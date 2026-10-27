"""Ground rejection, missing support, negative geometry and history invariants."""

from dataclasses import replace
import unittest

import numpy as np

from kinect.terrain.config import TerrainConfig, transform_points
from kinect.terrain.contracts import GroundModel
from kinect.terrain.fixtures import fixture, synthetic_config
from kinect.terrain.grid import build_grid, cell_indices, components
from kinect.terrain.ground import fit_support
from kinect.terrain.pipeline import TerrainPipeline


class TerrainGridTests(unittest.TestCase):
    def test_flooring_negative_coordinates_and_exclusive_upper_bounds(self):
        config = replace(synthetic_config(), bounds_xy_m=[-1, 1, -1, 1], seed_region_xy_m=[-0.5, 0.5, -0.5, 0.5])
        points = np.array([[-1, -1, 0], [-0.001, 0, 0], [0.999, 0.999, 0], [1, 0, 0], [-1.001, 0, 0]])
        indices, valid, shape = cell_indices(points, config)
        self.assertEqual(shape, (20, 20))
        np.testing.assert_array_equal(indices[:3], [[0, 0], [10, 9], [19, 19]])
        np.testing.assert_array_equal(valid, [True, True, True, False, False])

    def test_many_points_in_a_tiny_patch_do_not_establish_broad_support(self):
        c = replace(synthetic_config(), bounds_xy_m=[0, 1, 0, 1], seed_region_xy_m=[0, 1, 0, 1])
        points = np.tile([0.01, 0.01, 0.0], (1000, 1))
        model = GroundModel("VALID", "analytic", np.array([0, 0, 1, 0]), uncertainty_proxy_m=0.001)
        grid = build_grid(points, np.zeros(1000), model, c, 0, np.full((10, 10), "OUT_OF_VIEW", dtype="<U32"))
        self.assertEqual(grid["point_count"][0, 0], 1000)
        self.assertEqual(grid["spatial_support_coverage"][0, 0], 0.25)
        self.assertFalse(grid["observed_support"].any())
        self.assertFalse(grid["slope_valid"].any())
        self.assertTrue(np.isnan(grid["elevation_estimate_m"][1:, :]).all())

    def test_touching_raised_geometry_is_reported_as_one_connected_extent(self):
        mask = np.zeros((5, 5), bool)
        mask[1:3, 1:3] = True
        mask[3, 2] = True
        mask[4, 4] = True
        groups = list(components(mask))
        self.assertEqual(sorted(map(len, groups)), [1, 5])


class TerrainPipelineTests(unittest.TestCase):
    def process(self, layout, **kwargs):
        return TerrainPipeline(synthetic_config(), **kwargs).process(fixture(layout)[0])

    def test_flat_support_and_unknown_outside_view(self):
        result = self.process("flat")
        self.assertEqual(result.ground.status, "VALID")
        self.assertGreater(result.grid["observed_support"].sum(), 500)
        self.assertTrue((result.grid["unknown_reason"] == "OUT_OF_VIEW").any())
        self.assertFalse(result.grid["raised_evidence"].any())
        self.assertFalse(result.grid["negative_evidence"].any())
        self.assertFalse(result.grid["edge_evidence"].any())
        self.assertEqual(result.health["state"], "DEGRADED")  # synthetic is never qualified

    def test_raised_visible_extent_and_unknown_behind(self):
        result = self.process("raised")
        objects = [o for o in result.observations if o["type"] == "RAISED_GEOMETRY"]
        self.assertEqual(len(objects), 1)
        self.assertAlmostEqual(objects[0]["observed_height_range_m"][1], 0.22, delta=0.002)
        self.assertTrue((result.grid["unknown_reason"] == "OCCLUDED_OR_UNOBSERVED").any())
        raised_only = result.grid["raised_evidence"] & ~result.grid["observed_support"]
        candidate = result.grid["elevation_estimate_m"][raised_only]
        # Partial reference returns may provide a candidate estimate, but the
        # box's top must never become that estimate or verified support.
        self.assertTrue((np.abs(candidate[np.isfinite(candidate)]) <= 0.02).all())

    def test_small_raised_return_survives_fitting_downsampling(self):
        c = replace(synthetic_config(), voxel_size_m=0.1)
        frame = fixture("flat")[0]
        depth = frame.depth_raw.copy()
        ray_y = (120 - c.intrinsics_fx_fy_cx_cy[3]) / c.intrinsics_fx_fy_cx_cy[1]
        depth[120, 120] = round((0.8 - 0.12) / ray_y * 1000)
        result = TerrainPipeline(c).process(replace(frame, depth_raw=depth))
        self.assertEqual(result.ground.status, "VALID")
        self.assertTrue(result.grid["raised_evidence"].any())
        self.assertGreater(result.point_residual_m.max(), 0.1)
        self.assertIn([120, 120], result.metadata["point_pixels_vu"].tolist())

    def test_depression_is_distinct_from_dropout_and_occluded_edge(self):
        depressed = self.process("depression")
        missing = self.process("dropout")
        edge = self.process("edge")
        self.assertTrue(depressed.grid["negative_evidence"].any())
        observation = next(o for o in depressed.observations if o["type"] == "VISIBLE_DEPRESSION")
        self.assertAlmostEqual(observation["observed_height_range_m"][0], -0.18, delta=0.002)
        self.assertFalse(missing.grid["negative_evidence"].any())
        self.assertFalse(missing.grid["edge_evidence"].any())
        self.assertTrue((missing.grid["unknown_reason"] == "INVALID_DEPTH").any())
        self.assertTrue(edge.grid["edge_evidence"].any())
        self.assertFalse(edge.grid["negative_evidence"].any())

    def test_invalid_only_frame_never_creates_ground_or_crater(self):
        frame = fixture("flat")[0]
        for sentinel in (0, 65535, 10001):
            result = TerrainPipeline(synthetic_config()).process(replace(frame, depth_raw=np.full_like(frame.depth_raw, sentinel)))
            self.assertEqual(len(result.points_test_ground), 0)
            self.assertFalse(result.depth_valid_mask.any())
            self.assertFalse(result.grid["observed_support"].any())
            self.assertFalse(result.grid["negative_evidence"].any())
            self.assertTrue(np.isnan(result.grid["elevation_estimate_m"]).all())

    def test_wrong_plane_rejected_but_measured_elevations_preserved(self):
        result = self.process("wall")
        self.assertNotEqual(result.ground.status, "VALID")
        self.assertFalse(result.grid["observed_support"].any())
        self.assertTrue(np.isfinite(result.grid["elevation_min_m"]).any())
        self.assertTrue((result.grid["unknown_reason"] == "GROUND_MODEL_INVALID").all())

    def test_bad_residual_and_narrow_support_are_rejected(self):
        c = synthetic_config()
        rng = np.random.default_rng(5)
        points = np.column_stack((rng.uniform(1.05, 1.75, 500), rng.uniform(-0.7, 0.7, 500), rng.normal(0, 0.01, 500)))
        optical = transform_points(points, np.linalg.inv(c.T_test_ground_from_depth_optical))
        rejected = fit_support(optical, points, replace(c, max_ground_residual_m=0.002))
        self.assertEqual(rejected.status, "HIGH_RESIDUAL")
        points[:, 1] = rng.uniform(0, 0.02, 500)
        points[:, 2] = 0
        optical = transform_points(points, np.linalg.inv(c.T_test_ground_from_depth_optical))
        rejected = fit_support(optical, points, c)
        self.assertEqual(rejected.status, "INSUFFICIENT_SUPPORT")
        self.assertIn("concentrated", rejected.reason)
        for z in (0.25, -0.2):
            points[:, 1] = rng.uniform(-0.7, 0.7, 500)
            points[:, 2] = z
            optical = transform_points(points, np.linalg.inv(c.T_test_ground_from_depth_optical))
            self.assertNotEqual(fit_support(optical, points, c).status, "VALID")

    def test_freshness_receive_proxy_freeze_and_clock_fault(self):
        c, frame = synthetic_config(), fixture("flat")[0]
        stale = TerrainPipeline(c).process(frame, now=1)
        self.assertEqual(stale.health["state"], "STALE")
        self.assertFalse(stale.grid["observed_support"].any())
        pipeline = TerrainPipeline(c)
        first = replace(frame, device_timestamp_optional=10, capture_time_is_receive_proxy=True)
        pipeline.process(first)
        second = replace(first, sequence_id=1, receive_timestamp=1, depth_capture_timestamp=1)
        frozen = pipeline.process(second)
        self.assertEqual(frozen.health["state"], "STALE")
        self.assertIn("proxy", frozen.health["age_semantics"])
        pipeline = TerrainPipeline(c)
        pipeline.process(frame)
        duplicate = pipeline.process(frame)
        self.assertEqual(duplicate.health["state"], "ERROR")

    def test_history_ages_support_retains_hazards_and_resets_for_new_session(self):
        c, first = synthetic_config(), fixture("raised")[0]
        pipeline = TerrainPipeline(c, accumulate=True)
        initial = pipeline.process(first)
        missing = replace(first, sequence_id=1, receive_timestamp=2, depth_capture_timestamp=2,
                          depth_raw=np.zeros_like(first.depth_raw))
        result = pipeline.process(missing)
        history = result.metadata["history"]
        self.assertFalse(result.grid["raised_evidence"].any())
        np.testing.assert_array_equal(history["raised_evidence"], initial.grid["raised_evidence"])
        self.assertFalse(history["observed_support"].any())
        reset = pipeline.process(replace(missing, session_id="new-mount-session"))
        self.assertFalse(reset.metadata["history"]["raised_evidence"].any())

    def test_mount_mismatch_and_missing_profile_disable_dependent_outputs(self):
        c, frame = synthetic_config(), fixture()[0]
        for config, capture in ((TerrainConfig(), frame), (c, replace(frame, mount_transform_id="moved"))):
            result = TerrainPipeline(config).process(capture)
            self.assertEqual(result.health["state"], "CALIBRATION_REQUIRED")
            self.assertEqual(result.grid, {})
            self.assertFalse(result.observations)
        pipeline = TerrainPipeline(c)
        first = replace(frame, device_timestamp_optional=10)
        pipeline.process(first)
        bad_mount = replace(first, sequence_id=1, receive_timestamp=2, depth_capture_timestamp=2,
                            mount_transform_id="moved")
        result = pipeline.process(bad_mount)
        self.assertEqual(result.health["state"], "CALIBRATION_REQUIRED")
        self.assertEqual(result.grid, {})
        restored_metadata = replace(first, sequence_id=2, receive_timestamp=2.1, depth_capture_timestamp=2.1,
                                    device_timestamp_optional=11)
        self.assertEqual(pipeline.process(restored_metadata).ground.status, "INCONSISTENT")

    def test_temporal_plane_change_latches_until_new_session(self):
        config = replace(synthetic_config(), ground_temporal_offset_m=0.005)
        frame = fixture("flat")[0]
        pipeline = TerrainPipeline(config)
        pipeline.process(frame)
        moved = replace(frame, sequence_id=1, receive_timestamp=0.1, depth_capture_timestamp=0.1,
                        depth_raw=np.rint(frame.depth_raw.astype(float) * 0.975).astype(np.uint16))
        result = pipeline.process(moved)
        self.assertEqual(result.ground.status, "INCONSISTENT")
        self.assertTrue(result.health["reset_required"])
        stable = replace(frame, sequence_id=2, receive_timestamp=0.2, depth_capture_timestamp=0.2)
        self.assertEqual(pipeline.process(stable).ground.status, "INCONSISTENT")
        self.assertEqual(pipeline.process(replace(stable, session_id="new-calibrated-session")).ground.status, "VALID")


if __name__ == "__main__":
    unittest.main()
