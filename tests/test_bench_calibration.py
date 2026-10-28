"""Known geometry, held-out target separation and real input rejection policy."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from kinect.geometry import CameraIntrinsics
from kinect.terrain.bench_calibration import build_candidate, calibration_plan, fit_rigid_mount, sample_target
from kinect.terrain.config import TerrainConfig, transform_points
from kinect.terrain.contracts import write_json


T = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0.8], [0, 0, 0, 1]], dtype=float)
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bench_calibration.py"


def measured_fixture():
    plan = calibration_plan("test-mount")
    plan.update(origin_description="independently measured test origin", measurement_method="synthetic test fixture, not physical qualification")
    expected = np.array([[1, -0.4, 0], [2, 0.4, 0], [1.5, 0.1, 0.3],
                         [1.1, -0.2, 0], [1.7, 0.5, 0.2], [2.4, -0.5, 0.1],
                         [2.8, 0.3, 0.4], [1.8, -0.1, 0.05]])
    optical = transform_points(expected, np.linalg.inv(T))
    observations = {}
    for index, target in enumerate(plan["targets"]):
        target.update(expected_test_ground_m=expected[index].tolist(), measurement_uncertainty_m=0.002,
                      observation_path=f"{target['target_id']}.json")
        observations[target["target_id"]] = dict(
            schema_version=1, target_id=target["target_id"], mount_transform_id="test-mount",
            status="ACCEPTED_SAMPLE", frame_id="depth_optical", units="metres",
            depth_format="uint16_mm_optical_z", observed_depth_optical_m=optical[index].tolist(),
            stream_width=640, stream_height=480, intrinsics_fx_fy_cx_cy=[594, 591, 339, 242],
            sources=[dict(source_id=f"fixture:{index}", depth_sha256=f"synthetic:{index}")],
            pixel_uv=[index, index], source_calibration_kind="synthetic")
    return plan, observations, optical, expected


class TargetSamplingTests(unittest.TestCase):
    def sample(self, depths, **kwargs):
        frames = ((d, float(i), f"frame:{i}") for i, d in enumerate(depths))
        return sample_target(frames, (1, 1), CameraIntrinsics(100, 100, 1, 1),
                             mount_id="fixed", target_id="target", **kwargs)

    def test_stable_metric_sample_and_invalid_sentinel_rejection(self):
        result = self.sample([np.full((3, 3), 1200, np.uint16) for _ in range(5)])
        self.assertEqual(result["status"], "ACCEPTED_SAMPLE")
        np.testing.assert_allclose(result["observed_depth_optical_m"], [0, 0, 1.2])
        self.assertIsNone(result["expected_test_ground_m"])
        self.assertEqual(result["frame_count"], 5)
        for invalid in (0, 10001, 65535):
            result = self.sample([np.full((3, 3), invalid, np.uint16) for _ in range(5)])
            self.assertEqual(result["status"], "INSUFFICIENT_VALID_DEPTH")
            self.assertIsNone(result["observed_depth_optical_m"])

    def test_noisy_or_mixed_surface_rejected_without_adjusting_limit(self):
        result = self.sample([np.full((3, 3), z, np.uint16) for z in (1000, 1200, 1000, 1100, 1000)])
        self.assertEqual(result["status"], "UNSTABLE_OR_MIXED_SURFACES")
        self.assertGreater(result["depth_p95_p5_spread_m"], 0.02)
        self.assertIsNone(result["observed_depth_optical_m"])

    def test_insufficient_frames_raw_units_roi_and_frozen_clock_rejected(self):
        good = np.full((3, 3), 1000, np.uint16)
        with self.assertRaises(ValueError):
            self.sample([good, good])
        with self.assertRaises(ValueError):
            self.sample([good.astype(float)] * 5)
        with self.assertRaises(ValueError):
            self.sample([good] * 5, radius_px=2)
        with self.assertRaisesRegex(ValueError, "timestamps"):
            sample_target([(good, 0, "f")] * 5, (1, 1), CameraIntrinsics(100, 100, 1, 1),
                          mount_id="fixed", target_id="target")


class MountFittingTests(unittest.TestCase):
    def test_known_transform_and_held_out_validation_never_qualifies(self):
        plan, observations, optical, expected = measured_fixture()
        transform, _ = fit_rigid_mount(optical[:3], expected[:3])
        np.testing.assert_allclose(transform, T, atol=1e-10)
        candidate, report = build_candidate(plan, observations, TerrainConfig())
        np.testing.assert_allclose(candidate.T_test_ground_from_depth_optical, T, atol=1e-10)
        self.assertEqual(candidate.calibration_kind, "synthetic")
        self.assertFalse(candidate.calibration_ready)
        self.assertFalse(report["hardware_qualified"])
        self.assertTrue(report["within_declared_error_budget"])
        self.assertTrue(report["validation_has_forward_lateral_height_variation"])

    def test_validation_failure_cannot_influence_fitted_transform(self):
        plan, observations, _, _ = measured_fixture()
        before, _ = build_candidate(plan, observations, TerrainConfig())
        plan["targets"][3]["expected_test_ground_m"][0] += 0.3
        after, report = build_candidate(plan, observations, TerrainConfig())
        np.testing.assert_array_equal(before.T_test_ground_from_depth_optical, after.T_test_ground_from_depth_optical)
        self.assertFalse(report["within_declared_error_budget"])

    def test_collinear_duplicate_wrong_units_and_incorrect_correspondences_fail(self):
        _, _, observed, expected = measured_fixture()
        for a, b in (([[0, 0, 1], [0, 0, 2], [0, 0, 3]], [[1, 0, 0], [2, 0, 0], [3, 0, 0]]),
                     (observed[:3], expected[:3] * 1000),
                     (observed[[0, 0, 1]], expected[[0, 0, 1]])):
            with self.subTest(), self.assertRaises(ValueError):
                fit_rigid_mount(a, b)

    def test_observation_reuse_movement_missing_measurements_and_stream_change_fail(self):
        for fault in ("reuse", "mount", "missing", "stream", "validation_count"):
            plan, observations, _, _ = measured_fixture()
            if fault == "reuse":
                observations["validation-1"]["sources"] = observations["fit-1"]["sources"]
                observations["validation-1"]["pixel_uv"] = observations["fit-1"]["pixel_uv"]
            elif fault == "mount":
                observations["validation-1"]["mount_transform_id"] = "moved"
            elif fault == "missing":
                plan["targets"][0]["expected_test_ground_m"] = None
            elif fault == "stream":
                observations["validation-1"]["stream_width"] = 320
            else:
                plan["targets"] = plan["targets"][:-1]
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                build_candidate(plan, observations, TerrainConfig())

    def test_real_source_remains_bench_unqualified_and_mixed_sources_fail(self):
        plan, observations, _, _ = measured_fixture()
        observations["fit-1"]["source_calibration_kind"] = "bench_unqualified"
        with self.assertRaisesRegex(ValueError, "mixed"):
            build_candidate(plan, observations, TerrainConfig())
        for observation in observations.values():
            observation["source_calibration_kind"] = "bench_unqualified"
        candidate, report = build_candidate(plan, observations, TerrainConfig())
        self.assertEqual(candidate.calibration_kind, "bench_unqualified")
        self.assertEqual(report["status"], "PENDING_PHYSICAL_REVIEW")
        self.assertFalse(report["hardware_qualified"])


class CalibrationCLITests(unittest.TestCase):
    def test_plan_and_fit_export_separate_candidate_preserve_base(self):
        plan, observations, _, _ = measured_fixture()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = root / "base.json"
            TerrainConfig().save(base)
            original = base.read_bytes()
            write_json(root / "plan.json", plan)
            for name, observation in observations.items():
                write_json(root / f"{name}.json", observation)
            result = subprocess.run([sys.executable, str(SCRIPT), "fit", "--plan", str(root / "plan.json"),
                                     "--config", str(base), "--output", str(root / "candidate")],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(base.read_bytes(), original)
            report = json.loads((root / "candidate/validation_report.json").read_text())
            self.assertFalse(report["hardware_qualified"])
            self.assertTrue((root / "candidate/measurements_snapshot.json").exists())
            result = subprocess.run([sys.executable, str(SCRIPT), "plan", "--mount-id", "bench",
                                     "--output", str(root / "empty.json")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(all(t["expected_test_ground_m"] is None for t in json.loads((root / "empty.json").read_text())["targets"]))

    def test_observe_saved_npy_and_rejection_diagnostic(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            files = []
            for i in range(3):
                p = root / f"{i}.npy"
                np.save(p, np.full((3, 3), 1200, np.uint16))
                files.append(str(p))
            command = [sys.executable, str(SCRIPT), "observe", "--npy", *files, "--intrinsics", "100", "100", "1", "1",
                       "--pixel", "1", "1", "--mount-id", "bench", "--target-id", "fit-1", "--output", str(root / "sample.json")]
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            sample = json.loads((root / "sample.json").read_text())
            self.assertEqual(sample["frame_count"], 3)
            self.assertIn("not_capture", sample["timestamp_clock_domain"])


if __name__ == "__main__":
    unittest.main()
