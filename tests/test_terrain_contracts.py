"""Metric transforms, schemas, clocks and lossless session failure contracts."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kinect.terrain.config import TerrainConfig, rigid_transform, transform_points
from kinect.terrain.contracts import FrameBundle, write_json
from kinect.terrain.evaluation import calibration_check
from kinect.terrain.fixtures import fixture, synthetic_config
from kinect.terrain.session import SessionReplay, SessionWriter


class CoordinateConfigTests(unittest.TestCase):
    def test_known_axes_transform_direction_and_inverse(self):
        c = synthetic_config()
        t = rigid_transform(c.T_test_ground_from_depth_optical)
        optical = np.array([[0, 0, 2], [1, 0, 2], [0, 0.8, 2], [-1, 0, 2]])
        expected = [[2, 0, 0.8], [2, -1, 0.8], [2, 0, 0], [2, 1, 0.8]]
        np.testing.assert_allclose(transform_points(optical, t), expected)
        np.testing.assert_allclose(transform_points(expected, np.linalg.inv(t)), optical)

    def test_invalid_config_units_transforms_and_null_calibration(self):
        self.assertFalse(TerrainConfig().calibration_ready)
        config = synthetic_config()
        for changes in ({"resolution_m": 0}, {"qualified_range_m": [500, 5000]},
                        {"bounds_xy_m": [1, 1, 0, 2]}, {"maximum_input_age_s": -1},
                        {"intrinsics_fx_fy_cx_cy": [0, 1, 2, 3]},
                        {"max_normal_deviation_rad": 90}, {"measurement_uncertainty_m": float("nan")},
                        {"depth_representation": "raw_codes"}, {"minimum_support_coverage": 0},
                        {"T_test_ground_from_depth_optical": np.eye(4).tolist()}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes)
        reflection = np.array(config.T_test_ground_from_depth_optical)
        reflection[:3, 0] *= -1
        with self.assertRaises(ValueError):
            rigid_transform(reflection)

    def test_effective_config_identity_roundtrip_and_strict_json(self):
        c = synthetic_config()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            c.save(p)
            self.assertEqual(c.config_id, TerrainConfig.load(p).config_id)
            self.assertNotEqual(c.config_id, replace(c, resolution_m=0.05).config_id)
            write_json(p, {"unmeasured": np.array([float("nan"), float("inf")])})
            self.assertEqual(json.loads(p.read_text()), {"unmeasured": [None, None]})

    def test_calibration_error_budget_reports_failure_without_autoqualification(self):
        c = synthetic_config()
        optical = [[0, 0.2, 1], [0.3, 0.1, 2], [-0.4, 0.3, 3], [0, 0.8, 2], [0.2, 0.7, 1.5]]
        expected = transform_points(optical, c.T_test_ground_from_depth_optical)
        targets = [dict(observed_depth_optical_m=p, expected_test_ground_m=q.tolist(), measurement_uncertainty_m=0.002)
                   for p, q in zip(optical, expected)]
        report = calibration_check(c, {"targets": targets})
        self.assertTrue(report["within_proposed_budget"])
        self.assertIn("REQUIRES_PHYSICAL", report["qualification"])
        targets[0]["expected_test_ground_m"][0] += 0.3
        self.assertFalse(calibration_check(c, {"targets": targets})["within_proposed_budget"])
        with self.assertRaises(ValueError):
            calibration_check(c, {"targets": targets[:4]})


class SessionTests(unittest.TestCase):
    def test_raw_invalid_values_optional_rgb_clock_and_drops_roundtrip(self):
        c = synthetic_config()
        frame, _ = fixture()
        frame.depth_raw[0, :3] = [0, 65535, 10001]
        frame.rgb = None
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "session"
            with SessionWriter(path, c, frame.session_id) as writer:
                writer.write(frame)
                second = replace(frame, sequence_id=3, depth_capture_timestamp=0.3, receive_timestamp=0.3)
                writer.write(second, dropped_frames=2)
            restored = list(SessionReplay(path))
            self.assertEqual([f.sequence_id for f in restored], [0, 3])
            np.testing.assert_array_equal(restored[0].depth_raw, frame.depth_raw)
            self.assertEqual(restored[0].depth_raw.dtype, np.uint16)
            self.assertIsNone(restored[0].rgb)
            self.assertEqual(restored[0].timestamp_clock_domain, frame.timestamp_clock_domain)
            entries = [json.loads(l) for l in (path / "frame_index.jsonl").read_text().splitlines()]
            self.assertEqual(entries[1]["missing_sequences_before"], 2)
            with self.assertRaises(FileExistsError):
                SessionWriter(path, c, frame.session_id)
            (path / "frames/00000000.npz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                list(SessionReplay(path))

    def test_session_rejects_clock_calibration_and_sequence_changes(self):
        c, frame = synthetic_config(), fixture()[0]
        with tempfile.TemporaryDirectory() as d, SessionWriter(Path(d) / "session", c, frame.session_id) as writer:
            writer.write(frame)
            for change in ({"sequence_id": 0}, {"sequence_id": 1, "mount_transform_id": "moved"},
                           {"sequence_id": 1, "timestamp_clock_domain": "different"}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    writer.write(replace(frame, **change))

    def test_frame_rejects_wrong_raw_units_and_clock_order(self):
        frame = fixture()[0]
        for change in ({"depth_raw": frame.depth_raw.astype(float)}, {"depth_capture_timestamp": 1},
                       {"receive_timestamp": float("nan")}, {"depth_format": "raw_codes"}):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                replace(frame, **change)


if __name__ == "__main__":
    unittest.main()
