"""Headless NPY/replay CLI and mocked Open3D display integration."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kinect.ground import segment_ground
from kinect.ground_simulation import synthetic_ground_frame
from kinect.pointcloud import depth_mm_to_point_cloud
from kinect.recording import RecordingWriter


SCRIPT = Path(__file__).resolve().parents[1] / "demos" / "06_ground_segmentation.py"


class GroundDemoTests(unittest.TestCase):
    def run_demo(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), "--headless", *map(str, args)],
                              capture_output=True, text=True, timeout=30)

    def test_synthetic_npy_and_recording_match(self):
        synthetic = self.run_demo("--synthetic")
        self.assertEqual(synthetic.returncode, 0, synthetic.stderr)
        self.assertIn("Plane:", synthetic.stdout)
        frame = synthetic_ground_frame()
        intrinsics = frame.intrinsics
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.npy"
            np.save(path, frame.depth_mm)
            npy = self.run_demo(path, "--intrinsics", intrinsics.fx, intrinsics.fy,
                                intrinsics.cx, intrinsics.cy)
            self.assertEqual(npy.returncode, 0, npy.stderr)
            recording = Path(directory) / "recording"
            with RecordingWriter(recording) as writer:
                writer.write(frame)
            replay = self.run_demo("--recording", recording)
            self.assertEqual(replay.returncode, 0, replay.stderr)
            expected = synthetic.stdout.split(": ", 1)[1]
            self.assertEqual(npy.stdout.split(": ", 1)[1], expected)
            self.assertEqual(replay.stdout.split(": ", 1)[1], expected)

    def test_invalid_depth_empty_cloud_and_malformed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.npy"
            np.save(path, [[0, -1, np.nan, np.inf, 10001]])
            result = self.run_demo(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("No ground plane detected", result.stdout)
            self.assertIn("total=0", result.stdout)
            np.save(path, np.ones((2, 2, 2)))
            result = self.run_demo(path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("two-dimensional", result.stderr)
            self.assertNotEqual(self.run_demo(Path(directory) / "missing.npy").returncode, 0)
        self.assertNotEqual(self.run_demo("--synthetic", "--distance-threshold", -1).returncode, 0)
        self.assertNotEqual(self.run_demo().returncode, 0)

    def test_display_colors_title_camera_and_empty_cloud(self):
        spec = importlib.util.spec_from_file_location("ground_demo", SCRIPT)
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        frame = synthetic_ground_frame()
        result = segment_ground(depth_mm_to_point_cloud(frame.depth_mm, frame.intrinsics))
        text = demo.summary(result)
        with patch.object(demo.o3d.visualization, "draw_geometries") as draw:
            demo.show_result(result, text)
            draw.assert_called_once()
            self.assertEqual(draw.call_args.kwargs["window_name"], text)
            self.assertEqual(draw.call_args.kwargs["up"], [0, -1, 0])
            colors = np.asarray(draw.call_args.args[0][0].colors)
            self.assertTrue((colors[result.ground_mask] == [0, 1, 0]).all())
            self.assertTrue((colors[~result.ground_mask] == [1, 0, 0]).all())
            draw.reset_mock()
            empty = segment_ground(depth_mm_to_point_cloud(np.zeros((2, 2))))
            demo.show_result(empty, demo.summary(empty))
            draw.assert_not_called()


if __name__ == "__main__":
    unittest.main()
