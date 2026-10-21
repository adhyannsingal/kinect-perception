"""Hardware-free camera, simulation, and native-adapter contract checks."""

import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from kinect.camera import CameraFrame, KinectCamera
from kinect.simulation import SimulatedCamera


class SimulationTests(unittest.TestCase):
    def test_deterministic_geometry_timestamps_and_eof(self):
        with SimulatedCamera(width=16, height=12, fps=20, frame_count=2) as camera:
            first, second = camera.read(), camera.read()
            self.assertEqual(first.rgb.shape, (12, 16, 3))
            self.assertEqual(set(np.unique(first.depth_mm)), {0, 1000, 2000})
            self.assertEqual(second.depth_timestamp_s, 0.05)
            self.assertEqual(first.intrinsics.fx, 16)
            self.assertFalse(np.array_equal(first.depth_mm, second.depth_mm))
            self.assertIsNone(camera.read())
            self.assertIsNone(camera.read())
        with self.assertRaises(RuntimeError):
            camera.read()
        duplicate = SimulatedCamera(width=16, height=12).read()
        np.testing.assert_array_equal(first.rgb, duplicate.rgb)
        np.testing.assert_array_equal(first.depth_mm, duplicate.depth_mm)

    def test_parameters_and_empty_sequence(self):
        for kwargs in ({"width": 0}, {"height": -1}, {"fps": 0},
                       {"fps": float("nan")}, {"frame_count": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SimulatedCamera(**kwargs)
        self.assertIsNone(SimulatedCamera(frame_count=0).read())


class KinectAdapterTests(unittest.TestCase):
    def driver(self):
        driver = Mock(VIDEO_RGB=1, DEPTH_MM=2)
        driver.sync_get_video.return_value = (np.zeros((2, 3, 3), np.uint8), 1200)
        driver.sync_get_depth.return_value = (np.full((2, 3), 1234, np.uint16), 1207)
        return driver

    def test_lazy_import_units_capture_modes_buffer_ownership_and_cleanup(self):
        driver = self.driver()
        with patch.dict(sys.modules, {"freenect": driver}):
            with KinectCamera(device_index=1) as camera:
                driver.sync_get_video.assert_not_called()
                frame = camera.read()
                driver.sync_get_video.assert_called_once_with(index=1, format=1)
                driver.sync_get_depth.assert_called_once_with(index=1, format=2)
                self.assertEqual(frame.rgb_timestamp_s, 1.2)
                self.assertEqual(frame.depth_timestamp_s, 1.207)
                self.assertEqual(frame.depth_space, "depth_camera")
                driver.sync_get_depth.return_value[0][:] = 0
                self.assertTrue((frame.depth_mm == 1234).all())
            camera.close()
            driver.sync_stop.assert_called_once()
            with self.assertRaises(RuntimeError):
                camera.read()

    def test_capture_failure_is_not_eof_and_cleans_up(self):
        driver = self.driver()
        driver.sync_get_depth.return_value = None
        with patch.dict(sys.modules, {"freenect": driver}):
            with self.assertRaises(RuntimeError), KinectCamera() as camera:
                camera.read()
        driver.sync_stop.assert_called_once()

    def test_closing_unused_camera_does_not_load_native_driver(self):
        with patch("kinect.camera.importlib.import_module") as load:
            KinectCamera().close()
        load.assert_not_called()


class FrameTests(unittest.TestCase):
    def test_invalid_contract(self):
        frame = SimulatedCamera(width=4, height=4).read()
        from dataclasses import replace
        for changes in ({"rgb": frame.rgb.astype(float)},
                        {"depth_mm": frame.depth_mm.astype(float)},
                        {"rgb_timestamp_s": -1}, {"depth_timestamp_s": float("inf")},
                        {"depth_space": "unknown"}):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                replace(frame, **changes)


if __name__ == "__main__":
    unittest.main()
