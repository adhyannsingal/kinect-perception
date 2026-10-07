"""Offline tests for Kinect depth-camera projection math."""

import unittest

import numpy as np

from kinect.geometry import CameraIntrinsics, depth_image_to_camera_points, pixel_depth_to_camera_xyz


INTRINSICS = CameraIntrinsics(fx=100.0, fy=200.0, cx=10.0, cy=20.0)


class GeometryTests(unittest.TestCase):
    def test_principal_point_projects_to_optical_axis(self) -> None:
        self.assertEqual(pixel_depth_to_camera_xyz(10, 20, 2.5, INTRINSICS), (0.0, 0.0, 2.5))

    def test_pixel_directions_follow_image_axes(self) -> None:
        x, y, z = pixel_depth_to_camera_xyz(20, 40, 2.0, INTRINSICS)
        self.assertEqual((x, y, z), (0.2, 0.2, 2.0))

    def test_invalid_depth_is_safe(self) -> None:
        self.assertIsNone(pixel_depth_to_camera_xyz(10, 20, 0.0, INTRINSICS))
        self.assertIsNone(pixel_depth_to_camera_xyz(10, 20, float("nan"), INTRINSICS))

    def test_depth_image_skips_invalid_values(self) -> None:
        depth_mm = np.array([[0, 1000], [2000, 10001]], dtype=np.uint16)
        points = depth_image_to_camera_points(depth_mm, INTRINSICS)
        np.testing.assert_allclose(points, [[-0.09, -0.1, 1.0], [-0.2, -0.19, 2.0]])


if __name__ == "__main__":
    unittest.main()
