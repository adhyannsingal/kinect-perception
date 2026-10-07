"""Offline Open3D tests that do not require Kinect hardware."""

import unittest

import numpy as np
import open3d as o3d

from kinect.geometry import CameraIntrinsics
from kinect.pointcloud import depth_mm_to_point_cloud


class PointCloudTests(unittest.TestCase):
    def test_synthetic_flat_plane(self) -> None:
        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=1.5, cy=1.5)
        depth_mm = np.full((4, 4), 2000, dtype=np.uint16)
        cloud = depth_mm_to_point_cloud(depth_mm, intrinsics)
        points = np.asarray(cloud.points)
        self.assertIsInstance(cloud, o3d.geometry.PointCloud)
        self.assertEqual(points.shape, (16, 3))
        self.assertTrue(np.isfinite(points).all())
        self.assertAlmostEqual(float(np.median(points[:, 2])), 2.0)

    def test_invalid_depth_is_ignored(self) -> None:
        cloud = depth_mm_to_point_cloud(np.array([[0, 1000, 10001]], dtype=np.uint16))
        self.assertEqual(np.asarray(cloud.points).shape, (1, 3))

    def test_voxel_downsampling_preserves_metric_points(self) -> None:
        depth_mm = np.full((10, 10), 2000, dtype=np.uint16)
        cloud = depth_mm_to_point_cloud(depth_mm, voxel_size_m=0.05)
        points = np.asarray(cloud.points)
        self.assertGreater(len(points), 0)
        self.assertLessEqual(len(points), depth_mm.size)
        self.assertTrue(np.isfinite(points).all())
        self.assertAlmostEqual(float(np.median(points[:, 2])), 2.0)


if __name__ == "__main__":
    unittest.main()
