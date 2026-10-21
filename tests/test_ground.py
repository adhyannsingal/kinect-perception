"""Known-geometry regression tests for constrained floor RANSAC."""

import unittest

import numpy as np
import open3d as o3d

from kinect.ground import segment_ground
from kinect.ground_simulation import synthetic_ground_frame
from kinect.pointcloud import depth_mm_to_point_cloud
from kinect.simulation import SimulatedCamera


def cloud_from(points):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float).reshape(-1, 3))
    return cloud


class GroundTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(32)

    def floor(self, count=500, height=1.0, noise=0.0):
        return np.column_stack((self.rng.uniform(-2, 2, count),
                                height + self.rng.normal(0, noise, count),
                                self.rng.uniform(1, 5, count)))

    def test_floor_detected_despite_larger_wall_and_obstacles(self):
        floor = self.floor(noise=0.002)
        wall = np.column_stack((self.rng.uniform(-2, 2, 1800),
                               self.rng.uniform(-1, 0.7, 1800), np.full(1800, 5)))
        obstacles = self.rng.uniform([-1, -0.2, 1], [1, 0.6, 3], (200, 3))
        cloud = cloud_from(np.vstack((floor, wall, obstacles)))
        original = np.asarray(cloud.points).copy()
        result = segment_ground(cloud, voxel_size_m=0, distance_threshold_m=0.01,
                                height_range_m=(0.9, 1.1), iterations=2000)
        self.assertIsNotNone(result.plane)
        np.testing.assert_allclose(result.plane, [0, 1, 0, -1], atol=0.002)
        self.assertEqual(result.ground_count, 500)
        self.assertTrue(result.ground_mask[:500].all())
        self.assertFalse(result.ground_mask[500:].any())
        self.assertAlmostEqual(result.inlier_percentage, 20)
        np.testing.assert_array_equal(cloud.points, original)
        colored = result.colored_cloud()
        np.testing.assert_array_equal(np.asarray(colored.colors)[:500], np.tile([0, 1, 0], (500, 1)))
        np.testing.assert_array_equal(np.asarray(colored.colors)[500:], np.tile([1, 0, 0], (2000, 1)))
        self.assertFalse(cloud.has_colors())

    def test_height_prior_rejects_larger_parallel_tabletop(self):
        points = np.vstack((self.floor(400), self.floor(1200, height=0.4)))
        result = segment_ground(cloud_from(points), voxel_size_m=0,
                                height_range_m=(0.95, 1.05), min_inlier_ratio=0.1)
        self.assertIsNotNone(result.plane)
        self.assertEqual(result.ground_count, 400)
        self.assertFalse(result.ground_mask[400:].any())

    def test_tilted_ground_requires_appropriate_orientation(self):
        points = self.floor()
        points[:, 1] += 0.3 * points[:, 2]
        cloud = cloud_from(points)
        self.assertIsNone(segment_ground(cloud, max_tilt_deg=5).plane)
        expected = np.array([0, 1, -0.3, -1]) / np.sqrt(1.09)
        result = segment_ground(cloud, expected_normal=(0, 1, -0.3), max_tilt_deg=5)
        np.testing.assert_allclose(result.plane, expected, atol=1e-10)
        self.assertEqual(result.ground_count, len(points))

    def test_distance_threshold_and_full_resolution_labels(self):
        floor = self.floor(800)
        # Widely separated non-ground points must not move the consensus plane.
        probes = np.array([[8, 1.01, 2], [9, 1.06, 3]])
        cloud = cloud_from(np.vstack((floor, probes)))
        result = segment_ground(cloud, voxel_size_m=0.15, distance_threshold_m=0.02)
        self.assertLess(result.fit_point_count, len(cloud.points))
        self.assertEqual(len(result.ground_mask), len(cloud.points))
        np.testing.assert_array_equal(result.ground_mask[-2:], [True, False])
        strict = segment_ground(cloud, voxel_size_m=0.15, distance_threshold_m=0.005)
        np.testing.assert_array_equal(strict.ground_mask[-2:], [False, False])

    def test_no_ground_empty_small_degenerate_and_nonfinite(self):
        front = SimulatedCamera(width=32, height=24).read()
        wall_cloud = depth_mm_to_point_cloud(front.depth_mm, front.intrinsics)
        self.assertIsNone(segment_ground(wall_cloud, max_tilt_deg=5).plane)
        for points in ([], [[0, 1, 1], [1, 1, 2]], np.zeros((200, 3)),
                       np.column_stack((np.arange(200), np.ones(200), np.ones(200)))):
            result = segment_ground(cloud_from(points))
            self.assertIsNone(result.plane)
            self.assertEqual(result.ground_count, 0)
            self.assertEqual(result.inlier_percentage, 0)
        points = np.vstack((self.floor(), [[np.nan, 0, 1], [0, np.inf, 1]]))
        result = segment_ground(cloud_from(points))
        self.assertEqual(len(result.cloud.points), 500)
        self.assertEqual(result.ground_count, 500)

    def test_minimum_support_can_reject_a_plane(self):
        cloud = cloud_from(self.floor(50))
        self.assertIsNone(segment_ground(cloud, voxel_size_m=0, min_inliers=51).plane)

    def test_synthetic_depth_projection_and_determinism(self):
        frame = synthetic_ground_frame()
        cloud = depth_mm_to_point_cloud(frame.depth_mm, frame.intrinsics)
        first, second = segment_ground(cloud), segment_ground(cloud)
        self.assertIsNotNone(first.plane)
        # Millimetre quantization, voxel averaging, and wall points within the
        # 2 cm inlier band allow a few mm of least-squares offset at the seam.
        np.testing.assert_allclose(first.plane, [0, 1, 0, -0.8], atol=0.005)
        np.testing.assert_array_equal(first.plane, second.plane)
        np.testing.assert_array_equal(first.ground_mask, second.ground_mask)
        # Includes millimetre quantization in the ray-cast depth image.
        truth = np.abs(np.asarray(cloud.points)[:, 1] - 0.8) < 0.02
        np.testing.assert_array_equal(first.ground_mask, truth)

    def test_invalid_options(self):
        for options in ({"distance_threshold_m": 0}, {"distance_threshold_m": float("nan")},
                        {"voxel_size_m": -1}, {"voxel_size_m": float("inf")},
                        {"expected_normal": (0, 0, 0)}, {"expected_normal": (0, 1)},
                        {"max_tilt_deg": 90}, {"height_range_m": (2, 1)},
                        {"iterations": 0}, {"min_inliers": 2}, {"min_inlier_ratio": 1.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                segment_ground(cloud_from([]), **options)


if __name__ == "__main__":
    unittest.main()
