"""Quality gates around the existing Demo 6 RANSAC utility."""

import numpy as np
import open3d as o3d

from ..ground import segment_ground
from .contracts import GroundModel


def fit_support(optical_points, ground_points, config, previous=None):
    bounds = config.seed_region_xy_m
    seed = ((ground_points[:, 0] >= bounds[0]) & (ground_points[:, 0] < bounds[1]) &
            (ground_points[:, 1] >= bounds[2]) & (ground_points[:, 1] < bounds[3]))
    if seed.sum() < config.ground_min_inliers:
        return GroundModel("INSUFFICIENT_SUPPORT", "too few visible points in measured support seed")
    transform = np.asarray(config.T_test_ground_from_depth_optical)
    toward_ground_optical = transform[:3, :3].T @ [0, 0, -1]
    height = transform[2, 3]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(optical_points[seed])
    candidate = segment_ground(
        cloud, expected_normal=toward_ground_optical,
        height_range_m=(height - config.ground_height_tolerance_m, height + config.ground_height_tolerance_m),
        max_tilt_deg=np.rad2deg(config.max_normal_deviation_rad),
        distance_threshold_m=config.ground_distance_m, voxel_size_m=config.voxel_size_m,
        min_inliers=config.ground_min_inliers, min_inlier_ratio=config.ground_min_inlier_ratio,
        iterations=config.ground_iterations, seed=config.seed)
    if candidate.plane is None:
        return GroundModel("INSUFFICIENT_SUPPORT", "no plane satisfied reference height, orientation and support")
    # Inverse-transpose plane mapping, then turn its normal upward.
    plane = np.linalg.inv(transform).T @ candidate.plane
    if plane[2] < 0:
        plane = -plane
    plane /= np.linalg.norm(plane[:3])
    if plane[2] < np.cos(config.max_normal_deviation_rad) or abs(plane[3]) > config.ground_height_tolerance_m:
        return GroundModel("BAD_ORIENTATION", "model inconsistent with measured reference transform", plane)
    residual = ground_points @ plane[:3] + plane[3]
    inliers = seed & (np.abs(residual) <= config.ground_distance_m)
    if inliers.sum() < config.ground_min_inliers:
        return GroundModel("INSUFFICIENT_SUPPORT", "too few original support points", plane)
    xy = ground_points[inliers, :2]
    bins = np.floor((xy - [bounds[0], bounds[2]]) /
                    [bounds[1] - bounds[0], bounds[3] - bounds[2]] * 4).astype(int)
    coverage = len(np.unique(bins, axis=0)) / 16
    median, p95 = np.percentile(np.abs(residual[inliers]), [50, 95])
    model = GroundModel("VALID", "constrained support fit accepted", plane, coverage,
                        float(median), float(p95), float(max(p95, config.measurement_uncertainty_m)))
    if coverage < config.minimum_ground_coverage:
        model.status, model.reason = "INSUFFICIENT_SUPPORT", "support concentrated in too little of the seed region"
    elif p95 > config.max_ground_residual_m:
        model.status, model.reason = "HIGH_RESIDUAL", "support residual exceeds configured error budget"
    elif previous is not None:
        angle = np.arccos(np.clip(plane[:3] @ previous[:3], -1, 1))
        if angle > config.ground_temporal_angle_rad or abs(plane[3] - previous[3]) > config.ground_temporal_offset_m:
            model.status, model.reason = "INCONSISTENT", "support changed; check mount and restart session"
    return model
