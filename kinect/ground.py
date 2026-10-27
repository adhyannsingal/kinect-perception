"""Orientation-constrained RANSAC ground segmentation in camera metres."""

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class GroundSegmentation:
    """Finite input points, normalized plane ax+by+cz+d=0, and ground mask.

    The normal points toward the configured ground direction. None means no
    model met the orientation, height, and support constraints. The cloud is
    an independent copy; ground_mask indexes its full-resolution points.
    """

    cloud: o3d.geometry.PointCloud
    plane: np.ndarray | None
    ground_mask: np.ndarray
    fit_point_count: int

    @property
    def ground_count(self) -> int:
        return int(self.ground_mask.sum())

    @property
    def inlier_percentage(self) -> float:
        return 100 * self.ground_count / len(self.ground_mask) if len(self.ground_mask) else 0.0

    def colored_cloud(self) -> o3d.geometry.PointCloud:
        """Return green ground and red non-ground without mutating the input."""
        cloud = o3d.geometry.PointCloud(self.cloud)
        colors = np.zeros((len(self.ground_mask), 3))
        colors[:, 0] = 1
        colors[self.ground_mask] = (0, 1, 0)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud


def segment_ground(
    cloud: o3d.geometry.PointCloud,
    *,
    distance_threshold_m: float = 0.02,
    voxel_size_m: float = 0.03,
    expected_normal=(0.0, 1.0, 0.0),
    max_tilt_deg: float = 15.0,
    height_range_m=(0.1, 3.0),
    iterations: int = 1000,
    min_inliers: int = 100,
    min_inlier_ratio: float = 0.1,
    seed: int = 0,
) -> GroundSegmentation:
    """Fit ground with three-point RANSAC, rejecting inadmissible hypotheses.

    +Y is down in this repository. expected_normal points from camera toward
    ground (not necessarily +Y for a tilted camera). height_range_m constrains
    the signed perpendicular camera-to-plane distance, -d after normalizing
    and orienting the normal. Set it around a known camera height to exclude
    tabletops/ceilings. Orientation alone cannot distinguish parallel surfaces.

    Support thresholds apply to the voxelized fit cloud. The accepted model
    labels ALL finite original points by perpendicular distance. No input is
    mutated. voxel_size_m=0 disables downsampling. A seeded local RNG makes
    sampling repeatable without changing global NumPy/Open3D RNG state.
    """
    normal = np.asarray(expected_normal, dtype=float)
    heights = np.asarray(height_range_m, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
        raise ValueError("expected_normal must be a finite nonzero 3-vector")
    if heights.shape != (2,) or not np.isfinite(heights).all() or not 0 <= heights[0] < heights[1]:
        raise ValueError("height_range_m must satisfy 0 <= minimum < maximum")
    if not np.isfinite(distance_threshold_m) or distance_threshold_m <= 0:
        raise ValueError("distance_threshold_m must be finite and positive")
    if not np.isfinite(voxel_size_m) or voxel_size_m < 0:
        raise ValueError("voxel_size_m must be finite and nonnegative")
    if not np.isfinite(max_tilt_deg) or not 0 <= max_tilt_deg < 90:
        raise ValueError("max_tilt_deg must be in [0, 90)")
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if not isinstance(min_inliers, int) or min_inliers < 3:
        raise ValueError("min_inliers must be an integer >= 3")
    if not np.isfinite(min_inlier_ratio) or not 0 <= min_inlier_ratio <= 1:
        raise ValueError("min_inlier_ratio must be in [0, 1]")
    normal = normal / np.linalg.norm(normal)
    cos_limit = np.cos(np.deg2rad(max_tilt_deg))

    def orient_and_validate(model):
        if model[:3] @ normal < 0:
            model = -model
        if model[:3] @ normal + 1e-12 < cos_limit:
            return None
        if not heights[0] <= -model[3] <= heights[1]:
            return None
        return model

    def distances(points, model):
        return np.abs(np.einsum("ij,j->i", points, model[:3]) + model[3])

    points = np.asarray(cloud.points)
    finite = np.flatnonzero(np.isfinite(points).all(axis=1))
    clean = cloud.select_by_index(finite.tolist())
    fit = clean.voxel_down_sample(voxel_size_m) if voxel_size_m else clean
    sample_points = np.asarray(fit.points)
    count = len(sample_points)
    required = max(min_inliers, int(np.ceil(min_inlier_ratio * count)))
    no_ground = GroundSegmentation(clean, None, np.zeros(len(clean.points), dtype=bool), count)
    if count < required:
        return no_ground

    rng = np.random.default_rng(seed)
    best_model, best_count, best_error = None, 0, float("inf")
    for _ in range(iterations):
        a, b, c = sample_points[rng.choice(count, 3, replace=False)]
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length < 1e-10:
            continue
        n /= length
        model = orient_and_validate(np.r_[n, -n @ a])
        if model is None:
            continue
        residuals = distances(sample_points, model)
        mask = residuals <= distance_threshold_m
        support = int(mask.sum())
        if support < required:
            continue
        error = float(np.mean(residuals[mask]))
        if support > best_count or (support == best_count and error < best_error):
            best_model, best_count, best_error = model, support, error
    if best_model is None:
        return no_ground

    # Refine the winning consensus by least squares; retain the valid RANSAC
    # model if refinement drifts outside the configured constraints/support.
    for _ in range(3):
        inliers = sample_points[distances(sample_points, best_model) <= distance_threshold_m]
        center = inliers.mean(axis=0)
        _, singular, vh = np.linalg.svd(inliers - center, full_matrices=False)
        if singular[1] < 1e-10:
            break
        refined = orient_and_validate(np.r_[vh[-1], -vh[-1] @ center])
        if refined is None:
            break
        if np.count_nonzero(distances(sample_points, refined) <= distance_threshold_m) < required:
            break
        best_model = refined

    mask = distances(np.asarray(clean.points), best_model) <= distance_threshold_m
    return GroundSegmentation(clean, best_model, mask, count)
