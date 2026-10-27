"""Versioned effective configuration and rigid-transform validation."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def rigid_transform(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("T_test_ground_from_depth_optical must be finite 4x4")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("transform must be homogeneous")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError("transform rotation must be orthonormal with determinant +1")
    if matrix[2, 3] <= 0:
        raise ValueError("camera must be above the measured reference plane")
    return matrix


@dataclass(frozen=True)
class TerrainConfig:
    schema_version: int = 1
    mode: str = "stationary_bench_perception"
    calibration_id: str | None = None
    mount_transform_id: str | None = None
    calibration_kind: str = "unresolved"  # synthetic, bench_unqualified, qualified
    measurement_method: str | None = None
    stream_width: int | None = None
    stream_height: int | None = None
    intrinsics_fx_fy_cx_cy: list | None = None
    distortion_model: str | None = None
    depth_representation: str | None = None
    qualified_range_m: list | None = None
    T_test_ground_from_depth_optical: list | None = None
    bounds_xy_m: list | None = None  # xmin,xmax,ymin,ymax; upper bounds exclusive
    bounds_z_m: tuple = (-0.5, 1.5)
    resolution_m: float = 0.05
    seed_region_xy_m: list | None = None
    measurement_uncertainty_m: float | None = None
    ground_distance_m: float = 0.02
    ground_height_tolerance_m: float = 0.04
    max_normal_deviation_rad: float = 0.15
    max_ground_residual_m: float = 0.02
    minimum_ground_coverage: float = 0.25
    ground_min_inliers: int = 40
    ground_min_inlier_ratio: float = 0.3
    ground_iterations: int = 400
    voxel_size_m: float = 0.03
    ground_temporal_offset_m: float = 0.03
    ground_temporal_angle_rad: float = 0.1
    minimum_support_coverage: float = 0.5
    minimum_support_span_fraction: float = 0.25
    support_subdivisions: int = 2
    minimum_cell_points: int = 3
    raised_threshold_m: float = 0.06
    depression_threshold_m: float = 0.06
    edge_threshold_m: float = 0.08
    negative_context_radius_m: float = 0.4
    maximum_input_age_s: float = 0.5
    support_evidence_ttl_s: float = 1.0
    target_grid_hz: float = 10.0
    seed: int = 0

    def __post_init__(self):
        if self.schema_version != 1 or self.mode != "stationary_bench_perception":
            raise ValueError("unsupported configuration schema/mode")
        if self.calibration_kind not in ("unresolved", "synthetic", "bench_unqualified", "qualified"):
            raise ValueError("invalid calibration_kind")
        for name in ("calibration_id", "mount_transform_id", "measurement_method"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string")
        if self.distortion_model not in (None, "none_synthetic", "rectified_pinhole", "pinhole_approximation"):
            raise ValueError("unsupported distortion model; supply rectified depth or qualify the pinhole approximation")
        if self.calibration_kind != "synthetic" and self.distortion_model == "none_synthetic":
            raise ValueError("synthetic distortion metadata cannot qualify a physical camera")
        for name in ("resolution_m", "ground_distance_m", "ground_height_tolerance_m",
                     "max_ground_residual_m", "raised_threshold_m", "depression_threshold_m",
                     "edge_threshold_m", "negative_context_radius_m", "target_grid_hz"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("voxel_size_m", "maximum_input_age_s", "support_evidence_ttl_s",
                     "ground_temporal_offset_m", "ground_temporal_angle_rad"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("minimum_ground_coverage", "minimum_support_coverage", "minimum_support_span_fraction", "ground_min_inlier_ratio"):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        if not np.isfinite(self.max_normal_deviation_rad) or not 0 < self.max_normal_deviation_rad < np.pi / 2:
            raise ValueError("max_normal_deviation_rad must be in (0,pi/2)")
        for name in ("ground_min_inliers", "ground_iterations", "minimum_cell_points", "support_subdivisions"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.ground_min_inliers < 3 or not 1 <= self.support_subdivisions <= 8:
            raise ValueError("ground needs >=3 inliers; support_subdivisions must be <=8")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("bounds_xy_m", "seed_region_xy_m"):
            value = getattr(self, name)
            if value is not None:
                a = np.asarray(value, dtype=float)
                if a.shape != (4,) or not np.isfinite(a).all() or not (a[0] < a[1] and a[2] < a[3]):
                    raise ValueError(f"{name} must be [xmin,xmax,ymin,ymax]")
        for name in ("bounds_z_m", "qualified_range_m"):
            value = getattr(self, name)
            if value is not None:
                a = np.asarray(value, dtype=float)
                if a.shape != (2,) or not np.isfinite(a).all() or a[0] >= a[1]:
                    raise ValueError(f"{name} must be an increasing finite pair")
        if self.qualified_range_m is not None and not 0 < self.qualified_range_m[0] < self.qualified_range_m[1] <= 10:
            raise ValueError("qualified range must lie in (0,10] optical metres")
        if self.bounds_xy_m is not None:
            b = self.bounds_xy_m
            shape = np.array([b[1] - b[0], b[3] - b[2]]) / self.resolution_m
            if not np.allclose(shape, np.round(shape)) or np.prod(np.round(shape)) > 1_000_000:
                raise ValueError("grid extents must be whole cells and <=1000000 cells")
            if self.seed_region_xy_m is not None:
                seed = self.seed_region_xy_m
                if not (b[0] <= seed[0] < seed[1] <= b[1] and b[2] <= seed[2] < seed[3] <= b[3]):
                    raise ValueError("support seed region must lie inside the grid/qualified region")
        if self.measurement_uncertainty_m is not None:
            if not np.isfinite(self.measurement_uncertainty_m) or self.measurement_uncertainty_m < 0:
                raise ValueError("measurement_uncertainty_m must be finite and nonnegative")
        if self.intrinsics_fx_fy_cx_cy is not None:
            a = np.asarray(self.intrinsics_fx_fy_cx_cy, dtype=float)
            if a.shape != (4,) or not np.isfinite(a).all() or min(a[:2]) <= 0:
                raise ValueError("intrinsics require finite fx,fy>0,cx,cy")
        for name in ("stream_width", "stream_height"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if self.depth_representation not in (None, "uint16_mm_optical_z"):
            raise ValueError("only DEPTH_MM optical-Z uint16 is supported")
        if self.T_test_ground_from_depth_optical is not None:
            matrix = rigid_transform(self.T_test_ground_from_depth_optical)
            if matrix[2, 3] <= self.ground_height_tolerance_m:
                raise ValueError("camera height must exceed ground-height tolerance")

    @property
    def calibration_ready(self):
        fields = ("calibration_id", "mount_transform_id", "measurement_method", "stream_width",
                  "stream_height", "intrinsics_fx_fy_cx_cy", "distortion_model", "depth_representation",
                  "qualified_range_m", "T_test_ground_from_depth_optical", "bounds_xy_m",
                  "seed_region_xy_m", "measurement_uncertainty_m")
        return self.calibration_kind != "unresolved" and all(getattr(self, k) is not None for k in fields)

    @property
    def config_id(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def transform_points(points, matrix):
    return np.asarray(points) @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]
