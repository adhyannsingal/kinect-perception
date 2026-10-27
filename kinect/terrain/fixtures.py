"""Small deterministic sensor fixtures, distinct from physical qualification."""

import numpy as np

from .config import TerrainConfig
from .contracts import FrameBundle


LAYOUTS = ("flat", "raised", "depression", "dropout", "edge", "mixed", "wall")


def synthetic_config():
    # Optical (+X right,+Y down,+Z forward) -> ground (+x forward,+y left,+z up).
    return TerrainConfig(
        calibration_id="synthetic-pinhole-v1", mount_transform_id="synthetic-fixed-v1",
        calibration_kind="synthetic", measurement_method="analytic ray casting; NOT physical calibration",
        stream_width=320, stream_height=240, intrinsics_fx_fy_cx_cy=[220, 220, 159.5, 44],
        distortion_model="none_synthetic", depth_representation="uint16_mm_optical_z",
        qualified_range_m=[0.5, 5.0],
        T_test_ground_from_depth_optical=[[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0.8], [0, 0, 0, 1]],
        bounds_xy_m=[0.8, 4.0, -1.2, 1.2], seed_region_xy_m=[1.0, 1.8, -0.8, 0.8],
        measurement_uncertainty_m=0.005, ground_iterations=250,
        # Coarser cells in this LOW-resolution synthetic profile are deliberate.
        # The bench template retains the specification's 0.05 m default.
        resolution_m=0.1, minimum_support_coverage=0.5, minimum_cell_points=2,
        minimum_ground_coverage=0.4)


def fixture(layout="mixed", sequence_id=0, *, config=None, session_id="synthetic-development", timestamp=None):
    """Ray-cast nearest surfaces, preserving box occlusion and missing returns.

    Floor=0; box=+0.22 m; depression=-0.18 m. Edge includes a measured
    downward face, with the floor behind it hidden. Dropout has no geometry.
    Millimetre quantization is retained; these are logic fixtures, not noise models.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown fixture layout {layout}")
    config = config or synthetic_config()
    h, w = config.stream_height, config.stream_width
    fx, fy, cx, cy = config.intrinsics_fx_fy_cx_cy
    v, u = np.indices((h, w))
    rx, ry = (u - cx) / fx, (v - cy) / fy
    valid_ray = ry > 0
    ground_depth = np.full((h, w), np.inf)
    np.divide(0.8, ry, out=ground_depth, where=valid_ray)
    depth = ground_depth.copy()
    annotations = []
    if layout in ("raised", "mixed"):
        # Analytic axis-aligned box; intersect all six faces and keep the nearest.
        lo, hi = np.array([2.0, -0.65, 0]), np.array([2.45, -0.25, 0.22])
        origin = np.array([0, 0, 0.8])
        rays = np.stack((np.ones_like(rx), -rx, -ry), axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t0, t1 = (lo - origin) / rays, (hi - origin) / rays
        near = np.max(np.minimum(t0, t1), axis=-1)
        far = np.min(np.maximum(t0, t1), axis=-1)
        box = (far >= np.maximum(near, 0)) & (near > 0)
        depth[box] = np.minimum(depth[box], near[box])
        annotations.append(dict(type="RAISED_GEOMETRY", bounds_xy_m=[2.0, 2.45, -0.65, -0.25], height_m=0.22))
    if layout in ("depression", "mixed"):
        patch = ((ground_depth >= 2.0) & (ground_depth < 2.7) &
                 (-rx * ground_depth >= 0.2) & (-rx * ground_depth < 0.6))
        # Intersect depression floor; remove rays that hit its side walls.
        recessed = np.full_like(depth, np.inf)
        np.divide(0.98, ry, out=recessed, where=valid_ray)
        # First intersection with bottom, far wall, or lateral wall. Choosing
        # only the far wall would invent surfaces below the known pit bottom.
        lateral = np.full_like(depth, np.inf)
        np.divide(0.6, -rx, out=lateral, where=rx < 0)
        depth[patch] = np.minimum(np.minimum(recessed, 2.7), lateral)[patch]
        annotations.append(dict(type="VISIBLE_DEPRESSION", bounds_xy_m=[2.0, 2.7, 0.2, 0.6], depth_m=0.18))
    if layout == "dropout":
        patch = (ground_depth > 2) & (ground_depth < 2.5) & (np.abs(rx * ground_depth) < 0.3)
        depth[patch] = np.inf
        annotations.append(dict(type="UNKNOWN", bounds_xy_m=[2, 2.5, -0.3, 0.3]))
    if layout == "edge":
        beyond = (ground_depth >= 2.0) & (np.abs(rx * ground_depth) < 0.5)
        depth[beyond] = np.inf
        # Visible vertical face just below the rim; floor behind is absent.
        face_z = 0.8 - ry * 2.0
        face = (face_z < 0) & (face_z > -0.25) & (np.abs(rx * 2) < 0.5)
        depth[face] = 2.0
        annotations.append(dict(type="MEASURED_EDGE", bounds_xy_m=[1.9, 2.1, -0.5, 0.5]))
    if layout == "wall":
        depth[:] = 2.0
    depth[(depth < config.qualified_range_m[0]) | (depth > config.qualified_range_m[1])] = 0
    depth = np.rint(depth * 1000).astype(np.uint16)
    rgb = np.full((h, w, 3), [80, 100, 110], dtype=np.uint8)
    timestamp = sequence_id / 10 if timestamp is None else timestamp
    frame = FrameBundle(session_id, sequence_id, depth, timestamp, timestamp,
                        "synthetic_recorded_seconds", config.calibration_id, config.mount_transform_id,
                        rgb=rgb, capture_time_is_receive_proxy=False, source=f"synthetic:{layout}")
    return frame, annotations
