"""Pinhole projection helpers for Kinect v1 depth frames.

Camera coordinates are measured in metres: +X points right in the image,
+Y points down, and +Z points forward from the depth camera.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for a 640x480 camera image."""

    fx: float
    fy: float
    cx: float
    cy: float


# Common Kinect v1 depth-camera values.  They are a useful default only and
# must be replaced with values calibrated for a measurement-sensitive setup.
DEFAULT_KINECT_V1_DEPTH_INTRINSICS = CameraIntrinsics(
    fx=594.21434211923247,
    fy=591.04053696870778,
    cx=339.30780975300314,
    cy=242.73913761751615,
)


def pixel_depth_to_camera_xyz(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: CameraIntrinsics = DEFAULT_KINECT_V1_DEPTH_INTRINSICS,
) -> tuple[float, float, float] | None:
    """Project one image pixel and metric depth into depth-camera coordinates.

    Returns ``None`` for invalid (non-finite or non-positive) depth.
    """
    if not np.isfinite(depth_m) or depth_m <= 0:
        return None
    x = (u - intrinsics.cx) * depth_m / intrinsics.fx
    y = (v - intrinsics.cy) * depth_m / intrinsics.fy
    return float(x), float(y), float(depth_m)


def depth_image_to_camera_points(
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics = DEFAULT_KINECT_V1_DEPTH_INTRINSICS,
) -> np.ndarray:
    """Convert valid uint16 millimetre depths to an ``(N, 3)`` metre cloud."""
    if depth_mm.ndim != 2:
        raise ValueError("depth_mm must be a two-dimensional depth image")

    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    v, u = np.nonzero(valid)
    z = depth_mm[valid].astype(np.float64) / 1000.0
    x = (u.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (v.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    return np.column_stack((x, y, z))
