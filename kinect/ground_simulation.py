"""Offline ray-cast floor, wall, and box face for Demo 6 (no device access)."""

import numpy as np

from .camera import CameraFrame
from .geometry import CameraIntrinsics


def synthetic_ground_frame(width: int = 160, height: int = 120) -> CameraFrame:
    """Create metric depth with floor y=0.8 m, wall z=4 m, box front z=2 m.

    The wall occupies more pixels than the ground. Intrinsics are synthetic,
    and the RGB image is illustrative, on the same synthetic ray grid.
    """
    if width < 4 or height < 4:
        raise ValueError("synthetic image dimensions must be at least 4")
    intrinsics = CameraIntrinsics(0.75 * width, 0.75 * width,
                                  (width - 1) / 2, (height - 1) / 2)
    v, u = np.indices((height, width))
    ray_x = (u - intrinsics.cx) / intrinsics.fx
    ray_y = (v - intrinsics.cy) / intrinsics.fy
    ground_z = np.full((height, width), np.inf)
    np.divide(0.8, ray_y, out=ground_z, where=ray_y > 0)
    depth_m = np.minimum(ground_z, 4.0)
    box = (np.abs(ray_x * 2) <= 0.35) & (ray_y * 2 >= 0.2) & (ray_y * 2 <= 0.8)
    depth_m[box] = np.minimum(depth_m[box], 2.0)
    depth_mm = np.rint(depth_m * 1000).astype(np.uint16)
    rgb = np.full((height, width, 3), 90, dtype=np.uint8)
    rgb[ground_z < 4] = (80, 140, 80)
    rgb[box] = (180, 80, 40)
    depth_mm[0, 0] = 0
    return CameraFrame(rgb, depth_mm, 0.0, 0.0, intrinsics,
                       "synthetic_ground", "depth_camera")
