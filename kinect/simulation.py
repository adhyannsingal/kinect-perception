"""Deterministic synthetic RGB/depth for hardware-independent development."""

import math

import numpy as np

from .camera import Camera, CameraFrame
from .geometry import CameraIntrinsics


class SimulatedCamera(Camera):
    """Plane at 2 m with a moving 1 m square and a missing-depth corner.

    RGB and depth share a synthetic pinhole grid. Times start at zero and
    advance by 1/fps; reads do not sleep. No Kinect calibration is modified.
    """

    def __init__(self, width: int = 640, height: int = 480,
                 fps: float = 30, frame_count: int = 300):
        if width <= 0 or height <= 0 or frame_count < 0:
            raise ValueError("dimensions must be positive and frame_count nonnegative")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        self.width, self.height = width, height
        self.fps, self.frame_count = fps, frame_count
        self.intrinsics = CameraIntrinsics(float(width), float(width),
                                           (width - 1) / 2, (height - 1) / 2)
        self._index = 0
        self._closed = False

    def read(self) -> CameraFrame | None:
        if self._closed:
            raise RuntimeError("camera is closed")
        if self._index >= self.frame_count:
            return None
        depth = np.full((self.height, self.width), 2000, dtype=np.uint16)
        rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)
        rgb[:] = (30, 60, 120)
        side = max(1, min(self.width, self.height) // 4)
        x = self._index % (self.width - side + 1)
        y = (self.height - side) // 2
        depth[y:y + side, x:x + side] = 1000
        rgb[y:y + side, x:x + side] = (240, 100, 40)
        depth[0, 0] = 0
        timestamp = self._index / self.fps
        self._index += 1
        return CameraFrame(rgb, depth, timestamp, timestamp, self.intrinsics,
                           "simulation", "rgb_camera")

    def close(self) -> None:
        self._closed = True
