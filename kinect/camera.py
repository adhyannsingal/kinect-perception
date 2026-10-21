"""Opt-in frame sources; the original Kinect demos do not use this module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
import math

import numpy as np

from .geometry import CameraIntrinsics, DEFAULT_KINECT_V1_DEPTH_INTRINSICS


@dataclass(frozen=True)
class CameraFrame:
    """Owned RGB uint8 and depth uint16 arrays, with independent times in seconds.

    Depth is in millimetres; zero indicates missing depth. Equal image sizes
    do not imply registration. Intrinsics describe only the depth image.
    """

    rgb: np.ndarray
    depth_mm: np.ndarray
    rgb_timestamp_s: float
    depth_timestamp_s: float
    intrinsics: CameraIntrinsics | None
    source: str
    depth_space: str = "depth_camera"

    def __post_init__(self) -> None:
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("rgb must be an HxWx3 uint8 RGB image")
        if self.depth_mm.dtype != np.uint16 or self.depth_mm.ndim != 2:
            raise ValueError("depth_mm must be an HxW uint16 image")
        if not self.rgb.size or not self.depth_mm.size:
            raise ValueError("images must not be empty")
        if not all(math.isfinite(t) and t >= 0 for t in
                   (self.rgb_timestamp_s, self.depth_timestamp_s)):
            raise ValueError("timestamps must be finite nonnegative seconds")
        if self.depth_space not in ("depth_camera", "rgb_camera"):
            raise ValueError("unknown depth coordinate space")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a nonempty string")
        if self.intrinsics is not None:
            values = (self.intrinsics.fx, self.intrinsics.fy,
                      self.intrinsics.cx, self.intrinsics.cy)
            if not all(math.isfinite(v) for v in values) or min(values[:2]) <= 0:
                raise ValueError("intrinsics must be finite with positive focal lengths")
        # Native bindings can reuse their capture buffers. Keep frame ownership
        # independent of subsequent reads and caller-provided arrays.
        object.__setattr__(self, "rgb", self.rgb.copy())
        object.__setattr__(self, "depth_mm", self.depth_mm.copy())


class Camera(ABC):
    """Pull-based source. None means EOF; capture errors raise exceptions."""

    @abstractmethod
    def read(self) -> CameraFrame | None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class KinectCamera(Camera):
    """Lazy libfreenect adapter using existing DEPTH_MM device calibration.

    RGB/depth are acquired independently, without claiming temporal alignment
    or RGB registration. Device timestamps are converted from ms to seconds
    and may wrap with the device clock. The sync API has process-wide state:
    use only one live adapter at a time.
    """

    def __init__(self, device_index: int = 0,
                 intrinsics: CameraIntrinsics = DEFAULT_KINECT_V1_DEPTH_INTRINSICS):
        if device_index < 0:
            raise ValueError("device_index must be nonnegative")
        self._device_index = device_index
        self._intrinsics = intrinsics
        self._driver = None
        self._closed = False

    def read(self) -> CameraFrame:
        if self._closed:
            raise RuntimeError("camera is closed")
        if self._driver is None:
            self._driver = importlib.import_module("freenect")
        driver = self._driver
        rgb = driver.sync_get_video(index=self._device_index, format=driver.VIDEO_RGB)
        depth = driver.sync_get_depth(index=self._device_index, format=driver.DEPTH_MM)
        if rgb is None or depth is None:
            raise RuntimeError("Could not retrieve Kinect RGB/depth frames")
        return CameraFrame(rgb[0], depth[0], float(rgb[1]) / 1000,
                           float(depth[1]) / 1000, self._intrinsics, "kinect_v1")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._driver is not None:
                self._driver.sync_stop()
