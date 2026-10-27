"""Frame/output contracts; exported NaNs become JSON null, never fake zeros."""

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path

import numpy as np


def serializable(value):
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n")


@dataclass
class FrameBundle:
    session_id: str
    sequence_id: int
    depth_raw: np.ndarray
    depth_capture_timestamp: float
    receive_timestamp: float
    timestamp_clock_domain: str
    calibration_id: str | None
    mount_transform_id: str | None
    rgb: np.ndarray | None = None
    rgb_capture_timestamp: float | None = None
    device_timestamp_optional: float | None = None
    capture_time_is_receive_proxy: bool = True
    source: str = "kinect_v1"
    depth_format: str = "uint16_mm_optical_z"
    schema_version: int = 1
    depth_frame_id: str = "depth_optical"
    depth_conversion_id: str = "uint16_mm_times_0.001"
    device_to_host_clock_mapping_optional: dict | None = None
    pairing_error_optional: float | None = None
    rgb_device_timestamp_optional: float | None = None

    def __post_init__(self):
        if self.schema_version != 1 or self.depth_format != "uint16_mm_optical_z":
            raise ValueError("unsupported frame schema/depth format")
        if self.depth_frame_id != "depth_optical" or self.depth_conversion_id != "uint16_mm_times_0.001":
            raise ValueError("unsupported depth coordinate frame or conversion")
        if self.depth_raw.dtype != np.uint16 or self.depth_raw.ndim != 2 or not self.depth_raw.size:
            raise ValueError("depth_raw must be a nonempty 2D uint16 DEPTH_MM image")
        if type(self.sequence_id) is not int or self.sequence_id < 0 or not self.session_id:
            raise ValueError("invalid session/sequence ID")
        if not self.timestamp_clock_domain:
            raise ValueError("clock domain required")
        for value in (self.depth_capture_timestamp, self.receive_timestamp,
                      self.device_timestamp_optional, self.rgb_capture_timestamp, self.rgb_device_timestamp_optional):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError("timestamps must be finite nonnegative seconds")
        if self.receive_timestamp < self.depth_capture_timestamp:
            raise ValueError("capture and receive time must share a domain and ordered times")
        if self.rgb is not None and (self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3):
            raise ValueError("RGB must be uint8 HxWx3")
        self.depth_raw = self.depth_raw.copy()
        if self.rgb is not None:
            self.rgb = self.rgb.copy()

    def metadata(self):
        return {k: v for k, v in self.__dict__.items() if k not in ("depth_raw", "rgb")}

    @property
    def depth_m(self):
        return self.depth_raw.astype(float) / 1000


@dataclass
class GroundModel:
    status: str
    reason: str
    plane: np.ndarray | None = None
    support_spatial_coverage: float = 0.0
    residual_median_m: float = float("nan")
    residual_p95_m: float = float("nan")
    uncertainty_proxy_m: float = float("nan")
    frame_id: str = "test_ground"
    source_sequence_id: int = 0
    source_timestamp: float = 0.0
    model_id: str = ""
    support_indices: list | None = None


@dataclass
class TerrainResult:
    metadata: dict
    health: dict
    ground: GroundModel
    grid: dict
    observations: list
    points_test_ground: np.ndarray
    point_residual_m: np.ndarray
    depth_valid_mask: np.ndarray

    def export(self, path):
        payload = asdict(self)
        payload["ground"]["unit_normal"] = self.ground.plane[:3] if self.ground.plane is not None else None
        payload["ground"]["plane_offset"] = self.ground.plane[3] if self.ground.plane is not None else None
        write_json(path, payload)
