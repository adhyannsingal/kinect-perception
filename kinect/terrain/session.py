"""Phase 1 sessions: lossless frames, explicit clocks, config and calibration."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import TerrainConfig
from .contracts import FrameBundle, write_json


class SessionWriter:
    def __init__(self, directory, config: TerrainConfig, session_id: str, *, scene_metadata=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        for name in ("frames", "calibration", "config", "annotations", "outputs", "evaluation"):
            (self.directory / name).mkdir()
        config.save(self.directory / "config/effective.json")
        write_json(self.directory / "calibration/profile.json", {
            k: v for k, v in config.__dict__.items() if k in (
                "calibration_id", "calibration_kind", "mount_transform_id", "measurement_method",
                "T_test_ground_from_depth_optical", "intrinsics_fx_fy_cx_cy", "distortion_model",
                "stream_width", "stream_height", "depth_representation", "qualified_range_m", "measurement_uncertainty_m")})
        write_json(self.directory / "manifest.json", dict(
            schema_version=1, session_id=session_id, config_id=config.config_id,
            created_utc=datetime.now(timezone.utc).isoformat(),
            depth_format="uint16_mm_optical_z", depth_frame_id="depth_optical",
            device_model="analytic pinhole" if config.calibration_kind == "synthetic" else "Xbox 360 Kinect v1 (serial unmeasured)",
            stream_modes=dict(depth="DEPTH_MM", rgb="VIDEO_RGB optional; unregistered"),
            coordinate_frame="test_ground", rgb_registration_enabled=False,
            clock_policy="frame clock domain; live capture uses labelled receive-time proxy",
            scene_metadata=scene_metadata or {},
            hardware_qualified=config.calibration_kind == "qualified"))
        self._index = (self.directory / "frame_index.jsonl").open("x")
        self.session_id = session_id
        self.config = config
        self._previous = None
        self._clock = None

    def write(self, frame: FrameBundle, dropped_frames=0):
        if self._index.closed:
            raise RuntimeError("session writer is closed")
        if frame.session_id != self.session_id:
            raise ValueError("session mismatch")
        if (frame.calibration_id, frame.mount_transform_id) != (self.config.calibration_id, self.config.mount_transform_id):
            raise ValueError("frame calibration/mount does not match session")
        if self._clock is not None and frame.timestamp_clock_domain != self._clock:
            raise ValueError("clock domain changed within a session")
        if self._previous and (frame.sequence_id <= self._previous[0] or frame.receive_timestamp < self._previous[1]):
            raise ValueError("frames must retain original increasing sequence and nondecreasing time")
        path = self.directory / "frames" / f"{frame.sequence_id:08d}.npz"
        temporary = path.with_suffix(".partial")
        arrays = dict(depth_raw=frame.depth_raw, metadata=np.array(json.dumps(frame.metadata(), allow_nan=False)))
        if frame.rgb is not None:
            arrays["rgb"] = frame.rgb
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            temporary.rename(path)
        finally:
            temporary.unlink(missing_ok=True)
        missing_sequences = frame.sequence_id - self._previous[0] - 1 if self._previous else frame.sequence_id
        entry = dict(sequence_id=frame.sequence_id, receive_timestamp=frame.receive_timestamp,
                     path=path.relative_to(self.directory).as_posix(),
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     dropped_frames_total=dropped_frames, missing_sequences_before=missing_sequences)
        self._index.write(json.dumps(entry, allow_nan=False) + "\n")
        self._index.flush()
        self._previous = (frame.sequence_id, frame.receive_timestamp)
        self._clock = frame.timestamp_clock_domain

    def close(self):
        self._index.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SessionReplay:
    """Replay only indexed complete frames; algorithm time is recorded time."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("schema_version") != 1:
            raise ValueError("unsupported session schema")
        self.config = TerrainConfig.load(self.directory / "config/effective.json")
        if self.config.config_id != self.manifest["config_id"]:
            raise ValueError("session configuration hash mismatch")

    def __iter__(self):
        previous, domain = None, None
        with (self.directory / "frame_index.jsonl").open() as index:
            for line in index:
                entry = json.loads(line)
                relative = Path(entry["path"])
                if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "frames":
                    raise ValueError("invalid session frame path")
                path = self.directory / relative
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError(f"frame checksum mismatch: {relative}")
                with path.open("rb") as stream, np.load(stream, allow_pickle=False) as data:
                    frame = FrameBundle(depth_raw=data["depth_raw"],
                                        rgb=data["rgb"] if "rgb" in data else None,
                                        **json.loads(str(data["metadata"].item())))
                if frame.session_id != self.manifest["session_id"] or frame.sequence_id != entry["sequence_id"]:
                    raise ValueError("frame/index identity mismatch")
                if frame.receive_timestamp != entry["receive_timestamp"]:
                    raise ValueError("frame/index timestamp mismatch")
                if (frame.calibration_id, frame.mount_transform_id) != (self.config.calibration_id, self.config.mount_transform_id):
                    raise ValueError("recorded calibration mismatch")
                if previous and (frame.sequence_id <= previous[0] or frame.receive_timestamp < previous[1]):
                    raise ValueError("recording sequence/time is not ordered")
                if domain is not None and domain != frame.timestamp_clock_domain:
                    raise ValueError("recording clock changed")
                previous, domain = (frame.sequence_id, frame.receive_timestamp), frame.timestamp_clock_domain
                yield frame
