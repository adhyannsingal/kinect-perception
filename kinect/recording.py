"""Lossless per-frame NPZ recording, with JSON metadata and no pickle."""

from dataclasses import asdict
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from .camera import Camera, CameraFrame
from .geometry import CameraIntrinsics


class RecordingWriter:
    """Create a new recording directory; existing paths are never overwritten.

    Frames become visible via rename only after their file is complete. Each
    frame carries its own version and metadata, so completed frames remain
    usable if capture stops unexpectedly. Closing an empty writer is allowed.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self._index = 0
        self._closed = False

    def write(self, frame: CameraFrame) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        metadata = dict(version=1, rgb_timestamp_s=frame.rgb_timestamp_s,
                        depth_timestamp_s=frame.depth_timestamp_s,
                        intrinsics=asdict(frame.intrinsics) if frame.intrinsics else None,
                        source=frame.source, depth_space=frame.depth_space)
        path = self.directory / f"{self._index:08d}.npz"
        temporary = path.with_suffix(".partial")
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, rgb=frame.rgb, depth_mm=frame.depth_mm,
                                    metadata=np.array(json.dumps(metadata, allow_nan=False)))
            temporary.rename(path)
        finally:
            temporary.unlink(missing_ok=True)
        self._index += 1

    def close(self) -> None:
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ReplayCamera(Camera):
    """Read completed frames in order, without sleeping or altering timestamps.

    EOF is stable until rewind(). Corrupt frames raise ValueError rather than
    being silently skipped. Incomplete .partial files are ignored.
    """

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        self._paths = sorted(directory.glob("*.npz"))
        if any(path.name != f"{i:08d}.npz" for i, path in enumerate(self._paths)):
            raise ValueError("recording frames must have contiguous numbered filenames")
        self._index = 0
        self._closed = False

    def read(self) -> CameraFrame | None:
        if self._closed:
            raise RuntimeError("camera is closed")
        if self._index >= len(self._paths):
            return None
        path = self._paths[self._index]
        try:
            with path.open("rb") as stream, np.load(stream, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
                if metadata.pop("version") != 1:
                    raise ValueError("unsupported recording version")
                calibration = metadata.pop("intrinsics")
                frame = CameraFrame(data["rgb"], data["depth_mm"],
                                    intrinsics=CameraIntrinsics(**calibration) if calibration else None,
                                    **metadata)
        except (ValueError, TypeError, KeyError, OSError, EOFError, BadZipFile) as error:
            raise ValueError(f"Invalid recorded frame {path}: {error}") from error
        self._index += 1
        return frame

    def rewind(self) -> None:
        if self._closed:
            raise RuntimeError("camera is closed")
        self._index = 0

    def close(self) -> None:
        self._closed = True
