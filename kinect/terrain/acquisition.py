"""Bounded latest-frame acquisition wrapping the unchanged Kinect adapter."""

from queue import Empty, Full, Queue
from threading import Event, Thread
import time

from ..camera import KinectCamera
from .contracts import FrameBundle


class LatestFrameSource:
    """One queued frame, explicit drops, and no unbounded capture backlog.

    get(timeout) raises TimeoutError for a disconnected/frozen native read;
    caller must publish STALE. Native USB calls cannot be forcibly interrupted
    safely here. A daemon worker cleans up the adapter when the call returns.
    """

    def __init__(self, config, session_id, *, camera=None, clock=time.monotonic):
        if camera is None and config.calibration_kind == "synthetic":
            raise ValueError("live capture must never use a synthetic profile")
        self.config, self.session_id = config, session_id
        self.camera = camera if camera is not None else KinectCamera()
        self.clock = clock
        self.queue = Queue(maxsize=1)
        self.dropped_frames = 0
        self.error = None
        self._stop = Event()
        self._thread = Thread(target=self._run, name="terrain-capture", daemon=True)

    def _run(self):
        try:
            sequence = 0
            while not self._stop.is_set():
                capture = self.camera.read()
                if capture is None:
                    break
                receive = self.clock()
                frame = FrameBundle(
                    self.session_id, sequence, capture.depth_mm, receive, receive,
                    "host_monotonic_receive_proxy", self.config.calibration_id,
                    self.config.mount_transform_id, rgb=capture.rgb,
                    device_timestamp_optional=capture.depth_timestamp_s,
                    rgb_device_timestamp_optional=capture.rgb_timestamp_s,
                    source=capture.source,
                )
                sequence += 1
                while True:
                    try:
                        self.queue.put_nowait(frame)
                        break
                    except Full:
                        try:
                            self.queue.get_nowait()
                            self.dropped_frames += 1
                        except Empty:
                            pass
        except Exception as error:
            self.error = error
        finally:
            try:
                self.camera.close()
            except Exception as error:
                self.error = self.error or error

    def get(self, timeout=0.5):
        if self.error is not None:
            raise RuntimeError(f"capture failed: {self.error}") from self.error
        try:
            return self.queue.get(timeout=timeout)
        except Empty as error:
            if self.error is not None:
                raise RuntimeError(f"capture failed: {self.error}") from self.error
            raise TimeoutError("no new depth frame; current output is stale") from error

    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=1)
        return not self._thread.is_alive()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self.close()
