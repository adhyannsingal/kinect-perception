"""One headless pipeline for live and replay; display consumes these outputs."""

from datetime import datetime, timezone
import time

import numpy as np

from .contracts import GroundModel, TerrainResult
from .depth_quality import calibrated_points, reference_visibility
from .grid import StationaryHistory, build_grid, cell_indices, observations_from
from .ground import fit_support


class TerrainPipeline:
    def __init__(self, config, *, accumulate=False):
        self.config = config
        self._config_id = config.config_id
        self.accumulate = accumulate
        self.reset()

    def reset(self):
        self._session = None
        self._last_sequence = None
        self._last_receive = None
        self._last_domain = None
        self._device_timestamp = None
        self._device_change_time = None
        self._last_plane = None
        self._reset_required = False
        self._protocol_error = False
        self.history = StationaryHistory(self.config)

    def process(self, frame, *, now=None, dropped_frames=0):
        started = time.monotonic()
        # For replay, default algorithmic time is the recorded receive time.
        # A live caller explicitly passes its current monotonic clock.
        now = frame.receive_timestamp if now is None else now
        if not np.isfinite(now) or now < frame.receive_timestamp:
            raise ValueError("processing time must share the frame clock and not precede receive time")
        if self._session != frame.session_id:
            self.reset()
            self._session = frame.session_id
        state, fault = "RUNNING", ""
        age = now - frame.depth_capture_timestamp
        if not self.config.calibration_ready:
            state, fault = "CALIBRATION_REQUIRED", "complete calibration, range, uncertainty, bounds and seed profile"
        elif self.config.config_id != self._config_id:
            state, fault = "CALIBRATION_REQUIRED", "configuration changed; construct a new pipeline/session"
        elif (frame.calibration_id, frame.mount_transform_id) != (self.config.calibration_id, self.config.mount_transform_id):
            state, fault = "CALIBRATION_REQUIRED", "frame calibration or mount ID does not match profile"
            self._reset_required = True
        elif frame.depth_raw.shape != (self.config.stream_height, self.config.stream_width):
            state, fault = "CALIBRATION_REQUIRED", "depth resolution does not match calibrated intrinsics"
        elif self._last_sequence is not None and (frame.sequence_id <= self._last_sequence or
                 frame.receive_timestamp < self._last_receive or frame.timestamp_clock_domain != self._last_domain):
            state, fault = "ERROR", "sequence/time/clock discontinuity; restart session"
            self._protocol_error = True
        if self._protocol_error:
            state, fault = "ERROR", "sequence/time/clock discontinuity; restart session"
        if frame.device_timestamp_optional is not None:
            if self._device_timestamp != frame.device_timestamp_optional:
                self._device_timestamp = frame.device_timestamp_optional
                self._device_change_time = frame.receive_timestamp
            if (state not in ("ERROR", "CALIBRATION_REQUIRED") and self._device_change_time is not None
                    and now - self._device_change_time > self.config.maximum_input_age_s):
                state, fault = "STALE", "device timestamp frozen despite host receives"
        if age > self.config.maximum_input_age_s and state not in ("ERROR", "CALIBRATION_REQUIRED"):
            state, fault = "STALE", "input age exceeds freshness limit"
        self._last_sequence, self._last_receive = frame.sequence_id, frame.receive_timestamp
        self._last_domain = frame.timestamp_clock_domain
        ready = (self.config.calibration_ready and self.config.config_id == self._config_id and
                 (frame.calibration_id, frame.mount_transform_id) == (self.config.calibration_id, self.config.mount_transform_id) and
                 frame.depth_raw.shape == (self.config.stream_height, self.config.stream_width) and
                 state != "CALIBRATION_REQUIRED")
        if ready:
            optical, points, pixels, valid, original_valid = calibrated_points(frame, self.config)
            reasons = reference_visibility(frame, self.config)
            model = fit_support(optical, points, self.config, self._last_plane)
            if model.status == "INCONSISTENT":
                self._reset_required = True
            if self._reset_required:
                model.status, model.reason = "INCONSISTENT", "mount/support changed; new calibrated session required"
            if state in ("STALE", "ERROR"):
                model.status, model.reason = "INSUFFICIENT_SUPPORT", fault
            if model.status == "VALID":
                self._last_plane = model.plane.copy()
            elif state == "RUNNING":
                state, fault = "DEGRADED", model.reason
            if self.config.calibration_kind != "qualified" and state == "RUNNING":
                state, fault = "DEGRADED", f"{self.config.calibration_kind}: physical qualification incomplete"
            residual = points @ model.plane[:3] + model.plane[3] if model.status == "VALID" else np.full(len(points), np.nan)
            grid = build_grid(points, residual, model, self.config, frame.receive_timestamp, reasons)
            if state == "STALE":
                grid["unknown_reason"][:] = "STALE"
            model.source_sequence_id = frame.sequence_id
            model.source_timestamp = frame.depth_capture_timestamp
            model.model_id = f"{frame.session_id}:ground:{frame.sequence_id}"
            model.support_indices = np.flatnonzero(np.abs(residual) <= self.config.ground_distance_m).tolist() if model.status == "VALID" else []
            observations = observations_from(grid, self.config, frame.sequence_id, frame.receive_timestamp)
            if self.accumulate:
                usable = model.status == "VALID" and state not in ("STALE", "ERROR")
                history = self.history.update(grid, now, usable=usable)
            else:
                history = None
        else:
            points, residual, pixels = np.empty((0, 3)), np.empty(0), np.empty((0, 2), dtype=int)
            original_valid = (frame.depth_raw > 0) & (frame.depth_raw <= 10000)
            valid = np.zeros(frame.depth_raw.shape, dtype=bool)
            model = GroundModel("INSUFFICIENT_SUPPORT", fault)
            grid, history, observations = {}, None, []
        duration = time.monotonic() - started
        point_evidence = np.full(len(points), "UNKNOWN", dtype="<U20")
        if grid:
            cells, inside, _ = cell_indices(points, self.config)
            point_evidence[inside] = grid["summary"][tuple(cells[inside].T)]
        metadata = dict(
            schema_version=1, session_id=frame.session_id, source_sequence_id=frame.sequence_id,
            source_timestamp=frame.depth_capture_timestamp, timestamp_clock_domain=frame.timestamp_clock_domain,
            publication_utc=datetime.now(timezone.utc).isoformat(), frame_id="test_ground",
            depth_frame_id="depth_optical", calibration_id=self.config.calibration_id,
            mount_transform_id=self.config.mount_transform_id, config_id=self.config.config_id,
            effective_configuration=self.config.__dict__.copy(), grid_id=f"grid:{frame.sequence_id}",
            resolution_m=self.config.resolution_m, bounds_xy_m=self.config.bounds_xy_m,
            origin_xy_m=[self.config.bounds_xy_m[0], self.config.bounds_xy_m[2]] if self.config.bounds_xy_m else None,
            dimensions_yx=list(grid["point_count"].shape) if grid else None,
            point_pixels_vu=pixels, history=history,
            point_evidence=point_evidence,
            history_is_stationary_accumulation=self.accumulate,
            history_metadata=dict(frame_id="test_ground", as_of_timestamp=now,
                                  timestamp_clock_domain=frame.timestamp_clock_domain,
                                  clearing_policy="hazards persist until explicit reset/new session") if self.accumulate else None,
            geometry_qualification=self.config.calibration_kind,
            note="observed geometry only; no rover-specific safety classification")
        health = dict(
            state=state, fault_reason=fault, source=frame.source, source_age_s=age,
            age_semantics="receive proxy; transport delay unmeasured" if frame.capture_time_is_receive_proxy else "capture timestamp",
            queue_age_s=now - frame.receive_timestamp, dropped_frames=dropped_frames,
            valid_depth_fraction=float(valid.mean()), raw_valid_depth_fraction=float(original_valid.mean()),
            processing_time_s=duration, processing_capacity_hz=1 / duration if duration else None,
            processing_output_age_s=age + duration,
            target_grid_hz=self.config.target_grid_hz, reset_required=self._reset_required,
            rgb_pairing="unregistered; RGB display only", ground_status=model.status)
        return TerrainResult(metadata, health, model, grid, observations, points, residual, valid)
