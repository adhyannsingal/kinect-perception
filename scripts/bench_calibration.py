"""Collect measured targets and review a candidate fixed Kinect bench transform."""

from contextlib import ExitStack
import argparse
import itertools
import json
from pathlib import Path
import sys
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinect.geometry import CameraIntrinsics, DEFAULT_KINECT_V1_DEPTH_INTRINSICS
from kinect.recording import ReplayCamera
from kinect.terrain.acquisition import LatestFrameSource
from kinect.terrain.bench_calibration import build_candidate, calibration_plan, sample_target
from kinect.terrain.config import TerrainConfig
from kinect.terrain.contracts import write_json
from kinect.terrain.session import SessionReplay


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def pick_pixel(depth):
    """Select on the depth grid; never use unregistered RGB coordinates."""
    import cv2
    selected = []
    name = "Bench calibration: click target in DEPTH image; Q cancels"
    scaled = np.clip(depth.astype(float) / 10000 * 255, 0, 255).astype(np.uint8)
    image = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    image[(depth == 0) | (depth > 10000)] = (180, 0, 180)
    def click(event, u, v, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected.append((u, v))
    cv2.namedWindow(name)
    cv2.setMouseCallback(name, click)
    try:
        while not selected:
            cv2.imshow(name, image)
            if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                raise ValueError("target selection cancelled")
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("target selection window closed")
    finally:
        cv2.destroyWindow(name)
    return selected[0]


def observe(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if args.frames < 3 or args.radius < 0:
        raise ValueError("need at least three frames and a nonnegative ROI radius")
    override = CameraIntrinsics(*args.intrinsics) if args.intrinsics else None
    with ExitStack() as stack:
        if args.live:
            source = stack.enter_context(LatestFrameSource(TerrainConfig(), str(uuid.uuid4())))
            def live_frames():
                previous = None
                for _ in range(args.frames):
                    frame = source.get(timeout=args.timeout)
                    device = frame.device_timestamp_optional
                    if previous is not None and device == previous:
                        raise ValueError("device timestamp frozen while collecting calibration samples")
                    previous = device
                    yield frame.depth_raw, frame.receive_timestamp, f"live:{frame.sequence_id}"
            frames = live_frames()
            intrinsics = override or DEFAULT_KINECT_V1_DEPTH_INTRINSICS
            clock_domain = "host_monotonic_receive_proxy; device freeze checked independently"
            source_kind = "live_kinect_DEPTH_MM"
            source_calibration_kind = "bench_unqualified"
        elif args.recording:
            camera = stack.enter_context(ReplayCamera(args.recording))
            first = camera.read()
            if first is None:
                raise ValueError("recording is empty")
            intrinsics = override or first.intrinsics
            if intrinsics is None:
                raise ValueError("recording lacks intrinsics; supply --intrinsics")
            def legacy_frames():
                frame, index = first, 0
                while frame is not None and index < args.frames:
                    if frame.depth_space != "depth_camera" or frame.intrinsics != first.intrinsics:
                        raise ValueError("depth coordinate grid/intrinsics changed or is RGB-registered")
                    yield frame.depth_mm, frame.depth_timestamp_s, f"{args.recording.resolve()}#{index}"
                    index += 1
                    frame = camera.read()
            frames = legacy_frames()
            clock_domain, source_kind = "legacy_device_seconds", "legacy_recording"
            source_calibration_kind = "synthetic" if first.source.startswith(("simulat", "synthetic")) else "bench_unqualified"
        elif args.session:
            replay = SessionReplay(args.session)
            if replay.config.mount_transform_id is not None and replay.config.mount_transform_id != args.mount_id:
                raise ValueError("requested mount ID does not match the recorded session")
            values = replay.config.intrinsics_fx_fy_cx_cy
            if override is None and values is None:
                raise ValueError("session lacks intrinsics; supply --intrinsics")
            intrinsics = override or CameraIntrinsics(*values)
            iterator = iter(replay)
            first = next(iterator, None)
            if first is None:
                raise ValueError("session is empty")
            frames = ((f.depth_raw, f.receive_timestamp, f"{args.session.resolve()}#{f.sequence_id}")
                      for f in itertools.islice(itertools.chain([first], iterator), args.frames))
            clock_domain, source_kind = first.timestamp_clock_domain, "phase1_session"
            source_calibration_kind = replay.config.calibration_kind
        else:
            if override is None:
                raise ValueError("NPY observations require explicit --intrinsics FX FY CX CY")
            intrinsics = override
            def npy_frames():
                for index, path in enumerate(args.npy[:args.frames]):
                    with path.open("rb") as stream:
                        depth = np.load(stream, allow_pickle=False)
                        if not isinstance(depth, np.ndarray):
                            depth.close()
                            raise ValueError("expected NPY depth arrays")
                    yield depth, float(index), str(path.resolve())
            frames = npy_frames()
            clock_domain, source_kind = "assigned_frame_indices_not_capture_time", "npy_import"
            source_calibration_kind = "unresolved"
        frames = iter(frames)
        first_sample = next(frames, None)
        if first_sample is None:
            raise ValueError("no input frames")
        if args.live and first_sample[0].shape != (480, 640) and override is None:
            raise ValueError("default Kinect intrinsics require a 640x480 depth stream")
        pixel = pick_pixel(first_sample[0]) if args.pick else tuple(args.pixel)
        result = sample_target(itertools.chain([first_sample], frames), pixel, intrinsics,
                               mount_id=args.mount_id, target_id=args.target_id, radius_px=args.radius,
                               minimum_valid_fraction=args.minimum_valid_fraction,
                               maximum_spread_m=args.maximum_spread)
        result.update(timestamp_clock_domain=clock_domain, source_kind=source_kind,
                      source_calibration_kind=source_calibration_kind,
                      intrinsics_status="caller/recording pinhole parameters; accuracy unqualified",
                      observation_purpose=args.description)
        write_json(args.output, result)
        print(json.dumps(dict(status=result["status"], frames=result["frame_count"],
                              valid_fraction=result["valid_sample_fraction"],
                              spread_m=result["depth_p95_p5_spread_m"],
                              output=str(args.output), hardware_qualified=False)))
        return result["status"] == "ACCEPTED_SAMPLE"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="create a plan with unresolved physical target coordinates")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--mount-id", required=True)
    sample = commands.add_parser("observe", help="sample one identified target over repeated depth frames")
    source = sample.add_mutually_exclusive_group(required=True)
    source.add_argument("--live", action="store_true")
    source.add_argument("--recording", type=Path, help="existing RecordingWriter directory")
    source.add_argument("--session", type=Path, help="Phase 1 session directory")
    source.add_argument("--npy", nargs="+", type=Path)
    pixel = sample.add_mutually_exclusive_group(required=True)
    pixel.add_argument("--pixel", nargs=2, type=int, metavar=("U", "V"))
    pixel.add_argument("--pick", action="store_true", help="click target on the depth image")
    sample.add_argument("--frames", type=positive_int, default=30)
    sample.add_argument("--intrinsics", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"))
    sample.add_argument("--radius", type=int, default=0, help="ROI radius in depth pixels; default samples selected pixel only")
    sample.add_argument("--minimum-valid-fraction", type=float, default=0.8)
    sample.add_argument("--maximum-spread", type=float, default=0.02, help="p95-p5 stability limit in metres")
    sample.add_argument("--timeout", type=float, default=15, help="seconds to wait for each live frame")
    sample.add_argument("--mount-id", required=True)
    sample.add_argument("--target-id", required=True)
    sample.add_argument("--description", default="identified depth target; physical accuracy not established")
    sample.add_argument("--output", type=Path, required=True)
    fit = commands.add_parser("fit", help="write a NEW unqualified candidate profile and held-out report")
    fit.add_argument("--plan", type=Path, required=True)
    fit.add_argument("--config", type=Path, required=True, help="base terrain config; never overwritten")
    fit.add_argument("--output", type=Path, required=True, help="new result directory")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        if args.command == "plan":
            write_json(args.output, calibration_plan(args.mount_id))
        elif args.command == "observe":
            if not observe(args):
                parser.exit(2, "sample rejected; inspect the saved diagnostic and reacquire the target\n")
        else:
            plan = json.loads(args.plan.read_text())
            observations = {}
            for target in plan["targets"]:
                if not target["observation_path"]:
                    raise ValueError(f"{target['target_id']}: observation_path is unresolved")
                path = Path(target["observation_path"])
                if not path.is_absolute():
                    path = args.plan.parent / path
                observations[target["target_id"]] = json.loads(path.read_text())
            candidate, report = build_candidate(plan, observations, TerrainConfig.load(args.config))
            args.output.mkdir(parents=True, exist_ok=False)
            candidate.save(args.output / "candidate_profile.json")
            write_json(args.output / "validation_report.json", report)
            write_json(args.output / "measurements_snapshot.json", dict(plan=plan, observations=observations))
            print(json.dumps(dict(status=report["status"], hardware_qualified=False,
                                  within_declared_error_budget=report["within_declared_error_budget"],
                                  profile_complete=candidate.calibration_ready, output=str(args.output))))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, EOFError, TimeoutError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
