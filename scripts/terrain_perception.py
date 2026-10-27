"""Phase 1 stationary perception: capture, replay, inspect, and evaluate."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinect.recording import ReplayCamera
from kinect.terrain.acquisition import LatestFrameSource
from kinect.terrain.config import TerrainConfig
from kinect.terrain.contracts import FrameBundle, write_json
from kinect.terrain.evaluation import calibration_check, compare_annotations, synthetic_evaluation
from kinect.terrain.fixtures import LAYOUTS, fixture, synthetic_config
from kinect.terrain.pipeline import TerrainPipeline
from kinect.terrain.session import SessionReplay, SessionWriter


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


class RunOutput:
    def __init__(self, directory, config, mode):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        config.save(self.directory / "effective_config.json")
        write_json(self.directory / "run.json", dict(schema_version=1, mode=mode,
                    config_id=config.config_id, started_utc=datetime.now(timezone.utc).isoformat(),
                    classification="terrain evidence only; physical qualification separate"))
        self._previous_publish = None

    def write(self, frame, result):
        published = time.monotonic()
        result.health["grid_publication_interval_s"] = published - self._previous_publish if self._previous_publish else None
        self._previous_publish = published
        result.export(self.directory / f"{frame.sequence_id:08d}.json")
        write_json(self.directory / "health.json", result.health)

    def heartbeat(self, state, fault, **fields):
        health = dict(state=state, fault_reason=fault, **fields)
        write_json(self.directory / "health.json", health)
        print(json.dumps(health), flush=True)
        return health


def process_replay(args):
    replay = SessionReplay(args.session)
    config = replay.config
    pipeline = TerrainPipeline(config, accumulate=args.accumulate)
    output = RunOutput(args.output, config, "replay")
    viewer = None
    if args.display:
        from kinect.terrain.visualization import TerrainViewer
        viewer = TerrainViewer(cloud=args.cloud)
    annotations_path = Path(args.session) / "annotations/fixtures.json"
    annotations = json.loads(annotations_path.read_text()) if annotations_path.exists() else []
    comparison, previous, processed = [], None, 0
    output.heartbeat("INITIALIZING", "opening recorded session")
    try:
        for frame in replay:
            if args.frames is not None and processed >= args.frames:
                break
            if args.step:
                input(f"Frame {frame.sequence_id}: press Enter to process ")
            elif args.speed > 0 and previous is not None:
                delay = max(0, (frame.receive_timestamp - previous) / args.speed)
                # Keep displays responsive even across long recorded gaps.
                end = time.monotonic() + delay
                while time.monotonic() < end:
                    time.sleep(max(0, min(0.05, end - time.monotonic())))
            result = pipeline.process(frame)
            output.write(frame, result)
            print(f"frame={frame.sequence_id} {result.health['state']} ground={result.ground.status} "
                  f"objects={len(result.observations)}", flush=True)
            targets = next((a["targets"] for a in annotations if a["sequence_id"] == frame.sequence_id), [])
            comparison.append(dict(sequence_id=frame.sequence_id,
                                   comparisons=compare_annotations(result, targets) if result.grid else []))
            processed += 1
            previous = frame.receive_timestamp
            if viewer and not viewer.show(frame, result):
                break
        write_json(output.directory / "evaluation.json", dict(
            schema_version=1, source_session=replay.manifest["session_id"], frame_count=processed,
            annotation_frame_count=len(annotations), comparisons=comparison,
            qualification="fixture comparison only; review measured criteria and physical gates separately"))
        output.heartbeat("STALE", "replay ended; exported results are recorded history", processed_frames=processed)
    except Exception as error:
        output.heartbeat("ERROR", str(error))
        raise
    finally:
        if viewer:
            viewer.close()


def live(args):
    config = TerrainConfig.load(args.config)
    session_id = args.session_id or str(uuid.uuid4())
    output = RunOutput(args.output, config, "live_record" if args.command == "record" else "live")
    pipeline = TerrainPipeline(config, accumulate=args.accumulate)
    output.heartbeat("INITIALIZING", "starting bounded latest-frame source")
    viewer = None
    if args.display:
        from kinect.terrain.visualization import TerrainViewer
        viewer = TerrainViewer(cloud=args.cloud)
    try:
        with ExitStack() as stack:
            writer = stack.enter_context(SessionWriter(args.session, config, session_id,
                       scene_metadata={"physical_qualification": "not inferred", "recording_policy": "latest-frame; drops explicit"})) if args.session else None
            if args.command == "record" and writer is None:
                raise ValueError("record requires --session NEW_DIRECTORY")
            source = stack.enter_context(LatestFrameSource(config, session_id))
            processed, timeouts = 0, 0
            previous_frame, previous_result = None, None
            while processed < args.frames:
                started = time.monotonic()
                try:
                    frame = source.get(timeout=max(0.05, config.maximum_input_age_s))
                except TimeoutError as error:
                    timeouts += 1
                    health = output.heartbeat("STALE", str(error), dropped_frames=source.dropped_frames,
                        source_age_s=time.monotonic() - previous_frame.depth_capture_timestamp if previous_frame else None)
                    if viewer and previous_frame is not None and not viewer.show(previous_frame, previous_result, heartbeat=health):
                        break
                    if timeouts >= args.max_timeouts:
                        raise RuntimeError("no fresh depth after repeated timeouts; inspect Kinect connection")
                    continue
                timeouts = 0
                if args.no_rgb:
                    frame.rgb = None
                if writer:
                    writer.write(frame, source.dropped_frames)
                result = pipeline.process(frame, now=time.monotonic(), dropped_frames=source.dropped_frames)
                output.write(frame, result)
                processed += 1
                previous_frame, previous_result = frame, result
                print(f"frame={frame.sequence_id} {result.health['state']} drops={source.dropped_frames}", flush=True)
                if viewer and not viewer.show(frame, result):
                    break
                time.sleep(max(0, 1 / config.target_grid_hz - (time.monotonic() - started)))
            output.heartbeat("STALE", "capture stopped; outputs are history", processed_frames=processed,
                             dropped_frames=source.dropped_frames)
    except Exception as error:
        output.heartbeat("ERROR", str(error))
        raise
    finally:
        if viewer:
            viewer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("config-template", help="write unresolved bench or synthetic profile")
    template.add_argument("--output", required=True, type=Path)
    template.add_argument("--synthetic", action="store_true")
    synth = commands.add_parser("fixtures", help="write deterministic synthetic replay session")
    synth.add_argument("--output", required=True, type=Path)
    synth.add_argument("--layout", choices=(*LAYOUTS, "all"), default="all")
    replay = commands.add_parser("replay", help="headless by default; recorded algorithm time")
    replay.add_argument("--session", required=True, type=Path)
    replay.add_argument("--output", required=True, type=Path)
    replay.add_argument("--frames", type=positive_int)
    replay.add_argument("--speed", type=float, default=0, help="0=unpaced, 1=real time, >1=accelerated")
    replay.add_argument("--step", action="store_true")
    for name in ("live", "record"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--session", type=Path, help="new recording directory")
        command.add_argument("--session-id")
        command.add_argument("--frames", type=positive_int, default=100)
        command.add_argument("--max-timeouts", type=positive_int, default=10)
        command.add_argument("--no-rgb", action="store_true")
    for command in (replay, commands.choices["live"], commands.choices["record"]):
        command.add_argument("--display", action="store_true")
        command.add_argument("--cloud", action="store_true", help="also show Open3D; requires --display")
        command.add_argument("--accumulate", action="store_true", help="separate stationary history; hazards persist until reset")
    evaluate = commands.add_parser("evaluate", help="run deterministic synthetic acceptance cases")
    evaluate.add_argument("--output", required=True, type=Path)
    check = commands.add_parser("calibration-check", help="compare >=5 measured target positions")
    check.add_argument("--config", required=True, type=Path)
    check.add_argument("--measurements", required=True, type=Path)
    check.add_argument("--output", required=True, type=Path)
    for name in ("import-npy", "import-legacy"):
        command = commands.add_parser(name, help="lossless import with explicit user-supplied mount/stream profile")
        command.add_argument("--input", nargs="+" if name == "import-npy" else None, required=True, type=Path)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--fps", type=float, default=10, help="NPY assigned playback cadence; not measured capture time")
    args = parser.parse_args(argv)
    try:
        if getattr(args, "cloud", False) and not args.display:
            parser.error("--cloud requires --display")
        if args.command == "config-template":
            if args.output.exists():
                raise FileExistsError(args.output)
            (synthetic_config() if args.synthetic else TerrainConfig()).save(args.output)
        elif args.command == "fixtures":
            config, session_id = synthetic_config(), "synthetic-development-v1"
            annotations = []
            with SessionWriter(args.output, config, session_id, scene_metadata={"kind": "synthetic logic fixtures; not qualification"}) as writer:
                for sequence, layout in enumerate(LAYOUTS if args.layout == "all" else (args.layout,)):
                    frame, targets = fixture(layout, sequence, session_id=session_id)
                    writer.write(frame)
                    annotations.append(dict(sequence_id=sequence, layout=layout, targets=targets))
            write_json(args.output / "annotations/fixtures.json", annotations)
        elif args.command == "replay":
            if not np.isfinite(args.speed) or args.speed < 0:
                parser.error("--speed must be finite and nonnegative")
            process_replay(args)
        elif args.command in ("live", "record"):
            live(args)
        elif args.command == "evaluate":
            if args.output.exists():
                raise FileExistsError(args.output)
            report = synthetic_evaluation(args.output)
            print(json.dumps({"all_passed": report["all_passed"], "hardware_qualified": False}))
            if not report["all_passed"]:
                raise ValueError("synthetic acceptance cases failed; inspect report")
        elif args.command == "calibration-check":
            if args.output.exists():
                raise FileExistsError(args.output)
            write_json(args.output, calibration_check(TerrainConfig.load(args.config), json.loads(args.measurements.read_text())))
        else:
            config = TerrainConfig.load(args.config)
            if not np.isfinite(args.fps) or args.fps <= 0:
                raise ValueError("fps must be finite and positive")
            session_id = str(uuid.uuid4())
            with SessionWriter(args.output, config, session_id,
                               scene_metadata={"calibration_assignment": "manual retrospective; verify against original mount"}) as writer:
                if args.command == "import-npy":
                    for i, path in enumerate(args.input):
                        with path.open("rb") as stream:
                            depth = np.load(stream, allow_pickle=False)
                            if not isinstance(depth, np.ndarray):
                                depth.close()
                                raise ValueError("expected an NPY depth array, not an NPZ archive")
                        frame = FrameBundle(session_id, i, depth, i / args.fps, i / args.fps,
                                            "assigned_import_seconds_not_measured_capture", config.calibration_id,
                                            config.mount_transform_id, source="imported_npy")
                        writer.write(frame)
                else:
                    with ReplayCamera(args.input) as camera:
                        i = 0
                        while (legacy := camera.read()) is not None:
                            if legacy.depth_space != "depth_camera":
                                raise ValueError("legacy registered/RGB-grid depth needs a matching optical profile; automatic conversion is disabled")
                            frame = FrameBundle(session_id, i, legacy.depth_mm, legacy.depth_timestamp_s,
                                legacy.depth_timestamp_s, "legacy_device_seconds_no_host_mapping",
                                config.calibration_id, config.mount_transform_id, rgb=legacy.rgb,
                                rgb_capture_timestamp=legacy.rgb_timestamp_s,
                                device_timestamp_optional=legacy.depth_timestamp_s,
                                capture_time_is_receive_proxy=False, source=legacy.source)
                            writer.write(frame)
                            i += 1
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, EOFError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
