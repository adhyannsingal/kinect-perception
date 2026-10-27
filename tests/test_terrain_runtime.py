"""Queue bounds, capture faults, display consumption and complete CLI replay."""

import importlib.util
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import Mock

import numpy as np

from kinect.recording import RecordingWriter
from kinect.simulation import SimulatedCamera
from kinect.terrain.acquisition import LatestFrameSource
from kinect.terrain.config import TerrainConfig
from kinect.terrain.evaluation import synthetic_evaluation
from kinect.terrain.fixtures import fixture, synthetic_config
from kinect.terrain.pipeline import TerrainPipeline
from kinect.terrain.session import SessionReplay
from kinect.terrain.visualization import dashboard, grid_image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/terrain_perception.py"


class LatestSourceTests(unittest.TestCase):
    def test_queue_favors_latest_reports_exact_drops_and_closes_source(self):
        camera = SimulatedCamera(width=8, height=8, frame_count=100)
        with LatestFrameSource(synthetic_config(), "queue-test", camera=camera, clock=lambda: 5) as source:
            source._thread.join(timeout=2)
            self.assertFalse(source._thread.is_alive())
            self.assertEqual(source.queue.qsize(), 1)
            self.assertEqual(source.dropped_frames, 99)
            latest = source.get(timeout=0.1)
            self.assertEqual(latest.sequence_id, 99)
            self.assertTrue(latest.capture_time_is_receive_proxy)
            self.assertEqual(latest.receive_timestamp, 5)
            self.assertEqual(latest.timestamp_clock_domain, "host_monotonic_receive_proxy")
        with self.assertRaises(RuntimeError):
            camera.read()

    def test_native_capture_error_and_blocked_read_timeout_are_explicit(self):
        camera = Mock()
        camera.read.side_effect = RuntimeError("USB disconnected")
        with LatestFrameSource(TerrainConfig(), "error", camera=camera) as source:
            source._thread.join(timeout=1)
            with self.assertRaisesRegex(RuntimeError, "USB disconnected"):
                source.get(timeout=0.01)
        camera.close.assert_called_once()
        release = Event()
        camera = Mock()
        camera.read.side_effect = lambda: release.wait(1) and None
        with LatestFrameSource(TerrainConfig(), "blocked", camera=camera) as source:
            try:
                with self.assertRaises(TimeoutError):
                    source.get(timeout=0.01)
            finally:
                release.set()
        camera.close.assert_called_once()

    def test_synthetic_profile_cannot_drive_real_capture(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            LatestFrameSource(synthetic_config(), "invalid")


class RuntimeTests(unittest.TestCase):
    def run_cli(self, *args, input_text=None):
        result = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                                input=input_text, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_saved_session_replay_semantics_real_time_accelerated_and_single_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_cli("fixtures", "--output", root / "session")
            self.run_cli("replay", "--session", root / "session", "--output", root / "fast", "--frames", 2)
            self.run_cli("replay", "--session", root / "session", "--output", root / "real", "--speed", 1, "--frames", 2)
            self.run_cli("replay", "--session", root / "session", "--output", root / "step", "--step", "--frames", 2, input_text="\n\n")
            exported = [json.loads((root / name / "00000000.json").read_text()) for name in ("fast", "real", "step")]
            for result in exported[1:]:
                self.assertEqual(exported[0]["grid"], result["grid"])
                self.assertEqual(exported[0]["ground"], result["ground"])
            frame = next(iter(SessionReplay(root / "session")))
            direct = TerrainPipeline(synthetic_config()).process(frame)
            np.testing.assert_array_equal(direct.grid["observed_support"], exported[0]["grid"]["observed_support"])
            self.assertEqual(exported[0]["health"]["source_age_s"], 0)
            self.assertNotEqual(exported[0]["health"]["state"], "STALE")
            self.assertEqual(json.loads((root / "fast/health.json").read_text())["state"], "STALE")
            for name in ("real", "step"):
                fast_second = json.loads((root / "fast/00000001.json").read_text())
                other_second = json.loads((root / name / "00000001.json").read_text())
                self.assertEqual(fast_second["grid"], other_second["grid"])

    def test_npy_and_original_recording_imports_preserve_raw_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = synthetic_config()
            c.save(root / "config.json")
            frame = fixture()[0]
            frame.depth_raw[0, 0] = 65535
            np.save(root / "depth.npy", frame.depth_raw)
            self.run_cli("import-npy", "--input", root / "depth.npy", "--config", root / "config.json", "--output", root / "npy-session")
            restored = next(iter(SessionReplay(root / "npy-session")))
            np.testing.assert_array_equal(restored.depth_raw, frame.depth_raw)
            self.assertIn("not_measured", restored.timestamp_clock_domain)
            original = replace(SimulatedCamera(frame_count=1).read(), depth_space="depth_camera")
            with RecordingWriter(root / "legacy") as writer:
                writer.write(original)
            self.run_cli("import-legacy", "--input", root / "legacy", "--config", root / "config.json", "--output", root / "legacy-session")
            restored = next(iter(SessionReplay(root / "legacy-session")))
            np.testing.assert_array_equal(restored.depth_raw, original.depth_mm)
            self.assertEqual(restored.depth_capture_timestamp, original.depth_timestamp_s)

    def test_synthetic_acceptance_report_and_unresolved_profile_template(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = synthetic_evaluation(root / "report.json")
            self.assertTrue(report["all_passed"])
            self.assertFalse(report["hardware_qualified"])
            self.assertEqual(report["scene_count"], 7)
            self.run_cli("config-template", "--output", root / "unresolved.json")
            self.assertFalse(TerrainConfig.load(root / "unresolved.json").calibration_ready)

    def test_checked_in_session_replays_without_generator_or_hardware(self):
        path = SCRIPT.parents[1] / "fixtures/phase1/session"
        replay = SessionReplay(path)
        pipeline = TerrainPipeline(replay.config)
        results = [pipeline.process(frame) for frame in replay]
        self.assertEqual(len(results), 7)
        self.assertTrue(results[0].grid["observed_support"].any())
        self.assertTrue(results[2].grid["negative_evidence"].any())
        self.assertFalse(results[3].grid["negative_evidence"].any())
        self.assertNotEqual(results[-1].ground.status, "VALID")

    def test_dashboard_is_read_only_and_marks_stale_history(self):
        frame = fixture("mixed")[0]
        result = TerrainPipeline(synthetic_config()).process(frame)
        labels = result.grid["summary"].copy()
        image = dashboard(frame, result)
        self.assertEqual(image.shape, (600, 800, 3))
        self.assertEqual(grid_image(result).shape[:2], labels.shape)
        stale = dashboard(frame, result, heartbeat={"state": "STALE", "fault_reason": "sensor stopped"})
        self.assertFalse(np.array_equal(stale, image))
        np.testing.assert_array_equal(result.grid["summary"], labels)

    def test_mock_live_command_records_bounded_source_and_reports_capture_failure(self):
        # Exercise the real live CLI consumer with an injected source; no USB.
        from unittest.mock import patch
        spec = importlib.util.spec_from_file_location("terrain_cli", SCRIPT)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        config = synthetic_config()
        frame = fixture("flat")[0]
        source = Mock(dropped_frames=3)
        source.__enter__ = Mock(return_value=source)
        source.__exit__ = Mock(return_value=False)
        source.get.return_value = frame
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config.save(root / "config.json")
            with patch.object(cli, "LatestFrameSource", return_value=source), patch.object(cli.time, "monotonic", return_value=0):
                cli.main(["record", "--config", str(root / "config.json"), "--output", str(root / "run"),
                          "--session", str(root / "session"), "--session-id", frame.session_id, "--frames", "1"])
            self.assertEqual(len(list(SessionReplay(root / "session"))), 1)
            health = json.loads((root / "run/health.json").read_text())
            self.assertEqual(health["state"], "STALE")
            self.assertEqual(health["dropped_frames"], 3)


if __name__ == "__main__":
    unittest.main()
