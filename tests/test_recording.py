"""Lossless recording and replay checks, including error paths and CLI use."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from kinect.recording import RecordingWriter, ReplayCamera
from kinect.simulation import SimulatedCamera


class RecordingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "recording"

    def test_roundtrip_exact_arrays_metadata_rewind_and_close(self):
        camera = SimulatedCamera(width=16, height=12, frame_count=3)
        frames = [camera.read() for _ in range(3)]
        # Exercise invalid sensor sentinel values and independent timestamps,
        # dimensions, and absent intrinsics rather than only synthetic defaults.
        frames[0].depth_mm[0, 1] = 65535
        frames[1] = replace(frames[1], rgb=np.zeros((2, 3, 3), np.uint8),
                            rgb_timestamp_s=1.234, intrinsics=None)
        with RecordingWriter(self.path) as writer:
            for frame in frames:
                writer.write(frame)
        with self.assertRaises(RuntimeError):
            writer.write(frames[0])
        with ReplayCamera(self.path) as replay:
            for original in frames:
                restored = replay.read()
                np.testing.assert_array_equal(original.rgb, restored.rgb)
                np.testing.assert_array_equal(original.depth_mm, restored.depth_mm)
                self.assertEqual(restored.rgb.dtype, np.uint8)
                self.assertEqual(restored.depth_mm.dtype, np.uint16)
                for key in ("rgb_timestamp_s", "depth_timestamp_s", "source",
                            "intrinsics", "depth_space"):
                    self.assertEqual(getattr(original, key), getattr(restored, key))
            self.assertIsNone(replay.read())
            self.assertIsNone(replay.read())
            replay.rewind()
            np.testing.assert_array_equal(replay.read().depth_mm, frames[0].depth_mm)
        with self.assertRaises(RuntimeError):
            replay.read()
        with self.assertRaises(RuntimeError):
            replay.rewind()

    def test_existing_directory_refused(self):
        RecordingWriter(self.path).close()
        with self.assertRaises(FileExistsError):
            RecordingWriter(self.path)

    def test_empty_recording_and_partial_frame(self):
        RecordingWriter(self.path).close()
        (self.path / "00000000.partial").write_bytes(b"interrupted")
        self.assertIsNone(ReplayCamera(self.path).read())

    def test_missing_directory_and_missing_frame(self):
        with self.assertRaises(FileNotFoundError):
            ReplayCamera(self.path)
        self.path.mkdir()
        (self.path / "00000001.npz").touch()
        with self.assertRaises(ValueError):
            ReplayCamera(self.path)

    def test_corruption_and_unknown_version_fail_explicitly(self):
        self.path.mkdir()
        path = self.path / "00000000.npz"
        path.write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "Invalid recorded frame"):
            ReplayCamera(self.path).read()
        np.savez(path, metadata=np.array(json.dumps({"version": 999})))
        with self.assertRaisesRegex(ValueError, "unsupported recording version"):
            ReplayCamera(self.path).read()
        path.write_bytes(path.read_bytes()[:40])
        with self.assertRaisesRegex(ValueError, "Invalid recorded frame"):
            ReplayCamera(self.path).read()

    def test_cli_headless_simulation_and_replay(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "camera_stream.py"
        record = subprocess.run([sys.executable, str(script), "--frames", "3",
                                 "--record", str(self.path)], capture_output=True, text=True)
        self.assertEqual(record.returncode, 0, record.stderr)
        self.assertIn("Processed 3 frames from simulate", record.stdout)
        replay = subprocess.run([sys.executable, str(script), "--source", "replay",
                                 "--input", str(self.path)], capture_output=True, text=True)
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertIn("Processed 3 frames from replay", replay.stdout)


if __name__ == "__main__":
    unittest.main()
