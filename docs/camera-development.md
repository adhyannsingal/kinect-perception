# Isolated camera development

Known-working baseline supplied by the existing repository:
`71e502b0ad8a70ba6a6b32af214a92c0d2b6c01e` (branch `main`).
Development branch: `feature/camera-recording-simulation`.
The working tree was clean when this branch was created.

The original demos, geometry, point-cloud code, calibration, dependency file,
and environment configuration are unchanged. No merge into `main` is authorized
until the user validates the original Kinect demos on physical hardware.
Automated checks cannot establish hardware validity. The original README's
individual hardware-validation statuses still apply.

## Usage

Use the existing environment, without installing or updating packages:

```sh
conda activate kinect
python scripts/camera_stream.py --source simulate --frames 60
python scripts/camera_stream.py --source simulate --frames 60 --record /tmp/sim-capture
python scripts/camera_stream.py --source replay --input /tmp/sim-capture --display
python scripts/camera_stream.py --source kinect --frames 60 --record /tmp/live-capture --display
python -m unittest discover -s tests -v
```

Each output directory must be new. Recordings may contain private scenes and
should be stored outside the repository. Display is optional; default operation
is headless and does not import OpenCV, Open3D, or freenect. The live command
requires the existing native binding and hardware and remains hardware unvalidated.
The CLI limits reads to `--frames` (default 300), or replay EOF, whichever comes
first. Display q/Escape stops early. Reads are unpaced; replay preserves capture
timestamps but does not reproduce capture timing.

## Modules and data contract

- `kinect.camera`: `Camera`, `CameraFrame`, and lazy `KinectCamera` adapter.
  Sources implement `read()` and `close()` and support a context manager.
  `None` means EOF, errors raise, and reads after close raise.
- `kinect.simulation`: deterministic plane at 2 m, moving square at 1 m,
  missing-depth corner, and synthetic intrinsics independent of calibration.
- `kinect.recording`: `RecordingWriter` and `ReplayCamera`, with rewind and
  stable EOF. Each numbered NPZ stores lossless RGB/depth plus versioned JSON
  metadata. Pickle is disabled. Completed frames are published by rename;
  incomplete `.partial` files are ignored. There is no durability guarantee
  across power loss. Readers take a snapshot of completed files at construction.

Frames own copies of RGB uint8 H×W×3 and depth uint16 H×W millimetre arrays.
Zero depth means missing data. RGB and depth dimensions may differ. Independent
timestamps are in seconds in the source's time domain, not wall-clock time.
Recordings preserve the source, depth coordinate space, and optional depth
intrinsics. The live adapter uses DEPTH_MM and the existing approximate depth
intrinsics by default; callers may pass their own intrinsics. It does not modify
device calibration or register depth to RGB. Matching dimensions never establish
registration. libfreenect timestamps can wrap; no clock correction is applied.
The sync binding is process-global; use one live adapter at a time.

## Protection check

All pre-existing tracked files can be verified together:

```sh
git diff 71e502b0ad8a70ba6a6b32af214a92c0d2b6c01e -- \
  .gitignore README.md requirements.txt demos kinect/__init__.py \
  kinect/geometry.py kinect/pointcloud.py \
  tests/test_geometry.py tests/test_pointcloud.py tests/test_pointcloud_demo.py
```

That diff must remain empty. No changes to dependency versions or environment
configuration are needed. Hardware validation and permission to merge remain
with the user.

## Automated validation

The existing `/opt/anaconda3/envs/kinect/bin/python` passed all 8 original tests
before implementation. The full suite passed all 20 tests after implementation,
including headless CLI simulation/record/replay and mocked native capture.
The default base Python lacks Open3D; it was not modified. The existing Kinect
environment was used for the successful runs, with no package installation or
environment changes. The baseline diff covering every pre-existing tracked file
was empty. No physical-hardware validation or merge was performed.
