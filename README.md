# Kinect v1 perception fundamentals

Small, reusable perception utilities for an Xbox 360 Kinect v1.  The current
target platform is Apple Silicon macOS with Python 3.12, libfreenect, NumPy,
OpenCV, and Open3D.

```text
Kinect -> RGB + depth -> physical depth -> pixel + depth -> XYZ
       -> RGB-depth registration -> 3D point cloud
```

## Setup

Native dependencies are supplied outside this repository: Homebrew
`libfreenect` and `libusb`, plus the locally compiled arm64 `freenect` Python
extension in the `kinect` Conda environment.  That binding is intentionally
not listed as a pip dependency.  `requirements.txt` contains only the Python
packages used by this repository.

```bash
conda activate kinect
python -m pip install -r requirements.txt
```

## Demos

Run live demos from the repository root with `python demos/<name>.py`.

- `01_rgb_depth.py` — side-by-side RGB and depth viewer. **Implemented and
  hardware validated.**
- `02_depth_click.py` — inspect a depth pixel in physical millimetres.
  **Implemented and hardware validated.**
- `03_xyz_click.py` — project a selected depth pixel into depth-camera XYZ.
  **Implemented and offline validated; live validation pending.**
- `04_registration.py` — show libfreenect `DEPTH_REGISTERED` depth on the RGB
  grid. **Implemented and API validated; live visual alignment pending.**
- `05_pointcloud.py` — geometry-only Open3D cloud from metric depth.
  **Implemented and synthetic-data validated; live validation pending.**
- `06_ground_segmentation.py` — offline, orientation-constrained RANSAC ground
  segmentation. Green ground / red non-ground. **Automated offline validation;
  local visual validation pending.**

The XYZ convention is metres with +X right in the image, +Y down, and +Z
forward from the depth camera.  `kinect.geometry.DEFAULT_KINECT_V1_DEPTH_INTRINSICS`
contains common approximate Kinect v1 depth-camera values.  They are not a
replacement for calibration and should be replaced for measurement-sensitive
work.

`04_registration.py` uses libfreenect's actual `DEPTH_REGISTERED` mode; it
does not fake registration by resizing.  The current Python wrapper does not
expose lower-level registration controls, and independently captured RGB and
depth frames can be temporally offset during motion.

## Offline development

Use `python demos/capture_sample.py --prefix scene1` while the Kinect is
connected to save a color-correct RGB PNG and an exact uint16 millimetre depth
`.npy` frame.  Generated recordings in `sample_data/` are intentionally not
tracked.

Run hardware-independent checks with:

```bash
python -m unittest discover -s tests -v
```

## Demo 6: ground-plane segmentation (offline)

Use the existing `kinect` environment. No Kinect connection or new dependencies
are needed. Input `.npy` files must contain a 2D array of **metric millimetre
depth**, such as the `*_depth_mm.npy` files from `capture_sample.py`. Raw Kinect
disparity values and display-normalized depth are not supported. Multiple files
are processed in argument order; close each Open3D window to advance.

```bash
python demos/06_ground_segmentation.py sample_data/scene1_depth_mm.npy
python demos/06_ground_segmentation.py sample_data/*_depth_mm.npy --headless
python demos/06_ground_segmentation.py --synthetic
python demos/06_ground_segmentation.py --synthetic --headless
python demos/06_ground_segmentation.py --recording /tmp/live-capture --headless
```

The synthetic scene ray-casts a floor at `y=0.8 m`, a larger wall at `z=4 m`,
and a box front at `z=2 m`. It has its own intrinsics and leaves the existing
simulator/calibration unchanged. `--recording` reads the lossless NPZ directories
from `scripts/camera_stream.py` using `ReplayCamera`, including depth intrinsics.
The original simulator's front-facing planes are not floors under the default
ground direction and should yield no detected ground.

Configure the expected floor direction for the camera's actual mounting:

```bash
python demos/06_ground_segmentation.py sample_data/scene1_depth_mm.npy \
  --distance-threshold 0.02 --voxel-size 0.03 \
  --expected-normal 0 1 0 --max-tilt 15 --height-range 0.9 1.1 \
  --iterations 2000 --min-inliers 100 --min-inlier-ratio 0.1
```

Coordinates remain metres, +X right, +Y down, +Z forward. `--expected-normal`
points **from the camera toward ground**; the default `(0, 1, 0)` is suitable
for a level camera. `--max-tilt` is the allowed normal deviation in degrees.
`--height-range MIN MAX` bounds the signed perpendicular camera-to-plane
distance in metres (default `0.1..3.0`); for a camera 1 m above the floor, a
range such as `0.9..1.1` helps reject parallel tabletops. Orientation alone
cannot distinguish floors from parallel surfaces. Among admissible planes,
the strongest RANSAC consensus wins; a larger wall does not win merely by size.
There is no semantic guarantee if multiple surfaces satisfy all constraints,
and finite RANSAC sampling can miss a small floor. Adjust the height/orientation
priors and iteration/support settings for the scene.

`--distance-threshold` controls perpendicular inlier distance in metres.
`--voxel-size 0` disables downsampling; otherwise voxelization affects fitting
only. Support count/ratio thresholds apply to the fitting cloud. All valid
original points are then classified and displayed. `--seed` (default 0) makes
sampling deterministic. Plane refinement uses least squares on the consensus.

NPY inputs use the existing approximate Kinect depth intrinsics by default.
For another depth grid or known calibration, pass `--intrinsics FX FY CX CY`;
this overrides values only for Demo 6 and never changes stored calibration.
Synthetic and NPZ inputs use their own depth intrinsics unless overridden.
Projection reuses `depth_mm_to_point_cloud` and skips non-finite, nonpositive,
and greater-than-10000 mm depth. Empty clouds and scenes without an admissible
plane report zero ground; all remaining points are red. The normalized plane
equation `ax + by + cz + d = 0`, full-resolution ground/non-ground counts,
inlier percentage, and fitting-point count appear in the terminal and Open3D
window title. `--headless` skips windows for automated/offline runs.

The algorithm is isolated in `kinect/ground.py`; synthetic depth generation is
in `kinect/ground_simulation.py`. Tests cover noisy ground, larger walls,
parallel elevated planes, tilted ground, obstacles, invalid/empty data, NPY and
recording replay, and visualization colors. No clustering, mapping, or planning
is included. Demo 6 remains on its feature branch pending local validation.
