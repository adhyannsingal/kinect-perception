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
