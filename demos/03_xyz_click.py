"""Inspect a Kinect depth pixel as a 3D depth-camera coordinate.

This live demo uses approximate default Kinect v1 intrinsics.  Replace them
with a calibration for measurement-sensitive use.
"""

from pathlib import Path
import sys

import cv2
import freenect
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinect.geometry import pixel_depth_to_camera_xyz


WINDOW_NAME = "Kinect pixel to XYZ"


def depth_to_display(depth_mm: np.ndarray) -> np.ndarray:
    """Create a display-only color representation of metric depth."""
    valid = depth_mm > 0
    display = np.zeros(depth_mm.shape, dtype=np.uint8)
    if np.any(valid):
        display[valid] = cv2.normalize(
            depth_mm[valid], None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
    return cv2.applyColorMap(display, cv2.COLORMAP_TURBO)


def main() -> None:
    selected_pixel: tuple[int, int] | None = None

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal selected_pixel
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            selected_pixel = (x, y)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        while True:
            frame = freenect.sync_get_depth(format=freenect.DEPTH_MM)
            if frame is None:
                print("Could not retrieve a Kinect depth frame. Is the device connected?")
                break
            depth_mm, _timestamp = frame
            display = depth_to_display(depth_mm)

            if selected_pixel is not None:
                u, v = selected_pixel
                if 0 <= u < depth_mm.shape[1] and 0 <= v < depth_mm.shape[0]:
                    xyz = pixel_depth_to_camera_xyz(u, v, float(depth_mm[v, u]) / 1000.0)
                    color = (0, 255, 0) if xyz is not None else (0, 0, 255)
                    cv2.drawMarker(display, (u, v), color, cv2.MARKER_CROSS, 16, 2)
                    if xyz is None:
                        label = f"Pixel: ({u}, {v})  Depth: invalid"
                    else:
                        x, y, z = xyz
                        label = (
                            f"Pixel: ({u}, {v})  Depth: {z:.3f} m  "
                            f"X: {x:.3f}  Y: {y:.3f}  Z: {z:.3f}"
                        )
                    cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, color, 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        freenect.sync_stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
