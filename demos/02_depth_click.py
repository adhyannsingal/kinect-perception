"""Interactively inspect Kinect v1 depth in millimetres.

This is a live demo and therefore requires a connected Kinect.
"""

import cv2
import freenect
import numpy as np


WINDOW_NAME = "Kinect depth measurement"
INVALID_DEPTH_MM = 0


def depth_to_display(depth_mm: np.ndarray) -> np.ndarray:
    """Return a colorized display image without changing measurement data."""
    valid = depth_mm > INVALID_DEPTH_MM
    display = np.zeros(depth_mm.shape, dtype=np.uint8)
    if np.any(valid):
        display[valid] = cv2.normalize(
            depth_mm[valid], None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
    return cv2.applyColorMap(display, cv2.COLORMAP_TURBO)


def main() -> None:
    selected_pixel: tuple[int, int] | None = None
    latest_depth: np.ndarray | None = None

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

            latest_depth, _timestamp = frame
            display = depth_to_display(latest_depth)

            if selected_pixel is not None:
                u, v = selected_pixel
                if 0 <= u < latest_depth.shape[1] and 0 <= v < latest_depth.shape[0]:
                    distance_mm = int(latest_depth[v, u])
                    valid = distance_mm > INVALID_DEPTH_MM
                    color = (0, 255, 0) if valid else (0, 0, 255)
                    cv2.drawMarker(display, (u, v), color, cv2.MARKER_CROSS, 16, 2)
                    label = (
                        f"Pixel: ({u}, {v})  Distance: {distance_mm} mm "
                        f"({distance_mm / 1000.0:.2f} m)"
                        if valid
                        else f"Pixel: ({u}, {v})  Distance: invalid"
                    )
                    cv2.putText(
                        display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2, cv2.LINE_AA,
                    )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        freenect.sync_stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
