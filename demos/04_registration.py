"""Visualize libfreenect's RGB-aligned Kinect v1 depth output.

``FREENECT_DEPTH_REGISTERED`` is libfreenect's calibrated depth-to-RGB
registration mode.  It is not a resize or a same-pixel assumption: the
library uses the device registration data to produce metric depth on the RGB
image grid.  RGB and depth frames are acquired independently, so fast motion
can still introduce temporal misalignment.

This is a live demo and needs a connected Kinect.  Visual alignment remains
to be checked on hardware.
"""

import cv2
import freenect
import numpy as np


WINDOW_NAME = "Kinect RGB-depth registration"


def registered_depth_to_color(depth_mm: np.ndarray) -> np.ndarray:
    """Return a display-only color image, preserving raw uint16 depth."""
    valid = depth_mm > 0
    scaled = np.zeros(depth_mm.shape, dtype=np.uint8)
    if np.any(valid):
        scaled[valid] = cv2.normalize(
            depth_mm[valid], None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


def blend_registered_depth(rgb_bgr: np.ndarray, depth_color: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
    """Overlay only pixels for which registered depth is valid."""
    overlay = rgb_bgr.copy()
    valid = depth_mm > 0
    blended = cv2.addWeighted(rgb_bgr, 0.55, depth_color, 0.45, 0)
    overlay[valid] = blended[valid]
    return overlay


def main() -> None:
    if not hasattr(freenect, "DEPTH_REGISTERED"):
        raise RuntimeError(
            "This freenect binding does not expose DEPTH_REGISTERED; "
            "do not substitute a resize for calibrated registration."
        )

    try:
        while True:
            rgb_frame = freenect.sync_get_video(format=freenect.VIDEO_RGB)
            depth_frame = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
            if rgb_frame is None or depth_frame is None:
                print("Could not retrieve RGB/depth frames. Is the Kinect connected?")
                break

            rgb_rgb, _rgb_timestamp = rgb_frame
            registered_depth_mm, _depth_timestamp = depth_frame
            rgb_bgr = cv2.cvtColor(rgb_rgb, cv2.COLOR_RGB2BGR)
            depth_color = registered_depth_to_color(registered_depth_mm)
            overlay = blend_registered_depth(rgb_bgr, depth_color, registered_depth_mm)

            cv2.putText(overlay, "libfreenect DEPTH_REGISTERED overlay", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            combined = np.hstack((rgb_bgr, depth_color, overlay))
            cv2.imshow(WINDOW_NAME, combined)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        freenect.sync_stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
