import cv2
import freenect
import numpy as np
import time


def get_rgb():
    frame, _ = freenect.sync_get_video()

    if frame is None:
        return None

    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def get_depth():
    depth, _ = freenect.sync_get_depth()

    return depth


previous_time = time.time()

while True:
    rgb = get_rgb()
    depth = get_depth()

    if rgb is None or depth is None:
        print("Could not retrieve Kinect frame.")
        continue

    # Convert the 16-bit depth image into an 8-bit image
    # so that OpenCV can display it clearly.
    depth_display = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    depth_display = depth_display.astype(np.uint8)

    # Apply a color map only for visualization.
    depth_colored = cv2.applyColorMap(
        depth_display,
        cv2.COLORMAP_JET
    )

    # Calculate approximate FPS.
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(
        rgb,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Put RGB and depth next to each other.
    combined = np.hstack((rgb, depth_colored))

    cv2.imshow("Kinect RGB + Depth", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


freenect.sync_stop()
cv2.destroyAllWindows()
