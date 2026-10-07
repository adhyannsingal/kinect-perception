"""View a geometry-only Open3D point cloud from live Kinect v1 depth.

This intentionally does not color points from RGB: correct coloring depends on
live validation of RGB/depth registration.  The demo needs a connected Kinect.
"""

from pathlib import Path
import sys

import cv2
import freenect
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinect.pointcloud import depth_mm_to_point_cloud


DEPTH_WINDOW = "Kinect depth (point cloud source)"
VOXEL_SIZE_M = 0.02


def depth_to_display(depth_mm: np.ndarray) -> np.ndarray:
    """Create a display-only visualization of raw metric depth."""
    valid = depth_mm > 0
    display = np.zeros(depth_mm.shape, dtype=np.uint8)
    if np.any(valid):
        display[valid] = cv2.normalize(
            depth_mm[valid], None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
    return cv2.applyColorMap(display, cv2.COLORMAP_TURBO)


def main() -> None:
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window("Kinect geometry point cloud")
    cloud = o3d.geometry.PointCloud()
    visualizer.add_geometry(cloud)
    cv2.namedWindow(DEPTH_WINDOW)

    try:
        while True:
            frame = freenect.sync_get_depth(format=freenect.DEPTH_MM)
            if frame is None:
                print("Could not retrieve a Kinect depth frame. Is the device connected?")
                break
            depth_mm, _timestamp = frame
            latest_cloud = depth_mm_to_point_cloud(depth_mm, voxel_size_m=VOXEL_SIZE_M)
            cloud.points = latest_cloud.points
            visualizer.update_geometry(cloud)
            if not visualizer.poll_events():
                break
            visualizer.update_renderer()

            cv2.imshow(DEPTH_WINDOW, depth_to_display(depth_mm))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        freenect.sync_stop()
        visualizer.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
