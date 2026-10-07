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
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinect.pointcloud import depth_mm_to_point_cloud


DEPTH_WINDOW = "Kinect depth (point cloud source)"
VOXEL_SIZE_M = 0.02
DEBUG_INTERVAL_SECONDS = 5.0


def depth_to_display(depth_mm: np.ndarray) -> np.ndarray:
    """Create a display-only visualization of raw metric depth."""
    valid = depth_mm > 0
    display = np.zeros(depth_mm.shape, dtype=np.uint8)
    if np.any(valid):
        display[valid] = cv2.normalize(
            depth_mm[valid], None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
    return cv2.applyColorMap(display, cv2.COLORMAP_TURBO)


def read_nonempty_cloud() -> tuple[np.ndarray, o3d.geometry.PointCloud] | None:
    """Read one Kinect frame and return its downsampled nonempty cloud."""
    frame = freenect.sync_get_depth(format=freenect.DEPTH_MM)
    if frame is None:
        return None
    depth_mm, _timestamp = frame
    if depth_mm.ndim != 2:
        print(f"Ignoring unexpected depth frame shape: {depth_mm.shape}")
        return None
    cloud = depth_mm_to_point_cloud(depth_mm, voxel_size_m=VOXEL_SIZE_M)
    if not cloud.has_points():
        print("Ignoring depth frame with no valid 1..10000 mm measurements.")
        return None
    return depth_mm, cloud


def main() -> None:
    # Do not create or fit an Open3D view around an empty cloud.  Wait for a
    # real Kinect frame first so the initial camera bounds contain geometry.
    initial = read_nonempty_cloud()
    if initial is None:
        print("Could not retrieve a valid Kinect depth frame. Is the device connected?")
        freenect.sync_stop()
        return
    depth_mm, cloud = initial
    initial_count = len(cloud.points)
    print(f"Initial point count: {initial_count:,} (voxel size: {VOXEL_SIZE_M:.3f} m)")

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window("Kinect geometry point cloud")
    visualizer.add_geometry(cloud)
    # Fit the initial camera to valid points once.  Later updates preserve the
    # user's mouse rotation and zoom rather than resetting the viewpoint.
    visualizer.reset_view_point(True)
    visualizer.get_render_option().point_size = 2.0
    cv2.namedWindow(DEPTH_WINDOW)
    next_debug_time = time.monotonic() + DEBUG_INTERVAL_SECONDS

    try:
        while True:
            latest = read_nonempty_cloud()
            if latest is not None:
                depth_mm, latest_cloud = latest
                cloud.points = latest_cloud.points
                visualizer.update_geometry(cloud)
                if time.monotonic() >= next_debug_time:
                    print(f"Point count: {len(cloud.points):,}")
                    next_debug_time = time.monotonic() + DEBUG_INTERVAL_SECONDS
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
