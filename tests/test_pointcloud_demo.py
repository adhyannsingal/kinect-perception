"""Offline regression checks for Demo 5's view and live-update loop."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from kinect.geometry import CameraIntrinsics
from kinect.pointcloud import depth_mm_to_point_cloud


class PointCloudDemoTests(unittest.TestCase):
    def test_upright_initial_view_survives_live_updates(self) -> None:
        # Import without requiring the native Kinect binding or opening windows.
        path = Path(__file__).resolve().parents[1] / "demos" / "05_pointcloud.py"
        spec = importlib.util.spec_from_file_location("pointcloud_demo", path)
        demo = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"freenect": Mock()}):
            spec.loader.exec_module(demo)

        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=1.0, cy=1.0)
        depth = np.full((3, 3), 2000, dtype=np.uint16)
        cloud = depth_mm_to_point_cloud(depth, intrinsics)
        original_points = np.asarray(cloud.points).copy()
        center = cloud.get_axis_aligned_bounding_box().get_center().copy()
        latest = depth_mm_to_point_cloud(depth + 1000, intrinsics)
        latest_points = np.asarray(latest.points).copy()
        visualizer = Mock()
        view = visualizer.get_view_control.return_value

        def poll_events() -> bool:
            # First poll models a user rotating/zooming; the next frame must
            # update geometry without configuring or resetting the view again.
            if visualizer.poll_events.call_count == 1:
                view.reset_mock()
                visualizer.reset_view_point.reset_mock()
                return True
            view.set_up.assert_not_called()
            view.set_front.assert_not_called()
            view.set_lookat.assert_not_called()
            view.set_zoom.assert_not_called()
            visualizer.reset_view_point.assert_not_called()
            return False

        def check_initial_view() -> None:
            np.testing.assert_array_equal(cloud.points, original_points)
            visualizer.reset_view_point.assert_called_once_with(True)
            view.set_lookat.assert_called_once()
            np.testing.assert_allclose(view.set_lookat.call_args.args[0], center)
            view.set_zoom.assert_called_once_with(0.7)
            view.set_up.assert_called_once()
            view.set_front.assert_called_once()
            up = np.asarray(view.set_up.call_args.args[0])
            front = np.asarray(view.set_front.call_args.args[0])
            # Open3D's screen-right is up cross front.  Check the image's
            # top-left remains above and left, and positive Z faces the view.
            right = np.cross(up, front)
            top_left = original_points[0] - center
            self.assertGreater(top_left @ up, 0)
            self.assertLess(top_left @ right, 0)
            np.testing.assert_array_equal(-front, [0, 0, 1])

        visualizer.poll_events.side_effect = poll_events
        with (
            patch.object(demo.o3d.visualization, "Visualizer", return_value=visualizer),
            patch.object(demo, "cv2") as cv2,
            patch.object(demo, "depth_to_display"),
            patch.object(demo, "read_nonempty_cloud") as read_cloud,
        ):
            cv2.namedWindow.side_effect = lambda *_: check_initial_view()
            cv2.waitKey.return_value = -1
            # An empty read keeps the initial points; a subsequent valid frame
            # exercises geometry replacement after the simulated mouse input.
            read_cloud.side_effect = [(depth, cloud), None, (depth + 1000, latest)]
            demo.main()

        visualizer.add_geometry.assert_called_once_with(cloud)
        visualizer.update_geometry.assert_called_once_with(cloud)
        np.testing.assert_array_equal(cloud.points, latest_points)
        visualizer.update_renderer.assert_called_once()
        demo.freenect.sync_stop.assert_called_once()
        visualizer.destroy_window.assert_called_once()
        cv2.destroyAllWindows.assert_called_once()


if __name__ == "__main__":
    unittest.main()
