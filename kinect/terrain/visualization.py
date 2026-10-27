"""Read-only OpenCV dashboard and optional Open3D ground-aligned scene."""

import cv2
import numpy as np


COLORS_RGB = {
    "UNKNOWN": (90, 90, 90), "OBSERVED_GROUND": (40, 190, 60),
    "RAISED": (230, 60, 40), "NEGATIVE": (40, 110, 245),
    "EDGE": (220, 70, 220), "MIXED": (245, 170, 30),
}


def grid_image(result, *, elevation=False):
    if not result.grid:
        return np.zeros((240, 320, 3), np.uint8)
    labels = result.grid["summary"]
    image = np.empty((*labels.shape, 3), dtype=np.uint8)
    for label, rgb in COLORS_RGB.items():
        image[labels == label] = rgb[::-1]
    if elevation:
        values = result.grid["return_elevation_m"]
        finite = np.isfinite(values)
        scaled = np.clip((np.nan_to_num(values) + 0.5) / 2 * 255, 0, 255).astype(np.uint8)
        image[finite] = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)[finite]
    return np.flipud(image).copy()


def dashboard(frame, result, *, elevation=False, heartbeat=None):
    """Render existing evidence only; RGB is explicitly unregistered."""
    depth = np.clip(frame.depth_raw / 10000 * 255, 0, 255).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    depth[~result.depth_valid_mask] = (180, 0, 180)
    rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR) if frame.rgb is not None else np.zeros_like(depth)
    def panel(image, title):
        image = cv2.resize(image, (400, 300), interpolation=cv2.INTER_NEAREST)
        cv2.putText(image, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return image
    top = np.hstack((panel(rgb, "RGB (unregistered)"), panel(depth, "Depth; magenta=invalid/out-of-range")))
    grid = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.putText(grid, "test_ground; metres", (45, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
    grid[30:260, 45:385] = cv2.resize(grid_image(result, elevation=elevation), (340, 230), interpolation=cv2.INTER_NEAREST)
    bounds = result.metadata["bounds_xy_m"]
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        for fraction in (0, 0.5, 1):
            xpixel = 45 + round(fraction * 339)
            ypixel = 259 - round(fraction * 229)
            cv2.putText(grid, f"{xmin + fraction * (xmax-xmin):.1f}", (xpixel - 10, 276),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1)
            cv2.putText(grid, f"{ymin + fraction * (ymax-ymin):.1f}", (2, ypixel),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1)
        cv2.putText(grid, "x forward", (160, 294), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1)
        for observation in result.observations:
            lo, hi = observation["bounding_box_m"]
            x0, x1 = [int(np.clip(45 + (x-xmin)/(xmax-xmin)*339, 45, 384)) for x in (lo[0], hi[0])]
            y0, y1 = [int(np.clip(259 - (y-ymin)/(ymax-ymin)*229, 30, 259)) for y in (lo[1], hi[1])]
            cv2.rectangle(grid, (x0, y1), (x1, y0), (245, 245, 245), 1)
    status = np.zeros((300, 400, 3), np.uint8)
    health = heartbeat or result.health
    if health["state"] in ("STALE", "ERROR"):
        grid = (grid * 0.35).astype(np.uint8)
        cv2.putText(grid, "STALE / HISTORY", (25, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (80, 80, 255), 2)
    lines = [f"{health['state']}: {health.get('fault_reason', '')}",
             f"session={frame.session_id} frame={frame.sequence_id}",
             f"calibration={result.metadata['calibration_id']}",
             f"age={health.get('source_age_s', 0):.3f}s drops={health.get('dropped_frames', 0)}",
             f"valid={result.health['valid_depth_fraction']:.1%} ground={result.ground.status}",
             f"fit p95={result.ground.residual_p95_m:.4f}m coverage={result.ground.support_spatial_coverage:.0%}",
             f"processing={result.health['processing_time_s']:.3f}s grid=" +
                (f"{1/result.health['grid_publication_interval_s']:.1f}Hz" if result.health.get('grid_publication_interval_s') else "initial"),
             f"bounds={result.metadata['bounds_xy_m']}",
             "e: elevation/evidence | q: exit", "GEOMETRY ONLY; physical qualification pending"]
    for i, line in enumerate(lines):
        cv2.putText(status, line[:60], (5, 17 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (80, 80, 255) if i == 0 and health["state"] != "RUNNING" else (230, 230, 230), 1)
    for i, (label, color) in enumerate(COLORS_RGB.items()):
        y = 205 + i * 14
        cv2.rectangle(status, (5, y), (15, y + 9), color[::-1], -1)
        cv2.putText(status, label, (22, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color[::-1], 1)
    return np.vstack((top, np.hstack((grid, status))))


class TerrainViewer:
    def __init__(self, *, cloud=False):
        self.elevation = False
        self.visualizer = None
        self.geometry = None
        if cloud:
            import open3d as o3d
            self.o3d = o3d
            self.visualizer = o3d.visualization.Visualizer()
            if not self.visualizer.create_window("Phase 1: test_ground cloud; status in dashboard"):
                raise RuntimeError("Open3D window could not be created")

    def show(self, frame, result, *, heartbeat=None):
        cv2.imshow("Phase 1 stationary terrain evidence", dashboard(frame, result, elevation=self.elevation, heartbeat=heartbeat))
        if self.visualizer is not None and len(result.points_test_ground):
            cloud = self.o3d.geometry.PointCloud()
            cloud.points = self.o3d.utility.Vector3dVector(result.points_test_ground)
            colors = np.full((len(result.points_test_ground), 3), 0.4)
            labels = result.metadata["point_evidence"]
            for label, rgb in COLORS_RGB.items():
                colors[labels == label] = np.array(rgb) / 255
            if heartbeat and heartbeat["state"] in ("STALE", "ERROR"):
                colors[:] = 0.3
            cloud.colors = self.o3d.utility.Vector3dVector(colors)
            if self.geometry is None:
                self.geometry = cloud
                self.visualizer.add_geometry(self.geometry)
                view = self.visualizer.get_view_control()
                view.set_up([0, 0, 1])
                view.set_front([-1, 0, 0.5])
                view.set_lookat(cloud.get_center())
            else:
                self.geometry.points, self.geometry.colors = cloud.points, cloud.colors
                self.visualizer.update_geometry(self.geometry)
        elif self.visualizer is not None and self.geometry is not None:
            self.geometry.clear()
            self.visualizer.update_geometry(self.geometry)
        if self.visualizer is not None:
            if not self.visualizer.poll_events():
                return False
            self.visualizer.update_renderer()
        key = cv2.waitKey(1) & 0xFF
        if key == ord("e"):
            self.elevation = not self.elevation
        return key not in (27, ord("q"))

    def close(self):
        if self.visualizer is not None:
            self.visualizer.destroy_window()
        cv2.destroyAllWindows()
