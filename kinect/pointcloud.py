"""Open3D point-cloud conversion for Kinect v1 metric depth frames."""

import numpy as np
import open3d as o3d

from .geometry import DEFAULT_KINECT_V1_DEPTH_INTRINSICS, CameraIntrinsics, depth_image_to_camera_points


def depth_mm_to_point_cloud(
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics = DEFAULT_KINECT_V1_DEPTH_INTRINSICS,
    voxel_size_m: float | None = None,
) -> o3d.geometry.PointCloud:
    """Create a geometry-only Open3D cloud from a uint16 millimetre frame.

    Points use the depth-camera convention from :mod:`kinect.geometry`: metres,
    +X right, +Y down, +Z forward.  No RGB colors are attached because colors
    require trustworthy RGB/depth registration.
    """
    points = depth_image_to_camera_points(depth_mm, intrinsics)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if voxel_size_m is not None:
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        cloud = cloud.voxel_down_sample(voxel_size_m)
    return cloud
