"""Organized validity and calibrated projection; invalid samples stay explicit."""

import numpy as np

from ..geometry import CameraIntrinsics, depth_image_to_camera_points
from .config import transform_points


def calibrated_points(frame, config):
    depth = frame.depth_raw
    intrinsic = CameraIntrinsics(*config.intrinsics_fx_fy_cx_cy)
    depth_m = frame.depth_m
    original_valid = (depth > 0) & (depth <= 10000)
    valid = original_valid & (depth_m >= config.qualified_range_m[0]) & (depth_m <= config.qualified_range_m[1])
    masked = np.where(valid, depth, 0).astype(np.uint16)
    optical = depth_image_to_camera_points(masked, intrinsic)
    ground = transform_points(optical, config.T_test_ground_from_depth_optical)
    pixels_vu = np.column_stack(np.nonzero(valid))
    zvalid = (ground[:, 2] >= config.bounds_z_m[0]) & (ground[:, 2] <= config.bounds_z_m[1])
    bounds = config.bounds_xy_m
    zvalid &= ((ground[:, 0] >= bounds[0]) & (ground[:, 0] < bounds[1]) &
               (ground[:, 1] >= bounds[2]) & (ground[:, 1] < bounds[3]))
    return optical[zvalid], ground[zvalid], pixels_vu[zvalid], valid, original_valid


def reference_visibility(frame, config):
    """Project grid subcell centers on z=0 to diagnose potential support.

    This is an expected-reference visibility test, NEVER measured support.
    A closer return means occluded-or-unobserved; zero depth means invalid.
    """
    bounds, resolution = config.bounds_xy_m, config.resolution_m
    nx = round((bounds[1] - bounds[0]) / resolution)
    ny = round((bounds[3] - bounds[2]) / resolution)
    rows, cols = np.indices((ny, nx))
    s = config.support_subdivisions
    reasons = np.full((ny, nx), "OUT_OF_VIEW", dtype="<U32")
    any_visible, any_invalid, any_occluded = (np.zeros((ny, nx), dtype=bool) for _ in range(3))
    Tinv = np.linalg.inv(np.asarray(config.T_test_ground_from_depth_optical))
    fx, fy, cx, cy = config.intrinsics_fx_fy_cx_cy
    height, width = frame.depth_raw.shape
    for sy in range(s):
        for sx in range(s):
            points = np.column_stack((
                (bounds[0] + (cols + (sx + 0.5) / s) * resolution).ravel(),
                (bounds[2] + (rows + (sy + 0.5) / s) * resolution).ravel(),
                np.zeros(nx * ny)))
            optical = transform_points(points, Tinv)
            z = optical[:, 2]
            positive = z > 0
            u = np.zeros(len(z), dtype=int)
            v = np.zeros(len(z), dtype=int)
            u[positive] = np.rint(fx * optical[positive, 0] / z[positive] + cx).astype(int)
            v[positive] = np.rint(fy * optical[positive, 1] / z[positive] + cy).astype(int)
            inside = positive & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            inside &= (z >= config.qualified_range_m[0]) & (z <= config.qualified_range_m[1])
            depths = np.zeros(len(z))
            depths[inside] = frame.depth_raw[v[inside], u[inside]] / 1000
            invalid = inside & ((depths <= 0) | (depths > 10) |
                                (depths < config.qualified_range_m[0]) | (depths > config.qualified_range_m[1]))
            occluded = inside & ~invalid & (depths < z - config.ground_distance_m)
            any_visible |= inside.reshape(ny, nx)
            any_invalid |= invalid.reshape(ny, nx)
            any_occluded |= occluded.reshape(ny, nx)
    reasons[any_visible] = "OCCLUDED_OR_UNOBSERVED"
    reasons[any_invalid] = "INVALID_DEPTH"
    reasons[any_occluded] = "OCCLUDED_OR_UNOBSERVED"
    return reasons
