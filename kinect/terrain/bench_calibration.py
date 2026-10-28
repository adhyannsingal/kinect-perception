"""Measured-target sampling and a candidate fixed-bench transform.

No intrinsic/device calibration is modified. Transform fitting never uses
validation targets, and neither fitting nor validation qualifies hardware.
"""

from dataclasses import replace
import hashlib
import json

import numpy as np
import open3d as o3d

from ..geometry import CameraIntrinsics, pixel_depth_to_camera_xyz
from .config import rigid_transform, transform_points


def calibration_plan(mount_id):
    if not isinstance(mount_id, str) or not mount_id.strip():
        raise ValueError("mount ID must be a nonempty string")
    return dict(
        schema_version=1, mount_transform_id=mount_id,
        origin_description=None, measurement_method=None,
        criteria=dict(max_pair_scale_error_fraction=0.1, minimum_geometry_span_m=0.01,
                      median_error_limit_m=0.025, p95_error_limit_m=0.05),
        targets=[dict(target_id=f"{role}-{i+1}", role=role,
                      expected_test_ground_m=None, measurement_uncertainty_m=None,
                      observation_path=None)
                 for role, count in (("fit", 3), ("validation", 5)) for i in range(count)],
        qualification="UNMEASURED; fill physical coordinates independently of Kinect")


def sample_target(frames, pixel_uv, intrinsics: CameraIntrinsics, *, mount_id, target_id,
                  radius_px=0, minimum_valid_fraction=0.8, maximum_spread_m=0.02):
    """Sample one stationary target in depth pixels over distinct frames.

    frames yields (uint16_depth_mm, timestamp_seconds, source_id). The timestamp
    must use one declared domain (supplied by the caller). No RGB coordinates
    or fabricated expected target position are involved. Median depth is
    projected at the selected pixel; ROI radius zero avoids mixing surfaces.
    Spread is p95-p5 over raw ROI returns, a stability proxy, not calibrated noise.
    """
    if type(radius_px) is not int or radius_px < 0:
        raise ValueError("radius_px must be a nonnegative integer")
    if not 0 < minimum_valid_fraction <= 1 or not np.isfinite(maximum_spread_m) or maximum_spread_m <= 0:
        raise ValueError("valid fraction must be in (0,1]; spread must be finite and positive")
    if not target_id or not mount_id:
        raise ValueError("target and fixed-mount IDs are required")
    if len(pixel_uv) != 2 or any(type(p) is not int for p in pixel_uv):
        raise ValueError("pixel_uv must contain integer u,v coordinates")
    values = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
    if not np.isfinite(values).all() or min(values[:2]) <= 0:
        raise ValueError("invalid intrinsics")
    u, v = pixel_uv
    samples, sources, timestamps, total, valid_count = [], [], [], 0, 0
    shape = None
    for depth, timestamp, source_id in frames:
        if depth.dtype != np.uint16 or depth.ndim != 2:
            raise ValueError("target sampling requires uint16 metric millimetre depth")
        if shape is None:
            shape = depth.shape
        if shape != depth.shape:
            raise ValueError("depth resolution changed while sampling")
        if not np.isfinite(timestamp) or timestamp < 0 or (timestamps and timestamp <= timestamps[-1]):
            raise ValueError("sampling timestamps must be distinct and increasing; check source freeze/clock wrap")
        if not (radius_px <= u < depth.shape[1] - radius_px and radius_px <= v < depth.shape[0] - radius_px):
            raise ValueError("selected pixel/ROI lies outside the depth image")
        patch = depth[v-radius_px:v+radius_px+1, u-radius_px:u+radius_px+1]
        valid = (patch > 0) & (patch <= 10000)
        samples.extend((patch[valid].astype(float) / 1000).tolist())
        total += patch.size
        valid_count += int(valid.sum())
        sources.append(dict(source_id=str(source_id), depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest()))
        timestamps.append(float(timestamp))
    if len(timestamps) < 3:
        raise ValueError("at least three distinct frames are required for a target observation")
    fraction = valid_count / total
    spread = float(np.ptp(np.percentile(samples, [5, 95]))) if samples else None
    median = float(np.median(samples)) if samples else None
    status = "ACCEPTED_SAMPLE"
    if fraction < minimum_valid_fraction:
        status = "INSUFFICIENT_VALID_DEPTH"
    elif spread > maximum_spread_m:
        status = "UNSTABLE_OR_MIXED_SURFACES"
    observed = pixel_depth_to_camera_xyz(u, v, median, intrinsics) if status == "ACCEPTED_SAMPLE" else None
    return dict(
        schema_version=1, target_id=target_id, mount_transform_id=mount_id,
        status=status, frame_id="depth_optical", units="metres", depth_format="uint16_mm_optical_z",
        stream_width=shape[1], stream_height=shape[0], intrinsics_fx_fy_cx_cy=list(values),
        pixel_uv=[u, v], frame_count=len(timestamps), valid_sample_fraction=fraction,
        depth_median_m=median, depth_p95_p5_spread_m=spread,
        observed_depth_optical_m=observed, expected_test_ground_m=None,
        policy=dict(radius_px=radius_px, minimum_valid_fraction=minimum_valid_fraction,
                    maximum_spread_m=maximum_spread_m),
        source_timestamps=timestamps, sources=sources,
        qualification="UNQUALIFIED; sample stability does not establish metric accuracy")


def _points(value, name):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite metre 3-vectors")
    return points


def fit_rigid_mount(observed, expected, *, max_scale_error_fraction=0.1, minimum_span_m=0.01):
    """Known correspondences; Open3D rigid point-to-point fit with scaling off."""
    observed, expected = _points(observed, "observed"), _points(expected, "expected")
    if observed.shape != expected.shape or len(observed) < 3:
        raise ValueError("at least three paired fitting targets are required")
    if not np.isfinite(max_scale_error_fraction) or not 0 < max_scale_error_fraction < 1:
        raise ValueError("max_scale_error_fraction must be in (0,1)")
    if not np.isfinite(minimum_span_m) or minimum_span_m <= 0:
        raise ValueError("minimum_span_m must be finite and positive")
    for points in (observed, expected):
        singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        if singular[1] < minimum_span_m:
            raise ValueError("fitting targets are collinear or have inadequate spatial spread")
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        if np.any(distances[np.triu_indices(len(points), 1)] < minimum_span_m):
            raise ValueError("duplicate or insufficiently separated fitting targets")
    pairs = np.triu_indices(len(observed), 1)
    a = np.linalg.norm(observed[:, None] - observed[None, :], axis=2)[pairs]
    b = np.linalg.norm(expected[:, None] - expected[None, :], axis=2)[pairs]
    scale_error = np.abs(b / a - 1)
    if np.any(scale_error > max_scale_error_fraction):
        raise ValueError("target distances disagree; check units, correspondences and depth calibration (no scale correction applied)")
    source, target = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(observed)
    target.points = o3d.utility.Vector3dVector(expected)
    correspondence = o3d.utility.Vector2iVector(np.column_stack((np.arange(len(observed)), np.arange(len(observed)))))
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
    transform = estimator.compute_transformation(source, target, correspondence)
    rigid_transform(transform)
    errors = np.linalg.norm(transform_points(observed, transform) - expected, axis=1)
    return transform, dict(fit_target_count=len(observed), errors_m=errors,
                           maximum_pair_scale_error_fraction=float(scale_error.max()), scaling_enabled=False)


def build_candidate(plan, observations, config):
    """Fit the mount using fit targets, then report held-out validation errors.

    Remaining terrain-profile fields stay unresolved unless measured/provided.
    This function cannot create a qualified profile, even for a perfect fit.
    """
    if plan.get("schema_version") != 1 or not plan.get("origin_description") or not plan.get("measurement_method"):
        raise ValueError("schema 1, physical origin description and measurement method are required")
    targets = plan["targets"]
    ids = [t["target_id"] for t in targets]
    if len(set(ids)) != len(ids):
        raise ValueError("target IDs must be unique")
    fit, validation, signatures, streams, synthetic_sources = [], [], set(), [], []
    for target in targets:
        if target["role"] not in ("fit", "validation"):
            raise ValueError("target role must be fit or validation")
        observation = observations[target["target_id"]]
        if observation.get("schema_version") != 1 or observation["status"] != "ACCEPTED_SAMPLE":
            raise ValueError(f"target {target['target_id']} lacks an accepted observation")
        if observation["mount_transform_id"] != plan["mount_transform_id"] or observation["target_id"] != target["target_id"]:
            raise ValueError("observation identity/mount mismatch")
        if observation.get("frame_id") != "depth_optical" or observation.get("units") != "metres" or observation.get("depth_format") != "uint16_mm_optical_z":
            raise ValueError("observation coordinate frame/units/depth format mismatch")
        if target["expected_test_ground_m"] is None or target["measurement_uncertainty_m"] is None:
            raise ValueError("independent physical coordinates and uncertainty must not be null")
        uncertainty = float(target["measurement_uncertainty_m"])
        if not np.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError("measurement uncertainty must be finite and nonnegative")
        observed = _points([observation["observed_depth_optical_m"]], "observed")[0]
        expected = _points([target["expected_test_ground_m"]], "expected")[0]
        if not 0 < observed[2] <= 10:
            raise ValueError("observed optical Z must lie in the supported (0,10] metre range")
        if any(np.linalg.norm(expected - item[1]) < 1e-8 for item in fit + validation):
            raise ValueError("fitting and validation targets must be physically distinct")
        # Same archived frame and pixel cannot masquerade as independent targets.
        signature = json.dumps([observation["sources"], observation["pixel_uv"]], sort_keys=True)
        if signature in signatures:
            raise ValueError("an observation was reused across target IDs")
        signatures.add(signature)
        streams.append((observation["stream_width"], observation["stream_height"], observation["intrinsics_fx_fy_cx_cy"]))
        synthetic_sources.append(observation.get("source_calibration_kind") == "synthetic")
        (fit if target["role"] == "fit" else validation).append((observed, expected, uncertainty))
    if len(fit) < 3 or len(validation) < 5:
        raise ValueError("need >=3 fitting targets AND >=5 separate validation targets")
    if any(stream != streams[0] for stream in streams):
        raise ValueError("stream resolution/intrinsics changed between observations")
    if any(synthetic_sources) and not all(synthetic_sources):
        raise ValueError("synthetic and physical/unknown observations cannot be mixed in a mount fit")
    width, height, intrinsics = streams[0]
    if config.intrinsics_fx_fy_cx_cy is not None and config.intrinsics_fx_fy_cx_cy != intrinsics:
        raise ValueError("candidate config intrinsics disagree with observations")
    criteria = plan["criteria"]
    for name in ("median_error_limit_m", "p95_error_limit_m"):
        if not np.isfinite(criteria[name]) or criteria[name] <= 0:
            raise ValueError("validation error budgets must be finite and positive")
    if criteria["median_error_limit_m"] > criteria["p95_error_limit_m"]:
        raise ValueError("median limit must not exceed p95 limit")
    transform, fit_report = fit_rigid_mount(
        [r[0] for r in fit], [r[1] for r in fit],
        max_scale_error_fraction=criteria["max_pair_scale_error_fraction"],
        minimum_span_m=criteria["minimum_geometry_span_m"])
    observed = np.array([r[0] for r in validation])
    expected = np.array([r[1] for r in validation])
    uncertainty = np.array([r[2] for r in validation])
    errors_xyz = transform_points(observed, transform) - expected
    errors = np.linalg.norm(errors_xyz, axis=1)
    median, p95 = np.percentile(errors, [50, 95])
    span = np.ptp(expected, axis=0)
    diversity_ok = bool(np.all(span > max(2 * uncertainty.max(), 1e-8)))
    within_budget = bool(median < criteria["median_error_limit_m"] and p95 < criteria["p95_error_limit_m"])
    digest = hashlib.sha256(json.dumps([plan, observations], sort_keys=True).encode()).hexdigest()[:16]
    candidate = replace(config, calibration_id=f"bench-candidate-{digest}",
                        calibration_kind="synthetic" if all(synthetic_sources) else "bench_unqualified",
                        mount_transform_id=plan["mount_transform_id"],
                        measurement_method=plan["measurement_method"] + "; origin: " + plan["origin_description"],
                        stream_width=width, stream_height=height, intrinsics_fx_fy_cx_cy=intrinsics,
                        distortion_model="pinhole_approximation", depth_representation="uint16_mm_optical_z",
                        T_test_ground_from_depth_optical=transform.tolist())
    report = dict(schema_version=1, frame_id="test_ground", source_frame_id="depth_optical",
                  status="SYNTHETIC_WORKFLOW_TEST_ONLY" if all(synthetic_sources) else "PENDING_PHYSICAL_REVIEW", hardware_qualified=False,
                  candidate_config_id=candidate.config_id, T_test_ground_from_depth_optical=transform,
                  candidate_camera_position_test_ground_m=transform[:3, 3],
                  candidate_downward_pitch_rad=float(np.arctan2(-transform[2, 2], np.hypot(transform[0, 2], transform[1, 2]))),
                  fit_target_ids=[t["target_id"] for t in targets if t["role"] == "fit"],
                  validation_target_ids=[t["target_id"] for t in targets if t["role"] == "validation"],
                  fit=fit_report, validation_errors_m=errors, validation_axis_errors_m=errors_xyz,
                  validation_median_error_m=median, validation_p95_error_m=p95,
                  validation_measurement_uncertainty_m=uncertainty,
                  validation_optical_z_m=observed[:, 2], validation_spans_xyz_m=span,
                  validation_has_forward_lateral_height_variation=diversity_ok,
                  within_declared_error_budget=within_budget, declared_criteria=criteria,
                  terrain_profile_complete=candidate.calibration_ready,
                  pending=["human review of independent measurements, axes and fixture method",
                           "qualified range, blind zones and uncertainty budget",
                           "ground seed/grid bounds and physical scene qualification"])
    return candidate, report
