"""Fixture comparisons and calibration metrics; never auto-qualify hardware."""

import numpy as np

from .config import transform_points
from .contracts import write_json
from .fixtures import LAYOUTS, fixture, synthetic_config
from .pipeline import TerrainPipeline


def compare_annotations(result, annotations):
    """Compare independently supplied fixture extents with measured outputs.

    Detection requires a same-type visible bounding box overlap, not proximity
    alone. Unknown-area false ground and visible-ground coverage are separate.
    These coarse fixture metrics do not substitute for physical error budgets.
    """
    bounds = result.metadata["bounds_xy_m"]
    resolution = result.metadata["resolution_m"]
    rows, cols = np.indices(result.grid["point_count"].shape)
    x = bounds[0] + (cols + 0.5) * resolution
    y = bounds[2] + (rows + 0.5) * resolution
    comparisons = []
    for annotation in annotations:
        b = annotation["bounds_xy_m"]
        if np.asarray(b).shape != (4,) or not np.isfinite(b).all() or not (b[0] < b[1] and b[2] < b[3]):
            raise ValueError("annotation bounds must be finite [xmin,xmax,ymin,ymax]")
        area = (x >= b[0]) & (x < b[1]) & (y >= b[2]) & (y < b[3])
        matches = []
        for observation in result.observations:
            if observation["type"] != annotation["type"]:
                continue
            lo, hi = observation["bounding_box_m"]
            overlap = max(0, min(b[1], hi[0]) - max(b[0], lo[0])) * max(0, min(b[3], hi[1]) - max(b[2], lo[1]))
            if overlap > 0:
                matches.append(observation)
        false_ground = int(np.count_nonzero(area & result.grid["observed_support"]))
        comparisons.append(dict(
            type=annotation["type"], detected=bool(matches), target_cell_count=int(area.sum()),
            false_ground_cells=false_ground if annotation["type"] in ("UNKNOWN", "VISIBLE_DEPRESSION") else None,
            max_observed_height_m=max((o["observed_height_range_m"][1] for o in matches), default=None),
            min_observed_height_m=min((o["observed_height_range_m"][0] for o in matches), default=None),
            matched_observation_ids=[o["observation_id"] for o in matches]))
    return comparisons


def synthetic_evaluation(output):
    config = synthetic_config()
    layouts = []
    for layout in LAYOUTS:
        frame, annotations = fixture(layout)
        result = TerrainPipeline(config).process(frame)
        comparison = compare_annotations(result, annotations)
        hazards = any(result.grid[name].any() for name in ("raised_evidence", "negative_evidence", "edge_evidence"))
        if layout == "wall":
            passed = result.ground.status != "VALID" and not result.grid["observed_support"].any()
        elif layout in ("flat", "dropout"):
            passed = result.ground.status == "VALID" and not hazards
            passed &= all(c["false_ground_cells"] in (0, None) for c in comparison)
        else:
            passed = result.ground.status == "VALID" and all(c["detected"] for c in comparison)
        layouts.append(dict(layout=layout, passed=bool(passed), ground_status=result.ground.status,
                            support_cells=int(result.grid["observed_support"].sum()),
                            grid_cells=int(result.grid["point_count"].size),
                            ground_residual_p95_m=result.ground.residual_p95_m,
                            processing_time_s=result.health["processing_time_s"], comparisons=comparison))
    report = dict(schema_version=1, qualification="SYNTHETIC_LOGIC_ONLY", hardware_qualified=False,
                  config_id=config.config_id, effective_configuration=config.__dict__,
                  scene_count=len(layouts), frame_count=len(layouts), layouts=layouts,
                  all_passed=all(layout["passed"] for layout in layouts),
                  pending=["five measured target locations", "five independent physical layouts",
                           "Kinect noise/range/error budget", "live freeze and load measurement",
                           "real-recorded-session regression", "local display review"])
    write_json(output, report)
    return report


def calibration_check(config, measurements):
    if not config.calibration_ready:
        raise ValueError("complete the bench profile before evaluating calibration")
    targets = measurements["targets"]
    if len(targets) < 5:
        raise ValueError("at least five independently measured targets are required")
    observed = np.array([t["observed_depth_optical_m"] for t in targets], dtype=float)
    expected = np.array([t["expected_test_ground_m"] for t in targets], dtype=float)
    uncertainty = np.array([t["measurement_uncertainty_m"] for t in targets], dtype=float)
    if observed.shape != (len(targets), 3) or expected.shape != observed.shape or not np.isfinite(observed).all() or not np.isfinite(expected).all():
        raise ValueError("target positions must be finite metre 3-vectors")
    if not np.isfinite(uncertainty).all() or np.any(uncertainty < 0):
        raise ValueError("measurement uncertainty must be finite and nonnegative")
    errors = np.linalg.norm(transform_points(observed, config.T_test_ground_from_depth_optical) - expected, axis=1)
    median, p95 = np.percentile(errors, [50, 95])
    return dict(schema_version=1, calibration_id=config.calibration_id, config_id=config.config_id,
                frame_id="test_ground", observed_input_frame_id="depth_optical",
                target_count=len(targets), position_errors_m=errors, measurement_uncertainties_m=uncertainty,
                optical_ranges_m=np.linalg.norm(observed, axis=1),
                ground_axis_errors_m=transform_points(observed, config.T_test_ground_from_depth_optical) - expected,
                median_error_m=median, p95_error_m=p95,
                proposed_budget_m=dict(median=config.resolution_m / 2, p95=config.resolution_m),
                within_proposed_budget=bool(median < config.resolution_m / 2 and p95 < config.resolution_m),
                qualification="REQUIRES_PHYSICAL_MEASUREMENT_REVIEW; profile not automatically qualified")
