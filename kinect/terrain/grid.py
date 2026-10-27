"""Confidence-aware elevation/evidence grid; no filling or free-space clearing."""

import numpy as np


def cell_indices(points, config):
    b, r = config.bounds_xy_m, config.resolution_m
    nx, ny = round((b[1] - b[0]) / r), round((b[3] - b[2]) / r)
    xy = np.asarray(points)[:, :2]
    indices = np.floor((xy - [b[0], b[2]]) / r).astype(int)
    valid = ((xy[:, 0] >= b[0]) & (xy[:, 0] < b[1]) &
             (xy[:, 1] >= b[2]) & (xy[:, 1] < b[3]))
    return indices[:, ::-1], valid, (ny, nx)


def components(mask):
    """Four-connected observed cells; IDs are per output, never tracks."""
    remaining = set(map(tuple, np.argwhere(mask)))
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = [], [start]
        while stack:
            cell = stack.pop()
            group.append(cell)
            y, x = cell
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        yield np.asarray(sorted(group), dtype=int)


def summary_labels(grid):
    labels = np.full(grid["point_count"].shape, "UNKNOWN", dtype="<U20")
    labels[grid["observed_support"]] = "OBSERVED_GROUND"
    total = np.zeros(labels.shape, dtype=int)
    for field, name in (("raised_evidence", "RAISED"), ("negative_evidence", "NEGATIVE"), ("edge_evidence", "EDGE")):
        labels[grid[field]] = name
        total += grid[field]
    labels[total > 1] = "MIXED"
    return labels


def build_grid(points, residuals, model, config, timestamp, reasons):
    indices, inside, shape = cell_indices(points, config)
    grid = {}
    numeric = ("elevation_estimate_m", "elevation_min_m", "elevation_max_m", "return_elevation_m",
               "height_residual_m", "height_residual_min_m", "height_residual_max_m",
               "uncertainty_m", "slope_rad", "roughness_m",
               "last_support_timestamp", "last_hazard_timestamp")
    for name in numeric:
        grid[name] = np.full(shape, np.nan)
    for name in ("observed_support", "raised_evidence", "negative_evidence", "edge_evidence",
                 "slope_valid", "roughness_valid"):
        grid[name] = np.zeros(shape, dtype=bool)
    grid["point_count"] = np.zeros(shape, dtype=int)
    grid["spatial_support_coverage"] = np.zeros(shape)
    grid["surface_return_coverage"] = np.zeros(shape)
    grid["unknown_reason"] = reasons.copy()
    valid_ground = model.status == "VALID"
    uncertainty = config.measurement_uncertainty_m + (model.uncertainty_proxy_m if valid_ground else 0)
    order = np.flatnonzero(inside)
    if len(order):
        keys = indices[order, 0] * shape[1] + indices[order, 1]
        order = order[np.argsort(keys, kind="stable")]
        sorted_keys = indices[order, 0] * shape[1] + indices[order, 1]
        splits = np.flatnonzero(np.diff(sorted_keys)) + 1
        for group in np.split(order, splits):
            row, col = indices[group[0]]
            cell = (row, col)
            p = points[group]
            grid["point_count"][cell] = len(group)
            grid["elevation_min_m"][cell] = p[:, 2].min()
            grid["elevation_max_m"][cell] = p[:, 2].max()
            grid["return_elevation_m"][cell] = np.median(p[:, 2])
            grid["uncertainty_m"][cell] = uncertainty
            local = (p[:, :2] - [config.bounds_xy_m[0], config.bounds_xy_m[2]]) / config.resolution_m
            sub = np.minimum(np.floor((local - [col, row]) * config.support_subdivisions).astype(int),
                             config.support_subdivisions - 1)
            grid["surface_return_coverage"][cell] = len(np.unique(sub, axis=0)) / config.support_subdivisions ** 2
            if not valid_ground:
                continue
            h = residuals[group]
            support = np.abs(h) <= config.ground_distance_m
            coverage = len(np.unique(sub[support], axis=0)) / config.support_subdivisions ** 2
            grid["spatial_support_coverage"][cell] = coverage
            grid["height_residual_m"][cell] = np.median(h)
            grid["height_residual_min_m"][cell] = h.min()
            grid["height_residual_max_m"][cell] = h.max()
            grid["raised_evidence"][cell] = np.any(h > config.raised_threshold_m + uncertainty)
            if support.any():
                grid["elevation_estimate_m"][cell] = np.median(p[support, 2])
            # Vertical faces can put many returns into one horizontal strip.
            # Require actual span along BOTH ground axes, independent of count.
            span_ok = support.any() and np.all(np.ptp(p[support, :2], axis=0) >= config.resolution_m * config.minimum_support_span_fraction)
            measured = (support.sum() >= config.minimum_cell_points and coverage >= config.minimum_support_coverage and span_ok)
            grid["observed_support"][cell] = measured
            grid["unknown_reason"][cell] = "NONE" if measured else "SPARSE_SUPPORT"
            if measured:
                grid["last_support_timestamp"][cell] = timestamp
                # Local geometry only where two horizontal directions are constrained.
                support_points = p[support]
                if len(support_points) >= 6:
                    _, singular, vh = np.linalg.svd(support_points - support_points.mean(axis=0), full_matrices=False)
                    if singular[1] > config.resolution_m * 0.1:
                        normal = vh[-1]
                        residual = (support_points - support_points.mean(axis=0)) @ normal
                        grid["slope_rad"][cell] = np.arccos(np.clip(abs(normal[2]), 0, 1))
                        grid["roughness_m"][cell] = np.sqrt(np.mean(residual ** 2))
                        grid["slope_valid"][cell] = grid["roughness_valid"][cell] = True
    if not valid_ground:
        grid["unknown_reason"][:] = "GROUND_MODEL_INVALID"
    else:
        measured_surface = ((grid["point_count"] >= config.minimum_cell_points) &
                            (grid["surface_return_coverage"] >= config.minimum_support_coverage))
        below = measured_surface & (grid["height_residual_m"] < -(config.depression_threshold_m + uncertainty))
        support_components = list(components(grid["observed_support"]))
        # Confirm visible depressions only when measured reference support
        # surrounds the patch in >=3 quadrants inside the context radius.
        for group in components(below):
            center = group.mean(axis=0)
            for support_cells in support_components:
                delta = support_cells - center
                near = np.linalg.norm(delta, axis=1) * config.resolution_m <= config.negative_context_radius_m
                sectors = (delta[near, 0] >= 0).astype(int) * 2 + (delta[near, 1] >= 0)
                if len(np.unique(sectors)) >= 3:
                    grid["negative_evidence"][tuple(group.T)] = True
                    grid["elevation_estimate_m"][tuple(group.T)] = grid["return_elevation_m"][tuple(group.T)]
                    break
        # A missing return alone never establishes an edge. Require a measured
        # lower neighboring return or a visible below-support face in the cell.
        heights = grid["height_residual_m"]
        for dy, dx in ((1, 0), (0, 1)):
            left = (slice(None, -1), slice(None)) if dy else (slice(None), slice(None, -1))
            right = (slice(1, None), slice(None)) if dy else (slice(None), slice(1, None))
            pair = measured_surface[left] & measured_surface[right]
            change = np.abs(heights[left] - heights[right]) > config.edge_threshold_m + uncertainty
            lower = np.minimum(heights[left], heights[right]) < -(config.edge_threshold_m + uncertainty)
            anchored = grid["observed_support"][left] | grid["observed_support"][right]
            edges = pair & change & lower & anchored
            grid["edge_evidence"][left] |= edges
            grid["edge_evidence"][right] |= edges
        grid["edge_evidence"] |= (grid["observed_support"] &
                                  (grid["height_residual_min_m"] < -(config.edge_threshold_m + uncertainty)))
    hazard = grid["raised_evidence"] | grid["negative_evidence"] | grid["edge_evidence"]
    grid["last_hazard_timestamp"][hazard] = timestamp
    grid["summary"] = summary_labels(grid)
    return grid


def observations_from(grid, config, sequence_id, timestamp):
    result = []
    for field, kind in (("raised_evidence", "RAISED_GEOMETRY"),
                        ("negative_evidence", "VISIBLE_DEPRESSION"), ("edge_evidence", "MEASURED_EDGE")):
        for cells in components(grid[field]):
            rows, cols = cells.T
            x0 = config.bounds_xy_m[0] + cols.min() * config.resolution_m
            y0 = config.bounds_xy_m[2] + rows.min() * config.resolution_m
            x1 = config.bounds_xy_m[0] + (cols.max() + 1) * config.resolution_m
            y1 = config.bounds_xy_m[2] + (rows.max() + 1) * config.resolution_m
            low, high = grid["elevation_min_m"][rows, cols], grid["elevation_max_m"][rows, cols]
            finite = np.isfinite(low) & np.isfinite(high)
            z0, z1 = (float(low[finite].min()), float(high[finite].max())) if finite.any() else (None, None)
            result.append(dict(
                observation_id=f"{sequence_id}:{kind}:{len(result)}", type=kind,
                source_sequence_id=sequence_id, source_timestamp=timestamp, frame_id="test_ground",
                source_grid_id=f"grid:{sequence_id}",
                centroid_m=[float(config.bounds_xy_m[0] + (cols.mean() + 0.5) * config.resolution_m),
                            float(config.bounds_xy_m[2] + (rows.mean() + 0.5) * config.resolution_m),
                            (z0 + z1) / 2 if z0 is not None else None],
                bounding_box_m=[[x0, y0, z0], [x1, y1, z1]],
                observed_height_range_m=[z0, z1], supporting_cell_indices=cells.tolist(),
                uncertainty_m=float(np.nanmax(grid["uncertainty_m"][rows, cols])) if finite.any() else None,
                quality="visible extent only; touching geometry may be merged",
                reason_codes=["MEASURED_GEOMETRY", "TOTAL_EXTENT_UNKNOWN_UNDER_OCCLUSION"]
            ))
    return result


class StationaryHistory:
    """Separate optional history: hazards persist; missing returns never clear them."""

    def __init__(self, config):
        self.config = config
        self.grid = None

    def update(self, current, timestamp, *, usable):
        if self.grid is None:
            self.grid = {k: v.copy() for k, v in current.items()}
        elif usable:
            history = self.grid
            for name in ("raised_evidence", "negative_evidence", "edge_evidence"):
                history[name] |= current[name]
            measured = current["observed_support"]
            for name in ("elevation_estimate_m", "spatial_support_coverage", "last_support_timestamp"):
                history[name][measured] = current[name][measured]
            for name in ("last_hazard_timestamp",):
                mask = np.isfinite(current[name])
                history[name][mask] = current[name][mask]
            history["elevation_min_m"] = np.fmin(history["elevation_min_m"], current["elevation_min_m"])
            history["elevation_max_m"] = np.fmax(history["elevation_max_m"], current["elevation_max_m"])
            history["observed_support"] |= measured
            history["unknown_reason"][measured] = "NONE"
        history = self.grid
        stale = ~np.isfinite(history["last_support_timestamp"]) | (
            timestamp - history["last_support_timestamp"] > self.config.support_evidence_ttl_s)
        if not usable:
            stale[:] = True
        history["observed_support"][stale] = False
        history["unknown_reason"][stale] = "STALE"
        history["summary"] = summary_labels(history)
        fields = ("observed_support", "raised_evidence", "negative_evidence", "edge_evidence",
                  "last_support_timestamp", "last_hazard_timestamp", "spatial_support_coverage",
                  "elevation_estimate_m", "elevation_min_m", "elevation_max_m", "unknown_reason", "summary")
        return {k: history[k].copy() for k in fields}
