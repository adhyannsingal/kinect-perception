# Phase 1 — Kinect terrain perception: technical build specification

**Project:** NASA Lunabotics autonomous traversal  
**Specification version:** 1.0 — October 21, 2026  
**Suggested repository location:** `docs/specs/PHASE_1_KINECT_PERCEPTION_SPEC.md`  
**Audience:** student implementers, coding agents, and technical reviewers  
**Status:** implementation specification; no implementation or hardware qualification claimed

## 1. Objective and current baseline

Build a standalone, replayable terrain-perception application using the team's Xbox 360 Kinect (Kinect v1).

Already working, according to the project owner:

- Kinect libraries and device access.
- Live RGB and depth streams.
- Live 3D point-cloud generation.

The mechanical design, rover dimensions, final sensor mount, encoders, IMU, motor interface, and onboard deployment computer are not finalized. Phase 1 must proceed without them.

**Phase 1 completion means:** from a fixed Kinect observing a measured test scene, the application produces a metric, ground-aligned point cloud; a confidence-aware elevation grid; raised-object and negative-terrain evidence; and an explanatory display. The same processing must run on saved recordings, with documented limitations and qualification results.

This is a perception milestone. It must not claim that observed ground is safe for an unspecified rover, or that a stationary bench demonstration proves autonomous traversal.

## 2. Scope boundaries

### Required

1. Preserve and adapt the working Kinect acquisition pipeline.
2. Record and replay depth, optional RGB, calibration, timestamps, and session metadata.
3. Validate metric scale, coordinate conventions, and a temporary bench-mount transform.
4. Validate depth and expose coverage, missing observations, and stale inputs.
5. Estimate ground/support surfaces with explicit fit rejection.
6. Generate an elevation/evidence grid.
7. Detect raised objects, visible depressions, and drop/edge evidence.
8. Keep unobserved and occluded support distinguishable from measured terrain.
9. Visualize intermediate outputs, confidence, health, and geometry.
10. Supply meaningful tests, saved fixtures, documentation, and a release report.

### Deferred

Rover footprint/inflation, rover-specific traversability thresholds, localization/SLAM, encoders/IMU fusion, mission-frame coordinates, goal selection, route planning/A*, path following, motor commands, ROS integration, MuJoCo, learned perception, competition networking, and actual-rover testing.

**Do not implement deferred features as speculative scaffolding.** Provide clean data boundaries and metadata that allow them later. No motor transport or motion-enabling API belongs in this phase.

## 3. Binding implementation principles

The words MUST and MUST NOT define required behavior. SHOULD indicates a preferred approach that may be changed with a recorded reason.

- Preserve working device code unless a reproducible defect requires a change.
- Inspect the repository and its `AGENTS.md` before selecting file paths, language, dependencies, or tests.
- Prefer the existing language and environment. The module names below describe responsibilities, not a mandatory rewrite.
- Geometry uses meters and radians. Every geometric output declares its coordinate frame.
- Invalid depth MUST NOT become zero-height ground, interpolated support, or a confirmed crater by default.
- Unknown space MUST remain explicit, including outside the sensor's visible footprint.
- A bad ground model MUST produce unavailable/uncertain results rather than plausible-looking confident classifications.
- Phase 1 MUST NOT use final rover measurements or require their availability.
- The Kinect remains fixed relative to the test surface during an accumulated session. Moving or tilting it invalidates that session's transform and accumulated map.
- No global map accumulation across camera motion is allowed without later, separately qualified pose estimation.
- All tunable thresholds and their units MUST be saved with the output.
- Detect raised/negative **geometry**. Do not promise semantic identification of natural boulders or craters from appearance.

## 4. Coordinate and calibration contract

### Frames

| Frame | Definition |
|---|---|
| `depth_optical` | Depth camera optical frame; convention verified against the existing driver |
| `rgb_optical` | RGB camera frame, if color alignment is used |
| `test_ground` | Fixed bench frame: x forward horizontally, y left, z up; z=0 is the measured reference support plane |

The `test_ground` origin is a documented point on the reference plane, preferably the projected camera center. Store its definition in the calibration report. Do not call this an arena or rover frame.

Store `T_test_ground_from_depth_optical` as an explicit 4×4 rigid transform: it maps depth-optical points into test-ground coordinates. Record translation in meters, quaternion order if used, and calibration ID. Validate rotation orthonormality and determinant +1. Never infer transform direction from an ambiguous variable name.

For verified optical-axis metric depth Z, the pinhole relation after the appropriate calibration/undistortion is:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
p_test_ground = T_test_ground_from_depth_optical * [X, Y, Z, 1]
```

The adapter MUST establish whether the driver delivers raw codes, millimeters, meters, registered depth, or another representation. Do not apply this formula directly to raw codes or to ray distance. Intrinsics must match the stream's resolution and registration state. Avoid applying distortion correction twice.

### Required calibration artifacts

- Device/model identifier and stream mode.
- Intrinsics, distortion model, depth representation/conversion, and valid-value handling.
- Bench-mount transform and physical measurement method.
- Qualified distance region and observed blind zones.
- Measured depth/position/height errors versus range.
- RGB/depth extrinsics and pairing tolerance only if RGB registration is enabled.

Validate at least five known target locations across the intended working region. Include lateral points and surfaces with known height differences. Record measurement uncertainty; the measuring fixture itself is not perfect ground truth.

**Acceptance:** signs/axes are correct; increasing target distance behaves correctly; measured errors fit a declared error budget. A proposed initial target is median error below half a grid cell and 95th-percentile error below one cell inside the qualified region. If the camera cannot meet this, narrow the qualified region, adjust the grid/error budget, and report the result. Never manufacture a passing calibration by silently changing the criteria.

## 5. Runtime architecture

```text
Live Kinect adapter ──┐
                      ├─> FrameBundle -> validation -> calibrated cloud
Recording replay ─────┘                                 |
                                                       v
                                 ground/support estimation + quality
                                                       |
                                                       v
                             elevation, coverage, and geometry evidence
                                                       |
                                                       v
                                  object/edge/depression observations
                                                       |
                                                       v
                                    dashboard + logs + export reports
```

Processing MUST be usable without the display and without physical Kinect hardware when replaying. Visualization MUST consume processing outputs rather than implement a second classification pipeline.

Use a bounded live-processing queue that favors the latest frame; expose dropped frames and queue age. Recorders may have a different bounded queue and must report data loss. Do not accumulate an unbounded backlog that makes a live view appear current while showing old terrain.

### Suggested responsibilities

| Module | Responsibility |
|---|---|
| `acquisition` | Wrap existing hardware access and normalize metadata |
| `recording` / `replay` | Persist and reproduce the source contract |
| `calibration` / `coordinates` | Metric conversion and rigid transforms |
| `depth_quality` | Validity, range masks, timestamps, pairing, coverage |
| `ground` | Support estimation, residuals, model-quality rejection |
| `terrain_grid` | Spatial aggregation and evidence states |
| `raised_objects` | Connected raised-geometry observations |
| `negative_terrain` | Depressions, discontinuities, unknown-support reasons |
| `visualization` | Read-only scene, grid, health, and legend |
| `pipeline` | Scheduling, configuration, health propagation |
| `evaluation` | Fixture comparison, metrics, saved release report |

Optional initial tools include numeric arrays, existing image/calibration libraries, and a point-cloud library. Add a dependency only when it solves a concrete task and works in the team's environment. No exact API or library version is mandated here.

## 6. Data contracts and storage

Use serializable structures with an explicit schema version. These contracts may be native classes internally, but exported fields must preserve their meaning.

### FrameBundle

```text
schema_version
session_id, sequence_id
depth_capture_timestamp, receive_timestamp, timestamp_clock_domain
device_timestamp_optional, device_to_host_clock_mapping_optional
depth_raw, depth_format, depth_conversion_id
depth_m[H,W], depth_valid_mask[H,W]
rgb_optional, rgb_capture_timestamp_optional, pairing_error_optional
depth_intrinsics, distortion_model, calibration_id
depth_frame_id, mount_transform_id
```

Use monotonic time for live ages and watchdogs. Store wall-clock time as additional human-readable session metadata. When device capture time is unavailable, label receive time as a proxy and report the unmeasured transport delay. Do not claim true capture-to-output latency from a receive-only timestamp.

For replay, use recorded time for algorithmic aging and original frame order. Use a separate current clock for playback performance. Provide real-time, accelerated, and single-step replay where practical. Do not let an accelerated replay accidentally treat recorded timestamps as current live timestamps.

### GroundModel

```text
frame_id, source_sequence_id, source_timestamp, model_id
status: VALID | INSUFFICIENT_SUPPORT | BAD_ORIENTATION | HIGH_RESIDUAL | INCONSISTENT
unit_normal, plane_offset
support_mask_or_indices, support_spatial_coverage
residual_statistics, parameter_uncertainty_or_documented_proxy
```

For a unit normal pointing up, signed height relative to the plane is `h = n·p + d`. Positive values indicate geometry above the reference; negative values indicate geometry below it. Elevation z in `test_ground` and residual h against a locally fitted support plane are different quantities; label both correctly.

### TerrainGrid

Metadata: schema/version, session, grid ID, coordinate frame, source sequence/timestamp, publication timestamp, resolution, origin, dimensions, calibration/configuration IDs, and health.

Per cell:

```text
elevation_estimate_m, elevation_min_m, elevation_max_m
height_residual_m, uncertainty_m
point_count, spatial_support_coverage
slope_rad, roughness_m, feature_validity_flags
last_support_timestamp, last_hazard_timestamp
observed_support: true/false
raised_evidence: true/false
negative_evidence: true/false
edge_evidence: true/false
unknown_reason: NONE | OUT_OF_VIEW | INVALID_DEPTH | OCCLUDED_OR_UNOBSERVED
                | SPARSE_SUPPORT | STALE | GROUND_MODEL_INVALID
```

Numeric values without measurement must be NaN/null with explicit validity, not fabricated zeros. Evidence flags may coexist: a cell can contain both an object return and insufficient support below/behind it. A single exclusive class must not discard conflicting evidence.

For the display, derive a summary label: UNKNOWN, OBSERVED_GROUND, RAISED, NEGATIVE, EDGE, or MIXED. Hazard labels do not mean the rest of the cell is fully observed. No output class is named SAFE or TRAVERSABLE in Phase 1.

### ObjectObservation / NegativeTerrainObservation

Include observation ID, source grid/time/frame, type, centroid, bounding box/polygon in meters, observed height/depth range, supporting cell indices, uncertainty/quality information, and reason codes. Report measured visible extent; occlusion means total object dimensions or crater depth may be unknown.

Do not require persistent object tracking. IDs may be unique per output frame and must be documented as such.

### Session storage

```text
session/
  manifest.json             # Schema, device, stream modes, clocks, scene metadata
  calibration/              # Intrinsics, conversion, bench transform
  config/                   # Complete effective configuration
  frames/                   # Raw depth and optional RGB, lossless/chunked
  frame_index.jsonl         # Sequence IDs, timestamps, files, validity, drops
  annotations/              # Measured fixture extents and uncertainty
  outputs/                  # Optional derived grids/observations
  evaluation/               # Metrics and review images
```

The exact container may differ if the existing project already has an appropriate format. Raw depth must retain original bit depth and invalid values; a colorized screenshot is not a depth recording. Store calibration/configuration once per session where possible. Save derived clouds only if useful; they should be reproducible from the recorded source.

## 7. Ground segmentation algorithm requirements

1. Apply original depth-validity and qualified-range masks. Preserve organized-image correspondence before downsampling.
2. Transform valid points into `test_ground` and reject out-of-region measurements.
3. Select a credible support seed region from the measured fixture geometry and expected ground orientation. This region must actually contain visible support, not an assumed free patch.
4. Estimate a plane robustly for the initial flat-ground fixture. Constrain orientation and plausible height using the bench calibration; do not select the largest plane without checking whether it is ground.
5. Validate fit orientation, spatial distribution of inliers, residuals, temporal consistency, and uncertainty. Point count alone is insufficient.
6. Emit signed residuals and a valid/invalid model status.

A wall, tabletop, box top, or crater bottom must not silently replace the reference support surface. If credible ground is absent, preserve measured elevations but mark ground-relative interpretation unavailable. Show rejected fits explicitly.

For Phase 1, qualify the algorithm on a declared flat or gently varying reference surface. Local surface fitting for uneven terrain is an extension within perception; if not implemented, mark those scenes unsupported. Never claim a one-plane fit provides general terrain understanding. Estimate local slope/roughness only where sufficient local support exists.

Keep downsampling/outlier rejection configurable. Verify that small real objects and rim points survive preprocessing; otherwise retain a higher-resolution branch for hazard evidence.

## 8. Elevation and support grid requirements

Start with 0.05 m cells and configurable bounds in `test_ground`. The bounds are a storage region, not a claim of observed range. Initialize all cells as unknown.

For point (x,y), compute cell indices by flooring `(coordinate - origin) / resolution`. Reject out-of-bounds points; define boundary behavior explicitly and test it. Do not truncate negative coordinates toward zero.

Maintain a robust representative elevation plus min/max information. Separate candidate support points from raised geometry before computing support statistics so a rock's top does not become the ground estimate. A vertical surface's numerous pixels must not count as broad horizontal support.

Estimate spatial support coverage across a cell or local patch; many neighboring image pixels seeing a tiny part of one cell do not verify its full area. Document the coverage estimator and its limitations. Point count and coverage are separate fields.

Maintain current-frame evidence separately from optional stationary accumulated evidence. A stale accumulated map must not masquerade as a current observation. If accumulating:

- Use the fixed measured transform and preserve source times.
- Age observed-ground evidence back to unknown according to configuration.
- Retain confirmed hazard evidence until sufficient contrary observation or explicit map reset.
- Never clear hazards through missing returns alone.
- Reset/restart the session after camera relocation or mount changes.

No ray that travels through empty air establishes ground support under that ray. Do not apply ordinary occupancy free-space clearing to negative-terrain evidence.

## 9. Raised-object / boulder geometry requirements

Detect points/cells above credible support by more than the configured **bench feature threshold plus measurement uncertainty**. Group connected elevated cells into observations. Preserve the original evidence even if grouping fails.

Output position, visible horizontal extent, maximum observed height, and quality. Generic raised objects include boxes, rocks, supports, and unknown protrusions. Use the label BOULDER_PROXY in fixture metadata if appropriate, not an unverified semantic claim from the detector.

Do not discard small objects merely because they are below a chosen minimum boulder diameter. Record their geometry; use an explicit noise/feature-resolution policy. Test touching objects and partially visible objects. Report merged clusters as merged geometry when separation is not supported.

## 10. Crater, depression, edge, and unknown-support requirements

Implement and display these distinct cases:

| Observation | Evidence | Required interpretation |
|---|---|---|
| Visible depression | Valid surface below surrounding connected support | NEGATIVE evidence with observed depth; total depth may remain unknown |
| Drop or rim | Measured discontinuity/edge relative to surrounding support | EDGE evidence; do not require a visible crater floor |
| Missing expected ground | Invalid, occluded, sparse, or absent support | UNKNOWN support with a reason, not a confirmed crater |
| Ordinary raised object occlusion | Foreground geometry hides terrain | Raised evidence plus unknown terrain behind it |

Compare measured depression geometry against surrounding credible support, not an arbitrary global minimum z. Do not fit away the depression by allowing its bottom to redefine the reference surface.

Compute slope only between measured, spatially compatible support patches. Missing cells do not supply elevations for slope calculations. A discontinuity may justify edge evidence when geometry supports it; dropout alone does not.

Never hole-fill/interpolate missing ground before hazard classification. Any optional visualization interpolation must be visibly distinguished and must not modify exported measurement evidence.

Qualify on a real recessed fixture/depression, an edge with an occluded floor, and flat ground with artificially missing depth. Include shallow changes near the detection limit. Record false positives as well as misses.

## 11. Configuration and qualification policy

Use a validated, versioned configuration with explicit units. Example structure; null values are unresolved and must produce a calibration-required state or disable the dependent feature.

```yaml
schema_version: 1
mode: stationary_bench_perception
grid:
  resolution_m: 0.05
  bounds_xy_m: null                 # Choose from the measured fixture
camera:
  calibration_id: null
  depth_representation: null
  qualified_range_m: null
  T_test_ground_from_depth_optical: null
ground:
  seed_region_xy_m: null
  max_normal_deviation_rad: null
  max_residual_m: null
  minimum_spatial_coverage: null
features:
  raised_threshold_m: null
  depression_threshold_m: null
  edge_threshold_m: null
  minimum_support_coverage: null
timing:
  target_grid_hz: 10
  maximum_input_age_s: null
  support_evidence_ttl_s: null
```

The 10 Hz target is a proposed bench-performance objective, not a measured device capability or a driving authorization. Select freshness limits after measuring this system. Thresholds describe detectable bench geometry, not rover crossing capability.

Derive noise margins from calibration: compare flat-support residual distributions at different ranges/angles with the smallest target feature. If noise and target feature magnitudes overlap, restrict the qualified region or state that the feature cannot be reliably resolved. Do not tune only until the demonstration looks clean.

Validate finite values, positive cell size, consistent bounds, valid transforms, nonnegative timing limits, and compatible feature settings. Export the fully resolved configuration and its hash/ID with every run.

## 12. Visualization and health behavior

Required views:

1. RGB if available, plus depth with invalid pixels visibly distinct.
2. Ground-aligned cloud with support, raised points, negative points, and rejected/unknown evidence.
3. Top-down elevation/evidence grid with metric axes, legend, unknown regions, and observed geometry outlines.
4. Status panel: mode, frame/calibration ID, source age, grid rate, dropped frames, valid-depth fraction, ground-model quality, and fault reason.

Use consistent labels/colors; OBSERVED_GROUND must not be labeled rover-safe. Show elevation and class/confidence as separate selectable views where overlaying everything obscures meaning.

Health states: INITIALIZING, CALIBRATION_REQUIRED, RUNNING, DEGRADED, STALE, and ERROR. Document their transitions. A frozen/disconnected sensor marks current output stale; historical results may remain visible only with an unmistakable stale/history indication.

The display can inspect data and reset a stationary map. It must not hide poor-quality frames, silently replace measurements, or use manual object labels as detector inputs. Human annotations belong in the separate evaluation layer.

## 13. Evaluation dataset and acceptance matrix

Create development sessions for tuning and distinct qualification sessions for final evaluation. Include measured fixture dimensions/location, camera mount, surface/lighting, valid range, algorithm configuration, and annotation uncertainty. Reusing adjacent frames from the same scene does not create independent layouts.

Minimum qualification set: five physically distinct layouts covering clear ground, multiple raised objects, visible depression, occluded edge/depression, and mixed/partly unobservable terrain. Add specific failure recordings as needed. A bowl sitting above the floor is not a depression fixture.

| ID | Case | Required pass behavior |
|---|---|---|
| P1-01 | Coordinate fixtures | Correct signs, transforms, metric scale, and documented error |
| P1-02 | Flat support | Stable support estimate; no invented support outside coverage |
| P1-03 | Wall/table dominant scene | Ground fit rejected or constrained correctly; wrong plane not accepted |
| P1-04 | Raised objects at multiple ranges | Correct elevated geometry within declared error budget |
| P1-05 | Visible depression | Negative evidence preserved against surrounding support |
| P1-06 | Floor-occluded edge/depression | Missing support remains unknown; edge shown when measurable |
| P1-07 | Flat ground with masked depth | Unknown support, never fabricated ground or automatic confirmed crater |
| P1-08 | Foreground-object occlusion | Object plus unobserved region behind it |
| P1-09 | Camera movement/session reset | Old accumulated grid invalidated; new calibration/session required |
| P1-10 | Sensor freeze/disconnection | Stale/error state; no output presented as current |
| P1-11 | Replay | Equivalent processing semantics and repeatable results within tolerance |
| P1-12 | Invalid config/calibration | Explicit diagnostic; dependent geometry/classification disabled |
| P1-13 | Processing load | Bounded queue, measured latency, exposed drops, no hidden backlog |
| P1-14 | Evidence aging/conflict | Stale support becomes unknown; absent returns do not erase confirmed hazards |

Measure:

- Point/elevation/object-position error versus range.
- Ground-fit residuals and rejection behavior.
- Raised and negative fixture detection rates, misses, and false positives.
- **False-ground coverage:** how much annotated negative/unknown-support area is incorrectly called observed ground.
- Observed-ground coverage on genuinely visible support, preventing a trivial all-unknown solution from passing.
- Grid update rate, queue delay, processing time, and source-to-output age where capture timestamps permit it.

Report sample/frame counts and scene counts; a thousand correlated frames are not a thousand independent trials. Set coverage/error criteria before running qualification and document any revisions. Required invariant: no invalid-depth-only cell becomes observed ground in the deterministic masking tests; no accepted ground model silently substitutes a wall/crater floor in the failure fixtures. Finite fixture success does not establish field reliability.

## 14. Tests appropriate to this phase

Write tests for failure-prone contracts, not screenshots that merely mirror implementation:

- Known-point projection and transform direction, including negative coordinates and unit errors.
- Grid indexing at boundaries and out-of-bounds rejection.
- Invalid depth, missing support, and evidence conflict handling.
- Wrong-plane rejection and explicit invalid-ground output.
- A synthetic depression and an occluded/missing patch remaining distinct.
- Hazard persistence versus absent returns and stale support aging.
- Live/replay timestamp semantics, source freeze, and queue bounds.
- At least one real recorded-session regression with independently measured fixtures.

Synthetic scenes verify logic; real recordings verify interaction with Kinect noise and coverage. Keep both. Use fixed random seeds where relevant and numerical tolerances with a reason. Test operation without a display or attached Kinect.

## 15. Ordered implementation tickets for coding agents

| Ticket | Dependencies | Deliverable / review gate |
|---|---|---|
| T01 — Repository reconnaissance | None | Document working entry point, stream formats, existing dependencies, and source assumptions; demonstrate baseline unchanged |
| T02 — Source contract and recorder/replay | T01 | Common FrameBundle source; lossless depth session; headless replay |
| T03 — Calibration/coordinates | T02 | Validated transforms/units, known-point tests, measured bench report |
| T04 — Depth health/coverage | T02–T03 | Validity masks, qualified-range filtering, stale handling, observable blind regions |
| T05 — Ground estimation | T03–T04 | Model plus rejection reasons; clear-floor and wrong-plane fixtures pass |
| T06 — Elevation/evidence grid | T04–T05 | Correct indexing, support coverage, explicit unknown, evidence aging |
| T07 — Raised geometry | T05–T06 | Object observations and measured proxy comparisons |
| T08 — Negative terrain/edges | T05–T06 | Depression, edge, dropout, and occlusion cases distinguished |
| T09 — Integrated display/export | T07–T08 | Single-source display with legend, metrics, quality, recorded exports |
| T10 — Qualification and handoff | T01–T09 | Unseen-layout results, failure inventory, run instructions, Phase 1 release report |

Each ticket must report what changed, how it was verified, known limitations, and any changed public data contract. Agents must not declare hardware-calibration or physical-fixture gates passed using synthetic tests alone. If hardware/recordings are unavailable, implement replayable logic, list the unverified physical gates, and leave qualification incomplete.

No agent may invent a rover footprint, implement driving, or broaden this work into ROS/simulation/navigation to bypass a blocked perception gate. Changes to a shared schema must update consumers, replay compatibility policy, and documentation together.

## 16. Required repository deliverables

- This specification in the project's documentation directory.
- Minimal working live, record, replay, and evaluate entry points, following repository conventions.
- Configuration templates plus a separately identified calibrated bench profile.
- Documented schemas, units, frames, depth formats, and clock behavior.
- Calibration report and measured fixture annotations.
- Small, authorized replay fixtures or a documented method to obtain local fixtures; keep large recordings out of normal Git history.
- Headless automated checks and real-recording regression instructions.
- Example visual outputs and a release report with metrics, failures, qualified conditions, and deferred items.
- A README explaining setup, recording, replay, evaluation, and what the outputs do and do not establish.

Commands must be documented only after they exist and are verified. Do not put fictional command names in the README as if runnable.

## 17. Human review and phase-completion checklist

- [ ] Working Kinect acquisition preserved and documented.
- [ ] A raw recording can be replayed without camera hardware.
- [ ] Coordinates and metric scale physically checked.
- [ ] Fixed bench transform and valid operating region documented.
- [ ] Ground estimation rejects unsupported/wrong-plane cases.
- [ ] Grid distinguishes elevation, support coverage, confidence, and unknown regions.
- [ ] Raised geometry localized and measured.
- [ ] Real depression/drop fixture evaluated.
- [ ] Missing/occluded support never treated as verified ground by default.
- [ ] Health/freshness and all significant failure states visible.
- [ ] Qualification includes useful observed-ground coverage as well as hazard detection.
- [ ] Thresholds, limitations, datasets, tests, and results reproducible.
- [ ] No rover control, route planning, ROS, simulation, or ML dependency added.

**First implementation task:** inspect and preserve the current acquisition code, then add a recording/replay adapter with explicit depth format, timestamps, and calibration metadata. Once that is reproducible, implement coordinate validation and ground segmentation.

**Phase 2 handoff:** measured, timestamped terrain evidence in a declared frame. Later work will add moving-frame transforms, localization, rover-specific traversability, planning, simulation/ROS, and physical rover integration next semester.
