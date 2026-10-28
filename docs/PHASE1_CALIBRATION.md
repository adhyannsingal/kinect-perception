# Physical calibration and fixture handoff — incomplete

No calibrated physical bench profile is supplied. `configs/phase1_bench_template.json`
is intentionally unresolved. `configs/phase1_synthetic.json` describes an analytic
camera and must not be used on the Kinect. The approximate existing Kinect
intrinsics and all original demos remain unchanged.

An assisted [target collection and candidate-transform workflow](BENCH_CALIBRATION_WORKFLOW.md)
is now available. It uses separate fitting and validation targets and writes a
new unqualified profile. See [the actual hardware checks](PHASE1_HARDWARE_CHECKS.md)
for capture results; those checks do not establish metric accuracy.

## Fill the bench profile

Record these measurements before interpreting terrain:

- Device/model/stream resolution and DEPTH_MM representation. Verify optical Z
  units with targets at known distances; never import raw disparity as millimetres.
- Depth intrinsics matching that resolution and mode. Use `pinhole_approximation`
  only with a measured restricted-region error budget, or `rectified_pinhole`
  for already rectified input. Arbitrary distortion correction is unsupported.
- Unique calibration/mount IDs; `calibration_kind=bench_unqualified` initially.
- Physical reference plane and origin definition; a measured, rigid
  `T_test_ground_from_depth_optical`. Record the measuring method/uncertainty.
- Qualified optical-Z range, ground XY/Z storage bounds, visible seed-region
  bounds, and a measured uncertainty margin. Start grid cells at 0.05 m; revise
  only with a documented feature-resolution/error-budget reason.

For a level mount, depth optical X right/Y down/Z forward maps to ground x
forward/y left/z up with rotation `[[0,0,1],[-1,0,0],[0,-1,0]]`. Translation
is the camera position in the measured ground frame. At a projected-camera
origin it is `[0,0,height_m]`. For measured downward pitch theta, left-multiply
that rotation by `[[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]]`.
These are conventions, **not measurements of your mount**. Verify signs on
physical targets and record actual orientation rather than assuming level.

Set `measurement_method` to include the origin, fixture method, mount height/
orientation and measurement uncertainty. Set thresholds using flat-surface
residuals at several ranges and angles, then compare with the smallest intended
feature. Keep the camera fixed for the entire session. Repositioning requires
a new mount transform/ID and session; do not retain accumulated terrain.

## Target measurement file

Create JSON with a `targets` array of at least five independent measured points.
Each entry requires three finite metre values in `observed_depth_optical_m`,
three in `expected_test_ground_m`, and nonnegative `measurement_uncertainty_m`.
Use targets at several depths, lateral positions, and known height differences.
Do not derive the expected coordinates from the same depth image or transform
being evaluated. Include uncertainty of the physical measuring fixture.

After creating that file and completing the profile:

```sh
python scripts/terrain_perception.py calibration-check --config /tmp/my-bench.json \
  --measurements /tmp/measured-targets.json --output /tmp/calibration-review.json
```

The report contains per-target/range/axis errors, median/p95 positional error,
and the proposed half-cell/one-cell budget. It never changes the profile's
qualification status. Review the physical measurement method, signs/axes,
range behavior and blind zones, not just a boolean. Keep failing reports and
document any change to the range/grid/error budget before rerunning qualification.
Set `calibration_kind=qualified` only after human review of measured evidence.

## Independent scene qualification

Keep development/tuning and final qualification sessions separate. Capture at
least five physically distinct layouts: clear ground, multiple raised objects,
a recessed fixture with a visible bottom, an edge/depression with hidden floor,
and mixed/partly unobservable terrain. Add flat ground with masked depth,
wall/table-dominant scenes, shallow features near the detection limit, a frozen/
disconnected source, and processing load tests. A bowl above the floor is not
a depression fixture. Adjacent frames are not independent layouts.

Record fixture extents, camera mount, range, lighting/surface, measurement
uncertainty, and feature-detection/coverage/error criteria **before** final
qualification. Place annotations in `annotations/fixtures.json`: an array of
`{sequence_id, layout, targets}`. Each target has `type` (RAISED_GEOMETRY,
VISIBLE_DEPRESSION, MEASURED_EDGE, or UNKNOWN), `bounds_xy_m` ordered
xmin,xmax,ymin,ymax, and optionally measured `height_m` or `depth_m`.
Annotations feed evaluation only, never the detector.

Replay produces `evaluation.json` with extent matches and false-ground counts
inside annotated depression/unknown regions. Review partial observed extent,
misses, false positives, useful visible-ground coverage and range-dependent
height/position errors. Bounding-box overlap alone is only a coarse detection
metric; it does not establish measurement accuracy. The synthetic report cannot
replace these measurements. Keep large/local Kinect recordings outside Git.

Pending physical gates: five target checks; fixed mount/range qualification;
five independent scene evaluations; real-recorded regression; live USB freeze/
disconnect behavior; actual queue/load/latency measurements; local GUI review.
