# Measured-target bench calibration workflow

This implements the next T03 step after working Kinect acquisition. It creates
a **new candidate** fixed-bench transform and an independent validation report.
It never edits device/intrinsic calibration, existing demos, or a supplied base
profile. Physical qualification still requires measured fixtures and review.

## 1. Fix the mount and define the reference

Close other Kinect viewers with Q before acquisition; the synchronous driver
must have one owner. Keep the Kinect fixed throughout the measurements. Define
the physical `test_ground` origin on the reference floor, x forward, y left, z up.
Write down how positions and measurement uncertainty are obtained independently
of the Kinect (for example a measured fixture with a tape and level). Record the
camera height/tilt as useful independent checks when possible.

Create a plan before acquisition:

```sh
conda activate kinect
python scripts/bench_calibration.py plan --mount-id bench-mount-01 \
  --output /tmp/bench-plan.json
```

The plan has **three fitting targets plus five separate validation targets**.
Using held-out targets prevents the fitted transform from evaluating itself.
More targets can be added with unique IDs. Spread fitting targets across two
directions (not along a single line). Spread validation targets forward,
laterally, and across known height differences in the intended working region.
Use a flat, stationary, diffuse target in a visible depth region. Identify the
same physical point in the depth image that you measured on the fixture.

Fill `origin_description`, `measurement_method`, each target's
`expected_test_ground_m`, and nonnegative `measurement_uncertainty_m`. Null
coordinates are deliberately rejected; do not substitute zeros for unknowns.
Initial declared error budgets are 0.025 m median and 0.05 m p95 for the proposed
0.05 m bench grid. Criteria are explicit and saved with the report. Document
any budget revisions rather than adjusting them silently after a failed check.

## 2. Observe each target on the depth grid

For each identified physical target, click its location in the depth image:

```sh
python scripts/bench_calibration.py observe --live --pick --frames 30 \
  --mount-id bench-mount-01 --target-id fit-1 --output /tmp/fit-1.json
```

Repeat for `fit-2`, `fit-3`, and `validation-1` through `validation-5`, moving the
target rather than the camera. Alternatively use explicit `--pixel U V` on the
**depth grid**. Do not pick the same pixel in unregistered RGB and assume it
corresponds. Invalid depth is shown magenta. Q/Escape cancels selection.

Set each plan target's `observation_path` to its observation JSON. Relative paths
resolve relative to the plan. The collector saves resolution, intrinsics, source
IDs, depth-frame hashes, pixel/ROI, clock label, validity, median metric depth,
and p95-minus-p5 spread. It projects the median at the selected pixel using the
existing pinhole utility. Live input uses the existing bounded camera adapter;
device freeze, timeouts and resolution changes are explicit failures.

Default sampling policy: one pixel, >=80% valid returns, and <=0.02 m p95-p5
spread. These are conservative sample-admission settings, **not a calibrated
Kinect noise model**. Noisy or mixed-depth targets are rejected, with diagnostics
saved and exit code 2. Input/format errors return exit code 1. The collector
does not increase limits or replace rejected data to manufacture a good sample.
`--radius`, `--minimum-valid-fraction`, and `--maximum-spread` are explicit
settings, saved with every observation. A nonzero ROI can mix nearby surfaces.

The default live intrinsics are the existing approximate 640x480 Kinect values;
they are not newly calibrated. Supply `--intrinsics FX FY CX CY` if independently
calibrated values are available. Device intrinsics are never rewritten.

Saved sources work without hardware:

```sh
python scripts/bench_calibration.py observe --recording /tmp/kinect-capture \
  --pixel 320 240 --frames 5 --mount-id bench-mount-01 \
  --target-id fit-1 --output /tmp/fit-1-recorded.json
python scripts/bench_calibration.py observe --session /tmp/phase1-session \
  --pick --frames 30 --mount-id bench-mount-01 \
  --target-id fit-1 --output /tmp/fit-1-session.json
```

The first path is the original `RecordingWriter` format. Registered/RGB-grid
depth is rejected. Phase 1 sessions use their recorded intrinsics and enforce
the recorded mount ID when present. `--npy FILE FILE FILE ...` is also supported,
but requires explicit intrinsics and labels assigned frame-index times; it does
not claim measured capture timing. Legacy clock wrap/frozen timestamps are
rejected. Use a new session after mount changes or clock discontinuities.

## 3. Fit and independently review

Create/copy an unresolved terrain profile, then fit to **fitting targets only**:

```sh
python scripts/terrain_perception.py config-template --output /tmp/base-bench.json
python scripts/bench_calibration.py fit --plan /tmp/bench-plan.json \
  --config /tmp/base-bench.json --output /tmp/bench-candidate
```

The new directory contains `candidate_profile.json`, `validation_report.json`,
and a measurement snapshot. Existing output paths are refused. The base profile
is untouched. Open3D point-to-point registration uses **scaling disabled**.
Duplicate/collinear targets, inconsistent units/distances, changed mount/stream,
and reused observations are rejected. Scale disagreements must be investigated,
not hidden by fitting a scaling factor.

The validation report includes all held-out positional/axis errors, optical-Z
sample ranges, uncertainty, coordinate spans and median/p95 budgets. Validation
coordinates never enter the fitted transform. Success creating files means only
that the workflow ran; inspect report booleans, not just process exit status.
Synthetic sources stay labeled synthetic and cannot be mixed into a physical
mount fit. Physical candidates remain `bench_unqualified`, even for zero error.

Remaining unresolved terrain fields (qualified range, measurement uncertainty,
grid bounds and credible support seed) stay null. A candidate mount transform
does not resolve them automatically. Review those fields and physical gates in
[the calibration handoff](PHASE1_CALIBRATION.md) before terrain qualification.

Implementation reference: [Open3D rigid point-to-point estimation](https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.TransformationEstimationPointToPoint.html).
