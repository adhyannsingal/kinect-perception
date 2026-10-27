# Phase 1 software handoff — physical qualification incomplete

Date: 2026-10-27. Branch: `feature/phase1-terrain`.
Known-working starting commit: `78bd4744da584abe6481b606e67ac434a1e84d42`.
No merge into main or physical validation is claimed.

The Kinect was unavailable for this implementation. Synthetic results verify
software contracts and fixture logic; the project has **not** passed the Phase 1
physical-completion checklist. The calibrated bench profile and measured real
recording regression remain missing artifacts, deliberately not fabricated.

## Ticket status

| Ticket | Software result | Remaining physical gate |
|---|---|---|
| T01 | Existing RGB/DEPTH_MM adapter, geometry, demos, environment inspected; preserved | Recheck original live entry points after checkout |
| T02 | Versioned lossless sessions, ordered replay, clocks/calibration metadata, legacy/NPY import | Capture an actual fixed-mount session and inspect USB data loss |
| T03 | Rigid-transform validation, coordinate tests, target-error evaluation, unresolved template | Five measured targets, actual transform/intrinsics/range/error report |
| T04 | Organized validity, qualified-range masks, expected-footprint unknown reasons, stale/freeze handling | Measured blind zones and freshness limits |
| T05 | Constrained RANSAC, seed coverage, residual/temporal rejection | Kinect-noise robustness and visible support seed selection |
| T06 | Unknown-first elevation/evidence grid, area proxy, explicit stationary history/aging | Physical coverage/uncertainty thresholds |
| T07 | Raised evidence and connected visible extents | Measured objects at multiple ranges |
| T08 | Measured depression versus edge versus missing support | Real recessed/occluded fixtures and shallow limits |
| T09 | Headless outputs, OpenCV dashboard, optional Open3D cloud, health/export | Local interactive review and live load measurement |
| T10 | Synthetic evaluation, saved fixture, commands, failure inventory | Unseen physical layouts; release qualification remains incomplete |

## Verification

The original 20 tests passed before implementation in the existing `kinect`
environment. The full suite now includes 50 tests. Added tests cover transforms/units, config/session failure contracts,
boundary flooring, concentrated support, wrong planes, residual rejection,
depressions versus dropout, raised occlusion, temporal mount inconsistency,
stationary history, stale/frozen input, queue bounds/drop accounting, saved
replay equivalence, legacy import and read-only dashboard rendering.

See `phase1-synthetic-evaluation.json` for seven analytic layouts, actual
processing times and configuration hash. `phase1-example.png` is an offscreen
dashboard of the mixed synthetic fixture. Neither is a real Kinect test.
The saved development session is `fixtures/phase1/session`; it contains raw
uint16 depth and clearly synthetic annotations/calibration.

Run `python -m unittest discover -s tests -v` in the existing environment and
the evaluation command in [the run guide](PHASE1.md). Public live commands have
mocked-consumer tests and use the existing adapter; native behavior still needs
the device. Protected-file diffs are checked against the starting commit.

## Declared limitations and failure inventory

- Flat/gently varying reference surface only. One plane is not general uneven
  terrain understanding. No moving-camera accumulation or pose estimation.
- Seed region and mount must be measured; a bad mount can invalidate all geometry.
  A perfectly plausible wrong profile cannot be disproved from one depth image.
- Pinhole input only; no new distortion/registration calibration solver. Qualify
  the existing approximation in a restricted region, or supply rectified data.
- Coverage/subcell span and fit uncertainty are proxies. No probabilistic
  confidence calibration, hidden-surface inference or hole filling is claimed.
- Sparse/deep/wide depressions without surrounding visible support stay unknown.
  A hidden floor does not become confirmed negative terrain. Edge detection
  requires visible discontinuity geometry; dropout alone does not supply it.
- Connected touching geometry may merge; observed extents exclude unseen portions.
  No semantic boulder/crater identification or persistent tracking.
- Current and history are separate. Hazards clear only with reset/new session;
  automatic contrary-evidence clearing is intentionally not implemented.
- The live recorder saves processed/latest frames and reports skipped sequences.
  It cannot guarantee every sensor frame or interrupt a blocked native USB call.
- Python/JSON exports can dominate runtime. The 10 Hz bench target and 0.05 m
  physical grid are **unmeasured objectives**. Synthetic fixtures use 0.1 m cells
  at 320x240 to separate sampling limits from geometry logic.
- The synthetic layouts are development logic cases, not independent physical
  qualification layouts. No real-noise success rate is asserted.

No rover dimensions, traversability, controls, occupancy mapping, route planning,
SLAM, ROS, ML or new dependency/environment configuration was added.
