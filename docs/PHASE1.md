# Phase 1 stationary terrain perception

The application implements the offline and live software path described in
[the build spec](specs/PHASE_1_KINECT_PERCEPTION_SPEC.md). It reports geometry
and evidence in `test_ground`, not rover traversability. Physical qualification
is incomplete. See [the release report](PHASE1_RELEASE_REPORT.md) and
[calibration handoff](PHASE1_CALIBRATION.md).

## Run now, without a Kinect

Use the existing environment; no packages or configuration changes are needed:

```sh
conda activate kinect
python -m unittest discover -s tests -v
python scripts/terrain_perception.py evaluate --output /tmp/phase1-evaluation.json
python scripts/terrain_perception.py replay \
  --session fixtures/phase1/session --output /tmp/phase1-replay
```

Every output/session directory must be new. Individual evaluation reports and
configuration templates also refuse to overwrite an existing path.
Replay is headless by default. Inspect the same processing outputs with:

```sh
python scripts/terrain_perception.py replay --session fixtures/phase1/session \
  --output /tmp/phase1-view --display --cloud --step
```

Press Enter in the terminal for each recorded frame. `e` toggles the evidence
and elevation views; `q` or Escape exits. The cloud uses ground coordinates:
x forward, y left, z up. RGB is display-only and **unregistered**. The grid's
vertical screen direction is positive y, with bounds printed in metres.
Gray is UNKNOWN, green OBSERVED_GROUND, red RAISED, blue NEGATIVE, magenta EDGE,
and orange MIXED. These labels make no motion/safety claim.

Other playback choices:

```sh
python scripts/terrain_perception.py replay --session fixtures/phase1/session \
  --output /tmp/phase1-realtime --speed 1
python scripts/terrain_perception.py replay --session fixtures/phase1/session \
  --output /tmp/phase1-fast --speed 4 --accumulate
python scripts/terrain_perception.py fixtures --layout all --output /tmp/phase1-generated
```

`--speed 0` (default) is unpaced; positive values scale recorded delays.
Algorithmic age and evidence aging always use **recorded** time, independently
of playback speed. `--step` overrides delay pacing. The final `health.json`
says STALE because replay has ended; individual frame outputs preserve their
recorded-time health. Synthetic/unqualified profiles remain DEGRADED even
when the algorithm accepts a ground model.

## Existing recordings

The existing `RecordingWriter` / `ReplayCamera` format remains unchanged. The
Phase 1 session includes stronger mount, format, clock and configuration
metadata. Import metric uint16 NPY files or original unregistered recordings:

```sh
python scripts/terrain_perception.py import-npy \
  --input sample_data/scene1_depth_mm.npy \
  --config /tmp/my-bench.json --output /tmp/phase1-import
python scripts/terrain_perception.py import-legacy \
  --input /tmp/original-capture --config /tmp/my-bench.json \
  --output /tmp/phase1-legacy
```

`/tmp/my-bench.json` must describe the actual original stream and mount.
The NPY importer assigns playback timestamps at `--fps` (default 10), labels
them as assigned rather than measured capture times, and preserves depth
values exactly. Legacy device times remain in their own domain; there is no
invented host mapping. A device-time wrap requires a separate imported session.
Registered/RGB-grid legacy depth is rejected, since its intrinsics cannot be
silently relabeled as depth-optical calibration. Imported streams with a
resolution/profile mismatch produce CALIBRATION_REQUIRED when processed.

## Recording when the Kinect is available

Create an unresolved bench profile now:

```sh
python scripts/terrain_perception.py config-template --output /tmp/my-bench.json
```

Follow the calibration handoff to fill its null fields. Raw capture is allowed
with an unresolved profile; dependent geometry outputs remain unavailable.
A synthetic profile is explicitly rejected for physical device capture.

```sh
python scripts/terrain_perception.py record --config /tmp/my-bench.json \
  --session /tmp/bench-capture --output /tmp/bench-record-status --frames 100
python scripts/terrain_perception.py live --config /tmp/my-bench.json \
  --session /tmp/bench-live-capture --output /tmp/bench-live-results \
  --frames 100 --display --cloud
```

Both commands use the unchanged `KinectCamera` (libfreenect VIDEO_RGB/DEPTH_MM).
`record` includes status/processing where calibration permits it; `live` can
also record with `--session`. `--no-rgb` omits RGB from saved bundles and views.
The worker owns the adapter and closes it in `finally`. Capture and process
queues are bounded to **one latest frame**. Slow processing/recording discards
older pending frames and exposes sequence gaps and total drops in the index
and health. This is not an archival every-frame recorder. Hardware/USB loss
before the adapter delivers a frame cannot be counted by the Python queue.

The processing-rate cap is `target_grid_hz`, initially 10. Actual processing
time, publication interval, source/queue age and dropped frames are exported.
Live receive time is a monotonic **capture-time proxy**; transport delay and
true capture-to-output latency are unmeasured. Repeating device timestamps
trigger STALE. A blocked native read triggers watchdog timeouts; ten consecutive
timeouts (configurable `--max-timeouts`) terminate with ERROR. Python cannot
safely interrupt a USB call blocked inside the native driver; the daemon worker
cleans up when that call returns. This behavior needs hardware testing.

## Processing and uncertainty

1. Validate schema, depth units, session, clock, resolution and calibration IDs.
   DEPTH_MM is uint16 optical-Z millimetres; 0 and values above 10000 are invalid.
   Apply the declared qualified range while retaining the organized validity mask.
2. Reuse the existing pinhole geometry and transform to `test_ground`. Exports
   retain each projected point's `(v,u)` correspondence. Unsupported distortion
   models are rejected: supply rectified pinhole data, or explicitly qualify a
   pinhole approximation over a limited measured region. No implicit second
   undistortion or RGB registration is performed.
3. Fit only actual visible returns in the configured measured support-seed region.
   Reuse the Demo 6 RANSAC utility, constraining orientation and reference-plane
   height. Validate support across a 4x4 partition of the seed, p95 residual,
   and temporal angle/offset. A concentrated patch or a wrong-height tabletop/
   depression bottom cannot replace the reference. Temporal inconsistency
   latches until a new session; it is a motion warning, not pose estimation.
4. Preserve original points for evidence. Voxelization affects fitting only.
   Failed ground estimation preserves measured min/max/return elevation but
   disables support and ground-relative features. A point's signed residual is
   `n dot p + d` with an upward unit normal. This differs from ground-frame z.
5. Floor XY coordinates into lower-inclusive, upper-exclusive grid bounds.
   Support coverage counts occupied subcells (2x2 by default), separately from
   pixel count. Minimum span in **both** horizontal axes rejects narrow/vertical
   return strips. Coverage is a sampled-area proxy, not proof of full-cell visibility.
6. `elevation_estimate_m` uses near-reference candidate support returns, or a
   confirmed visible depression surface; box tops cannot define support.
   `return_elevation_m` is the median of all measured returns. Min/max retain
   all measured extents. Sparse candidate estimates are not observed support.
   Local slope/roughness requires >=6 support returns with two constrained
   spatial directions. Missing values remain NaN internally / null in JSON.
7. Raised evidence requires height above threshold **plus** uncertainty. Visible
   depression candidates require measured lower returns and a connected reference
   support component in >=3 surrounding quadrants within the context radius.
   Edge evidence requires a measured discontinuity anchored in credible support
   or a measured below-support face. Missing returns alone never create an edge
   or confirmed depression. Connected-cell observations retain measured visible
   extents; touching objects may be one merged observation.
8. Current-frame grids are independent. Optional history is separately labeled.
   Historical support ages to unknown; hazards persist until explicit reset/new
   session. Missing returns never clear hazards. No global mapping across motion.

The fit uncertainty field is a **documented proxy**, the larger of configured
measurement uncertainty and p95 support residual. Feature margins add the
measurement uncertainty to this proxy. It is not a calibrated probability or
covariance. The full effective configuration and hash are saved with every run
and each output, including all user-tunable thresholds and units.

## Sessions, schemas and outputs

Sessions use schema version 1, with manifest, config, calibration, lossless NPZ
frames, an ordered JSONL index with timestamps/gaps/checksums, annotations, outputs,
and evaluation folders. Complete frames are renamed before indexing. Incomplete
unindexed frames are ignored; a corrupt index or checksum raises an error rather
than silently skipping data. No power-loss durability guarantee is made.
Pickle is disabled. Raw bit depth and invalid sentinels survive round trips.

Frame metadata identifies `depth_optical`, conversion `uint16_mm_times_0.001`,
session/sequence, original clock domain, optional device time, proxy status,
calibration and mount IDs. Intrinsics/distortion/stream mode and transform live
in session config/calibration artifacts. RGB pairing error and device-to-host
mapping remain null when unmeasured. RGB/depth frames are not asserted synchronized.

Each output JSON contains source/publication metadata, effective config/hash,
health, ground model/rejection reason/support indices, grid fields, observations,
ground-aligned cloud, pixel correspondence, signed residuals and original-sized
qualified-depth validity. The grid includes support/raised/negative/edge flags
independently, counts, support/return coverage, elevation/residual statistics,
uncertainty proxy, slope/roughness validity, and support/hazard timestamps.
The optional history appears under metadata, never in place of the current grid.
Observation IDs are per output frame; no persistent tracking is claimed.

Unknown reasons are OUT_OF_VIEW, INVALID_DEPTH, OCCLUDED_OR_UNOBSERVED,
SPARSE_SUPPORT, STALE and GROUND_MODEL_INVALID. Expected-reference projection
only diagnoses why support may be missing; it never manufactures a surface.
Occlusion versus unobserved geometry cannot always be distinguished from one
depth view, hence the combined reason. Output directories should remain outside
normal Git history, except the small explicitly synthetic development fixture.

Consumers must check schema version and coordinate/clock fields, ignore new
optional fields, and reject unsupported required-field changes. Phase 1 readers
do not implicitly reinterpret legacy recordings; use the explicit importer.

Health transitions: INITIALIZING on opening a source; CALIBRATION_REQUIRED for
missing/mismatched profile; RUNNING only for an accepted model with a physically
qualified profile; DEGRADED for tentative calibration or rejected support; STALE
for old/frozen/ended input; ERROR for capture/storage/ordering failure. Historical
panels are dimmed and marked STALE/HISTORY. Sequence/clock faults and temporal
mount inconsistency require a new session. A feature-poor frame is not hidden.
