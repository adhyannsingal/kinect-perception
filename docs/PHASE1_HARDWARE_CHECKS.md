# Kinect hardware checks and next calibration gate

Date: 2026-10-28. Starting software: `1b62195`.

The connected Kinect initially failed native initialization with a USB
control-transfer error and a subsequent timeout. After the owner reconnected it,
the following checks passed using the existing `kinect` environment:

- 30 live RGB/depth pairs, with 30 distinct timestamps for each stream.
- RGB uint8 640x480x3 and DEPTH_MM uint16 640x480.
- Last-frame valid metric-depth fraction: 68.94%.
- Depth p5/median/p95: 2.114 / 5.125 / 6.235 m. These are sensor outputs, not
  independently measured distances and therefore do not verify metric accuracy.
- A 0.02 m voxel cloud contained 60,060 finite points.
- Five saved RGB/depth pairs replayed bit-for-bit through the existing recorder.
- Ten native DEPTH_REGISTERED frames with advancing timestamps and 55.3% valid
  depth. This verifies mode capture, not visual RGB/depth alignment accuracy.
- The original Demo 5 opened and continuously updated its point cloud. Its first
  reported cloud had 58,506 points. An initial USB warning recovered.

The 30-pair acquisition/recording startup run took 10.83 s including initialization
and saving. It is not an isolated steady-state frame-rate benchmark. Real camera
frames and raw diagnostic records remain outside Git in the local temporary
test directory; no room images are included in this repository.

## T03 sampling follow-up

The new collector was run against the five archived real frames with its
unchanged default policy (>=80% valid, <=0.02 m p95-p5 spread):

- Pixel (320,240) had no valid measurements: INSUFFICIENT_VALID_DEPTH.
- The nearest pixel valid in all five frames, (317,231), had 0.1496 m spread:
  UNSTABLE_OR_MIXED_SURFACES. Selection used validity only, not low measured noise.

These are useful rejection checks, **not physical target measurements**. The
camera scene may contain boundaries, motion or other unsuitable target geometry;
the spread does not identify the cause or quantify sensor noise in isolation.
The limits were not relaxed after failure.

The next required measurements are a fixed physical reference origin/mount,
three independent fitting targets, and at least five held-out validation targets
at known forward/lateral positions and heights, with measurement uncertainty.
No independently measured fixture coordinates have been supplied yet, so no
physical bench transform, qualified range, or accuracy pass is claimed.

The measurement workflow is implemented and synthetic known-transform tests
pass, including held-out separation and failure cases. Follow
[the workflow](BENCH_CALIBRATION_WORKFLOW.md) to collect the missing physical
measurements. Existing demos, camera calibration and environment remain unchanged.
