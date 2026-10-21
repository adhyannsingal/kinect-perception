"""View, record, or replay camera frames without changing the original demos."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys

# Allow direct invocation from the repository root without installing a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinect.camera import KinectCamera
from kinect.recording import RecordingWriter, ReplayCamera
from kinect.simulation import SimulatedCamera


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("simulate", "kinect", "replay"), default="simulate")
    parser.add_argument("--input", type=Path, help="recording directory for replay")
    parser.add_argument("--record", type=Path, help="new output directory (must not exist)")
    parser.add_argument("--frames", type=positive_int, default=300, help="maximum frames to read")
    parser.add_argument("--display", action="store_true", help="open RGB/depth windows; q or Escape exits")
    args = parser.parse_args()
    if args.source == "replay" and args.input is None:
        parser.error("--input is required for replay")
    if args.source != "replay" and args.input is not None:
        parser.error("--input applies only to replay")

    count = 0
    with ExitStack() as stack:
        if args.source == "simulate":
            camera = SimulatedCamera(frame_count=args.frames)
        elif args.source == "replay":
            camera = ReplayCamera(args.input)
        else:
            camera = KinectCamera()
        stack.enter_context(camera)
        writer = stack.enter_context(RecordingWriter(args.record)) if args.record else None
        if args.display:
            import cv2
            stack.callback(cv2.destroyAllWindows)
        for _ in range(args.frames):
            frame = camera.read()
            if frame is None:
                break
            if writer:
                writer.write(frame)
            count += 1
            if args.display:
                cv2.imshow("Camera RGB", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
                # Display-only scaling; recorded depth remains exact uint16 mm.
                preview = cv2.convertScaleAbs(frame.depth_mm, alpha=255 / 10000)
                cv2.imshow("Camera depth", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    print(f"Processed {count} frames from {args.source}")


if __name__ == "__main__":
    main()
