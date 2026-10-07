"""Save one live Kinect RGB/depth frame pair for offline development."""

import argparse
from pathlib import Path

import cv2
import freenect
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("sample_data"))
    parser.add_argument("--prefix", default="kinect_frame")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        rgb_frame = freenect.sync_get_video(format=freenect.VIDEO_RGB)
        depth_frame = freenect.sync_get_depth(format=freenect.DEPTH_MM)
        if rgb_frame is None or depth_frame is None:
            raise RuntimeError("Could not retrieve Kinect RGB/depth frames. Is the device connected?")
        rgb, _rgb_timestamp = rgb_frame
        depth_mm, _depth_timestamp = depth_frame
        rgb_path = args.output_dir / f"{args.prefix}_rgb.png"
        depth_path = args.output_dir / f"{args.prefix}_depth_mm.npy"
        if not cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write {rgb_path}")
        np.save(depth_path, depth_mm)
        print(f"Saved RGB PNG: {rgb_path}")
        print(f"Saved raw uint16 depth NPY: {depth_path}")
    finally:
        freenect.sync_stop()


if __name__ == "__main__":
    main()
