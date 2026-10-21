"""Offline ground segmentation from metric depth NPY files or synthetic depth."""

import argparse
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinect.geometry import CameraIntrinsics, DEFAULT_KINECT_V1_DEPTH_INTRINSICS
from kinect.ground import segment_ground
from kinect.ground_simulation import synthetic_ground_frame
from kinect.pointcloud import depth_mm_to_point_cloud
from kinect.recording import ReplayCamera


def summary(result) -> str:
    total = len(result.cloud.points)
    if result.plane is None:
        plane = "No ground plane detected"
    else:
        a, b, c, d = result.plane
        plane = f"Plane: {a:.4f}x {b:+.4f}y {c:+.4f}z {d:+.4f} = 0"
    return (f"{plane} | ground={result.ground_count:,} non-ground={total - result.ground_count:,} "
            f"total={total:,} inliers={result.inlier_percentage:.1f}% "
            f"fit-points={result.fit_point_count:,}")


def show_result(result, text: str) -> None:
    if not result.cloud.has_points():
        return
    # The window title and terminal both display the equation and counts.
    # Keep the same upright camera convention as Demo 5 without moving points.
    o3d.visualization.draw_geometries(
        [result.colored_cloud()], window_name=text, up=[0, -1, 0],
        front=[0, 0, -1], lookat=result.cloud.get_center(), zoom=0.7,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("depth_files", nargs="*", type=Path, help="2D metric millimetre .npy files")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true", help="use known floor/wall/box depth")
    source.add_argument("--recording", type=Path, help="NPZ recording directory from camera_stream.py")
    parser.add_argument("--headless", action="store_true", help="print results without opening windows")
    parser.add_argument("--distance-threshold", type=float, default=0.02, metavar="METRES")
    parser.add_argument("--voxel-size", type=float, default=0.03, metavar="METRES", help="0 disables downsampling")
    parser.add_argument("--expected-normal", nargs=3, type=float, default=(0, 1, 0), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--max-tilt", type=float, default=15, metavar="DEGREES")
    parser.add_argument("--height-range", nargs=2, type=float, default=(0.1, 3.0), metavar=("MIN", "MAX"))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--min-inliers", type=int, default=100, help="minimum fitting-cloud support")
    parser.add_argument("--min-inlier-ratio", type=float, default=0.1, help="minimum fitting-cloud support fraction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--intrinsics", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"),
                        help="depth-grid intrinsics override; NPY defaults to existing Kinect values")
    args = parser.parse_args(argv)
    if bool(args.depth_files) + args.synthetic + bool(args.recording) != 1:
        parser.error("choose NPY depth files, --synthetic, or --recording")
    override = CameraIntrinsics(*args.intrinsics) if args.intrinsics else None
    if override and (not np.isfinite(args.intrinsics).all() or min(override.fx, override.fy) <= 0):
        parser.error("intrinsics must be finite with positive focal lengths")

    def process(label, depth, intrinsics):
        if depth.ndim != 2 or depth.dtype.kind not in "uif":
            raise ValueError("depth must be a two-dimensional numeric millimetre array")
        cloud = depth_mm_to_point_cloud(depth, intrinsics)
        result = segment_ground(
            cloud, distance_threshold_m=args.distance_threshold, voxel_size_m=args.voxel_size,
            expected_normal=args.expected_normal, max_tilt_deg=args.max_tilt,
            height_range_m=args.height_range, iterations=args.iterations,
            min_inliers=args.min_inliers, min_inlier_ratio=args.min_inlier_ratio, seed=args.seed,
        )
        text = summary(result)
        print(f"{label}: {text}")
        if not args.headless:
            show_result(result, text)

    try:
        if args.synthetic:
            frame = synthetic_ground_frame()
            process("synthetic", frame.depth_mm, override or frame.intrinsics)
        elif args.recording:
            with ReplayCamera(args.recording) as camera:
                count = 0
                while (frame := camera.read()) is not None:
                    intrinsics = override or frame.intrinsics
                    if intrinsics is None:
                        raise ValueError("recording lacks depth intrinsics; supply --intrinsics")
                    process(f"frame {count}", frame.depth_mm, intrinsics)
                    count += 1
                if count == 0:
                    print("Recording has no completed frames.")
        else:
            for path in args.depth_files:
                with path.open("rb") as stream:
                    depth = np.load(stream, allow_pickle=False)
                    if not isinstance(depth, np.ndarray):
                        depth.close()
                        raise ValueError("expected an NPY depth array, not an NPZ archive")
                process(str(path), depth, override or DEFAULT_KINECT_V1_DEPTH_INTRINSICS)
    except (ValueError, OSError, EOFError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
