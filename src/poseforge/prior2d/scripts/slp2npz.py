#!/usr/bin/env python
"""Extract the highest-confidence pose per frame from a SLEAP predictions file.

For each frame, if multiple instances were detected, the instance with the
highest overall prediction score is kept (assumes one real animal per frame).
Optionally, also renders an annotated overview video with the skeleton drawn
over the source video (drawing done with OpenCV only, no matplotlib).

Usage:
    python extract_poses.py predictions.slp poses.npz
    python extract_poses.py predictions.slp poses.npz --video input_video.mkv
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import sleap

# --- Skeleton definition -----------------------------------------------
# Legs are connected as: Th -> *_ThC -> *_CTr -> *_FTi -> *_TiTa -> *_Cl
# Everything else (Th itself aside) is drawn as an unconnected dot.
LEG_PREFIXES = ["LF", "LM", "LH", "RF", "RM", "RH"]
LEG_JOINTS = ["ThC", "CTr", "FTi", "TiTa", "Cl"]

# BGR colors, one per leg chain.
LEG_COLORS = {
    "LF": (255, 80, 80),
    "LM": (80, 220, 80),
    "LH": (60, 60, 230),
    "RF": (230, 220, 60),
    "RM": (220, 80, 220),
    "RH": (60, 200, 230),
}
HUB_COLOR = (255, 255, 255)    # Th (thorax hub)
OTHER_COLOR = (180, 180, 180)  # N, A, LA, RA, LW, RW

POINT_RADIUS = 5
LINE_THICKNESS = 2


def build_skeleton(node_names):
    """Build (edges, point_colors) for the fly leg convention.

    Args:
        node_names: List of node names, in the same order as the pose array's
            node axis.

    Returns:
        edges: List of (idx_a, idx_b, color) tuples.
        point_colors: List of BGR colors, one per node, in node_names order.
    """
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    edges = []
    for leg in LEG_PREFIXES:
        chain = ["Th"] + [f"{leg}_{joint}" for joint in LEG_JOINTS]
        color = LEG_COLORS[leg]
        for a, b in zip(chain[:-1], chain[1:]):
            if a in name_to_idx and b in name_to_idx:
                edges.append((name_to_idx[a], name_to_idx[b], color))

    leg_node_names = {f"{leg}_{joint}" for leg in LEG_PREFIXES for joint in LEG_JOINTS}
    point_colors = []
    for name in node_names:
        if name == "Th":
            point_colors.append(HUB_COLOR)
        elif name in leg_node_names:
            point_colors.append(LEG_COLORS[name.split("_")[0]])
        else:
            point_colors.append(OTHER_COLOR)

    return edges, point_colors


def extract_poses(slp_path: Path) -> dict:
    """Load a .slp file and extract the best-scoring instance per frame.

    Args:
        slp_path: Path to a SLEAP predictions (.slp) file.

    Returns:
        Dict with keys "poses", "instance_scores", "keypoint_scores",
        "node_names", ready to be passed to np.savez_compressed.
    """
    labels = sleap.load_file(str(slp_path))

    video = labels.video
    n_frames = video.num_frames
    node_names = labels.skeleton.node_names
    n_nodes = len(node_names)

    poses = np.full((n_frames, n_nodes, 2), np.nan, dtype=np.float32)
    instance_scores = np.full(n_frames, np.nan, dtype=np.float32)
    keypoint_scores = np.full((n_frames, n_nodes), np.nan, dtype=np.float32)

    for lf in labels.labeled_frames:
        if len(lf.instances) == 0:
            continue
        best = max(lf.instances, key=lambda inst: inst.score)
        pts_scores = best.points_and_scores_array  # (n_nodes, 3): x, y, score
        poses[lf.frame_idx] = pts_scores[:, :2]
        keypoint_scores[lf.frame_idx] = pts_scores[:, 2]
        instance_scores[lf.frame_idx] = best.score

    return {
        "poses": poses,
        "instance_scores": instance_scores,
        "keypoint_scores": keypoint_scores,
        "node_names": np.array(node_names),
    }


def draw_pose(frame, points, edges, point_colors):
    """Draw skeleton edges and keypoint dots onto frame, in place."""
    for a, b, color in edges:
        pa, pb = points[a], points[b]
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            continue
        cv2.line(
            frame,
            (int(round(pa[0])), int(round(pa[1]))),
            (int(round(pb[0])), int(round(pb[1]))),
            color,
            LINE_THICKNESS,
            cv2.LINE_AA,
        )
    for idx, pt in enumerate(points):
        if np.any(np.isnan(pt)):
            continue
        cv2.circle(
            frame,
            (int(round(pt[0])), int(round(pt[1]))),
            POINT_RADIUS,
            point_colors[idx],
            -1,
            cv2.LINE_AA,
        )
    return frame


def resize_pad_and_scale_points(frame, points, target_height):
    """Resize frame to target_height (aspect-preserving), pad to multiple of
    16 on each axis, and return the transformed frame plus rescaled points.
    """
    h, w = frame.shape[:2]
    scale = target_height / h
    new_w = int(round(w * scale))
    resized = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)

    pad_w = (-new_w) % 16
    pad_h = (-target_height) % 16
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    if pad_w or pad_h:
        resized = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    scaled_points = points * scale
    scaled_points[:, 0] += left
    scaled_points[:, 1] += top
    return resized, scaled_points


def make_overview_video(
    video_path: Path,
    poses: np.ndarray,
    edges,
    point_colors,
    output_path: Path,
    crf: int = 23,
    target_height: int = 1024,
):
    """Render an annotated overview video with the skeleton drawn over each frame."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH; load/install ffmpeg to render the overview video.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = target_height / orig_h
    scaled_w = int(round(orig_w * scale))
    out_w = scaled_w + ((-scaled_w) % 16)
    out_h = target_height + ((-target_height) % 16)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", str(fps),
        "-i", "-",
        "-an", "-vcodec", "libx264", "-crf", str(crf),
        "-pix_fmt", "yuv420p", str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    n_frames = poses.shape[0]
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pts = poses[frame_idx] if frame_idx < n_frames else np.full((poses.shape[1], 2), np.nan)
            out_frame, scaled_pts = resize_pad_and_scale_points(frame, pts.copy(), target_height)
            draw_pose(out_frame, scaled_pts, edges, point_colors)
            proc.stdin.write(out_frame.tobytes())
            frame_idx += 1
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the highest-confidence pose per frame from a SLEAP "
            "predictions (.slp) file into a compressed .npz file, optionally "
            "rendering an annotated overview video."
        )
    )
    parser.add_argument("input_slp", type=Path, help="Path to the input SLEAP predictions file (.slp).")
    parser.add_argument("output_npz", type=Path, help="Path to write the output compressed .npz file.")
    parser.add_argument(
        "--video", type=Path, default=None,
        help="Path to the source video. If given, an annotated overview video is also rendered.",
    )
    parser.add_argument(
        "--overview-video", type=Path, default=None,
        help="Output path for the overview video. Defaults to '<output_npz stem>_overview.mp4'.",
    )
    parser.add_argument("--crf", type=int, default=23, help="x264 CRF for the overview video (default: 23).")
    parser.add_argument(
        "--height", type=int, default=1024,
        help="Target height in pixels for the overview video; width is scaled to preserve aspect "
             "ratio and padded with black to the next multiple of 16 (default: 1024).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_slp.is_file():
        raise SystemExit(f"Input file does not exist: {args.input_slp}")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)

    data = extract_poses(args.input_slp)
    np.savez_compressed(args.output_npz, **data)
    print(f"Saved poses for {data['poses'].shape[0]} frames to {args.output_npz}")

    if args.video is not None:
        if not args.video.is_file():
            raise SystemExit(f"Video file does not exist: {args.video}")

        overview_path = args.overview_video
        if overview_path is None:
            overview_path = args.output_npz.with_name(args.output_npz.stem + "_overview.mp4")
        overview_path.parent.mkdir(parents=True, exist_ok=True)

        node_names = [str(n) for n in data["node_names"]]
        edges, point_colors = build_skeleton(node_names)

        make_overview_video(
            video_path=args.video,
            poses=data["poses"],
            edges=edges,
            point_colors=point_colors,
            output_path=overview_path,
            crf=args.crf,
            target_height=args.height,
        )
        print(f"Saved overview video to {overview_path}")


if __name__ == "__main__":
    main()
