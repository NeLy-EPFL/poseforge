#!/usr/bin/env python
"""Convert between SLEAP `.slp` prediction files and `.h5` pose arrays.

Two modes, selected by `--slp2h5` or `--h52slp`:

- `--slp2h5`: extract the highest-confidence instance per frame from a `.slp`
  file into an `.h5` file with `poses`, `instance_scores`, `keypoint_scores`
  datasets and a `node_names` attr. Optionally also renders an annotated
  overview video (drawn with OpenCV only, no matplotlib). Pass
  `--include-acceptance` to also record, per frame, whether a user-labeled
  `Instance` (not just a prediction) is present, e.g. for the output of
  `filter_slp_labels.py`.
- `--h52slp`: build a new-format `.slp` file (one `PredictedInstance` per
  frame, no user labels) from such an `.h5`, linked to a given video, with
  the skeleton taken from a reference `.slp` file.

By default, `.slp` reading goes through `sleap_io`, which transparently
handles both the legacy (pre-2024) and current SLEAP file formats, and all
writing goes through `sleap_io`, which always writes the current format (see
https://sleap.ai/). Pass `--legacy-slp-format` to instead read via the older
`sleap` package's own loader (`sleap.load_file`) for `--slp2h5`, for
environments where only that package is installed and not `sleap_io` (e.g.
the standalone `sleap=1.4.1` conda environment described in
`prior2d/README.md`).

Usage:
    python convert_slp.py --slp2h5 --input-path predictions.slp --output-path poses.h5
    python convert_slp.py --h52slp --input-path poses.h5 --output-path poses.slp \
        --video-path video.mkv --reference-slp-path reference.slp
"""

import shutil
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np
import sleap_io as sio
import tyro
from loguru import logger

from poseforge.prior2d.skeleton_viz import build_skeleton, draw_pose

# Written as root h5 attrs (small, dataset-level metadata) rather than
# datasets (bulk per-frame arrays); see `save_poses_h5`.
ATTR_KEYS = {"node_names", "genotypes", "fly_trials", "n_frames_per_video"}


def save_poses_h5(data: dict, output_path: Path) -> None:
    """Save an `extract_poses`-style dict to an `.h5` file.

    Args:
        data: Dict as returned by `extract_poses`/`extract_poses_legacy`.
        output_path: Where to save the `.h5` file.
    """
    with h5py.File(output_path, "w") as f:
        for key, value in data.items():
            if key in ATTR_KEYS:
                # h5py can't store a fixed-length-unicode ("<U...") numpy
                # array directly as an attr; a plain list converts numpy
                # string arrays to `str` (and numeric ones to Python
                # ints/floats) uniformly, both of which h5py accepts.
                f.attrs[key] = value.tolist()
            else:
                f.create_dataset(key, data=value, compression="gzip")


def load_poses_h5(input_path: Path) -> dict:
    """Load an `.h5` file written by `save_poses_h5` back into a plain dict.

    Args:
        input_path: `.h5` file to load.

    Returns:
        Dict with the same keys/shapes `extract_poses`/`extract_poses_legacy`
        would have produced (attrs and datasets both as plain numpy arrays).
    """
    with h5py.File(input_path, "r") as f:
        data = {key: f.attrs[key] for key in f.attrs}
        data.update({key: f[key][:] for key in f})
    return data


def extract_poses_legacy(slp_path: Path) -> dict:
    """Load a `.slp` file via the legacy `sleap` package and extract the best
    instance per frame.

    Only works in environments where that package is installed and its
    `load_file` API is available (see module docstring).

    Args:
        slp_path: Path to a SLEAP predictions (.slp) file.

    Returns:
        Dict with keys "poses", "instance_scores", "keypoint_scores",
        "node_names", ready to be passed to `save_poses_h5`.
    """
    import sleap

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


def parse_trial_identity(video_path: str) -> tuple[str, str]:
    """Parse `(genotype, fly_trial)` from a video path.

    Assumes the `.../<genotype>/<fly_trial>/processed/<video>` directory
    convention used throughout this project (e.g.
    `.../G-213xCI55_260720/fly000_trial000/processed/aligned_behavior_video.mkv`).

    Args:
        video_path: Path to a trial's video file.

    Returns:
        `(genotype, fly_trial)`, e.g. `("G-213xCI55_260720", "fly000_trial000")`.
    """
    parts = Path(video_path).parts
    return parts[-4], parts[-3]


def _extract_video_poses(
    labeled_frames: list[sio.LabeledFrame],
    n_frames: int,
    n_nodes: int,
    include_acceptance: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract dense per-frame arrays for one video's labeled frames.

    Args:
        labeled_frames: This video's `LabeledFrame`s (any frame_idx < n_frames).
        n_frames: Number of frames to allocate (frames without a label stay NaN).
        n_nodes: Number of skeleton nodes.
        include_acceptance: See `extract_poses`.

    Returns:
        poses: `(n_frames, n_nodes, 2)`.
        instance_scores: `(n_frames,)`.
        keypoint_scores: `(n_frames, n_nodes)`.
        accepted: `(n_frames,)` bool; all False if not `include_acceptance`.
    """
    poses = np.full((n_frames, n_nodes, 2), np.nan, dtype=np.float32)
    instance_scores = np.full(n_frames, np.nan, dtype=np.float32)
    keypoint_scores = np.full((n_frames, n_nodes), np.nan, dtype=np.float32)
    accepted = np.zeros(n_frames, dtype=bool)

    for lf in labeled_frames:
        predicted = [
            inst for inst in lf.instances if isinstance(inst, sio.PredictedInstance)
        ]
        if predicted:
            best = max(predicted, key=lambda inst: inst.score)
            pts_scores = best.numpy(scores=True)  # (n_nodes, 3): x, y, score
            poses[lf.frame_idx] = pts_scores[:, :2]
            keypoint_scores[lf.frame_idx] = pts_scores[:, 2]
            instance_scores[lf.frame_idx] = best.score

        if include_acceptance:
            # PredictedInstance is a subclass of Instance, so exclude it explicitly
            # to isolate actual user-labeled instances.
            user_instances = [
                inst
                for inst in lf.instances
                if not isinstance(inst, sio.PredictedInstance)
            ]
            if user_instances:
                accepted[lf.frame_idx] = True
                poses[lf.frame_idx] = user_instances[0].numpy()

    return poses, instance_scores, keypoint_scores, accepted


def extract_poses(slp_path: Path, include_acceptance: bool = False) -> dict:
    """Load a `.slp` file (any format) via `sleap_io` and extract the best
    instance per frame.

    A single-video file returns flat, unpadded `(n_frames, ...)` arrays. A
    multi-video file (e.g. the combined output of `port_slp_labels.py`)
    instead returns `(n_videos, max_n_frames, ...)` arrays, one row per
    video padded with NaN/False up to the longest video, plus "genotypes",
    "fly_trials" (see `parse_trial_identity`), and "n_frames_per_video" so
    padding can be distinguished from real data.

    Args:
        slp_path: Path to a SLEAP predictions (.slp) file.
        include_acceptance: If True, also add an "accepted" boolean array,
            True for frames with a user-labeled `Instance` (e.g. the output
            of `filter_slp_labels.py`). "poses" then holds that instance's
            (possibly partially-NaN) points for accepted frames, rather than
            the raw prediction; "instance_scores"/"keypoint_scores" still
            always come from the underlying predicted instance, since a user
            instance has no score of its own. Off by default so the output
            schema matches plain prediction files exactly.

    Returns:
        Dict ready to be passed to `save_poses_h5`. See above for the
        single- vs. multi-video schemas.
    """
    labels = sio.load_file(str(slp_path))
    node_names = [node.name for node in labels.skeleton.nodes]
    n_nodes = len(node_names)

    if len(labels.videos) == 1:
        video = labels.videos[0]
        n_frames = max(lf.frame_idx for lf in labels.labeled_frames) + 1
        poses, instance_scores, keypoint_scores, accepted = _extract_video_poses(
            labels.labeled_frames, n_frames, n_nodes, include_acceptance
        )
        data = {
            "poses": poses,
            "instance_scores": instance_scores,
            "keypoint_scores": keypoint_scores,
            "node_names": np.array(node_names),
        }
        if include_acceptance:
            data["accepted"] = accepted
        return data

    per_video_frames = {video: [] for video in labels.videos}
    for lf in labels.labeled_frames:
        per_video_frames[lf.video].append(lf)
    n_frames_per_video = [
        max((lf.frame_idx for lf in frames), default=-1) + 1
        for frames in per_video_frames.values()
    ]
    max_n_frames = max(n_frames_per_video)

    poses = np.full(
        (len(labels.videos), max_n_frames, n_nodes, 2), np.nan, dtype=np.float32
    )
    instance_scores = np.full(
        (len(labels.videos), max_n_frames), np.nan, dtype=np.float32
    )
    keypoint_scores = np.full(
        (len(labels.videos), max_n_frames, n_nodes), np.nan, dtype=np.float32
    )
    accepted = np.zeros((len(labels.videos), max_n_frames), dtype=bool)
    genotypes, fly_trials = [], []

    for video_idx, (video, frames) in enumerate(per_video_frames.items()):
        genotype, fly_trial = parse_trial_identity(video.filename)
        genotypes.append(genotype)
        fly_trials.append(fly_trial)
        v_poses, v_instance_scores, v_keypoint_scores, v_accepted = (
            _extract_video_poses(
                frames, n_frames_per_video[video_idx], n_nodes, include_acceptance
            )
        )
        n = n_frames_per_video[video_idx]
        poses[video_idx, :n] = v_poses
        instance_scores[video_idx, :n] = v_instance_scores
        keypoint_scores[video_idx, :n] = v_keypoint_scores
        accepted[video_idx, :n] = v_accepted

    data = {
        "poses": poses,
        "instance_scores": instance_scores,
        "keypoint_scores": keypoint_scores,
        "node_names": np.array(node_names),
        "genotypes": np.array(genotypes),
        "fly_trials": np.array(fly_trials),
        "n_frames_per_video": np.array(n_frames_per_video),
    }
    if include_acceptance:
        data["accepted"] = accepted
    return data


def build_predicted_labels(
    poses: np.ndarray,
    instance_scores: np.ndarray,
    keypoint_scores: np.ndarray,
    skeleton: sio.Skeleton,
    video_path: Path,
) -> sio.Labels:
    """Build a `Labels` object with one `PredictedInstance` per detected frame.

    No filtering or promotion to user labels is done here; see
    `filter_slp_labels.py` for that.

    Args:
        poses: `(n_frames, n_nodes, 2)` keypoint coordinates, NaN where undetected.
        instance_scores: `(n_frames,)` overall instance detection scores.
        keypoint_scores: `(n_frames, n_nodes)` per-keypoint detection scores.
        skeleton: Skeleton to attach instances to.
        video_path: Path to the video these predictions belong to.

    Returns:
        A `Labels` object with a single video and one `LabeledFrame` per
        frame with a detected pose.
    """
    video = sio.Video(filename=str(Path(video_path).resolve()))

    labeled_frames = []
    for frame_idx in range(poses.shape[0]):
        if np.isnan(poses[frame_idx]).all():
            continue
        predicted_instance = sio.PredictedInstance.from_numpy(
            points_data=poses[frame_idx],
            skeleton=skeleton,
            point_scores=keypoint_scores[frame_idx],
            score=float(instance_scores[frame_idx]),
        )
        labeled_frames.append(
            sio.LabeledFrame(
                video=video, frame_idx=frame_idx, instances=[predicted_instance]
            )
        )

    return sio.Labels(
        labeled_frames=labeled_frames, videos=[video], skeletons=[skeleton]
    )


def resize_pad_and_scale_points(frame, points, target_height):
    """Resize frame to target_height (aspect-preserving), pad to multiple of
    16 on each axis, and return the transformed frame plus rescaled points.
    """
    h, w = frame.shape[:2]
    scale = target_height / h
    new_w = round(w * scale)
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
        raise SystemExit(
            "ffmpeg not found on PATH; load/install ffmpeg to render the overview video."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = target_height / orig_h
    scaled_w = round(orig_w * scale)
    out_w = scaled_w + ((-scaled_w) % 16)
    out_h = target_height + ((-target_height) % 16)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", str(fps),
        "-i", "-",
        "-an", "-vcodec", "libx264", "-crf", str(crf),
        "-pix_fmt", "yuv420p", str(output_path),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    n_frames = poses.shape[0]
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pts = (
                poses[frame_idx]
                if frame_idx < n_frames
                else np.full((poses.shape[1], 2), np.nan)
            )
            out_frame, scaled_pts = resize_pad_and_scale_points(
                frame, pts.copy(), target_height
            )
            draw_pose(out_frame, scaled_pts, edges, point_colors)
            proc.stdin.write(out_frame.tobytes())
            frame_idx += 1
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()


def run_slp2h5(
    input_path: Path,
    output_path: Path,
    legacy_slp_format: bool,
    include_acceptance: bool,
    video_path: Path | None,
    overview_video: Path | None,
    crf: int,
    height: int,
) -> None:
    """Extract poses from a `.slp` file into an `.h5`, optionally rendering an
    annotated overview video."""
    if not input_path.is_file():
        raise SystemExit(f"Input file does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = (
        extract_poses_legacy(input_path)
        if legacy_slp_format
        else extract_poses(input_path, include_acceptance)
    )
    save_poses_h5(data, output_path)
    if data["poses"].ndim == 3:
        logger.info(f"Saved poses for {data['poses'].shape[0]} frames to {output_path}")
    else:
        logger.info(
            f"Saved poses for {data['n_frames_per_video'].sum()} total frames "
            f"across {data['poses'].shape[0]} videos (padded to "
            f"{data['poses'].shape[1]} frames each) to {output_path}"
        )

    if video_path is not None:
        if not video_path.is_file():
            raise SystemExit(f"Video file does not exist: {video_path}")

        overview_path = overview_video
        if overview_path is None:
            overview_path = output_path.with_name(output_path.stem + "_overview.mp4")
        overview_path.parent.mkdir(parents=True, exist_ok=True)

        node_names = [str(n) for n in data["node_names"]]
        edges, point_colors = build_skeleton(node_names)

        make_overview_video(
            video_path=video_path,
            poses=data["poses"],
            edges=edges,
            point_colors=point_colors,
            output_path=overview_path,
            crf=crf,
            target_height=height,
        )
        logger.info(f"Saved overview video to {overview_path}")


def run_h52slp(
    input_path: Path,
    output_path: Path,
    video_path: Path,
    reference_slp_path: Path,
) -> None:
    """Build a new-format `.slp` file (predictions only) from an `.h5`."""
    if not input_path.is_file():
        raise SystemExit(f"Input file does not exist: {input_path}")
    if not video_path.is_file():
        raise SystemExit(f"Video file does not exist: {video_path}")
    if not reference_slp_path.is_file():
        raise SystemExit(f"Reference .slp file does not exist: {reference_slp_path}")

    with h5py.File(input_path, "r") as f:
        poses = f["poses"][:]
        instance_scores = f["instance_scores"][:]
        keypoint_scores = f["keypoint_scores"][:]

    skeleton = sio.load_file(str(reference_slp_path)).skeleton
    labels = build_predicted_labels(
        poses, instance_scores, keypoint_scores, skeleton, video_path
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_file(labels, str(output_path))
    logger.info(f"Saved {len(labels.labeled_frames)} predicted frames to {output_path}")


def main(
    input_path: Path,
    output_path: Path,
    slp2h5: bool = False,
    h52slp: bool = False,
    legacy_slp_format: bool = False,
    include_acceptance: bool = False,
    video_path: Path | None = None,
    reference_slp_path: Path | None = None,
    overview_video: Path | None = None,
    crf: int = 23,
    height: int = 1024,
) -> None:
    """Convert between SLEAP `.slp` prediction files and `.h5` pose arrays.

    Args:
        input_path: Path to the input file (`.slp` for --slp2h5, `.h5` for
            --h52slp).
        output_path: Path to write the output file (`.h5` for --slp2h5,
            `.slp` for --h52slp).
        slp2h5: Extract poses from `input_path` (a `.slp` file) into
            `output_path` (an `.h5` file). Exactly one of --slp2h5/--h52slp
            must be set.
        h52slp: Build a predictions-only `.slp` file at `output_path` from
            `input_path` (an `.h5` file). Requires --video-path and
            --reference-slp-path. Exactly one of --slp2h5/--h52slp must be set.
        legacy_slp_format: --slp2h5 only. Read `input_path` with the older
            `sleap` package's loader instead of `sleap_io`. See module docstring.
        include_acceptance: --slp2h5 only (and incompatible with
            legacy_slp_format). Also write an "accepted" dataset. See `extract_poses`.
        video_path: --slp2h5: if given, also render an annotated overview
            video from this source video. --h52slp: the video to link the
            output predictions to (required).
        reference_slp_path: --h52slp only. `.slp` file to take the skeleton
            from (required).
        overview_video: --slp2h5 only. Output path for the overview video.
            Defaults to '<output_path stem>_overview.mp4'.
        crf: --slp2h5 only. x264 CRF for the overview video.
        height: --slp2h5 only. Target height in pixels for the overview
            video; width is scaled to preserve aspect ratio and padded with
            black to the next multiple of 16.
    """
    if slp2h5 == h52slp:
        raise SystemExit("Specify exactly one of --slp2h5 or --h52slp.")

    if slp2h5:
        if legacy_slp_format and include_acceptance:
            raise SystemExit(
                "--include-acceptance requires reading via sleap_io; "
                "it is not supported with --legacy-slp-format."
            )
        run_slp2h5(
            input_path,
            output_path,
            legacy_slp_format,
            include_acceptance,
            video_path,
            overview_video,
            crf,
            height,
        )
    else:
        if video_path is None or reference_slp_path is None:
            raise SystemExit("--h52slp requires --video-path and --reference-slp-path.")
        run_h52slp(input_path, output_path, video_path, reference_slp_path)


if __name__ == "__main__":
    tyro.cli(main)
