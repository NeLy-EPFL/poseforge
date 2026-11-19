"""Unpack aligned Spotlight recordings into frames and categorize each one.

For every trial under ``--input-dir`` matching ``--glob-pattern`` this writes the
standard ``spotlight_aligned_and_cropped/<trial>/`` layout consumed by
``poseforge.style_transfer.scripts.extract_dataset`` (and the downstream CUT
training):

    <trial>/all/frame_%09d.jpg            every decoded frame
    <trial>/all/frame_%09d.metadata.json  that frame's metadata + category flags
    <trial>/predicted_flip_labels.csv     image,predicted_label  (see below)
    <trial>/model_prediction/{not_flipped,flipped}/   symlinks into all/
    <trial>/manual_label/                 empty, for later hand corrections
    <trial>/categorization_summary.json   counts + preservation

Categories per frame:
  * edge      -- a wall is visible (keypoint-independent corner test),
  * flipped   -- fly upside down (flip CNN + temporal filter),
  * untracked -- pose keypoints missing (reported, NOT excluded),
  * rearing   -- head lifted / antennae hidden (rearing CNN on a fixed crop).

A frame is ``usable`` iff it is not edge, not flipped, and not rearing (untracked is
allowed). ``predicted_flip_labels.csv`` marks every usable frame as ``"not flipped"``
and everything else as ``"flipped"``, so the CUT extractor -- which keeps only
``"not flipped"`` rows -- naturally takes only usable frames. The full per-category
truth is in each frame's ``metadata.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

cv2.setNumThreads(1)  # one process per trial already; keep each decoder single-threaded


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_flip_model(flip_config: dict, checkpoint_path: Path, device: torch.device):
    import poseforge.spotlight.flip_detection as flip_detection

    model = flip_detection.FlipDetectionCNN(**flip_config["init_params"])
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def _flip_working_tensor(gray_frame: np.ndarray, working_size: int) -> np.ndarray:
    resized = cv2.resize(gray_frame, (working_size, working_size)).astype(np.float32) / 255.0
    return (resized - 0.5) / 0.5


def process_trial(
    trial_dir: Path,
    arena_dir: Path,
    output_dir: Path,
    config: dict,
    flip_checkpoint: Path,
    rearing_checkpoint: Path,
    device_str: str,
    edge_clearance_mm: float,
    flip_min_run: int,
    flip_pad_seconds: float,
    behavior_fps: float,
    jpeg_quality: int,
    max_frames: int,
) -> dict:
    from poseforge.spotlight import rearing_detection as rearing
    from poseforge.spotlight import valid_frame_filtering as filtering

    device = _resolve_device(device_str)
    trial = trial_dir.name
    out_trial = output_dir / trial
    all_dir = out_trial / "all"
    all_dir.mkdir(parents=True, exist_ok=True)

    video_path = trial_dir / "processed" / "aligned_behavior_video.mkv"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = max(width, height)

    # Edge + untracked flags (no decode).
    too_close, keypoints_missing, n_edge = filtering.corner_wall_flags(
        trial_dir, arena_dir, frame_size, edge_clearance_mm
    )

    # Per-frame recorder metadata.
    metadata_df = pd.read_csv(trial_dir / "processed" / "behavior_frames_metadata.csv")
    metadata_by_frame = metadata_df.set_index("behavior_frame_id")

    n = min(n_edge, len(metadata_df))
    if max_frames > 0:
        n = min(n, max_frames)

    # Flip model + transform params.
    flip_config = config["flip_detection"]
    flip_model = _load_flip_model(flip_config, flip_checkpoint, device)
    flip_working_size = flip_config["working_size"]
    flip_labels = np.array(flip_config["output_labels"])  # index 1 == flipped

    # --- Pass 1: decode, write every frame, run flip detection ---
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    flip_raw = np.zeros(n, dtype=int)
    flip_conf = np.zeros(n, dtype=np.float32)
    batch_tensors: list[np.ndarray] = []
    batch_indices: list[int] = []

    def flush_flip():
        if not batch_tensors:
            return
        stacked = torch.from_numpy(np.stack(batch_tensors))[:, None].to(device)
        with torch.no_grad():
            probs = torch.softmax(flip_model(stacked), dim=1)
            conf, pred = torch.max(probs, dim=1)
        for k, idx in enumerate(batch_indices):
            flip_raw[idx] = int(pred[k].item())
            flip_conf[idx] = float(conf[k].item())
        batch_tensors.clear()
        batch_indices.clear()

    for idx in range(n):
        ok, frame = capture.read()
        if not ok:
            n = idx
            break
        cv2.imwrite(str(all_dir / f"frame_{idx:09d}.jpg"), frame, jpeg_params)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        batch_tensors.append(_flip_working_tensor(gray, flip_working_size))
        batch_indices.append(idx)
        if len(batch_tensors) >= 128:
            flush_flip()
    flush_flip()
    capture.release()

    # Trim all arrays to the number of frames actually decoded.
    too_close = too_close[:n]
    keypoints_missing = keypoints_missing[:n]
    flip_raw = flip_raw[:n]
    flip_conf = flip_conf[:n]

    flipped = filtering.temporal_flip_filter(
        flip_raw, behavior_fps, min_run=flip_min_run, pad_seconds=flip_pad_seconds
    ).astype(bool)

    # --- Pass 2: rearing on candidate frames (not edge, not flipped) ---
    model, snapshots, crop = rearing.load_checkpoint(rearing_checkpoint, device)
    neck = rearing.median_neck(trial_dir / "processed" / "behavior_alignment_transforms.h5")
    rearing_prob = np.full(n, np.nan, dtype=np.float32)
    candidate_idx = [
        i for i in range(n) if (not too_close[i]) and (not flipped[i])
    ]
    crops, cropped_idx = [], []
    for i in candidate_idx:
        gray = cv2.imread(str(all_dir / f"frame_{i:09d}.jpg"), cv2.IMREAD_GRAYSCALE)
        c = rearing.head_crop(gray, neck, half=crop["half"], y_offset=crop["y_offset"])
        if c is not None:
            crops.append(c)
            cropped_idx.append(i)
    probs = rearing.predict_rearing_probability(
        model, snapshots, crops, device, input_size=crop["input_size"]
    )
    for i, p in zip(cropped_idx, probs):
        rearing_prob[i] = p
    is_rearing = np.nan_to_num(rearing_prob, nan=0.0) >= 0.5

    usable = (~too_close) & (~flipped) & (~is_rearing)

    # --- Write per-frame metadata.json ---
    for i in range(n):
        row = (
            metadata_by_frame.loc[i].to_dict()
            if i in metadata_by_frame.index
            else {}
        )
        meta = {
            "behavior_frame_id": int(i),
            **{k: (None if pd.isna(v) else v) for k, v in row.items()},
            "flip_detection": str(flip_labels[flip_raw[i]]),
            "flip_conf": float(flip_conf[i]),
            "flipped": bool(flipped[i]),
            "too_close_to_edge": bool(too_close[i]),
            "keypoints_missing": bool(keypoints_missing[i]),
            "rearing_prob": (None if np.isnan(rearing_prob[i]) else float(rearing_prob[i])),
            "rearing": bool(is_rearing[i]),
            "usable": bool(usable[i]),
        }
        (all_dir / f"frame_{i:09d}.metadata.json").write_text(json.dumps(meta))

    # --- predicted_flip_labels.csv (usable -> "not flipped", else "flipped") ---
    images = [f"frame_{i:09d}.jpg" for i in range(n)]
    labels = ["not flipped" if usable[i] else "flipped" for i in range(n)]
    pd.DataFrame({"image": images, "predicted_label": labels}).to_csv(
        out_trial / "predicted_flip_labels.csv", index=False
    )

    # --- model_prediction/{not_flipped,flipped}/ symlinks + empty manual_label/ ---
    (out_trial / "manual_label").mkdir(exist_ok=True)
    label_dirs = {}
    for label in ["not flipped", "flipped"]:
        d = out_trial / "model_prediction" / label.replace(" ", "_")
        d.mkdir(parents=True, exist_ok=True)
        label_dirs[label] = d
    for i in range(n):
        src = (all_dir / images[i]).resolve()
        dst = label_dirs[labels[i]] / images[i]
        if not dst.exists():
            dst.symlink_to(src)

    summary = {
        "trial": trial,
        "n_frames": int(n),
        "edge": int(too_close.sum()),
        "flipped": int(flipped.sum()),
        "untracked": int(keypoints_missing.sum()),
        "rearing": int(is_rearing.sum()),
        "usable": int(usable.sum()),
        "usable_pct": round(100 * usable.sum() / max(n, 1), 1),
        "untracked_among_usable": int((keypoints_missing & usable).sum()),
    }
    (out_trial / "categorization_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{trial}] {summary}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--arena-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob-pattern", type=str, default="G*fly*")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(__file__).resolve().parents[1].parent
        / "production/spotlight/config.yaml",
    )
    parser.add_argument("--flip-checkpoint", type=Path, default=None)
    parser.add_argument("--rearing-checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--edge-clearance-mm", type=float, default=0.4)
    parser.add_argument("--flip-min-run", type=int, default=3)
    parser.add_argument("--flip-pad-seconds", type=float, default=0.3)
    parser.add_argument("--behavior-fps", type=float, default=396.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-frames-per-trial", type=int, default=0, dest="max_frames")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config_path).read_text())
    flip_checkpoint = args.flip_checkpoint or Path(config["flip_detection"]["checkpoint"])
    rearing_checkpoint = args.rearing_checkpoint or Path(
        config["rearing_detection"]["checkpoint"]
    )

    trials = sorted(
        d for d in args.input_dir.glob(args.glob_pattern) if (d / "processed").is_dir()
    )
    print(f"Found {len(trials)} trial(s) matching '{args.glob_pattern}' in {args.input_dir}")

    common = dict(
        arena_dir=args.arena_dir,
        output_dir=args.output_dir,
        config=config,
        flip_checkpoint=flip_checkpoint,
        rearing_checkpoint=rearing_checkpoint,
        device_str=args.device,
        edge_clearance_mm=args.edge_clearance_mm,
        flip_min_run=args.flip_min_run,
        flip_pad_seconds=args.flip_pad_seconds,
        behavior_fps=args.behavior_fps,
        jpeg_quality=args.jpeg_quality,
        max_frames=args.max_frames,
    )

    if args.n_workers <= 1 or len(trials) <= 1:
        for trial_dir in trials:
            process_trial(trial_dir=trial_dir, **common)
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {
                executor.submit(process_trial, trial_dir=t, **common): t.name
                for t in trials
            }
            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
