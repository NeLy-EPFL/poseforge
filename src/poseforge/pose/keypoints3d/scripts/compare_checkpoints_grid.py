"""Compare multiple checkpoints against hand-corrected ground truth on a grid.

Given a hardcoded list of checkpoints, a ground-truth h5 (as produced by
``annotate_keypoints_gui.py``: ``frame_indices`` + ``keypoints_xy`` + ``visible``),
and the full-resolution GT video, this script:

    1. Reads the GT frames (``frame_indices``) and GT keypoints.
    2. For each checkpoint, runs inference on exactly those frames and scales the
       predictions back to full resolution.
    3. Renders one grid per GT frame: each cell is a checkpoint, showing the frame
       with GT dots (lime) and predicted dots (red), and a YELLOW line linking each
       GT dot to its matching predicted dot. Only visible keypoints are drawn/scored.
    4. Each cell title has two lines: the run directory name, then
       ``epoch <e> step <s> | RMSE <px>`` (RMSE over the visible keypoints).
    5. Writes the grid as a video to the GT video's directory.

This mirrors ``run_video_inference.py`` (same model loading / inference / scaling),
but runs only on the chosen frames and compares checkpoints side by side.

Usage:
    # Edit CHECKPOINTS below, then:
    python compare_checkpoints_grid.py --gt-h5 GT.h5 --video /path/to/gt_video.mp4 \
        [--rows R --cols C] [--device cuda] [--fps 2]
"""

import argparse
import logging
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive; render grid frames to arrays
import matplotlib.pyplot as plt  # noqa: E402

import cv2  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from poseforge.pose.keypoints3d.scripts.run_video_inference import (  # noqa: E402
    _resolve_model_dir,
    _load_pipeline,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# Hardcoded checkpoints to compare. Edit this list.
# ===========================================================================
CHECKPOINTS = [
    "/Volumes/upramdya/data/VAS/poseforge/improve_kpt/trial_20260529_tiled_keypoints3d_freeze/checkpoints/epoch0_step1000.model.pth",
    # "/path/to/another/trial_.../checkpoints/epochN_stepM.model.pth",
]


# Colors (matplotlib named colors / RGBA)
GT_COLOR = "lime"
PRED_COLOR = "red"
LINK_COLOR = "yellow"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_epoch_step(checkpoint_path: Path) -> tuple[str, str]:
    """Extract epoch and step strings from a checkpoint filename like
    ``epoch0_step1000.model.pth``. Falls back to ('?', '?')."""
    m = re.search(r"epoch(\d+)_step(\d+)", checkpoint_path.name)
    if m:
        return m.group(1), m.group(2)
    return "?", "?"


def _read_gt(
    gt_h5_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str] | None, tuple[int, int] | None]:
    """Read GT frame indices, keypoints, visibility, names, full-res size.

    Returns:
        frame_indices: (m,) int.
        gt_xy: (m, n_kp, 2) float32 in full-resolution pixel space.
        visible: (m, n_kp) bool.
        kp_names: list[str] or None.
        full_res: (W, H) or None.
    """
    with h5py.File(gt_h5_path, "r") as f:
        if "keypoints_xy" not in f or "frame_indices" not in f:
            raise KeyError(
                f"GT h5 must contain 'frame_indices' and 'keypoints_xy'. "
                f"Found: {list(f.keys())}"
            )
        frame_indices = np.asarray(f["frame_indices"][:]).astype(int)
        gt_xy = np.asarray(f["keypoints_xy"][:], dtype=np.float32)
        if "visible" in f:
            visible = np.asarray(f["visible"][:]).astype(bool)
        else:
            visible = np.ones(gt_xy.shape[:2], dtype=bool)

        kp_names = None
        raw_names = f.attrs.get("keypoints")
        if raw_names is not None:
            kp_names = [
                n.decode("utf-8") if isinstance(n, bytes) else str(n)
                for n in list(raw_names)
            ]

        full_res = None
        raw_fr = f.attrs.get("full_resolution")
        if raw_fr is not None:
            full_res = tuple(int(v) for v in np.asarray(raw_fr).ravel()[:2])

    return frame_indices, gt_xy, visible, kp_names, full_res


def _read_frames_at(
    video_path: Path, frame_indices: np.ndarray
) -> tuple[list[np.ndarray], int, int, float]:
    """Read specific frames (RGB, full-res) from a video by index."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
        frames.append(frame[:, :, ::-1].copy())  # BGR -> RGB
    cap.release()
    return frames, orig_w, orig_h, fps


def _infer_full_res(
    pipeline,
    image_size: tuple[int, int],
    frames_rgb: list[np.ndarray],
    orig_size: tuple[int, int],
    batch_size: int,
) -> np.ndarray:
    """Run inference on RGB frames; return preds (m, n_kp, 2) in full-res pixels."""
    target_h, target_w = image_size  # data config stores (height, width)
    orig_w, orig_h = orig_size
    scale_x = orig_w / target_w
    scale_y = orig_h / target_h

    preds = []
    for start in range(0, len(frames_rgb), batch_size):
        chunk = frames_rgb[start : start + batch_size]
        batch = np.stack(
            [cv2.resize(f, (target_w, target_h)) for f in chunk], axis=0
        )  # (N, H, W, 3) RGB
        batch_float = batch.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        tensor = torch.from_numpy(batch_float)
        with torch.no_grad():
            pred_dict = pipeline.inference(tensor)
        pxy = pred_dict["pred_xy"].cpu().numpy().copy()  # (N, n_kp, 2)
        pxy[..., 0] *= scale_x
        pxy[..., 1] *= scale_y
        preds.append(pxy)
    return np.concatenate(preds, axis=0)


def _rmse(gt: np.ndarray, pred: np.ndarray, vis: np.ndarray) -> float:
    """RMS of per-keypoint pixel error over visible keypoints."""
    if vis.sum() == 0:
        return float("nan")
    d2 = ((gt[vis] - pred[vis]) ** 2).sum(axis=1)  # squared euclidean per kp
    return float(np.sqrt(d2.mean()))


def _render_grid_frame(
    frame_rgb: np.ndarray,
    cells: list[dict],
    rows: int,
    cols: int,
    frame_label: str,
    dpi: int = 100,
) -> np.ndarray:
    """Render one grid frame (RGB uint8 array) comparing all checkpoints."""
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4.0, rows * 4.3), dpi=dpi, squeeze=False
    )
    h, w = frame_rgb.shape[:2]
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= len(cells):
            ax.axis("off")
            continue
        cell = cells[idx]
        gt, pred, vis = cell["gt"], cell["pred"], cell["vis"]
        ax.imshow(frame_rgb)
        # Yellow links between matching GT and predicted points
        for k in range(gt.shape[0]):
            if not vis[k]:
                continue
            ax.plot(
                [gt[k, 0], pred[k, 0]], [gt[k, 1], pred[k, 1]],
                color=LINK_COLOR, linewidth=0.8, zorder=2,
            )
        ax.scatter(
            gt[vis, 0], gt[vis, 1], s=14, c=GT_COLOR,
            edgecolors="black", linewidths=0.4, zorder=3,
        )
        ax.scatter(
            pred[vis, 0], pred[vis, 1], s=14, c=PRED_COLOR,
            edgecolors="black", linewidths=0.4, zorder=3,
        )
        rmse_str = "nan" if math.isnan(cell["rmse"]) else f"{cell['rmse']:.1f}px"
        ax.set_title(
            f"{cell['run_name']}\n"
            f"epoch {cell['epoch']} step {cell['step']} | RMSE {rmse_str}",
            fontsize=8,
        )
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

    fig.suptitle(frame_label, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()  # RGB
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(
    gt_h5: Path,
    video_path: Path,
    rows: int | None = None,
    cols: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 16,
    fps: float = 2.0,
    output_path: Path | None = None,
):
    gt_h5 = Path(gt_h5).expanduser().resolve()
    video_path = Path(video_path).expanduser().resolve()
    if not gt_h5.is_file():
        raise FileNotFoundError(f"GT h5 not found: {gt_h5}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    checkpoints = [Path(c).expanduser().resolve() for c in CHECKPOINTS]
    checkpoints = [c for c in checkpoints if c.is_file()]
    if not checkpoints:
        raise ValueError(
            "No valid checkpoints found. Edit CHECKPOINTS at the top of the script."
        )
    n_ckpt = len(checkpoints)

    # Grid layout
    if cols is None:
        cols = math.ceil(math.sqrt(n_ckpt))
    if rows is None:
        rows = math.ceil(n_ckpt / cols)
    if rows * cols < n_ckpt:
        raise ValueError(f"Grid {rows}x{cols} too small for {n_ckpt} checkpoints.")

    # Load GT + frames
    frame_indices, gt_xy, visible, kp_names, full_res = _read_gt(gt_h5)
    frames_rgb, orig_w, orig_h, video_fps = _read_frames_at(video_path, frame_indices)
    logger.info(
        f"GT: {len(frame_indices)} frames, {gt_xy.shape[1]} keypoints. "
        f"Video: {orig_w}x{orig_h}."
    )
    if full_res is not None and tuple(full_res) != (orig_w, orig_h):
        logger.warning(
            f"GT full_resolution {tuple(full_res)} != video size {(orig_w, orig_h)}. "
            "Using the video size; check that the GT video matches."
        )

    # Run inference for each checkpoint
    per_ckpt = []  # list of dicts with run_name, epoch, step, pred (m, n_kp, 2)
    for ckpt in checkpoints:
        model_dir = _resolve_model_dir(ckpt)
        pipeline, image_size = _load_pipeline(model_dir, ckpt, device)
        epoch, step = _parse_epoch_step(ckpt)
        preds = _infer_full_res(
            pipeline, image_size, frames_rgb, (orig_w, orig_h), batch_size
        )
        per_ckpt.append(
            {"run_name": model_dir.name, "epoch": epoch, "step": step, "pred": preds}
        )
        logger.info(f"  Inferred {model_dir.name} (epoch {epoch} step {step}).")

    # Output path
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_checkpoint_grid.mp4"
    output_path = Path(output_path)

    # Render each GT frame as a grid and collect into a video
    writer = None
    for fi, frame_rgb in enumerate(frames_rgb):
        cells = []
        for ck in per_ckpt:
            pred = ck["pred"][fi]
            cells.append(
                {
                    "run_name": ck["run_name"],
                    "epoch": ck["epoch"],
                    "step": ck["step"],
                    "gt": gt_xy[fi],
                    "pred": pred,
                    "vis": visible[fi],
                    "rmse": _rmse(gt_xy[fi], pred, visible[fi]),
                }
            )
        label = f"frame index {int(frame_indices[fi])}  ({fi + 1}/{len(frames_rgb)})"
        grid_rgb = _render_grid_frame(frame_rgb, cells, rows, cols, label)

        if writer is None:
            gh, gw = grid_rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (gw, gh))
        writer.write(grid_rgb[:, :, ::-1])  # RGB -> BGR
        logger.info(f"Rendered grid for frame {fi + 1}/{len(frames_rgb)}")

    if writer is not None:
        writer.release()
    logger.info(f"Done. Grid video: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare checkpoints against GT on a grid with yellow error lines."
    )
    parser.add_argument("--gt-h5", type=Path, required=True, help="Ground-truth h5.")
    parser.add_argument(
        "--video", type=Path, required=True, dest="video_path",
        help="Full path to the GT video.",
    )
    parser.add_argument("--rows", type=int, default=None, help="Grid rows.")
    parser.add_argument("--cols", type=int, default=None, help="Grid cols.")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fps", type=float, default=2.0, help="Output video fps.")
    parser.add_argument("--output", type=Path, default=None, dest="output_path")
    args = parser.parse_args()

    run(
        gt_h5=args.gt_h5,
        video_path=args.video_path,
        rows=args.rows,
        cols=args.cols,
        device=args.device,
        batch_size=args.batch_size,
        fps=args.fps,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
