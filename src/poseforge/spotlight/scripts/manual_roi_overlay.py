#!/usr/bin/env python3
"""Overlay ROI segmentation on muscle frames and animate ROI traces.

Expected inputs in --base-dir:
- one TIFF matching muscle_frame*.tif with shape (T, H, W)
- one ROI segmentation TIFF with shape (T, H, W)
- steps.tif with shape (T, H, W)

The script reverses all stacks in time before plotting to match Napari ordering.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
from matplotlib.animation import FFMpegWriter, FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tifffile
from scipy.ndimage import median_filter

# The user-provided map included duplicate key 2; key 3 is used for RH.
DEFAULT_ROI_NAME_MAP = {1: "RF", 2: "RM", 3: "RH", 5: "LH", 6: "LM", 7: "LF"}


def natural_sort_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def load_twh(path: Path, name: str) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim != 3:
        raise ValueError(f"{name} must be a 3D stack (T, H, W), got {arr.shape} from {path}")
    return arr


def find_muscle_path(base_dir: Path) -> Path:
    candidates = sorted(base_dir.glob("muscle_frame*.tif"), key=natural_sort_key)
    if len(candidates) == 0:
        raise FileNotFoundError(f"No muscle frame file matching muscle_frame*.tif in {base_dir}")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple muscle frame files found in {base_dir}: {[p.name for p in candidates]}. "
            "Keep exactly one."
        )
    return candidates[0]


def find_steps_path(base_dir: Path) -> Path:
    steps_path = base_dir / "steps.tif"
    if not steps_path.exists():
        raise FileNotFoundError(f"Missing steps file: {steps_path}")
    return steps_path


def find_roi_path(base_dir: Path, muscle_path: Path, steps_path: Path) -> Path:
    explicit_candidates = [
        base_dir / "rois.tif",
        base_dir / "roi.tif",
        base_dir / "lt_muscles.tif",
        base_dir / "lt_muscle.tif",
        base_dir / "segmentation.tif",
        base_dir / "segmentation_map.tif",
    ]
    existing_explicit = [p for p in explicit_candidates if p.exists()]
    if len(existing_explicit) == 1:
        return existing_explicit[0]
    if len(existing_explicit) > 1:
        raise ValueError(
            f"Multiple explicit ROI candidates found in {base_dir}: {[p.name for p in existing_explicit]}. "
            "Keep exactly one ROI file."
        )

    all_tifs = sorted(base_dir.glob("*.tif"), key=natural_sort_key)
    excluded = {muscle_path.name, steps_path.name}
    remaining = [p for p in all_tifs if p.name not in excluded]
    if len(remaining) == 0:
        raise FileNotFoundError(
            f"Could not infer ROI segmentation TIFF in {base_dir}. "
            "Add one of: rois.tif, roi.tif, lt_muscles.tif, segmentation.tif"
        )
    if len(remaining) > 1:
        raise ValueError(
            f"Ambiguous ROI TIFFs in {base_dir}: {[p.name for p in remaining]}. "
            "Keep exactly one ROI segmentation file."
        )
    return remaining[0]


def normalize_stack_percentiles(stack: np.ndarray, vmin_pct: float, vmax_pct: float) -> tuple[np.ndarray, float, float]:
    flat = stack.astype(np.float64).ravel()
    vmin = float(np.percentile(flat, vmin_pct))
    vmax = float(np.percentile(flat, vmax_pct))
    if vmax <= vmin:
        raise ValueError(
            f"Invalid percentile range from image values: p{vmin_pct}={vmin}, p{vmax_pct}={vmax}."
        )
    norm = (stack.astype(np.float64) - vmin) / (vmax - vmin)
    return np.clip(norm, 0.0, 1.0), vmin, vmax


def load_roi_name_map(path: Path | None) -> dict[int, str]:
    raw_map = DEFAULT_ROI_NAME_MAP if path is None else json.loads(path.read_text())
    parsed: dict[int, str] = {}
    for k, v in raw_map.items():
        try:
            label = int(k)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ROI map key {k!r} is not integer-like") from exc
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"ROI map value for label {label} must be a non-empty string, got {v!r}")
        parsed[label] = v.strip()
    return parsed


def validate_label_stack(roi_stack: np.ndarray, roi_name_map: dict[int, str]) -> np.ndarray:
    if not np.issubdtype(roi_stack.dtype, np.integer):
        rounded = np.rint(roi_stack)
        if not np.allclose(roi_stack, rounded):
            raise ValueError("ROI stack contains non-integer label values")
        roi_stack = rounded.astype(np.int32)

    present_labels = np.unique(roi_stack)
    present_labels = present_labels[present_labels != 0]
    if present_labels.size == 0:
        raise ValueError("ROI stack has no non-zero labels")

    missing = sorted(int(l) for l in present_labels if int(l) not in roi_name_map)
    if missing:
        raise ValueError(
            f"ROI labels found in segmentation but missing from ROI name map: {missing}. "
            f"Map contains labels {sorted(roi_name_map.keys())}."
        )
    return present_labels.astype(int)


def compute_roi_traces(m_images: np.ndarray, roi_stack: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n_frames = m_images.shape[0]
    traces = np.full((len(labels), n_frames), np.nan, dtype=np.float64)
    for idx, label in enumerate(labels):
        mask_t = roi_stack == label
        if not np.any(mask_t):
            raise ValueError(f"ROI label {label} has no pixels in the stack")
        for t in range(n_frames):
            mask = mask_t[t]
            if np.any(mask):
                mask_pixels = m_images[t][mask]
                # take the 10 brightest pixels in the ROI for the trace value, to be more robust to noise and partial labeling
                top_pixels = mask_pixels #np.partition(mask_pixels, -5)[-5:]
                traces[idx, t] = float(np.std(top_pixels))
    return traces


def compute_step_events(steps_stack: np.ndarray, roi_stack: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n_labels = len(labels)
    n_frames = roi_stack.shape[0]
    step_events = np.zeros((n_labels, n_frames), dtype=bool)

    step_values = np.unique(steps_stack)
    non_zero_steps = step_values[step_values != 0]
    if non_zero_steps.size == 0:
        return step_events

    label_set = {int(v) for v in labels.tolist()}
    step_label_set = {int(v) for v in non_zero_steps.tolist()}
    label_encoded = step_label_set.issubset(label_set) and len(step_label_set) > 1

    for idx, label in enumerate(labels):
        if label_encoded:
            step_events[idx] = np.any(steps_stack == label, axis=(1, 2))
        else:
            # Binary (or non-label) steps: assign the event to ROI(s) that overlap the step pixels.
            overlap = (steps_stack > 0) & (roi_stack == label)
            step_events[idx] = np.any(overlap, axis=(1, 2))
    return step_events


def build_contour_rgba(roi_frame: np.ndarray, labels: np.ndarray, colors: np.ndarray) -> np.ndarray:
    h, w = roi_frame.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for idx, label in enumerate(labels):
        mask = roi_frame == label
        if not np.any(mask):
            continue
        mask_u8 = mask.astype(np.uint8) * 255
        # dilate the mask before getting the contour
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(mask_u8, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            overlay,
            contours,
            contourIdx=-1,
            color=(float(colors[idx, 0]), float(colors[idx, 1]), float(colors[idx, 2]), 1.0),
            thickness=2,
        )
    return overlay


def save_traces_csv(out_csv: Path, traces: np.ndarray, labels: np.ndarray, roi_names: list[str], fps: float) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(traces.shape[1]) / fps
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s"] + [f"{name} (label={label})" for name, label in zip(roi_names, labels)])
        for i in range(traces.shape[1]):
            writer.writerow([i, float(t[i]), *traces[:, i].tolist()])


def render_animation(
    norm_images: np.ndarray,
    roi_stack: np.ndarray,
    labels: np.ndarray,
    roi_names: list[str],
    traces: np.ndarray,
    step_events: np.ndarray,
    anim_fps: float,
    recording_fps: float,
    alpha: float,
    out_path: Path,
) -> None:
    n_frames = norm_images.shape[0]
    times = np.arange(n_frames) / recording_fps

    cmap = plt.get_cmap("tab10")
    colors = np.array([cmap(i % 10)[:3] for i in range(len(labels))], dtype=np.float32)

    # Reorder labels/names into a 2-column, 3-row layout:
    # left column: RF, RM, RH; right column: LF, LM, LH
    desired_order = [("RF", "LF"), ("RM", "LM"), ("RH", "LH")]

    fig = plt.figure(figsize=(10, 10))

    gs = fig.add_gridspec(4, 2, height_ratios=[7, 1, 1, 1])
    ax_img = fig.add_subplot(gs[0, :])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1], sharex=ax_left)
    ax_left_mid = fig.add_subplot(gs[2, 0], sharex=ax_left)
    ax_right_mid = fig.add_subplot(gs[2, 1], sharex=ax_right)
    ax_left_bot = fig.add_subplot(gs[3, 0], sharex=ax_left)
    ax_right_bot = fig.add_subplot(gs[3, 1], sharex=ax_right)
    ax_trace_grid = [ax_left, ax_right, ax_left_mid, ax_right_mid, ax_left_bot, ax_right_bot]

    # Display image and overlay
    image_artist = ax_img.imshow(norm_images[0], cmap="gray", vmin=0.0, vmax=1.0)
    overlay_artist = ax_img.imshow(build_contour_rgba(roi_stack[0], labels, colors))
    ax_img.axis("off")
    # For each desired subplot, plot the trace if label present; otherwise leave empty with label
    label_to_color = {int(l): colors[i] for i, l in enumerate(labels)}
    label_to_trace = {int(l): traces[i] for i, l in enumerate(labels)}
    label_to_steps = {int(l): step_events[i] for i, l in enumerate(labels)}

    time_span = max(float(times[-1] - times[0]), 1e-6)
    x_pad = 0.0 * time_span
    x_min = float(times[0] - x_pad)
    x_max = float(times[-1] + x_pad)

    # map desired names to labels if available
    name_label_map: dict[str, int | None] = {}
    for name in [pair[0] for pair in desired_order] + [pair[1] for pair in desired_order]:
        found_label = None
        for lab, nm in zip(labels, roi_names):
            if nm.lower() == name.lower():
                found_label = int(lab)
                break
        name_label_map[name] = found_label

    cursor_line_axes: list[plt.Axes] = []

    # plot each subplot: left and right columns share x independently
    for row_idx, (left_name, right_name) in enumerate(desired_order):
        for col_idx, name in enumerate((left_name, right_name)):
            ax = ax_trace_grid[row_idx * 2 + col_idx]
            ax.spines["left"].set_visible(True)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            lab = name_label_map[name]
            if lab is None:
                ax.text(0.5, 0.5, f"{name}\n(missing)", transform=ax.transAxes, va='center', ha='center', fontsize=9)
                # hide all spines and ticks for empty subplots
                ax.spines["left"].set_visible(False)
                ax.tick_params(labelleft=False)
            else:
                trace = label_to_trace[lab]
                color = label_to_color[lab]
                ax.plot(times, trace, color=color, linewidth=1.6)
                # place asterisks under the curve using each plot's own y-range
                step_idx = np.where(label_to_steps[lab])[0]
                y_min = float(np.nanmin(trace))
                y_max = float(np.nanmax(trace))
                if not np.isfinite(y_min) or not np.isfinite(y_max):
                    y_min, y_max = 0.0, 1.0
                if y_max <= y_min:
                    y_max = y_min + 1e-3
                y_span = max(y_max - y_min, 1e-6)
                y_pad = 0.1 * y_span
                y_lo = y_min - y_pad
                y_hi = y_max + y_pad
                y_star = y_lo + 0.15 * y_span
                if step_idx.size > 0:
                    ax.scatter(times[step_idx], np.full(step_idx.shape, y_star), marker='*', s=90, color=color, zorder=6)
                ax.text(0.05, 0.90, name, transform=ax.transAxes, va='top', ha='left', fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_lo, y_hi)
                if ax not in cursor_line_axes:
                    cursor_line_axes.append(ax)

                # show y-axis on every plot
                ax.tick_params(labelleft=True)

            # show x-axis only on the bottom row (plots 3 and 6 in column-major order)
            if row_idx == len(desired_order) - 1:
                ax.set_xlabel('Time (s)')
                ax.tick_params(labelbottom=True)
            else:
                ax.set_xlabel('')
                ax.tick_params(labelbottom=False)

            # keep the frame minimal but readable
            if row_idx == 2:
                ax.spines["bottom"].set_visible(True)
            else:
                ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="both", which="both", length=0)

    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.1, wspace=0.04, left=0.1, right=0.985, top=0.95, bottom=0.05)

    # create vertical cursor lines only for axes that actually have data
    cursor_lines = [ax.axvline(times[0], color="black", linestyle="--", linewidth=1.0) for ax in cursor_line_axes]

    def update(frame_idx: int):
        image_artist.set_data(norm_images[frame_idx])
        overlay_artist.set_data(build_contour_rgba(roi_stack[frame_idx], labels, colors))
        for cl in cursor_lines:
            cl.set_xdata([times[frame_idx], times[frame_idx]])
        return tuple([image_artist, overlay_artist] + cursor_lines)

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000.0 / anim_fps,
        blit=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=anim_fps, bitrate=2000)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create overlay animation of manual ROI segmentation on muscle frames"
    )
    parser.add_argument("--base-dir", type=Path, required=True, help="Folder with muscle_frame*.tif, ROI tif, and steps.tif")
    parser.add_argument("--roi-name-map", type=Path, default=None, help="Optional JSON map of ROI label to name")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output folder (default: <base-dir>/manual_roi_overlay)")
    parser.add_argument("--fps", type=float, default=1.0, help="Animation framerate")
    parser.add_argument("--alpha", type=float, default=0.15, help="ROI overlay alpha")
    parser.add_argument("--vmin-pct", type=float, default=50.0, help="Lower normalization percentile")
    parser.add_argument("--vmax-pct", type=float, default=99.9, help="Upper normalization percentile")
    parser.add_argument(
        "--time-order",
        choices=("reverse", "normal"),
        default="reverse",
        help="Playback order for the time axis and stacks",
    )
    parser.add_argument("--outfile_suffix", type=str, default=None, help="Optional suffix to add to output filenames")
    args = parser.parse_args()

    if not args.base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {args.base_dir}")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {args.alpha}")
    if args.vmax_pct <= args.vmin_pct:
        raise ValueError(f"vmax-pct must be > vmin-pct, got {args.vmin_pct} and {args.vmax_pct}")

    muscle_path = find_muscle_path(args.base_dir)
    steps_path = find_steps_path(args.base_dir)
    roi_path = find_roi_path(args.base_dir, muscle_path, steps_path)

    print(f"Muscle frames: {muscle_path}")
    print(f"ROI labels:   {roi_path}")
    print(f"Step labels:  {steps_path}")

    muscle_stack = load_twh(muscle_path, "muscle frame stack")
    roi_stack = load_twh(roi_path, "ROI stack")
    steps_stack = load_twh(steps_path, "steps stack")

    if muscle_stack.shape != roi_stack.shape:
        raise ValueError(
            f"ROI stack shape {roi_stack.shape} does not match muscle stack shape {muscle_stack.shape}"
        )
    if steps_stack.shape != muscle_stack.shape:
        raise ValueError(
            f"steps stack shape {steps_stack.shape} does not match muscle stack shape {muscle_stack.shape}"
        )

    if args.time_order == "reverse":
        muscle_stack = muscle_stack[::-1]
        roi_stack = roi_stack[::-1]
        steps_stack = steps_stack[::-1]

    roi_name_map = load_roi_name_map(args.roi_name_map)
    labels = validate_label_stack(roi_stack, roi_name_map)
    roi_names = [roi_name_map[int(label)] for label in labels]

    norm_images, vmin, vmax = normalize_stack_percentiles(muscle_stack.copy(), args.vmin_pct, args.vmax_pct)
    # Compute traces on raw muscle images, not normalized
    # Apply median filter to each frame to reduce noise and make the traces more stable, since the ROIs are often sparse and may have some mislabeled pixels
    # m_images = np.empty_like(muscle_stack, dtype=np.float32)
    # for t in range(muscle_stack.shape[0]):
    #     m_images[t] = median_filter(muscle_stack[t].astype(np.float32), size=5)
    traces = compute_roi_traces(muscle_stack, roi_stack, labels)
    step_events = compute_step_events(steps_stack, roi_stack, labels)

    out_dir = args.output_dir if args.output_dir is not None else (args.base_dir / "manual_roi_overlay")
    out_mp4 = out_dir / f"overlay_animation_{args.time_order}.mp4"
    if args.outfile_suffix:
        out_mp4 = out_dir / f"overlay_animation_{args.time_order}_{args.outfile_suffix}.mp4"
    out_csv = out_dir / "traces.csv"

    import yaml
    metadata_file = Path(args.base_dir).parent.parent.parent / "metadata/experiment_parameters.yaml"
    if not metadata_file.exists():
        print(f"Warning: experiment_parameters.yaml not found at {metadata_file}. Assuming recording FPS = animation FPS = {args.fps}")
        recording_fps = 25
    else:
        with metadata_file.open() as f:
            exp_params = yaml.safe_load(f)
        recording_fps = exp_params["behavior_fps"]/exp_params["muscle_sync_ratio"]

    save_traces_csv(out_csv, traces, labels, roi_names, args.fps)
    render_animation(
        norm_images=norm_images,
        roi_stack=roi_stack,
        labels=labels,
        roi_names=roi_names,
        traces=traces,
        step_events=step_events,
        anim_fps=args.fps,
        recording_fps=recording_fps,
        alpha=args.alpha,
        out_path=out_mp4,
    )

    print(f"Normalization percentiles: p{args.vmin_pct}={vmin:.4f}, p{args.vmax_pct}={vmax:.4f}")
    print(f"Time order:      {args.time_order}")
    print(f"Saved animation: {out_mp4}")
    print(f"Saved traces:    {out_csv}")


if __name__ == "__main__":
    main()