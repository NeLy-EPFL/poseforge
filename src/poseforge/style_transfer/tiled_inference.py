import logging
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw
from tqdm import trange
from torchvision.transforms.functional import to_pil_image
from pvio.io import read_frames_from_video, write_frames_to_video

from poseforge.style_transfer.cut_inference import InferencePipeline
from poseforge.util.sys import clear_memory_cache


def _normalize_weight_type(weight_type: str) -> str:
    normalized_weight_type = weight_type.strip().lower()
    if normalized_weight_type == "guassian":
        normalized_weight_type = "gaussian"

    if normalized_weight_type not in {"uniform", "cosine", "gaussian", "pyramid"}:
        raise ValueError(
            "weight_type must be one of 'uniform', 'cosine', 'gaussian', or 'pyramid', "
            f"got {weight_type!r}"
        )

    return normalized_weight_type


def _make_tile_weight_map(patch_h: int, patch_w: int, weight_type: str) -> np.ndarray:
    """Create a per-patch blending weight map with positive support everywhere."""
    if patch_h <= 0:
        raise ValueError(f"patch_h must be positive, got {patch_h}")
    if patch_w <= 0:
        raise ValueError(f"patch_w must be positive, got {patch_w}")

    weight_type = _normalize_weight_type(weight_type)

    if weight_type == "uniform":
        weight_map = np.ones((patch_h, patch_w), dtype=np.float32)
    else:
        if patch_h == 1:
            y_coords = np.zeros(1, dtype=np.float32)
        else:
            y_coords = np.linspace(0.0, 1.0, patch_h, dtype=np.float32)

        if patch_w == 1:
            x_coords = np.zeros(1, dtype=np.float32)
        else:
            x_coords = np.linspace(0.0, 1.0, patch_w, dtype=np.float32)

        if weight_type == "cosine":
            y_weights = 0.5 - 0.5 * np.cos(2.0 * np.pi * y_coords)
            x_weights = 0.5 - 0.5 * np.cos(2.0 * np.pi * x_coords)
            weight_map = np.outer(y_weights, x_weights)
        elif weight_type == "gaussian":
            y_centered = np.linspace(-1.0, 1.0, patch_h, dtype=np.float32)
            x_centered = np.linspace(-1.0, 1.0, patch_w, dtype=np.float32)
            sigma = 0.35
            y_weights = np.exp(-0.5 * (y_centered / sigma) ** 2)
            x_weights = np.exp(-0.5 * (x_centered / sigma) ** 2)
            weight_map = np.outer(y_weights, x_weights)
        elif weight_type == "pyramid":
            y_weights = 1.0 - np.abs(2.0 * y_coords - 1.0)
            x_weights = 1.0 - np.abs(2.0 * x_coords - 1.0)
            weight_map = np.outer(y_weights, x_weights)
        else:
            raise AssertionError(f"Unhandled weight_type: {weight_type}")

    weight_map = np.clip(weight_map, 1e-6, None)
    weight_map /= float(weight_map.max())
    return weight_map.astype(np.float32)[..., None]


def _compute_half_overlap_tile_starts(
    full_size: int, tile_size: int, phase: int = 0
) -> list[int]:
    """Compute tile start indices for a fixed 50% overlap grid with optional phase shift.

    The grid starts at (phase - tile_size) and continues in strides of half a tile,
    shifted by the given phase. This ensures every pixel along the axis receives
    exactly 2 overlapping tile contributions, even with random phase shifts.
    """
    if full_size <= 0:
        raise ValueError(f"full_size must be positive, got {full_size}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if tile_size % 2 != 0:
        raise ValueError(
            "tile_size must be even to use an exact 50% overlap strategy, "
            f"got {tile_size}"
        )

    stride = tile_size // 2
    if not (0 <= phase < stride):
        raise ValueError(
            f"phase must be in [0, {stride}), got {phase}"
        )

    first_start = phase - tile_size
    last_start = full_size - stride + phase
    starts = np.arange(first_start, last_start + 1, stride, dtype=np.int32)
    return starts.tolist()


def _mirror_indices(indices: np.ndarray, size: int) -> np.ndarray:
    """Map potentially out-of-bounds indices into a mirrored coordinate set."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if size == 1:
        return np.zeros_like(indices, dtype=np.int32)

    period = 2 * (size - 1)
    mirrored = np.abs(indices) % period
    mirrored = np.where(mirrored >= size, period - mirrored, mirrored)
    return mirrored.astype(np.int32)


def _extract_mirrored_tile(
    frame: np.ndarray,
    start_y: int,
    start_x: int,
    tile_size: int,
) -> np.ndarray:
    """Extract a fixed-size tile using mirrored coordinates outside the frame."""
    height, width, _ = frame.shape
    y_indices = _mirror_indices(np.arange(start_y, start_y + tile_size), height)
    x_indices = _mirror_indices(np.arange(start_x, start_x + tile_size), width)
    return frame[y_indices[:, None], x_indices[None, :]]


def _tile_debug_color(color_index: int) -> tuple[int, int, int]:
    """Return one of 4 high-contrast outline colors based on index mod 4."""
    colors = [
        (255, 64, 64),    # red
        (64, 192, 255),   # cyan
        (64, 255, 64),    # green
        (255, 192, 64),   # orange
    ]
    return colors[color_index % 4]


def _draw_tile_debug_overlay(
    frame: np.ndarray,
    specs: list[tuple[int, int, int, int]],
    tile_size: int,
) -> np.ndarray:
    """Draw tile outlines on top of a zero-padded frame for debugging."""
    min_start_y = min(y for y, _x, _patch_h, _patch_w in specs)
    min_start_x = min(x for _y, x, _patch_h, _patch_w in specs)
    max_end_y = max(y + tile_size for y, _x, _patch_h, _patch_w in specs)
    max_end_x = max(x + tile_size for _y, x, _patch_h, _patch_w in specs)

    height, width = frame.shape[:2]

    pad_top = max(0, -min_start_y)
    pad_left = max(0, -min_start_x)
    pad_bottom = max(0, max_end_y - height)
    pad_right = max(0, max_end_x - width)

    padded_h = height + pad_top + pad_bottom
    padded_w = width + pad_left + pad_right
    padded = np.zeros((padded_h, padded_w, frame.shape[2]), dtype=frame.dtype)
    padded[pad_top : pad_top + height, pad_left : pad_left + width] = frame

    overlay = Image.fromarray(padded)
    draw = ImageDraw.Draw(overlay)

    for tile_index, (y, x, _patch_h, _patch_w) in enumerate(specs):
        x0 = x + pad_left
        y0 = y + pad_top
        x1 = x0 + tile_size - 1
        y1 = y0 + tile_size - 1

        draw.rectangle(
            [(x0, y0), (x1, y1)],
            outline=_tile_debug_color(tile_index),
            width=3,
        )

    return np.asarray(overlay)


def _tile_specs(
    height: int, width: int, tile_size: int, y_phase: int = 0, x_phase: int = 0
) -> list[tuple[int, int, int, int]]:
    """Return tile specs as (y, x, patch_h, patch_w) with optional phase shifts."""
    y_starts = _compute_half_overlap_tile_starts(height, tile_size, phase=y_phase)
    x_starts = _compute_half_overlap_tile_starts(width, tile_size, phase=x_phase)

    specs: list[tuple[int, int, int, int]] = []
    for y in y_starts:
        for x in x_starts:
            patch_h = min(tile_size, max(0, height - y))
            patch_w = min(tile_size, max(0, width - x))
            specs.append((y, x, patch_h, patch_w))
    return specs


def stylize_frame_tiled(
    inference_pipeline: InferencePipeline,
    frame: np.ndarray,
    patch_batch_size: int,
    weight_type: str = "uniform",
    debug_mode: bool = False,
    y_phase: int = 0,
    x_phase: int = 0,
) -> np.ndarray:
    """Run style transfer on a full frame by tiled patch inference.

    Tiles are extracted on a 50% overlap grid and mirrored at the borders.
    Phases allow randomization of seam locations while preserving 2x2 coverage.
    """
    if frame.ndim != 3:
        raise ValueError(f"Expected frame with shape (H, W, C), got {frame.shape}")

    if patch_batch_size <= 0:
        raise ValueError(f"patch_batch_size must be > 0, got {patch_batch_size}")

    height, width, _ = frame.shape
    tile_size = inference_pipeline.image_side_length
    weight_type = _normalize_weight_type(weight_type)
    specs = _tile_specs(height, width, tile_size, y_phase=y_phase, x_phase=x_phase)
    weight_map = _make_tile_weight_map(tile_size, tile_size, weight_type)

    # Build PIL patch list once
    patches_pil: list[Image.Image] = []
    for y, x, patch_h, patch_w in specs:
        patch = _extract_mirrored_tile(frame, y, x, tile_size)
        patches_pil.append(to_pil_image(patch))

    out_sum: np.ndarray | None = None
    out_count = np.zeros((height, width, 1), dtype=np.float32)

    for i in range(0, len(specs), patch_batch_size):
        batch_specs = specs[i : i + patch_batch_size]
        batch_pil = patches_pil[i : i + patch_batch_size]
        batch_out = inference_pipeline.infer(batch_pil)

        if out_sum is None:
            out_channels = batch_out.shape[-1]
            out_sum = np.zeros((height, width, out_channels), dtype=np.float32)

        for j, (y, x, patch_h, patch_w) in enumerate(batch_specs):
            out_patch = batch_out[j].astype(np.float32)
            y0 = max(0, y)
            x0 = max(0, x)
            y1 = min(height, y + tile_size)
            x1 = min(width, x + tile_size)

            tile_y0 = y0 - y
            tile_x0 = x0 - x
            tile_y1 = tile_y0 + (y1 - y0)
            tile_x1 = tile_x0 + (x1 - x0)

            tile_weights = weight_map[tile_y0:tile_y1, tile_x0:tile_x1]
            out_sum[y0:y1, x0:x1] += out_patch[tile_y0:tile_y1, tile_x0:tile_x1] * tile_weights
            out_count[y0:y1, x0:x1] += tile_weights

    assert out_sum is not None
    output = np.zeros_like(out_sum, dtype=np.float32)
    valid_locations = out_count > 0
    np.divide(out_sum, out_count, out=output, where=valid_locations)
    output = output.clip(0, 255).astype(np.uint8)

    if debug_mode:
        output = _draw_tile_debug_overlay(output, specs, tile_size)
    return output


def process_simulation_tiled(
    inference_pipeline: InferencePipeline,
    input_video_path: Path,
    output_video_path: Path,
    weight_type: str,
    randomize_seams: bool,
    seed: int | None,
    patch_batch_size: int | None,
    debug_mode: bool = False,
    progress_bar: bool = True,
    clear_memory_cache_after: bool = True,
) -> None:
    """Run tiled style transfer over all frames in a simulation video.
    
    If randomize_seams is True, each frame gets a random y_phase and x_phase
    sampled uniformly from [0, tile_size/2), giving different seam patterns
    per image while maintaining exact 2x2 tile coverage.
    """
    input_frames, fps = read_frames_from_video(input_video_path)

    if len(input_frames) == 0:
        raise RuntimeError(f"Input video has no frames: {input_video_path}")

    if patch_batch_size is None:
        tile_size = inference_pipeline.image_side_length
        patch_input_shape = (tile_size, tile_size, input_frames[0].shape[-1])
        patch_batch_size = inference_pipeline.detect_max_batch_size(
            patch_input_shape,
            start=1,
            end=1024,
        )

    if randomize_seams:
        print("USING RANDOMIZED SEAMS WITH SEED", seed)
        rng = np.random.RandomState(seed)
        tile_size = inference_pipeline.image_side_length
        stride = tile_size // 2
    else:
        rng = None
        stride = None

    output_frames = []
    for i in trange(len(input_frames), disable=not progress_bar):
        if randomize_seams and rng is not None:
            y_phase = rng.randint(0, stride)
            x_phase = rng.randint(0, stride)
        else:
            y_phase = 0
            x_phase = 0

        frame_out = stylize_frame_tiled(
            inference_pipeline,
            input_frames[i],
            patch_batch_size=patch_batch_size,
            weight_type=weight_type,
            debug_mode=debug_mode,
            y_phase=y_phase,
            x_phase=x_phase,
        )
        output_frames.append(frame_out)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    write_frames_to_video(output_video_path, output_frames, fps=fps)

    del input_frames, output_frames
    if clear_memory_cache_after:
        clear_memory_cache()

    logging.info(f"Saved tiled inference output to {output_video_path}")
