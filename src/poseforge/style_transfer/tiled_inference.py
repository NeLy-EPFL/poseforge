import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import trange
from torchvision.transforms.functional import to_pil_image
from pvio.io import read_frames_from_video, write_frames_to_video

from poseforge.style_transfer.cut_inference import InferencePipeline
from poseforge.util.sys import clear_memory_cache


def _compute_even_tile_starts(full_size: int, tile_size: int) -> list[int]:
    """Compute tile start indices with minimal tile count and even spacing.

    Uses the minimum number of tiles needed to cover the axis, then spaces
    starts linearly from 0 to (full_size - tile_size) so overlaps are as even
    as possible.
    """
    if full_size <= 0:
        raise ValueError(f"full_size must be positive, got {full_size}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")

    if full_size <= tile_size:
        return [0]

    num_tiles = int(np.ceil(full_size / tile_size))
    last_start = full_size - tile_size
    starts = np.linspace(0, last_start, num_tiles)
    starts = np.rint(starts).astype(np.int32)

    # Guard against numerical quirks and enforce boundary coverage
    starts[0] = 0
    starts[-1] = last_start

    # Ensure monotonic non-decreasing starts while preserving last boundary
    starts = np.maximum.accumulate(starts)
    starts[-1] = last_start

    return starts.tolist()


def _tile_specs(height: int, width: int, tile_size: int) -> list[tuple[int, int, int, int]]:
    """Return tile specs as (y, x, patch_h, patch_w)."""
    y_starts = _compute_even_tile_starts(height, tile_size)
    x_starts = _compute_even_tile_starts(width, tile_size)

    specs: list[tuple[int, int, int, int]] = []
    for y in y_starts:
        for x in x_starts:
            patch_h = min(tile_size, height - y)
            patch_w = min(tile_size, width - x)
            specs.append((y, x, patch_h, patch_w))
    return specs


def stylize_frame_tiled(
    inference_pipeline: InferencePipeline,
    frame: np.ndarray,
    patch_batch_size: int,
) -> np.ndarray:
    """Run style transfer on a full frame by tiled patch inference.

    Overlap handling is done via simple arithmetic mean in overlapping pixels.
    """
    if frame.ndim != 3:
        raise ValueError(f"Expected frame with shape (H, W, C), got {frame.shape}")

    if patch_batch_size <= 0:
        raise ValueError(f"patch_batch_size must be > 0, got {patch_batch_size}")

    height, width, _ = frame.shape
    tile_size = inference_pipeline.image_side_length
    specs = _tile_specs(height, width, tile_size)

    # Build PIL patch list once
    patches_pil: list[Image.Image] = []
    for y, x, patch_h, patch_w in specs:
        patch = frame[y : y + patch_h, x : x + patch_w]
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
            out_sum[y : y + patch_h, x : x + patch_w] += out_patch[:patch_h, :patch_w]
            out_count[y : y + patch_h, x : x + patch_w] += 1.0

    assert out_sum is not None
    out_count = np.maximum(out_count, 1.0)
    output = (out_sum / out_count).clip(0, 255).astype(np.uint8)
    return output


def process_simulation_tiled(
    inference_pipeline: InferencePipeline,
    input_video_path: Path,
    output_video_path: Path,
    patch_batch_size: int | None = None,
    progress_bar: bool = True,
    clear_memory_cache_after: bool = True,
) -> None:
    """Run tiled style transfer over all frames in a simulation video."""
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

    output_frames = []
    for i in trange(len(input_frames), disable=not progress_bar):
        frame_out = stylize_frame_tiled(
            inference_pipeline,
            input_frames[i],
            patch_batch_size=patch_batch_size,
        )
        output_frames.append(frame_out)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    write_frames_to_video(output_video_path, output_frames, fps=fps)

    del input_frames, output_frames
    if clear_memory_cache_after:
        clear_memory_cache()

    logging.info(f"Saved tiled inference output to {output_video_path}")
