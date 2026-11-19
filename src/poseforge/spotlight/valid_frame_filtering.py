"""Keypoint-independent arena-edge detection and temporal flip filtering.

These two filters refine which aligned behavior frames are usable for downstream
image translation (CUT) training:

* ``corner_wall_flags`` flags frames in which the arena wall is visible, by mapping
  the four corners of the aligned frame to arena millimetres and checking whether
  any corner lies outside the arena or inside a wall. Unlike the keypoint-based
  edge test in :mod:`poseforge.production.spotlight.filtering`, this uses the
  stored per-frame alignment transform (finite for every frame), so wall frames on
  which the pose keypoints are missing are still caught.

* ``suppress_short_runs`` and ``dilate_flip_blocks`` turn the raw per-frame flip
  predictions into a temporally smoothed mask: isolated single-frame flips are
  removed and genuine flip blocks are padded on both sides, because the frames
  immediately around a flip are usually mid-roll and also unusable.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

# Order of the affine inputs in the behavior calibration yaml.
AFFINE_INPUT_KEYS = ["stage_pos_x", "stage_pos_y", "pixel_pos_x", "pixel_pos_y", "bias"]


def calibration_weights(calibration_yaml_path: Path):
    """Read the (stage, raw pixel) -> physical arena-mm affine weights."""
    params = yaml.safe_load(Path(calibration_yaml_path).read_text())[
        "stage_and_pixel_to_physical"
    ]
    weights_x = np.array([params["physical_pos_x"][k] for k in AFFINE_INPUT_KEYS])
    weights_y = np.array([params["physical_pos_y"][k] for k in AFFINE_INPUT_KEYS])
    return weights_x, weights_y


def raw_pixel_to_physical(stage_xy, pixel_xy, weights_x, weights_y):
    """Map raw camera pixels to arena millimetres for one frame's stage position.

    Args:
        stage_xy: ``(2,)`` stage position (mm) for the frame.
        pixel_xy: ``(..., 2)`` raw camera pixel coordinates.
        weights_x, weights_y: affine weights from :func:`calibration_weights`.

    Returns:
        ``(..., 2)`` physical arena-mm coordinates.
    """
    pixel_xy = np.asarray(pixel_xy, float)
    leading_shape = pixel_xy.shape[:-1]
    flat = pixel_xy.reshape(-1, 2)
    stage_repeated = np.broadcast_to(stage_xy, (len(flat), 2))
    inputs = np.column_stack([stage_repeated, flat, np.ones(len(flat))])
    physical = np.column_stack([inputs @ weights_x, inputs @ weights_y])
    return physical.reshape(*leading_shape, 2)


def load_arena(arena_dir: Path):
    """Load the arena accessible-area mask and distance-to-wall rasters.

    Returns ``(accessible, dist_to_wall_mm, depth_into_wall_mm, raster_resolution,
    arena_x_mm, arena_y_mm)``. ``accessible`` is a boolean raster of the reachable
    area; the two distance rasters give, per raster cell, the millimetre distance to
    the nearest wall (inside) and the depth into the wall (outside).
    """
    arena_dir = Path(arena_dir)
    meta = yaml.safe_load((arena_dir / "metadata.yaml").read_text())
    raster_resolution = float(meta["active_area_raster_resolution"])
    arena_x_mm, arena_y_mm = (float(v) for v in meta["arena_dim"])
    image = cv2.imread(str(arena_dir / "active_area.png"), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read {arena_dir / 'active_area.png'}")
    accessible = image > 127
    # The arena perimeter counts as a wall: distance into the accessible region is
    # measured with the border treated as inaccessible.
    padded_accessible = np.pad(accessible, 1, mode="constant", constant_values=False)
    dist_to_wall_mm = (
        ndimage.distance_transform_edt(padded_accessible)[1:-1, 1:-1] * raster_resolution
    )
    depth_into_wall_mm = ndimage.distance_transform_edt(~accessible) * raster_resolution
    return (
        accessible,
        dist_to_wall_mm,
        depth_into_wall_mm,
        raster_resolution,
        arena_x_mm,
        arena_y_mm,
    )


def corner_margins(
    physical_corners,
    accessible,
    depth_into_wall_mm,
    dist_to_wall_mm,
    raster_resolution,
    arena_x_mm,
    arena_y_mm,
):
    """Signed millimetre margin to the wall boundary for each of 4 corners.

    Positive = inside the accessible area (distance to the nearest wall); negative =
    past the boundary (outside the arena, or inside a wall).
    """
    height, width = accessible.shape
    margins = []
    for x, y in physical_corners:
        if not (np.isfinite(x) and np.isfinite(y)):
            margins.append(np.nan)
            continue
        if x < 0 or y < 0 or x > arena_x_mm or y > arena_y_mm:
            delta_x = max(-x, x - arena_x_mm, 0.0)
            delta_y = max(-y, y - arena_y_mm, 0.0)
            margins.append(-float(np.hypot(delta_x, delta_y)) - 0.001)
            continue
        col = int(np.clip(round(x / raster_resolution), 0, width - 1))
        row = int(np.clip(round(y / raster_resolution), 0, height - 1))
        if accessible[row, col]:
            margins.append(float(dist_to_wall_mm[row, col]))
        else:
            margins.append(-float(depth_into_wall_mm[row, col]))
    return np.array(margins)


def corner_wall_flags(trial_dir, arena_dir, frame_size, clearance_mm):
    """Flag frames whose footprint crosses a wall, keypoint-independently.

    Uses the stored per-frame alignment transform (``transform_matrices``:
    raw->aligned 2x3 affine, finite for every frame) rather than re-estimating it
    from the keypoints. That matters because the keypoints can be flagged missing on
    exactly the frames where the fly is at a wall -- those frames still have a valid
    transform, so their corners can be mapped and the wall caught. Maps the four
    aligned-frame corners to arena mm (inverse stored transform -> raw pixel ->
    calibration) and flags a frame if any corner is within ``clearance_mm`` of a
    wall. A frame whose transform is non-finite (should not occur) is flagged,
    conservatively.

    Returns ``(too_close, keypoints_missing, n_frames)``. ``keypoints_missing`` is
    reported separately (from NaN pre-alignment keypoints) so that callers can keep
    untracked frames while the edge decision no longer depends on the keypoints.
    """
    trial_dir = Path(trial_dir)
    processed = trial_dir / "processed"
    with h5py.File(processed / "behavior_alignment_transforms.h5", "r") as f:
        transforms = f["transform_matrices"][:]  # (n, 2, 3) raw -> aligned
        pre_alignment_keypoints = f["keypoints_xy_pre_alignment"][:]
    stage_dataframe = pd.read_csv(processed / "behavior_frames_metadata.csv")
    stage = stage_dataframe.set_index("behavior_frame_id")[
        ["x_pos_mm_interp", "y_pos_mm_interp"]
    ].to_numpy()
    n = min(len(transforms), len(pre_alignment_keypoints), len(stage))
    transforms = transforms[:n]
    pre_alignment_keypoints = pre_alignment_keypoints[:n]
    stage = stage[:n]

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(trial_dir / "metadata.zip") as zf:
            zf.extractall(tmp)
        weights_x, weights_y = calibration_weights(
            Path(tmp) / "metadata" / "calibration_parameters_behavior.yaml"
        )

    (
        accessible,
        dist_to_wall_mm,
        depth_into_wall_mm,
        raster_resolution,
        arena_x_mm,
        arena_y_mm,
    ) = load_arena(arena_dir)

    width = height = float(frame_size)
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], float)
    ones = np.ones((4, 1))
    too_close = np.zeros(n, bool)
    keypoints_missing = ~np.isfinite(pre_alignment_keypoints).all(axis=(1, 2))
    for i in range(n):
        transform = transforms[i].astype(np.float64)
        if not np.isfinite(transform).all():
            too_close[i] = True  # cannot verify -> treat as edge
            continue
        inverse_transform = cv2.invertAffineTransform(transform)  # aligned -> raw
        corners_raw = np.hstack([corners, ones]) @ inverse_transform.T
        physical = raw_pixel_to_physical(stage[i], corners_raw, weights_x, weights_y)
        margins = corner_margins(
            physical,
            accessible,
            depth_into_wall_mm,
            dist_to_wall_mm,
            raster_resolution,
            arena_x_mm,
            arena_y_mm,
        )
        too_close[i] = bool(np.nanmin(margins) < clearance_mm)
    return too_close, keypoints_missing, n


def suppress_short_runs(flip: np.ndarray, min_run: int) -> np.ndarray:
    """Set flipped runs shorter than ``min_run`` back to not-flipped (kill blips)."""
    out = flip.copy()
    n = len(flip)
    i = 0
    while i < n:
        if flip[i] == 1:
            j = i
            while j < n and flip[j] == 1:
                j += 1
            if (j - i) < min_run:
                out[i:j] = 0
            i = j
        else:
            i += 1
    return out


def dilate_flip_blocks(
    flip: np.ndarray,
    pad_seconds: float,
    behavior_fps: float,
    frame_step: int = 1,
) -> np.ndarray:
    """Grow flip blocks by ``pad_seconds`` on each side (both edges are unusable).

    ``frame_step`` is the spacing between the entries of ``flip`` in original
    behavior frames (1 when every frame is labelled).
    """
    pad = int(round(pad_seconds * behavior_fps / max(frame_step, 1)))
    if pad <= 0:
        return flip.astype(int)
    return ndimage.binary_dilation(flip.astype(bool), iterations=pad).astype(int)


def temporal_flip_filter(
    raw_flip: np.ndarray,
    behavior_fps: float,
    min_run: int = 3,
    pad_seconds: float = 0.3,
    frame_step: int = 1,
) -> np.ndarray:
    """Full temporal flip filter: suppress short runs, then dilate the blocks."""
    suppressed = suppress_short_runs(np.asarray(raw_flip).astype(int), min_run)
    return dilate_flip_blocks(suppressed, pad_seconds, behavior_fps, frame_step)
