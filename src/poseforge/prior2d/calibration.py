"""Shared per-trial calibration loading and pixel/mm conversion helpers.

Used by `scripts/select_continuous_periods.py` (aligned pixels -> mm) and
`scripts/solve_ik.py` (mm -> raw pixels -> aligned pixels, for `fk_2d_px`).
"""

import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from spotlight_tools.calibration.mapper import SpotlightPositionMapper

from poseforge.prior2d.geometry import apply_affine, invert_affine

CALIBRATION_ZIP_RELPATH = Path("metadata.zip")
CALIBRATION_YAML_NAME = "metadata/calibration_parameters_behavior.yaml"
TRANSFORMS_RELPATH = Path("processed/behavior_alignment_transforms.h5")
STAGE_POSITION_CSV_RELPATH = Path("processed/behavior_frames_metadata.csv")


def load_calibration_mapper(trial_dir: Path) -> SpotlightPositionMapper:
    """Load a trial's stage/pixel/physical calibration mapper.

    Reads the calibration YAML directly out of `metadata.zip` (never
    extracted to disk, since `trial_dir` typically lives on a read-only
    mount).

    Args:
        trial_dir: Trial directory containing `metadata.zip`.

    Returns:
        Mapper fitted for this trial's camera and stage.
    """
    with zipfile.ZipFile(trial_dir / CALIBRATION_ZIP_RELPATH) as zf:
        raw_yaml = zf.read(CALIBRATION_YAML_NAME)
    return SpotlightPositionMapper(yaml.safe_load(raw_yaml))


def load_transform_matrices(trial_dir: Path) -> np.ndarray:
    """Load a trial's per-frame raw-to-aligned affine transforms.

    Args:
        trial_dir: Trial directory containing `TRANSFORMS_RELPATH`.

    Returns:
        `(n_frames, 2, 3)` affine matrices, one per frame.
    """
    with h5py.File(trial_dir / TRANSFORMS_RELPATH, "r") as f:
        return f["transform_matrices"][:]


def load_stage_positions_mm(trial_dir: Path) -> np.ndarray:
    """Load a trial's per-frame stage position, in mm.

    Args:
        trial_dir: Trial directory containing `STAGE_POSITION_CSV_RELPATH`.

    Returns:
        `(n_frames, 2)` stage (x, y) position, in mm.
    """
    df = pd.read_csv(trial_dir / STAGE_POSITION_CSV_RELPATH)
    return df[["x_pos_mm_interp", "y_pos_mm_interp"]].to_numpy(dtype=np.float64)


def convert_px_to_mm(
    aligned_px: np.ndarray,
    transform_matrices: np.ndarray,
    stage_positions_mm: np.ndarray,
    mapper: SpotlightPositionMapper,
) -> np.ndarray:
    """Convert aligned-video-domain pixel keypoints to physical (mm) coordinates.

    Args:
        aligned_px: `(period_length, n_nodes, 2)` keypoints in the aligned
            (cropped) video domain, may contain NaN.
        transform_matrices: `(period_length, 2, 3)` per-frame affine matrices
            mapping raw camera pixels to the aligned domain, for the same
            frame range as `aligned_px`.
        stage_positions_mm: `(period_length, 2)` stage (x, y) position, in mm,
            for the same frame range.
        mapper: Fitted stage+pixel -> physical mapper for this trial.

    Returns:
        `(period_length, n_nodes, 2)` physical (x, y) coordinates, in mm.
    """
    raw_px = apply_affine(aligned_px, invert_affine(transform_matrices))
    stage_pos = np.broadcast_to(stage_positions_mm[:, np.newaxis, :], raw_px.shape)
    return mapper.stage_and_pixel_to_physical(stage_pos, raw_px)


def convert_mm_to_px(
    physical_mm: np.ndarray,
    transform_matrices: np.ndarray,
    stage_positions_mm: np.ndarray,
    mapper: SpotlightPositionMapper,
) -> np.ndarray:
    """Convert physical (mm) keypoints to aligned-video-domain pixel coordinates.

    Inverse of `convert_px_to_mm`.

    Args:
        physical_mm: `(period_length, n_nodes, 2)` physical (x, y)
            coordinates, in mm, may contain NaN.
        transform_matrices: `(period_length, 2, 3)` per-frame affine matrices
            mapping raw camera pixels to the aligned domain, for the same
            frame range as `physical_mm`.
        stage_positions_mm: `(period_length, 2)` stage (x, y) position, in mm,
            for the same frame range.
        mapper: Fitted stage+pixel -> physical mapper for this trial.

    Returns:
        `(period_length, n_nodes, 2)` keypoints in the aligned video domain.
    """
    stage_pos = np.broadcast_to(stage_positions_mm[:, np.newaxis, :], physical_mm.shape)
    raw_px = mapper.stage_and_physical_to_pixel(stage_pos, physical_mm)
    return apply_affine(raw_px, transform_matrices)
