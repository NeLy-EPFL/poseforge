"""Select wall-bearing frames by the fly's stage-derived distance to the arena wall.

The behaviour recordings are centred on a tethered fly whose position in the
arena is set by the stage. Each trial stores, per frame, the stage position (in
``processed/behavior_frames_metadata.csv``) and the fly's body-centre keypoint in
raw pixels (in ``processed/behavior_alignment_transforms.h5``). The arena's
``model/calibration_result.yaml`` gives an affine map from (stage position, fly
pixel) to the fly's physical position in the arena. From that we get the fly's
distance to the nearest arena wall every frame -- a far more reliable cue than
image thresholding for whether a wall is in view and how close it is to the fly.

Only frames in the distance band (see :class:`WallDistanceConfig`) are used as
wall sources; the residual wall-presence check in
:func:`poseforge.dust.walls.extract_wall_from_frame` then drops the few in-band
frames where that arena side has no bright edge in the crop.

The heavy dependencies (pandas, h5py, yaml, cv2) are imported lazily so importing
``poseforge.dust`` stays cheap and does not hard-depend on them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from poseforge.dust.config import WallDistanceConfig, WallExtractionConfig
from poseforge.dust.walls import WallLibrary, WallPrimitive, extract_wall_from_frame


def _arena_extent(arena_dir: Path, config: WallDistanceConfig) -> tuple[float, float]:
    """Arena (width_x, height_y) in mm, from config override or the arena metadata."""
    if config.arena_width_mm is not None and config.arena_height_mm is not None:
        return float(config.arena_width_mm), float(config.arena_height_mm)
    import yaml

    with open(Path(arena_dir) / "metadata.yaml") as f:
        dims = yaml.safe_load(f)["arena_dim"]  # [x_width, y_height] mm
    return float(dims[0]), float(dims[1])


def _calibration(arena_dir: Path) -> dict:
    import yaml

    with open(Path(arena_dir) / "model" / "calibration_result.yaml") as f:
        return yaml.safe_load(f)["stage_and_pixel_to_physical"]


def fly_wall_distances(
    trial_dir: str | Path,
    arena_dir: str | Path,
    config: WallDistanceConfig,
) -> np.ndarray:
    """Per-frame distance (mm) from the fly's body centre to the nearest arena wall.

    Returns an array of length ``n_behavior_frames``; entries are ``NaN`` where the
    anchor keypoint is missing (tracking gap) so they are never selected.
    """
    import h5py
    import pandas as pd

    processed = Path(trial_dir) / "processed"
    meta = pd.read_csv(processed / config.behavior_metadata_name)
    stage_x = meta[config.stage_x_column].to_numpy()
    stage_y = meta[config.stage_y_column].to_numpy()

    with h5py.File(processed / config.alignment_transforms_name, "r") as f:
        keypoints = f["keypoints_xy_pre_alignment"][:, config.anchor_keypoint_index, :]

    # The stage metadata and the alignment transforms are occasionally off by a
    # frame; align them by truncating to their common length.
    n = min(len(stage_x), len(keypoints))
    stage_x = stage_x[:n]
    stage_y = stage_y[:n]
    keypoints = keypoints[:n]
    valid = np.isfinite(keypoints).all(axis=1)
    pixel_x = keypoints[:, 0]
    pixel_y = keypoints[:, 1]

    calibration = _calibration(Path(arena_dir))
    cx = calibration["physical_pos_x"]
    cy = calibration["physical_pos_y"]
    physical_x = (
        cx["stage_pos_x"] * stage_x
        + cx["stage_pos_y"] * stage_y
        + cx["pixel_pos_x"] * pixel_x
        + cx["pixel_pos_y"] * pixel_y
        + cx["bias"]
    )
    physical_y = (
        cy["stage_pos_x"] * stage_x
        + cy["stage_pos_y"] * stage_y
        + cy["pixel_pos_x"] * pixel_x
        + cy["pixel_pos_y"] * pixel_y
        + cy["bias"]
    )

    width, height = _arena_extent(Path(arena_dir), config)
    distance = np.minimum.reduce(
        [physical_x, width - physical_x, physical_y, height - physical_y]
    )
    distance[~valid] = np.nan
    return distance


def find_trials(dataset_root: str | Path, config: WallDistanceConfig) -> list[Path]:
    """Trial folders under ``dataset_root`` that have the needed processed files.

    A trial qualifies when its ``processed/`` folder holds the aligned video, the
    behaviour metadata and the alignment transforms.
    """
    root = Path(dataset_root)
    trials = []
    for processed in sorted(root.glob("*/processed")):
        if (
            (processed / config.aligned_video_name).exists()
            and (processed / config.behavior_metadata_name).exists()
            and (processed / config.alignment_transforms_name).exists()
        ):
            trials.append(processed.parent)
    return trials


def iter_trial_frames(
    trial_dir: str | Path,
    config: WallDistanceConfig,
    stride: int = 15,
    max_frames: int | None = None,
) -> Iterator[tuple[np.ndarray, str]]:
    """Yield ``(grayscale_frame, source)`` from a trial's aligned video, every ``stride`` frames.

    Reads the video sequentially and keeps every ``stride``-th frame. Sequential
    decoding is ~100x faster than per-frame seeking on the all-intra aligned
    videos, where OpenCV re-decodes from the file start on each seek.

    Used to source dust primitives / statistics from the same centred domain the
    walls come from (and that the translated videos are in).
    """
    import cv2

    trial_dir = Path(trial_dir)
    video_path = trial_dir / "processed" / config.aligned_video_name
    capture = cv2.VideoCapture(str(video_path))
    try:
        index = 0
        yielded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yield gray, f"{trial_dir.name}:{index}"
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            index += 1
    finally:
        capture.release()


def band_frame_indices(
    trial_dir: str | Path,
    arena_dir: str | Path,
    config: WallDistanceConfig,
) -> np.ndarray:
    """Indices of frames whose fly-wall distance is within the configured band."""
    distance = fly_wall_distances(trial_dir, arena_dir, config)
    return np.where(
        np.isfinite(distance)
        & (distance >= config.min_distance_mm)
        & (distance <= config.max_distance_mm)
    )[0]


def iter_trial_band_frames(
    trial_dir: str | Path,
    arena_dir: str | Path,
    config: WallDistanceConfig,
    max_frames: int | None = None,
) -> Iterator[tuple[np.ndarray, str]]:
    """Yield ``(grayscale_frame, source)`` for the trial's in-band aligned frames.

    Reads only the selected frames from the aligned behaviour video, so a long
    recording is decoded sparsely.
    """
    import cv2

    trial_dir = Path(trial_dir)
    indices = band_frame_indices(trial_dir, arena_dir, config)
    if max_frames is not None and len(indices) > max_frames:
        indices = indices[np.linspace(0, len(indices) - 1, max_frames).astype(int)]
    wanted = set(int(i) for i in indices)
    if not wanted:
        return

    # Read sequentially and keep the in-band frames: ~100x faster than seeking to
    # each one, since the all-intra aligned videos re-decode from the file start
    # on every OpenCV seek. Stop once the last wanted frame has been read.
    last_wanted = max(wanted)
    video_path = trial_dir / "processed" / config.aligned_video_name
    capture = cv2.VideoCapture(str(video_path))
    try:
        index = 0
        while index <= last_wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yield gray, f"{trial_dir.name}:{index}"
            index += 1
    finally:
        capture.release()


def build_wall_library_from_trials(
    trial_dirs: list[str | Path],
    arena_dir: str | Path,
    extraction_config: WallExtractionConfig,
    distance_config: WallDistanceConfig,
    max_frames_per_trial: int | None = None,
) -> tuple[WallLibrary, "WallStatistics"]:
    """Build the wall library + statistics from many trials, gated by stage distance.

    For each trial, the in-band frames (distance in the configured band) are
    passed through :func:`extract_wall_from_frame`, which removes the fly and
    keeps the residual only when it holds a real border-touching wall (the
    presence check). Walls are reservoir-capped to ``extraction_config.max_walls``.

    Returns ``(wall_library, wall_statistics)``. The statistics' presence rate is
    the fraction of all behaviour frames (across the trials) that yielded a wall --
    the natural frequency the compositor reproduces by default.
    """
    from poseforge.dust.statistics import WallStatistics

    walls: list[WallPrimitive] = []
    wall_areas: list[int] = []
    rng = np.random.default_rng(0)
    seen = 0
    total_frames = 0
    walls_kept = 0

    import pandas as pd

    for trial_dir in trial_dirs:
        trial_dir = Path(trial_dir)
        metadata_path = trial_dir / "processed" / distance_config.behavior_metadata_name
        total_frames += len(pd.read_csv(metadata_path, usecols=[0]))
        for gray, source in iter_trial_band_frames(
            trial_dir, arena_dir, distance_config, max_frames=max_frames_per_trial
        ):
            wall = extract_wall_from_frame(gray, extraction_config, source)
            if wall is None:
                continue
            walls_kept += 1
            wall_areas.append(int(wall.wall_area))
            seen += 1
            if len(walls) < extraction_config.max_walls:
                walls.append(wall)
            else:
                replace = int(rng.integers(seen))
                if replace < extraction_config.max_walls:
                    walls[replace] = wall

    statistics = WallStatistics(
        wall_presence_rate=(walls_kept / total_frames) if total_frames else 0.0,
        wall_areas=wall_areas,
        n_frames_measured=total_frames,
    )
    return WallLibrary(walls), statistics
