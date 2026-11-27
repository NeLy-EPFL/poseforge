"""Dust / fiber and arena-wall augmentation for dark-field fly recordings.

Two artifacts, extracted independently from the real recordings and composited
back onto clean (style-transferred) videos:

- **Dust** -- individual fibers, clumps and specks, cut out as a size-balanced
  library of primitives and added additively over the whole frame
  (:mod:`poseforge.dust.primitives`).
- **Walls** -- the bright arena edge, taken as a full-frame residual (fly
  removed) from frames whose stage-derived fly-to-wall distance is in a chosen
  band (:mod:`poseforge.dust.walls`, :mod:`poseforge.dust.stage`).

Each has its own measured *presence probability* (:mod:`poseforge.dust.statistics`),
applied independently at compositing time (:mod:`poseforge.dust.compositing`).
"""

from poseforge.dust.compositing import (
    add_artifacts_to_frames,
    add_dust,
    add_wall,
    compute_size_matched_weights,
)
from poseforge.dust.config import (
    DustCompositingConfig,
    DustExtractionConfig,
    LibraryBalanceConfig,
    WallCompositingConfig,
    WallDistanceConfig,
    WallExtractionConfig,
)
from poseforge.dust.primitives import (
    DustLibrary,
    DustPrimitive,
    extract_primitives_from_frame,
)
from poseforge.dust.stage import (
    band_frame_indices,
    build_wall_library_from_trials,
    find_trials,
    fly_wall_distances,
    iter_trial_band_frames,
    iter_trial_frames,
)
from poseforge.dust.statistics import (
    DustStatistics,
    WallStatistics,
    build_dust_artifacts,
)
from poseforge.dust.walls import (
    WallLibrary,
    WallPrimitive,
    extract_wall_from_frame,
)

__all__ = [
    "add_artifacts_to_frames",
    "add_dust",
    "add_wall",
    "compute_size_matched_weights",
    "DustCompositingConfig",
    "DustExtractionConfig",
    "LibraryBalanceConfig",
    "WallCompositingConfig",
    "WallDistanceConfig",
    "WallExtractionConfig",
    "DustLibrary",
    "DustPrimitive",
    "extract_primitives_from_frame",
    "band_frame_indices",
    "build_wall_library_from_trials",
    "find_trials",
    "fly_wall_distances",
    "iter_trial_band_frames",
    "iter_trial_frames",
    "DustStatistics",
    "WallStatistics",
    "build_dust_artifacts",
    "WallLibrary",
    "WallPrimitive",
    "extract_wall_from_frame",
]
