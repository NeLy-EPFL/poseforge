"""Measured natural distributions of dust and walls, kept as separate objects.

Dust and walls are extracted independently (see the ``extract_dust`` and
``extract_walls`` scripts), so each has its own statistics object with its own
data-derived *presence probability*:

- :class:`DustStatistics` -- how often a frame has any dust, how many primitives
  it has when it does, and their size distribution.
- :class:`WallStatistics` -- how often a frame shows an arena wall.

At compositing time the two probabilities are applied independently (each
overridable), so dust and walls appear at their own measured rates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from poseforge.dust.config import DustExtractionConfig, LibraryBalanceConfig
from poseforge.dust.primitives import (
    DustLibrary,
    _SizeBalancedCollector,
    extract_primitives_from_frame,
)


@dataclass
class DustStatistics:
    """Empirical dust distribution measured from real frames."""

    # Fraction of measured frames that contain at least one dust primitive.
    dust_presence_rate: float = 0.0
    # Primitive counts of the frames that had dust (used to draw how many to add).
    dust_counts_when_present: list[int] = field(default_factory=list)
    # Areas (pixels) of every dust primitive found, for the size distribution.
    dust_areas: list[int] = field(default_factory=list)
    n_frames_measured: int = 0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "DustStatistics":
        with open(path) as f:
            return cls(**json.load(f))

    def dust_present(self, rng: np.random.Generator, override: float | None = None) -> bool:
        """Whether to add dust to a frame, using the measured rate unless overridden."""
        probability = self.dust_presence_rate if override is None else override
        return bool(rng.random() < probability)

    def sample_count(self, rng: np.random.Generator) -> int:
        """Draw how many primitives to add, from the measured present-frame counts."""
        if not self.dust_counts_when_present:
            return 0
        return int(rng.choice(self.dust_counts_when_present))


@dataclass
class WallStatistics:
    """Empirical wall distribution measured from real frames."""

    # Fraction of measured frames that show an arena wall (in the distance band).
    wall_presence_rate: float = 0.0
    # Pixel areas of the detected walls, for reference / tuning.
    wall_areas: list[int] = field(default_factory=list)
    n_frames_measured: int = 0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "WallStatistics":
        with open(path) as f:
            return cls(**json.load(f))

    def wall_present(self, rng: np.random.Generator, override: float | None = None) -> bool:
        """Whether to add a wall to a frame, using the measured rate unless overridden."""
        probability = self.wall_presence_rate if override is None else override
        return bool(rng.random() < probability)


def build_dust_artifacts(
    frames: Iterable[tuple[np.ndarray, str]],
    extraction_config: DustExtractionConfig,
    balance_config: LibraryBalanceConfig,
) -> tuple[DustLibrary, list[int], DustStatistics]:
    """Single pass over frames: build the size-balanced dust library and statistics.

    Returns ``(dust_library, dust_bin_counts, dust_statistics)``.
    """
    collector = _SizeBalancedCollector(
        balance_config, extraction_config.deduplicate, extraction_config.deduplicate_tolerance
    )
    counts_when_present: list[int] = []
    areas: list[int] = []
    n_frames = 0
    frames_with_dust = 0

    for gray, source in frames:
        n_frames += 1
        primitives = extract_primitives_from_frame(gray, extraction_config, source)
        if primitives:
            frames_with_dust += 1
            counts_when_present.append(len(primitives))
            areas.extend(int(p.area) for p in primitives)
        for primitive in primitives:
            collector.offer(primitive)

    statistics = DustStatistics(
        dust_presence_rate=(frames_with_dust / n_frames) if n_frames else 0.0,
        dust_counts_when_present=counts_when_present,
        dust_areas=areas,
        n_frames_measured=n_frames,
    )
    return DustLibrary(collector.primitives()), collector.bin_counts(), statistics
