"""Extract dust / fiber primitives from real dark-field recordings.

A *primitive* is a single connected dust region (a fiber, a clump, a speck) cut
out as a small floating-point patch holding its brightness above background,
with zero elsewhere. A library of many primitives, sampled and transformed
independently at compositing time, yields effectively unlimited distinct dusty
images.

The fly-removal strategy exploits the dark-field geometry: the background is
near-black, so everything bright is either the fly or junk. The fly is the
single largest bright connected component; we remove it (dilated to also strip
its out-of-focus halo) and keep the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from poseforge.dust.config import (
    DustExtractionConfig,
    LibraryBalanceConfig,
)


@dataclass
class DustPrimitive:
    """One isolated dust region.

    Attributes:
        patch: float32 array holding dust brightness above background; zero where
            there is no dust. Variable size (the region's bounding box).
        area: number of non-zero pixels.
        max_intensity: peak brightness of the primitive.
        source: name of the frame/file the primitive came from.
    """

    patch: np.ndarray
    area: int
    max_intensity: float
    source: str


def compute_fly_mask(gray: np.ndarray, config: DustExtractionConfig) -> np.ndarray:
    """Return a boolean mask of the fly (body + legs), deliberately over-inclusive.

    Thresholds low (just above the dark background) rather than at Otsu, so the
    dim wings and leg tips join the body as one connected component instead of
    leaking into the dust. The fly is then the single largest bright component.
    """
    median = float(np.median(gray))
    threshold = int(
        np.clip(
            median + config.foreground_threshold_offset,
            config.foreground_threshold_min,
            config.foreground_threshold_max,
        )
    )
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.fly_close_kernel_size, config.fly_close_kernel_size),
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(gray, dtype=bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def isolate_dust(gray: np.ndarray, config: DustExtractionConfig) -> np.ndarray:
    """Return a float32 layer holding only bright dust, with the fly removed.

    The fly mask is dilated well beyond the body so the diffuse halo is removed
    too; a low background floor is then subtracted and clipped at zero, leaving
    dust brightness above background.
    """
    fly_mask = compute_fly_mask(gray, config)
    halo_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.fly_halo_dilate_kernel_size, config.fly_halo_dilate_kernel_size),
    )
    fly_mask = cv2.dilate(fly_mask.astype(np.uint8), halo_kernel).astype(bool)

    layer = gray.astype(np.float32)
    layer[fly_mask] = 0.0
    floor = float(
        np.clip(
            np.median(gray) + config.dust_floor_offset,
            config.dust_floor_min,
            config.dust_floor_max,
        )
    )
    return np.clip(layer - floor, 0.0, None)


def extract_primitives_from_frame(
    gray: np.ndarray, config: DustExtractionConfig, source: str
) -> list[DustPrimitive]:
    """Cut each connected dust region out of one frame as a primitive."""
    dust = isolate_dust(gray, config)
    binary = (dust > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    frame_height, frame_width = gray.shape
    frame_area = frame_height * frame_width
    max_area = config.max_primitive_area_fraction * frame_area
    max_extent = config.max_primitive_extent_fraction * max(gray.shape)
    margin_x = config.edge_exclude_margin_fraction * frame_width
    margin_y = config.edge_exclude_margin_fraction * frame_height

    primitives: list[DustPrimitive] = []
    for label in range(1, num_labels):
        x, y, width, height, area = stats[label]
        if area < config.min_primitive_area or area > max_area:
            continue
        if max(width, height) > max_extent:
            continue
        # Drop primitives whose bounding box reaches into the frame border.
        if (
            x < margin_x
            or y < margin_y
            or (x + width) > (frame_width - margin_x)
            or (y + height) > (frame_height - margin_y)
        ):
            continue
        region_mask = labels[y : y + height, x : x + width] == label
        patch = dust[y : y + height, x : x + width].copy()
        patch[~region_mask] = 0.0  # keep only this region's pixels
        primitives.append(
            DustPrimitive(
                patch=patch.astype(np.float32),
                area=int(area),
                max_intensity=float(patch.max()),
                source=source,
            )
        )
    return primitives


def _signature(patch: np.ndarray, size: int = 16) -> np.ndarray:
    """A small fixed-size intensity signature for near-duplicate detection."""
    resized = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    peak = resized.max()
    if peak > 0:
        resized = resized / peak * 255.0
    return resized.astype(np.float32)


def _assign_bin(area: int, area_bin_edges: tuple[int, ...]) -> int:
    """Return the size-bin index for a primitive of the given area.

    Bins are [0, e0], (e0, e1], ..., (e_{k-1}, inf). There are len(edges)+1 bins.
    """
    for index, edge in enumerate(area_bin_edges):
        if area <= edge:
            return index
    return len(area_bin_edges)


class _SizeBalancedCollector:
    """Keep a size-balanced, deduplicated, memory-bounded sample of primitives.

    For each size bin, keeps at most ``max_per_bin`` primitives via reservoir
    sampling (so the result is a uniform random sample across all frames seen,
    without ever holding the full set in memory), skipping near-duplicates of
    primitives already kept in that bin.
    """

    def __init__(
        self, balance_config: LibraryBalanceConfig, deduplicate: bool, tolerance: float
    ):
        self.config = balance_config
        self.deduplicate = deduplicate
        self.tolerance = tolerance
        self.rng = np.random.default_rng(balance_config.seed)
        n_bins = len(balance_config.area_bin_edges) + 1
        self.kept: list[list[DustPrimitive]] = [[] for _ in range(n_bins)]
        self.signatures: list[np.ndarray | None] = [None] * n_bins  # (K, s, s) per bin
        self.seen: list[int] = [0] * n_bins  # non-duplicate primitives offered per bin

    def _is_duplicate(self, bin_index: int, signature: np.ndarray) -> bool:
        stack = self.signatures[bin_index]
        if stack is None or len(stack) == 0:
            return False
        differences = np.mean(np.abs(stack - signature[None]), axis=(1, 2))
        return bool(differences.min() < self.tolerance)

    def _append_signature(self, bin_index: int, signature: np.ndarray) -> None:
        stack = self.signatures[bin_index]
        self.signatures[bin_index] = (
            signature[None] if stack is None else np.concatenate([stack, signature[None]])
        )

    def offer(self, primitive: DustPrimitive) -> None:
        bin_index = _assign_bin(primitive.area, self.config.area_bin_edges)
        signature = _signature(primitive.patch)
        if self.deduplicate and self._is_duplicate(bin_index, signature):
            return

        self.seen[bin_index] += 1
        kept = self.kept[bin_index]
        if len(kept) < self.config.max_per_bin:
            kept.append(primitive)
            self._append_signature(bin_index, signature)
        else:
            # Reservoir replacement, keeping a uniform random sample of the bin.
            replace_index = int(self.rng.integers(self.seen[bin_index]))
            if replace_index < self.config.max_per_bin:
                kept[replace_index] = primitive
                self.signatures[bin_index][replace_index] = signature

    def primitives(self) -> list[DustPrimitive]:
        return [primitive for bin_kept in self.kept for primitive in bin_kept]

    def bin_counts(self) -> list[int]:
        return [len(bin_kept) for bin_kept in self.kept]


class DustLibrary:
    """A collection of dust primitives, savable as a single ``.npz`` archive."""

    def __init__(self, primitives: list[DustPrimitive]):
        self.primitives = primitives

    def __len__(self) -> int:
        return len(self.primitives)

    def patches(self) -> list[np.ndarray]:
        return [primitive.patch for primitive in self.primitives]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            patches=np.array([p.patch for p in self.primitives], dtype=object),
            areas=np.array([p.area for p in self.primitives], dtype=np.int64),
            max_intensities=np.array(
                [p.max_intensity for p in self.primitives], dtype=np.float32
            ),
            sources=np.array([p.source for p in self.primitives], dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DustLibrary":
        data = np.load(path, allow_pickle=True)
        primitives = [
            DustPrimitive(
                patch=patch.astype(np.float32),
                area=int(area),
                max_intensity=float(max_intensity),
                source=str(source),
            )
            for patch, area, max_intensity, source in zip(
                data["patches"], data["areas"], data["max_intensities"], data["sources"]
            )
        ]
        return cls(primitives)

    def make_preview_sheet(
        self, path: str | Path, n_columns: int = 12, cell_size: int = 72, max_cells: int = 144
    ) -> None:
        """Render a contact sheet of primitives (brightness-normalized) for inspection.

        Primitives are sampled evenly across the (bin-ordered) library so every
        size class is represented, rather than showing only the smallest bin.
        """
        if len(self.primitives) <= max_cells:
            chosen = list(self.primitives)
        else:
            indices = np.linspace(0, len(self.primitives) - 1, max_cells).astype(int)
            chosen = [self.primitives[i] for i in indices]
        n_rows = int(np.ceil(len(chosen) / n_columns)) if chosen else 1
        sheet = np.zeros((n_rows * cell_size, n_columns * cell_size), dtype=np.uint8)
        for index, primitive in enumerate(chosen):
            patch = primitive.patch
            peak = patch.max()
            normalized = (patch / peak * 255.0).astype(np.uint8) if peak > 0 else patch.astype(np.uint8)
            # Fit the patch inside a cell, preserving aspect ratio.
            height, width = normalized.shape
            scale = min(cell_size / max(height, 1), cell_size / max(width, 1), 1.0)
            resized = cv2.resize(
                normalized, (max(int(width * scale), 1), max(int(height * scale), 1))
            )
            row, column = divmod(index, n_columns)
            y0 = row * cell_size + (cell_size - resized.shape[0]) // 2
            x0 = column * cell_size + (cell_size - resized.shape[1]) // 2
            sheet[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), sheet)


