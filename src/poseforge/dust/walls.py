"""Extract arena-wall frames from centred dark-field fly recordings.

A *wall* is the bright chamber edge that shows up near a frame border when the
fly is close to it. We isolate it by reusing the dust pipeline's fly removal,
exploiting that the recordings are centred on the fly:

1. the fly is the single largest bright connected component at a very low
   threshold (so its dim leg tips are included), and it must cover the frame
   centre;
2. the frame is kept only when the fly component's area is within the expected
   range -- a wall *fused* (touching) to the fly inflates it, and those frames
   are rejected, so a stored wall never touches the fly;
3. the fly is removed (dilated to strip its halo) and the *residual full frame*
   is kept; residuals holding a sizeable border-touching structure are stored as
   wall samples.

Because the fly sits at the centre, a residual has a hole at the centre, so when
it is composited (additively, mirrored on the two axes only) onto a new centred
frame, the wall adds its peripheral brightness while a new centred fly is left
essentially untouched -- see :func:`poseforge.dust.compositing.add_wall`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from poseforge.dust.config import WallExtractionConfig


@dataclass
class WallPrimitive:
    """One wall sample: a full-frame residual with the fly removed.

    Attributes:
        layer: full-frame uint8 array holding the wall (and any residual dust)
            brightness above background, with the fly region zeroed.
        wall_area: pixel area of the largest border-touching structure (the wall).
        source: name of the frame the wall came from.
    """

    layer: np.ndarray
    wall_area: int
    source: str


def _low_threshold(gray: np.ndarray, config: WallExtractionConfig) -> int:
    median = float(np.median(gray))
    return int(
        np.clip(
            median + config.low_threshold_offset,
            config.low_threshold_min,
            config.low_threshold_max,
        )
    )


def fly_component_mask(
    gray: np.ndarray, config: WallExtractionConfig
) -> tuple[np.ndarray | None, int]:
    """Largest bright connected component at a very low threshold (fly incl. legs).

    Returns ``(mask, area)``. The largest component must also cover the frame
    centre (the fly is centred); otherwise returns ``(None, 0)`` so the frame is
    skipped.
    """
    height, width = gray.shape
    threshold = _low_threshold(gray, config)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.fly_close_kernel_size, config.fly_close_kernel_size)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return None, 0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if labels[height // 2, width // 2] != largest:
        # The centred component is not the biggest -- an odd frame (or a wall
        # larger than the fly fused in). Skip it rather than mistake the wall for
        # the fly.
        return None, 0
    return labels == largest, int(stats[largest, cv2.CC_STAT_AREA])


def isolate_residual(
    gray: np.ndarray, fly_mask: np.ndarray, config: WallExtractionConfig
) -> np.ndarray:
    """Full-frame residual (float32) with the fly and its halo removed, floor subtracted."""
    halo = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.fly_halo_dilate_kernel_size, config.fly_halo_dilate_kernel_size),
    )
    fly_dilated = cv2.dilate(fly_mask.astype(np.uint8), halo).astype(bool)
    layer = gray.astype(np.float32)
    layer[fly_dilated] = 0.0
    floor = float(
        np.clip(np.median(gray) + config.floor_offset, config.floor_min, config.floor_max)
    )
    return np.clip(layer - floor, 0.0, None)


def largest_border_structure(residual: np.ndarray, config: WallExtractionConfig) -> int:
    """Area of the largest border-touching connected component of the residual."""
    height, width = residual.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (residual > 0).astype(np.uint8), connectivity=8
    )
    margin_x = int(config.wall_border_margin_fraction * width)
    margin_y = int(config.wall_border_margin_fraction * height)
    best = 0
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        touches_border = (
            x < margin_x
            or y < margin_y
            or (x + w) > (width - margin_x)
            or (y + h) > (height - margin_y)
        )
        if touches_border:
            best = max(best, int(area))
    return best


def extract_wall_from_frame(
    gray: np.ndarray, config: WallExtractionConfig, source: str
) -> WallPrimitive | None:
    """Return the full-frame wall residual for this frame, or None.

    None when the fly is not cleanly detected / isolated (including a wall fused
    to the fly, which inflates its area), or when the residual holds no sizeable
    border-touching structure (i.e. no wall is present).
    """
    fly_mask, fly_area = fly_component_mask(gray, config)
    if fly_mask is None:
        return None
    fraction = fly_area / gray.size
    if not (config.min_fly_area_fraction <= fraction <= config.max_fly_area_fraction):
        return None
    residual = isolate_residual(gray, fly_mask, config)
    wall_area = largest_border_structure(residual, config)
    if wall_area < config.min_wall_area_fraction * gray.size:
        return None
    return WallPrimitive(layer=residual.astype(np.uint8), wall_area=wall_area, source=source)


class WallLibrary:
    """A collection of wall residuals, savable as a single ``.npz`` archive."""

    def __init__(self, walls: list[WallPrimitive]):
        self.walls = walls

    def __len__(self) -> int:
        return len(self.walls)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            layers=np.array([w.layer for w in self.walls], dtype=object),
            wall_areas=np.array([w.wall_area for w in self.walls], dtype=np.int64),
            sources=np.array([w.source for w in self.walls], dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "WallLibrary":
        data = np.load(path, allow_pickle=True)
        walls = [
            WallPrimitive(
                layer=layer.astype(np.uint8),
                wall_area=int(wall_area),
                source=str(source),
            )
            for layer, wall_area, source in zip(
                data["layers"], data["wall_areas"], data["sources"]
            )
        ]
        return cls(walls)

    def make_preview_sheet(
        self, path: str | Path, n_columns: int = 4, cell: int = 220, max_cells: int = 16
    ) -> None:
        """Contact sheet of full-frame wall residuals (downscaled), for inspection.

        Samples evenly across the library so the sheet is representative, not just
        the first few (consecutive) walls captured.
        """
        if len(self.walls) <= max_cells:
            chosen = list(self.walls)
        else:
            indices = np.linspace(0, len(self.walls) - 1, max_cells).astype(int)
            chosen = [self.walls[i] for i in indices]
        n_rows = int(np.ceil(len(chosen) / n_columns)) if chosen else 1
        sheet = np.zeros((n_rows * cell, n_columns * cell), dtype=np.uint8)
        for index, wall in enumerate(chosen):
            layer = wall.layer
            peak = layer.max()
            normalized = (
                (layer.astype(np.float32) / peak * 255).astype(np.uint8) if peak > 0 else layer
            )
            resized = cv2.resize(normalized, (cell, cell))
            row, column = divmod(index, n_columns)
            sheet[row * cell : (row + 1) * cell, column * cell : (column + 1) * cell] = resized
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), sheet)
