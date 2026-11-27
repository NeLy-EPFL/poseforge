"""Composite dust primitives onto frames.

Dark-field junk between the lens and the subject scatters light, so it *adds*
brightness wherever it falls -- background or fly alike. Compositing is therefore
additive over the whole frame, which naturally produces dust that covers the fly.

``add_dust`` is a pure, seedable ``frame -> frame`` transform. It is used both by
the batch folder-to-folder augmentation script and, later, as an on-the-fly
augmentation inside the contrastive data loader.
"""

from __future__ import annotations

import cv2
import numpy as np

from poseforge.dust.config import (
    DustCompositingConfig,
    WallCompositingConfig,
)
from poseforge.dust.primitives import DustPrimitive
from poseforge.dust.walls import WallPrimitive


def _transform_primitive(
    patch: np.ndarray, config: DustCompositingConfig, rng: np.random.Generator
) -> np.ndarray:
    """Flip / rotate / scale / defocus / intensity-jitter a single primitive patch."""
    if config.random_flip:
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        if rng.random() < 0.5:
            patch = patch[::-1, :]

    height, width = patch.shape
    size_scale = rng.uniform(config.min_size_scale, config.max_size_scale)
    angle = rng.uniform(0.0, 360.0)
    # Rotate+scale about the patch centre on a canvas padded to avoid clipping.
    diagonal = int(np.ceil(np.hypot(height, width) * max(size_scale, 1.0))) + 2
    canvas = np.zeros((diagonal, diagonal), dtype=np.float32)
    offset_y = (diagonal - height) // 2
    offset_x = (diagonal - width) // 2
    canvas[offset_y : offset_y + height, offset_x : offset_x + width] = patch
    matrix = cv2.getRotationMatrix2D((diagonal / 2, diagonal / 2), angle, size_scale)
    transformed = cv2.warpAffine(
        canvas, matrix, (diagonal, diagonal), flags=cv2.INTER_LINEAR, borderValue=0
    )

    blur_sigma = rng.uniform(config.min_blur_sigma, config.max_blur_sigma)
    transformed = cv2.GaussianBlur(transformed, (0, 0), blur_sigma)

    intensity_scale = rng.uniform(config.min_intensity_scale, config.max_intensity_scale)
    transformed *= intensity_scale

    # Crop back to the tight non-zero bounding box so placement is exact.
    nonzero = transformed > 0
    if not nonzero.any():
        return transformed
    rows = np.where(nonzero.any(axis=1))[0]
    columns = np.where(nonzero.any(axis=0))[0]
    return transformed[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]


def compute_sampling_weights(
    primitives: list[DustPrimitive], power: float
) -> np.ndarray | None:
    """Sampling probabilities proportional to each primitive's integrated brightness.

    The library is dominated by faint specks; weighting by integrated brightness
    (to the given power) lets the prominent fibers and clumps appear often enough
    to be visible. Returns None for power 0 (uniform sampling).
    """
    if power == 0:
        return None
    integrated = np.array(
        [float(primitive.patch.sum()) for primitive in primitives], dtype=np.float64
    )
    weights = np.power(integrated, power)
    total = weights.sum()
    if total <= 0:
        return None
    return weights / total


def _paste_additive(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    """Additively blend ``patch`` into ``canvas`` at (top, left), clipping to bounds."""
    canvas_height, canvas_width = canvas.shape
    patch_height, patch_width = patch.shape

    y0 = max(top, 0)
    x0 = max(left, 0)
    y1 = min(top + patch_height, canvas_height)
    x1 = min(left + patch_width, canvas_width)
    if y0 >= y1 or x0 >= x1:
        return  # fully off-frame
    patch_region = patch[y0 - top : y1 - top, x0 - left : x1 - left]
    canvas[y0:y1, x0:x1] += patch_region


def add_dust(
    frame: np.ndarray,
    primitives: list[DustPrimitive],
    config: DustCompositingConfig,
    rng: np.random.Generator,
    weights: np.ndarray | None = None,
    n_layers: int | None = None,
) -> np.ndarray:
    """Return ``frame`` with a random number of transformed dust primitives added.

    Args:
        frame: grayscale uint8 frame.
        primitives: the dust library's primitives to sample from.
        config: compositing parameters (layer count range, jitter ranges, ...).
        rng: random generator; pass a per-frame seeded generator for reproducibility.
        weights: optional sampling probabilities over ``primitives``. If None,
            they are computed from ``config.sampling_power`` (recompute once and
            pass them in when composing many frames).
        n_layers: optional exact number of primitives to overlay (e.g. drawn from
            the measured natural distribution). If None, drawn from the config's
            min_layers/max_layers range.

    Returns:
        A uint8 frame with dust additively composited over the whole image,
        including over the subject.
    """
    if len(primitives) == 0:
        return frame.copy()

    if weights is None:
        weights = compute_sampling_weights(primitives, config.sampling_power)

    canvas = frame.astype(np.float32)
    height, width = frame.shape
    if n_layers is None:
        n_layers = int(rng.integers(config.min_layers, config.max_layers + 1))

    for _ in range(n_layers):
        primitive = primitives[int(rng.choice(len(primitives), p=weights))]
        patch = _transform_primitive(primitive.patch, config, rng)
        patch_height, patch_width = patch.shape

        if rng.random() < config.over_subject_fraction:
            # Bias toward the frame centre, where the tethered fly sits, so some
            # dust reliably lands on the animal.
            center_y = rng.normal(height / 2, height / 8)
            center_x = rng.normal(width / 2, width / 8)
            top = int(center_y - patch_height / 2)
            left = int(center_x - patch_width / 2)
        else:
            # Uniform placement; allow partial overhang off any edge.
            top = int(rng.integers(-patch_height // 2, height - patch_height // 2))
            left = int(rng.integers(-patch_width // 2, width - patch_width // 2))

        _paste_additive(canvas, patch, top, left)

    return np.clip(canvas, 0, 255).astype(np.uint8)


def compute_size_matched_weights(
    primitives: list[DustPrimitive],
    natural_areas: list[int],
    area_bin_edges: tuple[int, ...] = (20, 60, 200, 800),
) -> np.ndarray | None:
    """Sampling weights that reproduce the natural dust size distribution.

    Each primitive is binned by area; the target probability of a bin is its
    frequency among ``natural_areas``. Within a bin, primitives are equally
    likely, so sampling with these weights reproduces the measured size mix while
    still drawing on the full (balanced) library for variety.
    """
    if not natural_areas or not primitives:
        return None
    n_bins = len(area_bin_edges) + 1

    def bin_of(area: int) -> int:
        for index, edge in enumerate(area_bin_edges):
            if area <= edge:
                return index
        return len(area_bin_edges)

    target = np.zeros(n_bins)
    for area in natural_areas:
        target[bin_of(area)] += 1
    if target.sum() == 0:
        return None
    target /= target.sum()

    primitive_bins = np.array([bin_of(p.area) for p in primitives])
    bin_counts = np.array([(primitive_bins == b).sum() for b in range(n_bins)])
    weights = np.zeros(len(primitives), dtype=np.float64)
    for index, bin_index in enumerate(primitive_bins):
        if bin_counts[bin_index] > 0:
            weights[index] = target[bin_index] / bin_counts[bin_index]
    total = weights.sum()
    if total <= 0:
        return None
    return weights / total


def add_wall(
    frame: np.ndarray,
    walls: list[WallPrimitive],
    config: WallCompositingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add one wall residual to the frame, additively -- exactly like dust.

    Wall samples are full-frame residuals (fly removed) anchored to a frame
    border. They are only mirrored on the horizontal and/or vertical axis (never
    rotated or translated), so a border wall maps to another border and keeps its
    distance from every edge. The residual is then simply *added* to the frame
    (dark-field brightness adds), with no masking: because each residual already
    has a hole where its source fly was and the recordings are fly-centred, a
    mirrored wall lands in the periphery and adds almost nothing over a new centred
    fly -- and wherever it does graze the fly it just adds light, as real
    out-of-focus structure does. This avoids the hard, unrealistic cut-out a fly
    occlusion mask would leave.
    """
    if len(walls) == 0:
        return frame.copy()
    canvas = frame.astype(np.float32)
    height, width = frame.shape

    wall = walls[int(rng.integers(len(walls)))]
    layer = wall.layer.astype(np.float32)
    if layer.shape != (height, width):
        layer = cv2.resize(layer, (width, height))

    # Mirroring on both axes only -- no rotation, no translation, no scaling.
    if rng.random() < 0.5:
        layer = layer[:, ::-1]
    if rng.random() < 0.5:
        layer = layer[::-1, :]
    layer = np.ascontiguousarray(layer)

    layer = cv2.GaussianBlur(layer, (0, 0), rng.uniform(config.min_blur_sigma, config.max_blur_sigma))
    layer *= rng.uniform(config.min_intensity_scale, config.max_intensity_scale)

    return np.clip(canvas + layer, 0, 255).astype(np.uint8)


def add_artifacts_to_frames(
    frames: list[np.ndarray],
    dust_primitives: list[DustPrimitive],
    walls: list[WallPrimitive],
    dust_config: DustCompositingConfig,
    wall_config: WallCompositingConfig,
    dust_statistics,
    wall_statistics,
    seed: int,
    video_key: str,
) -> list[np.ndarray]:
    """Add dust and walls to every frame at their own data-derived probabilities.

    Dust and walls are independent: each frame gets dust with probability
    ``dust_statistics.dust_presence_rate`` (overridable via
    ``dust_config.dust_probability``) and a wall with probability
    ``wall_statistics.wall_presence_rate`` (overridable via
    ``wall_config.wall_probability``). When dust is added and
    ``dust_config.use_measured_distribution`` is True, the number of primitives
    and their sizes are drawn from the measured distributions; otherwise the
    manual ``min_layers``/``max_layers``/``sampling_power`` knobs are used. Both
    dust and walls are composited additively (no masking).

    ``dust_statistics`` / ``wall_statistics`` may be None to disable that artifact.
    """
    use_measured = dust_statistics is not None and dust_config.use_measured_distribution
    if use_measured:
        dust_weights = compute_size_matched_weights(dust_primitives, dust_statistics.dust_areas)
    else:
        dust_weights = compute_sampling_weights(dust_primitives, dust_config.sampling_power)

    output: list[np.ndarray] = []
    base = f"{seed}:{video_key}"
    for frame_index, frame in enumerate(frames):
        rng = np.random.default_rng(abs(hash(f"{base}:{frame_index}")) % (2**32))
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        result = gray

        add_dust_here = (
            len(dust_primitives) > 0
            and (
                dust_statistics.dust_present(rng, override=dust_config.dust_probability)
                if dust_statistics is not None
                else dust_config.dust_probability is None
                or rng.random() < dust_config.dust_probability
            )
        )
        if add_dust_here:
            n_dust = dust_statistics.sample_count(rng) if use_measured else None
            result = add_dust(result, dust_primitives, dust_config, rng, dust_weights, n_layers=n_dust)

        add_wall_here = (
            len(walls) > 0
            and wall_statistics is not None
            and wall_statistics.wall_present(rng, override=wall_config.wall_probability)
        )
        if add_wall_here:
            result = add_wall(result, walls, wall_config, rng)

        output.append(result if result.dtype == np.uint8 else result.astype(np.uint8))
    return output
