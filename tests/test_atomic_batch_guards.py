"""Tests for the atomic-batch loader resolution guard (audit issue #48, I1-C).

`AtomicBatchDataset.load_atomic_batch_frames` reconstructs each per-variant
tile via a top-left crop `frame[:n_rows, start_col:end_col]`. That is only
correct when the on-disk tiles were serialized at exactly `image_size`. If an
atomic batch was extracted at a different (larger) per-variant resolution, the
crop silently grabs the wrong columns while the stored labels stay at the
extraction resolution -> corrupted training data. The loader must instead fail
loudly. These tests pin both behaviours against a REAL shipped 256-tiled batch.
"""

from pathlib import Path

import pytest
import torch

from poseforge.pose.data.synthetic.atomic_batch import AtomicBatchDataset

# The shipped atomic batches under this directory are tiled at 256x256 with
# 4 variants and a serialization spacing of 10 px.
ATOMIC_BATCHES_4VARIANTS_DIR = Path(
    "bulk_data/pose_estimation/atomic_batches/4variants"
)
N_VARIANTS = 4
SPACING = 10
TILE_SIZE = (256, 256)


def _find_one_atomic_batch_frames() -> Path:
    """Return the path to a single real `*_frames.mp4`, or skip the test."""
    if not ATOMIC_BATCHES_4VARIANTS_DIR.is_dir():
        pytest.skip(
            f"Real atomic batches not available at {ATOMIC_BATCHES_4VARIANTS_DIR}"
        )
    frames_videos = sorted(
        ATOMIC_BATCHES_4VARIANTS_DIR.rglob("atomicbatch*_frames.mp4")
    )
    if not frames_videos:
        pytest.skip(
            f"No atomicbatch *_frames.mp4 found under {ATOMIC_BATCHES_4VARIANTS_DIR}"
        )
    return frames_videos[0]


def test_load_atomic_batch_frames_matching_resolution():
    """The loader works on a real 256-tiled batch with image_size=(256, 256)."""
    frames_video = _find_one_atomic_batch_frames()

    atomic_batch = AtomicBatchDataset.load_atomic_batch_frames(
        frames_video,
        n_variants=N_VARIANTS,
        image_size=TILE_SIZE,
        n_channels=1,
        spacing=SPACING,
    )

    assert isinstance(atomic_batch, torch.Tensor)
    assert atomic_batch.dtype == torch.float32
    # (n_variants, n_frames, n_channels, height, width)
    assert atomic_batch.ndim == 5
    assert atomic_batch.shape[0] == N_VARIANTS
    assert atomic_batch.shape[2] == 1
    assert atomic_batch.shape[3:] == TILE_SIZE
    assert atomic_batch.shape[1] > 0  # at least one frame
    # Pixels are normalized into [0, 1] and the tile is not all-zero/all-one.
    assert atomic_batch.min() >= 0.0 and atomic_batch.max() <= 1.0
    assert 0.0 < float(atomic_batch.mean()) < 1.0


def test_load_atomic_batch_frames_resolution_mismatch_raises():
    """A mismatched image_size must raise a clear error, not mis-slice silently.

    The shipped tiles are 256x256; asking the loader to treat them as 512x512
    is exactly the I1-C footgun. The guard must refuse rather than top-left
    crop the wrong columns.
    """
    frames_video = _find_one_atomic_batch_frames()

    with pytest.raises(ValueError) as excinfo:
        AtomicBatchDataset.load_atomic_batch_frames(
            frames_video,
            n_variants=N_VARIANTS,
            image_size=(512, 512),
            n_channels=1,
            spacing=SPACING,
        )

    message = str(excinfo.value)
    # The error must name the mismatch and how to fix it so it is actionable.
    assert "inconsistent" in message
    assert "image_size=(512, 512)" in message
    assert "re-extract" in message


def test_infer_stored_variant_geometry_inversion():
    """Unit-level check of the geometry inversion that backs the guard."""
    infer = AtomicBatchDataset._infer_stored_variant_geometry

    # Real shipped layout: 4 tiles of 256 + 3*10 spacing = 1054 -> round16 1056.
    assert infer(1056, 256, SPACING) == (4, 1056)

    # A 512-tiled, 4-variant video: 4*512 + 3*10 = 2078 -> round16 2080.
    assert infer(2080, 512, SPACING) == (4, 2080)

    # Asking for 512-wide tiles against a 256-tiled (1056-wide) video is
    # incompatible -> no integer variant count reproduces the width.
    n_variants, _ = infer(1056, 512, SPACING)
    assert n_variants is None

    # ...and vice versa: 256-wide tiles against a 512-tiled (2080-wide) video.
    n_variants, _ = infer(2080, 256, SPACING)
    assert n_variants is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
