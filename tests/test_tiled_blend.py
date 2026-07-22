"""Tests for the tiled style-transfer blending weight maps (audit issue #48).

These tests cover the pure-numpy tile blending helpers in
``poseforge.style_transfer.tiled_inference`` and the resolution-parsing helper
that the (GAN) inference paths rely on. They do NOT exercise the GAN itself.

Issue #48 findings under test:

- I2-B: the default ``uniform`` weight map does no feathering, and the
  ``cosine``/``pyramid`` windows are zero at the tile border (so a seam pixel
  gets ~0 weight from one of the two overlapping tiles) and do not form a
  partition of unity. The new ``feather`` window must be a partition of unity on
  the 50%-overlap grid AND strictly nonzero at the seam. Mirror-padded
  out-of-frame tile regions must receive zero weight.
- I2-A: the non-tiled inference path must source the resolution / preprocessing
  from the trained model's ``train_options.json`` (``crop_size`` becomes
  ``image_side_length``) rather than hard-coding 256.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import the production tiled-inference helpers without pulling in the CUT GAN
# library (the `cut` package is not installed in this test environment, and
# `poseforge.style_transfer.__init__` eagerly imports it via cut_inference).
# We stub the `cut` submodules that cut_inference imports at module load time;
# the helpers under test are pure numpy and never touch them.
# ---------------------------------------------------------------------------
def _load_tiled_inference_module():
    for name in [
        "cut",
        "cut.models",
        "cut.models.cut_model",
        "cut.options",
        "cut.options.option_stats",
        "cut.data",
        "cut.data.base_dataset",
    ]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["cut.models.cut_model"].CUTModel = object
    sys.modules["cut.options.option_stats"].OptionsWrapper = object
    sys.modules["cut.data.base_dataset"].get_transform = lambda *a, **k: None

    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from poseforge.style_transfer import tiled_inference  # noqa: WPS433

    return tiled_inference


tiled_inference = _load_tiled_inference_module()

_make_tile_weight_map = tiled_inference._make_tile_weight_map
_normalize_weight_type = tiled_inference._normalize_weight_type
_feather_ramp_1d = tiled_inference._feather_ramp_1d
_clamp_weights_to_frame = tiled_inference._clamp_weights_to_frame
_compute_half_overlap_tile_starts = tiled_inference._compute_half_overlap_tile_starts


def _load_style_transfer_util_module():
    """Load style_transfer/util.py standalone (it is pure stdlib)."""
    repo_root = Path(__file__).resolve().parent.parent
    util_path = repo_root / "src" / "poseforge" / "style_transfer" / "util.py"
    spec = importlib.util.spec_from_file_location("_st_util_for_test", util_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


st_util = _load_style_transfer_util_module()


# ---------------------------------------------------------------------------
# Helpers that mirror the real overlap-add used by ``stylize_frame_tiled``.
# ---------------------------------------------------------------------------
def _overlap_add_1d(weights_1d: np.ndarray, tile_size: int, n_tiles: int) -> np.ndarray:
    stride = tile_size // 2
    full = stride * (n_tiles - 1) + tile_size
    acc = np.zeros(full, dtype=np.float64)
    for k in range(n_tiles):
        start = k * stride
        acc[start : start + tile_size] += weights_1d
    return acc


def _weight_map_1d(tile_size: int, weight_type: str) -> np.ndarray:
    """Return the pure 1D window the code blends along each axis.

    ``_make_tile_weight_map`` builds the 2D map as ``np.outer(y_window,
    x_window)``; the per-axis windows are what determine the overlap-add
    behaviour, so we reproduce them directly (matching the exact formulas used in
    ``_make_tile_weight_map``) rather than slicing the 2D map, whose extra
    normalisation entangles the two axes.
    """
    if weight_type == "feather":
        return _feather_ramp_1d(tile_size).astype(np.float64)

    t = np.linspace(0.0, 1.0, tile_size)
    if weight_type == "cosine":
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * t)
    if weight_type == "pyramid":
        return 1.0 - np.abs(2.0 * t - 1.0)
    if weight_type == "uniform":
        return np.ones(tile_size, dtype=np.float64)
    raise ValueError(f"unhandled weight_type in test helper: {weight_type}")


# ===========================================================================
# (c) Existing weight types keep working -- no regression in normalization.
# ===========================================================================
class TestNormalizeWeightType:
    def test_all_supported_types_pass_through(self):
        for name in ["feather", "uniform", "cosine", "gaussian", "pyramid"]:
            assert _normalize_weight_type(name) == name

    def test_case_and_whitespace_insensitive(self):
        assert _normalize_weight_type("  FEATHER ") == "feather"
        assert _normalize_weight_type("Uniform") == "uniform"

    def test_common_misspelling_of_gaussian_is_corrected(self):
        assert _normalize_weight_type("guassian") == "gaussian"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            _normalize_weight_type("definitely-not-a-window")

    def test_every_legacy_type_still_builds_a_weight_map(self):
        for name in ["uniform", "cosine", "gaussian", "pyramid"]:
            wm = _make_tile_weight_map(16, 16, name)
            assert wm.shape == (16, 16, 1)
            assert np.all(wm > 0.0)  # legacy maps are clipped to a tiny positive floor
            assert wm.max() == pytest.approx(1.0)


# ===========================================================================
# (a) The new feather window: partition of unity + nonzero at seams,
#     contrasted with cosine/pyramid which collapse to ~0 at the border.
# ===========================================================================
class TestFeatherWindow:
    @pytest.mark.parametrize("tile_size", [8, 16, 64, 256])
    def test_feather_is_nonzero_at_borders(self, tile_size):
        w = _feather_ramp_1d(tile_size)
        assert w[0] > 0.0
        assert w[-1] > 0.0
        # The border value is the first midpoint of the ramp: 0.5 / (tile/2).
        expected_border = 0.5 / (tile_size // 2)
        assert w[0] == pytest.approx(expected_border)
        assert w[-1] == pytest.approx(expected_border)

    @pytest.mark.parametrize("tile_size", [8, 16, 64])
    def test_feather_1d_is_partition_of_unity(self, tile_size):
        w = _weight_map_1d(tile_size, "feather")
        acc = _overlap_add_1d(w, tile_size, n_tiles=8)
        # Strip the first/last tile, which lie partly off the "image" and so are
        # only singly covered (handled by renormalisation in production).
        interior = acc[tile_size:-tile_size]
        assert interior.size > 0
        np.testing.assert_allclose(interior, 1.0, atol=1e-5)

    @pytest.mark.parametrize("tile_size", [8, 16, 64])
    def test_feather_2d_is_partition_of_unity(self, tile_size):
        wm = _make_tile_weight_map(tile_size, tile_size, "feather")[..., 0]
        stride = tile_size // 2
        n_tiles = 6
        full = stride * (n_tiles - 1) + tile_size
        acc = np.zeros((full, full), dtype=np.float64)
        for i in range(n_tiles):
            for j in range(n_tiles):
                acc[
                    i * stride : i * stride + tile_size,
                    j * stride : j * stride + tile_size,
                ] += wm
        interior = acc[tile_size:-tile_size, tile_size:-tile_size]
        assert interior.size > 0
        np.testing.assert_allclose(interior, 1.0, atol=1e-5)

    def test_feather_seam_pixels_get_meaningful_weight_from_both_tiles(self):
        # On a 50%-overlap grid the seam between two tiles is the overlap centre.
        # Each of the two overlapping tiles must contribute a non-negligible
        # weight there (unlike cosine/pyramid, which contribute ~0 from one side).
        tile_size = 16
        w = _weight_map_1d(tile_size, "feather")
        stride = tile_size // 2
        # Tile A occupies [0, tile_size); tile B occupies [stride, stride+tile).
        # The overlap is [stride, tile_size). Inspect its centre.
        overlap_centre = stride + (tile_size - stride) // 2  # global coordinate
        a_local = overlap_centre  # tile A starts at 0
        b_local = overlap_centre - stride  # tile B starts at stride
        weight_from_a = w[a_local]
        weight_from_b = w[b_local]
        assert weight_from_a > 0.2
        assert weight_from_b > 0.2
        assert (weight_from_a + weight_from_b) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_and_pyramid_collapse_to_zero_at_border(self):
        # Demonstrates the I2-B problem with the non-default windows: the seam
        # pixel receives ~0 weight from one of the two overlapping tiles.
        tile_size = 16
        for name in ["cosine", "pyramid"]:
            w = _weight_map_1d(tile_size, name)
            # These windows clip to a tiny positive floor (1e-6 after rescale),
            # i.e. effectively zero at the border -- far below the feather's
            # border weight.
            assert w[0] < 1e-3
            assert w[-1] < 1e-3
        feather_border = _weight_map_1d(tile_size, "feather")[0]
        assert feather_border > 0.05  # strictly, meaningfully nonzero

    def test_cosine_is_not_a_partition_of_unity(self):
        # Contrast: cosine's overlap-add is not constant across the interior.
        tile_size = 16
        w = _weight_map_1d(tile_size, "cosine")
        acc = _overlap_add_1d(w, tile_size, n_tiles=8)
        interior = acc[tile_size:-tile_size]
        spread = interior.max() - interior.min()
        assert spread > 0.05  # visibly non-uniform -> seams/banding


# ===========================================================================
# (b) Mirror-padded / out-of-frame regions get exactly zero weight.
# ===========================================================================
class TestOutOfFrameWeightClamping:
    def test_top_left_out_of_frame_region_is_zeroed(self):
        tile_size = 8
        weight_map = _make_tile_weight_map(tile_size, tile_size, "feather")
        height = width = 20
        # A tile that hangs off the top-left corner: rows/cols [0:2] are padded.
        y, x = -2, -3
        masked, (y0, y1, x0, x1), (ty0, ty1, tx0, tx1) = _clamp_weights_to_frame(
            weight_map, y, x, height, width
        )

        # In-frame destination starts at the image origin.
        assert (y0, x0) == (0, 0)
        # The off-frame tile rows/cols carry zero weight.
        assert ty0 == 2 and tx0 == 3
        assert np.all(masked[:ty0, :] == 0.0)
        assert np.all(masked[:, :tx0] == 0.0)
        # The in-frame portion keeps its (positive) feather weights.
        assert np.all(masked[ty0:ty1, tx0:tx1] > 0.0)
        # Total weight equals only the in-frame contribution.
        assert masked.sum() == pytest.approx(weight_map[ty0:ty1, tx0:tx1].sum())

    def test_bottom_right_out_of_frame_region_is_zeroed(self):
        tile_size = 8
        weight_map = _make_tile_weight_map(tile_size, tile_size, "uniform")
        height = width = 10
        # Tile placed so it spills past the bottom-right edge.
        y, x = 6, 5  # tile spans rows [6:14], cols [5:13]; frame is 10x10
        masked, (y0, y1, x0, x1), (ty0, ty1, tx0, tx1) = _clamp_weights_to_frame(
            weight_map, y, x, height, width
        )
        assert (y1, x1) == (10, 10)
        # Rows/cols beyond the in-frame slice are zeroed.
        assert np.all(masked[ty1:, :] == 0.0)
        assert np.all(masked[:, tx1:] == 0.0)
        # In-frame region keeps full (uniform) weight.
        assert np.all(masked[ty0:ty1, tx0:tx1] == 1.0)

    def test_fully_in_frame_tile_is_unchanged(self):
        tile_size = 8
        weight_map = _make_tile_weight_map(tile_size, tile_size, "feather")
        height = width = 64
        y, x = 10, 12
        masked, _, _ = _clamp_weights_to_frame(weight_map, y, x, height, width)
        np.testing.assert_array_equal(masked, weight_map)

    def test_edge_tile_from_real_grid_zeroes_padded_rows(self):
        # Use the actual tile grid. The very first start (== -tile_size for
        # phase 0) is fully out of frame; the next one is partially padded above
        # the frame -- a genuine mirror-padded edge tile whose padded rows must
        # be zeroed.
        tile_size = 8
        height = width = 24
        weight_map = _make_tile_weight_map(tile_size, tile_size, "feather")
        y_starts = _compute_half_overlap_tile_starts(height, tile_size, phase=0)

        # First tile is entirely off-frame -> all-zero weight, nothing accumulated.
        fully_oof_start = y_starts[0]
        assert fully_oof_start + tile_size <= 0
        masked_oof, (y0, y1, x0, x1), _ = _clamp_weights_to_frame(
            weight_map, fully_oof_start, 0, height, width
        )
        assert y1 <= y0  # empty in-frame slice
        assert np.all(masked_oof == 0.0)

        # Second tile straddles the top edge: rows above the frame are padded.
        partial_start = y_starts[1]
        assert partial_start < 0 < partial_start + tile_size
        masked, (y0, y1, _x0, _x1), (ty0, _ty1, _tx0, _tx1) = _clamp_weights_to_frame(
            weight_map, partial_start, 0, height, width
        )
        assert y0 == 0
        assert ty0 == -partial_start  # padded rows skipped
        assert np.all(masked[:ty0, :] == 0.0)
        assert np.all(masked[ty0:, :] >= 0.0)
        assert masked[ty0:].sum() > 0.0


# ===========================================================================
# I2-A: resolution parsing helper sources crop_size from train_options.json.
# ===========================================================================
class TestTrainedResolutionParsing:
    def _write_train_options(self, tmp_path: Path, values: dict) -> Path:
        # parse_hyperparameters_from_checkpoint_path expects the json at
        # checkpoint_path.parent.parent / "train_options.json".
        model_dir = tmp_path / "some_model"
        ckpt_dir = model_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (model_dir / "train_options.json").write_text(json.dumps({"values": values}))
        return ckpt_dir / "100_net_G.pth"

    def test_image_side_length_comes_from_crop_size(self, tmp_path):
        ckpt = self._write_train_options(
            tmp_path,
            {
                "netG": "smallstylegan2",
                "ngf": 16,
                "crop_size": 512,
                "load_size": 512,
                "preprocess": "resize_and_crop",
                "n_epochs_decay": 0,
            },
        )
        hparams = st_util.parse_hyperparameters_from_checkpoint_path(ckpt)
        # The trained crop_size (512), NOT the hard-coded 256, is the resolution.
        assert hparams["image_side_length"] == 512
        assert hparams["preprocess_opt"]["crop_size"] == 512
        assert hparams["preprocess_opt"]["load_size"] == 512
        assert hparams["preprocess_opt"]["preprocess"] == "resize_and_crop"

    def test_finetune_load_size_used_when_decaying(self, tmp_path):
        ckpt = self._write_train_options(
            tmp_path,
            {
                "netG": "resnet_9blocks",
                "ngf": 64,
                "crop_size": 256,
                "load_size": 286,
                "finetune_load_size": 300,
                "preprocess": "resize_and_crop",
                "n_epochs_decay": 50,
            },
        )
        hparams = st_util.parse_hyperparameters_from_checkpoint_path(ckpt)
        assert hparams["image_side_length"] == 256
        assert hparams["preprocess_opt"]["load_size"] == 300

    def test_missing_train_options_raises(self, tmp_path):
        # Mirrors the fallback branch in run_inference.resolve_trained_resolution.
        ckpt_dir = tmp_path / "model" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        ckpt = ckpt_dir / "100_net_G.pth"
        with pytest.raises(FileNotFoundError):
            st_util.parse_hyperparameters_from_checkpoint_path(ckpt)
