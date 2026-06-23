"""Tests for the depth heads of ``Pose2p5DModel`` (audit #48, I1-B/I1-D/I1-G).

These cover:
  (a) both depth-head types ("global" and "spatial") run a forward on a small
      random batch and produce finite depth_logits of shape
      (N, n_kp, depth_n_bins);
  (b) a GROUNDING test that the spatial head's depth output for a keypoint
      actually depends on the predicted (x, y) location (whereas the global
      avg-pool head is location-invariant);
  (c) gradients flow to the new spatial head's parameters.

Import note: ``poseforge.pose.keypoints3d`` -> pipeline -> atomic_batch imports
a private pvio symbol that is absent in this venv, so we shim it before
importing the model. We also make sure ``poseforge`` resolves to THIS worktree's
``src`` (not a separately editable-installed copy), so the tests exercise the
code under review.
"""

import sys
from pathlib import Path

# Ensure we import the worktree copy of poseforge (this file lives in
# <repo>/tests/, so the package is at <repo>/src). Prepend so it wins over any
# editable install that may point elsewhere.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Shim the private pvio symbol pulled in transitively by the keypoints3d package.
import pvio.io as _pio  # noqa: E402

if not hasattr(_pio, "_default_ffmpeg_params_for_video_writing"):
    _pio._default_ffmpeg_params_for_video_writing = [
        "-crf",
        "15",
        "-preset",
        "slow",
        "-profile:v",
        "high",
        "-level",
        "4.0",
    ]

import torch  # noqa: E402

from poseforge.pose.keypoints3d.model import (  # noqa: E402
    Pose2p5DModel,
    SpatialDepthHead,
)
from poseforge.pose.common import ResNetFeatureExtractor  # noqa: E402

# Tiny CPU config. Note hidden_channels (32) is a multiple of groupnorm_n_groups
# (8) with >1 channel/group, matching the production ratio (128/32).
N_KP = 4
DEPTH_N_BINS = 8
HIDDEN_CHANNELS = 32
GN_GROUPS = 8
CORE_CHANNELS = 64


def _build_model(depth_head_type: str) -> Pose2p5DModel:
    # Feature extractor WITHOUT weights (random init) — fine for shape/grad tests.
    feature_extractor = ResNetFeatureExtractor()
    return Pose2p5DModel(
        n_keypoints=N_KP,
        feature_extractor=feature_extractor,
        depth_n_bins=DEPTH_N_BINS,
        depth_min=-103.0,
        depth_max=-99.0,
        xy_temperature=0.8,
        depth_temperature=0.8,
        upsample_core_out_channels=CORE_CHANNELS,
        depth_hidden_channels=HIDDEN_CHANNELS,
        depth_head_type=depth_head_type,
        groupnorm_n_groups=GN_GROUPS,
    )


def test_both_head_types_forward_shapes_and_finite():
    """(a) Both head types produce finite (N, n_kp, depth_n_bins) logits."""
    batch_size = 2
    x = torch.rand(batch_size, 3, 64, 64)
    for depth_head_type in ("global", "spatial"):
        model = _build_model(depth_head_type).eval()
        with torch.no_grad():
            out = model(x)
        depth_logits = out["depth_logits"]
        assert depth_logits.shape == (batch_size, N_KP, DEPTH_N_BINS), (
            f"[{depth_head_type}] depth_logits shape {tuple(depth_logits.shape)} "
            f"!= {(batch_size, N_KP, DEPTH_N_BINS)}"
        )
        assert torch.isfinite(depth_logits).all(), (
            f"[{depth_head_type}] depth_logits contains non-finite values"
        )
        # Sanity: other outputs keep their contract too.
        assert out["pred_depth"].shape == (batch_size, N_KP)
        assert out["xy_heatmaps"].shape[:2] == (batch_size, N_KP)


def test_global_vs_spatial_head_class():
    """The two configs build genuinely different head modules (so checkpoints
    are not interchangeable — see PR notes)."""
    global_model = _build_model("global")
    spatial_model = _build_model("spatial")
    assert not isinstance(global_model.depth_head, SpatialDepthHead)
    assert isinstance(spatial_model.depth_head, SpatialDepthHead)
    # state_dict keys differ -> a "global" checkpoint cannot load into "spatial".
    assert set(global_model.depth_head.state_dict().keys()) != set(
        spatial_model.depth_head.state_dict().keys()
    )


def test_spatial_head_depends_on_location():
    """(b) GROUNDING: the spatial head reads features AT the predicted (x, y),
    so its depth logits change when the keypoint location moves. A global
    avg-pool head would be ~invariant to location on the same feature map.

    We test ``SpatialDepthHead`` directly with a feature map whose content
    differs between the left and right halves (and whose global average is
    exactly zero, so a global-pool head would yield identical output for any
    location)."""
    torch.manual_seed(0)
    channels, height, width = 16, 32, 32
    head = SpatialDepthHead(
        in_channels=channels,
        hidden_channels=HIDDEN_CHANNELS,
        depth_n_bins=DEPTH_N_BINS,
        groupnorm_n_groups=GN_GROUPS,
        pose_head_init_std=1e-3,
    ).eval()

    # Spatially-varying feature map: +1 in the left half, -1 in the right half.
    # Its spatial mean per channel is exactly 0 -> a global avg-pool descriptor
    # is location-invariant (and identical) for this input.
    feature_map = torch.zeros(1, channels, height, width)
    feature_map[:, :, :, : width // 2] = 1.0
    feature_map[:, :, :, width // 2 :] = -1.0
    assert torch.allclose(
        feature_map.mean(dim=(2, 3)), torch.zeros(1, channels)
    ), "test fixture should have zero global average per channel"

    xy_left = torch.tensor([[[5.0, height / 2.0]]])  # x in left half
    xy_right = torch.tensor([[[width - 6.0, height / 2.0]]])  # x in right half

    with torch.no_grad():
        logits_left = head(feature_map, xy_left)
        logits_right = head(feature_map, xy_right)

    assert logits_left.shape == (1, 1, DEPTH_N_BINS)
    max_abs_diff = (logits_left - logits_right).abs().max().item()
    # Decisively above float noise: location genuinely drives the output.
    assert max_abs_diff > 1e-4, (
        "spatial depth head output did not change when the keypoint location "
        f"moved across the feature map (max|diff|={max_abs_diff:.2e}); it is "
        "behaving like a location-invariant global head."
    )


def test_spatial_head_full_model_follows_heatmap_peak():
    """(b, full-model variant) When the heatmap peaks move to different image
    locations, the depth logits change through the full model's spatial head.
    This exercises the soft-argmax -> grid_sample -> depth-head plumbing end to
    end (not just the head in isolation).

    A freshly-initialized model has a near-zero-weight heatmap head (small-std
    init), so its heatmaps are nearly uniform and the soft-argmax barely moves
    for any input — that is an artifact of being untrained, not of the spatial
    head. To get a meaningful end-to-end signal without training, we amplify
    the (single-conv) heatmap head's weights so the peaks actually localize,
    then feed two clearly different images."""
    torch.manual_seed(1)
    model = _build_model("spatial").eval()

    # Make the heatmaps responsive to the input so the soft-argmax localizes.
    # (Default init is intentionally near-zero; that is not what we are testing.)
    assert isinstance(model.heatmap_head, torch.nn.Conv2d)
    with torch.no_grad():
        model.heatmap_head.weight.mul_(500.0)

    x_a = torch.rand(1, 3, 64, 64)
    # A clearly different image (inverted + shifted) to move the predicted xy.
    x_b = 1.0 - torch.roll(x_a, shifts=(16, 16), dims=(2, 3))

    with torch.no_grad():
        out_a = model(x_a)
        out_b = model(x_b)

    # Predicted locations should differ...
    xy_diff = (out_a["pred_xy"] - out_b["pred_xy"]).abs().max().item()
    # ...and the spatially-grounded depth logits should differ as a result.
    depth_diff = (out_a["depth_logits"] - out_b["depth_logits"]).abs().max().item()
    assert xy_diff > 1e-2, f"predicted xy did not move between inputs ({xy_diff:.2e})"
    assert depth_diff > 1e-6, (
        f"spatial depth logits did not respond to changed inputs ({depth_diff:.2e})"
    )


def test_spatial_head_gradients_flow():
    """(c) Backward pass produces non-zero gradients on the spatial head."""
    model = _build_model("spatial").train()
    x = torch.rand(2, 3, 64, 64)
    out = model(x)
    loss = out["depth_logits"].sum()
    loss.backward()

    head_params = list(model.depth_head.parameters())
    assert len(head_params) > 0, "spatial head has no parameters"
    grads_present = [
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in head_params
    ]
    assert all(grads_present), (
        "expected a finite, non-zero gradient on every spatial depth-head "
        f"parameter; got {grads_present}"
    )
