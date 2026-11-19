"""Rearing detection for aligned Spotlight behavior frames.

A fly is "rearing" when it lifts its head so the antennae are no longer clearly
visible. This is detected from a fixed head crop of the aligned frame with a small
pretrained classifier (MobileNetV3-small). The crop window is fixed per trial (at
the median neck position, in the aligned frame), so a frame on which the pose
keypoints are missing is still scored -- tracking often fails precisely because the
fly is rearing.

The production checkpoint stores several epoch snapshots; predictions are averaged
over them (a snapshot ensemble), which calibrates the otherwise volatile small-data
model. See :func:`load_checkpoint` for the checkpoint format.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn

# Fixed head-crop geometry (pixels in the aligned frame), as validated.
CROP_HALF = 130  # half-size -> 260 px crop
CROP_Y_OFFSET = -60  # shift the crop up from the neck toward the head
MODEL_INPUT_SIZE = 160  # the crop is resized to this before the network
CLASS_ORDER = ["not_rearing", "rearing"]  # index 1 == rearing (positive)


class RearingClassifier(nn.Module):
    """Pretrained MobileNetV3-small; grayscale replicated to 3 channels."""

    def __init__(self):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        self.net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.net.classifier[-1] = nn.Linear(
            self.net.classifier[-1].in_features, len(CLASS_ORDER)
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # x: (B, 1, H, W) in [0, 1]. Replicate to 3 channels and ImageNet-normalize.
        return self.net((x.repeat(1, 3, 1, 1) - self.mean) / self.std)


def create_model(device: torch.device | str = "cpu") -> RearingClassifier:
    """Build an untrained rearing classifier on ``device``."""
    return RearingClassifier().to(device)


def load_checkpoint(checkpoint_path: Path, device: torch.device | str = "cpu"):
    """Load the rearing production checkpoint.

    The checkpoint is a dict with:
        - ``snapshots``: list of ``state_dict`` (the ensemble members),
        - ``crop``: ``{half, y_offset, input_size}``,
        - ``class_order``: label order (index 1 == rearing).

    Returns ``(model, snapshots, crop)`` where ``model`` is a single
    :class:`RearingClassifier` whose weights are swapped per snapshot at inference.
    """
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
    snapshots = checkpoint["snapshots"]
    crop = checkpoint.get(
        "crop", {"half": CROP_HALF, "y_offset": CROP_Y_OFFSET, "input_size": MODEL_INPUT_SIZE}
    )
    model = create_model(device)
    model.load_state_dict(snapshots[0])
    return model, snapshots, crop


def median_neck(alignment_metadata_path: Path) -> np.ndarray:
    """Median neck position (x, y) in the aligned frame, over all frames."""
    with h5py.File(Path(alignment_metadata_path), "r") as f:
        post = f["keypoints_xy_post_alignment"]
        names = [
            n.decode() if isinstance(n, bytes) else n
            for n in post.attrs["keypoint_names"]
        ]
        keypoints = post[:]
    return np.nanmedian(keypoints[:, names.index("neck")], axis=0)


def head_crop(
    gray_frame: np.ndarray,
    neck_xy,
    half: int = CROP_HALF,
    y_offset: int = CROP_Y_OFFSET,
) -> np.ndarray | None:
    """Fixed head crop around ``neck_xy``; ``None`` if it would fall off the frame."""
    center_x = int(round(float(neck_xy[0])))
    center_y = int(round(float(neck_xy[1]))) + y_offset
    crop = gray_frame[
        center_y - half : center_y + half, center_x - half : center_x + half
    ]
    return crop if crop.shape == (2 * half, 2 * half) else None


def predict_rearing_probability(
    model: RearingClassifier,
    snapshots: list,
    crops: list[np.ndarray],
    device: torch.device | str,
    input_size: int = MODEL_INPUT_SIZE,
    batch_size: int = 128,
) -> np.ndarray:
    """Average P(rearing) over the snapshot ensemble for a list of head crops."""
    if not crops:
        return np.zeros(0, dtype=np.float32)
    batch = (
        np.stack([cv2.resize(c, (input_size, input_size)) for c in crops]).astype(
            np.float32
        )
        / 255.0
    )
    x = torch.from_numpy(batch)[:, None].to(device)
    accumulated = np.zeros(len(crops), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for state_dict in snapshots:
            model.load_state_dict(state_dict)
            for start in range(0, len(crops), batch_size):
                logits = model(x[start : start + batch_size])
                probability = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                accumulated[start : start + batch_size] += probability
    return (accumulated / len(snapshots)).astype(np.float32)
