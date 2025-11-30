import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from pathlib import Path

import poseforge.pose.pose6d.config as config
from poseforge.pose.common import ResNetFeatureExtractor, DecoderBlock


class Pose6DModel(nn.Module):
    def __init__(
        self,
        n_segments: int,
        feature_extractor: ResNetFeatureExtractor,
        final_upsampler_n_hidden_channels: int,
        pose6d_head_hidden_sizes: list[int],
    ):
        super(Pose6DModel, self).__init__()
        self.n_segments = n_segments
        self.feature_extractor = feature_extractor
        self.final_upsampler_n_hidden_channels = final_upsampler_n_hidden_channels
        self.pose6d_head_hidden_sizes = pose6d_head_hidden_sizes

        # Create decoder (decoder1/2/3/4 mirror encoder layers 1/2/3/4)
        # Note that when upsampling, we actually run decoder4 first, decoder1 last
        # Also note that the number of channels along the decoding path is higher than
        # in keypoints3d and bodyseg models. This because here we will eventually pool
        # the features globally for FC pose6d heads, so we want more channels to retain
        # information.
        self.dec_layer4 = DecoderBlock(512, 256, 512)
        self.dec_layer3 = DecoderBlock(512, 128, 256)
        self.dec_layer2 = DecoderBlock(256, 64, 256)
        self.dec_layer1 = DecoderBlock(256, 64, final_upsampler_n_hidden_channels)

        # 6D pose prediction head (one per segment)
        self.pose6d_heads = nn.ModuleList(
            [self._make_pose6d_head() for _ in range(n_segments)]
        )

    @classmethod
    def from_config(
        cls, architecture_config: config.ModelArchitectureConfig | Path | str
    ) -> "Pose6DModel":
        # Load from file if config is a path
        if isinstance(architecture_config, (str, Path)):
            architecture_config = config.ModelArchitectureConfig.load(
                architecture_config
            )
            logger.info(f"Loaded model architecture config from {architecture_config}")

        # Initialize feature extractor (WITHOUT WEIGHTS at this step!)
        feature_extractor = ResNetFeatureExtractor()

        # Initialize Pose6DModel (self) from config (WITHOUT WEIGHTS at this step!)
        try:
            # Parse pose6d_head_hidden_sizes from string
            # (don't specify directly as list[int] in YAML - mutability issues)
            pose6d_head_hidden_sizes = [
                int(x.strip())
                for x in architecture_config.pose6d_head_hidden_sizes.split(",")
            ]
        except ValueError as e:
            logger.critical(
                f"Invalid pose6d_head_hidden_sizes in ModelArchitectureConfig: {e}. "
                f"Expected a comma-separated list of integers."
            )
            raise e
        obj = cls(
            n_segments=architecture_config.n_segments,
            feature_extractor=feature_extractor,
            final_upsampler_n_hidden_channels=architecture_config.final_upsampler_n_hidden_channels,
            pose6d_head_hidden_sizes=pose6d_head_hidden_sizes,
        )

        logger.info("Initialized Pose6DModel from architecture config")
        return obj

    def load_weights_from_config(
        self, weights_config: config.ModelWeightsConfig | Path | str
    ):
        # Load from file if config is given as a path
        if isinstance(weights_config, (str, Path)):
            weights_config = config.ModelWeightsConfig.load(weights_config)
            logger.info(f"Loaded model weights config from {weights_config}")

        # Check if config has either feature extractor weights or model weights
        if (
            weights_config.feature_extractor_weights is None
            and weights_config.model_weights is None
        ):
            logger.warning("weights_config contains nothing useful. No action taken")
            return

        # If full model weights are provided, load them directly
        if weights_config.model_weights is not None:
            checkpoint_path = Path(weights_config.model_weights)
            if not checkpoint_path.exists():
                logger.critical(f"Model weights path {checkpoint_path} does not exist")
                raise FileNotFoundError(f"Model weights file does not exist")
            weights = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(weights)
            logger.info(
                f"Loaded Pose6DModel weights (inc. feature extractor) "
                f"from {checkpoint_path}"
            )
            return

        # Otherwise, init feature extractor weights if provided
        self.feature_extractor = ResNetFeatureExtractor(
            # Path, str, or "IMAGENET1K_V1"
            weights=weights_config.feature_extractor_weights
        )
        logger.info(
            f"Initialized feature extractor weights from "
            f"{weights_config.feature_extractor_weights}"
        )

    def _make_pose6d_head(self) -> nn.Module:
        all_layers = []
        n_channels_in = self.final_upsampler_n_hidden_channels
        for hidden_size in self.pose6d_head_hidden_sizes:
            layers_within_block = [
                nn.Linear(n_channels_in, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
            ]
            all_layers.extend(layers_within_block)
            n_channels_in = hidden_size
        # 3 translation + 4 rotation (quaternion)
        all_layers.append(nn.Linear(n_channels_in, 7))
        return nn.Sequential(*all_layers)

    def forward(self, input_img: torch.Tensor, bodyseg_probs: torch.Tensor) -> dict:
        # Extract features with ResNet backbone and upsample to 128x128
        features = self.feature_extractor(input_img)
        x = self.dec_layer4(features)
        x = self.dec_layer3(x)
        x = self.dec_layer2(x)
        x = self.dec_layer1(x)
        # Now x has shape (B, final_upsampler_n_hidden_channels, 128, 128)

        # Process each segment separately
        translation_pred_list = []
        quaternion_pred_list = []

        for seg_idx in range(self.n_segments):
            # Confidence-weighted global average pooling
            mask_probs = bodyseg_probs[:, seg_idx, :, :].unsqueeze(1)  # (B, 1, H, W)
            weighted_features = x * mask_probs  # (B, C, H, W)
            feature_sums = weighted_features.sum(dim=(2, 3))  # (B, C)
            confidence_sums = mask_probs.sum(dim=(2, 3)) + 1e-6  # (B, 1)
            pooled_features = feature_sums / confidence_sums.clamp_min(1e-6)  # (B, C)

            # Predict 6D pose
            pose_pred = self.pose6d_heads[seg_idx](pooled_features)  # (B, 7)
            translation_pred = pose_pred[:, 0:3]  # (B, 3)
            quaternion_pred = pose_pred[:, 3:7]  # (B, 4)
            # Normalize quaternion to unit length
            quaternion_pred = F.normalize(quaternion_pred, p=2, dim=1)

            translation_pred_list.append(translation_pred)
            quaternion_pred_list.append(quaternion_pred)

        # Stack predictions into single tensors of shape (B, n_segments, 3 or 4)
        translation_pred_all = torch.stack(translation_pred_list, dim=1)
        quaternion_pred_all = torch.stack(quaternion_pred_list, dim=1)

        return translation_pred_all, quaternion_pred_all


class Pose6DLoss(nn.Module):
    def __init__(self, translation_weight: float = 1.0, rotation_weight: float = 1.0):
        super(Pose6DLoss, self).__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight

    def forward(
        self,
        translation_pred: torch.Tensor,
        quaternion_pred: torch.Tensor,
        translation_label: torch.Tensor,
        quaternion_label: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        # If the segment is too small, don't include it in loss computation
        translation_pred = translation_pred[valid_mask, ...]
        quaternion_pred = quaternion_pred[valid_mask, ...]
        translation_label = translation_label[valid_mask, ...]
        quaternion_label = quaternion_label[valid_mask, ...]

        # Compute losses
        translation_loss = self.translation_loss(
            translation_pred.view(-1, 3), translation_label.view(-1, 3)
        )
        rotation_loss = self.rotation_loss(
            quaternion_pred.view(-1, 4), quaternion_label.view(-1, 4)
        )
        return (
            self.translation_weight * translation_loss
            + self.rotation_weight * rotation_loss
        )

    @classmethod
    def create_from_config(
        cls, loss_config: config.LossConfig | Path | str
    ) -> "Pose6DLoss":
        # Load from file if config is a path
        if isinstance(loss_config, (str, Path)):
            loss_config = config.LossConfig.load(loss_config)
            logger.info(f"Loaded loss config from {loss_config}")

        obj = cls(
            translation_weight=loss_config.translation_weight,
            rotation_weight=loss_config.rotation_weight,
        )
        logger.info("Initialized Pose6DLoss from loss config")
        return obj

    @staticmethod
    def translation_loss(
        translation_pred: torch.Tensor,
        translation_label: torch.Tensor,
    ) -> torch.Tensor:
        """Computes L1 translation loss. Both predictions and labels are should have
        shape (N, 3), where N is the number of valid samples."""
        return F.l1_loss(translation_pred, translation_label)

    @staticmethod
    def rotation_loss(
        quaternion_pred: torch.Tensor,
        quaternion_label: torch.Tensor,
    ) -> torch.Tensor:
        """Computes rotation loss based on quaternion dot product. Both predictions and
        labels are should have shape (N, 4), where N is the number of valid samples."""
        # Normalize both quaternions to unit length (note: dim 1 is quat components)
        quaternion_pred = F.normalize(quaternion_pred, p=2, dim=1)
        quaternion_label = F.normalize(quaternion_label, p=2, dim=1)

        # Compute absolute quaternion dot product (note: dim 1 is quat components)
        dot_prod = torch.abs(torch.sum(quaternion_pred * quaternion_label, dim=1))

        return torch.mean(1.0 - dot_prod)
