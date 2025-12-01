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
        n_attention_gated_feature_channels: int,
        n_global_feature_channels: int,
        camera_distance: float,
    ):
        super(Pose6DModel, self).__init__()
        self.n_segments = n_segments
        self.feature_extractor = feature_extractor
        self.n_attention_gated_feature_channels = n_attention_gated_feature_channels
        self.n_global_feature_channels = n_global_feature_channels
        self.n_feature_channels_total = (
            n_attention_gated_feature_channels + n_global_feature_channels
        )
        self.camera_distance = camera_distance
        # MuJoCo convention: in front of camera = negative z, so floor is at
        # (*, *, -camera_distance)
        self.z_center = -camera_distance

        # Create decoder (decoder1/2/3/4 mirror encoder layers 1/2/3/4)
        # Note that when upsampling, we actually run decoder4 first, decoder1 last
        # Also note that the number of channels along the decoding path is higher than
        # in keypoints3d and bodyseg models. This because here we will eventually pool
        # the features globally for FC pose6d heads, so we want more channels to retain
        # information.
        self.dec_layer4 = DecoderBlock(512, 256, 512)
        self.dec_layer3 = DecoderBlock(512, 128, 256)
        self.dec_layer2 = DecoderBlock(256, 64, self.n_feature_channels_total)

        # Attention layer
        if n_attention_gated_feature_channels > 0:
            _attention_heads = [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.n_feature_channels_total + n_segments,
                        out_channels=128,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
                    nn.Sigmoid(),
                )
                for _ in range(n_segments)
            ]
            self.attention_heads = nn.ModuleList(_attention_heads)
        else:
            self.attention_heads = None

        # 6D pose prediction head (one per segment)
        _pose6d_heads = [
            nn.Sequential(
                nn.Linear(in_features=self.n_feature_channels_total, out_features=512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(in_features=512, out_features=256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(in_features=256, out_features=7),
            )
            for _ in range(n_segments)
        ]
        self.pose6d_heads = nn.ModuleList(_pose6d_heads)

    @classmethod
    def create_from_config(
        cls, arch_config: config.ModelArchitectureConfig | Path | str
    ) -> "Pose6DModel":
        # Load from file if config is a path
        if isinstance(arch_config, (str, Path)):
            arch_config = config.ModelArchitectureConfig.load(arch_config)
            logger.info(f"Loaded model architecture config from {arch_config}")

        # Initialize feature extractor (WITHOUT WEIGHTS at this step!)
        feature_extractor = ResNetFeatureExtractor()

        # Initialize Pose6DModel (self) from config (WITHOUT WEIGHTS at this step!)
        obj = cls(
            n_segments=arch_config.n_segments,
            feature_extractor=feature_extractor,
            n_attention_gated_feature_channels=arch_config.n_attention_gated_feature_channels,
            n_global_feature_channels=arch_config.n_global_feature_channels,
            camera_distance=arch_config.camera_distance,
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

    def forward(self, image: torch.Tensor, bodyseg_prob: torch.Tensor) -> dict:
        bat_size = image.shape[0]
        assert image.shape == (bat_size, 3, 256, 256)

        # Run feature extractor and upsample to 64x64
        e0, e1, e2, e3, e4 = self.feature_extractor.forward_with_intermediates(image)
        d4 = e4  # this is just the bottleneck
        d3 = self.dec_layer4(d4, e3)
        d2 = self.dec_layer3(d3, e2)
        features = self.dec_layer2(d2, e1)  # (B, n_feature_channels_total, 64, 64)

        # Attention using bodyseg prediction happens at 64x64 resolution (stride 4)
        assert features.shape == (bat_size, self.n_feature_channels_total, 64, 64)
        assert bodyseg_prob.shape == (bat_size, self.n_segments, 64, 64)
        # (B, n_channels_feature_map + n_segments, H, W)
        feature_map_with_seg_prob = torch.cat([features, bodyseg_prob], dim=1)

        # Process each segment separately
        translation_pred_list = []
        quaternion_pred_list = []
        for seg_idx in range(self.n_segments):
            # Global features
            global_features_pooled = torch.mean(
                features[:, : self.n_global_feature_channels, :, :], dim=[2, 3]
            )

            # Attention gated features
            if self.n_attention_gated_feature_channels > 0:
                attn_head = self.attention_heads[seg_idx]
                attn_map = attn_head(feature_map_with_seg_prob)  # (B, 1, H, W)
                gated_features = (
                    features[:, -self.n_attention_gated_feature_channels :, :, :]
                    * attn_map
                )
                attn_weight_total = torch.sum(attn_map, dim=[2, 3]) + 1e-6  # (B, 1)
                gated_features_pooled = (
                    gated_features.sum(dim=[2, 3]) / attn_weight_total
                )  # (B, C)
                all_features_pooled = torch.cat(
                    [global_features_pooled, gated_features_pooled], dim=1
                )
            else:
                all_features_pooled = global_features_pooled

            # FC layers for 6D pose prediction
            head = self.pose6d_heads[seg_idx]
            pose6d_pred = head(all_features_pooled)  # (B, 7)
            translation_pred = pose6d_pred[:, 0:3]  # (B, 3)
            quaternion_pred = pose6d_pred[:, 3:7]  # (B, 4)
            quaternion_pred = F.normalize(quaternion_pred, p=2, dim=1)  # normalize quat
            translation_pred_list.append(translation_pred)
            quaternion_pred_list.append(quaternion_pred)

        # Stack predictions into single tensors of shape (B, n_segments, 3 or 4)
        translation_pred_all = torch.stack(translation_pred_list, dim=1)
        translation_pred_all[:, :, 2] += self.z_center
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
    ) -> torch.Tensor:
        # Compute losses
        translation_loss = self.translation_loss(
            translation_pred.view(-1, 3), translation_label.view(-1, 3)
        )
        rotation_loss = self.rotation_loss(
            quaternion_pred.view(-1, 4), quaternion_label.view(-1, 4)
        )
        total_loss = (
            self.translation_weight * translation_loss
            + self.rotation_weight * rotation_loss
        )

        return {
            "translation_loss_unweighted": translation_loss,
            "rotation_loss_unweighted": rotation_loss,
            "total_loss": total_loss,
        }

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
