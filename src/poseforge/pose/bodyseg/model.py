import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

import poseforge.pose.bodyseg.config as config
from poseforge.pose.common import ResNetFeatureExtractor, DecoderBlock


class BodySegmentationModel(nn.Module):
    def __init__(
        self,
        n_classes: int,
        feature_extractor: ResNetFeatureExtractor,
        final_upsampler_n_hidden_channels: int,
        confidence_method: str = "entropy",
    ):
        """
        Args:
            n_classes (int): Number of classes to predict.
            feature_extractor (ResNetFeatureExtractor): (Pretrained)
                feature extractor.
            final_upsampler_n_hidden_channels (int): Number of hidden
                channels in the final upsampling layer before
                classification.
            confidence_method (str): Method to compute confidence scores
                from the output logits. Options: "entropy"
                (1 - normalized entropy in predicted distribution), or
                "peak" (max probability among classes).
        """
        super(BodySegmentationModel, self).__init__()
        self.n_classes = n_classes
        self.feature_extractor = feature_extractor
        self.final_upsampler_n_hidden_channels = final_upsampler_n_hidden_channels
        self.confidence_method = confidence_method

        if confidence_method not in ["entropy", "peak"]:
            raise ValueError(
                f"Invalid confidence_method {confidence_method}. "
                'Must be "entropy" or "peak".'
            )

        # Create decoder (decoder1/2/3/4 mirror encoder layers 1/2/3/4)
        # Note that when upsampling, we actually run decoder4 first, decoder1 last
        self.dec_layer4 = DecoderBlock(512, 256, 256)  # 512ch 8x8 -> 256ch 16x16
        self.dec_layer3 = DecoderBlock(256, 128, 128)  # 256ch 16x16 -> 128ch 32x32
        self.dec_layer2 = DecoderBlock(128, 64, 64)  # 128ch 32x32 -> 64ch 64x64
        self.dec_layer1 = DecoderBlock(64, 64, 64)  # 64ch 64x64 -> 64ch 128x128

        # Final upsampling layer to reach input size
        self.final_upsampler = nn.Sequential(
            nn.ConvTranspose2d(
                64, self.final_upsampler_n_hidden_channels, kernel_size=2, stride=2
            ),  # c=64, 128x128 -> c=final_upsampler_n_hidden_channels, 256x256
            nn.BatchNorm2d(self.final_upsampler_n_hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Final classification layer
        self.classifier = nn.Conv2d(
            self.final_upsampler_n_hidden_channels, self.n_classes, kernel_size=1
        )

        self._first_time_forward = True

    @classmethod
    def create_architecture_from_config(
        cls, architecture_config: config.ModelArchitectureConfig | Path | str
    ) -> "BodySegmentationModel":
        # Load from file if config is given as a path
        if isinstance(architecture_config, (Path, str)):
            architecture_config = config.ModelArchitectureConfig.load(
                architecture_config
            )
            logging.info(f"Loaded model architecture config from {architecture_config}")

        # Initialize feature extractor (WITHOUT WEIGHTS at this step!)
        feature_extractor = ResNetFeatureExtractor()

        # Initialize model from config (WITHOUT WEIGHTS at this step!)
        obj = cls(
            n_classes=architecture_config.n_classes,
            feature_extractor=feature_extractor,
            final_upsampler_n_hidden_channels=architecture_config.final_upsampler_n_hidden_channels,
            confidence_method=architecture_config.confidence_method,
        )

        logging.info("Created BodySegmentationModel from architecture config")
        return obj

    def load_weights_from_config(
        self, weights_config: config.ModelWeightsConfig | Path | str
    ):
        # Load from file if config is given as a path
        if isinstance(weights_config, (Path, str)):
            weights_config = config.ModelWeightsConfig.load(weights_config)
            logging.info(f"Loaded model weights config from {weights_config}")

        # Check if config has either feature extractor weights or full model weights
        if (
            weights_config.feature_extractor_weights is None
            and weights_config.model_weights is None
        ):
            logging.warning("weights_config contains nothing useful. No action taken.")

        # If full model weights are provided, load them directly
        if weights_config.model_weights is not None:
            checkpoint_path = Path(weights_config.model_weights)
            if not checkpoint_path.is_file():
                raise ValueError(f"Model weights path {checkpoint_path} is not a file")
            weights = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(weights)
            logging.info(
                f"Loaded BodySegmentationModel weights (inc. feature extractor) from config"
            )
            return

        # Otherwise, init feature extractor first
        self.feature_extractor = ResNetFeatureExtractor(
            # Path, str, or "IMAGENET1K_V1"
            weights=weights_config.feature_extractor_weights
        )
        logging.info("Set up feature extractor from config")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor (batch_size, 3, 256, 256)

        Returns:
            segmentation_logits (torch.Tensor): Output logits, tensor
                (batch_size, n_classes, 256, 256)

        We use a UNet-like architecture with skip connections:

        Encoder (downsampling)     Decoder (upsampling)
        ──────────────────────     ────────────────────
          e0: 64ch 128x128        **d0: 64ch 128x128**    (end of upsampling core)
                │ │                       ↑
                │ └────────(skip)───────→(+)  dec_layer1: 64(up)+64(skip)->64 ch
                ↓                         ↑(up)           64x64(up)+128x128(skip)->128x128
          e1: 64ch 64x64            d1: 64ch 64x64
                │ │                       ↑
                │ └────────(skip)───────→(+)  dec_layer2: 128(up)+64(skip)->64 ch
                ↓                         ↑(up)           32x32(up)+64x64(skip)->64x64
          e2: 128ch 32x32           d2: 128ch 32x32
                │ │                       ↑
                │ └────────(skip)───────→(+)  dec_layer3: 256(up)+128(skip)->128 ch
                ↓                         ↑(up)           16x16(up)+32x32(skip)->32x32
          e3: 256ch 16x16           d3: 256ch 16x16
                │ │                       ↑
                │ └────────(skip)───────→(+)  dec_layer4: 512(up)+256(skip)->256 ch
                ↓                         ↑(up)           8x8(up)+16x16(skip)->16x16
          e4: 512ch 8x8             d4: 512ch 8x8
                |                         ↑
                └──(bottleneck/identity)──┘
        """
        # Run feature extractor
        e0, e1, e2, e3, e4 = self.feature_extractor.forward(
            x, return_intermediates=True
        )

        d4 = e4  # this is just the bottleneck

        # Upsample with skip connections
        d3 = self.dec_layer4(d4, e3)
        d2 = self.dec_layer3(d3, e2)
        d1 = self.dec_layer2(d2, e1)
        d0 = self.dec_layer1(d1, e0)

        # Final upsampling and classification
        upsampled = self.final_upsampler(d0)
        segmentation_logits = self.classifier(upsampled)

        # Compute confidence scores
        if self.confidence_method == "entropy":
            # Softmax to get class probabilities
            probs = F.softmax(segmentation_logits, dim=1)  # (B, n_classes, H, W)
            # Compute normalized entropy --- everything below has shape (B, H, W)
            entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=1)
            # Normalize by the maximum possible entropy (uniform distribution)
            n_classes = torch.tensor(self.n_classes, dtype=entropy.dtype)  # scalar
            normalized_entropy = entropy / torch.log(n_classes)
            confidence = 1.0 - normalized_entropy  # Higher confidence for lower entropy
        elif self.confidence_method == "peak":
            probs = F.softmax(segmentation_logits, dim=1)  # (B, n_classes, H, W)
            confidence, dim = torch.max(probs, dim=1)  # (B, H, W)

        # If this is the first forward pass, check if the shapes are as expected
        if self._first_time_forward:
            batch_size = x.shape[0]
            assert self.feature_extractor.input_size == (256, 256)
            assert x.shape == (batch_size, 3, 256, 256)
            assert e0.shape == (batch_size, 64, 128, 128)
            assert e1.shape == (batch_size, 64, 64, 64)
            assert e2.shape == (batch_size, 128, 32, 32)
            assert e3.shape == (batch_size, 256, 16, 16)
            assert e4.shape == (batch_size, 512, 8, 8)
            assert d4.shape == (batch_size, 512, 8, 8)
            assert d3.shape == (batch_size, 256, 16, 16)
            assert d2.shape == (batch_size, 128, 32, 32)
            assert d1.shape == (batch_size, 64, 64, 64)
            assert d0.shape == (batch_size, 64, 128, 128)
            upsample_hidden_channels = self.final_upsampler_n_hidden_channels
            assert upsampled.shape == (batch_size, upsample_hidden_channels, 256, 256)
            assert segmentation_logits.shape == (batch_size, self.n_classes, 256, 256)
            assert confidence.shape == (batch_size, 256, 256)
            self._first_time_forward = False

        return {"logits": segmentation_logits, "confidence": confidence}


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target_indices):
        """
        Args:
            pred_logits (torch.Tensor):
                Predicted logits (batch_size, n_classes, height, width).
            target_indices (torch.Tensor):
                Ground truth class indices (batch_size, height, width).

        Returns:
            loss (torch.Tensor): Dice loss (scalar).
        """
        n_classes = pred_logits.shape[1]

        # Get class probabilities
        probs = F.softmax(pred_logits, dim=1)  # (batch_size, n_classes, H, W)

        # Get ground truth in one-hot format
        # F.one_hot gives n_classes at the end (batch_size, H, W, n_classes)
        # We need to permute it to (batch_size, n_classes, H, W)
        targets_1hot = F.one_hot(target_indices, num_classes=n_classes)
        targets_1hot = targets_1hot.permute(0, 3, 1, 2).float()

        # Compute Dice loss
        spatial_dims = (2, 3)  # height and width
        intersection = (probs * targets_1hot).sum(dim=spatial_dims)
        union = probs.sum(dim=spatial_dims) + targets_1hot.sum(dim=spatial_dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice.mean()  # average over classes and batch
        return loss


class CombinedDiceCELoss(nn.Module):
    def __init__(
        self,
        weight_dice: float = 0.5,
        weight_ce: float = 0.5,
        ce_class_weights: torch.Tensor = None,
    ):
        """
        Args:
            weight_dice (float): Weight for Dice loss component.
            weight_ce (float): Weight for Cross-Entropy loss component.
        """
        super(CombinedDiceCELoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce_class_weights = ce_class_weights

        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_class_weights)

    @classmethod
    def create_from_config(
        cls, loss_config: config.LossConfig | Path | str
    ) -> "CombinedDiceCELoss":
        # Load from file if config is given as a path
        if isinstance(loss_config, (Path, str)):
            loss_config = config.LossConfig.load(loss_config)
            logging.info(f"Loaded model loss config from {loss_config}")

        # Initialize loss from config
        obj = cls(
            weight_dice=loss_config.weight_dice,
            weight_ce=loss_config.weight_ce,
            ce_class_weights=loss_config.ce_class_weights,
        )

        logging.info("Created CombinedDiceCELoss from loss config")
        return obj

    def forward(self, pred_logits, target_indices):
        """
        Args:
            pred_logits (torch.Tensor):
                Predicted logits (batch_size, n_classes, height, width).
            target_indices (torch.Tensor):
                Ground truth class indices (batch_size, height, width).

        Returns:
            loss (torch.Tensor): Combined loss (scalar).
        """
        ce = self.ce_loss(pred_logits, target_indices)
        dice = self.dice_loss(pred_logits, target_indices)
        total_loss = self.weight_ce * ce + self.weight_dice * dice
        return {"total_loss": total_loss, "ce_loss": ce, "dice_loss": dice}
