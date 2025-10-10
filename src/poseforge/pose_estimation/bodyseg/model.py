import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

import poseforge.pose_estimation.bodyseg.config as config
from poseforge.pose_estimation.feature_extractor import ResNetFeatureExtractor


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class BodySegModel(nn.Module):
    def __init__(
        self,
        n_classes: int,
        feature_extractor: ResNetFeatureExtractor,
        temperature: float,
        upsample_n_layers: int = 3,
        final_upsample_n_hidden_channels: int = 32,
        confidence_method: str = "entropy",
    ):
        """
        Args:
            n_classes (int): Number of classes to predict.
            feature_extractor (ResNetFeatureExtractor): (Pretrained)
                feature extractor.
            temperature (float): Temperature for soft-argmax in heatmaps.
            upsample_n_layers (int): Number of upsampling layers (deconvs).
            upsample_n_hidden_channels (int): Number of hidden channels in
                upsampling layers.
            confidence_method (str): Method to compute confidence scores in
                soft argmax of x-y heatmaps and depth logits. Options:
                "entropy" (1 - normalized entropy in predicted
                distribution, default) or "peak" (max probability).
        """
        super(BodySegModel, self).__init__()
        self.n_classes = n_classes
        self.feature_extractor = feature_extractor
        self.temperature = temperature
        self.upsample_n_layers = upsample_n_layers
        self.final_upsample_n_hidden_channels = final_upsample_n_hidden_channels
        self.confidence_method = confidence_method

        if confidence_method not in ["entropy", "peak"]:
            raise ValueError(
                f"Invalid confidence_method {confidence_method}. "
                'Must be "entropy" or "peak".'
            )

        # Create decoder (decoder1/2/3/4 mirror encoder layers 1/2/3/4)
        # Note that when upsampling, we actually run decoder4 first, decoder1 last
        self.decoder4 = DecoderBlock(512, 256, 256)  # 512ch   8x8 -> 256ch   16x16
        self.decoder3 = DecoderBlock(256, 128, 128)  # 256ch 16x16 -> 128ch   32x32
        self.decoder2 = DecoderBlock(128, 64, 64)  #   128ch 32x32 ->  64ch   64x64
        self.decoder1 = DecoderBlock(64, 64, 64)  #     64ch 64x64 ->  64ch 128x128

        # Final upsampling layer to reach input size
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(
                64, self.final_upsample_n_hidden_channels, kernel_size=2, stride=2
            ),  # c=64, 128x128 -> c=final_upsample_n_hidden_channels, 256x256
            nn.BatchNorm2d(self.final_upsample_n_hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Final classification layer
        self.classifier = nn.Conv2d(
            self.final_upsample_n_hidden_channels, self.n_classes, kernel_size=1
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (batch_size, 3, 256, 256)

        Returns:
            segmentation_logits (torch.Tensor): Output logits, tensor
                (batch_size, n_classes, 256, 256)
        """
        # Run feature extractor
        e0, e1, e2, e3, e4 = self.feature_extractor.forward(
            x, return_intermediates=True
        )

        # Decoder with skip connections
        #  Encoder (downsampling)             Decoder (upsampling)
        #  ──────────────────────             ────────────────────
        #    e0:  64ch 128x128                  d0:  64ch 128x128
        #              │ │                       ↑
        #              │ └────────(skip)───────→(+)
        #              ↓                         ↑(up)
        #    e1:  64ch   64x64                  d1:  64ch   64x64
        #              │ │                       ↑
        #              │ └────────(skip)───────→(+)
        #              ↓                         ↑(up)
        #    e2: 128ch   32x32                  d2: 128ch   32x32
        #              │ │                       ↑
        #              │ └────────(skip)───────→(+)
        #              ↓                         ↑(up)
        #    e3: 256ch   16x16                  d3: 256ch   16x16
        #              │ │                       ↑
        #              │ └────────(skip)───────→(+)
        #              ↓                         ↑(up)
        #    e4: 512ch     8x8                  d4: 512ch     8x8
        #              |                         ↑
        #              └──(bottleneck/identity)──┘

        d4 = e4  # this is just the bottleneck
        d3 = self.decoder4(d4, e3)
        d2 = self.decoder3(d3, e2)
        d1 = self.decoder2(d2, e1)
        d0 = self.decoder1(d1, e0)

        # Final upsampling and classification
        upsampled = self.final_upsample(d0)
        segmentation_logits = self.classifier(upsampled)

        return segmentation_logits


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
        intersection = probs * targets_1hot.sum(dim=spatial_dims)
        union = probs.sum(dim=spatial_dims) + targets_1hot.sum(dim=spatial_dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice.mean()  # average over classes and batch
        return loss


class CombinedLoss(nn.Module):
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
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce_class_weights = ce_class_weights

        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_class_weights)

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
        return self.weight_ce * ce + self.weight_dice * dice
