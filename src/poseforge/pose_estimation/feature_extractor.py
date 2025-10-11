import torch
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNetFeatureExtractor(nn.Module):
    """Feature extractor using a ResNet-18 backbone."""

    def __init__(self, weights: str | Path | ResNet18_Weights | None = "IMAGENET1K_V1"):
        """
        Args:
            weights (str | Path | ResNet18_Weights | None): Weights to use
                for the backbone. In practice, use "IMAGENET1K_V1" for
                off-the-shelf ImageNet weights from torchvision, or a path
                to a .pth file with weights for this nn.Module (e.g. from
                pretraining). If None, start from scratch.
        """
        super(ResNetFeatureExtractor, self).__init__()

        # Figure out which weights to use
        if weights == "IMAGENET1K_V1":  # only option as of 2025-09
            weights = ResNet18_Weights.IMAGENET1K_V1

        if isinstance(weights, ResNet18_Weights):
            # Use an off-the-shelf pretrained ResNet backbone from torchvision.models
            backbone_weights = weights  # used to initialize models.resnet18
            my_module_weights = None  # load weights for this very nn.Module
        elif isinstance(weights, (str, Path)):
            # Instead of using off-the-shelf weights for the ResNet backbone, this very
            # nn.Module has been pretrained or partially trained (though possibly from
            # off-the-shelf ResNet weights as a starting point). Load those weights once
            # the architecture of this nn.Module is defined.
            backbone_weights = None
            if not Path(weights).is_file():
                raise ValueError(f"Provided weights path {weights} is not a file")
            my_module_weights = torch.load(weights, map_location="cpu")
        elif weights is None:
            # Start from scratch
            backbone_weights = None
            my_module_weights = None
        else:
            raise ValueError(f"Invalid weights argument: {weights}")

        # Initialize ResNet-18 backbone
        self.resnet = models.resnet18(weights=backbone_weights)

        # Find out the output size of the ResNet feature extractor
        self.output_channels = 512  # ResNet-18 layer4 output channels

        # Load weights for this very nn.Module if provided
        if my_module_weights is not None:
            self.load_state_dict(my_module_weights)

    @staticmethod
    def _apply_imagenet_normalization(
        x: torch.Tensor,
        mean: list = [0.485, 0.456, 0.406],
        std: list = [0.229, 0.224, 0.225],
    ) -> torch.Tensor:
        """Normalize input image tensor to the format expected by
        ImageNet-pretrained models from torchvision.

        See https://docs.pytorch.org/vision/0.8/models.html

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3,
                height, width), with pixel values in [0, 1].
                `SimulatedDataLoader` and `SyntheticFramesSampler`, and
                `AtomDataset` already do this normalization. Note that
                even after the data is converted to the range [0, 1], they
                still need to be normalized using the ImageNet mean and
                std. This is handled by this method.
            mean (list): Per-channel mean for normalization.
            std (list): Per-channel standard deviation for normalization.

        Returns:
            x_normalized (torch.Tensor): Image tensor further normalized
                to the format expected by ImageNet-pretrained models.
        """
        # ImageNet mean and std
        mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
        x_normalized = (x - mean) / std
        return x_normalized

    def forward(self, x, return_intermediates: bool = False):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3,
                height, width), with pixel values in [0, 1].
            return_intermediates (bool): Whether to return intermediate
                feature maps from various layers. Default False.

        Returns:
            If return_intermediates is False:
                features (torch.Tensor): Extracted features. The shape is
                    (batch_size, out_channels, *output_feature_map_size)
                    where output_feature_map_size depends on the input
                    image size.
            If return_intermediates is True:
                A tuple of 5 torch.Tensors:
                - Features after initial Conv-BN-ReLU but before maxpool:
                      tensor of shape (batch_size, 64, 128, 128)
                - Features after layer1:
                      tensor of shape (batch_size, 64, 64, 64)
                - Features after layer2:
                      tensor of shape (batch_size, 128, 32, 32)
                - Features after layer3:
                      tensor of shape (batch_size, 256, 16, 16)
                - Features after layer4:
                      tensor of shape (batch_size, 512, 8, 8)
                      This is the same as the single output returned if
                      return_intermediates is False.
        """
        x_norm = self._apply_imagenet_normalization(x)

        # Remove the final classification head (avgpool + fc)
        # ResNet architecture:
        #     conv1 -> bn1 -> relu -> maxpool
        #           -> layer1 -> layer2 -> layer3 -> layer4
        #           -> avgpool -> fc
        # Discard the avgpool and fc; keep everything up to the last conv layer
        conv1_out = self.resnet.conv1(x_norm)
        bn1_out = self.resnet.bn1(conv1_out)
        x0 = self.resnet.relu(bn1_out)
        x0_maxpool_out = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(x0_maxpool_out)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        if return_intermediates:
            return x0, x1, x2, x3, x4
        else:
            return x4
