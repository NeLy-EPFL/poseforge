import torch
import torch.nn as nn
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
        self.input_size = (256, 256)  # input image size (height, width), fixed
        self.output_channels = 512  # ResNet-18 layer4 output channels

        # Load weights for this very nn.Module if provided
        if my_module_weights is not None:
            self.load_state_dict(my_module_weights)

        self._first_time_forward = True

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
    
    def forward_with_intermediates(self, x: torch.Tensor):
        """Run forward pass, returning intermediate feature maps as well as
        the final features.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3,
                height, width), with pixel values in [0, 1].

        Returns:
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
        """
        if x.shape[-2:] != self.input_size:
            raise NotImplementedError(
                f"Input image size {x.shape[-2:]} not supported; "
                f"expected {self.input_size}"
            )

        x_norm = self._apply_imagenet_normalization(x)

        # Remove the final classification head (avgpool + fc)
        # ResNet architecture:
        #     conv1 -> bn1 -> relu -> maxpool
        #           -> layer1 -> layer2 -> layer3 -> layer4
        #           -> avgpool -> fc
        # Discard the avgpool and fc; keep everything up to the last conv layer
        conv1_out = self.resnet.conv1(x_norm)  # (batch_size, 64, 128, 128)
        bn1_out = self.resnet.bn1(conv1_out)  # same shape
        x0 = self.resnet.relu(bn1_out)  # same shape
        x0_maxpool_out = self.resnet.maxpool(x0)  # (batch_size, 64, 64, 64)
        x1 = self.resnet.layer1(x0_maxpool_out)  # (batch_size, 64, 64, 64)
        x2 = self.resnet.layer2(x1)  # (batch_size, 128, 32, 32)
        x3 = self.resnet.layer3(x2)  # (batch_size, 256, 16, 16)
        x4 = self.resnet.layer4(x3)  # (batch_size, 512, 8, 8)

        # If this is the first forward pass, check if the shapes are as expected
        if self._first_time_forward:
            batch_size = x.shape[0]
            assert x.shape == (batch_size, 3, *self.input_size)
            assert x_norm.shape == (batch_size, 3, *self.input_size)
            assert conv1_out.shape == (batch_size, 64, 128, 128)
            assert bn1_out.shape == (batch_size, 64, 128, 128)
            assert x0.shape == (batch_size, 64, 128, 128)
            assert x0_maxpool_out.shape == (batch_size, 64, 64, 64)
            assert x1.shape == (batch_size, 64, 64, 64)
            assert x2.shape == (batch_size, 128, 32, 32)
            assert x3.shape == (batch_size, 256, 16, 16)
            assert x4.shape == (batch_size, 512, 8, 8)
            self._first_time_forward = False

        return x0, x1, x2, x3, x4
        
    def forward(self, x: torch.Tensor):
        """Run forward pass through the ResNet-18 backbone and return the
        final extracted features.

        See also `.forward_with_intermediates`, which returns intermediate
        feature maps as well (useful for identity connections in U-Net-like
        architectures).

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3,
                height, width), with pixel values in [0, 1].

        Returns:
            features (torch.Tensor): Extracted features. The shape is
                (batch_size, out_channels, *output_feature_map_size)
                where output_feature_map_size depends on the input
                image size.
        """
        x0, x1, x2, x3, x4 = self.forward_with_intermediates(x)
        return x4


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
