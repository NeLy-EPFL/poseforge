import torch
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNetFeatureExtractor(nn.Module):
    """Feature extractor using a ResNet-18 backbone."""

    def __init__(
        self,
        weights: str | Path | ResNet18_Weights = "IMAGENET1K_V1",
        global_pool: bool = True,
    ):
        """
        Args:
            weights (str | Path | ResNet18_Weights): Weights to use for the
                backbone. In practice, use "IMAGENET1K_V1" for
                off-the-shelf ImageNet weights from torchvision, or a path
                to a .pth file with weights for this nn.Module (e.g. from
                pretraining). If None, start from scratch.
            global_pool (bool): Whether to use the last global adaptive
                average pooling layer. Keeping this layer is handy for
                obtaining a fixed-size output regardless of input image
                size. However, it gets rid of spatial information needed
                for things like pose estimation via heatmaps.
        """
        super(ResNetFeatureExtractor, self).__init__()

        self.global_pool = global_pool
        self._out_feature_map_size_cache = None  # to be determined based on input data

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

        # Remove the final classification head (avgpool + fc)
        # ResNet architecture:
        #     conv1 -> bn1 -> relu -> maxpool
        #           -> layer1 -> layer2 -> layer3 -> layer4
        #           -> avgpool -> fc
        # Remove the avgpool and fc; keep everything up to the last conv layer
        model_elements = OrderedDict(
            [
                ("conv1", self.resnet.conv1),
                ("bn1", self.resnet.bn1),
                ("relu", self.resnet.relu),
                ("maxpool", self.resnet.maxpool),
                ("layer1", self.resnet.layer1),
                ("layer2", self.resnet.layer2),
                ("layer3", self.resnet.layer3),
                ("layer4", self.resnet.layer4),
            ]
        )
        self.resnet_feature_extractor = nn.Sequential(model_elements)

        # Find out the output size of the ResNet feature extractor
        self.out_channels = 512  # ResNet-18 layer4 output channels
        # These need to be determined based on input data size. Call
        # `data_dependent_init(x)` with a sample input to set them.
        self.out_feature_map_size = (None, None)
        self.output_dim = None

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
                `AtomDataset` already do this normalization.
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

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3,
                height, width), with pixel values in [0, 1].

        Returns:
            features (torch.Tensor): Extracted features. The shape is
                (batch_size, out_channels, *self.out_feature_map_size)
                where self._out_feature_map_size depends on the input size.
        """
        x_norm = self._apply_imagenet_normalization(x)
        return self.resnet_feature_extractor(x_norm)

    def data_dependent_init(self, x: torch.Tensor):
        """Initialize data-dependent parameters based on a sample input.
        Specifically, determine the feature map size in the last layer
        right before global adaptive pooling. When `global_pool` is False,
        this is useful to determine the spatial dimensions of the output.

        This method only needs to be called once and only if `global_pool`
        is False.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3,
                height, width), with pixel values in [0, 1].
        """
        if self.global_pool:
            pass  # no need for data-dependent init
        else:
            with torch.no_grad():
                _, _, n_rows_in, n_cols_in = x.shape
                dummy_batch_size = 1
                dummy_input = torch.zeros((dummy_batch_size, 3, n_rows_in, n_cols_in))
                feature_map = self.resnet_feature_extractor(dummy_input)
                _, _, n_rows_out, n_cols_out = feature_map.shape
            self.out_feature_map_size = (n_rows_out, n_cols_out)
            self.output_dim = self.out_channels * n_rows_out * n_cols_out
