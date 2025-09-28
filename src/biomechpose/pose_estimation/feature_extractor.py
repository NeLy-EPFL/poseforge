import torch
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, weights: str | Path | ResNet18_Weights = "IMAGENET1K_V1"):
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

        # Remove the final classification head (avgpool + fc)
        # ResNet architecture:
        #     conv1 -> bn1 -> relu -> maxpool
        #           -> layer1 -> layer2 -> layer3 -> layer4
        #           -> avgpool -> fc
        # Remove the head (fc); keep everything up to avgpool
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
                ("avgpool", self.resnet.avgpool),
            ]
        )
        self.resnet_feature_extractor = nn.Sequential(model_elements)

        # Find out the output size of the ResNet feature extractor
        # For ResNet-18, layer4 has 512 output channels
        # The avgpool layer reduces spatial dimensions to 1x1
        self.resnet_out_channels = 512  # ResNet-18 layer4 output channels
        self.global_pool_output_size = (1, 1)  # avgpool output size
        self.output_dim = (
            self.resnet_out_channels
            * self.global_pool_output_size[0]
            * self.global_pool_output_size[1]
        )

        # Load weights for this very nn.Module if provided
        if my_module_weights is not None:
            self.load_state_dict(my_module_weights)

    def forward(self, x):
        features = self.resnet_feature_extractor(x)
        return features
