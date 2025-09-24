import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetFeatureExtractor, self).__init__()

        # Load pretrained ResNet-18
        if pretrained:
            self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet18(weights=None)

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

    def forward(self, x):
        features = self.resnet_feature_extractor(x)
        return features


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ContrastiveProjectionHead, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Flatten the features
        projected = self.projection_head(x)
        return projected
