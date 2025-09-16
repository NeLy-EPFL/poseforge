import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import RegNet_Y_800MF_Weights


class RegNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(RegNetFeatureExtractor, self).__init__()

        # Load pretrained RegNet_Y_800MF
        if pretrained:
            weights = RegNet_Y_800MF_Weights.IMAGENET1K_V2
            self.regnet = models.regnet_y_800mf(weights=weights)
        else:
            self.regnet = models.regnet_y_800mf(weights=None)

        # Remove the final classification head (avgpool + fc)
        # RegNet architecture: stem -> trunk -> head. Keep stem & trunk, remove head
        self.regnet_feature_extractor = nn.Sequential(
            self.regnet.stem, self.regnet.trunk
        )

    def forward(self, x):
        features = self.regnet_feature_extractor(x)
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
