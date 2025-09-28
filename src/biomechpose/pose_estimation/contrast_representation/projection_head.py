import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models
from torchvision.models import ResNet18_Weights


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
