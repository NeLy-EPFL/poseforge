import torch
import torch.nn as nn
from collections import OrderedDict
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
        # RegNet architecture:  stem -> trunk_output -> avgpool -> fc
        # Remove the head (fc); keep stem, trunk, and avgpool (adaptive global pooling)
        model_elements = OrderedDict(
            [
                ("stem", self.regnet.stem),
                ("trunk", self.regnet.trunk_output),
                ("avgpool", self.regnet.avgpool),
            ]
        )
        self.regnet_feature_extractor = nn.Sequential(model_elements)

        # Find out the output size of the RegNet feature extractor
        # For this, we need to know the number of channels of the last conv layer in the
        # trunk and the output size of the adaptive pooling layer (the final feature
        # dimension is just (last_n_channel, pooled_H, pooled_W))
        trunk = self.regnet_feature_extractor.trunk
        trunk_last_block = trunk[-1]
        trunk_last_subblock = trunk_last_block[-1]
        trunk_last_subblock_transform = trunk_last_subblock.f
        trunk_last_norm_conv_activation_block = trunk_last_subblock_transform.c
        trunk_last_batch_norm = trunk_last_norm_conv_activation_block[1]
        assert isinstance(
            trunk_last_batch_norm, nn.BatchNorm2d
        ), "Error identifying the batch norm layer of the last conv block in the trunk"
        # The number of features of the last batch norm layer is the number of output
        # channels of the last conv layer
        self.regnet_out_channels = trunk_last_batch_norm.num_features
        self.global_pool_output_size = self.regnet_feature_extractor.avgpool.output_size
        self.output_dim = (
            self.regnet_out_channels
            * self.global_pool_output_size[0]
            * self.global_pool_output_size[1]
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
