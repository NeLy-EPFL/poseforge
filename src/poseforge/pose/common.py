import math
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights
from torchvision.models._api import WeightsEnum


# Supported ResNet backbones, mapping the string used in configs/CLI to the
# torchvision constructor and its ImageNet weights enum. Only BasicBlock
# variants (resnet18, resnet34) are included: they share the exact same
# per-stage channel layout (64/64/128/256/512) and feature-map sizes, so a
# ResNetFeatureExtractor and every downstream head that consumes its
# intermediates is agnostic to which of the two is used. ResNet-34 is the
# deeper option and therefore has the larger receptive field.
_BACKBONES: dict[str, tuple] = {
    "resnet18": (models.resnet18, ResNet18_Weights),
    "resnet34": (models.resnet34, ResNet34_Weights),
}


class ResNetFeatureExtractor(nn.Module):
    """Feature extractor using a ResNet (18 or 34) backbone."""

    def __init__(
        self,
        weights: str | Path | WeightsEnum | None = "IMAGENET1K_V1",
        backbone: str = "resnet18",
    ):
        """
        Args:
            weights (str | Path | WeightsEnum | None): Weights to use
                for the backbone. In practice, use "IMAGENET1K_V1" for
                off-the-shelf ImageNet weights from torchvision, or a path
                to a .pth file with weights for this nn.Module (e.g. from
                pretraining). If None, start from scratch. When loading
                pretrained weights for this nn.Module from a path, ``backbone``
                must match the backbone those weights were trained with.
            backbone (str): Which ResNet variant to use as the backbone. One
                of "resnet18" (default) or "resnet34". ResNet-34 is deeper and
                has a larger receptive field, while keeping the same channel
                layout and feature-map sizes as ResNet-18, so it is a drop-in
                replacement everywhere the intermediates are consumed.
        """
        super(ResNetFeatureExtractor, self).__init__()

        if backbone not in _BACKBONES:
            raise ValueError(
                f"Invalid backbone {backbone!r}. Must be one of "
                f"{sorted(_BACKBONES)}."
            )
        self.backbone = backbone
        backbone_fn, weights_enum = _BACKBONES[backbone]

        # Figure out which weights to use
        if weights == "IMAGENET1K_V1":  # only option as of 2025-09
            weights = weights_enum.IMAGENET1K_V1

        if isinstance(weights, WeightsEnum):
            # Use an off-the-shelf pretrained ResNet backbone from torchvision.models
            backbone_weights = weights  # used to initialize the torchvision backbone
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

        # Initialize the ResNet backbone
        self.resnet = backbone_fn(weights=backbone_weights)

        # Find out the output size of the ResNet feature extractor
        # input_size will be detected dynamically on first forward pass
        self.input_size = None  # will be set during first forward pass
        # layer4 output channels == fc.in_features (512 for BasicBlock resnets)
        self.output_channels = self.resnet.fc.in_features

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
        # Detect and store actual input size on first forward pass
        if self.input_size is None:
            self.input_size = (x.shape[2], x.shape[3])  # (height, width)
        
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
        # if self._first_time_forward:
        #     batch_size = x.shape[0]
        #     assert x.shape == (batch_size, 3, *self.input_size)
        #     assert x_norm.shape == (batch_size, 3, *self.input_size)
        #     assert conv1_out.shape == (batch_size, 64, 128, 128)
        #     assert bn1_out.shape == (batch_size, 64, 128, 128)
        #     assert x0.shape == (batch_size, 64, 128, 128)
        #     assert x0_maxpool_out.shape == (batch_size, 64, 64, 64)
        #     assert x1.shape == (batch_size, 64, 64, 64)
        #     assert x2.shape == (batch_size, 128, 32, 32)
        #     assert x3.shape == (batch_size, 256, 16, 16)
        #     assert x4.shape == (batch_size, 512, 8, 8)
        #     self._first_time_forward = False

        if return_intermediates:
            return x0, x1, x2, x3, x4
        else:
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


class BottleneckSelfAttention(nn.Module):
    """Multi-head self-attention over the spatial tokens of a (N, C, H, W)
    feature map, with fixed 2D sinusoidal positional encodings and a residual
    connection.

    Intended for the low-resolution ResNet bottleneck (``e4``, 512ch and
    ~8x8 at 256px input), where the token count is tiny so full all-pairs
    attention is cheap. It lets every spatial location aggregate
    content-selected global context in a single hop -- something stacked
    convolutions can only do weakly and indirectly (their effective receptive
    field is a small Gaussian, and at large inputs even the theoretical field
    is smaller than the animal). This directly targets keypoint
    identity/plausibility ("is this a real claw, or clutter?", "which leg am
    I?").

    Applied as a pre-norm residual sub-layer:
        ``x <- x + Proj(SelfAttention(LayerNorm(tokens) + pos_enc))``
    The output projection is zero-initialized so the block starts as an exact
    identity (residual only) and attention is learned gradually, which avoids
    disrupting the pretrained backbone at the start of training.
    """

    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        if channels % n_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by n_heads ({n_heads})."
            )
        if channels % 4 != 0:
            # 2D sinusoidal encoding splits channels into row/col halves, each
            # of which needs an even size for the sin/cos interleave.
            raise ValueError(
                f"channels ({channels}) must be divisible by 4 for the 2D "
                "sinusoidal positional encoding."
            )
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        # Identity at init: zero the output projection so x + proj(...) == x.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    @staticmethod
    def _sinusoidal_pos_encoding_2d(
        h: int, w: int, dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return a (h*w, dim) 2D sinusoidal positional encoding: the first
        half of the channels encode the row index, the second half the column
        index."""
        d = dim // 2  # channels for row, and for col
        div = torch.exp(
            torch.arange(0, d, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / d)
        )  # (d/2,)
        y = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1)  # (h,1)
        x = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(1)  # (w,1)
        pe_y = torch.zeros(h, d, device=device, dtype=torch.float32)
        pe_y[:, 0::2] = torch.sin(y * div)
        pe_y[:, 1::2] = torch.cos(y * div)
        pe_x = torch.zeros(w, d, device=device, dtype=torch.float32)
        pe_x[:, 0::2] = torch.sin(x * div)
        pe_x[:, 1::2] = torch.cos(x * div)
        pe_y = pe_y.unsqueeze(1).expand(h, w, d)  # (h,w,d)
        pe_x = pe_x.unsqueeze(0).expand(h, w, d)  # (h,w,d)
        pe = torch.cat([pe_y, pe_x], dim=-1)  # (h,w,dim)
        return pe.reshape(h * w, dim).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        # (N, C, H, W) -> (N, H*W, C) sequence of tokens
        tokens = x.flatten(2).transpose(1, 2)
        residual = tokens
        t = self.norm(tokens)
        # Inject absolute position so attention is not permutation-blind.
        t = t + self._sinusoidal_pos_encoding_2d(h, w, c, x.device, t.dtype).unsqueeze(0)

        qkv = self.qkv(t)  # (N, H*W, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(z: torch.Tensor) -> torch.Tensor:
            # (N, H*W, C) -> (N, n_heads, H*W, head_dim)
            return z.view(n, h * w, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # (N, n_heads, H*W, head_dim)
        out = out.transpose(1, 2).reshape(n, h * w, c)  # (N, H*W, C)
        out = self.proj(out)

        tokens = residual + out
        # (N, H*W, C) -> (N, C, H, W)
        return tokens.transpose(1, 2).view(n, c, h, w)
