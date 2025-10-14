import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

import poseforge.pose.keypoints3d.config as config
from poseforge.pose.common import ResNetFeatureExtractor, DecoderBlock


class Pose2p5DModel(nn.Module):
    """A 3D keypoint detection model, but implemented in "2.5D", i.e:
        - A x-y pathway predicts heatmaps for each keypoint in the 2D image
          plane of the camera. The (x, y) coordinates of each keypoint are
          obtained by taking the soft-argmax (expectation) of the predicted
          heatmap.
        - A depth pathway predicts a probability distribution over
          quantized depth bins for each keypoint. The depth value of each
          keypoint is obtained by taking the expectation of the predicted
          distribution.

    The feature extractor is a ResNet18 model. The intended approach is
    that this model has been pretrained on ImageNet (published by
    torchvision), and pretrained again contrastively on synthetic data.
    When the same simulated frame is rendered by different style transfer
    models into the experimental domain, their feature representations
    should be similar.

    Following the feature extractor, there is an upsampling core consisting
    of several ConvTranspose2d (decov) layers. A specialized x-y heatmap
    head and a specialized depth head branch off from the upsampling core,
    producing x-y heatmaps and depth logits respectively.
    """

    def __init__(
        self,
        n_keypoints: int,
        feature_extractor: ResNetFeatureExtractor,
        depth_n_bins: int,
        depth_min: float,
        depth_max: float,
        xy_temperature: float,
        depth_temperature: float,
        upsample_core_out_channels: int = 64,
        depth_hidden_channels: int = 64,
        confidence_method: str = "entropy",
        groupnorm_n_groups: int = 32,
        pose_head_init_std: float = 1e-3,
    ):
        """
        Args:
            n_keypoints (int): Number of keypoints to predict.
            feature_extractor (ResNetFeatureExtractor): (Pretrained)
                feature extractor.
            depth_n_bins (int): Number of discrete bins for depth
                prediction.
            depth_min (float): Minimum depth value (closest to camera).
            depth_max (float): Maximum depth value (farthest from camera).
            xy_temperature (float): Temperature for soft-argmax in x-y
                heatmaps.
            depth_temperature (float): Temperature for soft-argmax in depth
                logits.
            upsample_core_out_channels (int): Number of hidden channels in
                upsampling layers.
            depth_hidden_channels (int): Number of hidden channels in
                depth head.
            confidence_method (str): Method to compute confidence scores in
                soft argmax of x-y heatmaps and depth logits. Options:
                "entropy" (1 - normalized entropy in predicted
                distribution, default) or "peak" (max probability).
            groupnorm_n_groups (int): Number of groups for GroupNorm layers
                (BatchNorm is not suitable if batch size is small, so we
                use GroupNorm instead). Must be a divisor of numbers of
                channels in various layers that precede GroupNorm.
            pose_head_init_std (float): Standard deviation for initializing
                heatmap/depth head layers that are not followed by ReLU.
        """
        super().__init__()
        self.n_keypoints = n_keypoints
        self.depth_n_bins = depth_n_bins
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.xy_temperature = xy_temperature
        self.depth_temperature = depth_temperature
        self.upsample_core_out_channels = upsample_core_out_channels
        self.depth_hidden_channels = depth_hidden_channels
        self.confidence_method = confidence_method.lower()
        self.groupnorm_n_groups = groupnorm_n_groups
        self.pose_head_init_std = pose_head_init_std

        # Check input validity
        if confidence_method not in ["entropy", "peak"]:
            raise ValueError(
                f"Invalid confidence_method: {confidence_method}. "
                'Must be "entropy" or "peak".'
            )
        if (
            (upsample_core_out_channels % groupnorm_n_groups) != 0
            or (depth_hidden_channels % groupnorm_n_groups) != 0
            or groupnorm_n_groups > upsample_core_out_channels
            or groupnorm_n_groups > depth_hidden_channels
        ):
            raise ValueError(
                "groupnorm_n_groups must be a divisor of "
                "upsample_n_hidden_channels and depth_n_hidden_channels, "
                "and it cannot be greater than either of them."
            )

        self.feature_extractor = feature_extractor

        # Create decoder core with skipped connection for upsampling
        # We use decoder4/3/2 to mirror layers2/3/4 in the ResNet encoder
        # Note that when upsampling, we actually go in the reverse order (4-3-2) from
        # the bottleneck. We skip "decoder1" because decoder2 already produces a 64x64
        # feature map (stride 4 compared to the input). This is already a good
        # resolution to predict keypoint heatmaps from.
        assert feature_extractor.output_channels == 512  # expected from ResNet18 layer4
        self.dec_layer4 = DecoderBlock(512, 256, 256)  # 512ch 8x8 -> 256ch 16x16
        self.dec_layer3 = DecoderBlock(256, 128, 128)  # 256ch 16x16 -> 128ch 32x32
        # 128ch 32x32 -> 64ch 64x64
        self.dec_layer2 = DecoderBlock(128, 64, upsample_core_out_channels)

        # Heatmap head for (x, y) keypoint locations
        self.heatmap_head = self._build_heatmap_head(
            in_channels=upsample_core_out_channels, out_channels=n_keypoints
        )

        # Depth head for distance from camera
        self.depth_head = self._build_depth_head(
            in_channels=upsample_core_out_channels,
            hidden_channels=depth_hidden_channels,
            n_keypoints=n_keypoints,
            depth_n_bins=depth_n_bins,
        )

        # Precompute depth bin centers
        depth_bin_centers = torch.linspace(
            depth_min, depth_max, depth_n_bins, dtype=torch.float32
        )
        self.register_buffer("depth_bin_centers", depth_bin_centers, persistent=False)
        self._first_time_forward = True

    @classmethod
    def create_architecture_from_config(
        cls, architecture_config: config.ModelArchitectureConfig | Path | str
    ) -> "Pose2p5DModel":
        # Load from file if config is given as a path
        if isinstance(architecture_config, (Path, str)):
            architecture_config = config.ModelArchitectureConfig.load(
                architecture_config
            )
            logging.info(f"Loaded model architecture config from {architecture_config}")
        # Initialize feature extractor (WITHOUT WEIGHTS at this step!)
        feature_extractor = ResNetFeatureExtractor()

        # Initialize model from config (WITHOUT WEIGHTS at this step!)
        obj = cls(
            n_keypoints=architecture_config.n_keypoints,
            feature_extractor=feature_extractor,
            depth_n_bins=architecture_config.depth_n_bins,
            depth_min=architecture_config.depth_min,
            depth_max=architecture_config.depth_max,
            xy_temperature=architecture_config.xy_temperature,
            depth_temperature=architecture_config.depth_temperature,
            upsample_core_out_channels=architecture_config.upsample_core_out_channels,
            depth_hidden_channels=architecture_config.depth_hidden_channels,
            confidence_method=architecture_config.confidence_method,
            groupnorm_n_groups=architecture_config.groupnorm_n_groups,
            pose_head_init_std=architecture_config.pose_head_init_std,
        )

        logging.info("Created Pose2p5DModel from architecture config")
        return obj

    def load_weights_from_config(
        self, weights_config: config.ModelWeightsConfig | Path | str
    ):
        # Load from file if config is given as a path
        if isinstance(weights_config, (Path, str)):
            weights_config = config.ModelWeightsConfig.load(weights_config)
            logging.info(f"Loaded model weights config from {weights_config}")

        # Check if config has either feature extractor weights or full model weights
        if (
            weights_config.feature_extractor_weights is None
            and weights_config.model_weights is None
        ):
            logging.warning("weights_config contains nothing useful. No action taken.")

        # If full model weights are provided, load them directly
        if weights_config.model_weights is not None:
            checkpoint_path = Path(weights_config.model_weights)
            if not checkpoint_path.is_file():
                raise ValueError(f"Model weights path {checkpoint_path} is not a file")
            weights = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(weights)
            logging.info(
                f"Loaded Pose2p5DModel weights (inc. feature extractor) from config"
            )
            return

        # Otherwise, init feature extractor first
        self.feature_extractor = ResNetFeatureExtractor(
            # Path, str, or "IMAGENET1K_V1"
            weights=weights_config.feature_extractor_weights
        )
        logging.info("Set up feature extractor from config")

    def _build_heatmap_head(self, in_channels: int, out_channels: int) -> nn.Sequential:
        # Use kernel_size=3 to collect some spatial information, but keep the same
        # feature map size: padding = (k-1)//2, stride=1
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        # Initialize conv layer weights using normal initialization with small std
        # (this is not followed by ReLU, so Kaiming init is not appropriate here)
        nn.init.normal_(conv.weight, std=self.pose_head_init_std)
        nn.init.zeros_(conv.bias)
        return conv

    def _build_depth_head(
        self,
        in_channels: int,
        hidden_channels: int,
        n_keypoints: int,
        depth_n_bins: int,
    ) -> nn.Sequential:
        adaptive_pool = nn.AdaptiveAvgPool2d(1)
        # Initialize conv layer weights using normal initialization with small std
        # (this is not followed by ReLU, so Kaiming init is not appropriate here)
        conv = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        groupnorm = nn.GroupNorm(self.groupnorm_n_groups, hidden_channels)
        relu = nn.ReLU(inplace=True)
        flatten = nn.Flatten()
        fc = nn.Linear(hidden_channels, n_keypoints * depth_n_bins)
        reshape = nn.Unflatten(1, (n_keypoints, depth_n_bins))

        # Initialize layers weights using Kaiming normal initialization
        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(groupnorm.weight, 1)
        nn.init.constant_(groupnorm.bias, 0)
        # For the fc layer, don't use Kaiming init because it's not followed by ReLU
        nn.init.normal_(fc.weight, std=self.pose_head_init_std)
        nn.init.zeros_(fc.bias)

        return nn.Sequential(adaptive_pool, conv, groupnorm, relu, flatten, fc, reshape)

    @staticmethod
    def _softmax_with_temp(
        logits: torch.Tensor, dim: int, temperature: float
    ) -> torch.Tensor:
        """Softmax with temperature scaling. Higher temperature makes the
        distribution more uniform (random); lower temperature makes the
        distribution peakier around the most likely value."""
        return F.softmax(logits / max(1e-6, temperature), dim=dim)

    def _get_heatmap_xy_grid(
        self, heatmap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, n_rows, n_cols = heatmap.shape

        if (
            not hasattr(self, "heatmap_grid_x")
            or not hasattr(self, "heatmap_grid_y")
            or self.heatmap_grid_x.shape[-1] != n_cols
            or self.heatmap_grid_y.shape[-2] != n_rows
        ):
            # Create grid of (x, y) coordinates corresponding to the heatmap
            x_grid = torch.linspace(0, n_cols - 1, n_cols, device=heatmap.device).view(
                1, 1, 1, n_cols
            )
            y_grid = torch.linspace(0, n_rows - 1, n_rows, device=heatmap.device).view(
                1, 1, n_rows, 1
            )
            self.register_buffer("heatmap_grid_x", x_grid, persistent=False)
            self.register_buffer("heatmap_grid_y", y_grid, persistent=False)

        return self.heatmap_grid_x, self.heatmap_grid_y

    def _soft_argmax_2d(
        self, heatmaps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            heatmaps (torch.Tensor): Logits of shape
                (batch_size, n_keypoints, n_rows, n_cols).

        Returns:
            xy (torch.Tensor): X-Y coordinates of peaks in probability heat
                map for each keypoint. Shape: (batch_size, n_keypoints, 2).
            conf (torch.Tensor): Confidence scores for each keypoint.
                Shape: (batch_size, n_keypoints).
        """
        batch_size, n_keypoints, n_rows, n_cols = heatmaps.shape
        # Flat logits, shape: (batch_size, n_keypoints, n_rows*n_cols), then apply
        # softmax along the flattened spatial dimension (n_rows*n_cols)
        probs_flat = Pose2p5DModel._softmax_with_temp(
            heatmaps.view(batch_size, n_keypoints, -1),
            dim=-1,
            temperature=self.xy_temperature,
        )
        probs = probs_flat.view(batch_size, n_keypoints, n_rows, n_cols)

        # Extract the X-Y coordinates from the heatmaps. We do this by computing the
        # expected values of the X and Y coordinates, i.e. summing over all possible
        # coordinates weighted by their probabilities.
        x_grid, y_grid = self._get_heatmap_xy_grid(heatmaps)
        # Dimensions 2 and 3 are rows and cols
        x_expected = (probs * x_grid).sum(dim=(2, 3))  # (batch_size, n_keypoints)
        y_expected = (probs * y_grid).sum(dim=(2, 3))  # (batch_size, n_keypoints)
        xy_expected = torch.stack([x_expected, y_expected], dim=-1)

        # Compute the confidence of the prediction (one scalar per keypoint per image)
        if self.confidence_method == "peak":
            # Option 1: use the max probability as a confidence score
            confidence = probs_flat.max(dim=-1).values  # (batch_size, n_keypoints)
        elif self.confidence_method == "entropy":
            # Option 2: use the 1 - normalized entropy as a confidence score
            # entropy = $- \sum_i p_i \log p_i$, shape (batch_size, n_keypoints)
            entropy = -(probs_flat * torch.log(probs_flat.clamp_min(1e-6))).sum(dim=-1)
            # Normalize by the maximum possible entropy (uniform distribution)
            n_cells = torch.tensor(
                n_rows * n_cols, device=entropy.device, dtype=entropy.dtype
            )
            entropy_norm = entropy / torch.log(n_cells)
            confidence = 1.0 - entropy_norm  # (batch_size, n_keypoints)
        else:
            raise ValueError(f"Invalid confidence_method: {self.confidence_method}.")

        return xy_expected, confidence

    def _soft_argmax_1d(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits (torch.Tensor): Logits of shape
                (batch_size, n_keypoints, depth_n_bins).

        Returns:
            depth (torch.Tensor): Expected depth for each keypoint. Shape:
                (batch_size, n_keypoints).
            conf (torch.Tensor): Confidence scores for each keypoint.
                Shape: (batch_size, n_keypoints).
        """
        # probs: shape (batch_size, n_keypoints, depth_n_bins), same as logits
        probs = Pose2p5DModel._softmax_with_temp(
            logits, dim=-1, temperature=self.depth_temperature
        )
        # Compute expected depth (one scalar per keypoint per image)
        depth_expected = (probs * self.depth_bin_centers.view(1, 1, -1)).sum(dim=-1)

        # Compute confidence of prediction (one scalar per keypoint per image)
        if self.confidence_method == "peak":
            confidence = probs.max(dim=-1).values  # (batch_size, n_keypoints)
        elif self.confidence_method == "entropy":
            # See same operation in _soft_argmax_2d
            entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=-1)
            n_bins = torch.tensor(
                len(self.depth_bin_centers), device=entropy.device, dtype=entropy.dtype
            )
            entropy_norm = entropy / torch.log(n_bins)
            confidence = 1.0 - entropy_norm  # (batch_size, n_keypoints)
        else:
            raise ValueError(f"Invalid confidence_method: {self.confidence_method}.")

        return depth_expected, confidence

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor (batch_size, 3, 256, 256).

        Returns:
            dict with keys:
                "xy_heatmaps": (torch.Tensor) Predicted heatmaps of shape
                    (n_batches, n_keypoints, n_rows_out, n_cols_out).
                "depth_logits": (torch.Tensor) Predicted depth logits of
                    shape (n_batches, n_keypoints, depth_n_bins).
                "pred_xy": (torch.Tensor) Predicted x-y coordinates in
                    input image pixel space of shape
                    (n_batches, n_keypoints, 2).
                "pred_depth": (torch.Tensor) Predicted depth values of
                    shape (n_batches, n_keypoints).
                "conf_xy": (torch.Tensor) Confidence scores for x-y
                    predictions of shape (n_batches, n_keypoints).
                "conf_depth": (torch.Tensor) Confidence scores for depth
                    predictions of shape (n_batches, n_keypoints).

        We use a UNet-like architecture with skip connections:

        Encoder (downsampling)     Decoder (upsampling)
        ──────────────────────     ────────────────────
          e0: 64ch
                │
                │
                ↓                 **d1: upsample_core_out_channels 64x64**
          e1: 64ch 64x64            (end of upsampling core)
                │ │                       ↑
                │ └────────(skip)───────→(+)  dec_layer2: 128(up)+64(skip)->upsample_core_out_channels
                ↓                         ↑(up)           32x32(up)+64x64(skip)->64x64
          e2: 128ch 32x32           d2: 128ch 32x32
                │ │                       ↑
                │ └────────(skip)───────→(+)  dec_layer3: 256(up)+128(skip)->128 ch
                ↓                         ↑(up)           16x16(up)+32x32(skip)->32x32
          e3: 256ch 16x16           d3: 256ch 16x16
                │ │                       ↑
                │ └────────(skip)───────→(+)  dec_layer4: 512(up)+256(skip)->256 ch
                ↓                         ↑(up)           8x8(up)+16x16(skip)->16x16
          e4: 512ch 8x8             d4: 512ch 8x8
                |                         ↑
                └──(bottleneck/identity)──┘
        """
        # Run feature extractor
        e0, e1, e2, e3, e4 = self.feature_extractor.forward(
            x, return_intermediates=True
        )

        d4 = e4  # this is just the bottleneck

        # Upsample with skip connections
        d3 = self.dec_layer4(d4, e3)
        d2 = self.dec_layer3(d3, e2)
        d1 = self.dec_layer2(d2, e1)  # (N, upsample_core_out_channels, 64, 64)

        # Compute x-y heatmaps
        # Compute logits using heatmap head
        heatmaps = self.heatmap_head(d1)  # (N, n_keypoints, nrows_out, ncols_out)
        # Decode x-y coordinates from logits using soft-argmax
        # xy_px_out: x, y in heatmap pixel space, shape (N, n_keypoints, 2)
        # xy_conf: shape (N, n_keypoints)
        xy_px_out, xy_conf = self._soft_argmax_2d(heatmaps)

        # Map to input image pixel coordinates (input images are 256x256, but heatmaps
        # are predicted at 64x64, so there is a stride of 4)
        heatmap_size = heatmaps.shape[-2:]  # (n_rows_out, n_cols_out)
        stride = self.feature_extractor.input_size[0] / heatmap_size[0]
        assert stride == 4, "Expected input size=256x256 and output heatmap size=64x64"
        xy_px_in = xy_px_out * stride  # (N, n_keypoints, 2)

        # Compute depth distributions
        # Compute logits using depth head
        depth_logits = self.depth_head(d1)  # (N, n_keypoints, depth_n_bins)
        # Decode depth from logits using soft-argmax
        # depth_pos and depth_conf both of shape (N, n_keypoints)
        depth_pos, depth_conf = self._soft_argmax_1d(depth_logits)

        # If this is the first forward pass, check if the shapes are as expected
        if self._first_time_forward:
            batch_size = x.shape[0]
            assert self.feature_extractor.input_size == (256, 256)
            assert x.shape == (batch_size, 3, 256, 256)
            assert e0.shape == (batch_size, 64, 128, 128)
            assert e1.shape == (batch_size, 64, 64, 64)
            assert e2.shape == (batch_size, 128, 32, 32)
            assert e3.shape == (batch_size, 256, 16, 16)
            assert e4.shape == (batch_size, 512, 8, 8)
            assert d4.shape == (batch_size, 512, 8, 8)
            assert d3.shape == (batch_size, 256, 16, 16)
            assert d2.shape == (batch_size, 128, 32, 32)
            assert d1.shape == (batch_size, self.upsample_core_out_channels, 64, 64)

            assert heatmaps.shape == (batch_size, self.n_keypoints, *heatmap_size)
            assert xy_px_in.shape == (batch_size, self.n_keypoints, 2)
            assert xy_conf.shape == (batch_size, self.n_keypoints)
            assert xy_px_out.shape == (batch_size, self.n_keypoints, 2)
            assert stride == self.feature_extractor.input_size[0] / heatmap_size[0]
            assert stride == self.feature_extractor.input_size[1] / heatmap_size[1]

            depth_n_bins = self.depth_n_bins
            assert depth_logits.shape == (batch_size, self.n_keypoints, depth_n_bins)
            assert depth_pos.shape == (batch_size, self.n_keypoints)
            assert depth_conf.shape == (batch_size, self.n_keypoints)

            self._first_time_forward = False

        return {
            "xy_heatmaps": heatmaps,
            "depth_logits": depth_logits,
            "pred_xy": xy_px_in,
            "pred_depth": depth_pos,
            "conf_xy": xy_conf,
            "conf_depth": depth_conf,
            "heatmap_stride": stride,
        }


class Pose2p5DLoss(nn.Module):
    """Loss function for 2.5D pose estimation model. Combines loss on
    predicted heatmaps (x-y coordinates) and loss on predicted depth."""

    def __init__(
        self,
        heatmap_loss_func: str,
        heatmap_sigma: float = 2.0,
        depth_sigma_bins: float = 1.0,
        xy_loss_weight: float = 4.0,
        depth_ce_loss_weight: float = 1.0,
        depth_l1_loss_weight: float = 0.25,
        oob_treatment: str = "drop",
    ):
        """
        Args:
            heatmap_loss_func (str): Loss function to use for heatmaps.
                Options: "mse" or "kl".
            heatmap_sigma (float): Standard deviation of Gaussian used to
                create ground truth heatmaps from x-y labels.
            depth_sigma_bins (float): Standard deviation of Gaussian used
                to soften one-hot labels in depth cross-entropy loss (in
                number of bins). Higher values make the labels softer.
            xy_loss_weight (float): Weight for x-y loss.
            depth_ce_loss_weight (float): Weight for cross-entropy term in
                depth loss.
            depth_l1_loss_weight (float): Weight for L1 term in depth loss.
            oob_treatment (str): What to do with out-of-bounds depth
                labels. Options: "clamp" (clamp to valid range), "drop"
                (ignore OOB labels) or "ignore".
        """
        super().__init__()
        self.heatmap_loss_func = heatmap_loss_func.lower()
        self.heatmap_sigma = heatmap_sigma
        self.depth_sigma_bins = depth_sigma_bins
        self.xy_loss_weight = xy_loss_weight
        self.depth_ce_loss_weight = depth_ce_loss_weight
        self.depth_l1_loss_weight = depth_l1_loss_weight
        self.oob_treatment = oob_treatment.lower()

        # Check input validity
        if heatmap_loss_func not in ["mse", "kl"]:
            raise ValueError(
                f'Invalid heatmap_loss: {heatmap_loss_func}. Must be "mse" or "kl".'
            )
        if oob_treatment not in ["clamp", "drop", "ignore"]:
            raise ValueError(
                f"Invalid depth_oob_treatment: {oob_treatment}. "
                'Must be "clamp", "drop" or "ignore".'
            )

    @classmethod
    def create_from_config(
        cls, loss_config: config.LossConfig | Path | str
    ) -> "Pose2p5DLoss":
        # Load from file if config is given as a path
        if isinstance(loss_config, (Path, str)):
            loss_config = config.LossConfig.load(loss_config)
            logging.info(f"Loaded model loss config from {loss_config}")

        # Initialize loss from config
        obj = cls(
            heatmap_loss_func=loss_config.heatmap_loss_func,
            heatmap_sigma=loss_config.heatmap_sigma,
            depth_sigma_bins=loss_config.depth_sigma_bins,
            xy_loss_weight=loss_config.xy_loss_weight,
            depth_ce_loss_weight=loss_config.depth_ce_loss_weight,
            depth_l1_loss_weight=loss_config.depth_l1_loss_weight,
            oob_treatment=loss_config.oob_treatment,
        )

        logging.info("Created Pose2p5DLoss from loss config")
        return obj

    @staticmethod
    def _expand_xy_labels_to_gaussian_heatmaps(
        xy_labels: torch.Tensor, out_dim: tuple[int, int], sigma: float
    ) -> torch.Tensor:
        """Make ground truth Gaussian heatmaps from x-y labels.

        Args:
            xy_labels (torch.Tensor): X-Y coordinates of shape
                (batch_size, n_keypoints, 2) in heatmap pixel space (NOT
                input image pixel space!).
            out_dim (tuple[int, int]): Output dimensions (n_rows, n_cols).
            sigma (float): Standard deviation of Gaussian.

        Returns:
            torch.Tensor: Ground truth heatmaps of shape
                (batch_size, n_keypoints, *out_dim).
        """
        batch_size, n_keypoints, _ = xy_labels.shape
        n_rows_heatmap, n_cols_heatmap = out_dim
        device = xy_labels.device

        # Create meshgrid for heatmap coordinates
        rows_grid = torch.arange(n_rows_heatmap, device=device).view(
            1, 1, n_rows_heatmap, 1
        )
        cols_grid = torch.arange(n_cols_heatmap, device=device).view(
            1, 1, 1, n_cols_heatmap
        )

        # Expand xy_labels to match heatmap dimensions
        mu_col = xy_labels[..., 0].view(batch_size, n_keypoints, 1, 1)
        mu_row = xy_labels[..., 1].view(batch_size, n_keypoints, 1, 1)

        # Compute Gaussian heatmaps
        heatmaps = torch.exp(
            -(((cols_grid - mu_col) ** 2) + ((rows_grid - mu_row) ** 2))
            / (2 * sigma**2)
        )

        # Normalize each joint map to integrate to 1
        heatmaps = heatmaps / heatmaps.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

        return heatmaps

    @staticmethod
    def _compute_xy_heatmap_loss(
        loss_function: str, heatmaps: torch.Tensor, heatmap_labels: torch.Tensor
    ) -> torch.Tensor:
        """Loss on the X-Y coordinates prediction: compute the loss between
        predicted heatmaps and ground truth heatmaps.

        Args:
            loss_function (str): Loss function to use. Options: "mse" or
                "kl".
            heatmaps (torch.Tensor): Predicted heatmaps of shape
                (batch_size, n_keypoints, n_rows_out, n_cols_out).
            heatmap_labels (torch.Tensor): Ground truth heatmaps of shape
                (batch_size, n_keypoints, n_rows_out, n_cols_out).

        Returns:
            torch.Tensor: Computed loss (scalar).
        """
        batch_size, n_keypoints, n_rows_out, n_cols_out = heatmaps.shape
        if loss_function == "mse":
            return F.mse_loss(heatmaps, heatmap_labels)
        elif loss_function == "kl":
            # KL(target || pred); compute log-softmax over pixels
            logp = F.log_softmax(
                heatmaps.view(batch_size, n_keypoints, -1), dim=-1
            ).view(batch_size, n_keypoints, n_rows_out, n_cols_out)
            kl = (
                heatmap_labels * (torch.log(heatmap_labels.clamp_min(1e-6)) - logp)
            ).sum(dim=(2, 3))
            return kl.mean()
        else:
            raise ValueError(f"Invalid loss_function: {loss_function}.")

    @staticmethod
    def _compute_depth_ce_loss(
        depth_logits: torch.Tensor,
        z_labels: torch.Tensor,
        bin_values: torch.Tensor,
        sigma_bins: float = 0.1,
    ) -> torch.Tensor:
        """Compute the cross-entropy loss on the depth prediction (depth
        is treated as a classification problem over discrete bins). We use
        a "soft" version of the one-hot labels, where the label for each
        bin is computed based on a Gaussian centered at the ground truth
        depth value.

        Args:
            depth_logits (torch.Tensor): Predicted depth logits of shape
                (batch_size, n_keypoints, depth_n_bins).
            z_labels (torch.Tensor): Ground truth depth values of shape
                (batch_size, n_keypoints).
            bin_values (torch.Tensor): Center values (i.e. depths) of each
                discrete bin. Shape: (depth_n_bins,).
            sigma_bins (float): Standard deviation of Gaussian used to
                soften the one-hot labels (in number of bins).

        Returns:
            torch.Tensor: Computed cross-entropy loss (scalar).
        """
        bin_width = bin_values[1] - bin_values[0]
        sigma = sigma_bins * bin_width

        # Soften labels: compute normalized distances from each bin center and apply
        # a Gaussian function to get "soft" labels (actually implemented via softmax)
        normalized_dist_from_bin_centers = (
            z_labels.unsqueeze(-1) - bin_values.view(1, 1, -1)
        ) / sigma.clamp_min(1e-6)
        soft_labels = torch.softmax(
            -0.5 * (normalized_dist_from_bin_centers**2), dim=-1
        )

        # Cross-entropy with model predictions:
        # H(label, pred) = \sum_i label_i * log(pred_i)
        logp_pred = F.log_softmax(depth_logits, dim=-1)
        ce = -(soft_labels * logp_pred).sum(dim=-1)  # (batch_size, n_keypoints)
        return ce.mean()

    @staticmethod
    def _compute_depth_l1_loss(
        depth_logits: torch.Tensor, z_labels: torch.Tensor, bin_values: torch.Tensor
    ) -> torch.Tensor:
        """Compute the L1 loss on the depth prediction (regression on
        expected depth computed based on logits).

        Args:
            depth_logits (torch.Tensor): Predicted depth logits of shape
                (batch_size, n_keypoints, depth_n_bins).
            z_labels (torch.Tensor): Ground truth depth values of shape
                (batch_size, n_keypoints).
            bin_values (torch.Tensor): Center values (i.e. depths) of each
                discrete bin. Shape: (depth_n_bins,).

        Returns:
            torch.Tensor: Computed L1 loss (scalar).
        """
        # Compute probs, shape still (batch_size, n_keypoints, depth_n_bins)
        bin_probs = F.softmax(depth_logits, dim=-1)
        # Compute expected depth, shape is now (batch_size, n_keypoints)
        depth_expected = (bin_probs * bin_values.view(1, 1, -1)).sum(dim=-1)
        l1_loss = F.l1_loss(depth_expected, z_labels, reduction="mean")
        return l1_loss

    def _get_oob_masks(
        self,
        xy_labels_in_output_dim: torch.Tensor,
        depth_labels: torch.Tensor,
        heatmap_size: tuple[int, int],
        bin_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # heatmap_size is (n_rows, n_cols)
        n_rows, n_cols = heatmap_size
        # xy_oob: (batch_size,), .any(dim=1) over keypoints
        xy_oob = (
            (xy_labels_in_output_dim[..., 0] < 0)
            | (xy_labels_in_output_dim[..., 0] >= n_cols)
            | (xy_labels_in_output_dim[..., 1] < 0)
            | (xy_labels_in_output_dim[..., 1] >= n_rows)
        ).any(dim=1)

        depth_too_small = depth_labels < bin_values[0]
        depth_too_large = depth_labels > bin_values[-1]
        depth_oob = (depth_too_small | depth_too_large).any(dim=1)  # (batch_size,)

        combined_oob = xy_oob | depth_oob
        return xy_oob, depth_oob, combined_oob  # all (batch_size,)

    def _treat_oob(
        self,
        xy_labels_in_output_dim: torch.Tensor,
        depth_labels: torch.Tensor,
        xy_heatmaps: torch.Tensor,
        depth_logits: torch.Tensor,
        heatmap_size: tuple[int, int],
        bin_values: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        xy_oob, depth_oob, combined_oob = self._get_oob_masks(
            xy_labels_in_output_dim, depth_labels, heatmap_size, bin_values
        )
        if combined_oob.any():
            n_xy_oob = xy_oob.sum().item()
            n_depth_oob = depth_oob.sum().item()
            n_combined_oob = combined_oob.sum().item()
            logging.warning(
                f"Found {n_combined_oob} samples with OOB labels "
                f"({n_xy_oob} with OOB x-y, {n_depth_oob} with OOB depth) "
                f"out of {xy_labels_in_output_dim.shape[0]} in the current batch. "
                f"Using oob_treatment='{self.oob_treatment}'."
            )

            device = xy_labels_in_output_dim.device

            if self.oob_treatment == "clamp":
                n_rows, n_cols = heatmap_size
                min_xy = torch.tensor([0, 0], device=device)
                max_xy = torch.tensor([n_cols - 1, n_rows - 1], device=device)
                xy_labels_in_output_dim = torch.clamp(
                    xy_labels_in_output_dim, min=min_xy, max=max_xy
                )

                min_depth = bin_values[0]
                max_depth = bin_values[-1]
                depth_labels = torch.clamp(depth_labels, min=min_depth, max=max_depth)

            elif self.oob_treatment == "drop":
                if combined_oob.all():
                    logging.error(
                        "All labels are out-of-bounds. This should be very alarming."
                    )
                    return None
                if combined_oob.any():
                    keep_mask = ~combined_oob  # (batch_size,)
                    xy_labels_in_output_dim = xy_labels_in_output_dim[keep_mask, :, :]
                    depth_labels = depth_labels[keep_mask, :]
                    xy_heatmaps = xy_heatmaps[keep_mask, :, :, :]
                    depth_logits = depth_logits[keep_mask, :, :]

            elif self.oob_treatment == "ignore":
                pass  # do nothing

            else:
                raise ValueError(f"Invalid oob_treatment: {self.oob_treatment}.")

        return {
            "xy_labels_in_output_dim": xy_labels_in_output_dim,
            "depth_labels": depth_labels,
            "xy_heatmaps": xy_heatmaps,
            "depth_logits": depth_logits,
        }

    def forward(
        self,
        pred_dict: dict[str, torch.Tensor],
        xy_labels: torch.Tensor,
        depth_labels: torch.Tensor,
        bin_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred_dict (dict[str, torch.Tensor]): Output of
                Pose2p5DModel.forward().
            xy_labels (torch.Tensor): Ground truth x-y coordinates of shape
                (batch_size, n_keypoints, 2) in input image pixel space.
            depth_labels (torch.Tensor): Ground truth depth values of shape
                (batch_size, n_keypoints).
            bin_values (torch.Tensor): Center values (i.e. depths) of each
                discrete bin. Shape: (depth_n_bins,).

        Returns:
            dict with keys:
                "total_loss": (torch.Tensor) Total loss (scalar).
                "xy_heatmap_loss": (torch.Tensor) Loss on X-Y prediction
                    (scalar).
                "depth_ce_loss": (torch.Tensor) Cross-entropy loss on depth
                    prediction (scalar).
                "depth_l1_loss": (torch.Tensor) L1 loss on depth prediction
                    (scalar).
        """
        xy_heatmaps = pred_dict["xy_heatmaps"]
        depth_logits = pred_dict["depth_logits"]
        heatmap_stride = pred_dict["heatmap_stride"]
        heatmap_size = xy_heatmaps.shape[-2:]

        # Convert xy labels (image coords) -> heatmap coords
        xy_labels_in_output_dim = xy_labels / heatmap_stride

        # Treat out-of-bounds (OOB) labels according to self.oob_treatment
        oob_treated_data = self._treat_oob(
            xy_labels_in_output_dim,
            depth_labels,
            xy_heatmaps,
            depth_logits,
            heatmap_size,
            bin_values,
        )
        if oob_treated_data is None:
            logging.error(
                "All labels are out-of-bounds. Returning zero loss to avoid NaNs. "
                "This should be very alarming."
            )
            device = xy_labels_in_output_dim.device
            return {
                "total_loss": torch.tensor(0.0, device=device),
                "xy_heatmap_loss": torch.tensor(0.0, device=device),
                "depth_ce_loss": torch.tensor(0.0, device=device),
                "depth_l1_loss": torch.tensor(0.0, device=device),
            }
        xy_labels_in_output_dim = oob_treated_data["xy_labels_in_output_dim"]
        depth_labels = oob_treated_data["depth_labels"]
        xy_heatmaps = oob_treated_data["xy_heatmaps"]
        depth_logits = oob_treated_data["depth_logits"]

        # Expand xy labels to heatmap labels
        heatmap_labels = self._expand_xy_labels_to_gaussian_heatmaps(
            xy_labels_in_output_dim,
            heatmap_size,
            sigma=self.heatmap_sigma,
        )

        # Compute x-y prediction loss on heatmaps
        xy_heatmap_loss = self._compute_xy_heatmap_loss(
            self.heatmap_loss_func, xy_heatmaps, heatmap_labels
        )

        # Compute depth losses
        depth_ce_loss = self._compute_depth_ce_loss(
            depth_logits,
            depth_labels,
            bin_values,
            self.depth_sigma_bins,
        )
        depth_l1_loss = self._compute_depth_l1_loss(
            depth_logits, depth_labels, bin_values
        )

        # Compute final loss
        total_loss = (
            self.xy_loss_weight * xy_heatmap_loss
            + self.depth_ce_loss_weight * depth_ce_loss
            + self.depth_l1_loss_weight * depth_l1_loss
        )
        return {
            "total_loss": total_loss,
            "xy_heatmap_loss": xy_heatmap_loss,
            "depth_ce_loss": depth_ce_loss,
            "depth_l1_loss": depth_l1_loss,
        }
