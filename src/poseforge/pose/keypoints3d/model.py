import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

import poseforge.pose.keypoints3d.config as config
from poseforge.pose.common import (
    ResNetFeatureExtractor,
    DecoderBlock,
    BottleneckSelfAttention,
)


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
        activation_noise_std: float = 0.0,
        decoder_spatial_dropout_p: float = 0.0,
        heatmap_n_hidden_layers: int = 0,
        heatmap_hidden_channels: int = 64,
        xy_decode_mode: str = "local_window",
        xy_decode_window: int = 11,
        coord_conv_enabled: bool = False,
        bottleneck_attention_enabled: bool = False,
        bottleneck_attention_n_heads: int = 4,
        excluded_keypoint_indices: tuple[int, ...] = (),
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
            activation_noise_std (float): Standard deviation of multiplicative
                noise applied to activations during training.
            decoder_spatial_dropout_p (float): Probability for spatial
                dropout (Dropout2d) applied between decoder layers during
                training. Drops entire feature map channels to prevent
                the decoder from overfitting to synthetic-specific spatial
                patterns. Set to 0.0 to disable (default).
            heatmap_n_hidden_layers (int): Number of hidden (3x3 conv ->
                GroupNorm -> ReLU) layers inserted in the x-y heatmap head
                BEFORE the existing final 3x3 conv. 0 = single-conv head
                (default, backward-compatible). Each added layer grows the
                head's receptive field by 2 heatmap pixels.
            heatmap_hidden_channels (int): Width of the hidden layers in
                the heatmap head (only used when heatmap_n_hidden_layers >
                0). Must be a divisor multiple of groupnorm_n_groups.
            xy_decode_mode (str): How the x-y heatmap is turned into a
                coordinate. "local_window" (default) takes the argmax peak
                and computes the soft-argmax (expectation) only within a
                window of side ``xy_decode_window`` around it, so a spurious
                secondary bump elsewhere cannot drag the estimate into the
                empty space between two modes. "global" computes the
                expectation over the entire heatmap (legacy behavior; kept
                for A/B comparison and reproducing older results). This only
                affects decoding at inference/eval time -- the training loss
                is computed on the raw heatmap, not on the decoded point, so
                switching modes does not require retraining.
            xy_decode_window (int): Side length, in heatmap pixels, of the
                window used when ``xy_decode_mode == "local_window"``. The
                window spans peak +/- (xy_decode_window // 2). Must be a
                positive integer. Ignored for "global".
            coord_conv_enabled (bool): If True, append two normalized
                coordinate channels (x and y, each in [-1, 1]) to the x-y
                heatmap head's input (CoordConv, Liu et al. 2018). This gives
                the otherwise translation-equivariant head an absolute-position
                signal, so a keypoint that always appears in a specific image
                region (e.g. a foreleg in an aligned/cropped frame) is less
                likely to be predicted in an impossible location. Only the
                heatmap head gets the extra channels; the depth head is
                global-pooled, so per-pixel coordinates would wash out there.
                Default False (opt-in). NOTE: this relies on the input being
                spatially registered -- if alignment drifts at test time, a
                position-conditioned head can hurt.
            bottleneck_attention_enabled (bool): If True, insert a multi-head
                self-attention block (with 2D sinusoidal positional encoding
                and a residual connection) on the ResNet bottleneck ``e4``
                before decoding. This gives every bottleneck location direct,
                content-selected global context in one hop -- targeting
                keypoint identity/plausibility errors that a small effective
                receptive field cannot resolve. Cheap because ``e4`` has very
                few spatial tokens. The block is identity at init (zero-init
                output projection), so training starts unchanged. Default
                False (opt-in).
            bottleneck_attention_n_heads (int): Number of attention heads for
                the bottleneck self-attention block. Must divide the
                bottleneck channel count (512 for ResNet-18). Only used when
                ``bottleneck_attention_enabled`` is True.
            excluded_keypoint_indices (tuple[int, ...]): Indices (into the
                full set of ``n_keypoints`` label keypoints) that the model
                should NOT predict. The prediction heads are sized to emit
                only the remaining (kept) keypoints, in their original
                relative order, so these keypoints are removed from the model
                entirely rather than left unsupervised. Empty (default) =
                predict all ``n_keypoints`` keypoints.
        """
        super().__init__()
        # ``n_keypoints`` is the size of the full label set (e.g. 32 canonical
        # keypoints). ``excluded_keypoint_indices`` are dropped, so the model
        # actually predicts ``n_predicted_keypoints`` keypoints, namely the
        # ``included_keypoint_indices`` (kept in original relative order).
        self.n_keypoints = n_keypoints
        self.excluded_keypoint_indices = tuple(sorted(set(excluded_keypoint_indices)))
        if any(i < 0 or i >= n_keypoints for i in self.excluded_keypoint_indices):
            raise ValueError(
                f"excluded_keypoint_indices {self.excluded_keypoint_indices} contains "
                f"indices out of range for n_keypoints={n_keypoints}."
            )
        self.included_keypoint_indices = tuple(
            i for i in range(n_keypoints) if i not in set(self.excluded_keypoint_indices)
        )
        self.n_predicted_keypoints = len(self.included_keypoint_indices)
        if self.n_predicted_keypoints == 0:
            raise ValueError("All keypoints are excluded; nothing left to predict.")
        # Long tensor of the kept indices, used to slice the (full) keypoint
        # labels down to the predicted subset. Non-persistent: it is derived
        # from config and need not live in the checkpoint.
        self.register_buffer(
            "included_keypoint_indices_t",
            torch.tensor(self.included_keypoint_indices, dtype=torch.long),
            persistent=False,
        )
        if self.excluded_keypoint_indices:
            logging.info(
                f"Model predicts {self.n_predicted_keypoints}/{n_keypoints} keypoints; "
                f"excluding indices {self.excluded_keypoint_indices}."
            )
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
        self.activation_noise_std = activation_noise_std
        self.decoder_spatial_dropout_p = decoder_spatial_dropout_p
        self.heatmap_n_hidden_layers = heatmap_n_hidden_layers
        self.heatmap_hidden_channels = heatmap_hidden_channels
        self.xy_decode_mode = xy_decode_mode.lower()
        self.xy_decode_window = xy_decode_window
        self.coord_conv_enabled = coord_conv_enabled
        # Number of coordinate channels appended to the heatmap head input when
        # CoordConv is enabled (x and y).
        self._n_coord_channels = 2 if coord_conv_enabled else 0
        self.bottleneck_attention_enabled = bottleneck_attention_enabled
        self.bottleneck_attention_n_heads = bottleneck_attention_n_heads

        # Spatial dropout for decoder (drops entire channels)
        # nn.Dropout2d is a no-op when p=0.0 or in eval mode
        self.decoder_dropout = nn.Dropout2d(p=decoder_spatial_dropout_p)

        # Check input validity
        if confidence_method not in ["entropy", "peak"]:
            raise ValueError(
                f"Invalid confidence_method: {confidence_method}. "
                'Must be "entropy" or "peak".'
            )
        if self.xy_decode_mode not in ["local_window", "global"]:
            raise ValueError(
                f"Invalid xy_decode_mode: {xy_decode_mode}. "
                'Must be "local_window" or "global".'
            )
        if self.xy_decode_mode == "local_window" and self.xy_decode_window < 1:
            raise ValueError(
                f"xy_decode_window must be a positive integer, got {xy_decode_window}."
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
        if heatmap_n_hidden_layers > 0 and (
            (heatmap_hidden_channels % groupnorm_n_groups) != 0
            or groupnorm_n_groups > heatmap_hidden_channels
        ):
            raise ValueError(
                "When heatmap_n_hidden_layers > 0, heatmap_hidden_channels must be "
                "a multiple of groupnorm_n_groups and >= groupnorm_n_groups "
                f"(got heatmap_hidden_channels={heatmap_hidden_channels}, "
                f"groupnorm_n_groups={groupnorm_n_groups})."
            )

        self.feature_extractor = feature_extractor

        # Optional multi-head self-attention on the bottleneck (e4) for global
        # context. Identity at init, so it does not disturb early training.
        if bottleneck_attention_enabled:
            self.bottleneck_attention = BottleneckSelfAttention(
                channels=feature_extractor.output_channels,
                n_heads=bottleneck_attention_n_heads,
            )
        else:
            self.bottleneck_attention = None

        # Create decoder core with skipped connections for upsampling
        # We use decoder4/3/2/1 to mirror layers1/2/3/4 in the ResNet encoder
        # Note that when upsampling, we actually go in the reverse order (4-3-2-1) from
        # the bottleneck.
        # Also note that ResNet18 operates at 128x128 after the initial
        # Conv-BN-ReLU (but before maxpool). We upsample to 128x128 to match this
        # (stride 2 compared to the input). The output heatmaps will thus be 128x128.
        assert feature_extractor.output_channels == 512  # expected from ResNet18 layer4
        self.dec_layer4 = DecoderBlock(512, 256, 256)  # 512ch 8x8 -> 256ch 16x16
        self.dec_layer3 = DecoderBlock(256, 128, 128)  # 256ch 16x16 -> 128ch 32x32
        self.dec_layer2 = DecoderBlock(128, 64, 64)  # 128ch 32x32 -> 64ch 64x64
        # last layer (dec_layer1): # 64ch 64x64 -> upsample_core_out_channels, 128x128
        self.dec_layer1 = DecoderBlock(64, 64, upsample_core_out_channels)

        # Heatmap head for (x, y) keypoint locations. Sized to the predicted
        # (kept) keypoints only. When CoordConv is enabled, the head also
        # consumes the two appended coordinate channels, so widen its input.
        self.heatmap_head = self._build_heatmap_head(
            in_channels=upsample_core_out_channels + self._n_coord_channels,
            out_channels=self.n_predicted_keypoints,
            n_hidden_layers=heatmap_n_hidden_layers,
            hidden_channels=heatmap_hidden_channels,
        )

        # Depth head for distance from camera
        self.depth_head = self._build_depth_head(
            in_channels=upsample_core_out_channels,
            hidden_channels=depth_hidden_channels,
            n_keypoints=self.n_predicted_keypoints,
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
        feature_extractor = ResNetFeatureExtractor(
            backbone=architecture_config.backbone
        )

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
            activation_noise_std=architecture_config.activation_noise_std,
            decoder_spatial_dropout_p=architecture_config.decoder_spatial_dropout_p,
            heatmap_n_hidden_layers=architecture_config.heatmap_n_hidden_layers,
            heatmap_hidden_channels=architecture_config.heatmap_hidden_channels,
            xy_decode_mode=architecture_config.xy_decode_mode,
            xy_decode_window=architecture_config.xy_decode_window,
            coord_conv_enabled=architecture_config.coord_conv_enabled,
            bottleneck_attention_enabled=architecture_config.bottleneck_attention_enabled,
            bottleneck_attention_n_heads=architecture_config.bottleneck_attention_n_heads,
            excluded_keypoint_indices=architecture_config.excluded_keypoint_indices,
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
            # A joint segpose checkpoint stores {"pose": ..., "bodyseg": ...};
            # transparently pick the pose slice so this inference path stays unchanged.
            if isinstance(weights, dict) and "pose" in weights and "bodyseg" in weights:
                weights = weights["pose"]
            self.load_state_dict(weights)
            logging.info(
                f"Loaded Pose2p5DModel weights (inc. feature extractor) from config"
            )
            return

        # Otherwise, init feature extractor first. Preserve the backbone chosen
        # at architecture-creation time so pretrained resnet34 weights load into
        # a resnet34 (not the default resnet18).
        self.feature_extractor = ResNetFeatureExtractor(
            # Path, str, or "IMAGENET1K_V1"
            weights=weights_config.feature_extractor_weights,
            backbone=self.feature_extractor.backbone,
        )
        logging.info("Set up feature extractor from config")

    def _build_heatmap_head(
        self,
        in_channels: int,
        out_channels: int,
        n_hidden_layers: int = 0,
        hidden_channels: int = 64,
    ) -> nn.Module:
        """Build the x-y heatmap head.

        - ``n_hidden_layers == 0`` (default): a single 3x3 conv mapping
          decoder features straight to per-keypoint heatmap logits. This is
          the backward-compatible behavior — the returned module is a single
          ``nn.Conv2d``.
        - ``n_hidden_layers > 0``: prepend ``n_hidden_layers`` blocks of
          ``Conv2d(3x3, no bias) -> GroupNorm -> ReLU`` at width
          ``hidden_channels`` BEFORE the existing final 3x3 conv. The first
          hidden block projects ``in_channels -> hidden_channels``; any
          subsequent hidden block is ``hidden_channels -> hidden_channels``.
          The final conv (``hidden_channels -> out_channels``) keeps the
          small-std init since it is not followed by ReLU.
        """
        # Final 3x3 conv (always present). Its in_channels depends on whether
        # we have any hidden layers in front of it.
        final_in = hidden_channels if n_hidden_layers > 0 else in_channels
        final_conv = nn.Conv2d(
            final_in, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        # Initialize final conv weights using normal initialization with small std
        # (this is not followed by ReLU, so Kaiming init is not appropriate here)
        nn.init.normal_(final_conv.weight, std=self.pose_head_init_std)
        nn.init.zeros_(final_conv.bias)

        if n_hidden_layers == 0:
            return final_conv

        layers: list[nn.Module] = []
        c_in = in_channels
        for _ in range(n_hidden_layers):
            conv = nn.Conv2d(
                c_in, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False
            )
            # Conv is followed by ReLU -> Kaiming init is appropriate
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            groupnorm = nn.GroupNorm(self.groupnorm_n_groups, hidden_channels)
            nn.init.constant_(groupnorm.weight, 1)
            nn.init.constant_(groupnorm.bias, 0)
            layers += [conv, groupnorm, nn.ReLU(inplace=True)]
            c_in = hidden_channels
        layers.append(final_conv)
        return nn.Sequential(*layers)

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
    def _coord_channels(feat: torch.Tensor) -> torch.Tensor:
        """Build two normalized coordinate channels (x, y in [-1, 1]) matching
        the spatial size, batch, dtype and device of ``feat``.

        Args:
            feat (torch.Tensor): Feature map of shape (N, C, H, W).

        Returns:
            torch.Tensor: Coordinate channels of shape (N, 2, H, W); channel 0
                is x (varies across columns), channel 1 is y (varies across
                rows).
        """
        n, _, h, w = feat.shape
        x = torch.linspace(-1.0, 1.0, w, device=feat.device, dtype=feat.dtype)
        y = torch.linspace(-1.0, 1.0, h, device=feat.device, dtype=feat.dtype)
        x_channel = x.view(1, 1, 1, w).expand(n, 1, h, w)
        y_channel = y.view(1, 1, h, 1).expand(n, 1, h, w)
        return torch.cat([x_channel, y_channel], dim=1)

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

        # Extract the X-Y coordinates from the heatmaps as the expected (x, y)
        # position, i.e. the probability-weighted average of the grid coordinates.
        x_grid, y_grid = self._get_heatmap_xy_grid(heatmaps)

        if self.xy_decode_mode == "local_window":
            # Anchor the expectation at the argmax peak and only average within a
            # +/- half-window box around it. A distant spurious bump then falls
            # outside the window and cannot drag the estimate into the empty
            # valley between two modes. Restricting-then-renormalizing the
            # softmax over the window is exactly a softmax over that window, so
            # this is a local soft-argmax (still sub-pixel within the window).
            half = self.xy_decode_window // 2
            peak_idx = heatmaps.reshape(batch_size, n_keypoints, -1).argmax(dim=-1)
            peak_col = (peak_idx % n_cols).to(probs.dtype).view(
                batch_size, n_keypoints, 1, 1
            )
            peak_row = (peak_idx // n_cols).to(probs.dtype).view(
                batch_size, n_keypoints, 1, 1
            )
            within_window = ((x_grid - peak_col).abs() <= half) & (
                (y_grid - peak_row).abs() <= half
            )
            masked = probs * within_window.to(probs.dtype)
            masked = masked / masked.sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)
        else:  # "global": expectation over the entire heatmap (legacy behavior)
            masked = probs

        # Dimensions 2 and 3 are rows and cols
        x_expected = (masked * x_grid).sum(dim=(2, 3))  # (batch_size, n_keypoints)
        y_expected = (masked * y_grid).sum(dim=(2, 3))  # (batch_size, n_keypoints)
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

    @staticmethod
    def pad_inputs(
        x: torch.Tensor, multiple: int = 32
    ) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        """Pad an input image tensor to a size divisible by ``multiple``.

        The padding is applied on the bottom and right only, so x-y labels in
        the original image coordinate system do not need to be shifted.

        Returns:
            padded_x: The padded input tensor.
            orig_size: The original (height, width).
            padded_size: The padded (height, width).
        """
        orig_height, orig_width = x.shape[2], x.shape[3]

        def pad_to_multiple(size: int) -> int:
            return ((size + multiple - 1) // multiple) * multiple

        padded_height = pad_to_multiple(orig_height)
        padded_width = pad_to_multiple(orig_width)

        if padded_height != orig_height or padded_width != orig_width:
            pad_height = padded_height - orig_height
            pad_width = padded_width - orig_width
            x = F.pad(x, (0, pad_width, 0, pad_height), mode="constant", value=0)

        return x, (orig_height, orig_width), (padded_height, padded_width)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).
                Height and width will be padded to nearest multiple of 32 if necessary.

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
          e0: 64ch 128x128          d0: upsample_core_out_channels ch, 128x128 (end of upsampling core)
                │ │                       ↑
                │ └────────(skip)───────→(+)  dec_layer1: 64(up)+64(skip)->upsample_core_out_channels ch
                ↓                         ↑(up)
          e1: 64ch 64x64            d1: 64ch 64x64
                │ │                       ↑
                │ └────────(skip)───────→(+)  dec_layer2: 128(up)+64(skip)->64 ch
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
        # Pad the input to a size divisible by 32. Because the padding is only
        # on the bottom and right, keypoint coordinates remain valid as-is.
        x, orig_size, padded_size = self.pad_inputs(x, multiple=32)
        orig_height, orig_width = orig_size
        padded_height, padded_width = padded_size
        
        # Run feature extractor
        e0, e1, e2, e3, e4 = self.feature_extractor.forward(
            x, return_intermediates=True
        )

        if self.training and self.activation_noise_std > 0:
            e0 = e0 * (1.0 + torch.randn_like(e0) * self.activation_noise_std)
            e1 = e1 * (1.0 + torch.randn_like(e1) * self.activation_noise_std)
            e2 = e2 * (1.0 + torch.randn_like(e2) * self.activation_noise_std)
            e3 = e3 * (1.0 + torch.randn_like(e3) * self.activation_noise_std)
            e4 = e4 * (1.0 + torch.randn_like(e4) * self.activation_noise_std)

        # Optional global-context mixing on the bottleneck via self-attention.
        # Shape (N, 512, H, W) is preserved, so the decoder is unaffected.
        if self.bottleneck_attention is not None:
            e4 = self.bottleneck_attention(e4)

        d4 = e4  # this is just the bottleneck

        # Upsample with skip connections and spatial dropout between layers
        d3 = self.decoder_dropout(self.dec_layer4(d4, e3))
        d2 = self.decoder_dropout(self.dec_layer3(d3, e2))
        d1 = self.decoder_dropout(self.dec_layer2(d2, e1))
        d0 = self.dec_layer1(d1, e0)  # (N, upsample_core_out_channels, 128, 128)
        # No dropout after the last decoder layer — let the heads see clean features

        # Compute x-y heatmaps
        # Optionally append normalized coordinate channels (CoordConv) so the
        # otherwise position-agnostic head can condition on absolute location.
        # Only the heatmap head sees them; the depth head keeps clean features.
        if self.coord_conv_enabled:
            heatmap_head_input = torch.cat([d0, self._coord_channels(d0)], dim=1)
        else:
            heatmap_head_input = d0
        # Compute logits using heatmap head
        heatmaps = self.heatmap_head(heatmap_head_input)  # (N, n_keypoints, nrows_out, ncols_out)
        # Decode x-y coordinates from logits using soft-argmax
        # xy_px_out: x, y in heatmap pixel space, shape (N, n_keypoints, 2)
        # xy_conf: shape (N, n_keypoints)
        xy_px_out, xy_conf = self._soft_argmax_2d(heatmaps)

        # Map to input image pixel coordinates
        heatmap_size = heatmaps.shape[-2:]  # (n_rows_out, n_cols_out)
        # Stride from heatmap -> padded input space
        stride_padded = padded_height / heatmap_size[0]
        xy_px_padded = xy_px_out * stride_padded  # (N, n_keypoints, 2) - in padded space

        # Convert back to original input space if input was padded
        if orig_size != padded_size:
            scale_factor_h = orig_height / padded_height
            scale_factor_w = orig_width / padded_width
            xy_px_in = xy_px_padded.clone()
            xy_px_in[..., 0] *= scale_factor_w  # x coordinate (width)
            xy_px_in[..., 1] *= scale_factor_h  # y coordinate (height)
        else:
            xy_px_in = xy_px_padded

        # Compute the stride that maps heatmap -> original image pixel space
        # This is what the loss function expects for label conversion
        stride = orig_height / heatmap_size[0]

        # Compute depth distributions
        # Compute logits using depth head
        depth_logits = self.depth_head(d0)  # (N, n_keypoints, depth_n_bins)
        # Decode depth from logits using soft-argmax
        # depth_pos and depth_conf both of shape (N, n_keypoints)
        depth_pos, depth_conf = self._soft_argmax_1d(depth_logits)

        # If this is the first forward pass, check if the shapes are as expected
        if self._first_time_forward:
            batch_size = x.shape[0]
            # Check that padded input size is divisible by 32 (total stride of ResNet)
            assert padded_height % 32 == 0 and padded_width % 32 == 0, \
                f"Padded input spatial dims must be divisible by 32, got {(padded_height, padded_width)}"
            
            # Check intermediate feature map shapes follow expected downsampling pattern
            # Each layer should be half the spatial size of the previous with correct channels
            assert e0.shape[1] == 64, f"e0 should have 64 channels, got {e0.shape[1]}"
            assert e1.shape[1] == 64, f"e1 should have 64 channels, got {e1.shape[1]}"
            assert e2.shape[1] == 128, f"e2 should have 128 channels, got {e2.shape[1]}"
            assert e3.shape[1] == 256, f"e3 should have 256 channels, got {e3.shape[1]}"
            assert e4.shape[1] == 512, f"e4 should have 512 channels, got {e4.shape[1]}"
            
            # Check spatial downsampling: each should be half the previous
            assert e0.shape[2] == x.shape[2] // 2, "e0 spatial size mismatch"
            assert e1.shape[2] == e0.shape[2] // 2, "e1 spatial size mismatch"
            assert e2.shape[2] == e1.shape[2] // 2, "e2 spatial size mismatch"
            assert e3.shape[2] == e2.shape[2] // 2, "e3 spatial size mismatch"
            assert e4.shape[2] == e3.shape[2] // 2, "e4 spatial size mismatch"
            
            # Check decoder shapes
            assert d0.shape == (batch_size, self.upsample_core_out_channels, e0.shape[2], e0.shape[3]), \
                f"d0 shape mismatch: expected (*,{self.upsample_core_out_channels}, {e0.shape[2]}, {e0.shape[3]}), got {d0.shape}"

            # Check output shapes (the model emits only the predicted/kept keypoints)
            n_pred = self.n_predicted_keypoints
            assert heatmaps.shape == (batch_size, n_pred, *heatmap_size), \
                f"heatmaps shape mismatch"
            assert xy_px_in.shape == (batch_size, n_pred, 2), \
                f"xy_px_in shape mismatch"
            assert xy_conf.shape == (batch_size, n_pred), \
                f"xy_conf shape mismatch"

            # Check strides are positive
            assert stride_padded > 0, f"stride_padded should be positive, got {stride_padded}"
            assert stride > 0, f"stride (original-image) should be positive, got {stride}"

            depth_n_bins = self.depth_n_bins
            assert depth_logits.shape == (batch_size, n_pred, depth_n_bins), \
                f"depth_logits shape mismatch"
            assert depth_pos.shape == (batch_size, n_pred), \
                f"depth_pos shape mismatch"
            assert depth_conf.shape == (batch_size, n_pred), \
                f"depth_conf shape mismatch"

            self._first_time_forward = False

        return {
            "xy_heatmaps": heatmaps,
            "depth_logits": depth_logits,
            "pred_xy": xy_px_in,
            "pred_depth": depth_pos,
            "conf_xy": xy_conf,
            "conf_depth": depth_conf,
            # Return the stride that maps heatmap coordinates to ORIGINAL image pixels.
            # The loss / target creation expects labels to be divided by this value
            # (i.e. heatmap_px = img_px / heatmap_stride). Using the original-image
            # based stride ensures consistency when inputs were padded on the
            # bottom/right only.
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
