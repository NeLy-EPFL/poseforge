import torch
import torch.nn as nn
import torch.nn.functional as F

from biomechpose.pose_estimation.feature_extractor import ResNetFeatureExtractor


class Pose2p5DModel(nn.Module):
    """2.5D Pose estimation model (for each keypoint, camera col-row coords
    are predicted via a 2D dense heatmap; depth is predicted separately via
    another 1D heatmap)."""

    def __init__(
        self,
        n_keypoints: int,
        backbone: ResNetFeatureExtractor,
        depth_n_bins: int,
        depth_min: float,
        depth_max: float,
        xy_temperature: float,
        depth_temperature: float,
        upsample_n_layers: int = 3,
        upsample_n_hidden_channels: int = 256,
        depth_n_hidden_channels: int = 256,
        confidence_method: str = "entropy",
    ):
        """
        Args:
            n_keypoints (int): Number of keypoints to predict.
            backbone (ResNetFeatureExtractor): Backbone feature extractor.
            depth_n_bins (int): Number of discrete bins for depth
                prediction.
            depth_min (float): Minimum depth value (closest to camera).
            depth_max (float): Maximum depth value (farthest from camera).
            xy_temperature (float): Temperature for soft-argmax in x-y
                heatmaps.
            depth_temperature (float): Temperature for soft-argmax in depth
                logits.
            upsample_n_layers (int): Number of upsampling layers (deconvs).
            upsample_n_hidden_channels (int): Number of hidden channels in
                upsampling layers.
            depth_n_hidden_channels (int): Number of hidden channels in
                depth head.
            confidence_method (str): Method to compute confidence scores in
                soft argmax of x-y heatmaps and depth logits. Options:
                "entropy" (1 - normalized entropy in predicted
                distribution, default) or "peak" (max probability).
        """
        super().__init__()
        self.n_keypoints = n_keypoints
        self.backbone = backbone
        self.depth_n_bins = depth_n_bins
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.xy_temperature = xy_temperature
        self.depth_temperature = depth_temperature
        self.upsample_n_layers = upsample_n_layers
        self.upsample_n_hidden_channels = upsample_n_hidden_channels
        self.depth_n_hidden_channels = depth_n_hidden_channels
        self.confidence_method = confidence_method

        if confidence_method not in ["entropy", "peak"]:
            raise ValueError(
                f"Invalid confidence_method: {confidence_method}. "
                'Must be "entropy" or "peak".'
            )

        # Build upsampling core. This is the first level of processing after the ResNet
        # feature extractor, shared by both the heatmap head and the depth head.
        self.upsampling_core = self._build_upsampling_core(
            n_layers=upsample_n_layers,
            n_hidden_channels=upsample_n_hidden_channels,
            n_channels_in=self.backbone.out_channels,
        )

        # Heatmap head for (x, y) keypoint locations
        self.heatmap_head = self._build_heatmap_head(
            n_channels_in=self.backbone.out_channels, n_keypoints=n_keypoints
        )

        # Depth head for distance from camera
        self.depth_head = self._build_depth_head(
            n_keypoints=n_keypoints,
            n_bins=depth_n_bins,
            n_hidden_channels=depth_n_hidden_channels,
            n_channels_in=self.backbone.out_channels,
        )
        # Precompute depth bin centers
        self.register_buffer(
            "depth_bin_centers",
            torch.linspace(depth_min, depth_max, depth_n_bins, dtype=torch.float16),
        )

    @staticmethod
    def _build_upsampling_core(
        n_layers: int, n_hidden_channels: int, n_channels_in: int
    ) -> nn.Sequential:
        layers = []
        for i in range(n_layers):
            if i == 0:
                in_channels = n_channels_in
            else:
                in_channels = n_hidden_channels
            deconv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=n_hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,  # no bias because batchnorm already normalizes it
            )
            batchnorm = nn.BatchNorm2d(n_hidden_channels)
            relu = nn.ReLU(inplace=True)
            layers.extend([deconv, batchnorm, relu])
        return nn.Sequential(*layers)

    @staticmethod
    def _build_heatmap_head(n_channels_in: int, n_keypoints: int) -> nn.Sequential:
        return nn.Conv2d(n_channels_in, n_keypoints, kernel_size=1, bias=True)

    @staticmethod
    def _build_depth_head(
        n_keypoints: int, n_bins: int, n_hidden_channels: int, n_channels_in: int
    ) -> nn.Sequential:
        adaptive_pool = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv2d(n_channels_in, n_hidden_channels, kernel_size=1, bias=False)
        batchnorm = nn.BatchNorm2d(n_hidden_channels)
        relu = nn.ReLU(inplace=True)
        flatten = nn.Flatten()
        fc = nn.Linear(n_hidden_channels, n_keypoints * n_bins)
        return nn.Sequential(adaptive_pool, conv, batchnorm, relu, flatten, fc)

    @staticmethod
    def _softmax_with_temp(
        logits: torch.Tensor, dim: int, temperature: float
    ) -> torch.Tensor:
        return F.softmax(logits / max(1e-6, temperature), dim=dim)

    @staticmethod
    def _soft_argmax_2d(
        heatmaps: torch.Tensor,
        temperature: float = 1.0,
        confidence_method: str = "entropy",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            heatmaps (torch.Tensor): Logits of shape
                (batch_size, n_keypoints, n_rows, n_cols).
            temperature (float): Temperature for softmax.
            confidence_method (str): Method to compute confidence scores in
                soft argmax.

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
            heatmaps.view(batch_size, n_keypoints, -1), dim=-1, temperature=temperature
        )
        probs = probs_flat.view(batch_size, n_keypoints, n_rows, n_cols)

        # Extract the X-Y coordinates from the heatmaps. We do this by computing the
        # expected values of the X and Y coordinates, i.e. summing over all possible
        # coordinates weighted by their probabilities.
        y_grid = torch.linspace(0, n_rows - 1, n_rows, device=heatmaps.device).view(
            1, 1, n_rows, 1
        )
        x_grid = torch.linspace(0, n_cols - 1, n_cols, device=heatmaps.device).view(
            1, 1, 1, n_cols
        )
        # Dimensions 2 and 3 are rows and cols
        x_expected = (probs * x_grid).sum(dim=(2, 3))  # (batch_size, n_keypoints)
        y_expected = (probs * y_grid).sum(dim=(2, 3))  # (batch_size, n_keypoints)
        xy_expected = torch.stack([x_expected, y_expected], dim=-1)

        # Compute the confidence of the prediction (one scalar per keypoint per image)
        if confidence_method == "peak":
            # Option 1: use the max probability as a confidence score
            confidence = probs_flat.max(dim=-1).values  # (batch_size, n_keypoints)
        elif confidence_method == "entropy":
            # Option 2: use the 1 - normalized entropy as a confidence score
            # entropy = $- \sum_i p_i \log p_i$, shape (batch_size, n_keypoints)
            entropy = -(probs_flat * torch.log(probs_flat + 1e-6)).sum(dim=-1)
            # Normalize by the maximum possible entropy (uniform distribution)
            entropy_norm = entropy / torch.log(n_rows * n_cols)
            confidence = 1.0 - entropy_norm  # (batch_size, n_keypoints)
        return xy_expected, confidence

    @staticmethod
    def _soft_argmax_1d(
        logits: torch.Tensor,  # (batch_size, n_keypoints, depth_n_bins)
        bin_values: torch.Tensor,  # (depth_n_bins,)
        temperature: float = 1.0,
        confidence_method: str = "entropy",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits (torch.Tensor): Logits of shape
                (batch_size, n_keypoints, depth_n_bins).
            bin_values (torch.Tensor): Values of each discrete bin. Shape:
                (depth_n_bins,).
            temperature (float): Temperature for softmax.
            confidence_method (str): Method to compute confidence scores in
                soft argmax.
        Returns:
            depth (torch.Tensor): Expected depth for each keypoint. Shape:
                (batch_size, n_keypoints).
            conf (torch.Tensor): Confidence scores for each keypoint.
                Shape: (batch_size, n_keypoints).
        """
        # probs: shape (batch_size, n_keypoints, depth_n_bins), same as logits
        probs = Pose2p5DModel._softmax_with_temp(
            logits, dim=-1, temperature=temperature
        )
        # Compute expected depth (one scalar per keypoint per image)
        depth_expected = (probs * bin_values.view(1, 1, -1)).sum(dim=-1)

        # Compute confidence of prediction (one scalar per keypoint per image)
        if confidence_method == "peak":
            confidence = probs.max(dim=-1).values  # (batch_size, n_keypoints)
        elif confidence_method == "entropy":
            # See same operation in _soft_argmax_2d
            entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=-1)
            entropy_norm = entropy / torch.log(len(bin_values))
            confidence = 1.0 - entropy_norm  # (batch_size, n_keypoints)
        return depth_expected, confidence

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (n_batches, n_channels,
                n_rows, n_cols).

        Returns:
            dict with keys:
                "heatmaps": (torch.Tensor) Predicted heatmaps of shape
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
                "conf_z": (torch.Tensor) Confidence scores for depth
                    predictions of shape (n_batches, n_keypoints).
                "heatmap_stride_h": (float) Stride of heatmap in height
                    dimension relative to input image.
                "heatmap_stride_w": (float) Stride of heatmap in width
                    dimension relative to input image.
        """
        batch_size, _, nrows_in, ncols_in = x.shape

        # Extract features using backbone
        features = self.backbone(x)

        # Run upsampling core (deconv layers)
        up = self.upsampling_core(features)

        # Compute x-y heatmaps
        # Note heatmaps are basically matrices of logits
        heatmaps = self.heatmap_head(up)  # (N, n_keypoints, nrows_out, ncols_out)
        # Decode x-y coordinates from heatmaps using soft-argmax
        # xy_output_coords: shape (N, n_keypoints, 2); xy_conf: shape (N, n_keypoints)
        xy_output_coords, xy_conf = self._soft_argmax_2d(
            heatmaps,
            temperature=self.xy_temperature,
            confidence_method=self.confidence_method,
        )

        # Map to input image pixel coordinates (account for output stride)
        _, _, nrows_out, ncols_out = heatmaps.shape
        stride_rows = nrows_in / nrows_out
        stride_cols = ncols_in / ncols_out
        x = xy_output_coords[..., 0] * stride_cols
        y = xy_output_coords[..., 1] * stride_rows
        xy_input_coords = torch.stack([x, y], dim=-1)  # (N, n_keypoints, 2)

        # Compute depth distributions (logits)
        depth_logits = self.depth_head(up).view(
            batch_size, self.n_keypoints, self.depth_n_bins
        )
        # depth_pos and depth_conf both of shape (N, n_keypoints)
        depth_pos, depth_conf = self._soft_argmax_1d(
            depth_logits,
            bin_values=getattr(self, "depth_bin_centers"),  # pre-registered buffer
            temperature=self.depth_temperature,
            confidence_method=self.confidence_method,
        )

        return {
            "heatmaps": heatmaps,
            "depth_logits": depth_logits,
            "pred_xy": xy_input_coords,
            "pred_depth": depth_pos,
            "conf_xy": xy_conf,
            "conf_z": depth_conf,
            "heatmap_stride_rows": stride_rows,
            "heatmap_stride_cols": stride_cols,
        }


class Pose2p5DLoss(nn.Module):
    def __init__(
        self,
        heatmap_sigma: float = 2.0,
        heatmap_loss: str = "mse",  # "mse" or "kl"
        depth_loss_ce_weight: float = 1.0,
        depth_loss_l1_weight: float = 0.0,
        use_root_relative_z: bool = True,
        root_joint_idx: int = 0,
    ):
        super().__init__()
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_loss = heatmap_loss
        self.depth_loss_ce_weight = depth_loss_ce_weight
        self.depth_loss_l1_weight = depth_loss_l1_weight
        self.use_root_relative_z = use_root_relative_z
        self.root_joint_idx = root_joint_idx

        if heatmap_loss not in ["mse", "kl"]:
            raise ValueError(
                f"Invalid heatmap_loss: {heatmap_loss}. Must be 'mse' or 'kl'."
            )

    @staticmethod
    def _make_gaussian_heatmaps(
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
        heatmaps = heatmaps / torch.max(heatmaps.sum(dim=(-2, -1), keepdim=True), 1e-6)

        return heatmaps

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        xy_labels: torch.Tensor,  # (B,K,2) in input image coords
        depth_labels: torch.Tensor,  # (B,K) in same units as bin values (or normalized)
        bin_values: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        heatmaps = preds["heatmaps"]
        depth_logits = preds["depth_logits"]
        heatmap_stride_rows = preds["heatmap_stride_rows"]
        heatmap_stride_cols = preds["heatmap_stride_cols"]
        batch_size, n_keypoints, n_rows_out, n_cols_out = heatmaps.shape
        device = heatmaps.device

        # Convert xy labels (image coords) -> heatmap coords
        xy_labels_heatmap = xy_labels.clone()
        xy_labels_heatmap[..., 0] = xy_labels[..., 0] / heatmap_stride_cols
        xy_labels_heatmap[..., 1] = xy_labels[..., 1] / heatmap_stride_rows

        # Expand xy labels to heatmap labels
        heatmap_labels = self._make_gaussian_heatmaps(
            xy_labels_heatmap, (n_rows_out, n_cols_out), sigma=self.heatmap_sigma
        )

        # Compute XY heatmap loss
        if self.heatmap_loss == "mse":
            xy_heatmap_loss = F.mse_loss(heatmaps, heatmap_labels)
        elif self.heatmap_loss == "kl":
            # KL(target || pred); compute log-softmax over pixels
            logp = F.log_softmax(
                heatmaps.view(batch_size, n_keypoints, -1), dim=-1
            ).view(batch_size, n_keypoints, n_rows_out, n_cols_out)
            kl = (
                heatmap_labels * (torch.log(heatmap_labels.clamp_min(1e-6)) - logp)
            ).sum(dim=(2, 3))
            xy_heatmap_loss = kl.mean()

        # Depth cross-entropy loss
        if self.depth_loss_ce_weight > 0.0:
            depth_ce_loss = ...  # TODO
        else:
            depth_ce_loss = torch.tensor(0.0, device=device)

        # Depth L1 loss
        if self.depth_loss_l1_weight > 0.0:
            depth_l1_loss = ...  # TODO
        else:
            depth_l1_loss = torch.tensor(0.0, device=device)

        # Compute final loss
        total_loss = xy_heatmap_loss + depth_ce_loss + depth_l1_loss
        return {
            "total_loss": total_loss,
            "xy_heatmap_loss": xy_heatmap_loss,
            "depth_ce_loss": depth_ce_loss,
            "depth_l1_loss": depth_l1_loss,
        }
