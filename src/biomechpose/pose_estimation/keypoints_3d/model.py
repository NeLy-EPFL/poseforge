import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from biomechpose.pose_estimation.feature_extractor import ResNetFeatureExtractor


class Pose2p5DModel(nn.Module):
    """2.5D Pose estimation model (for each keypoint, camera col-row coords
    are predicted via a 2D dense heatmap; depth is predicted separately via
    another 1D heatmap)."""

    def __init__(
        self,
        n_keypoints: int,
        feature_extractor: ResNetFeatureExtractor,
        depth_n_bins: int,
        depth_min: float,
        depth_max: float,
        xy_temperature: float,
        depth_temperature: float,
        upsample_n_layers: int = 3,
        upsample_n_hidden_channels: int = 256,
        depth_n_hidden_channels: int = 256,
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
            upsample_n_layers (int): Number of upsampling layers (deconvs).
            upsample_n_hidden_channels (int): Number of hidden channels in
                upsampling layers.
            depth_n_hidden_channels (int): Number of hidden channels in
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
        self.feature_extractor = feature_extractor
        self.depth_n_bins = depth_n_bins
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.xy_temperature = xy_temperature
        self.depth_temperature = depth_temperature
        self.upsample_n_layers = upsample_n_layers
        self.upsample_n_hidden_channels = upsample_n_hidden_channels
        self.depth_n_hidden_channels = depth_n_hidden_channels
        self.confidence_method = confidence_method
        self.groupnorm_n_groups = groupnorm_n_groups
        self.pose_head_init_std = pose_head_init_std

        # Check input validity
        if confidence_method not in ["entropy", "peak"]:
            raise ValueError(
                f"Invalid confidence_method: {confidence_method}. "
                'Must be "entropy" or "peak".'
            )
        if (
            (upsample_n_hidden_channels % groupnorm_n_groups) != 0
            or (depth_n_hidden_channels % groupnorm_n_groups) != 0
            or groupnorm_n_groups > upsample_n_hidden_channels
            or groupnorm_n_groups > depth_n_hidden_channels
        ):
            raise ValueError(
                "groupnorm_n_groups must be a divisor of "
                "upsample_n_hidden_channels and depth_n_hidden_channels, "
                "and it cannot be greater than either of them."
            )

        # Build upsampling core. This is the first level of processing after the ResNet
        # feature extractor, shared by both the heatmap head and the depth head.
        self.upsampling_core = self._build_upsampling_core()

        # Heatmap head for (x, y) keypoint locations
        self.heatmap_head = self._build_heatmap_head()

        # Depth head for distance from camera
        self.depth_head = self._build_depth_head()
        # Precompute depth bin centers
        self.register_buffer(
            "depth_bin_centers",
            torch.linspace(depth_min, depth_max, depth_n_bins, dtype=torch.float32),
            persistent=False,
        )

    def _build_upsampling_core(self) -> nn.Sequential:
        layers = []
        for i in range(self.upsample_n_layers):
            if i == 0:
                in_channels = self.feature_extractor.output_channels
            else:
                in_channels = self.upsample_n_hidden_channels
            deconv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=self.upsample_n_hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,  # no bias because groupnorm already normalizes it
            )
            groupnorm = nn.GroupNorm(
                self.groupnorm_n_groups, self.upsample_n_hidden_channels
            )
            relu = nn.ReLU(inplace=True)

            # Initialize layers weights using Kaiming normal initialization
            nn.init.kaiming_normal_(deconv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(groupnorm.weight, 1)
            nn.init.constant_(groupnorm.bias, 0)

            layers.extend([deconv, groupnorm, relu])
        return nn.Sequential(*layers)

    def _build_heatmap_head(self) -> nn.Sequential:
        conv = nn.Conv2d(
            self.upsample_n_hidden_channels, self.n_keypoints, kernel_size=1, bias=True
        )

        # Initialize conv layer weights using normal initialization with small std
        # (this is not followed by ReLU, so Kaiming init is not appropriate here)
        nn.init.normal_(conv.weight, std=self.pose_head_init_std)
        nn.init.zeros_(conv.bias)

        return conv

    def _build_depth_head(self) -> nn.Sequential:
        adaptive_pool = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv2d(
            self.upsample_n_hidden_channels,
            self.depth_n_hidden_channels,
            kernel_size=1,
            bias=False,
        )
        groupnorm = nn.GroupNorm(self.groupnorm_n_groups, self.depth_n_hidden_channels)
        relu = nn.ReLU(inplace=True)
        flatten = nn.Flatten()
        fc = nn.Linear(
            self.depth_n_hidden_channels, self.n_keypoints * self.depth_n_bins
        )

        # Initialize layers weights using Kaiming normal initialization
        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(groupnorm.weight, 1)
        nn.init.constant_(groupnorm.bias, 0)
        # For the fc layer, don't use Kaiming init because it's not followed by ReLU
        nn.init.normal_(fc.weight, std=self.pose_head_init_std)
        nn.init.zeros_(fc.bias)

        return nn.Sequential(adaptive_pool, conv, groupnorm, relu, flatten, fc)

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
            x (torch.Tensor): Input tensor of shape (n_batches, n_channels,
                n_rows, n_cols).

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
        """
        batch_size, _, nrows_in, ncols_in = x.shape

        # Run (pretrained) feature extractor (e.g. ResNet-18)
        features = self.feature_extractor(x)

        # Run upsampling core (deconv layers)
        up = self.upsampling_core(features)

        # Compute x-y heatmaps
        # Note heatmaps are basically matrices of logits
        heatmaps = self.heatmap_head(up)  # (N, n_keypoints, nrows_out, ncols_out)
        # Decode x-y coordinates from heatmaps using soft-argmax
        # xy_output_coords: shape (N, n_keypoints, 2); xy_conf: shape (N, n_keypoints)
        xy_output_coords, xy_conf = self._soft_argmax_2d(heatmaps)

        # Map to input image pixel coordinates (account for output stride)
        _, _, nrows_out, ncols_out = heatmaps.shape
        stride_rows = nrows_in / nrows_out
        stride_cols = ncols_in / ncols_out
        x_px_in = xy_output_coords[..., 0] * stride_cols
        y_px_in = xy_output_coords[..., 1] * stride_rows
        xy_input_coords = torch.stack([x_px_in, y_px_in], dim=-1)  # (N, n_keypoints, 2)

        # Compute depth distributions (logits)
        depth_logits = self.depth_head(up).view(
            batch_size, self.n_keypoints, self.depth_n_bins
        )
        # depth_pos and depth_conf both of shape (N, n_keypoints)
        depth_pos, depth_conf = self._soft_argmax_1d(depth_logits)

        return {
            "xy_heatmaps": heatmaps,
            "depth_logits": depth_logits,
            "pred_xy": xy_input_coords,
            "pred_depth": depth_pos,
            "conf_xy": xy_conf,
            "conf_depth": depth_conf,
            "heatmap_stride_rows": stride_rows,
            "heatmap_stride_cols": stride_cols,
        }


class Pose2p5DLoss(nn.Module):
    """Loss function for 2.5D pose estimation model. Combines loss on
    predicted heatmaps (x-y coordinates) and loss on predicted depth."""

    def __init__(
        self,
        heatmap_loss_func: str,
        heatmap_sigma: float = 2.0,
        xy_loss_weight: float = 4.0,
        depth_ce_loss_weight: float = 1.0,
        depth_l1_loss_weight: float = 0.25,
        depth_sigma_bins: float = 1.0,
        clamp_labels: bool = True,
    ):
        """
        Args:
            heatmap_loss_func (str): Loss function to use for heatmaps.
                Options: "mse" or "kl".
            heatmap_sigma (float): Standard deviation of Gaussian used to
                create ground truth heatmaps from x-y labels.
            xy_loss_weight (float): Weight for x-y loss.
            depth_ce_loss_weight (float): Weight for cross-entropy term in
                depth loss.
            depth_l1_loss_weight (float): Weight for L1 term in depth loss.
            depth_sigma_bins (float): Standard deviation of Gaussian used
                to soften one-hot labels in depth cross-entropy loss (in
                number of bins). Higher values make the labels softer.
            clamp_labels (bool): If True, clamp out-of-bounds labels for
                both x-y and depth to the valid range.
        """
        super().__init__()
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_loss_func = heatmap_loss_func
        self.xy_loss_weight = xy_loss_weight
        self.depth_ce_loss_weight = depth_ce_loss_weight
        self.depth_l1_loss_weight = depth_l1_loss_weight
        self.depth_sigma_bins = depth_sigma_bins
        self.clamp_labels = clamp_labels

        # Check input validity
        if heatmap_loss_func not in ["mse", "kl"]:
            raise ValueError(
                f'Invalid heatmap_loss: {heatmap_loss_func}. Must be "mse" or "kl".'
            )

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

    @staticmethod
    def _check_xy_labels(
        xy_labels_heatmap: torch.Tensor,
        heatmap_shape: tuple[int, int],
        clamp_to_range: bool = False,
    ) -> tuple[bool, torch.Tensor]:
        """Check if x-y labels are within the heatmap dimensions. Log a
        warning if any label is out of bounds.

        Args:
            xy_labels_heatmap (torch.Tensor): X-Y coordinates of shape
                (batch_size, n_keypoints, 2) in heatmap pixel space (NOT
                input image pixel space!).
            heatmap_shape (tuple[int, int]): Heatmap dimensions
                (n_rows, n_cols).
            clamp_to_range (bool): If True, clamp out-of-bounds labels to
                the valid range and return the clamped version of
                xy_labels_heatmap. If False, the original labels are
                returned.

        Returns:
            bool: True if any label is out of bounds, False otherwise.
            torch.Tensor: Clamped version of xy_labels_heatmap if
                clamp_to_range is True, original xy_labels_heatmap
                otherwise.
        """
        n_rows, n_cols = heatmap_shape
        max_xy = torch.tensor([n_cols - 1, n_rows - 1], device=xy_labels_heatmap.device)
        is_bad = (xy_labels_heatmap < 0).any() or (xy_labels_heatmap > max_xy).any()
        if is_bad:
            logging.warning(
                "Some x-y labels are outside the heatmap dimensions. "
                "This may cause bad values in the loss."
            )

        if clamp_to_range:
            xy_labels_heatmap = torch.clamp(xy_labels_heatmap, min=0, max=max_xy)

        return is_bad, xy_labels_heatmap

    @staticmethod
    def _check_depth_labels(
        depth_labels: torch.Tensor,
        bin_values: torch.Tensor,
        clamp_to_range: bool = False,
    ) -> tuple[bool, torch.Tensor]:
        """Check if depth labels are within the range of depth bins. Log a
        warning if any label is out of bounds.

        Args:
            depth_labels (torch.Tensor): Ground truth depth values of shape
                (batch_size, n_keypoints).
            bin_values (torch.Tensor): Center values (i.e. depths) of each
                discrete bin. Shape: (depth_n_bins,).
            clamp_to_range (bool): If True, clamp out-of-bounds labels to
                the valid range and return the clamped version of
                depth_labels. If False, the clamping step is skipped and
                the original labels are returned.

        Returns:
            bool: True if any label is out of bounds, False otherwise.
            torch.Tensor: Clamped version of depth_labels if clamp_to_range
                is True, original depth_labels otherwise.
        """
        is_bad = (depth_labels < bin_values[0]).any() or (
            depth_labels > bin_values[-1]
        ).any()
        if is_bad:
            logging.warning(
                "Some depth labels are outside the range of depth bins. "
                "This may cause bad values in the loss."
            )

        if clamp_to_range:
            dmin, dmax = bin_values[0], bin_values[-1]
            depth_labels = torch.clamp(depth_labels, min=dmin, max=dmax)

        return is_bad, depth_labels

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
        heatmap_stride_rows = pred_dict["heatmap_stride_rows"]
        heatmap_stride_cols = pred_dict["heatmap_stride_cols"]
        batch_size, n_keypoints, n_rows_out, n_cols_out = xy_heatmaps.shape

        # Convert xy labels (image coords) -> heatmap coords
        xy_labels_heatmap = xy_labels.clone()
        xy_labels_heatmap[..., 0] = xy_labels[..., 0] / heatmap_stride_cols
        xy_labels_heatmap[..., 1] = xy_labels[..., 1] / heatmap_stride_rows
        _, xy_labels_heatmap = self._check_xy_labels(
            xy_labels_heatmap,
            (n_rows_out, n_cols_out),
            clamp_to_range=self.clamp_labels,
        )

        # Expand xy labels to heatmap labels
        heatmap_labels = self._expand_xy_labels_to_gaussian_heatmaps(
            xy_labels_heatmap, (n_rows_out, n_cols_out), sigma=self.heatmap_sigma
        )

        # Compute x-y prediction loss on heatmaps
        xy_heatmap_loss = self._compute_xy_heatmap_loss(
            self.heatmap_loss_func, xy_heatmaps, heatmap_labels
        )

        # Compute depth losses
        _, depth_labels = self._check_depth_labels(
            depth_labels, bin_values, clamp_to_range=self.clamp_labels
        )
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
