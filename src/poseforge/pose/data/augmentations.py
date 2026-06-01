"""
Training-time augmentations for sim-to-real transfer in pose estimation.

Two independent augmentation modules:

1. **RandomScaleCrop** ("scale jittering" / "random resized crop")
   Simulates the animal appearing at different distances from the camera
   by cropping a random sub-region and resizing it back to the original
   input dimensions. Forces the model to become invariant to the scale
   at which anatomical structures appear in pixels — counteracting
   overfitting to the fixed object scale present in synthetic data.
   Geometry-aware: jointly transforms xy keypoint labels.

2. **DomainRandomization** ("domain randomization" / "photometric jitter")
   Destroys synthetic-specific high-frequency cues (render aliasing,
   style-transfer texture fingerprints, unrealistic noise spectrum) by
   injecting realistic low-level perturbations: blur, downsample-then-
   upsample, JPEG compression artefacts, sensor-style noise, and
   contrast/gamma shifts. Label-agnostic: only modifies pixel values.

Both operate on GPU tensors of shape (N, C, H, W) with values in [0, 1]
and are designed to be inserted in the training loop between data loading
and the forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import math


class RandomScaleCrop(nn.Module):
    """Scale jittering via random resized crop.

    Crops a random sub-region of the image (simulating zoom-in / zoom-out)
    and resizes it back to the original spatial dimensions. Keypoint labels
    are transformed accordingly.

    At higher scale factors the animal fills more of the frame (zoom-in);
    at lower scale factors, more background is visible (zoom-out, padded).

    The input resolution seen by the network stays constant (matching
    the pretrained encoder), but the *content scale* varies — which is
    exactly the augmentation needed when the model overfits to a fixed
    object-to-image-size ratio in synthetic data.
    """

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.7, 1.3),
        p: float = 0.5,
    ):
        """
        Args:
            scale_range: (min_scale, max_scale) for the crop size relative
                to the full image. scale=1.0 means the crop covers the
                entire image (identity). scale<1.0 means zoom-in (crop is
                smaller → content appears larger after resize). scale>1.0
                means zoom-out (crop is larger than image → padded, content
                appears smaller after resize).
            p: Probability of applying the augmentation to each sample.
        """
        super().__init__()
        self.scale_min, self.scale_max = scale_range
        self.p = p

        if self.scale_min <= 0 or self.scale_max <= 0:
            raise ValueError("scale_range values must be positive")
        if self.scale_min > self.scale_max:
            raise ValueError("scale_range[0] must be <= scale_range[1]")

    def forward(
        self,
        frames: torch.Tensor,
        xy_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: (N, C, H, W), float in [0, 1].
            xy_labels: (N, n_keypoints, 2), keypoint positions in image
                pixel coordinates (x, y).

        Returns:
            augmented_frames: (N, C, H, W), same size as input.
            augmented_xy_labels: (N, n_keypoints, 2), transformed labels.
        """
        if not self.training:
            return frames, xy_labels

        N, C, H, W = frames.shape
        device = frames.device

        # Decide which samples get augmented
        mask = torch.rand(N, device=device) < self.p  # (N,)
        if not mask.any():
            return frames, xy_labels

        # Sample scales on a log-uniform distribution (symmetric around 1.0
        # in log space — zoom-in and zoom-out are equally likely)
        log_scale = torch.empty(N, device=device).uniform_(
            math.log(self.scale_min), math.log(self.scale_max)
        )
        scales = log_scale.exp()  # (N,)

        # Crop dimensions (before resize back to H×W)
        crop_h = (H * scales).clamp(min=1).int()
        crop_w = (W * scales).clamp(min=1).int()

        augmented_frames = frames.clone()
        augmented_xy = xy_labels.clone()

        for i in range(N):
            if not mask[i]:
                continue

            s = scales[i].item()
            ch, cw = crop_h[i].item(), crop_w[i].item()

            if s <= 1.0:
                # Zoom-in: crop a smaller region, then resize up to H×W
                # Random top-left corner within valid range
                top = torch.randint(0, max(1, H - ch + 1), (1,), device=device).item()
                left = torch.randint(0, max(1, W - cw + 1), (1,), device=device).item()

                crop = frames[i : i + 1, :, top : top + ch, left : left + cw]
                augmented_frames[i] = F.interpolate(
                    crop, size=(H, W), mode="bilinear", align_corners=False
                ).squeeze(0)

                # Transform labels: shift then scale
                augmented_xy[i, :, 0] = (xy_labels[i, :, 0] - left) * (W / cw)
                augmented_xy[i, :, 1] = (xy_labels[i, :, 1] - top) * (H / ch)

            else:
                # Zoom-out: the "crop" is larger than the image — pad first,
                # then take the ch×cw region, then resize to H×W
                pad_h = ch - H
                pad_w = cw - W
                top_pad = torch.randint(0, max(1, pad_h + 1), (1,), device=device).item()
                left_pad = torch.randint(0, max(1, pad_w + 1), (1,), device=device).item()

                padded = F.pad(
                    frames[i : i + 1],
                    (left_pad, pad_w - left_pad, top_pad, pad_h - top_pad),
                    mode="constant",
                    value=0,
                )
                # padded is now (1, C, ch, cw) — resize to H×W
                augmented_frames[i] = F.interpolate(
                    padded, size=(H, W), mode="bilinear", align_corners=False
                ).squeeze(0)

                # Transform labels: shift (by padding offset) then scale
                augmented_xy[i, :, 0] = (xy_labels[i, :, 0] + left_pad) * (W / cw)
                augmented_xy[i, :, 1] = (xy_labels[i, :, 1] + top_pad) * (H / ch)

        return augmented_frames, augmented_xy

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"scale_range=({self.scale_min}, {self.scale_max}), p={self.p})"
        )


class DomainRandomization(nn.Module):
    """Photometric domain randomization for sim-to-real transfer.

    Applies a stochastic pipeline of low-level image perturbations that
    destroy synthetic-specific visual cues without altering pose geometry.
    Each perturbation is applied independently with its own probability.

    Perturbations (in order of application):
        1. **Gaussian blur** — softens render-specific edge sharpness.
        2. **Downsample-then-upsample** — simulates resolution mismatch
           between synthetic renders and real camera captures.
        3. **Additive Gaussian noise** — simulates camera sensor noise.
        4. **Contrast and brightness jitter** — simulates varying exposure
           and lighting conditions.
        5. **Gamma correction** — simulates non-linear camera response.

    All operations are differentiable and run on GPU. Labels are not
    modified (photometric augmentations do not affect keypoint positions).
    """

    def __init__(
        self,
        blur_sigma_range: tuple[float, float] = (0.1, 2.0),
        blur_p: float = 0.3,
        downsample_factor_range: tuple[float, float] = (0.25, 0.8),
        downsample_p: float = 0.3,
        noise_std_range: tuple[float, float] = (0.0, 0.05),
        noise_p: float = 0.4,
        contrast_range: tuple[float, float] = (0.6, 1.4),
        brightness_range: tuple[float, float] = (-0.1, 0.1),
        contrast_p: float = 0.4,
        gamma_range: tuple[float, float] = (0.7, 1.5),
        gamma_p: float = 0.3,
    ):
        """
        Args:
            blur_sigma_range: Range for Gaussian blur sigma.
            blur_p: Probability of applying blur per sample.
            downsample_factor_range: Range for the downsampling factor
                (fraction of original resolution to downsample to before
                upsampling back). Lower = more aggressive degradation.
            downsample_p: Probability of applying downsample-upsample.
            noise_std_range: Range for additive Gaussian noise std dev
                (relative to [0, 1] pixel range).
            noise_p: Probability of applying sensor noise.
            contrast_range: Multiplicative contrast factor range.
            brightness_range: Additive brightness shift range.
            contrast_p: Probability of applying contrast/brightness jitter.
            gamma_range: Range for gamma correction exponent.
            gamma_p: Probability of applying gamma correction.
        """
        super().__init__()
        self.blur_sigma_range = blur_sigma_range
        self.blur_p = blur_p
        self.downsample_factor_range = downsample_factor_range
        self.downsample_p = downsample_p
        self.noise_std_range = noise_std_range
        self.noise_p = noise_p
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.contrast_p = contrast_p
        self.gamma_range = gamma_range
        self.gamma_p = gamma_p

    @staticmethod
    def _gaussian_blur(images: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Apply per-sample Gaussian blur with varying sigma.

        Args:
            images: (N, C, H, W)
            sigma: (N,) per-sample sigma values. Samples with sigma <= 0
                are left unblurred.

        Returns:
            Blurred images: (N, C, H, W)
        """
        # Use a fixed kernel size based on max sigma in batch (for efficiency)
        max_sigma = sigma.max().item()
        if max_sigma <= 0:
            return images

        # Kernel radius: 3*sigma rounded up to ensure coverage, minimum 1
        radius = max(1, int(math.ceil(3.0 * max_sigma)))
        kernel_size = 2 * radius + 1

        # Build per-sample 1D Gaussian kernels
        device = images.device
        N = images.shape[0]
        x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        x = x.unsqueeze(0).expand(N, -1)  # (N, kernel_size)

        # Gaussian: exp(-x^2 / (2*sigma^2)), with sigma (N, 1)
        sigma_safe = sigma.clamp(min=1e-6).unsqueeze(1)  # (N, 1)
        kernels_1d = torch.exp(-x**2 / (2 * sigma_safe**2))  # (N, kernel_size)
        kernels_1d = kernels_1d / kernels_1d.sum(dim=1, keepdim=True)  # normalize

        # Separable 2D blur: horizontal then vertical
        # Use grouped conv where each group is one channel of one sample.
        # Input must be (1, N*C, H, W) so N*C is the channel dim for groups.
        C = images.shape[1]
        G = N * C
        imgs = images.reshape(1, G, images.shape[2], images.shape[3])

        # Horizontal kernel: (N*C, 1, 1, kernel_size)
        kh = kernels_1d.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, ks)
        kh = kh.repeat(1, C, 1, 1).reshape(G, 1, 1, kernel_size)

        # Vertical kernel: (N*C, 1, kernel_size, 1)
        kv = kernels_1d.unsqueeze(1).unsqueeze(3)  # (N, 1, ks, 1)
        kv = kv.repeat(1, C, 1, 1).reshape(G, 1, kernel_size, 1)

        # Apply as depthwise conv (groups = N*C)
        padded = F.pad(imgs, (radius, radius, 0, 0), mode="reflect")
        blurred = F.conv2d(padded, kh.flip(-1), groups=G)
        padded = F.pad(blurred, (0, 0, radius, radius), mode="reflect")
        blurred = F.conv2d(padded, kv.flip(-2), groups=G)

        return blurred.reshape(images.shape)

    @staticmethod
    def _downsample_upsample(
        images: torch.Tensor, factors: torch.Tensor
    ) -> torch.Tensor:
        """Apply per-sample downsample-then-upsample to simulate resolution
        mismatch.

        Args:
            images: (N, C, H, W)
            factors: (N,) per-sample downsampling factors in (0, 1].
                factor=1.0 means no change; factor=0.5 halves the
                resolution before upsampling back.

        Returns:
            Degraded images: (N, C, H, W)
        """
        N, C, H, W = images.shape
        result = images.clone()
        # Group samples by similar factor to batch the resizes
        for i in range(N):
            f = factors[i].item()
            if f >= 0.99:
                continue
            h_small = max(1, int(H * f))
            w_small = max(1, int(W * f))
            small = F.interpolate(
                images[i : i + 1], size=(h_small, w_small), mode="bilinear",
                align_corners=False,
            )
            result[i] = F.interpolate(
                small, size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze(0)
        return result

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (N, C, H, W), float in [0, 1].

        Returns:
            Augmented frames: (N, C, H, W), clamped to [0, 1].
        """
        if not self.training:
            return frames

        N, C, H, W = frames.shape
        device = frames.device
        x = frames

        # 1. Gaussian blur
        if self.blur_p > 0:
            mask = torch.rand(N, device=device) < self.blur_p
            sigma = torch.zeros(N, device=device)
            sigma[mask] = torch.empty(mask.sum(), device=device).uniform_(
                *self.blur_sigma_range
            )
            if mask.any():
                blurred = self._gaussian_blur(x, sigma)
                # Only replace augmented samples
                x = torch.where(mask.view(N, 1, 1, 1), blurred, x)

        # 2. Downsample-then-upsample
        if self.downsample_p > 0:
            mask = torch.rand(N, device=device) < self.downsample_p
            if mask.any():
                factors = torch.ones(N, device=device)
                factors[mask] = torch.empty(mask.sum(), device=device).uniform_(
                    *self.downsample_factor_range
                )
                degraded = self._downsample_upsample(x, factors)
                x = torch.where(mask.view(N, 1, 1, 1), degraded, x)

        # 3. Additive Gaussian noise (sensor noise)
        if self.noise_p > 0:
            mask = torch.rand(N, device=device) < self.noise_p
            if mask.any():
                noise_std = torch.zeros(N, device=device)
                noise_std[mask] = torch.empty(mask.sum(), device=device).uniform_(
                    *self.noise_std_range
                )
                noise = torch.randn_like(x) * noise_std.view(N, 1, 1, 1)
                x = x + noise

        # 4. Contrast and brightness jitter
        if self.contrast_p > 0:
            mask = torch.rand(N, device=device) < self.contrast_p
            if mask.any():
                contrast = torch.ones(N, device=device)
                brightness = torch.zeros(N, device=device)
                n_aug = mask.sum()
                contrast[mask] = torch.empty(n_aug, device=device).uniform_(
                    *self.contrast_range
                )
                brightness[mask] = torch.empty(n_aug, device=device).uniform_(
                    *self.brightness_range
                )
                # Apply: pixel = contrast * (pixel - mean) + mean + brightness
                mean = x.mean(dim=(2, 3), keepdim=True)  # per-sample, per-channel
                x_jittered = contrast.view(N, 1, 1, 1) * (x - mean) + mean + brightness.view(N, 1, 1, 1)
                x = torch.where(mask.view(N, 1, 1, 1), x_jittered, x)

        # 5. Gamma correction
        if self.gamma_p > 0:
            mask = torch.rand(N, device=device) < self.gamma_p
            if mask.any():
                gamma = torch.ones(N, device=device)
                gamma[mask] = torch.empty(mask.sum(), device=device).uniform_(
                    *self.gamma_range
                )
                # Gamma: pixel = pixel^gamma (requires pixel in [0, 1])
                x_clamped = x.clamp(0, 1)
                x_gamma = x_clamped.pow(gamma.view(N, 1, 1, 1))
                x = torch.where(mask.view(N, 1, 1, 1), x_gamma, x)

        return x.clamp(0, 1)

    def __repr__(self):
        parts = []
        if self.blur_p > 0:
            parts.append(f"blur(σ={self.blur_sigma_range}, p={self.blur_p})")
        if self.downsample_p > 0:
            parts.append(f"downsample(f={self.downsample_factor_range}, p={self.downsample_p})")
        if self.noise_p > 0:
            parts.append(f"noise(σ={self.noise_std_range}, p={self.noise_p})")
        if self.contrast_p > 0:
            parts.append(f"contrast({self.contrast_range}, p={self.contrast_p})")
        if self.gamma_p > 0:
            parts.append(f"gamma({self.gamma_range}, p={self.gamma_p})")
        return f"{self.__class__.__name__}({', '.join(parts)})"
