"""
Dataset implementation for CycleGAN fruit fly image translation.
Handles RGB (simulated) and grayscale (experimental) images.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class FlyImageDataset(Dataset):
    """Dataset for unpaired fruit fly images."""

    def __init__(
        self,
        sim_image_paths: List[Path],
        exp_image_paths: List[Path],
        image_size: int = 900,
        # Data augmentation hyperparameters
        random_crop_size: Optional[int] = None,  # If None, no random crop
        horizontal_flip_prob: float = 0.5,
        brightness_jitter: float = 0.1,
        contrast_jitter: float = 0.1,
        # Normalization (set to None to use [-1, 1] normalization)
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
        # Rotate simulated dataset clockwise by 90 degrees
        rotate_simulated_90deg_clockwise: bool = True,
        # Final resize after all transformations
        final_resize: int = 512,  # NEW
    ):
        self.rotate_simulated_90deg_clockwise = rotate_simulated_90deg_clockwise
        self.sim_image_paths = list(sim_image_paths)
        self.exp_image_paths = list(exp_image_paths)
        self.image_size = image_size
        self.final_resize = final_resize

        # Create transforms for simulated images (RGB)
        sim_transforms = []

        # Resize to target size
        sim_transforms.append(transforms.Resize((image_size, image_size)))

        # Random crop if specified
        if random_crop_size is not None:
            sim_transforms.extend(
                [
                    transforms.Resize(
                        int(image_size * 1.12)
                    ),  # Slightly larger for cropping
                    transforms.RandomCrop(random_crop_size),
                    transforms.Resize((image_size, image_size)),
                ]
            )

        # Data augmentation
        if horizontal_flip_prob > 0:
            sim_transforms.append(transforms.RandomHorizontalFlip(horizontal_flip_prob))

        if brightness_jitter > 0 or contrast_jitter > 0:
            sim_transforms.append(
                transforms.ColorJitter(
                    brightness=brightness_jitter, contrast=contrast_jitter
                )
            )

        # Convert to tensor and normalize
        sim_transforms.append(transforms.ToTensor())

        if normalize_mean is not None and normalize_std is not None:
            sim_transforms.append(transforms.Normalize(normalize_mean, normalize_std))
        else:
            # Normalize to [-1, 1] (standard for GANs)
            sim_transforms.append(
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            )

        # Final resize
        sim_transforms.append(transforms.Resize((final_resize, final_resize)))

        self.sim_transform = transforms.Compose(sim_transforms)

        # Create transforms for experimental images (Grayscale)
        exp_transforms = []

        # Resize to target size
        exp_transforms.append(transforms.Resize((image_size, image_size)))

        # Random crop if specified
        if random_crop_size is not None:
            exp_transforms.extend(
                [
                    transforms.Resize(int(image_size * 1.12)),
                    transforms.RandomCrop(random_crop_size),
                    transforms.Resize((image_size, image_size)),
                ]
            )

        # Data augmentation
        if horizontal_flip_prob > 0:
            exp_transforms.append(transforms.RandomHorizontalFlip(horizontal_flip_prob))

        if brightness_jitter > 0 or contrast_jitter > 0:
            exp_transforms.append(
                transforms.ColorJitter(
                    brightness=brightness_jitter, contrast=contrast_jitter
                )
            )

        # Convert to tensor and normalize
        exp_transforms.append(transforms.ToTensor())

        if normalize_mean is not None and normalize_std is not None:
            # For grayscale, use only first channel of normalization
            exp_transforms.append(
                transforms.Normalize([normalize_mean[0]], [normalize_std[0]])
            )
        else:
            # Normalize to [-1, 1]
            exp_transforms.append(transforms.Normalize([0.5], [0.5]))

        # Final resize
        exp_transforms.append(transforms.Resize((final_resize, final_resize)))

        self.exp_transform = transforms.Compose(exp_transforms)

    def __len__(self) -> int:
        # Return length of the larger dataset
        return max(len(self.sim_image_paths), len(self.exp_image_paths))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use modulo to cycle through shorter dataset
        sim_idx = idx % len(self.sim_image_paths)
        exp_idx = idx % len(self.exp_image_paths)

        # Load simulated image (RGB)
        sim_path = self.sim_image_paths[sim_idx]
        sim_image = Image.open(sim_path).convert("RGB")
        if self.rotate_simulated_90deg_clockwise:
            sim_image = sim_image.rotate(-90, expand=True)
        sim_tensor = self.sim_transform(sim_image)

        # Load experimental image (convert to grayscale if needed)
        exp_path = self.exp_image_paths[exp_idx]
        exp_image = Image.open(exp_path).convert("L")  # Convert to grayscale
        exp_tensor = self.exp_transform(exp_image)

        return sim_tensor, exp_tensor


def create_dataloaders(
    sim_image_paths: List[Path],
    exp_image_paths: List[Path],
    train_split: float = 0.8,
    batch_size: int = 1,
    num_workers: int = 4,
    # Dataset hyperparameters
    image_size: int = 900,
    random_crop_size: Optional[int] = None,
    horizontal_flip_prob: float = 0.5,
    brightness_jitter: float = 0.1,
    contrast_jitter: float = 0.1,
    # Random seed for reproducible splits
    random_seed: int = 42,
    final_resize: int = 512,  # NEW
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders."""

    # Set random seed for reproducible splits
    random.seed(random_seed)

    # Split data
    sim_shuffled = sim_image_paths.copy()
    exp_shuffled = exp_image_paths.copy()
    random.shuffle(sim_shuffled)
    random.shuffle(exp_shuffled)

    sim_train_size = int(len(sim_shuffled) * train_split)
    exp_train_size = int(len(exp_shuffled) * train_split)

    sim_train_paths = sim_shuffled[:sim_train_size]
    sim_val_paths = sim_shuffled[sim_train_size:]

    exp_train_paths = exp_shuffled[:exp_train_size]
    exp_val_paths = exp_shuffled[exp_train_size:]

    # Create datasets
    train_dataset = FlyImageDataset(
        sim_train_paths,
        exp_train_paths,
        image_size=image_size,
        random_crop_size=random_crop_size,
        horizontal_flip_prob=horizontal_flip_prob,
        brightness_jitter=brightness_jitter,
        contrast_jitter=contrast_jitter,
        final_resize=final_resize,  # Pass through
    )

    # Validation dataset without augmentation
    val_dataset = FlyImageDataset(
        sim_val_paths,
        exp_val_paths,
        image_size=image_size,
        random_crop_size=None,  # No random crop for validation
        horizontal_flip_prob=0.0,  # No flip for validation
        brightness_jitter=0.0,  # No jitter for validation
        contrast_jitter=0.0,
        final_resize=final_resize,  # Pass through
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Important for stable GAN training
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def denormalize_tensor(
    tensor: torch.Tensor, mean: List[float] = [0.5], std: List[float] = [0.5]
) -> torch.Tensor:
    """Denormalize tensor from [-1, 1] to [0, 1] for visualization. Handles (C,H,W) and (B,C,H,W)."""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)  # Add batch dim
        squeeze = True
    else:
        squeeze = False
    c = tensor.shape[1]
    if len(mean) == 1 and c == 3:
        mean = mean * 3
        std = std * 3
    elif len(mean) == 3 and c == 1:
        mean = mean[:1]
        std = std[:1]
    mean = torch.tensor(mean, device=tensor.device).view(1, c, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, c, 1, 1)
    out = torch.clamp(tensor * std + mean, 0, 1)
    if squeeze:
        out = out.squeeze(0)
    return out
