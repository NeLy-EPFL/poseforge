"""
Utility functions for CycleGAN fruit fly image translation.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import shutil


def initialize_weights(
    net: nn.Module, init_type: str = "normal", init_gain: float = 0.02
):
    """Initialize network weights."""

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f"Initialization method {init_type} is not implemented"
                )

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class ImageBuffer:
    """Buffer to store previously generated images for discriminator training.

    This helps stabilize GAN training by showing the discriminator a history
    of generated images rather than only the most recent ones.
    """

    def __init__(self, buffer_size: int = 50):
        self.buffer_size = buffer_size
        self.buffer = []

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Return images from buffer, potentially mixing with new images."""
        if self.buffer_size == 0:
            return images

        return_images = []
        for image in images:
            image = image.unsqueeze(0)  # Add batch dimension

            if len(self.buffer) < self.buffer_size:
                # Buffer not full, just add the image
                self.buffer.append(image)
                return_images.append(image)
            else:
                # 50% chance to return buffered image, 50% chance to return new image
                if torch.rand(1).item() > 0.5:
                    # Return random buffered image and replace it with new image
                    random_idx = torch.randint(0, self.buffer_size, (1,)).item()
                    temp = self.buffer[random_idx].clone()
                    self.buffer[random_idx] = image
                    return_images.append(temp)
                else:
                    # Return new image
                    return_images.append(image)

        return torch.cat(return_images, dim=0)


def _to_jsonable(obj):
    """Recursively convert Path objects to str for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_to_jsonable(v) for v in obj)
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    losses: Dict[str, float],
    hyperparameters: Dict,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dicts": {
            name: opt.state_dict() for name, opt in optimizers.items()
        },
        "scheduler_state_dicts": {
            name: sch.state_dict() for name, sch in schedulers.items()
        },
        "losses": losses,
        "hyperparameters": hyperparameters,
    }

    # Create directory if it doesn't exist
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

    # Save hyperparameters as JSON for easy reading
    hyperparams_path = (
        checkpoint_path.parent / f"{checkpoint_path.stem}_hyperparams.json"
    )
    with open(hyperparams_path, "w") as f:
        json.dump(_to_jsonable(hyperparameters), f, indent=2)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    schedulers: Optional[Dict[str, torch.optim.lr_scheduler._LRScheduler]] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, Dict[str, float]]:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer states
    if optimizers is not None:
        for name, optimizer in optimizers.items():
            if name in checkpoint["optimizer_state_dicts"]:
                optimizer.load_state_dict(checkpoint["optimizer_state_dicts"][name])

    # Load scheduler states
    if schedulers is not None:
        for name, scheduler in schedulers.items():
            if name in checkpoint["scheduler_state_dicts"]:
                scheduler.load_state_dict(checkpoint["scheduler_state_dicts"][name])

    epoch = checkpoint["epoch"]
    losses = checkpoint.get("losses", {})

    return epoch, losses


def save_best_model(model_path: Path, model: nn.Module, hyperparameters: Dict):
    """Save the best model (inference only)."""
    save_dict = {
        "model_state_dict": model.state_dict(),
        "hyperparameters": hyperparameters,
    }

    # Create directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(save_dict, model_path)


def load_model_for_inference(
    model_path: Path, model: nn.Module, device: torch.device = torch.device("cpu")
) -> Dict:
    """Load model for inference."""
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return checkpoint.get("hyperparameters", {})


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_directories(output_dir: Path) -> Dict[str, Path]:
    """Set up output directories for training."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dirs = {
        "checkpoints": output_dir / "checkpoints",
        "visualizations": output_dir / "visualizations",
        "logs": output_dir / "logs",
        "models": output_dir / "models",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True, parents=True)

    return dirs


def cleanup_old_checkpoints(
    checkpoint_dir: Path, keep_last_n: int = 5, keep_best: bool = True
):
    """Clean up old checkpoints, keeping only the most recent ones."""
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))

    if len(checkpoint_files) <= keep_last_n:
        return

    # Sort by modification time
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime)

    # Remove old checkpoints
    files_to_remove = checkpoint_files[:-keep_last_n]

    for file_path in files_to_remove:
        # Don't remove if it's marked as best
        if keep_best and "best" in file_path.name:
            continue

        file_path.unlink()

        # Also remove corresponding hyperparameters file
        hyperparams_file = checkpoint_dir / f"{file_path.stem}_hyperparams.json"
        if hyperparams_file.exists():
            hyperparams_file.unlink()


def compute_gradient_penalty(
    discriminator: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute gradient penalty for WGAN-GP (optional enhancement)."""
    batch_size = real_samples.size(0)

    # Random weight for interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)

    # Interpolated samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)

    # Get discriminator output
    d_interpolated = discriminator(interpolated)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
