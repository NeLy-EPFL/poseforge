"""
Visualization utilities for CycleGAN fruit fly image translation.
"""

import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import wandb

from biomechpose.style_transfer.dataset import denormalize_tensor


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for matplotlib."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension

    # Denormalize from [-1, 1] to [0, 1]
    tensor = denormalize_tensor(tensor.clone())

    if tensor.shape[0] == 1:  # Grayscale
        return tensor.squeeze(0).cpu().numpy()
    elif tensor.shape[0] == 3:  # RGB
        return tensor.permute(1, 2, 0).cpu().numpy()
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")


def save_image_grid(
    images: List[torch.Tensor],
    titles: List[str],
    save_path: Path,
    nrow: int = None,
    figsize: Tuple[int, int] = (15, 10),
):
    """Save a grid of images with titles."""
    if nrow is None:
        nrow = len(images)

    nrows = (len(images) + nrow - 1) // nrow

    fig, axes = plt.subplots(nrows, nrow, figsize=figsize)
    if nrows == 1:
        axes = [axes] if nrow == 1 else axes
    else:
        axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        img_np = tensor_to_numpy(img)

        if len(img_np.shape) == 2:  # Grayscale
            axes[i].imshow(img_np, cmap="gray")
        else:  # RGB
            axes[i].imshow(img_np)

        axes[i].set_title(title)
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_training_batch(
    model_outputs: Dict[str, torch.Tensor],
    sim_real: torch.Tensor,
    exp_real: torch.Tensor,
    save_path: Path,
    max_images: int = 4,
):
    """Visualize a training batch with real and generated images."""
    batch_size = min(sim_real.shape[0], max_images)

    images = []
    titles = []

    for i in range(batch_size):
        # Original images
        images.append(sim_real[i])
        titles.append(f"Real Sim {i+1}")

        images.append(exp_real[i])
        titles.append(f"Real Exp {i+1}")

        # Generated images
        if "fake_exp" in model_outputs:
            images.append(model_outputs["fake_exp"][i])
            titles.append(f"Fake Exp {i+1}")

        if "fake_sim" in model_outputs:
            images.append(model_outputs["fake_sim"][i])
            titles.append(f"Fake Sim {i+1}")

        # Reconstructed images
        if "reconstructed_sim" in model_outputs:
            images.append(model_outputs["reconstructed_sim"][i])
            titles.append(f"Recon Sim {i+1}")

        if "reconstructed_exp" in model_outputs:
            images.append(model_outputs["reconstructed_exp"][i])
            titles.append(f"Recon Exp {i+1}")

    save_image_grid(images, titles, save_path, nrow=6)


def visualize_cycle_consistency(
    sim_real: torch.Tensor,
    fake_exp: torch.Tensor,
    reconstructed_sim: torch.Tensor,
    exp_real: torch.Tensor,
    fake_sim: torch.Tensor,
    reconstructed_exp: torch.Tensor,
    save_path: Path,
    max_images: int = 4,
):
    """Visualize cycle consistency for a few samples."""
    batch_size = min(sim_real.shape[0], max_images)

    images = []
    titles = []

    for i in range(batch_size):
        # Sim -> Exp -> Sim cycle
        images.extend([sim_real[i], fake_exp[i], reconstructed_sim[i]])
        titles.extend([f"Sim Real {i+1}", f"→ Exp Fake {i+1}", f"→ Sim Recon {i+1}"])

        # Exp -> Sim -> Exp cycle
        images.extend([exp_real[i], fake_sim[i], reconstructed_exp[i]])
        titles.extend([f"Exp Real {i+1}", f"→ Sim Fake {i+1}", f"→ Exp Recon {i+1}"])

    save_image_grid(images, titles, save_path, nrow=6)


def log_images_to_tensorboard(
    writer,
    model_outputs: Dict[str, torch.Tensor],
    sim_real: torch.Tensor,
    exp_real: torch.Tensor,
    step: int,
    max_images: int = 4,
):
    """Log images to TensorBoard."""
    batch_size = min(sim_real.shape[0], max_images)

    # Denormalize images for logging
    sim_real_norm = denormalize_tensor(sim_real[:batch_size].clone())
    exp_real_norm = denormalize_tensor(exp_real[:batch_size].clone())

    # Log real images
    writer.add_images("Real/Simulated", sim_real_norm, step)
    writer.add_images("Real/Experimental", exp_real_norm, step)

    # Log generated images
    if "fake_exp" in model_outputs:
        fake_exp_norm = denormalize_tensor(
            model_outputs["fake_exp"][:batch_size].clone()
        )
        writer.add_images("Generated/Sim_to_Exp", fake_exp_norm, step)

    if "fake_sim" in model_outputs:
        fake_sim_norm = denormalize_tensor(
            model_outputs["fake_sim"][:batch_size].clone()
        )
        writer.add_images("Generated/Exp_to_Sim", fake_sim_norm, step)

    # Log reconstructed images
    if "reconstructed_sim" in model_outputs:
        recon_sim_norm = denormalize_tensor(
            model_outputs["reconstructed_sim"][:batch_size].clone()
        )
        writer.add_images("Reconstructed/Simulated", recon_sim_norm, step)

    if "reconstructed_exp" in model_outputs:
        recon_exp_norm = denormalize_tensor(
            model_outputs["reconstructed_exp"][:batch_size].clone()
        )
        writer.add_images("Reconstructed/Experimental", recon_exp_norm, step)


def log_images_to_wandb(
    model_outputs: Dict[str, torch.Tensor],
    sim_real: torch.Tensor,
    exp_real: torch.Tensor,
    step: int,
    max_images: int = 4,
):
    """Log images to Weights & Biases."""
    batch_size = min(sim_real.shape[0], max_images)

    images_dict = {}

    # Create comparison images
    for i in range(batch_size):
        # Real images
        sim_real_np = tensor_to_numpy(sim_real[i])
        exp_real_np = tensor_to_numpy(exp_real[i])

        images_dict[f"real_sim_{i}"] = wandb.Image(
            sim_real_np, caption=f"Real Simulated {i+1}"
        )
        images_dict[f"real_exp_{i}"] = wandb.Image(
            exp_real_np, caption=f"Real Experimental {i+1}"
        )

        # Generated images
        if "fake_exp" in model_outputs:
            fake_exp_np = tensor_to_numpy(model_outputs["fake_exp"][i])
            images_dict[f"fake_exp_{i}"] = wandb.Image(
                fake_exp_np, caption=f"Sim→Exp {i+1}"
            )

        if "fake_sim" in model_outputs:
            fake_sim_np = tensor_to_numpy(model_outputs["fake_sim"][i])
            images_dict[f"fake_sim_{i}"] = wandb.Image(
                fake_sim_np, caption=f"Exp→Sim {i+1}"
            )

        # Reconstructed images
        if "reconstructed_sim" in model_outputs:
            recon_sim_np = tensor_to_numpy(model_outputs["reconstructed_sim"][i])
            images_dict[f"recon_sim_{i}"] = wandb.Image(
                recon_sim_np, caption=f"Reconstructed Sim {i+1}"
            )

        if "reconstructed_exp" in model_outputs:
            recon_exp_np = tensor_to_numpy(model_outputs["reconstructed_exp"][i])
            images_dict[f"recon_exp_{i}"] = wandb.Image(
                recon_exp_np, caption=f"Reconstructed Exp {i+1}"
            )

    wandb.log(images_dict, step=step)


def create_inference_visualization(
    input_image: torch.Tensor,
    generated_image: torch.Tensor,
    direction: str,  # "sim_to_exp" or "exp_to_sim"
    save_path: Path,
):
    """Create visualization for inference results."""
    images = [input_image.squeeze(0), generated_image.squeeze(0)]

    if direction == "sim_to_exp":
        titles = ["Input (Simulated)", "Generated (Experimental)"]
    else:
        titles = ["Input (Experimental)", "Generated (Simulated)"]

    save_image_grid(images, titles, save_path, nrow=2, figsize=(10, 5))
