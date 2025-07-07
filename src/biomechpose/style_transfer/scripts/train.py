"""
Training script for CycleGAN fruit fly image translation.

Example usage:
    python train.py --sim_images_dir /path/to/simulated --exp_images_dir /path/to/experimental
    python train.py --sim_images_dir /path/to/simulated --exp_images_dir /path/to/experimental --use_wandb
"""

import tyro
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb

from biomechpose.style_transfer.model import CycleGAN
from biomechpose.style_transfer.dataset import create_dataloaders
from datetime import datetime
from biomechpose.style_transfer.visualize import (
    visualize_training_batch,
    visualize_cycle_consistency,
    log_images_to_tensorboard,
    log_images_to_wandb,
)
from biomechpose.style_transfer.utils import (
    initialize_weights,
    ImageBuffer,
    save_checkpoint,
    load_checkpoint,
    save_best_model,
    count_parameters,
    get_device,
    setup_directories,
    cleanup_old_checkpoints,
)


# fmt:off
@dataclass
class TrainingConfig:
    """Training configuration with all hyperparameters."""
    
    # === Data Configuration ===
    sim_images_dir: Path  # Directory containing simulated (RGB) images
    exp_images_dir: Path  # Directory containing experimental (grayscale) images
    output_basedir: Path = Path("./bulk_data/style_transfer/training/")  # Output directory for checkpoints, logs, etc.
    train_split: float = 0.8  # Fraction of data to use for training
    
    # === Model Architecture Hyperparameters ===
    generator_base_filters: int = 64  # Base number of filters in generator
    generator_n_residual_blocks: int = 6  # Number of ResNet blocks in generator (6 for 256/512, 9 for 900)
    generator_use_dropout: bool = False  # Use dropout in generator ResNet blocks
    discriminator_base_filters: int = 64  # Base number of filters in discriminator
    discriminator_n_layers: int = 3  # Number of layers in discriminator
    
    # === Training Hyperparameters ===
    batch_size: int = 1  # Batch size (CycleGAN typically uses 1)
    num_epochs: int = 200  # Number of training epochs
    lr_generator: float = 0.0002  # Learning rate for generators
    lr_discriminator: float = 0.0002  # Learning rate for discriminators
    beta1: float = 0.5  # Beta1 for Adam optimizer
    beta2: float = 0.999  # Beta2 for Adam optimizer
    
    # === Loss Hyperparameters ===
    lambda_cycle: float = 10.0  # Weight for cycle consistency loss
    lambda_identity: float = 0.5  # Weight for identity loss (0 to disable)
    
    # === Learning Rate Schedule ===
    lr_decay_start_epoch: int = 100  # Epoch to start linear learning rate decay
    
    # === Data Augmentation Hyperparameters ===
    image_size: int = 900  # Image size (images will be resized to this)
    random_crop_size: Optional[int] = None  # Random crop size (None to disable)
    horizontal_flip_prob: float = 0.5  # Probability of horizontal flip
    brightness_jitter: float = 0.1  # Brightness jitter amount
    contrast_jitter: float = 0.1  # Contrast jitter amount
    
    # === Training Stability ===
    buffer_size: int = 50  # Size of image buffer for discriminator training
    init_type: Literal['normal', 'xavier', 'kaiming', 'orthogonal'] = 'normal'  # Weight initialization
    init_gain: float = 0.02  # Gain for weight initialization
    
    # === Hardware and Performance ===
    num_workers: int = 4  # Number of dataloader workers
    device: Optional[str] = None  # Device to use ('cuda', 'cpu', 'mps', or None for auto)
    
    # === Logging and Checkpointing ===
    log_interval: int = 100  # Log every N batches
    save_interval: int = 1  # Save checkpoint every N epochs
    vis_interval: int = 500  # Save visualizations every N batches
    keep_last_n_checkpoints: int = 99999  # Keep only the last N checkpoints
    
    # === Weights & Biases ===
    use_wandb: bool = False  # Use Weights & Biases logging
    wandb_project: str = "biomechfly_style_transfer"  # W&B project name
    wandb_run_name: Optional[str] = None  # W&B run name (None for auto)
    
    # === Resume Training ===
    resume_from: Optional[Path] = None  # Path to checkpoint to resume from
    
    # === Random Seed ===
    random_seed: int = 42  # Random seed for reproducibility

    # === Final Resize ===
    final_resize: int = 512  # Final resize after all augmentations (output images will be this size)
# fmt:on


def get_image_paths(image_dir: Path) -> List[Path]:
    """Get all image paths from directory."""
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    image_paths = []

    for ext in extensions:
        image_paths.extend(image_dir.glob(f"*{ext}"))
        image_paths.extend(image_dir.glob(f"*{ext.upper()}"))

    return sorted(image_paths)


class GANLoss(nn.Module):
    """GAN loss with different loss types."""

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type

        if loss_type == "mse":
            self.loss = nn.MSELoss()
        elif loss_type == "bce":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def __call__(self, prediction, target_is_real):
        if self.loss_type == "mse":
            target = (
                torch.ones_like(prediction)
                if target_is_real
                else torch.zeros_like(prediction)
            )
        else:  # bce
            target = (
                torch.ones_like(prediction)
                if target_is_real
                else torch.zeros_like(prediction)
            )

        return self.loss(prediction, target)


def train_one_epoch(
    model: CycleGAN,
    dataloader: torch.utils.data.DataLoader,
    optimizers: dict,
    schedulers: dict,
    criteria: dict,
    buffers: dict,
    config: TrainingConfig,
    output_rundir: Path,
    epoch: int,
    writer: SummaryWriter,
    device: torch.device,
) -> dict:
    """Train for one epoch."""

    model.train()
    epoch_losses = {
        "G_total": 0.0,
        "G_adv": 0.0,
        "G_cycle": 0.0,
        "G_identity": 0.0,
        "D_sim": 0.0,
        "D_exp": 0.0,
    }

    total_batches = len(dataloader)

    for batch_idx, (sim_real, exp_real) in enumerate(dataloader):
        sim_real = sim_real.to(device)
        exp_real = exp_real.to(device)

        global_step = epoch * total_batches + batch_idx

        # =================== Train Generators ===================
        # Zero gradients
        optimizers["G"].zero_grad()

        # Forward pass
        outputs = model(sim_real, exp_real)

        # Adversarial losses
        loss_G_adv_sim = criteria["gan"](outputs["D_sim_fake"], True)
        loss_G_adv_exp = criteria["gan"](outputs["D_exp_fake"], True)
        loss_G_adv = loss_G_adv_sim + loss_G_adv_exp

        # Cycle consistency losses
        loss_cycle_sim = criteria["cycle"](outputs["reconstructed_sim"], sim_real)
        loss_cycle_exp = criteria["cycle"](outputs["reconstructed_exp"], exp_real)
        loss_cycle = loss_cycle_sim + loss_cycle_exp

        # Identity losses (optional)
        loss_identity = torch.tensor(0.0, device=device)
        if config.lambda_identity > 0:
            # General note: because our simulated images are RGB and experimental images
            # are grayscale, in the calculation of the identity loss, we will convert
            # the simulated images to grayscale by averaging the channels, and convert
            # the experimental images to RGB by repeating the channel 3 times.
            
            # G_exp_to_sim should be identity on simulated images (RGB->grayscale)
            identity_sim = model.G_exp_to_sim(sim_real.mean(dim=1, keepdim=True))
            loss_identity_sim = criteria["identity"](identity_sim, sim_real)

            # G_sim_to_exp should be identity on experimental images (grayscale->RGB)
            identity_exp = model.G_sim_to_exp(exp_real.repeat(1, 3, 1, 1))
            loss_identity_exp = criteria["identity"](identity_exp, exp_real)

            loss_identity = loss_identity_sim + loss_identity_exp

        # Total generator loss
        loss_G = (
            loss_G_adv
            + config.lambda_cycle * loss_cycle
            + config.lambda_identity * loss_identity
        )

        # Backward and optimize
        loss_G.backward()
        optimizers["G"].step()

        # =================== Train Discriminators ===================
        # Train D_sim
        optimizers["D_sim"].zero_grad()

        # Real images
        pred_real = model.D_sim(sim_real)
        loss_D_sim_real = criteria["gan"](pred_real, True)

        # Fake images (from buffer)
        fake_sim_buffered = buffers["sim"](outputs["fake_sim"].detach())
        pred_fake = model.D_sim(fake_sim_buffered)
        loss_D_sim_fake = criteria["gan"](pred_fake, False)

        loss_D_sim = (loss_D_sim_real + loss_D_sim_fake) * 0.5
        loss_D_sim.backward()
        optimizers["D_sim"].step()

        # Train D_exp
        optimizers["D_exp"].zero_grad()

        # Real images
        pred_real = model.D_exp(exp_real)
        loss_D_exp_real = criteria["gan"](pred_real, True)

        # Fake images (from buffer)
        fake_exp_buffered = buffers["exp"](outputs["fake_exp"].detach())
        pred_fake = model.D_exp(fake_exp_buffered)
        loss_D_exp_fake = criteria["gan"](pred_fake, False)

        loss_D_exp = (loss_D_exp_real + loss_D_exp_fake) * 0.5
        loss_D_exp.backward()
        optimizers["D_exp"].step()

        # =================== Logging ===================
        # Accumulate losses
        epoch_losses["G_total"] += loss_G.item()
        epoch_losses["G_adv"] += loss_G_adv.item()
        epoch_losses["G_cycle"] += (config.lambda_cycle * loss_cycle).item()
        epoch_losses["G_identity"] += (config.lambda_identity * loss_identity).item()
        epoch_losses["D_sim"] += loss_D_sim.item()
        epoch_losses["D_exp"] += loss_D_exp.item()

        # Log to TensorBoard
        if batch_idx % config.log_interval == 0:
            writer.add_scalar("Loss/G_total", loss_G.item(), global_step)
            writer.add_scalar("Loss/G_adversarial", loss_G_adv.item(), global_step)
            writer.add_scalar("Loss/G_cycle", loss_cycle.item(), global_step)
            writer.add_scalar("Loss/G_identity", loss_identity.item(), global_step)
            writer.add_scalar("Loss/D_sim", loss_D_sim.item(), global_step)
            writer.add_scalar("Loss/D_exp", loss_D_exp.item(), global_step)

            # Log learning rates
            for name, scheduler in schedulers.items():
                writer.add_scalar(f"LR/{name}", scheduler.get_last_lr()[0], global_step)

            # Log to W&B
            if config.use_wandb:
                wandb.log(
                    {
                        "Loss/G_total": loss_G.item(),
                        "Loss/G_adversarial": loss_G_adv.item(),
                        "Loss/G_cycle": loss_cycle.item(),
                        "Loss/G_identity": loss_identity.item(),
                        "Loss/D_sim": loss_D_sim.item(),
                        "Loss/D_exp": loss_D_exp.item(),
                        "epoch": epoch,
                        "batch": batch_idx,
                    },
                    step=global_step,
                )

            print(
                f"Epoch [{epoch}/{config.num_epochs}], Batch [{batch_idx}/{total_batches}], "
                f"G_loss: {loss_G.item():.4f}, D_sim: {loss_D_sim.item():.4f}, "
                f"D_exp: {loss_D_exp.item():.4f}"
            )

        # Save visualizations
        if batch_idx % config.vis_interval == 0:
            vis_path = (
                output_rundir
                / "visualizations"
                / f"epoch_{epoch}_batch_{batch_idx}.png"
            )
            visualize_training_batch(outputs, sim_real, exp_real, vis_path)

            # Log images
            log_images_to_tensorboard(writer, outputs, sim_real, exp_real, global_step)

            if config.use_wandb:
                log_images_to_wandb(outputs, sim_real, exp_real, global_step)

    # Average losses over epoch
    for key in epoch_losses:
        epoch_losses[key] /= total_batches

    return epoch_losses


def main(config: TrainingConfig):
    """Main training function."""

    # Set random seed
    torch.manual_seed(config.random_seed)

    # Setup device
    if config.device is None:
        device = get_device()
    else:
        device = torch.device(config.device)

    print(f"Using device: {device}")

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_rundir = config.output_basedir / f"trial_{timestamp}"
    dirs = setup_directories(output_rundir)

    # Get image paths
    sim_paths = get_image_paths(config.sim_images_dir)
    exp_paths = get_image_paths(config.exp_images_dir)

    print(f"Found {len(sim_paths)} simulated images")
    print(f"Found {len(exp_paths)} experimental images")

    if len(sim_paths) == 0 or len(exp_paths) == 0:
        raise ValueError("No images found in the specified directories")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        sim_paths,
        exp_paths,
        train_split=config.train_split,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
        random_crop_size=config.random_crop_size,
        horizontal_flip_prob=config.horizontal_flip_prob,
        brightness_jitter=config.brightness_jitter,
        contrast_jitter=config.contrast_jitter,
        random_seed=config.random_seed,
        final_resize=config.final_resize,  # Pass through
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Create model
    model = CycleGAN(
        generator_base_filters=config.generator_base_filters,
        generator_n_residual_blocks=config.generator_n_residual_blocks,
        generator_use_dropout=config.generator_use_dropout,
        discriminator_base_filters=config.discriminator_base_filters,
        discriminator_n_layers=config.discriminator_n_layers,
    ).to(device)

    # Initialize weights
    initialize_weights(model, config.init_type, config.init_gain)

    # Print model info
    print(f"Model parameters: {count_parameters(model):,}")

    # Create optimizers
    optimizers = {
        "G": optim.Adam(
            list(model.G_sim_to_exp.parameters())
            + list(model.G_exp_to_sim.parameters()),
            lr=config.lr_generator,
            betas=(config.beta1, config.beta2),
        ),
        "D_sim": optim.Adam(
            model.D_sim.parameters(),
            lr=config.lr_discriminator,
            betas=(config.beta1, config.beta2),
        ),
        "D_exp": optim.Adam(
            model.D_exp.parameters(),
            lr=config.lr_discriminator,
            betas=(config.beta1, config.beta2),
        ),
    }

    # Create learning rate schedulers
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - config.lr_decay_start_epoch) / (
            config.num_epochs - config.lr_decay_start_epoch
        )
        return lr_l

    schedulers = {
        "G": optim.lr_scheduler.LambdaLR(optimizers["G"], lr_lambda=lambda_rule),
        "D_sim": optim.lr_scheduler.LambdaLR(
            optimizers["D_sim"], lr_lambda=lambda_rule
        ),
        "D_exp": optim.lr_scheduler.LambdaLR(
            optimizers["D_exp"], lr_lambda=lambda_rule
        ),
    }

    # Create loss functions
    criteria = {
        "gan": GANLoss("mse").to(device),
        "cycle": nn.L1Loss(),
        "identity": nn.L1Loss(),
    }

    # Create image buffers
    buffers = {
        "sim": ImageBuffer(config.buffer_size),
        "exp": ImageBuffer(config.buffer_size),
    }

    # Setup logging
    writer = SummaryWriter(dirs["logs"])

    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )

    # Resume training if specified
    start_epoch = 0
    best_loss = float("inf")

    if config.resume_from is not None:
        print(f"Resuming training from {config.resume_from}")
        start_epoch, _ = load_checkpoint(
            config.resume_from, model, optimizers, schedulers, device
        )
        start_epoch += 1

    # Convert config to dict for saving
    config_dict = vars(config)
    config_dict["device"] = str(device)

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch}/{config.num_epochs}")

        # Train
        epoch_losses = train_one_epoch(
            model,
            train_loader,
            optimizers,
            schedulers,
            criteria,
            buffers,
            config,
            output_rundir,
            epoch,
            writer,
            device,
        )

        # Update learning rates
        for scheduler in schedulers.values():
            scheduler.step()

        # Log epoch losses
        for name, loss in epoch_losses.items():
            writer.add_scalar(f"Epoch/{name}", loss, epoch)

        if config.use_wandb:
            wandb.log(
                {f"Epoch/{name}": loss for name, loss in epoch_losses.items()},
                step=epoch,
            )

        print(
            f"Epoch {epoch} - G_total: {epoch_losses['G_total']:.4f}, "
            f"D_sim: {epoch_losses['D_sim']:.4f}, D_exp: {epoch_losses['D_exp']:.4f}"
        )

        # Save checkpoint
        if epoch % config.save_interval == 0 or epoch == config.num_epochs - 1:
            checkpoint_path = dirs["checkpoints"] / f"checkpoint_epoch_{epoch}.pth"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizers,
                schedulers,
                epoch,
                epoch_losses,
                config_dict,
            )

        # Save best model
        current_loss = epoch_losses["G_total"]
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_path = dirs["models"] / "best_model.pth"
            save_best_model(best_model_path, model, config_dict)
            print(f"New best model saved with loss: {best_loss:.4f}")

        # Cleanup old checkpoints
        if epoch % (config.save_interval * 2) == 0:
            cleanup_old_checkpoints(dirs["checkpoints"], config.keep_last_n_checkpoints)

    # Save final model
    final_model_path = dirs["models"] / "final_model.pth"
    save_best_model(final_model_path, model, config_dict)

    writer.close()

    if config.use_wandb:
        wandb.finish()

    print("Training completed!")


if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)
    main(config)
