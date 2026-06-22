"""
Visualization script to verify that input images and target keypoint positions are properly aligned.

Uses the same hyperparameters as the training job (hardcoded).
Loads one batch from the training dataloader and saves visualizations showing:
- Input image on the left
- Input image with ground truth keypoints marked on the right
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import torch
from poseforge.pose.data.synthetic.atomic_batch import (
    init_atomic_dataset_and_dataloader,
    atomic_batches_to_simple_batch,
)

def visualize_batch(
    batch_frames: torch.Tensor,
    batch_keypoints: torch.Tensor,
    output_dir: Path,
):
    """Create visualizations for the batch."""
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = batch_frames.shape[0]
    logging.info(f"Visualizing {batch_size} samples from batch")

    for idx in range(batch_size):
        # Get input image and keypoints
        input_image = batch_frames[idx].cpu().numpy()  # (3, H, W) in [0, 1]
        keypoints = batch_keypoints[idx].cpu().numpy()  # (n_keypoints, 2) in image coords

        # Convert input to HWC format for visualization
        input_image_hwc = np.transpose(input_image, (1, 2, 0))  # (H, W, 3)
        input_image_uint8 = (input_image_hwc * 255).astype(np.uint8)

        # Get image dimensions
        h, w = input_image_hwc.shape[:2]

        # Create figure with side-by-side layout
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left: Input image only
        axes[0].imshow(input_image_uint8, interpolation="nearest", origin="upper")
        axes[0].set_title("Input Image", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        # Right: Input image with keypoints
        axes[1].imshow(input_image_uint8, interpolation="nearest", origin="upper")
        
        # Draw keypoints
        axes[1].scatter(
            keypoints[:, 0],
            keypoints[:, 1],
            s=50,
            c="red",
            marker="x",
            linewidths=2,
            alpha=0.8
        )
        
        axes[1].set_title("Input with Ground Truth Keypoints", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        plt.tight_layout()

        # Save figure
        output_path = output_dir / f"batch_sample_{idx:04d}.png"
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        logging.info(f"Saved visualization to {output_path}")
        plt.close(fig)

        if idx % 8 == 0:
            logging.info(f"  Processed {idx+1}/{batch_size} samples")


def main(
    n_samples_to_visualize: int = 32,
):
    """Main function to load data and create visualizations."""

    # ===== HARDCODED HYPERPARAMETERS (same as bodyseg training) =====
    n_classes = 17
    input_image_size = (912, 912)
    atomic_batch_n_samples = 32
    atomic_batch_n_variants = 1
    train_batch_size = 32
    n_workers = 8

    # Data paths
    atomic_batch_base_dir = Path(
        "/Users/stimpfli/Downloads/atomic_batches"
    )

    # Use validation data for safer visualization (single directory)
    train_data_dirs = [
        str(atomic_batch_base_dir / "BO_Gal4_fly4_trial001"),
    ]

    output_dir = atomic_batch_base_dir / "keypoint_visualization_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 60)
    logging.info("KEYPOINTS3D DATA ALIGNMENT VISUALIZATION")
    logging.info("=" * 60)
    logging.info(f"Input image size: {input_image_size}")
    logging.info(f"Atomic batch n_samples: {atomic_batch_n_samples}")
    logging.info(f"Atomic batch n_variants: {atomic_batch_n_variants}")
    logging.info(f"Train batch size: {train_batch_size}")
    logging.info(f"Data directories: {train_data_dirs}")
    logging.info("=" * 60)

    # Initialize dataset and dataloader
    logging.info("Initializing dataset...")
    dataset, dataloader = init_atomic_dataset_and_dataloader(
        data_dirs=train_data_dirs,
        atomic_batch_n_samples=atomic_batch_n_samples,
        atomic_batch_n_variants=atomic_batch_n_variants,
        input_image_size=input_image_size,
        batch_size=train_batch_size,
        load_dof_angles=False,
        load_keypoint_positions=True,
        load_body_segment_maps=False,
        shuffle=False,
        n_workers=n_workers,
        n_channels=3,
        pin_memory=True,
        drop_last=False,
    )

    # Load one batch
    logging.info("Loading first batch from dataloader...")
    for batch_idx, (atomic_batches_frames, atomic_batches_sim_data) in enumerate(
        dataloader
    ):
        logging.info(
            f"Loaded batch {batch_idx}: frames shape {atomic_batches_frames.shape}"
        )

        # Convert atomic batches to simple batch
        frames, sim_data = atomic_batches_to_simple_batch(
            atomic_batches_frames, atomic_batches_sim_data, device="cpu"
        )

        logging.info(f"After collapsing: frames shape {frames.shape}")
        logging.info(f"Keypoint positions shape: {sim_data['keypoint_pos'].shape}")

        # Get keypoint positions (B, n_keypoints, 3) -> take only (x, y) 
        keypoint_xy = sim_data["keypoint_pos"][:, :, :2]

        # Limit to n_samples_to_visualize
        frames = frames[:n_samples_to_visualize]
        keypoint_xy = keypoint_xy[:n_samples_to_visualize]

        # Create visualizations
        output_path = Path(output_dir) / f"batch_{batch_idx:04d}"
        logging.info(f"Creating visualizations and saving to {output_path}")
        visualize_batch(
            batch_frames=frames,
            batch_keypoints=keypoint_xy,
            output_dir=output_path,
        )

        logging.info("Visualization complete!")
        break  # Only process first batch


if __name__ == "__main__":
    import tyro

    tyro.cli(
        main,
        prog="python visualize_keypoint_alignment.py",
        description="Visualize data alignment between input images and target keypoint positions.",
    )
