"""
Visualization script to verify that input images and target segmentation masks are properly aligned.

Uses the same hyperparameters as the training job.run file (hardcoded).
Loads one batch from the training dataloader and saves visualizations showing:
- Input image on the left
- Target segmentation mask overlay on the right
- Color-coded with a legend
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import torch
import poseforge.pose.bodyseg.config as config
from poseforge.pose.data.synthetic.atomic_batch import (
    init_atomic_dataset_and_dataloader,
    atomic_batches_to_simple_batch,
)
from poseforge.pose.bodyseg.pipeline import BodySegmentationPipeline
from poseforge.util.plot import get_segmentation_color_palette

def visualize_batch(
    batch_frames: torch.Tensor,
    batch_seg_maps: torch.Tensor,
    target_label_mapper,
    output_dir: Path,
    class_names: dict[int, str],
    color_palette,
):
    """Create visualizations for the batch."""
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = batch_frames.shape[0]
    logging.info(f"Visualizing {batch_size} samples from batch")

    for idx in range(batch_size):
        # Get input image and target map
        input_image = batch_frames[idx].cpu().numpy()  # (3, H, W) in [0, 1]
        target_map = batch_seg_maps[idx].cpu().numpy()  # (H, W) with class indices

        # Convert input to HWC format for visualization
        input_image_hwc = np.transpose(input_image, (1, 2, 0))  # (H, W, 3)
        input_image_uint8 = (input_image_hwc * 255).astype(np.uint8)

        # Get image dimensions
        h, w = target_map.shape
        n_classes = len(color_palette)

        # Create figure with side-by-side layout
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left: Input image
        axes[0].imshow(input_image_uint8, interpolation="nearest", origin="upper")
        axes[0].set_title("Input Image", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        # Right: Segmentation overlay
        # Create RGB visualization of segmentation
        seg_rgb = np.zeros((h, w, 3), dtype=np.float32)
        for class_id, color in enumerate(color_palette):
            mask = target_map == class_id
            seg_rgb[mask] = np.array(color)

        # Blend input image with segmentation
        # input_image_hwc is already (H, W, 3) in [0, 1] range
        input_image_rgb = input_image_hwc
        alpha = 0.6
        blended = (
            alpha * input_image_rgb + (1 - alpha) * seg_rgb
        )

        axes[1].imshow(blended, interpolation="nearest", origin="upper")
        axes[1].set_title("Target Segmentation (overlaid)", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        # Add legend below the segmentation image
        # Only show classes that are present in this sample
        unique_classes = np.unique(target_map)
        legend_patches = []
        for class_id in sorted(unique_classes):
            if class_id < len(color_palette):
                color = color_palette[int(class_id)]
                label = class_names.get(int(class_id), f"Class-{class_id}")
                legend_patches.append(
                    mpatches.Patch(
                        facecolor=color,
                        edgecolor="black",
                        linewidth=0.5,
                        label=label,
                    )
                )

        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=min(6, len(legend_patches)),
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        plt.tight_layout(rect=[0, 0.08, 1, 1])

        # Save figure
        output_path = output_dir / f"batch_sample_{idx:04d}.png"
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        logging.info(f"Saved visualization to {output_path}")
        plt.close(fig)

        # Also save a combined image for easier viewing
        if idx % 8 == 0:
            logging.info(f"  Processed {idx+1}/{batch_size} samples")


def main(
    n_samples_to_visualize: int = 32,
):
    """Main function to load data and create visualizations."""

    # ===== HARDCODED HYPERPARAMETERS FROM job.run =====
    selected_original_class_indices = np.array([
        39, 40, 41,
        47, 48, 49,
        55, 56, 57,
        63, 64, 65,
        71, 72, 73,
        79, 80, 81])
    n_classes = 19
    input_image_size = (912, 912)
    atomic_batch_n_samples = 32
    atomic_batch_n_variants = 1
    train_batch_size = 32
    n_workers = 8

    # These paths would need to be adapted or passed as arguments
    # For now, we'll use placeholder paths that the user can modify
    atomic_batch_base_dir = Path(
        "/Users/stimpfli/Downloads/atomic_batches"
        #"/work/upramdya/alfie/style_transfer/contrastive_pretraining/atomic_batches"
    )

    # Use validation data for safer visualization (single directory)
    train_data_dirs = [
        str(atomic_batch_base_dir / "BO_Gal4_fly4_trial001"),
    ]

    output_dir =atomic_batch_base_dir / "visualization_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)


    logging.info("=" * 60)
    logging.info("BODYSEG DATA ALIGNMENT VISUALIZATION")
    logging.info("=" * 60)
    logging.info(f"Input image size: {input_image_size}")
    logging.info(f"Number of classes: {n_classes}")
    logging.info(f"Selected original class indices: {selected_original_class_indices}")
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
        load_keypoint_positions=False,
        load_body_segment_maps=True,
        shuffle=False,
        n_workers=n_workers,
        n_channels=3,
        pin_memory=True,
        drop_last=False,
    )

    # Create target label mapper
    logging.info("Creating target label mapper...")
    target_label_mapper = (
        BodySegmentationPipeline.create_target_label_mapper(
            selected_original_class_indices=selected_original_class_indices,
            n_output_classes=n_classes,
        )
    )

    # Load class names directly from segmentation HDF5 (hardcoded path)
    # The HDF5 stores the original 87 class names in
    # segmentation_maps.attrs["keys"]. We index into that list using the
    # original class indices to get readable labels.
    import h5py

    segmentation_h5 = Path(
        "/Users/stimpfli/Desktop/prototype_data/bulk_data_flybody/BO_Gal4_fly2_trial005/segment_001/simulation_data.h5"
    )
    logging.info(f"Loading segment names from {segmentation_h5}")
    with h5py.File(segmentation_h5, "r") as f:
        keys_attr = f["segmentation_maps"].attrs["keys"]
        
    import json
    is_new_format = False
    if isinstance(keys_attr, (str, bytes, np.bytes_, np.str_)):
        keys_data = keys_attr.decode('utf-8') if isinstance(keys_attr, (bytes, np.bytes_)) else str(keys_attr)
        try:
            keys_dict = json.loads(keys_data)
            is_new_format = True
        except json.JSONDecodeError:
            pass
            
    if is_new_format:
        segments = {v: k for k, v in keys_dict.items()}
    else:
        segments = keys_attr.tolist() if hasattr(keys_attr, 'tolist') else keys_attr
        if isinstance(segments, list):
            segments = [s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s) for s in segments]
        else:
            segments = [str(segments)]
        segments.insert(0, "Background")  # Add background class at index 0
        segments = {i: name for i, name in enumerate(segments)}

    class_names = {0: "Background"}
    for new_idx, old_idx in enumerate(selected_original_class_indices, start=1):
        if old_idx in segments:
            class_names[new_idx] = segments[old_idx]
        else:
            class_names[new_idx] = f"Class-{old_idx}"

    # Get color palette
    color_palette = get_segmentation_color_palette(n_classes)
    logging.info(f"Generated color palette with {len(color_palette)} colors")

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
        logging.info(
            f"Target maps shape: {sim_data['body_seg_maps'].shape}"
        )

        # Apply target label mapper
        target_indices = sim_data["body_seg_maps"].long()
        target_indices_mapped = target_label_mapper(target_indices)

        print(frames.shape)
        print(target_indices.shape)

        # Limit to n_samples_to_visualize
        frames = frames[:n_samples_to_visualize]
        target_indices_mapped = target_indices_mapped[:n_samples_to_visualize]

        # Create visualizations
        output_path = Path(output_dir) / f"batch_{batch_idx:04d}"
        logging.info(f"Creating visualizations and saving to {output_path}")
        visualize_batch(
            batch_frames=frames,
            batch_seg_maps=target_indices_mapped,
            target_label_mapper=target_label_mapper,
            output_dir=output_path,
            class_names=class_names,
            color_palette=color_palette,
        )

        logging.info("Visualization complete!")
        break  # Only process first batch


if __name__ == "__main__":
    import tyro

    tyro.cli(
        main,
        prog="python visualize_data_alignment.py",
        description="Visualize data alignment between input images and target segmentation masks.",
    )
