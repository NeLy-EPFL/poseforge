import torch
import torch.nn.functional as F
import h5py
from pathlib import Path
import re
import yaml
import math
import argparse
from importlib.resources import files

from poseforge.pose.bodyseg.scripts.run_bodyseg_inference import (
    run_bodyseg_inference_generic,
    compute_confidence_from_probs,
)


def temporal_smooth_probabilities(probs, window_size=5, std=1.0):
    """Apply 1D Gaussian temporal smoothing to the class probabilities for each pixel."""
    T, C, H, W = probs.shape
    # Create 1D Gaussian kernel
    kernel = [math.exp(-i**2 / (2 * std**2)) for i in range(-window_size//2 + 1, window_size//2 + 1)]
    kernel_tensor = torch.tensor(kernel, dtype=probs.dtype, device=probs.device)
    kernel_tensor = kernel_tensor / kernel_tensor.sum() # Normalize
    
    # Pad probs along time dimension to keep size T
    pad_size = window_size // 2
    # Permute to [C, H, W, T]
    p = probs.permute(1, 2, 3, 0)
    # Pad the last dimension (T) using replicate padding
    p_padded = F.pad(p, (pad_size, pad_size), mode='replicate')
    
    # Now reshape to [C * H * W, 1, T + 2*pad_size] to use F.conv1d
    C_dim, H_dim, W_dim, T_padded = p_padded.shape
    p_conv = p_padded.reshape(C_dim * H_dim * W_dim, 1, T_padded)
    
    # conv1d weight needs to be [out_channels, in_channels, kernel_width] = [1, 1, window_size]
    weight = kernel_tensor.view(1, 1, -1)
    
    # Apply conv1d
    smoothed = F.conv1d(p_conv, weight) # Shape: [C * H * W, 1, T]
    
    # Reshape and permute back to [T, C, H, W]
    smoothed = smoothed.view(C_dim, H_dim, W_dim, -1).permute(3, 0, 1, 2)
    return smoothed


def process_batch_smoothing(pipeline, batch):
    """Run standard inference and return softmax probabilities along with raw predictions."""
    pred_dict = pipeline.inference(batch["frames"])
    logits = pred_dict["logits"]
    
    # Extract softmax probabilities as float32 for CPU processing safety
    probs = torch.softmax(logits, dim=1).to(torch.float32).detach().cpu()
    
    # Extract raw predictions
    raw_seg = torch.argmax(logits, dim=1).to(torch.uint8).detach().cpu()
    raw_conf = (pred_dict["confidence"] * 100).to(torch.uint8).detach().cpu()
    
    data_items = []
    for i in range(logits.shape[0]):
        data_items.append((
            probs[i, :, :, :],
            raw_seg[i, :, :],
            raw_conf[i, :, :]
        ))
    return data_items


def make_save_predictions_smoothing(window_size, std):
    """Create a save_predictions closure with custom window_size and std parameters."""
    def save_predictions_smoothing(f, pipeline, data_items, video_obj):
        # Save raw predictions
        pred_segmaps_raw = torch.stack([x[1] for x in data_items], dim=0).cpu().numpy()
        ds_raw = f.create_dataset(
            "pred_segmap_raw",
            data=pred_segmaps_raw,
            dtype="uint8",
            compression="gzip",
            shuffle=True,
        )
        ds_raw.attrs["class_labels"] = pipeline.class_labels

        confs_raw = torch.stack([x[2] for x in data_items], dim=0).cpu().numpy()
        ds_conf_raw = f.create_dataset(
            "pred_confidence_raw",
            data=confs_raw,
            dtype="uint8",
            compression="gzip",
            shuffle=True,
        )
        ds_conf_raw.attrs["scale"] = 100
        ds_conf_raw.attrs["method"] = pipeline.model.confidence_method

        # Stack probabilities: [T, n_classes, H, W]
        all_probs = torch.stack([x[0] for x in data_items], dim=0)

        # Smooth probabilities
        smoothed_probs = temporal_smooth_probabilities(all_probs, window_size=window_size, std=std)

        # Compute smoothed predictions
        pred_segmaps_smoothed = torch.argmax(smoothed_probs, dim=1).to(torch.uint8).cpu().numpy()
        confidence_smoothed = (
            compute_confidence_from_probs(smoothed_probs, pipeline.model.confidence_method) * 100
        ).to(torch.uint8).cpu().numpy()

        # Save smoothed predictions as the primary datasets
        ds = f.create_dataset(
            "pred_segmap",
            data=pred_segmaps_smoothed,
            dtype="uint8",
            compression="gzip",
            shuffle=True,
        )
        ds.attrs["class_labels"] = pipeline.class_labels

        ds_conf = f.create_dataset(
            "pred_confidence",
            data=confidence_smoothed,
            dtype="uint8",
            compression="gzip",
            shuffle=True,
        )
        ds_conf.attrs["scale"] = 100
        ds_conf.attrs["method"] = pipeline.model.confidence_method

    return save_predictions_smoothing


def start_smoothing():
    parser = argparse.ArgumentParser(
        description="Run body segmentation inference with temporal probability smoothing."
    )
    parser.add_argument(
        "aligned_data_dir",
        type=Path,
        default=Path("bulk_data/spotlight_aligned_and_cropped"),
        help="Base directory containing aligned and cropped spotlight recording trials.",
    )
    parser.add_argument(
        "glob_pattern",
        type=str,
        default="fly*",
        help="Glob pattern to match spotlight trial directories.",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=files("poseforge").joinpath(
            "production/spotlight/config.yaml"
        ),
        help="Path to config file containing model paths and parameters.",
    )
    parser.add_argument(
        "--output_basedir",
        type=Path,
        help="Base directory to save output predictions. If not provided, will be saved in the model directory under production/epoch{epoch}_step{step}/",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Temporal window size for Gaussian smoothing filter (must be odd).",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=1.0,
        help="Standard deviation for the Gaussian filter kernel.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = start_smoothing()

    # Verify window size is odd
    if args.window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {args.window_size}")

    with open(args.config_path, "r") as f:
        prod_config = yaml.safe_load(f)

    checkpoint_path = Path(prod_config["bodyseg"]["checkpoint"])
    match = re.search(r"epoch(\d+)_step(\d+)", checkpoint_path.stem)
    if not match:
        raise ValueError(
            f"Could not extract epoch and step from checkpoint path: {checkpoint_path}"
        )
    epoch, step = int(match.group(1)), int(match.group(2))

    contrastive_checkpoint_path = prod_config.get("common", {}).get("feature_extractor_checkpoint") or \
                                  prod_config["bodyseg"].get("contrastive_checkpoint")

    model_dir = checkpoint_path.parent.parent
    batch_size = prod_config["bodyseg"]["batch_size"]
    n_workers = prod_config.get("common", {}).get("n_workers", prod_config["bodyseg"].get("n_workers", 16))
    inference_image_size = tuple(prod_config.get("common", {}).get("inference_image_size") or \
                                 prod_config["bodyseg"]["inference_image_size"])
    output_buffer_log_interval = prod_config["bodyseg"]["output_buffer_log_interval"]

    if args.output_basedir is None:
        output_basedir = model_dir / f"production/epoch{epoch}_step{step}/"
    else:
        output_basedir = args.output_basedir / f"bodyseg/epoch{epoch}_step{step}/"
    output_basedir.mkdir(parents=True, exist_ok=True)

    # Run generic inference with smoothing callbacks
    run_bodyseg_inference_generic(
        input_basedir=args.aligned_data_dir,
        model_dir=model_dir,
        model_checkpoint_path=checkpoint_path,
        contrastive_checkpoint_path=contrastive_checkpoint_path,
        output_basedir=output_basedir,
        batch_size=batch_size,
        n_workers=n_workers,
        inference_image_size=inference_image_size,
        output_buffer_log_interval=output_buffer_log_interval,
        glob_pattern=args.glob_pattern,
        output_filename="bodyseg_pred_smoothing.h5",
        process_batch_func=process_batch_smoothing,
        save_predictions_func=make_save_predictions_smoothing(args.window_size, args.std),
    )
