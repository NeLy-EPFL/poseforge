import torch
import torch.nn.functional as F
import h5py
from pathlib import Path
import re
import yaml
import math
import argparse
import numpy as np
from importlib.resources import files

from poseforge.pose.bodyseg.scripts.run_bodyseg_inference import (
    run_bodyseg_inference_generic,
    compute_confidence_from_probs,
)


def temporal_smooth_probabilities_online(probs, window_size=5, std=1.0):
    """Apply 1D Gaussian temporal smoothing to the class probabilities (no internal padding)."""
    T_total, C, H, W = probs.shape
    kernel = [math.exp(-i**2 / (2 * std**2)) for i in range(-window_size//2 + 1, window_size//2 + 1)]
    kernel_tensor = torch.tensor(kernel, dtype=probs.dtype, device=probs.device)
    kernel_tensor = kernel_tensor / kernel_tensor.sum()
    
    p = probs.permute(1, 2, 3, 0)
    C_dim, H_dim, W_dim, T_dim = p.shape
    p_conv = p.reshape(C_dim * H_dim * W_dim, 1, T_dim)
    
    weight = kernel_tensor.view(1, 1, -1)
    smoothed = F.conv1d(p_conv, weight)
    
    smoothed = smoothed.view(C_dim, H_dim, W_dim, -1).permute(3, 0, 1, 2)
    return smoothed


class BatchSmoother:
    def __init__(self, window_size=5, std=1.0):
        self.window_size = window_size
        self.std = std
        self.pad_size = window_size // 2
        self.past_probs = {}

    def __call__(self, pipeline, batch):
        pred_dict = pipeline.inference(batch["frames"])
        logits = pred_dict["logits"]
        probs = torch.softmax(logits, dim=1).to(torch.float32).detach().cpu()
        
        raw_seg = torch.argmax(logits, dim=1).to(torch.uint8).detach().cpu()
        raw_conf = (pred_dict["confidence"] * 100).to(torch.uint8).detach().cpu()
        
        video_indices = batch["video_indices"].cpu().numpy()
        smoothed_probs = torch.empty_like(probs)
        
        unique_vids = np.unique(video_indices)
        for vid in unique_vids:
            mask = (video_indices == vid)
            vid_probs = probs[mask]
            
            if vid in self.past_probs:
                past = self.past_probs[vid]
            else:
                past = vid_probs[0:1].expand(self.pad_size, -1, -1, -1)
                
            future = vid_probs[-1:].expand(self.pad_size, -1, -1, -1)
            concat_probs = torch.cat([past, vid_probs, future], dim=0)
            
            smoothed_vid_probs = temporal_smooth_probabilities_online(concat_probs, self.window_size, self.std)
            smoothed_probs[mask] = smoothed_vid_probs
            
            self.past_probs[vid] = vid_probs[-self.pad_size:].clone()

        pred_segmaps_smoothed = torch.argmax(smoothed_probs, dim=1).to(torch.uint8)
        confidence_smoothed = (compute_confidence_from_probs(smoothed_probs, pipeline.model.confidence_method) * 100).to(torch.uint8)
        
        data_items = []
        for i in range(probs.shape[0]):
            data_items.append((
                pred_segmaps_smoothed[i],
                confidence_smoothed[i],
                raw_seg[i],
                raw_conf[i]
            ))
        return data_items


def make_save_predictions_smoothing():
    """Create a save_predictions closure for the smoothed and raw outputs."""
    def save_predictions_smoothing(f, pipeline, data_items, video_obj):
        # Save smoothed predictions as the primary datasets
        pred_segmaps_smoothed = torch.stack([x[0] for x in data_items], dim=0).cpu().numpy()
        ds = f.create_dataset(
            "pred_segmap",
            data=pred_segmaps_smoothed,
            dtype="uint8",
            compression="gzip",
            shuffle=True,
        )
        ds.attrs["class_labels"] = pipeline.class_labels

        confidence_smoothed = torch.stack([x[1] for x in data_items], dim=0).cpu().numpy()
        ds_conf = f.create_dataset(
            "pred_confidence",
            data=confidence_smoothed,
            dtype="uint8",
            compression="gzip",
            shuffle=True,
        )
        ds_conf.attrs["scale"] = 100
        ds_conf.attrs["method"] = pipeline.model.confidence_method

        # Save raw predictions
        pred_segmaps_raw = torch.stack([x[2] for x in data_items], dim=0).cpu().numpy()
        ds_raw = f.create_dataset(
            "pred_segmap_raw",
            data=pred_segmaps_raw,
            dtype="uint8",
            compression="gzip",
            shuffle=True,
        )
        ds_raw.attrs["class_labels"] = pipeline.class_labels

        confs_raw = torch.stack([x[3] for x in data_items], dim=0).cpu().numpy()
        ds_conf_raw = f.create_dataset(
            "pred_confidence_raw",
            data=confs_raw,
            dtype="uint8",
            compression="gzip",
            shuffle=True,
        )
        ds_conf_raw.attrs["scale"] = 100
        ds_conf_raw.attrs["method"] = pipeline.model.confidence_method

        frame_ids = [
            int(p.stem.split("_")[1])
            for p in video_obj.phy_frame_id_to_path.values()
        ]
        f.create_dataset(
            "frame_ids",
            data=frame_ids,
            dtype="int",
            compression="gzip",
            shuffle=True,
        )

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

    model_dir = checkpoint_path.parent.parent
    batch_size = prod_config["bodyseg"]["batch_size"]
    n_workers = prod_config.get("common", {}).get("n_workers", prod_config["bodyseg"].get("n_workers", 16))
    inference_image_size = tuple(prod_config.get("common", {}).get("inference_image_size") or \
                                 prod_config["bodyseg"]["inference_image_size"])
    class_labels = prod_config.get("common", {}).get("class_labels")
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
        output_basedir=output_basedir,
        batch_size=batch_size,
        n_workers=n_workers,
        inference_image_size=inference_image_size,
        class_labels=class_labels,
        output_buffer_log_interval=output_buffer_log_interval,
        glob_pattern=args.glob_pattern,
        output_filename="bodyseg_pred_smoothing.h5",
        process_batch_func=BatchSmoother(args.window_size, args.std),
        save_predictions_func=make_save_predictions_smoothing(),
    )
