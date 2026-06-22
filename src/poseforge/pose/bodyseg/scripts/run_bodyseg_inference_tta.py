import torch
import h5py
from pathlib import Path
import re
import yaml

from poseforge.pose.bodyseg.scripts.run_bodyseg_inference import (
    start,
    run_bodyseg_inference_generic,
    compute_confidence_from_probs,
)


def process_batch_tta(pipeline, batch):
    """Run translation-based Test-Time Augmentation (TTA) on a batch of frames."""
    frames = batch["frames"]
    shifts = [(0, 0), (2, 2), (-2, -2), (2, -2), (-2, 2)]

    accumulated_probs = None
    raw_seg = None
    raw_conf = None

    for dy, dx in shifts:
        if dy == 0 and dx == 0:
            shifted_frames = frames
        else:
            shifted_frames = torch.roll(frames, shifts=(dy, dx), dims=(2, 3))

        pred_dict = pipeline.inference(shifted_frames)
        logits = pred_dict["logits"]

        if dy != 0 or dx != 0:
            logits = torch.roll(logits, shifts=(-dy, -dx), dims=(2, 3))

        probs = torch.softmax(logits, dim=1)

        if dy == 0 and dx == 0:
            raw_seg = torch.argmax(logits, dim=1).to(torch.uint8).detach().cpu()
            raw_conf = (pred_dict["confidence"] * 100).to(torch.uint8).detach().cpu()

        if accumulated_probs is None:
            accumulated_probs = probs
        else:
            accumulated_probs += probs

    avg_probs = accumulated_probs / len(shifts)
    pred_seg_tta = torch.argmax(avg_probs, dim=1).to(torch.uint8).detach().cpu()
    confidence_tta = (
        compute_confidence_from_probs(avg_probs, pipeline.model.confidence_method) * 100
    ).to(torch.uint8).detach().cpu()

    data_items = []
    for i in range(frames.shape[0]):
        data_items.append((
            pred_seg_tta[i, :, :],
            confidence_tta[i, :, :],
            raw_seg[i, :, :],
            raw_conf[i, :, :]
        ))
    return data_items


def make_save_predictions_tta(class_labels=None):
    """Create a save_predictions closure for the TTA stabilized and raw outputs."""
    def save_predictions_tta(f, pipeline, data_items, video_obj):
        # Save TTA results as the primary datasets
        pred_segmaps = torch.stack([x[0] for x in data_items], dim=0).cpu().numpy()
        ds = f.create_dataset(
            "pred_segmap",
            data=pred_segmaps,
            dtype="uint8",
            compression="gzip",
            shuffle=True,
        )
        ds.attrs["class_labels"] = class_labels if class_labels is not None else pipeline.class_labels

        confs = torch.stack([x[1] for x in data_items], dim=0).cpu().numpy()
        ds_conf = f.create_dataset(
            "pred_confidence",
            data=confs,
            dtype="uint8",
            compression="gzip",
            shuffle=True,
        )
        ds_conf.attrs["scale"] = 100
        ds_conf.attrs["method"] = pipeline.model.confidence_method

        # Save raw (intermediate) results
        pred_segmaps_raw = torch.stack([x[2] for x in data_items], dim=0).cpu().numpy()
        ds_raw = f.create_dataset(
            "pred_segmap_raw",
            data=pred_segmaps_raw,
            dtype="uint8",
            compression="gzip",
            shuffle=True,
        )
        ds_raw.attrs["class_labels"] = class_labels if class_labels is not None else pipeline.class_labels

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

    return save_predictions_tta


if __name__ == "__main__":
    # Reuse argument parsing
    input_basedir, glob_pattern, config_path, output_basedir = start()

    with open(config_path, "r") as f:
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

    if output_basedir is None:
        output_basedir = model_dir / f"production/epoch{epoch}_step{step}/"
    else:
        output_basedir = output_basedir / f"bodyseg/epoch{epoch}_step{step}/"
    output_basedir.mkdir(parents=True, exist_ok=True)

    # Run generic inference with TTA callbacks
    run_bodyseg_inference_generic(
        input_basedir=input_basedir,
        model_dir=model_dir,
        model_checkpoint_path=checkpoint_path,
        output_basedir=output_basedir,
        batch_size=batch_size,
        n_workers=n_workers,
        inference_image_size=inference_image_size,
        class_labels=class_labels,
        output_buffer_log_interval=output_buffer_log_interval,
        glob_pattern=glob_pattern,
        output_filename="bodyseg_pred_tta.h5",
        process_batch_func=process_batch_tta,
        save_predictions_func=make_save_predictions_tta(class_labels),
    )
