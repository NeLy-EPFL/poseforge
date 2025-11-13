import numpy as np
import torch
import torchvision.transforms as transforms
import h5py
from time import perf_counter
from tqdm import tqdm
from pathlib import Path
from typing import Any
from loguru import logger

from pvio.torch_tools import SimpleVideoCollectionLoader

import poseforge.pose.bodyseg as bodyseg


def predict_body_segmentation(
    *,
    bodyseg_output_path: Path,
    aligned_behavior_video_path: Path,
    bodyseg_model_config: dict[str, Any],
    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    loading_batch_size: int = 128,
    loading_n_workers: int = 8,
    loading_buffer_size: int = 128,
    loading_cache_video_metadata: bool = True,
) -> None:
    logger.info("Estimating body segmentation masks for behavior video")

    # Set up model and pipeline
    architecture_config_path = Path(
        bodyseg_model_config["architecture_config"]
    ).expanduser()
    logger.info(
        f"Setting up body segmentation model from architecture config "
        f"{architecture_config_path}"
    )
    model = bodyseg.BodySegmentationModel.create_architecture_from_config(
        architecture_config_path
    ).cuda()
    ckpt_path = Path(bodyseg_model_config["checkpoint"]).expanduser()
    logger.info(f"Loading body segmentation model weights from {ckpt_path}")
    weights_config = bodyseg.config.ModelWeightsConfig(model_weights=ckpt_path)
    model.load_weights_from_config(weights_config)
    logger.info("Creating body segmentation inference pipeline")
    pipeline = bodyseg.BodySegmentationPipeline(model, device=device, use_float16=True)

    # Create video loader
    logger.info("Creating video loader for body segmentation")
    working_size = bodyseg_model_config["working_size"]
    video_loader = SimpleVideoCollectionLoader(
        [aligned_behavior_video_path],
        transform=transforms.Resize((working_size, working_size)),
        batch_size=loading_batch_size,
        num_workers=loading_n_workers,
        buffer_size=loading_buffer_size,
        use_cached_video_metadata=loading_cache_video_metadata,
    )

    # Create output file
    logger.info(f"Creating H5 file for bodyseg predictions: {bodyseg_output_path}")
    n_frames_total = len(video_loader.dataset)
    with h5py.File(bodyseg_output_path, "w") as f:
        ds_confidence = f.create_dataset(
            "confidence",
            shape=(n_frames_total, working_size, working_size),
            dtype="uint8",
            compression="gzip",
        )
        ds_labels = f.create_dataset(
            "labels",
            shape=(n_frames_total, working_size, working_size),
            dtype="uint8",
            compression="gzip",
        )
        f.attrs["class_labels"] = pipeline.class_labels

        # Run inference
        logger.info("Running inference for body segmentation")
        for batch in tqdm(
            video_loader, desc="Predicting bodyseg", unit="batch", disable=None
        ):
            # Forward pass
            # No need to move data to GPU and back, Pipeline.inference handles that
            frames = batch["frames"]
            frame_ids = batch["frame_indices"]
            assert (np.array(batch["video_indices"]) == 0).all()
            pred_dict = pipeline.inference(frames)
            logits = pred_dict["logits"]  # (B, n_classes, H, W)
            confidence = pred_dict["confidence"]  # (B, H, W)
            confidence = (confidence * 100).to(torch.uint8)
            labels = torch.argmax(logits, dim=1).to(torch.uint8)  # (B, H, W)

            # Save to H5 file
            start_time = perf_counter()
            ds_labels[frame_ids, :, :] = labels.numpy()
            ds_confidence[frame_ids, :, :] = confidence.numpy()
            elapsed = perf_counter() - start_time
            logger.debug(f"Saved output for {len(frame_ids)} frames in {elapsed:.3f}s")

    logger.info("Body segmentation prediction complete")
