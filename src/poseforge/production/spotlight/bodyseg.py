import matplotlib
from poseforge.util.plot import configure_matplotlib_style

matplotlib.use("Agg")
configure_matplotlib_style()

import numpy as np
import torch
import torchvision.transforms as transforms
import h5py
import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cmasher as cmr
from fractions import Fraction
from time import perf_counter
from tqdm import tqdm
from pathlib import Path
from typing import Any
from loguru import logger

from pvio.torch_tools import SimpleVideoCollectionLoader
from parallel_animate import Animator, IndexedFrameParams
from parallel_animate.util import get_rendered_frame_ids

import poseforge.pose.bodyseg as bodyseg
from poseforge.util.plot import (
    get_segmentation_color_palette,
    configure_matplotlib_style,
)


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
    with h5py.File(bodyseg_output_path, "w") as f_h5:
        ds_confidence = f_h5.create_dataset(
            "confidence",
            shape=(n_frames_total, working_size, working_size),
            dtype="uint8",
            compression="gzip",
        )
        ds_labels = f_h5.create_dataset(
            "labels",
            shape=(n_frames_total, working_size, working_size),
            dtype="uint8",
            compression="gzip",
        )
        ds_labels.attrs["class_labels"] = pipeline.class_labels
        with open(architecture_config_path) as f_:
            architecture_config = yaml.safe_load(f_)
            confidence_method = architecture_config["confidence_method"]
        ds_confidence.attrs["confidence_method"] = confidence_method

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


def visualize_body_segmentation(
    *,
    visualization_output_path: Path,
    bodyseg_output_path: Path,
    aligned_behavior_video_path: Path,
    recording_fps: Fraction | int,
    play_speed: float,
    rendered_fps: int,
    plotted_image_size: int = 256,
    loading_batch_size: int = 128,
    loading_n_workers: int = 4,
    loading_buffer_size: int = 128,
    loading_cache_video_metadata: bool = True,
    rendering_n_workers: int = 12,
):
    logger.info("Visualizing body segmentation predictions")

    # Create video loader
    logger.info("Creating video loader for body segmentation")
    image_shape = (plotted_image_size, plotted_image_size)
    video_loader = SimpleVideoCollectionLoader(
        [aligned_behavior_video_path],
        transform=transforms.Resize(image_shape),
        batch_size=loading_batch_size,
        num_workers=loading_n_workers,
        buffer_size=loading_buffer_size,
        use_cached_video_metadata=loading_cache_video_metadata,
    )

    # Open bodyseg predictions
    logger.info(f"Loading bodyseg predictions from {bodyseg_output_path}")
    with h5py.File(bodyseg_output_path, "r") as f:
        ds_confidence = f["confidence"]
        ds_labels = f["labels"]
        confidence_method = ds_confidence.attrs["confidence_method"]
        class_labels = ds_labels.attrs["class_labels"]

        # Define animator
        color_palette = get_segmentation_color_palette(len(class_labels))
        animator = _BodySegAnimator(
            image_shape=image_shape,
            confidence_method=confidence_method,
            color_palette=color_palette,
        )

        # Define frame params iterator and make video
        rendered_ids = get_rendered_frame_ids(
            data_fps=recording_fps,
            play_speed=play_speed,
            rendered_fps=rendered_fps,
            n_data_frames=len(video_loader.dataset),
        )

        def iter_frames():
            for batch in video_loader:
                frames = batch["frames"]  # (B, 1, H, W)
                frame_ids = batch["frame_indices"]
                for i in range(frames.shape[0]):
                    frame_id = frame_ids[i]
                    if frame_id not in rendered_ids:
                        continue
                    frame = frames[i, 0, :, :].numpy()
                    labels = ds_labels[frame_id, :, :]
                    confidence = ds_confidence[frame_id, :, :]
                    yield IndexedFrameParams(
                        frame_id=frame_id, params=(frame, labels, confidence)
                    )

        animator.make_video(
            visualization_output_path,
            iter_frames(),
            n_frames=int(len(rendered_ids)),
            fps=Fraction(rendered_fps).limit_denominator(100),
            num_workers=rendering_n_workers,
        )


class _BodySegAnimator(Animator):
    def __init__(
        self,
        image_shape: tuple[int, int],
        confidence_method: str,
        color_palette: list[tuple[float, float, float]],
    ):
        self.image_shape = image_shape
        self.confidence_method = confidence_method
        self.color_palette = color_palette

    def setup(self):
        fig = plt.figure(figsize=(13, 4))

        # Create a more sophisticated layout: 3 equal panels + space for colorbar
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.05)

        # Create the three main axes (equal size)
        ax_input = fig.add_subplot(gs[0, 0])
        ax_segmap = fig.add_subplot(gs[0, 1])
        ax_conf = fig.add_subplot(gs[0, 2])
        ax_conf_colorbar = fig.add_subplot(gs[0, 3])

        # Configure all main axes with equal aspect ratio
        for ax in [ax_input, ax_segmap, ax_conf]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, self.image_shape[1])
            ax.set_ylim(self.image_shape[0], 0)
            ax.set_aspect("equal", adjustable="box")

        ax_input.set_title("Input")
        self.input_artist = ax_input.imshow(
            np.zeros(self.image_shape), vmin=0, vmax=1, cmap="gray"
        )
        ax_segmap.set_title("Predicted labels")
        self.labels_artist = ax_segmap.imshow(
            np.zeros((*self.image_shape, 3), dtype=np.uint8),
        )
        ax_conf.set_title("Confidence")
        self.conf_artist = ax_conf.imshow(
            np.zeros(self.image_shape), vmin=0, vmax=1, cmap=cmr.eclipse
        )

        # Create colorbar in the reserved space without affecting the uncertainty panel
        label = f"1 - {self.confidence_method}"
        cbar = fig.colorbar(self.conf_artist, cax=ax_conf_colorbar, label=label)

        # Increase the distance between colorbar and its label text
        cbar.ax.yaxis.labelpad = 20  # Increase padding between colorbar and label

        return fig

    def update(self, frame_id: int, data: Any):
        input_image, segmap, confidence = data

        self.input_artist.set_data(input_image)

        # Convert segmap to RGB using a color palette
        segmap_rgb = np.zeros((*segmap.shape, 3), dtype=np.float32)
        for class_id, color in enumerate(self.color_palette):
            segmap_rgb[segmap == class_id, :] = np.array(color)
        self.labels_artist.set_data(segmap_rgb)

        self.conf_artist.set_data(1 - confidence / 100)
