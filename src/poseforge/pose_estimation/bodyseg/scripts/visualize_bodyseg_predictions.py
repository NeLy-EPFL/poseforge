import matplotlib

matplotlib.use("Agg")

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v2 as imageio
import cmasher as cmr
import logging
from PIL import Image
from tempfile import TemporaryDirectory
from pathlib import Path
from joblib import Parallel, delayed
from pvio.video_io import write_frames_to_video

from poseforge.util.plot import (
    configure_matplotlib_style,
    get_segmentation_color_palette,
)

logging.basicConfig(level=logging.INFO)


def set_up_figure(image_shape: tuple[int, int], confidence_measure: str | None = None):
    fig = plt.figure(figsize=(13, 4))

    # Create a more sophisticated layout: 3 equal panels + space for colorbar
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.05)

    # Create the three main axes (equal size)
    ax_input = fig.add_subplot(gs[0, 0])
    ax_segmap = fig.add_subplot(gs[0, 1])
    ax_conf = fig.add_subplot(gs[0, 2])

    # Configure all main axes with equal aspect ratio
    for ax in [ax_input, ax_segmap, ax_conf]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, image_shape[1])
        ax.set_ylim(image_shape[0], 0)
        ax.set_aspect("equal", adjustable="box")

    elements = {
        "ax_input_image": ax_input,
        "ax_pred_segmap": ax_segmap,
        "ax_pred_conf": ax_conf,
    }
    elements["ax_input_image"].set_title("Input image")
    elements["ax_pred_segmap"].set_title("Predicted segmentation")
    elements["ax_pred_conf"].set_title("Uncertainty")

    elements["im_input_image"] = elements["ax_input_image"].imshow(
        np.zeros(image_shape, dtype=np.uint8), vmin=0, vmax=1, cmap="gray"
    )
    elements["im_pred_segmap"] = elements["ax_pred_segmap"].imshow(
        np.zeros((*image_shape, 3), dtype=np.uint8)
    )
    elements["im_pred_conf"] = elements["ax_pred_conf"].imshow(
        np.zeros(image_shape, dtype=np.uint8), vmin=0, vmax=1, cmap=cmr.eclipse
    )

    # Create colorbar in the reserved space without affecting the uncertainty panel
    cbar_ax = fig.add_subplot(gs[0, 3])
    if confidence_measure is None:
        label = None
    else:
        label = f"1 - {confidence_measure}"
    cbar = fig.colorbar(elements["im_pred_conf"], cax=cbar_ax, label=label)

    # Increase the distance between colorbar and its label text
    cbar.ax.yaxis.labelpad = 20  # Increase padding between colorbar and label

    return fig, elements


def update_figure(
    elements: dict[str, plt.Axes | matplotlib.image.AxesImage],
    input_image: np.ndarray,
    segmap: np.ndarray,
    color_palette: list[np.ndarray],
    input_alpha: float,
    conf: np.ndarray | None = None,
):
    input_image_pil = Image.fromarray(input_image)
    input_image_pil = input_image_pil.resize(
        (segmap.shape[1], segmap.shape[0]), Image.Resampling.BILINEAR
    )
    input_image = np.array(input_image_pil) / 255
    elements["im_input_image"].set_data(input_image)
    segmap_rgb = np.zeros((*segmap.shape, 3), dtype=np.float32)
    for class_id, color in enumerate(color_palette):
        segmap_rgb[segmap == class_id, :] = np.array(color)
    input_image_rgb = np.repeat(input_image[:, :, None], 3, axis=2)
    segmap_rgb = input_alpha * input_image_rgb + (1 - input_alpha) * segmap_rgb
    elements["im_pred_segmap"].set_data(segmap_rgb)
    if conf is not None and elements["im_pred_conf"] is not None:
        elements["im_pred_conf"].set_data(1 - (conf / 100))


def worker_payload(
    frame_indices: list[int],
    input_frame_paths: list[Path],
    pred_segmaps: np.ndarray,
    pred_confs: np.ndarray,
    confidence_measure: str | None,
    color_palette: list[np.ndarray],
    input_alpha: float,
    out_dir: Path,
):
    logging.basicConfig(level=logging.INFO)

    # Skip if no frames to process
    if len(input_frame_paths) == 0:
        return

    fig, elements = set_up_figure(
        image_shape=plt.imread(input_frame_paths[0]).shape[:2],
        confidence_measure=confidence_measure,
    )

    for i, frame_idx in enumerate(frame_indices):
        input_image = plt.imread(input_frame_paths[i])[:, :]  # already grayscale
        segmap = pred_segmaps[i, :, :]
        conf = pred_confs[i, :, :]
        update_figure(
            elements=elements,
            input_image=input_image,
            segmap=segmap,
            color_palette=color_palette,
            input_alpha=input_alpha,
            conf=conf,
        )
        out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
        fig.savefig(out_path)

        if (i + 1) % 300 == 0:
            logging.info(f"Worker saved {i + 1} / {len(frame_indices)} frames")


def visualize_bodyseg_prediction(
    recording_dir: Path,
    pred_path: Path,
    output_path: Path,
    output_fps: int = 30,
    label_alpha: float = 0.6,
    confidence_measure: str | None = None,
    n_workers: int = -1,
    max_n_frames: int | None = None,
):
    n_workers_eff = Parallel(n_jobs=n_workers)._effective_n_jobs()

    input_frame_paths = list(recording_dir.glob("*.jpg"))
    input_frame_paths.sort(key=lambda x: int(x.stem.replace("frame_", "")))

    with h5py.File(pred_path, "r") as f:
        class_labels = f["pred_segmap"].attrs["class_labels"].tolist()
        color_palette = get_segmentation_color_palette(len(class_labels))  #
        pred_segmaps = f["pred_segmap"]
        pred_confs = f["pred_confidence"]

        # Apply max_n_frames limit after loading data
        if max_n_frames is not None:
            input_frame_paths = input_frame_paths[:max_n_frames]
            pred_segmaps = pred_segmaps[:max_n_frames]
            pred_confs = pred_confs[:max_n_frames]

        n_frames_per_worker = int(np.ceil(len(input_frame_paths) / n_workers_eff))
        payloads = []
        with TemporaryDirectory() as tmpdir:
            logging.info(f"Using {tmpdir} for storing intermediate files")
            for w in range(n_workers_eff):
                start = w * n_frames_per_worker
                end = min((w + 1) * n_frames_per_worker, len(input_frame_paths))

                # Only create payload if there are frames to process
                if start < len(input_frame_paths):
                    input_paths_chunk = input_frame_paths[start:end]
                    frame_indices = [
                        int(path.stem.replace("frame_", ""))
                        for path in input_paths_chunk
                    ]
                    pred_segmaps_chunk = pred_segmaps[start:end, :, :]
                    pred_confs_chunk = pred_confs[start:end, :, :]

                    payloads.append(
                        (
                            frame_indices,
                            input_paths_chunk,
                            pred_segmaps_chunk,
                            pred_confs_chunk,
                            confidence_measure,
                            color_palette,
                            label_alpha,
                            Path(tmpdir),
                        )
                    )
            Parallel(n_jobs=n_workers_eff)(
                delayed(worker_payload)(*payload) for payload in payloads
            )

            # Combine all output images into a video
            frame_paths = list(Path(tmpdir).glob("frame_*.jpg"))
            frame_paths.sort(key=lambda x: int(x.stem.replace("frame_", "")))
            frames = []
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                frames.append(image)
            image_shape = frames[0].shape
            frames = [x[: image_shape[0], : image_shape[1]] for x in frames]
            logging.info(f"Writing video to {output_path}, {len(frames)} frames")
            write_frames_to_video(output_path, frames, fps=output_fps)
            logging.info(f"Finished writing video to {output_path}")


if __name__ == "__main__":
    configure_matplotlib_style()

    recording_basedir = Path("bulk_data/behavior_images/spotlight_aligned_and_cropped")
    pred_basedir = Path("bulk_data/pose_estimation/bodyseg/trial_20251012a/inference")

    trial = "20250613-fly1b-013"
    recording_dir = recording_basedir / trial / "model_prediction/not_flipped"
    pred_path = pred_basedir / f"{trial}_model_prediction_not_flipped/bodyseg_pred.h5"
    output_path = pred_basedir / f"{trial}_model_prediction_not_flipped/viz.mp4"

    output_fps = 30
    label_alpha = 0.3
    n_workers = -1

    visualize_bodyseg_prediction(
        recording_dir=recording_dir,
        pred_path=pred_path,
        output_path=output_path,
        output_fps=output_fps,
        label_alpha=label_alpha,
        n_workers=n_workers,
        confidence_measure="normalized prediction entropy",
        # max_n_frames=100,
    )
    print(f"Video saved to {output_path}")
