import matplotlib

matplotlib.use("Agg")

import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image
from tempfile import TemporaryDirectory
from pathlib import Path
from joblib import Parallel, delayed
from pvio.video_io import write_frames_to_video

from poseforge.util.plot import (
    configure_matplotlib_style,
    get_segmentation_color_palette,
)


def set_up_figure(image_shape: tuple[int, int]):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, image_shape[1])
        ax.set_ylim(image_shape[0], 0)

    elements = {
        "ax_input_image": axes[0],
        "ax_pred_segmap": axes[1],
        "ax_pred_conf": axes[2],
    }
    elements["ax_input_image"].set_title("Input image")
    elements["ax_pred_segmap"].set_title("Predicted segmentation")
    elements["ax_pred_conf"].set_title("Confidence")

    elements["im_input_image"] = elements["ax_input_image"].imshow(
        np.zeros(image_shape, dtype=np.uint8), vmin=0, vmax=1, cmap="gray"
    )
    elements["im_pred_segmap"] = elements["ax_pred_segmap"].imshow(
        np.zeros((*image_shape, 3), dtype=np.uint8)
    )
    elements["im_pred_conf"] = elements["ax_pred_conf"].imshow(
        np.zeros(image_shape, dtype=np.uint8), vmin=0, vmax=100, cmap="viridis"
    )
    # Create a colorbar for the confidence image. Use the figure-level
    # colorbar API and pass the AxesImage returned by imshow.
    fig.colorbar(elements["im_pred_conf"], ax=elements["ax_pred_conf"])

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
        elements["im_pred_conf"].set_data(conf)


def worker_payload(
    frame_indices: list[int],
    input_frame_paths: list[Path],
    pred_segmaps: np.ndarray,
    pred_confs: np.ndarray | None,
    color_palette: list[np.ndarray],
    label_alpha: float,
    out_dir: Path,
):
    fig, elements = set_up_figure(
        image_shape=plt.imread(input_frame_paths[0]).shape[:2]
    )

    for i, frame_idx in enumerate(frame_indices):
        input_image = plt.imread(input_frame_paths[i])[:, :]  # already grayscale
        segmap = pred_segmaps[i, :, :]
        conf = pred_confs[i, :, :] if pred_confs is not None else None
        update_figure(
            elements=elements,
            input_image=input_image,
            segmap=segmap,
            color_palette=color_palette,
            input_alpha=label_alpha,
            conf=conf,
        )
        out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
        fig.savefig(out_path)


def visualize_bodyseg_prediction(
    recording_dir: Path,
    pred_path: Path,
    output_path: Path,
    output_fps: int = 30,
    label_alpha: float = 0.6,
    n_workers: int = -1,
):
    n_workers_eff = Parallel(n_jobs=n_workers)._effective_n_jobs()

    input_frame_paths = list(recording_dir.glob("*.jpg"))
    input_frame_paths.sort(key=lambda x: int(x.stem.replace("frame_", "")))
    with h5py.File(pred_path, "r") as f:
        class_labels = f["pred_segmap"].attrs["class_labels"].tolist()
        color_palette = get_segmentation_color_palette(len(class_labels))  #
        pred_segmaps = f["pred_segmap"]
        pred_confs = f["pred_confidence"]

        n_frames_per_worker = int(np.ceil(len(input_frame_paths) / n_workers_eff))
        payloads = []
        with TemporaryDirectory() as tmpdir:
            for w in range(n_workers_eff):
                start = w * n_frames_per_worker
                end = min((w + 1) * n_frames_per_worker, len(input_frame_paths))
                input_paths_chunk = input_frame_paths[start:end]
                frame_indices = [
                    int(path.stem.replace("frame_", "")) for path in input_paths_chunk
                ]
                pred_segmaps_chunk = pred_segmaps[start:end, :, :]
                pred_confs_chunk = pred_confs[start:end, :, :]

                payloads.append(
                    (
                        frame_indices,
                        input_paths_chunk,
                        pred_segmaps_chunk,
                        pred_confs_chunk,
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
            write_frames_to_video(output_path, frames, fps=output_fps)


if __name__ == "__main__":
    configure_matplotlib_style()

    recording_basedir = Path("bulk_data/behavior_images/spotlight_aligned_and_cropped")
    pred_basedir = Path("bulk_data/pose_estimation/bodyseg/trial_20251011a/inference")

    trial = "20250613-fly1b-013"
    recording_dir = recording_basedir / trial / "model_prediction/not_flipped"
    pred_path = pred_basedir / f"{trial}_model_prediction_not_flipped/bodyseg_pred.h5"
    output_path = pred_basedir / f"{trial}_model_prediction_not_flipped/viz.mp4"

    output_fps = 30
    label_alpha = 0.6
    n_workers = -1

    visualize_bodyseg_prediction(
        recording_dir=recording_dir,
        pred_path=pred_path,
        output_path=output_path,
        output_fps=output_fps,
        label_alpha=label_alpha,
        n_workers=n_workers,
    )
    print(f"Video saved to {output_path}")
