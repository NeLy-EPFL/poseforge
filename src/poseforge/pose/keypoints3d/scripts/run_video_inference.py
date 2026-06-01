"""Run 2.5D keypoint inference on a video and produce an output video
with predicted keypoints overlaid.

Given a checkpoint path like:
    .../trial_20260529_tiled_keypoints3d_freeze/checkpoints/epoch0_step1000.model.pth

The script:
    1. Infers the run directory (parent of ``checkpoints/``)
    2. Loads ``configs/model_architecture_config.yaml`` and
       ``configs/data_config.yaml`` from that directory
    3. Loads model weights from the checkpoint
    4. Runs batched inference on the input video
    5. Writes an output video with keypoints drawn, saved next to the
       input video as ``<video_stem>_<run_name>.mp4``

Usage:
    python run_video_inference.py /path/to/video.mp4 \\
        --checkpoint /path/to/trial_.../checkpoints/epoch0_step1000.model.pth \\
        [--batch_size 16] [--device cuda]
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from poseforge.pose.keypoints3d import Pose2p5DModel, Pose2p5DPipeline
from poseforge.pose.keypoints3d import config as keypoints3d_config
from poseforge.neuromechfly.constants import kchain_plotting_colors


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Keypoint coloring (reused from visualizer conventions)
def _get_keypoint_color_bgr(keypoint_idx: int, n_keypoints: int) -> tuple[int, int, int]:
    """Return a distinct BGR color for each keypoint, cycling through a
    perceptually-spaced palette."""
    # Use a fixed palette that's easy to distinguish on dark and light backgrounds
    palette_rgb = [
        (255, 0, 0), (0, 255, 0), (0, 100, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
        (0, 255, 128), (128, 255, 0), (255, 0, 128), (0, 128, 255),
    ]
    rgb = palette_rgb[keypoint_idx % len(palette_rgb)]
    return (rgb[2], rgb[1], rgb[0])  # BGR for OpenCV


def _resolve_model_dir(checkpoint_path: Path) -> Path:
    """Infer the model/run directory from a checkpoint path.

    Expects: ``<run_dir>/checkpoints/<checkpoint_file>``
    """
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path.parent.name != "checkpoints":
        raise ValueError(
            "Cannot infer model directory from checkpoint path. "
            "Expected '<run_dir>/checkpoints/*.model.pth', got: "
            f"{checkpoint_path}"
        )
    return checkpoint_path.parent.parent


def _load_pipeline(
    model_dir: Path,
    checkpoint_path: Path,
    device: str,
) -> tuple[Pose2p5DPipeline, tuple[int, int]]:
    """Load model, weights, and return the pipeline + expected image size."""
    # Architecture config
    arch_config_path = model_dir / "configs" / "model_architecture_config.yaml"
    if not arch_config_path.is_file():
        arch_config_path = model_dir / "model_architecture_config.yaml"
    if not arch_config_path.is_file():
        raise FileNotFoundError(
            f"Architecture config not found in {model_dir}/configs/ or {model_dir}/"
        )

    # Data config (for image size)
    data_config_path = model_dir / "configs" / "data_config.yaml"
    if not data_config_path.is_file():
        data_config_path = model_dir / "data_config.yaml"
    if not data_config_path.is_file():
        raise FileNotFoundError(
            f"Data config not found in {model_dir}/configs/ or {model_dir}/"
        )

    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)
    image_size = tuple(data_config.get("input_image_size", (256, 256)))

    # Build model
    arch_config = keypoints3d_config.ModelArchitectureConfig.load(arch_config_path)
    model = Pose2p5DModel.create_architecture_from_config(arch_config)
    model.load_weights_from_config(
        keypoints3d_config.ModelWeightsConfig(model_weights=str(checkpoint_path))
    )
    pipeline = Pose2p5DPipeline(model, device=device, use_float16=True)

    logger.info(f"Loaded model from {model_dir.name}")
    logger.info(f"  Checkpoint: {checkpoint_path.name}")
    logger.info(f"  Image size: {image_size}")
    logger.info(f"  Keypoints: {arch_config.n_keypoints}")

    return pipeline, image_size


def _draw_keypoints(
    frame_bgr: np.ndarray,
    keypoints_xy: np.ndarray,
    confidence_xy: np.ndarray | None = None,
    n_keypoints: int = 32,
    radius: int = 3,
    conf_threshold: float = 0.0,
) -> np.ndarray:
    """Draw keypoints on a BGR frame.

    Args:
        frame_bgr: (H, W, 3) uint8 BGR image.
        keypoints_xy: (n_keypoints, 2) predicted x, y in image pixels.
        confidence_xy: (n_keypoints,) optional confidence per keypoint.
        n_keypoints: total number of keypoints (for color cycling).
        radius: circle radius in pixels.
        conf_threshold: hide keypoints below this confidence.

    Returns:
        Annotated frame (H, W, 3) uint8 BGR.
    """
    out = frame_bgr.copy()
    for k in range(keypoints_xy.shape[0]):
        if confidence_xy is not None and confidence_xy[k] < conf_threshold:
            continue
        x, y = int(round(keypoints_xy[k, 0])), int(round(keypoints_xy[k, 1]))
        color = _get_keypoint_color_bgr(k, n_keypoints)
        cv2.circle(out, (x, y), radius, color, thickness=-1)
        cv2.circle(out, (x, y), radius, (255, 255, 255), thickness=1)
    return out


def run_video_inference(
    video_path: Path,
    checkpoint_path: Path,
    batch_size: int = 16,
    device: str = "cuda",
    output_path: Path | None = None,
    conf_threshold: float = 0.0,
    marker_radius: int = 3,
) -> Path:
    """Run inference on a video and write an annotated output video.

    Args:
        video_path: Path to input video.
        checkpoint_path: Path to model checkpoint
            (``<run_dir>/checkpoints/<name>.model.pth``).
        batch_size: Frames per inference batch.
        device: 'cuda' or 'cpu'.
        output_path: Explicit output path. If None, auto-generated as
            ``<video_dir>/<video_stem>_<run_name>.mp4``.
        conf_threshold: Hide keypoints with confidence below this value.
        marker_radius: Keypoint marker radius in pixels.

    Returns:
        Path to the output video.
    """
    video_path = Path(video_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Resolve model directory and run name
    model_dir = _resolve_model_dir(checkpoint_path)
    run_name = model_dir.name
    checkpoint_stem = checkpoint_path.stem.replace(".model", "")

    # Determine output path
    if output_path is None:
        output_path = (
            video_path.parent / f"{video_path.stem}_{run_name}_{checkpoint_stem}.mp4"
        )
    output_path = Path(output_path)
    logger.info(f"Output will be saved to: {output_path}")

    # Load model
    pipeline, image_size = _load_pipeline(model_dir, checkpoint_path, device)
    n_keypoints = pipeline.model.n_keypoints

    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Input video: {orig_w}x{orig_h}, {total_frames} frames, {fps:.1f} fps")

    # Open output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (orig_w, orig_h))

    # Process in batches
    target_h, target_w = image_size
    frames_buffer = []
    orig_frames_buffer = []
    pbar = tqdm(total=total_frames, desc="Inference")

    def _flush_batch():
        if not frames_buffer:
            return
        # Build tensor: (N, C, H, W) float32 in [0, 1]
        batch = np.stack(frames_buffer, axis=0)  # (N, H, W, 3) uint8
        batch_rgb = batch[:, :, :, ::-1].copy()  # BGR -> RGB
        batch_float = batch_rgb.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        tensor = torch.from_numpy(batch_float)

        with torch.no_grad():
            pred_dict = pipeline.inference(tensor)

        pred_xy = pred_dict["pred_xy"].cpu().numpy()  # (N, n_kp, 2)
        conf_xy = pred_dict["conf_xy"].cpu().numpy()  # (N, n_kp)

        # Scale predictions back to original resolution
        scale_x = orig_w / target_w
        scale_y = orig_h / target_h

        for i in range(len(frames_buffer)):
            kps = pred_xy[i].copy()
            kps[:, 0] *= scale_x
            kps[:, 1] *= scale_y
            annotated = _draw_keypoints(
                orig_frames_buffer[i],
                kps,
                confidence_xy=conf_xy[i],
                n_keypoints=n_keypoints,
                radius=marker_radius,
                conf_threshold=conf_threshold,
            )
            writer.write(annotated)

        frames_buffer.clear()
        orig_frames_buffer.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_frames_buffer.append(frame)
        resized = cv2.resize(frame, (target_w, target_h))
        frames_buffer.append(resized)

        if len(frames_buffer) >= batch_size:
            _flush_batch()

        pbar.update(1)

    # Flush remaining frames
    _flush_batch()

    pbar.close()
    cap.release()
    writer.release()

    logger.info(f"Done. Output video: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run 2.5D keypoint inference on a video and overlay predictions."
    )
    parser.add_argument(
        "video_path", type=Path, help="Path to input video file."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to checkpoint under <run_dir>/checkpoints/."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for inference (default: 16)."
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (default: cuda if available)."
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output video path. Default: <video_dir>/<stem>_<run_name>.mp4."
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=0.0,
        help="Hide keypoints below this confidence (default: 0.0)."
    )
    parser.add_argument(
        "--marker_radius", type=int, default=3,
        help="Keypoint marker radius in pixels (default: 3)."
    )
    args = parser.parse_args()

    run_video_inference(
        video_path=args.video_path,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
        output_path=args.output,
        conf_threshold=args.conf_threshold,
        marker_radius=args.marker_radius,
    )


if __name__ == "__main__":
    main()
