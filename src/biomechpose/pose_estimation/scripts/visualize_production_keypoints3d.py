#!/usr/bin/env python3
"""
Script to create visualization videos for production 3D keypoint inference results.
This script takes inference results and creates two-panel videos showing:
- Left panel: Real data frames
- Right panel: 3D skeleton animation with camera movement
"""

import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio
from joblib import Parallel, delayed
import logging
import shutil
import tempfile

from biomechpose.pose_estimation.keypoints_3d.visualizer import (
    get_keypoint_color,
    get_skeleton_connections,
)
from biomechpose.simulate_nmf.constants import (
    keypoint_segments_canonical,
    kchain_plotting_colors,
)
from biomechpose.util import (
    configure_matplotlib_style,
    default_video_writing_ffmpeg_params,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configure matplotlib style
configure_matplotlib_style()

# Animation constants
CAMERA_PAN_PERIOD = 300.0  # number of frames during a full camera pan cycle
ELEVATION_ANGLE = 30.0
AZIMUTH_AMPLITUDE = 30.0


def setup_two_panel_figure(
    img_shape, x_lim, y_lim, z_lim, keypoints_order, skeleton_connections
):
    """Setup figure with two panels: left for image, right for 3D skeleton"""
    fig, (ax_img, ax_3d) = plt.subplots(1, 2, figsize=(8, 4))

    # Setup left panel for image display
    ax_img.set_title("Real Data", fontweight="bold")
    ax_img.axis("off")
    # Create empty image plot that we'll update
    # Determine whether image is grayscale or color
    is_grayscale = (len(img_shape) == 2) or (len(img_shape) == 3 and img_shape[-1] == 1)

    # Use float range [0, 1] since images are normalized before plotting
    # Use origin='upper' so plotted 2D keypoints (pixel coordinates) match image coords
    if is_grayscale:
        img_plot = ax_img.imshow(
            np.zeros(img_shape),
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="gray",
            origin="upper",
        )
    else:
        # For color images (H, W, 3) use RGB plotting without a colormap
        img_plot = ax_img.imshow(
            np.zeros(img_shape), aspect="auto", vmin=0, vmax=1, origin="upper"
        )

    # Prepare 2D skeleton overlay artists on the image axis
    # Helper to detect leg vs antennae
    def _is_leg(name: str) -> bool:
        return (
            name[:2] in {"LF", "LM", "LH", "RF", "RM", "RH"} and "Pedicel" not in name
        )

    def _is_antenna(name: str) -> bool:
        return "Pedicel" in name or "Antenna" in name

    two_d_lines = []
    two_d_scatters = {}
    for connection in skeleton_connections:
        start_idx, end_idx = connection
        start_name = keypoints_order[start_idx]
        leg_id = start_name[:2] if len(start_name) >= 2 else None
        color = kchain_plotting_colors.get(leg_id, np.array([0.5, 0.5, 0.5]))
        (line2d,) = ax_img.plot([0, 0], [0, 0], color=color, linewidth=3)
        two_d_lines.append(line2d)

    # Create scatter artists only for antennae and non-leg keypoints
    for kp_idx, keypoint_name in enumerate(keypoints_order):
        if _is_leg(keypoint_name):
            # legs: no scatter on 2D overlay
            continue
        color = get_keypoint_color(keypoint_name)
        sc = ax_img.scatter([], [], color=color, s=15)
        two_d_scatters[kp_idx] = sc

    # Setup right panel for 3D skeleton
    ax_3d.remove()  # Remove the 2D axes
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    ax_3d.set_xlabel("X (mm)")
    ax_3d.set_ylabel("Y (mm)")
    ax_3d.set_zlabel("Z (mm)")
    ax_3d.set_title("3D Skeleton", fontweight="bold")
    # Use the requested, fixed axis limits and set an equal aspect ratio
    ax_3d.set_xlim(0.5, 3.5)
    ax_3d.set_ylim(-3.5, -0.5)
    ax_3d.set_zlim(-0.5, 2.0)
    # Set a box aspect proportional to the axis ranges so scaling looks equal
    try:
        xr = 3.5 - 0.5
        yr = -0.5 - (-3.5)
        zr = 2.0 - (-0.5)
        ax_3d.set_box_aspect((xr, yr, zr))
    except Exception:
        # Older matplotlib versions may not support set_box_aspect; ignore
        pass

    return fig, ax_img, ax_3d, img_plot, two_d_lines, two_d_scatters


def render_frame_chunk_worker(
    frame_indices,
    frames_dir,
    frame_ids,
    img_dir,
    keypoints_pos,
    keypoints_order,
    skeleton_connections,
    img_shape,
    x_lim,
    y_lim,
    z_lim,
    keypoints_pos_2d=None,
    camera_image_size: tuple[int, int] | None = None,
):
    """Worker function to render a chunk of frames with figure reuse"""
    # Setup figure once for this worker (also get 2D overlay artists)
    fig, ax_img, ax_3d, img_plot, two_d_lines, two_d_scatters = setup_two_panel_figure(
        img_shape, x_lim, y_lim, z_lim, keypoints_order, skeleton_connections
    )

    # Pre-allocate line and scatter plot objects for 3D skeleton
    skeleton_lines = []
    keypoint_scatters = [None] * len(keypoints_order)

    # Initialize with dummy data for 3D lines
    for connection in skeleton_connections:
        start_idx, end_idx = connection
        start_name = keypoints_order[start_idx]
        leg_id = start_name[:2] if len(start_name) >= 2 else None
        color = kchain_plotting_colors.get(leg_id, np.array([0.5, 0.5, 0.5]))
        (line,) = ax_3d.plot3D([0, 0], [0, 0], [0, 0], color=color, linewidth=3)
        skeleton_lines.append(line)

    # For 3D: only create scatter markers for antennae and non-leg keypoints
    def _is_leg(name: str) -> bool:
        return (
            name[:2] in {"LF", "LM", "LH", "RF", "RM", "RH"} and "Pedicel" not in name
        )

    def _is_antenna(name: str) -> bool:
        return "Pedicel" in name or "Antenna" in name

    for kp_idx, keypoint_name in enumerate(keypoints_order):
        color = get_keypoint_color(keypoint_name)
        if _is_leg(keypoint_name):
            # skip scatter for leg keypoints (lines will represent legs)
            keypoint_scatters[kp_idx] = None
        else:
            # scatter for antennae and other keypoints
            scatter = ax_3d.scatter([0], [0], [0], color=color, s=50)
            keypoint_scatters[kp_idx] = scatter

    # Render each frame in this chunk

    for i in frame_indices:
        frame_id = frame_ids[i]

        # Load and update image (convert to RGB and resize to display size)
        img_path = img_dir / f"frame_{frame_id:09d}.jpg"
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue
        img = Image.open(img_path).convert("RGB")
        # Resize image to display size (W,H)
        display_size = (img_shape[1], img_shape[0])
        img = img.resize(display_size, Image.Resampling.LANCZOS)
        img_array = np.array(img).astype(np.float32) / 255.0
        if img_array.ndim == 2:
            display_array = img_array
        elif img_array.ndim == 3 and img_array.shape[2] == 4:
            display_array = img_array[:, :, :3]
        else:
            display_array = img_array
        img_plot.set_array(np.clip(display_array, 0.0, 1.0))

        ax_img.set_title("Spotlight recording")

        # Get 3D keypoints for this frame
        keypoints_3d = keypoints_pos[i]  # (32, 3)

        # Update 3D skeleton lines
        for line_idx, connection in enumerate(skeleton_connections):
            start_idx, end_idx = connection
            start_pos = keypoints_3d[start_idx]
            end_pos = keypoints_3d[end_idx]
            skeleton_lines[line_idx].set_data_3d(
                [start_pos[0], end_pos[0]],
                [start_pos[1], end_pos[1]],
                [start_pos[2], end_pos[2]],
            )

        # Update 3D keypoint scatters (skip None entries for leg keypoints)
        for kp_idx, scatter in enumerate(keypoint_scatters):
            if scatter is None:
                continue
            pos = keypoints_3d[kp_idx]
            scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        # Update 3D plot title and camera view
        ax_3d.set_title("Predicted 3D pose")
        azimuth = np.cos(2 * np.pi * i / CAMERA_PAN_PERIOD) * AZIMUTH_AMPLITUDE
        ax_3d.view_init(elev=ELEVATION_ANGLE, azim=azimuth)

        # Update 2D overlay if provided (scale from camera pixel coords to display size)
        if keypoints_pos_2d is not None:
            keypoints_2d_frame = keypoints_pos_2d[i].astype(np.float32)
            # display size (W,H)
            disp_w = img_shape[1]
            disp_h = img_shape[0]
            if camera_image_size is not None:
                orig_w, orig_h = camera_image_size
                sx = float(disp_w) / float(orig_w)
                sy = float(disp_h) / float(orig_h)
                keypoints_2d_frame = keypoints_2d_frame.copy()
                keypoints_2d_frame[:, 0] = keypoints_2d_frame[:, 0] * sx
                keypoints_2d_frame[:, 1] = keypoints_2d_frame[:, 1] * sy

            # Heuristic detection for coordinate ordering/origin issues.
            # Try four transforms: none, swap (x<->y), flip (y -> H - y), swap+flip.
            def count_in_bounds(kps):
                xs = kps[:, 0]
                ys = kps[:, 1]
                inside = (xs >= 0) & (xs < disp_w) & (ys >= 0) & (ys < disp_h)
                return int(np.sum(inside))

            kps = keypoints_2d_frame.copy()
            transforms = {
                "none": kps,
                "swap": kps[:, [1, 0]],
                "flip": np.stack([kps[:, 0], disp_h - kps[:, 1]], axis=1),
                "swap_flip": np.stack([kps[:, 1], disp_h - kps[:, 0]], axis=1),
            }
            best_name = "none"
            best_cnt = -1
            for name, tk in transforms.items():
                try:
                    cnt = count_in_bounds(tk)
                except Exception:
                    cnt = -1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_name = name
            if best_name != "none":
                keypoints_2d_frame = transforms[best_name]
                logger.debug(f"Applied 2D keypoint transform: {best_name}")

            # Update 2D lines
            for line_idx, connection in enumerate(skeleton_connections):
                s, e = connection
                sxy = keypoints_2d_frame[s]
                exy = keypoints_2d_frame[e]
                two_d_lines[line_idx].set_data([sxy[0], exy[0]], [sxy[1], exy[1]])
            # Update 2D scatters (antennae/others)
            for kp_idx, sc in two_d_scatters.items():
                xy = keypoints_2d_frame[kp_idx]
                sc.set_offsets([xy[0], xy[1]])

        # Save frame
        frame_path = frames_dir / f"frame_{i:06d}.png"
        # Use no tight bbox and a small pad to keep titles/labels from being cut
        fig.savefig(frame_path, dpi=100, bbox_inches=None, pad_inches=0.1)

    plt.close(fig)


def render_frames_parallel(
    frames_dir,
    frame_ids,
    img_dir,
    keypoints_pos,
    keypoints_order,
    skeleton_connections,
    img_shape,
    x_lim,
    y_lim,
    z_lim,
    n_workers=-2,
    keypoints_pos_2d=None,
    camera_image_size: tuple[int, int] | None = None,
):
    """Render frames in parallel using joblib"""
    n_frames = len(frame_ids)

    # Determine number of workers
    if n_workers == -1:
        import multiprocessing

        n_workers = multiprocessing.cpu_count()
    elif n_workers == -2:
        import multiprocessing

        n_workers = max(1, multiprocessing.cpu_count() - 1)
    elif n_workers <= 0:
        n_workers = 1

    logger.info(f"Rendering {n_frames} frames using {n_workers} workers")

    # Split frame indices into chunks for workers
    frame_indices = list(range(n_frames))
    chunk_size = max(1, n_frames // n_workers)
    frame_chunks = [
        frame_indices[i : i + chunk_size] for i in range(0, n_frames, chunk_size)
    ]

    # Render chunks in parallel
    Parallel(n_jobs=n_workers)(
        delayed(render_frame_chunk_worker)(
            chunk,
            frames_dir,
            frame_ids,
            img_dir,
            keypoints_pos,
            keypoints_order,
            skeleton_connections,
            img_shape,
            x_lim,
            y_lim,
            z_lim,
            keypoints_pos_2d,
            camera_image_size,
        )
        for chunk in frame_chunks
    )


def visualize_predictions(
    inference_output_path: Path,
    input_images_dir: Path,
    output_video_path: Path,
    fps: int = 30,
    n_workers: int = -2,
):
    """Create a two-panel video showing real frames (left) and 3D skeleton (right)

    Args:
        trial_name: Name of the trial to visualize
        output_basedir: Base directory containing inference results
        input_basedir: Base directory containing original image data
        fps: Frames per second for output video
        n_workers: Number of parallel workers (-2 = all cores except 1)
    """
    logger.info(f"Creating visualization for: {inference_output_path}")

    # Load inference results
    with h5py.File(inference_output_path, "r") as f:
        frame_ids = f["frame_ids"][:]
        # 3D world coordinates (n_frames, n_kp, 3)
        # Upstream inference currently writes this as 'keypoints_world_xyz' in run_keypoints3d_inference
        if "keypoints_pos" in f:
            keypoints_pos = f["keypoints_pos"][:]
            keypoints_order = list(f["keypoints_pos"].attrs["keypoints"])
        elif "keypoints_world_xyz" in f:
            keypoints_pos = f["keypoints_world_xyz"][:]
            keypoints_order = list(f["keypoints_world_xyz"].attrs["keypoints"])
        else:
            raise RuntimeError("No 3D keypoints found in results HDF5")

        # 2D camera pixel coordinates (n_frames, n_kp, 2) - REQUIRED
        # These should be pixel coordinates in the original camera image used for inference.
        # Upstream inference writes this as 'keypoints_camera_xy'.
        if "keypoints_camera_xy" in f:
            keypoints_pos_2d = f["keypoints_camera_xy"][:].astype(np.float32)
            # Read the original image size used during inference (W, H) if present
            cam_img_size_attr = f["keypoints_camera_xy"].attrs.get("image_size", None)
            if cam_img_size_attr is not None:
                try:
                    camera_image_size = (
                        int(cam_img_size_attr[0]),
                        int(cam_img_size_attr[1]),
                    )
                except Exception:
                    camera_image_size = None
            else:
                camera_image_size = None
        elif "keypoints_pos_2d" in f:
            keypoints_pos_2d = f["keypoints_pos_2d"][:].astype(np.float32)
            camera_image_size = None
        elif "keypoints_2d" in f:
            keypoints_pos_2d = f["keypoints_2d"][:].astype(np.float32)
            camera_image_size = None
        else:
            # Explicitly require 2D camera coordinates for visualization overlays
            raise RuntimeError(
                "No 2D camera pixel coordinates (keypoints_camera_xy) found in results HDF5"
            )

    logger.info(f"Loaded {len(frame_ids)} frames of keypoint data")

    # Load input images and resize to resolution at which inference was run
    display_size = (256, 256)  # (W, H)
    img_shape = (display_size[1], display_size[0], 3)  # H, W, C

    # Determine coordinate ranges for proper axis scaling
    x_coords = keypoints_pos[:, :, 0].flatten()
    y_coords = keypoints_pos[:, :, 1].flatten()
    z_coords = keypoints_pos[:, :, 2].flatten()

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)

    # Add some padding to the ranges
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    z_padding = (z_max - z_min) * 0.1

    x_lim = (x_min - x_padding, x_max + x_padding)
    y_lim = (y_min - y_padding, y_max + y_padding)
    z_lim = (z_min - z_padding, z_max + z_padding)

    logger.info(f"3D plot limits: X={x_lim}, Y={y_lim}, Z={z_lim}")

    # Get skeleton connections once
    skeleton_connections = get_skeleton_connections(keypoints_order)

    # Create a temporary frames directory using tempfile
    with tempfile.TemporaryDirectory(prefix="keypoints3d_frames_") as temp_dir:
        frames_dir = Path(temp_dir)

        # Render frames in parallel
        # camera_image_size was set while reading the HDF5 (may be None)
        # it represents (W, H) of the original camera images used during inference
        render_frames_parallel(
            frames_dir=frames_dir,
            frame_ids=frame_ids,
            img_dir=input_images_dir,
            keypoints_pos=keypoints_pos,
            keypoints_order=keypoints_order,
            skeleton_connections=skeleton_connections,
            img_shape=img_shape,
            x_lim=x_lim,
            y_lim=y_lim,
            z_lim=z_lim,
            n_workers=n_workers,
            keypoints_pos_2d=keypoints_pos_2d,
            camera_image_size=camera_image_size,
        )

        # Convert frames to video
        logger.info("Converting frames to video...")
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if not frame_files:
            raise RuntimeError("No frame files were created")

        # Ensure consistent frame sizes (resize any mismatches) before writing video
        # Detect target size from first few frames
        frame_sizes = {}
        target_size = None
        for i, frame_file in enumerate(frame_files[:5]):
            frame = imageio.imread(frame_file)
            size = frame.shape[:2]  # (height, width)
            frame_sizes[i] = size
            if target_size is None:
                target_size = size
            elif size != target_size:
                logger.warning(
                    f"Frame size inconsistency detected: frame {i} has size {size}, expected {target_size}"
                )

        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            str(output_video_path),
            "ffmpeg",
            fps=fps,
            codec="libx264",
            quality=None,
            ffmpeg_params=default_video_writing_ffmpeg_params,
        ) as video_writer:
            for frame_file in frame_files:
                frame = imageio.imread(frame_file)
                # Resize if needed to match target
                if target_size and frame.shape[:2] != target_size:
                    pil_frame = Image.fromarray(frame)
                    pil_frame = pil_frame.resize(
                        (target_size[1], target_size[0]), Image.Resampling.LANCZOS
                    )
                    frame = np.array(pil_frame)
                video_writer.append_data(frame)

        # Temporary directory and all frames are automatically cleaned up when exiting the context

    logger.info(f"Visualization video saved to: {output_video_path}")


if __name__ == "__main__":
    input_basedir = Path("bulk_data/behavior_images/spotlight_aligned_and_cropped/")
    model_dir = Path("bulk_data/pose_estimation/keypoints3d/trial_20251007a")
    epochs_to_try = list(range(0, 21, 2))
    recordings = ["20250613-fly1b-013"]

    for recording in recordings:
        for epoch in epochs_to_try:
            print(
                f"Creating visualization for recording {recording} "
                f"using model at the end of epoch {epoch}"
            )
            recording_dir = input_basedir / recording / "model_prediction/not_flipped"
            keypoints3d_data_dir = model_dir / f"production/epoch{epoch}" / recording
            inference_output_path = keypoints3d_data_dir / "keypoints3d.h5"
            output_video_path = keypoints3d_data_dir / "predictions.mp4"
            visualize_predictions(
                inference_output_path,
                recording_dir,
                output_video_path,
                fps=30,
                n_workers=-2,
            )
            print(f"Visualization complete for recording {recording}, epoch {epoch}.")
