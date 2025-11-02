import matplotlib

matplotlib.use("Agg")

import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import logging
import tempfile
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from joblib import Parallel, delayed
from pvio.video_io import write_frames_to_video

from poseforge.pose.keypoints3d.invkin import run_seqikpy, save_seqikpy_output
from poseforge.pose.keypoints3d.visualizer import (
    get_keypoint_color,
    get_skeleton_connections,
)
from poseforge.neuromechfly.constants import kchain_plotting_colors
from poseforge.util import configure_matplotlib_style

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure matplotlib style
configure_matplotlib_style()

# Animation constants
CAMERA_PAN_PERIOD = 300.0  # number of frames during a full camera pan cycle
ELEVATION_ANGLE = 30.0
AZIMUTH_AMPLITUDE = 30.0


def _convert_fwdkin_to_canonical_format(
    forward_kinematics_world_xyz: np.ndarray,
    legs: list[str],
    leg_keypoints_canonical: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Convert forward kinematics data to canonical format matching original keypoints.

    Args:
        forward_kinematics_world_xyz: Shape (n_frames, 6, 5, 3) - 6 legs, 5 keypoints per leg
        legs: List of leg names ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
        leg_keypoints_canonical: List of keypoint names per leg ['ThC', 'CTr', 'FTi', 'TiTa', 'Claw']

    Returns:
        keypoints_pos: Shape (n_frames, n_keypoints, 3) where n_keypoints = 6*5 + 2 (antennae)
        keypoints_order: List of keypoint names in canonical format
    """
    n_frames, n_legs, n_keypoints_per_leg, _ = forward_kinematics_world_xyz.shape

    # Create keypoint names in canonical format for legs
    keypoints_order = []
    for leg in legs:
        for keypoint in leg_keypoints_canonical:
            keypoints_order.append(f"{leg}{keypoint}")

    # Add antenna keypoints (these won't be present in forward kinematics but needed for consistency)
    keypoints_order.extend(["LPedicel", "RPedicel"])

    # Reshape forward kinematics to (n_frames, n_keypoints, 3)
    n_leg_keypoints = n_legs * n_keypoints_per_leg
    keypoints_pos = np.full(
        (n_frames, len(keypoints_order), 3), np.nan, dtype=np.float32
    )

    # Fill in leg keypoints
    leg_keypoints_flat = forward_kinematics_world_xyz.reshape(
        n_frames, n_leg_keypoints, 3
    )
    keypoints_pos[:, :n_leg_keypoints, :] = leg_keypoints_flat

    return keypoints_pos, keypoints_order


def _align_constrained_poses_to_raw_poses(
    keypoints_pos_raw: np.ndarray,
    keypoints_pos_constrained: np.ndarray,
    keypoints_order: list[str],
    legs: list[str],
    leg_keypoints_canonical: list[str],
) -> np.ndarray:
    """Align constrained poses to raw poses by shifting each leg's kinematic chain.

    The inverse kinematics process aligns each leg to a template position. For visualization,
    we want to shift each leg back so that the first keypoint (ThC/Coxa) has the same 3D
    position as in the raw poses.

    Args:
        keypoints_pos_raw: Raw keypoint positions (n_frames, n_keypoints, 3)
        keypoints_pos_constrained: Constrained keypoint positions (n_frames, n_keypoints, 3)
        keypoints_order: List of keypoint names
        legs: List of leg names ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
        leg_keypoints_canonical: List of keypoint names per leg ['ThC', 'CTr', 'FTi', 'TiTa', 'Claw']

    Returns:
        keypoints_pos_constrained_aligned: Aligned constrained poses
    """
    keypoints_pos_constrained_aligned = keypoints_pos_constrained.copy()
    n_frames = keypoints_pos_raw.shape[0]

    # For each leg, align the constrained pose to the raw pose
    for leg in legs:
        # Get the first keypoint (ThC/Coxa) for this leg
        first_keypoint_name = f"{leg}{leg_keypoints_canonical[0]}"  # e.g., "LFThC"

        try:
            first_keypoint_idx = keypoints_order.index(first_keypoint_name)
        except ValueError:
            logger.warning(
                f"Keypoint {first_keypoint_name} not found in keypoints_order"
            )
            continue

        # Get all keypoint indices for this leg
        leg_keypoint_indices = []
        for keypoint in leg_keypoints_canonical:
            keypoint_name = f"{leg}{keypoint}"
            try:
                idx = keypoints_order.index(keypoint_name)
                leg_keypoint_indices.append(idx)
            except ValueError:
                logger.warning(f"Keypoint {keypoint_name} not found in keypoints_order")
                continue

        if not leg_keypoint_indices:
            continue

        # For each frame, compute the translation needed to align the first keypoint
        for frame_idx in range(n_frames):
            # Get the positions of the first keypoint in raw and constrained poses
            raw_first_pos = keypoints_pos_raw[frame_idx, first_keypoint_idx]
            constrained_first_pos = keypoints_pos_constrained[
                frame_idx, first_keypoint_idx
            ]

            # Skip if either position has NaN values
            if np.isnan(raw_first_pos).any() or np.isnan(constrained_first_pos).any():
                continue

            # Compute translation vector
            translation = raw_first_pos - constrained_first_pos

            # Apply translation to all keypoints of this leg
            for leg_kp_idx in leg_keypoint_indices:
                current_pos = keypoints_pos_constrained_aligned[frame_idx, leg_kp_idx]
                if not np.isnan(current_pos).any():
                    keypoints_pos_constrained_aligned[frame_idx, leg_kp_idx] = (
                        current_pos + translation
                    )

    return keypoints_pos_constrained_aligned


def setup_comparison_figure(
    img_shape, x_lim, y_lim, z_lim, keypoints_order, skeleton_connections
):
    """Setup figure with two panels: left for image, right for 3D skeleton comparison"""
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
    # Only show raw pose for 2D
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

        # Raw pose lines (colored)
        (line2d,) = ax_img.plot([0, 0], [0, 0], color=color, linewidth=3)
        two_d_lines.append(line2d)

    # Create scatter artists only for antennae and non-leg keypoints
    for kp_idx, keypoint_name in enumerate(keypoints_order):
        if _is_leg(keypoint_name):
            # legs: no scatter on 2D overlay
            continue
        color = get_keypoint_color(keypoint_name)

        # Raw pose scatters (colored)
        sc = ax_img.scatter([], [], color=color, s=15)
        two_d_scatters[kp_idx] = sc

    # Setup right panel for 3D skeleton
    ax_3d.remove()  # Remove the 2D axes
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    ax_3d.set_xlabel("X (mm)")
    ax_3d.set_ylabel("Y (mm)")
    ax_3d.set_zlabel("Z (mm)")
    ax_3d.set_title("3D Skeleton", fontweight="bold")
    # Use the fixed axis limits from original visualization
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

    return (
        fig,
        ax_img,
        ax_3d,
        img_plot,
        two_d_lines,
        two_d_scatters,
    )


def render_comparison_frame_chunk_worker(
    frame_indices,
    frames_dir,
    frame_ids,
    img_dir,
    keypoints_pos_raw,
    keypoints_pos_constrained,
    keypoints_order,
    skeleton_connections,
    img_shape,
    x_lim,
    y_lim,
    z_lim,
    keypoints_pos_2d_raw=None,
    keypoints_pos_2d_constrained=None,
    camera_image_size: tuple[int, int] | None = None,
):
    """Worker function to render a chunk of frames comparing raw and constrained poses"""
    # Setup figure once for this worker
    (
        fig,
        ax_img,
        ax_3d,
        img_plot,
        two_d_lines,
        two_d_scatters,
    ) = setup_comparison_figure(
        img_shape, x_lim, y_lim, z_lim, keypoints_order, skeleton_connections
    )

    # Pre-allocate line and scatter plot objects for 3D skeleton
    skeleton_lines_raw = []
    skeleton_lines_constrained = []
    keypoint_scatters_raw = [None] * len(keypoints_order)
    keypoint_scatters_constrained = [None] * len(keypoints_order)

    # Initialize with dummy data for 3D lines
    for connection in skeleton_connections:
        start_idx, end_idx = connection
        start_name = keypoints_order[start_idx]
        leg_id = start_name[:2] if len(start_name) >= 2 else None
        color = kchain_plotting_colors.get(leg_id, np.array([0.5, 0.5, 0.5]))

        # Raw pose lines (colored)
        (line_raw,) = ax_3d.plot3D([0, 0], [0, 0], [0, 0], color=color, linewidth=3)
        skeleton_lines_raw.append(line_raw)

        # Constrained pose lines (black, on top)
        (line_constrained,) = ax_3d.plot3D(
            [0, 0], [0, 0], [0, 0], color="black", linewidth=1
        )
        skeleton_lines_constrained.append(line_constrained)

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
            keypoint_scatters_raw[kp_idx] = None
            keypoint_scatters_constrained[kp_idx] = None
        else:
            # Raw pose scatters (colored, not gray)
            scatter_raw = ax_3d.scatter([0], [0], [0], color=color, s=30)
            keypoint_scatters_raw[kp_idx] = scatter_raw

            # Constrained pose scatters (black)
            scatter_constrained = ax_3d.scatter([0], [0], [0], color="black", s=50)
            keypoint_scatters_constrained[kp_idx] = scatter_constrained

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
        keypoints_3d_raw = keypoints_pos_raw[i]  # (n_keypoints, 3)
        keypoints_3d_constrained = keypoints_pos_constrained[i]  # (n_keypoints, 3)

        # Update 3D skeleton lines for raw pose
        for line_idx, connection in enumerate(skeleton_connections):
            start_idx, end_idx = connection
            start_pos = keypoints_3d_raw[start_idx]
            end_pos = keypoints_3d_raw[end_idx]
            if not (np.isnan(start_pos).any() or np.isnan(end_pos).any()):
                skeleton_lines_raw[line_idx].set_data_3d(
                    [start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    [start_pos[2], end_pos[2]],
                )
            else:
                skeleton_lines_raw[line_idx].set_data_3d([np.nan], [np.nan], [np.nan])

        # Update 3D skeleton lines for constrained pose
        for line_idx, connection in enumerate(skeleton_connections):
            start_idx, end_idx = connection
            start_pos = keypoints_3d_constrained[start_idx]
            end_pos = keypoints_3d_constrained[end_idx]
            if not (np.isnan(start_pos).any() or np.isnan(end_pos).any()):
                skeleton_lines_constrained[line_idx].set_data_3d(
                    [start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    [start_pos[2], end_pos[2]],
                )
            else:
                skeleton_lines_constrained[line_idx].set_data_3d(
                    [np.nan], [np.nan], [np.nan]
                )

        # Update 3D keypoint scatters for raw pose (skip None entries for leg keypoints)
        for kp_idx, scatter in enumerate(keypoint_scatters_raw):
            if scatter is None:
                continue
            pos = keypoints_3d_raw[kp_idx]
            if not np.isnan(pos).any():
                scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
            else:
                scatter._offsets3d = ([np.nan], [np.nan], [np.nan])

        # Update 3D keypoint scatters for constrained pose (skip None entries for leg keypoints)
        for kp_idx, scatter in enumerate(keypoint_scatters_constrained):
            if scatter is None:
                continue
            pos = keypoints_3d_constrained[kp_idx]
            if not np.isnan(pos).any():
                scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
            else:
                scatter._offsets3d = ([np.nan], [np.nan], [np.nan])

        # Update 3D plot title and camera view
        ax_3d.set_title("Raw (colored) vs constrained (black) pose")
        azimuth = np.cos(2 * np.pi * i / CAMERA_PAN_PERIOD) * AZIMUTH_AMPLITUDE
        ax_3d.view_init(elev=ELEVATION_ANGLE, azim=azimuth)

        # Update 2D overlay if provided (for both raw and constrained poses)
        def update_2d_overlay(keypoints_2d_frame, lines, scatters, alpha=1.0):
            if keypoints_2d_frame is None:
                return

            keypoints_2d_frame = keypoints_2d_frame.astype(np.float32)
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

            # Update 2D lines
            for line_idx, connection in enumerate(skeleton_connections):
                s, e = connection
                sxy = keypoints_2d_frame[s]
                exy = keypoints_2d_frame[e]
                lines[line_idx].set_data([sxy[0], exy[0]], [sxy[1], exy[1]])
                lines[line_idx].set_alpha(alpha)

            # Update 2D scatters (antennae/others)
            for kp_idx, sc in scatters.items():
                xy = keypoints_2d_frame[kp_idx]
                sc.set_offsets([xy[0], xy[1]])
                sc.set_alpha(alpha)

        # Update raw 2D overlay only
        if keypoints_pos_2d_raw is not None:
            update_2d_overlay(
                keypoints_pos_2d_raw[i], two_d_lines, two_d_scatters, alpha=0.7
            )

        # Save frame
        frame_path = frames_dir / f"frame_{i:06d}.png"
        # Use no tight bbox and a small pad to keep titles/labels from being cut
        fig.savefig(frame_path, dpi=100, bbox_inches=None, pad_inches=0.1)

    plt.close(fig)


def render_comparison_frames_parallel(
    frames_dir,
    frame_ids,
    img_dir,
    keypoints_pos_raw,
    keypoints_pos_constrained,
    keypoints_order,
    skeleton_connections,
    img_shape,
    x_lim,
    y_lim,
    z_lim,
    n_workers=-2,
    keypoints_pos_2d_raw=None,
    keypoints_pos_2d_constrained=None,
    camera_image_size: tuple[int, int] | None = None,
):
    """Render frames in parallel using joblib for pose comparison visualization"""
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

    logger.info(f"Rendering {n_frames} comparison frames using {n_workers} workers")

    # Split frame indices into chunks for workers
    frame_indices = list(range(n_frames))
    chunk_size = max(1, n_frames // n_workers)
    frame_chunks = [
        frame_indices[i : i + chunk_size] for i in range(0, n_frames, chunk_size)
    ]

    # Render chunks in parallel
    Parallel(n_jobs=n_workers)(
        delayed(render_comparison_frame_chunk_worker)(
            chunk,
            frames_dir,
            frame_ids,
            img_dir,
            keypoints_pos_raw,
            keypoints_pos_constrained,
            keypoints_order,
            skeleton_connections,
            img_shape,
            x_lim,
            y_lim,
            z_lim,
            keypoints_pos_2d_raw,
            keypoints_pos_2d_constrained,
            camera_image_size,
        )
        for chunk in frame_chunks
    )


def visualize_inverse_kinematics_comparison(
    inference_output_path: Path,
    inverse_kinematics_path: Path,
    input_images_dir: Path,
    output_video_path: Path,
    fps: int = 30,
    n_workers: int = -2,
    max_n_frames: int | None = None,
):
    """Create a two-panel video comparing raw and constrained 3D poses

    Args:
        inference_output_path: Path to the original keypoints3d.h5 inference results
        inverse_kinematics_path: Path to the inverse_kinematics.h5 file
        input_images_dir: Directory containing original image data
        output_video_path: Path where the output video will be saved
        fps: Frames per second for output video
        n_workers: Number of parallel workers (-2 = all cores except 1)
        max_n_frames: Maximum number of frames to process (for debugging)
    """
    logger.info(f"Creating IK comparison visualization for: {inference_output_path}")

    # Load original inference results (raw poses)
    with h5py.File(inference_output_path, "r") as f:
        frame_ids = f["frame_ids"][:]
        # Raw 3D world coordinates (n_frames, n_kp, 3)
        if "keypoints_pos" in f:
            keypoints_pos_raw = f["keypoints_pos"][:]
            keypoints_order = list(f["keypoints_pos"].attrs["keypoints"])
        elif "keypoints_world_xyz" in f:
            keypoints_pos_raw = f["keypoints_world_xyz"][:]
            keypoints_order = list(f["keypoints_world_xyz"].attrs["keypoints"])
        else:
            raise RuntimeError("No 3D keypoints found in results HDF5")

        # 2D camera pixel coordinates for raw poses
        if "keypoints_camera_xy" in f:
            keypoints_pos_2d_raw = f["keypoints_camera_xy"][:].astype(np.float32)
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
        else:
            keypoints_pos_2d_raw = None
            camera_image_size = None

    # Load inverse kinematics results (constrained poses)
    with h5py.File(inverse_kinematics_path, "r") as f:
        forward_kinematics_world_xyz = f["forward_kinematics_world_xyz"][:]
        legs_ik = list(f["forward_kinematics_world_xyz"].attrs["legs"])
        leg_keypoints_canonical_ik = list(
            f["forward_kinematics_world_xyz"].attrs["keypoint_names_per_leg"]
        )
        # Get frame_ids from IK file to ensure consistency
        ik_frame_ids = f["frame_ids"][:]

    # Convert forward kinematics to canonical format
    keypoints_pos_constrained, keypoints_order_constrained = (
        _convert_fwdkin_to_canonical_format(
            forward_kinematics_world_xyz, legs_ik, leg_keypoints_canonical_ik
        )
    )

    # Ensure we only process frames that exist in both datasets
    # The IK processing may have been limited by max_n_frames
    n_frames_ik = len(ik_frame_ids)
    if max_n_frames is not None:
        n_frames_to_process = min(max_n_frames, n_frames_ik, len(frame_ids))
    else:
        n_frames_to_process = min(n_frames_ik, len(frame_ids))

    logger.info(f"Processing {n_frames_to_process} frames for visualization")

    # Slice all data to match the number of frames we can actually process
    frame_ids = frame_ids[:n_frames_to_process]
    keypoints_pos_raw = keypoints_pos_raw[:n_frames_to_process]
    keypoints_pos_constrained = keypoints_pos_constrained[:n_frames_to_process]

    if keypoints_pos_2d_raw is not None:
        keypoints_pos_2d_raw = keypoints_pos_2d_raw[:n_frames_to_process]

    # Align constrained poses to raw poses for visualization
    # This shifts each leg's kinematic chain so that the first keypoint (ThC/Coxa)
    # matches the position in the raw poses
    logger.info("Aligning constrained poses to raw poses for visualization...")
    keypoints_pos_constrained = _align_constrained_poses_to_raw_poses(
        keypoints_pos_raw=keypoints_pos_raw,
        keypoints_pos_constrained=keypoints_pos_constrained,
        keypoints_order=keypoints_order,
        legs=legs_ik,
        leg_keypoints_canonical=leg_keypoints_canonical_ik,
    )

    # For now, we don't have 2D projections of the constrained poses, so set to None
    keypoints_pos_2d_constrained = None

    logger.info(f"Loaded {len(frame_ids)} frames of keypoint data")

    # Load input images and resize to resolution at which inference was run
    display_size = (256, 256)  # (W, H)
    img_shape = (display_size[1], display_size[0], 3)  # H, W, C

    # Use fixed coordinate limits matching the original visualization
    x_lim = (0.5, 3.5)
    y_lim = (-3.5, -0.5)
    z_lim = (-0.5, 2.0)
    logger.info(f"3D plot limits: X={x_lim}, Y={y_lim}, Z={z_lim}")

    # Get skeleton connections once
    skeleton_connections = get_skeleton_connections(keypoints_order)

    # Create a temporary frames directory using tempfile
    with tempfile.TemporaryDirectory(
        prefix="keypoints3d_ik_comparison_frames_"
    ) as temp_dir:
        frames_dir = Path(temp_dir)

        # Render frames in parallel
        render_comparison_frames_parallel(
            frames_dir=frames_dir,
            frame_ids=frame_ids,
            img_dir=input_images_dir,
            keypoints_pos_raw=keypoints_pos_raw,
            keypoints_pos_constrained=keypoints_pos_constrained,
            keypoints_order=keypoints_order,
            skeleton_connections=skeleton_connections,
            img_shape=img_shape,
            x_lim=x_lim,
            y_lim=y_lim,
            z_lim=z_lim,
            n_workers=n_workers,
            keypoints_pos_2d_raw=keypoints_pos_2d_raw,
            keypoints_pos_2d_constrained=keypoints_pos_2d_constrained,
            camera_image_size=camera_image_size,
        )

        # Convert frames to video
        logger.info("Converting frames to video...")
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if not frame_files:
            raise RuntimeError("No frame files were created")

        # Load all frames into memory for write_frames_to_video
        frames = []
        for frame_file in frame_files:
            frame = imageio.imread(frame_file)
            frames.append(frame)

        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        write_frames_to_video(output_video_path, frames, fps=fps)

        # Temporary directory and all frames are automatically cleaned up when exiting the context

    logger.info(f"IK comparison visualization video saved to: {output_video_path}")


def process_all(
    input_dirs: list[str],
    max_n_frames: int | None = None,
    n_workers_per_dataset: int = 6,
    create_visualization: bool = False,
    input_images_basedir: str | None = None,
) -> None:
    # Index all keypoints3d output files to process
    all_keypoints3d_output_files = []
    for input_dir in input_dirs:
        keypoints3d_output_files = list(Path(input_dir).rglob("keypoints3d.h5"))
        all_keypoints3d_output_files.extend(keypoints3d_output_files)
    all_keypoints3d_output_files = sorted(all_keypoints3d_output_files)

    # Run inverse kinematics
    for keypoints3d_output_file in tqdm(all_keypoints3d_output_files):
        with h5py.File(keypoints3d_output_file, "r") as f:
            frame_ids = f["frame_ids"][:]
            world_xyz = f["keypoints_world_xyz"][:]
            keypoint_names_canonical = f["keypoints_world_xyz"].attrs["keypoints"]
        output_path = keypoints3d_output_file.parent / "inverse_kinematics.h5"
        joint_angles, forward_kinematics = run_seqikpy(
            world_xyz=world_xyz,
            keypoint_names_canonical=keypoint_names_canonical,
            max_n_frames=max_n_frames,
            n_workers=n_workers_per_dataset,
            debug_plots_dir=keypoints3d_output_file.parent / "ik_debug_plots/",
        )
        save_seqikpy_output(
            output_path, joint_angles, forward_kinematics, frame_ids=frame_ids
        )

        # Create visualization if requested
        if create_visualization:
            if input_images_basedir is None:
                raise ValueError(
                    "input_images_basedir must be provided to create visualizations"
                )
            logger.info(f"Creating visualization for {keypoints3d_output_file}")

            # Try to infer the input images directory structure
            # Assuming structure like: .../epoch{N}/recording_name/keypoints3d.h5
            # And images are in: input_images_basedir/recording_name/model_prediction/not_flipped/
            # Extract recording name from path
            path_parts = keypoints3d_output_file.parts
            recording_name = None
            for i, part in enumerate(path_parts):
                if part.startswith("epoch"):
                    if i + 1 < len(path_parts):
                        recording_name = path_parts[i + 1]
                        break

            if recording_name is None:
                logger.warning(
                    f"Could not infer recording name from path: {keypoints3d_output_file}"
                )
                continue

            input_images_dir = (
                Path(input_images_basedir)
                / recording_name
                / "model_prediction"
                / "not_flipped"
            )
            if not input_images_dir.exists():
                logger.warning(f"Input images directory not found: {input_images_dir}")
                continue

            output_video_path = keypoints3d_output_file.parent / "ik_comparison.mp4"

            visualize_inverse_kinematics_comparison(
                inference_output_path=keypoints3d_output_file,
                inverse_kinematics_path=output_path,
                input_images_dir=input_images_dir,
                output_video_path=output_video_path,
                fps=30,
                n_workers=-2,
                max_n_frames=max_n_frames,
            )
            logger.info(f"Visualization saved to: {output_video_path}")


def create_ik_visualization_for_trial(
    trial_name: str,
    model_basedir: str,
    input_images_basedir: str,
    epoch: int,
    fps: int = 30,
    n_workers: int = -2,
    max_n_frames: int | None = None,
) -> None:
    """Create inverse kinematics comparison visualization for a specific trial

    Args:
        trial_name: Name of the trial (e.g., "20250613-fly1b-013")
        model_basedir: Base directory containing model outputs (e.g., "bulk_data/pose_estimation/keypoints3d/trial_20251013b")
        input_images_basedir: Base directory containing input images (e.g., "bulk_data/behavior_images/spotlight_aligned_and_cropped/")
        epoch: Epoch number to use for visualization
        fps: Frames per second for output video
        n_workers: Number of parallel workers
        max_n_frames: Maximum number of frames to process (for debugging)
    """
    model_basedir = Path(model_basedir)
    input_images_basedir = Path(input_images_basedir)

    # Construct paths
    keypoints3d_output_file = (
        model_basedir / f"production/epoch{epoch}" / trial_name / "keypoints3d.h5"
    )
    inverse_kinematics_file = (
        model_basedir
        / f"production/epoch{epoch}"
        / trial_name
        / "inverse_kinematics.h5"
    )
    input_images_dir = (
        input_images_basedir / trial_name / "model_prediction" / "not_flipped"
    )
    output_video_path = (
        model_basedir / f"production/epoch{epoch}" / trial_name / "ik_comparison.mp4"
    )

    # Check if files exist
    if not keypoints3d_output_file.exists():
        raise FileNotFoundError(
            f"Keypoints3D file not found: {keypoints3d_output_file}"
        )
    if not inverse_kinematics_file.exists():
        raise FileNotFoundError(
            f"Inverse kinematics file not found: {inverse_kinematics_file}"
        )
    if not input_images_dir.exists():
        raise FileNotFoundError(f"Input images directory not found: {input_images_dir}")

    logger.info(
        f"Creating IK comparison visualization for trial: {trial_name}, epoch: {epoch}"
    )

    visualize_inverse_kinematics_comparison(
        inference_output_path=keypoints3d_output_file,
        inverse_kinematics_path=inverse_kinematics_file,
        input_images_dir=input_images_dir,
        output_video_path=output_video_path,
        fps=fps,
        n_workers=n_workers,
        max_n_frames=max_n_frames,
    )

    print(f"IK comparison visualization saved to: {output_video_path}")


if __name__ == "__main__":
    import tyro
    from poseforge.util.sys import get_hardware_availability

    # * Processing by CLI
    # tyro.cli(
    #     process_all,
    #     prog=f"python {Path(__file__).name}",
    #     description="Run inverse kinematics on all keypoints3d output files in the given directories.",
    # )
    # Example:
    # python src/poseforge/pose/keypoints3d/scripts/run_inverse_kinematics.py \
    #     --input-dirs bulk_data/pose_estimation/keypoints3d/trial_20251013b/production/epoch14_step9167/20250613-fly1b-005/ \
    #     --create-visualization \
    #     --input-images-basedir bulk_data/behavior_images/spotlight_aligned_and_cropped/

    # * Processing from this script directly
    epoch = 14  # these must be consistent with run_keypoints3d_inference.py
    step = 9167  # same as above
    production_model_basedir = Path(
        f"bulk_data/pose_estimation/keypoints3d/trial_20251013b/production/epoch{epoch}_step{step}/"
    )
    input_images_basedir = Path(
        "bulk_data/behavior_images/spotlight_aligned_and_cropped/"
    )
    input_dirs = sorted(list(production_model_basedir.glob("20250613-fly1b-*/")))
    print(f"Found {len(input_dirs)} directories to process")

    # Process directories in parallel. Note that each task is parallelized internally
    # among 6 legs, so the theoretical optimal max number of top-level workers is
    # (n_cpu_cores // 6). However, actual CPU utilization is low, so we can further
    # multiply this by a tasks_to_core_ratio factor.
    avail = get_hardware_availability()
    n_cores_available = avail["num_cpu_cores_available"]
    tasks_to_core_ratio = 4
    n_workers_top_level = tasks_to_core_ratio * (n_cores_available // 6)
    n_workers_top_level = max(1, min(n_workers_top_level, len(input_dirs)))

    Parallel(n_jobs=n_workers_top_level, verbose=1)(
        delayed(process_all)(
            input_dirs=[input_dir],
            max_n_frames=None,
            n_workers_per_dataset=6,
            create_visualization=True,
            input_images_basedir=str(input_images_basedir),
        )
        for input_dir in input_dirs
    )

    print("All directories processed!")

    # # Smaller example
    # trials_basedir = Path(
    #     "bulk_data/pose_estimation/keypoints3d/trial_20251013b/production/epoch14_step9167/"
    # )
    # input_images_basedir = Path(
    #     "bulk_data/behavior_images/spotlight_aligned_and_cropped/"
    # )
    # trial_dirs = sorted(list(trials_basedir.glob("20250613-fly1b-002/")))
    # process_all(
    #     input_dirs=trial_dirs,
    #     max_n_frames=100,
    #     n_workers_per_dataset=6,
    #     create_visualization=True,
    #     input_images_basedir=input_images_basedir,
    # )
