import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import logging
import cmasher
import imageio.v2 as imageio
import time
from PIL import Image
from pathlib import Path
from joblib import Parallel, delayed

from biomechpose.util import (
    configure_matplotlib_style,
    default_video_writing_ffmpeg_params,
    read_frames_from_video,
)
from biomechpose.simulate_nmf.utils import kchain_plotting_colors


configure_matplotlib_style()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Constants
AZIMUTH_ANIMATION_PERIOD = 300.0  # frames for one full rotation
ELEVATION_ANGLE = 30.0
AZIMUTH_AMPLITUDE = 30.0


# Helper functions for parallel processing (must be pickle-able)
def get_keypoint_color(keypoint_name: str) -> np.ndarray:
    """Get color for a keypoint based on kchain_plotting_colors dictionary"""
    if keypoint_name in ["LPedicel", "RPedicel"]:
        # Map antenna pedicels to antenna colors
        antenna_key = "LAntenna" if keypoint_name.startswith("L") else "RAntenna"
        return kchain_plotting_colors[antenna_key]
    else:
        # Use first two characters (leg identifier)
        leg_key = keypoint_name[:2]
        return kchain_plotting_colors.get(leg_key, np.array([0.5, 0.5, 0.5]))  # Default gray


def get_skeleton_connections(keypoints_order: list[str]) -> list[tuple[int, int]]:
    """Define skeleton connections between keypoints on the same leg (excluding antennae)"""
    connections = []
    
    # Group keypoints by leg (first two characters) - exclude last 2 antennae
    leg_groups = {}
    leg_keypoints = keypoints_order[:-2]  # Exclude last 2 antennae keypoints
    
    for i, keypoint_name in enumerate(leg_keypoints):
        leg_id = keypoint_name[:2]  # First two characters identify the leg
        if leg_id not in leg_groups:
            leg_groups[leg_id] = []
        leg_groups[leg_id].append(i)
    
    # Connect consecutive keypoints within each leg
    for leg_id, keypoint_indices in leg_groups.items():
        # Sort indices to ensure proper proximal-to-distal order
        keypoint_indices.sort()
        # Connect consecutive keypoints in this leg
        for i in range(len(keypoint_indices) - 1):
            connections.append((keypoint_indices[i], keypoint_indices[i + 1]))
    
    return connections



def update_heatmap_data(ax, frame_idx: int, variant_idx: int, pred_xy_heatmaps: np.ndarray,
                        label_xy: np.ndarray, stride_x: int, stride_y: int, 
                        n_keypoints: int, marker_size: int, cmap):
    """Update heatmap data only - preserves axes formatting from setup"""
    # Clear only the plotted data, not the axes formatting
    for artist in ax.get_children():
        if hasattr(artist, 'remove') and artist.__class__.__name__ in ['AxesImage', 'PathCollection']:
            artist.remove()
    
    # Get heatmaps for this frame and variant
    heatmaps = pred_xy_heatmaps[variant_idx, frame_idx]  # (n_keypoints, H, W)

    # Convert each keypoint's heatmap to probabilities independently
    heatmaps_prob = np.zeros_like(heatmaps)
    for kp_idx in range(n_keypoints):
        heatmap_flat = heatmaps[kp_idx].flatten()
        heatmap_flat_shifted = heatmap_flat - np.max(heatmap_flat)
        probs_flat = np.exp(heatmap_flat_shifted) / np.sum(np.exp(heatmap_flat_shifted))
        heatmaps_prob[kp_idx] = probs_flat.reshape(heatmaps[kp_idx].shape)

    # Merge all keypoint heatmaps (take maximum across keypoints)
    merged_heatmap_prob = np.max(heatmaps_prob, axis=0)  # (H, W)
    ax.imshow(merged_heatmap_prob, cmap=cmap, vmin=0, vmax=0.04)

    # Plot label points
    for kp_idx in range(n_keypoints):
        label_x = label_xy[frame_idx, kp_idx, 0] / stride_x
        label_y = label_xy[frame_idx, kp_idx, 1] / stride_y
        ax.scatter(label_x, label_y, color="white", s=marker_size, 
                  edgecolors="black", linewidth=1)


def update_depth_data(ax, frame_idx: int, variant_idx: int, pred_depth_logits: np.ndarray,
                      label_depth: np.ndarray, depth_min: float, depth_max: float, 
                      depth_n_bins: int, n_keypoints: int, marker_size: int, cmap):
    """Update depth data only - preserves axes formatting from setup"""
    # Clear only the plotted data, not the axes formatting
    for artist in ax.get_children():
        if hasattr(artist, 'remove') and artist.__class__.__name__ in ['AxesImage', 'PathCollection']:
            artist.remove()
    
    # Get depth logits for this frame and variant
    depth_logits = pred_depth_logits[variant_idx, frame_idx]  # (n_keypoints, depth_bins)
    depth_logits_shifted = depth_logits - np.max(depth_logits, axis=1, keepdims=True)
    depth_probs = np.exp(depth_logits_shifted) / np.sum(np.exp(depth_logits_shifted), axis=1, keepdims=True)

    ax.imshow(depth_probs.T, cmap=cmap, aspect='auto', vmin=0, vmax=0.5)
    
    # Add ground truth depth as white dots
    depth_bin_centers = np.linspace(depth_min, depth_max, depth_n_bins)
    for kp_idx in range(n_keypoints):
        label_depth_val = label_depth[frame_idx, kp_idx]
        depth_bin = np.argmin(np.abs(depth_bin_centers - label_depth_val))
        ax.scatter(kp_idx, depth_bin, color='white', s=marker_size, marker='o', 
                  edgecolors='black', linewidth=1)


def update_3d_skeleton_data(ax, frame_idx: int, variant_idx: int, pred_world_xyz: np.ndarray,
                           label_world_xyz: np.ndarray, keypoints_order: list[str],
                           n_keypoints: int, marker_size: int):
    """Update 3D skeleton data only - preserves axes formatting from setup"""
    # Clear only the plotted data, not the axes formatting
    for artist in ax.get_children():
        if hasattr(artist, 'remove') and artist.__class__.__name__ in ['Line3D', 'Path3DCollection']:
            artist.remove()
    
    skeleton_connections = get_skeleton_connections(keypoints_order)

    # Plot ground truth
    gt_points = label_world_xyz[frame_idx]
    for connection in skeleton_connections:
        if connection[0] < n_keypoints and connection[1] < n_keypoints:
            start_point = gt_points[connection[0]]
            end_point = gt_points[connection[1]]
            keypoint_name = keypoints_order[connection[0]]
            line_color = get_keypoint_color(keypoint_name)
            ax.plot3D([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                     [start_point[2], end_point[2]], color=line_color, linewidth=3)

    # Plot antennae keypoints
    if n_keypoints >= 2:
        for i in range(2):
            antenna_idx = n_keypoints - 2 + i
            antenna_point = gt_points[antenna_idx]
            keypoint_name = keypoints_order[antenna_idx]
            point_color = get_keypoint_color(keypoint_name)
            ax.scatter(antenna_point[0], antenna_point[1], antenna_point[2], 
                      color=point_color, s=marker_size)

    # Plot predictions
    if variant_idx is not None:
        pred_points = pred_world_xyz[variant_idx, frame_idx]
        for connection in skeleton_connections:
            if connection[0] < n_keypoints and connection[1] < n_keypoints:
                start_point = pred_points[connection[0]]
                end_point = pred_points[connection[1]]
                keypoint_name = keypoints_order[connection[0]]
                line_color = get_keypoint_color(keypoint_name)
                ax.plot3D([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                         [start_point[2], end_point[2]], color=line_color, linewidth=2)

        if n_keypoints >= 2:
            for i in range(2):
                antenna_idx = n_keypoints - 2 + i
                antenna_point = pred_points[antenna_idx]
                keypoint_name = keypoints_order[antenna_idx]
                point_color = get_keypoint_color(keypoint_name)
                ax.scatter(antenna_point[0], antenna_point[1], antenna_point[2], 
                          color=point_color, s=marker_size)

    # Set viewing angle
    azimuth = np.cos(2 * np.pi * frame_idx / AZIMUTH_ANIMATION_PERIOD) * AZIMUTH_AMPLITUDE
    ax.view_init(elev=ELEVATION_ANGLE, azim=azimuth)


def setup_figure_and_axes(
    n_cols: int,
    n_keypoints: int,
    depth_min: float,
    depth_max: float,
    depth_n_bins: int,
) -> tuple[plt.Figure, list, list, list, list, list]:
    """Setup figure and all axes with complete formatting - no need to repeat later"""
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(4, n_cols, height_ratios=[1, 1, 1, 1])  #, hspace=0.3)
    
    video_axes = []
    video_images = []  # Store image artists for updating
    heatmap_axes = []
    depth_axes = []
    skeleton_axes = []
    
    # Row 0: Video axes
    for col_idx in range(n_cols):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_xticks([])
        ax.set_yticks([])
        if col_idx == 0:
            ax.set_title("NeuroMechFly\nsimulation", fontweight="bold", pad=20)
            # Create placeholder image
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            im = ax.imshow(dummy_image)
        else:
            model_idx = col_idx - 1
            ax.set_title(f"Synthetic rendering\n(variant {model_idx})", fontweight="bold", pad=20)
            # Create placeholder image
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            im = ax.imshow(dummy_image)
        video_axes.append(ax)
        video_images.append(im)
    
    # Row 1: Heatmap axes with complete setup
    for col_idx in range(n_cols):
        ax = fig.add_subplot(gs[1, col_idx])
        if col_idx == 0:
            ax.axis("off")
            ax.text(0.6, 0.5, "2D pose", transform=ax.transAxes, rotation=90,
                   va='center', ha='center', fontsize=12, fontweight='bold', color='black')
        else:
            if col_idx == 1:
                ax.set_xlabel("column (px)")
                ax.set_ylabel("row (px)")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        heatmap_axes.append(ax)
    
    # Row 2: Depth axes with complete setup
    depth_bin_centers = np.linspace(depth_min, depth_max, depth_n_bins)
    for col_idx in range(n_cols):
        ax = fig.add_subplot(gs[2, col_idx])
        if col_idx == 0:
            ax.axis("off")
            ax.text(0.6, 0.5, "Distance from camera", transform=ax.transAxes, rotation=90,
                   va='center', ha='center', fontsize=12, fontweight='bold', color='black')
        else:
            # Complete depth plot setup
            for leg_end in range(5, n_keypoints, 5):
                ax.axvline(x=leg_end - 0.5, color='white', linewidth=1)
            
            n_legs = 6
            n_keypoints_per_leg = 5
            n_antennae = 2
            x_ticks = [
                (n_keypoints_per_leg * i) + (n_keypoints_per_leg / 2) - 0.5
                for i in range(n_legs)
            ] + [n_legs * n_keypoints_per_leg + (n_antennae / 2) - 0.5]
            k_ticklabels = [f"{side}{pos}" for side in "LR" for pos in "FMH"] + ["A"]
            ax.set_xticks(x_ticks)
            ax.set_xlim(-0.5, n_keypoints - 0.5)
            ax.set_ylim(depth_n_bins - 0.5, 0)
            
            if col_idx == 1:
                ax.set_xticklabels(k_ticklabels)
                ax.set_ylabel("depth (mm)")
                if depth_n_bins > 10:
                    desired_depth_values = np.linspace(depth_min, depth_max, 5)
                    fracs = (desired_depth_values - depth_min) / (depth_max - depth_min)
                    tick_indices = (fracs * (depth_n_bins - 1)).round().astype(int)
                    ax.set_yticks(tick_indices)
                    ax.set_yticklabels([f"{-d - 100}" for d in desired_depth_values], fontsize=8)
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        depth_axes.append(ax)
    
    # Row 3: 3D skeleton axes with complete setup
    for col_idx in range(n_cols):
        if col_idx == 0:
            ax = fig.add_subplot(gs[3, col_idx], projection="3d")
            ax.axis("off")
            ax.text2D(0.6, 0.5, "3D reconstruction", transform=ax.transAxes, rotation=90,
                     va='center', ha='center', fontsize=12, fontweight='bold', color='black')
        else:
            ax = fig.add_subplot(gs[3, col_idx], projection="3d")
            variant_idx = col_idx - 1
            # Complete 3D setup
            ax.set_xlim(0.5, 3.5)
            ax.set_ylim(-3.5, -0.5)
            ax.set_zlim(-0.5, 2.0)
            ax.set_box_aspect([1, 1, 1])
            
            if variant_idx == 0:
                ax.set_xlabel("x (mm)", labelpad=-8)
                ax.set_ylabel("y (mm)", labelpad=-8)
                ax.set_zlabel("z (mm)", rotation=270, labelpad=-8)
                ax.xaxis.set_tick_params(pad=-5)
                ax.yaxis.set_tick_params(pad=-5)
                ax.zaxis.set_tick_params(pad=-5)
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
        skeleton_axes.append(ax)
    
    return fig, video_axes, video_images, heatmap_axes, depth_axes, skeleton_axes


def render_frames_with_reused_figure(
    fig: plt.Figure,
    video_axes: list,
    video_images: list,
    heatmap_axes: list, 
    depth_axes: list,
    skeleton_axes: list,
    frames_dir: Path,
    all_video_frames: dict,
    n_cols: int,
    n_frames: int,
    pred_xy_heatmaps: np.ndarray,
    pred_depth_logits: np.ndarray,
    pred_world_xyz: np.ndarray,
    label_xy: np.ndarray,
    label_depth: np.ndarray,
    label_world_xyz: np.ndarray,
    keypoints_order: list[str],
    stride_x: int,
    stride_y: int,
    depth_min: float,
    depth_max: float,
    depth_n_bins: int,
    marker_size: int,
    cmap,
    n_keypoints: int,
) -> None:
    """Render frames by updating data only in pre-setup figure axes"""
    
    for frame_idx in range(n_frames):
        # Update video frames - just replace image data, no clearing needed!
        for col_idx in range(n_cols):
            im = video_images[col_idx]
            
            if col_idx == 0:
                video_frame = all_video_frames["original"][frame_idx]
            else:
                model_idx = col_idx - 1
                model_key = f"synthetic_{model_idx}"
                video_frame = all_video_frames[model_key][frame_idx]
            
            # Update image data directly - much more efficient!
            im.set_array(video_frame)
            im.set_extent([0, video_frame.shape[1], video_frame.shape[0], 0])
        
        # Update heatmaps - only data
        for col_idx in range(1, n_cols):
            variant_idx = col_idx - 1
            update_heatmap_data(heatmap_axes[col_idx], frame_idx, variant_idx, pred_xy_heatmaps,
                              label_xy, stride_x, stride_y, n_keypoints, marker_size, cmap)
        
        # Update depth plots - only data  
        for col_idx in range(1, n_cols):
            variant_idx = col_idx - 1
            update_depth_data(depth_axes[col_idx], frame_idx, variant_idx, pred_depth_logits,
                             label_depth, depth_min, depth_max, depth_n_bins,
                             n_keypoints, marker_size, cmap)
        
        # Update 3D skeletons - only data
        for col_idx in range(1, n_cols):
            variant_idx = col_idx - 1
            update_3d_skeleton_data(skeleton_axes[col_idx], frame_idx, variant_idx, pred_world_xyz,
                                   label_world_xyz, keypoints_order, n_keypoints, marker_size)
        
        # Save frame
        frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
        fig.savefig(frame_path, dpi=100, bbox_inches=None, pad_inches=0.1)
        
        if frame_idx % 50 == 0:
            print(f"Rendered frame {frame_idx}/{n_frames}")


def render_frames_parallel(
    frames_dir: Path,
    all_video_frames: dict,  # {"original": list[np.ndarray], "synthetic_0": list[np.ndarray], ...}
    n_cols: int,
    n_frames: int,
    # Data arrays - all with consistent frame indexing
    pred_xy_heatmaps: np.ndarray,    # (n_variants, n_frames, n_keypoints, H, W)
    pred_depth_logits: np.ndarray,   # (n_variants, n_frames, n_keypoints, depth_bins)
    pred_world_xyz: np.ndarray,      # (n_variants, n_frames, n_keypoints, 3)
    label_xy: np.ndarray,            # (n_frames, n_keypoints, 2)
    label_depth: np.ndarray,         # (n_frames, n_keypoints)
    label_world_xyz: np.ndarray,     # (n_frames, n_keypoints, 3)
    keypoints_order: list[str],      # list of keypoint names
    # Parameters
    stride_x: int,
    stride_y: int,
    depth_min: float,
    depth_max: float,
    depth_n_bins: int,
    marker_size: int,
    cmap,  # matplotlib colormap
    n_keypoints: int,
    n_workers: int = -2,
) -> None:
    """Optimized frame rendering by reusing figure elements and using parallel processing"""
    
    # Handle sequential processing case
    if n_workers == 1:
        logger.info("Using sequential processing (n_workers=1)")
        render_frames_sequential(
            frames_dir, all_video_frames, n_cols, n_frames,
            pred_xy_heatmaps, pred_depth_logits, pred_world_xyz,
            label_xy, label_depth, label_world_xyz, keypoints_order,
            stride_x, stride_y, depth_min, depth_max, depth_n_bins,
            marker_size, cmap, n_keypoints
        )
        return
    
    logger.info(f"Starting parallel frame rendering with {n_workers} workers...")
    start_time = time.time()
    
    # Split frame indices into chunks for parallel processing
    frame_indices = list(range(n_frames))
    
    # Determine actual number of workers
    if n_workers == -2:
        import multiprocessing
        actual_workers = max(1, multiprocessing.cpu_count() - 1)
    elif n_workers == -1:
        import multiprocessing
        actual_workers = multiprocessing.cpu_count()
    else:
        actual_workers = max(1, n_workers)
    
    # Split frames into chunks for each worker
    chunk_size = max(1, n_frames // actual_workers)
    frame_chunks = [frame_indices[i:i + chunk_size] for i in range(0, n_frames, chunk_size)]
    
    logger.info(f"Using {actual_workers} workers to process {n_frames} frames")
    logger.info(f"Frame chunks: {[len(chunk) for chunk in frame_chunks]}")
    
    # Run parallel processing
    Parallel(n_jobs=actual_workers, prefer="processes")(
        delayed(render_frame_chunk_worker)(
            frame_chunk,
            frames_dir,
            all_video_frames,
            n_cols,
            pred_xy_heatmaps,
            pred_depth_logits,
            pred_world_xyz,
            label_xy,
            label_depth,
            label_world_xyz,
            keypoints_order,
            stride_x,
            stride_y,
            depth_min,
            depth_max,
            depth_n_bins,
            marker_size,
            cmap,
            n_keypoints,
            worker_id
        ) for worker_id, frame_chunk in enumerate(frame_chunks)
    )
    
    end_time = time.time()
    logger.info(f"Parallel frame rendering completed in {end_time - start_time:.2f} seconds")


def render_frame_chunk_worker(
    frame_indices: list[int],
    frames_dir: Path,
    all_video_frames: dict,
    n_cols: int,
    pred_xy_heatmaps: np.ndarray,
    pred_depth_logits: np.ndarray,
    pred_world_xyz: np.ndarray,
    label_xy: np.ndarray,
    label_depth: np.ndarray,
    label_world_xyz: np.ndarray,
    keypoints_order: list[str],
    stride_x: int,
    stride_y: int,
    depth_min: float,
    depth_max: float,
    depth_n_bins: int,
    marker_size: int,
    cmap,
    n_keypoints: int,
    worker_id: int,
) -> None:
    """Worker function that processes a chunk of frames with figure reuse"""
    if not frame_indices:
        return
        
    worker_start_time = time.time()
    logger.info(f"Worker {worker_id}: Processing {len(frame_indices)} frames: {frame_indices[0]}-{frame_indices[-1]}")
    
    # Setup figure and axes once per worker
    fig, video_axes, video_images, heatmap_axes, depth_axes, skeleton_axes = setup_figure_and_axes(
        n_cols, n_keypoints, depth_min, depth_max, depth_n_bins
    )
    
    # Process each frame in this chunk using the same figure
    for i, frame_idx in enumerate(frame_indices):
        # Update video frames - just replace image data, no clearing needed!
        for col_idx in range(n_cols):
            im = video_images[col_idx]
            
            if col_idx == 0:
                video_frame = all_video_frames["original"][frame_idx]
            else:
                model_idx = col_idx - 1
                model_key = f"synthetic_{model_idx}"
                video_frame = all_video_frames[model_key][frame_idx]
            
            # Update image data directly - much more efficient!
            im.set_array(video_frame)
            im.set_extent([0, video_frame.shape[1], video_frame.shape[0], 0])
        
        # Update heatmaps - only data
        for col_idx in range(1, n_cols):
            variant_idx = col_idx - 1
            update_heatmap_data(heatmap_axes[col_idx], frame_idx, variant_idx, pred_xy_heatmaps,
                              label_xy, stride_x, stride_y, n_keypoints, marker_size, cmap)
        
        # Update depth plots - only data
        for col_idx in range(1, n_cols):
            variant_idx = col_idx - 1
            update_depth_data(depth_axes[col_idx], frame_idx, variant_idx, pred_depth_logits,
                             label_depth, depth_min, depth_max, depth_n_bins,
                             n_keypoints, marker_size, cmap)
        
        # Update 3D skeletons - only data
        for col_idx in range(1, n_cols):
            variant_idx = col_idx - 1
            update_3d_skeleton_data(skeleton_axes[col_idx], frame_idx, variant_idx, pred_world_xyz,
                                   label_world_xyz, keypoints_order, n_keypoints, marker_size)
        
        # Save frame
        frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
        fig.savefig(frame_path, dpi=100, bbox_inches=None, pad_inches=0.1)
        
        # Progress reporting (less frequent to avoid spam)
        if i % 20 == 0 or i == len(frame_indices) - 1:
            logger.info(f"Worker {worker_id}: Rendered frame {frame_idx} ({i+1}/{len(frame_indices)})")
                
    plt.close(fig)
    
    worker_end_time = time.time()
    logger.info(f"Worker {worker_id}: Completed {len(frame_indices)} frames in {worker_end_time - worker_start_time:.2f} seconds")


def render_frames_sequential(
    frames_dir: Path,
    all_video_frames: dict,  # {"original": list[np.ndarray], "synthetic_0": list[np.ndarray], ...}
    n_cols: int,
    n_frames: int,
    # Data arrays - all with consistent frame indexing
    pred_xy_heatmaps: np.ndarray,    # (n_variants, n_frames, n_keypoints, H, W)
    pred_depth_logits: np.ndarray,   # (n_variants, n_frames, n_keypoints, depth_bins)
    pred_world_xyz: np.ndarray,      # (n_variants, n_frames, n_keypoints, 3)
    label_xy: np.ndarray,            # (n_frames, n_keypoints, 2)
    label_depth: np.ndarray,         # (n_frames, n_keypoints)
    label_world_xyz: np.ndarray,     # (n_frames, n_keypoints, 3)
    keypoints_order: list[str],      # list of keypoint names
    # Parameters
    stride_x: int,
    stride_y: int,
    depth_min: float,
    depth_max: float,
    depth_n_bins: int,
    marker_size: int,
    cmap,  # matplotlib colormap
    n_keypoints: int,
) -> None:
    """Original sequential rendering - kept for fallback"""
    print("Setting up reusable figure...")
    
    # Setup figure and axes once
    fig, video_axes, video_images, heatmap_axes, depth_axes, skeleton_axes = setup_figure_and_axes(
        n_cols, n_keypoints, depth_min, depth_max, depth_n_bins
    )
    
    # Render all frames with the reused figure
    render_frames_with_reused_figure(
        fig, video_axes, video_images, heatmap_axes, depth_axes, skeleton_axes,
        frames_dir, all_video_frames, n_cols, n_frames,
        pred_xy_heatmaps, pred_depth_logits, pred_world_xyz,
        label_xy, label_depth, label_world_xyz, keypoints_order,
        stride_x, stride_y, depth_min, depth_max, depth_n_bins,
        marker_size, cmap, n_keypoints
    )
    
    plt.close(fig)
    print("Frame rendering completed!")


class Keypoints3DVisualizer:
    def __init__(
        self,
        preds: dict[str, np.ndarray],
        labels: dict[str, np.ndarray],
        keypoints_order: list[str],
        output_path: Path,
        data_freq: int = 300,
        nmf_rendering_basedir: Path | None = None,
        synthetic_videos_basedir: Path | None = None,
        exp_trial: str | None = None,
        segment: str | None = None,
        subsegment: str | None = None,
        style_transfer_models: list[str] | None = None,
        depth_min: float = -102.0,
        depth_max: float = -98.0,
        depth_n_bins: int = 64,
        cmap=cmasher.cm.nuclear,
        marker_size: int = 15,
        n_workers: int = -2,
    ):
        """
        Initialize the 3D keypoints visualizer.
        
        Parameters:
        -----------
        n_workers : int, default=-2
            Number of worker processes for parallel frame rendering.
            - -2: Use all CPU cores except 1 (recommended for most cases)
            - -1: Use all CPU cores
            - 1: Sequential processing (no parallelization) 
            - N > 1: Use exactly N workers
        """
        self.preds = preds
        self.labels = labels
        self.keypoints_order = keypoints_order
        self.output_path = output_path
        self.data_freq = data_freq
        self.nmf_rendering_basedir = nmf_rendering_basedir
        self.synthetic_videos_basedir = synthetic_videos_basedir
        self.exp_trial = exp_trial
        self.segment = segment
        self.subsegment = subsegment
        self.style_transfer_models = style_transfer_models
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_n_bins = depth_n_bins
        self.cmap = cmap
        self.marker_size = marker_size
        self.n_workers = n_workers

        self.stride_x = preds["heatmap_stride_cols"][0]
        self.stride_y = preds["heatmap_stride_rows"][0]
        # (variants, frames, keypoints, H, W)
        self.pred_xy_heatmaps = preds["xy_heatmaps"]
        # (variants, frames, keypoints, depth_bins)
        self.pred_depth_logits = preds["depth_logits"]
        # (variants, frames, keypoints, 2)
        self.pred_xy = preds["pred_xy"]
        # (variants, frames, keypoints)
        self.pred_depth = preds["pred_depth"]
        # (variants, frames, keypoints, 3)
        self.pred_world_xyz = preds["pred_world_xyz"]
        # (frames, keypoints, 2)
        self.label_xy = labels["keypoint_pos"][:, :, :2]
        # (frames, keypoints)
        self.label_depth = labels["keypoint_pos"][:, :, 2]
        # (frames, keypoints, 3)
        self.label_world_xyz = labels["keypoint_pos_world_xyz"]
        self.n_variants, self.n_frames, self.n_keypoints = self.pred_depth.shape

    def plot_keypoints_over_time(self, output_path: Path) -> None:
        fig, axes = plt.subplots(
            self.n_keypoints,
            6,
            figsize=(6 * 3, self.n_keypoints * 2),
            tight_layout=True,
        )
        t_grid = np.arange(self.n_frames) / self.data_freq
        for i_keypoint in range(self.n_keypoints):
            for i_panel, panel_name in enumerate(
                ["column", "row", "depth", "x", "y", "z"]
            ):
                ax = axes[i_keypoint, i_panel]
                keypoint_name = self.keypoints_order[i_keypoint]

                # Plot col, row, depth in camera coords
                if panel_name in ("column", "row", "depth"):
                    is_depth = panel_name == "depth"
                    if is_depth:
                        pred = self.pred_depth[:, :, i_keypoint]
                        label = self.label_depth[:, i_keypoint]
                    else:
                        pred = self.pred_xy[:, :, i_keypoint, i_panel]
                        label = self.label_xy[:, i_keypoint, i_panel]
                    for i_variant in range(self.n_variants):
                        ax.plot(t_grid, pred[i_variant, :], linewidth=1)
                    ax.plot(t_grid, label, color="black", linewidth=2)
                    ax.set_xlabel("time (s)")
                    ax.set_ylabel(
                        "depth (mm)" if is_depth else f"{panel_name} (pixels)"
                    )
                    ax.set_title(f"{keypoint_name}, {panel_name}")

                # Plot x, y, z in world coords, but individually
                if panel_name in ("x", "y", "z"):
                    pred = self.pred_world_xyz[:, :, i_keypoint, i_panel - 3]
                    label = self.label_world_xyz[:, i_keypoint, i_panel - 3]
                    for i_variant in range(self.n_variants):
                        ax.plot(t_grid, pred[i_variant, :], linewidth=1)
                    ax.plot(t_grid, label, color="black", linewidth=2)
                    ax.set_xlabel("time (s)")
                    ax.set_ylabel(f"{panel_name} (mm)")
                    ax.set_title(f"{keypoint_name}, {panel_name}")

        fig.savefig(output_path)
        plt.close(fig)

    def make_summary_video(self, output_path: Path) -> None:
        """Create a comprehensive video with 4 rows showing:
        - Row 0: Original simulation video + synthetic videos from each style transfer model
        - Row 1: Ground truth heatmap + predicted heatmaps for body keypoints (all keypoints merged)
        - Row 2: Depth distributions as heatmaps + labels as red dots
        - Row 3: 3D body skeleton in world coordinates
        """
        sim_rendering_path = (
            Path(self.nmf_rendering_basedir)
            / self.exp_trial
            / self.segment
            / self.subsegment
            / "processed_nmf_sim_render_colorcode_0.mp4"
        )
        synthetic_videos_dir = (
            Path(self.synthetic_videos_basedir)
            / self.exp_trial
            / self.segment
            / self.subsegment
        )

        n_cols = 1 + self.n_variants  # 1 for original + n_variants for style transfer models

        # Pre-load all video frames for efficiency
        print("Loading video frames...")
        all_video_frames = self._load_all_video_frames(sim_rendering_path, synthetic_videos_dir)
        print(f"Loaded {len(all_video_frames)} video streams")

        # Create frames directory
        frames_dir = Path("plot_frames")
        frames_dir.mkdir(exist_ok=True)
        print(f"DEBUG: Frames will be saved to: {frames_dir.absolute()}")

        # Clear old frames
        for f in frames_dir.glob("frame_*.png"):
            f.unlink()

        # Use optimized sequential rendering with figure reuse
        self._render_frames(frames_dir, all_video_frames, n_cols)
        # Convert frames to video
        self._frames_to_video(frames_dir, output_path)

        print(f"DEBUG: Frames preserved in: {frames_dir.absolute()}")
        print(f"Video saved to: {output_path}")

    def _render_frames(self, frames_dir: Path, all_video_frames: dict, n_cols: int) -> None:
        """Optimized frame rendering by reusing figure elements and only updating data - delegates to standalone function"""
        render_frames_parallel(
            frames_dir=frames_dir,
            all_video_frames=all_video_frames,
            n_cols=n_cols,
            n_frames=self.n_frames,
            # Data arrays
            pred_xy_heatmaps=self.pred_xy_heatmaps,
            pred_depth_logits=self.pred_depth_logits,
            pred_world_xyz=self.pred_world_xyz,
            label_xy=self.label_xy,
            label_depth=self.label_depth,
            label_world_xyz=self.label_world_xyz,
            keypoints_order=self.keypoints_order,
            # Parameters
            stride_x=self.stride_x,
            stride_y=self.stride_y,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
            depth_n_bins=self.depth_n_bins,
            marker_size=self.marker_size,
            cmap=self.cmap,
            n_keypoints=self.n_keypoints,
            n_workers=self.n_workers,
        )

    def _load_all_video_frames(
        self, sim_rendering_path: Path, synthetic_videos_dir: Path
    ) -> dict:
        """Efficiently load all video frames into memory for fast access"""
        all_frames = {}

        # Load original simulation video
        print(f"Loading original video: {sim_rendering_path}")
        frames, fps = read_frames_from_video(sim_rendering_path, frame_indices=None)
        all_frames["original"] = frames
        print(f"Loaded {len(frames)} frames from {sim_rendering_path}")

        # Load synthetic videos for each style transfer model
        for model_idx, model_name in enumerate(self.style_transfer_models):
            synthetic_video_path = synthetic_videos_dir / f"translated_{model_name}.mp4"
            model_key = f"synthetic_{model_idx}"
            print(f"Loading synthetic video {model_idx}: {synthetic_video_path}")
            frames, fps = read_frames_from_video(synthetic_video_path, frame_indices=None)
            all_frames[model_key] = frames
            print(f"Loaded {len(frames)} frames from {synthetic_video_path}")

        return all_frames

    def _frames_to_video(self, frames_dir: Path, output_path: Path, fps: int = 30):
        """Convert frame images to video using imageio"""
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if not frame_files:
            raise ValueError("No frame files found")

        logger.info(f"Converting {len(frame_files)} frames to video using imageio...")

        # Check frame sizes first to detect inconsistencies
        frame_sizes = {}
        target_size = None
        for i, frame_file in enumerate(frame_files[:5]):  # Check first 5 frames
            frame = imageio.imread(str(frame_file))
            size = frame.shape[:2]  # (height, width)
            frame_sizes[i] = size
            if target_size is None:
                target_size = size
            elif size != target_size:
                logger.warning(
                    f"Frame size inconsistency detected: frame {i} has size {size}, expected {target_size}"
                )
        logger.info(f"Target frame size: {target_size}")

        with imageio.get_writer(
            str(output_path),
            "ffmpeg",
            fps=fps,
            codec="libx264",
            quality=None,  # Use CRF instead of quality
            ffmpeg_params=default_video_writing_ffmpeg_params,
        ) as video_writer:
            successful_frames = 0
            for i, frame_file in enumerate(frame_files):
                # Read frame using imageio (consistent with video loading)
                frame = imageio.imread(str(frame_file))

                # Ensure consistent frame size
                if target_size and frame.shape[:2] != target_size:
                    logger.debug(
                        f"Resizing frame {i} from {frame.shape[:2]} to {target_size}"
                    )
                    pil_frame = Image.fromarray(frame)
                    pil_frame = pil_frame.resize(
                        (target_size[1], target_size[0]),
                        Image.Resampling.LANCZOS,
                    )
                    frame = np.array(pil_frame)

                video_writer.append_data(frame)
                successful_frames += 1

                if i % 100 == 0:
                    logger.debug(f"Processed frame {i}/{len(frame_files)}")

        logger.info(
            f"Successfully wrote {successful_frames}/{len(frame_files)} frames to video"
        )

        if successful_frames == 0:
            raise RuntimeError("Failed to write any frames to video")