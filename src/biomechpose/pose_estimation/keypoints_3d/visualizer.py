import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
import cmasher
import imageio.v2 as imageio
from PIL import Image
from pathlib import Path

from biomechpose.util import (
    configure_matplotlib_style,
    default_video_writing_ffmpeg_params,
)
from biomechpose.simulate_nmf.utils import kchain_plotting_colors


configure_matplotlib_style()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Standalone functions for parallel processing (must be pickle-able)
def _get_keypoint_color_standalone(keypoint_name: str) -> np.ndarray:
    """Get color for a keypoint based on kchain_plotting_colors dictionary"""
    if keypoint_name in ["LPedicel", "RPedicel"]:
        # Map antenna pedicels to antenna colors
        antenna_key = "LAntenna" if keypoint_name.startswith("L") else "RAntenna"
        return kchain_plotting_colors[antenna_key]
    else:
        # Use first two characters (leg identifier)
        leg_key = keypoint_name[:2]
        return kchain_plotting_colors.get(leg_key, np.array([0.5, 0.5, 0.5]))  # Default gray


def _get_skeleton_connections_standalone(keypoints_order: list[str]) -> list[tuple[int, int]]:
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


def render_single_frame(
    frame_idx: int,
    frames_dir: Path,
    all_video_frames: dict,
    n_cols: int,
    # Data needed for rendering
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
    style_transfer_models: list[str],
) -> None:
    """Standalone function to render a single frame - can be pickled for multiprocessing"""
    try:
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, n_cols, height_ratios=[1, 1, 1, 1], hspace=0.3)

        # Row 0: Videos
        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs[0, col_idx])
            if col_idx == 0:
                if "original" in all_video_frames and frame_idx < len(all_video_frames["original"]):
                    video_frame = all_video_frames["original"][frame_idx]
                    if video_frame is not None:
                        ax.imshow(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))
                        ax.set_title("NeuroMechFly\nsimulation", fontweight="bold", pad=20)
                    else:
                        ax.text(0.5, 0.5, f"Frame {frame_idx}\nNot Available", 
                               ha="center", va="center", transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, f"Original Video\nNot Found", 
                           ha="center", va="center", transform=ax.transAxes)
            else:
                model_idx = col_idx - 1
                model_key = f"synthetic_{model_idx}"
                if model_key in all_video_frames and frame_idx < len(all_video_frames[model_key]):
                    video_frame = all_video_frames[model_key][frame_idx]
                    if video_frame is not None:
                        ax.imshow(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))
                        ax.set_title(f"Synthetic rendering\n(variant {model_idx})", 
                                   fontweight="bold", pad=20)
                    else:
                        ax.text(0.5, 0.5, f"Frame {frame_idx}\nNot Available", 
                               ha="center", va="center", transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, f"Synthetic Video\nModel {model_idx}\nNot Found", 
                           ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

        # Row 1: Heatmaps
        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs[1, col_idx])
            if col_idx == 0:
                ax.axis("off")
                ax.text(0.6, 0.5, "2D pose", transform=ax.transAxes, rotation=90,
                       va='center', ha='center', fontsize=12, fontweight='bold', color='black')
            else:
                variant_idx = col_idx - 1
                _plot_merged_heatmaps_standalone(ax, frame_idx, variant_idx, pred_xy_heatmaps, 
                                               label_xy, stride_x, stride_y, n_keypoints, 
                                               marker_size, cmap)

        # Row 2: Depth
        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs[2, col_idx])
            if col_idx == 0:
                ax.axis("off")
                ax.text(0.6, 0.5, "Distance from camera", transform=ax.transAxes, rotation=90,
                       va='center', ha='center', fontsize=12, fontweight='bold', color='black')
            else:
                variant_idx = col_idx - 1
                _plot_depth_distributions_standalone(ax, frame_idx, variant_idx, pred_depth_logits,
                                                    label_depth, depth_min, depth_max, depth_n_bins, 
                                                    n_keypoints, marker_size, cmap)

        # Row 3: 3D skeleton
        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs[3, col_idx], projection="3d")
            if col_idx == 0:
                ax.axis("off")
                ax.text2D(0.6, 0.5, "3D reconstruction", transform=ax.transAxes, rotation=90,
                         va='center', ha='center', fontsize=12, fontweight='bold', color='black')
            else:
                variant_idx = col_idx - 1
                _plot_3d_skeleton_standalone(ax, frame_idx, variant_idx, pred_world_xyz,
                                           label_world_xyz, keypoints_order, n_keypoints,
                                           marker_size)

        frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
        fig.savefig(frame_path, dpi=100, bbox_inches=None, pad_inches=0.1)
        plt.close(fig)
        
    except Exception as e:
        logger.exception(f"Failed to render frame {frame_idx}: {e}")


def _plot_merged_heatmaps_standalone(ax, frame_idx: int, variant_idx: int, pred_xy_heatmaps: np.ndarray,
                                   label_xy: np.ndarray, stride_x: int, stride_y: int, 
                                   n_keypoints: int, marker_size: int, cmap):
    """Plot merged heatmaps for all keypoints with label dots"""
    # Get heatmaps for this frame and variant
    heatmaps = pred_xy_heatmaps[variant_idx, frame_idx]  # (n_keypoints, H, W)

    # Convert each keypoint's heatmap to probabilities independently
    heatmaps_prob = np.zeros_like(heatmaps)
    for kp_idx in range(n_keypoints):
        # Normalize each keypoint's heatmap to probabilities
        heatmap_flat = heatmaps[kp_idx].flatten()
        heatmap_flat_shifted = heatmap_flat - np.max(heatmap_flat)
        probs_flat = np.exp(heatmap_flat_shifted) / np.sum(np.exp(heatmap_flat_shifted))
        heatmaps_prob[kp_idx] = probs_flat.reshape(heatmaps[kp_idx].shape)

    # Merge all keypoint heatmaps (take maximum across keypoints)
    merged_heatmap_prob = np.max(heatmaps_prob, axis=0)  # (H, W)

    # Display merged heatmap
    im = ax.imshow(merged_heatmap_prob, cmap=cmap, vmin=0, vmax=0.04)

    # Plot label points as prominent dots
    for kp_idx in range(n_keypoints):
        label_x = label_xy[frame_idx, kp_idx, 0] / stride_x
        label_y = label_xy[frame_idx, kp_idx, 1] / stride_y
        ax.scatter(label_x, label_y, color="white", s=marker_size, 
                  edgecolors="black", linewidth=2)

    ax.set_xlim(0, merged_heatmap_prob.shape[1])
    ax.set_ylim(merged_heatmap_prob.shape[0], 0)  # Flip y-axis for image coordinates
    ax.set_aspect("equal")


def _plot_depth_distributions_standalone(ax, frame_idx: int, variant_idx: int, pred_depth_logits: np.ndarray,
                                        label_depth: np.ndarray, depth_min: float, depth_max: float, 
                                        depth_n_bins: int, n_keypoints: int, marker_size: int, cmap):
    """Plot depth predictions as a heatmap"""
    # Get depth logits for this frame and variant
    depth_logits = pred_depth_logits[variant_idx, frame_idx]  # (n_keypoints, depth_bins)

    # Convert logits to probabilities
    depth_logits_shifted = depth_logits - np.max(depth_logits, axis=1, keepdims=True)
    depth_probs = np.exp(depth_logits_shifted) / np.sum(np.exp(depth_logits_shifted), axis=1, keepdims=True)

    # Show as heatmap: depth bins (y-axis) vs keypoints (x-axis)
    im = ax.imshow(depth_probs.T, cmap=cmap, aspect='auto', vmin=0, vmax=0.5)
    
    # Add ground truth depth as white dots
    depth_bin_centers = np.linspace(depth_min, depth_max, depth_n_bins)
    for kp_idx in range(n_keypoints):
        label_depth_val = label_depth[frame_idx, kp_idx]
        # Convert depth to bin index using proper mapping
        depth_bin = np.argmin(np.abs(depth_bin_centers - label_depth_val))
        ax.scatter(kp_idx, depth_bin, color='white', s=marker_size, marker='o', 
                  edgecolors='black', linewidth=1)

    # Add white vertical lines after each leg (every 6 keypoints)
    for leg_end in range(6, n_keypoints, 6):
        ax.axvline(x=leg_end - 0.5, color='white', linewidth=1)

    ax.set_xlabel("Keypoint Index")
    ax.set_ylabel("Depth Bins")
    ax.set_xticks(range(n_keypoints))
    ax.set_xticklabels([])


def _plot_3d_skeleton_standalone(ax, frame_idx: int, variant_idx: int, pred_world_xyz: np.ndarray,
                                label_world_xyz: np.ndarray, keypoints_order: list[str],
                                n_keypoints: int, marker_size: int):
    """Plot 3D skeleton with connections between keypoints"""
    skeleton_connections = _get_skeleton_connections_standalone(keypoints_order)

    # Plot ground truth first
    gt_points = label_world_xyz[frame_idx]  # (n_keypoints, 3)
    
    # Plot leg connections with colors
    for connection in skeleton_connections:
        if connection[0] < n_keypoints and connection[1] < n_keypoints:
            start_point = gt_points[connection[0]]
            end_point = gt_points[connection[1]]
            
            keypoint_name = keypoints_order[connection[0]]
            line_color = _get_keypoint_color_standalone(keypoint_name)
            
            ax.plot3D([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                     [start_point[2], end_point[2]], color=line_color, linewidth=3)

    # Plot antennae keypoints as scatter points (last 2 keypoints)
    if n_keypoints >= 2:
        for i in range(2):
            antenna_idx = n_keypoints - 2 + i
            antenna_point = gt_points[antenna_idx]
            keypoint_name = keypoints_order[antenna_idx]
            point_color = _get_keypoint_color_standalone(keypoint_name)
            ax.scatter(antenna_point[0], antenna_point[1], antenna_point[2], 
                      color=point_color, s=marker_size)

    # Plot predictions
    if variant_idx is not None:
        pred_points = pred_world_xyz[variant_idx, frame_idx]  # (n_keypoints, 3)

        # Plot leg connections
        for connection in skeleton_connections:
            if connection[0] < n_keypoints and connection[1] < n_keypoints:
                start_point = pred_points[connection[0]]
                end_point = pred_points[connection[1]]
                
                keypoint_name = keypoints_order[connection[0]]
                line_color = _get_keypoint_color_standalone(keypoint_name)
                
                ax.plot3D([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                         [start_point[2], end_point[2]], color=line_color, linewidth=2)

        # Plot antennae keypoints
        if n_keypoints >= 2:
            for i in range(2):
                antenna_idx = n_keypoints - 2 + i
                antenna_point = pred_points[antenna_idx]
                keypoint_name = keypoints_order[antenna_idx]
                point_color = _get_keypoint_color_standalone(keypoint_name)
                ax.scatter(antenna_point[0], antenna_point[1], antenna_point[2], 
                          color=point_color, s=marker_size)

    # Set labels and limits
    if variant_idx == 0:
        ax.set_xlabel("ant.-post. (mm)")
        ax.set_ylabel("lateral (mm)")
        ax.set_zlabel("dors.-vent. (mm)")
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # Set equal aspect ratio and limits
    all_points = label_world_xyz[frame_idx]
    if variant_idx is not None:
        all_points = np.vstack([all_points, pred_world_xyz[variant_idx, frame_idx]])

    # Calculate ranges for each dimension
    x_range = np.ptp(all_points[:, 0])
    y_range = np.ptp(all_points[:, 1])
    z_range = np.ptp(all_points[:, 2])
    
    # Use the maximum range to ensure equal scaling
    max_range = max(x_range, y_range, z_range) / 2.0
    
    # Center points for each dimension
    mid_x = np.mean(all_points[:, 0])
    mid_y = np.mean(all_points[:, 1])
    mid_z = np.mean(all_points[:, 2])

    # Set equal limits for all axes
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Ensure equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Set viewing angle
    azimuth = np.cos(2 * np.pi * frame_idx / 300.0) * 30.0
    ax.view_init(elev=30.0, azim=azimuth)



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
    ):
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
            try:
                f.unlink()
            except Exception:
                pass

        # Use optimized sequential rendering with figure reuse
        try:
            self._render_frames_optimized(frames_dir, all_video_frames, n_cols)
            # Convert frames to video
            self._frames_to_video(frames_dir, output_path)
        except Exception as e:
            print(f"Error generating video: {e}")
            print(f"Frames saved in: {frames_dir.absolute()}")
            raise

        print(f"DEBUG: Frames preserved in: {frames_dir.absolute()}")
        print(f"Video saved to: {output_path}")

    def _render_frames_optimized(self, frames_dir: Path, all_video_frames: dict, n_cols: int) -> None:
        """Optimized frame rendering by reusing figure elements and only updating data"""
        print("Setting up reusable figure...")
        
        # Create figure and axes once
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, n_cols, height_ratios=[1, 1, 1, 1], hspace=0.3)
        
        # Store all axes for reuse
        video_axes = []
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
            else:
                model_idx = col_idx - 1
                ax.set_title(f"Synthetic rendering\n(variant {model_idx})", fontweight="bold", pad=20)
            video_axes.append(ax)
        
        # Row 1: Heatmap axes
        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs[1, col_idx])
            if col_idx == 0:
                ax.axis("off")
                ax.text(0.6, 0.5, "2D pose", transform=ax.transAxes, rotation=90,
                       va='center', ha='center', fontsize=12, fontweight='bold', color='black')
            else:
                ax.set_aspect("equal")
                if col_idx == 1:  # Only for first data column
                    ax.set_ylabel("row (px)")
                    ax.set_xlabel("column (px)")
                else:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
            heatmap_axes.append(ax)
        
        # Row 2: Depth axes
        depth_bin_centers = np.linspace(self.depth_min, self.depth_max, self.depth_n_bins)
        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs[2, col_idx])
            if col_idx == 0:
                ax.axis("off")
                ax.text(0.6, 0.5, "Distance from camera", transform=ax.transAxes, rotation=90,
                       va='center', ha='center', fontsize=12, fontweight='bold', color='black')
            else:
                # Pre-setup depth plot formatting
                for leg_end in range(6, self.n_keypoints, 6):
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
                
                if col_idx == 1:  # Only for first data column
                    ax.set_xticklabels(k_ticklabels)
                    ax.set_ylabel("depth (mm)")
                    if self.depth_n_bins > 10:
                        tick_indices = range(0, self.depth_n_bins, max(1, self.depth_n_bins // 5))
                        ax.set_yticks(tick_indices)
                        ax.set_yticklabels([f"{depth_bin_centers[i]:.1f}" for i in tick_indices], fontsize=8)
                else:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
            depth_axes.append(ax)
        
        # Row 3: 3D skeleton axes
        for col_idx in range(n_cols):
            if col_idx == 0:
                ax = fig.add_subplot(gs[3, col_idx], projection="3d")
                ax.axis("off")
                ax.text2D(0.6, 0.5, "3D reconstruction", transform=ax.transAxes, rotation=90,
                         va='center', ha='center', fontsize=12, fontweight='bold', color='black')
            else:
                ax = fig.add_subplot(gs[3, col_idx], projection="3d")
                variant_idx = col_idx - 1
                if variant_idx == 0:
                    ax.set_xlabel("ant.-post. (mm)")
                    ax.set_ylabel("lateral (mm)")
                    ax.set_zlabel("dors.-vent. (mm)", rotation=90)
                else:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_zticklabels([])
                ax.set_box_aspect([1, 1, 1])
            skeleton_axes.append(ax)
        
        # Pre-compute skeleton connections
        skeleton_connections = self._get_skeleton_connections()
        
        print(f"Rendering {self.n_frames} frames with optimized approach...")
        
        # Render each frame by updating data only
        for frame_idx in range(self.n_frames):
            try:
                # Update video frames
                for col_idx in range(n_cols):
                    ax = video_axes[col_idx]
                    ax.clear()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    if col_idx == 0:
                        ax.set_title("NeuroMechFly\nsimulation", fontweight="bold", pad=20)
                        if "original" in all_video_frames and frame_idx < len(all_video_frames["original"]):
                            video_frame = all_video_frames["original"][frame_idx]
                            if video_frame is not None:
                                ax.imshow(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))
                            else:
                                ax.text(0.5, 0.5, f"Frame {frame_idx}\nNot Available", 
                                       ha="center", va="center", transform=ax.transAxes)
                        else:
                            ax.text(0.5, 0.5, f"Original Video\nNot Found", 
                                   ha="center", va="center", transform=ax.transAxes)
                    else:
                        model_idx = col_idx - 1
                        ax.set_title(f"Synthetic rendering\n(variant {model_idx})", fontweight="bold", pad=20)
                        model_key = f"synthetic_{model_idx}"
                        if model_key in all_video_frames and frame_idx < len(all_video_frames[model_key]):
                            video_frame = all_video_frames[model_key][frame_idx]
                            if video_frame is not None:
                                ax.imshow(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))
                            else:
                                ax.text(0.5, 0.5, f"Frame {frame_idx}\nNot Available", 
                                       ha="center", va="center", transform=ax.transAxes)
                        else:
                            ax.text(0.5, 0.5, f"Synthetic Video\nModel {model_idx}\nNot Found", 
                                   ha="center", va="center", transform=ax.transAxes)
                
                # Update heatmaps (only data columns)
                for col_idx in range(1, n_cols):
                    ax = heatmap_axes[col_idx]
                    ax.clear()
                    variant_idx = col_idx - 1
                    self._plot_merged_heatmaps(ax, frame_idx, variant_idx)
                
                # Update depth plots (only data columns)
                for col_idx in range(1, n_cols):
                    ax = depth_axes[col_idx]
                    ax.clear()
                    variant_idx = col_idx - 1
                    self._plot_depth_distributions(ax, frame_idx, variant_idx)
                
                # Update 3D skeletons (only data columns)
                for col_idx in range(1, n_cols):
                    ax = skeleton_axes[col_idx]
                    ax.clear()
                    variant_idx = col_idx - 1
                    self._plot_3d_skeleton(ax, frame_idx, variant_idx, show_ground_truth=True)
                
                # Save frame
                frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
                fig.savefig(frame_path, dpi=100, bbox_inches=None, pad_inches=0.1)
                
                # Progress reporting
                if frame_idx % 50 == 0:
                    print(f"Rendered frame {frame_idx}/{self.n_frames}")
                    
            except Exception as e:
                logger.error(f"Failed to render frame {frame_idx}: {e}")
        
        plt.close(fig)
        print("Frame rendering completed!")

    def _load_all_video_frames(
        self, sim_rendering_path: Path, synthetic_videos_dir: Path
    ) -> dict:
        """Efficiently load all video frames into memory for fast access"""
        all_frames = {}

        # Load original simulation video
        if sim_rendering_path.exists():
            print(f"Loading original video: {sim_rendering_path}")
            all_frames["original"] = self._load_video_frames(sim_rendering_path)
        else:
            print(f"Warning: Original video not found at {sim_rendering_path}")
            all_frames["original"] = []

        # Load synthetic videos for each style transfer model
        for model_idx, model_name in enumerate(self.style_transfer_models):
            synthetic_video_path = synthetic_videos_dir / f"translated_{model_name}.mp4"
            model_key = f"synthetic_{model_idx}"

            if synthetic_video_path.exists():
                print(f"Loading synthetic video {model_idx}: {synthetic_video_path}")
                all_frames[model_key] = self._load_video_frames(synthetic_video_path)
            else:
                print(f"Warning: Synthetic video not found at {synthetic_video_path}")
                all_frames[model_key] = []

        return all_frames

    def _load_video_frames(self, video_path: Path) -> list:
        """Load all frames from a video file"""
        frames = []
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return frames

        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1

                # Limit to expected number of frames to avoid loading too much
                if frame_count >= self.n_frames * 2:  # Allow some buffer
                    break
        finally:
            cap.release()

        print(f"Loaded {len(frames)} frames from {video_path}")
        return frames

    def _render_and_save_frame(
        self, frame_idx: int, frames_dir: Path, all_video_frames: dict, n_cols: int
    ) -> None:
        """Render a single frame (all subplots) and save to disk.
        This runs in a worker thread and shares data in memory.
        """
        try:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(4, n_cols, height_ratios=[1, 1, 1, 1], hspace=0.3)

            # Row 0: Videos
            for col_idx in range(n_cols):
                ax = fig.add_subplot(gs[0, col_idx])
                if col_idx == 0:
                    if "original" in all_video_frames and frame_idx < len(
                        all_video_frames["original"]
                    ):
                        video_frame = all_video_frames["original"][frame_idx]
                        if video_frame is not None:
                            ax.imshow(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))
                            ax.set_title("NeuroMechFly\nsimulation", fontweight="bold", pad=20)
                        else:
                            ax.text(
                                0.5,
                                0.5,
                                f"Frame {frame_idx}\nNot Available",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            f"Original Video\nNot Found",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                else:
                    model_idx = col_idx - 1
                    model_key = f"synthetic_{model_idx}"
                    if model_key in all_video_frames and frame_idx < len(
                        all_video_frames[model_key]
                    ):
                        video_frame = all_video_frames[model_key][frame_idx]
                        if video_frame is not None:
                            ax.imshow(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))
                            ax.set_title(
                                f"Synthetic rendering\n(variant {model_idx})",
                                fontweight="bold",
                                pad=20,
                            )
                        else:
                            ax.text(
                                0.5,
                                0.5,
                                f"Frame {frame_idx}\nNot Available",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            f"Synthetic Video\nModel {model_idx}\nNot Found",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                ax.set_xticks([])
                ax.set_yticks([])

            # Row 1: Heatmaps
            for col_idx in range(n_cols):
                ax = fig.add_subplot(gs[1, col_idx])
                if col_idx == 0:
                    ax.axis("off")
                    # Add vertical row label
                    ax.text(
                        0.7,
                        0.5,
                        "2D pose",
                        transform=ax.transAxes,
                        rotation=90,  # Vertical text
                        va="center",
                        ha="center",
                        fontsize=12,  # Match default title size
                        fontweight="bold",  # Make bold like titles
                        color="black",
                    )
                else:
                    variant_idx = col_idx - 1
                    self._plot_merged_heatmaps(ax, frame_idx, variant_idx)
                    # ax.set_title(f"Row-column probabilities")

            # Row 2: Depth
            for col_idx in range(n_cols):
                ax = fig.add_subplot(gs[2, col_idx])
                if col_idx == 0:
                    ax.axis("off")
                    # Add vertical row label
                    ax.text(
                        0.7,
                        0.5,
                        "Distance from camera",
                        transform=ax.transAxes,
                        rotation=90,  # Vertical text
                        va="center",
                        ha="center",
                        fontsize=12,  # Match default title size
                        fontweight="bold",  # Make bold like titles
                        color="black",
                    )
                else:
                    variant_idx = col_idx - 1
                    self._plot_depth_distributions(ax, frame_idx, variant_idx)
                    # ax.set_title(f"Depth probabilities")

            # Row 3: 3D skeleton
            for col_idx in range(n_cols):
                ax = fig.add_subplot(gs[3, col_idx], projection="3d")
                if col_idx == 0:
                    ax.axis("off")
                    # Add vertical row label - use text2D for 3D axes
                    ax.text2D(
                        0.7,
                        0.5,
                        "3D reconstruction",
                        transform=ax.transAxes,
                        rotation=90,  # Vertical text
                        va="center",
                        ha="center",
                        fontsize=12,  # Match default title size
                        fontweight="bold",  # Make bold like titles
                        color="black",
                    )
                else:
                    variant_idx = col_idx - 1
                    self._plot_3d_skeleton(
                        ax, frame_idx, variant_idx, show_ground_truth=True
                    )
                    # ax.set_title(f"3D skeleton")
                    # if col_idx == 1:
                    #     # Only add ylabel once for the whole row (leftmost plot)
                    #     ax.text(
                    #         -0.25, 0.5, "3D pose (mm)", transform=ax.transAxes,
                    #         rotation="vertical", va="center", ha="center", fontsize=16
                    #     )

            # plt.tight_layout()
            frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
            # Use fixed figure size instead of bbox_inches="tight" to ensure consistent frame dimensions
            fig.savefig(frame_path, dpi=100, bbox_inches=0, pad_inches=0)

            # Debug: Check saved frame size for first few frames
            if frame_idx < 3:
                try:
                    test_frame = imageio.imread(str(frame_path))
                    logger.debug(
                        f"Frame {frame_idx} saved with size: {test_frame.shape}"
                    )
                except Exception:
                    pass

            plt.close(fig)
        except Exception as e:
            logger.exception(f"Failed to render frame {frame_idx}: {e}")
            # Ensure figure is closed even on error
            try:
                plt.close(fig)
            except:
                pass

    def _plot_merged_heatmaps(self, ax, frame_idx: int, variant_idx: int):
        """Plot merged heatmaps for all keypoints with label dots"""
        # Get heatmaps for this frame and variant
        heatmaps = self.pred_xy_heatmaps[variant_idx, frame_idx]  # (n_keypoints, H, W)

        # Convert each keypoint's heatmap to probabilities independently
        heatmaps_prob = np.zeros_like(heatmaps)
        for kp_idx in range(self.n_keypoints):
            # Normalize each keypoint's heatmap to probabilities
            heatmap_flat = heatmaps[kp_idx].flatten()
            heatmap_flat_shifted = heatmap_flat - np.max(heatmap_flat)
            probs_flat = np.exp(heatmap_flat_shifted) / np.sum(
                np.exp(heatmap_flat_shifted)
            )
            heatmaps_prob[kp_idx] = probs_flat.reshape(heatmaps[kp_idx].shape)

        # Merge all keypoint heatmaps (take maximum across keypoints)
        merged_heatmap_prob = np.max(heatmaps_prob, axis=0)  # (H, W)

        # Display merged heatmap with fixed vmin/vmax for xy heatmaps
        im = ax.imshow(merged_heatmap_prob, cmap=self.cmap, vmin=0, vmax=0.04)

        # Plot label points as prominent dots
        for kp_idx in range(self.n_keypoints):
            label_x = self.label_xy[frame_idx, kp_idx, 0] / self.stride_x
            label_y = self.label_xy[frame_idx, kp_idx, 1] / self.stride_y
            ax.scatter(
                label_x,
                label_y,
                color="white",
                s=self.marker_size,
                edgecolors="black",
                linewidth=1,
            )

        ax.set_aspect("equal")
        if variant_idx == 0:
            ax.set_ylabel("row (px)")
            ax.set_xlabel("column (px)")
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    def _plot_depth_distributions(self, ax, frame_idx: int, variant_idx: int):
        """Plot depth predictions as a simple heatmap showing distribution across keypoints for this variant"""
        # Get depth logits for this frame and variant
        depth_logits = self.pred_depth_logits[
            variant_idx, frame_idx
        ]  # (n_keypoints, depth_bins)

        # Convert logits to probabilities using more stable computation
        depth_logits_shifted = depth_logits - np.max(
            depth_logits, axis=1, keepdims=True
        )
        depth_probs = np.exp(depth_logits_shifted) / np.sum(
            np.exp(depth_logits_shifted), axis=1, keepdims=True
        )

        # # Debug: Print some statistics
        # print(f"Frame {frame_idx}, Variant {variant_idx}:")
        # print(f"  Depth logits range: [{depth_logits.min():.2f}, {depth_logits.max():.2f}]")
        # print(f"  Depth probs range: [{depth_probs.min():.6f}, {depth_probs.max():.6f}]")
        # print(f"  Keypoints with max prob < 0.1: {np.sum(depth_probs.max(axis=1) < 0.1)}/{self.n_keypoints}")

        # Use fixed vmin/vmax for depth distributions

        # Show as heatmap: depth bins (y-axis) vs keypoints (x-axis) - SWAPPED AXES
        im = ax.imshow(depth_probs.T, cmap=self.cmap, aspect="auto", vmin=0, vmax=0.5)

        # Compute depth bin centers for proper mapping
        depth_bin_centers = np.linspace(
            self.depth_min, self.depth_max, self.depth_n_bins
        )

        # Add ground truth depth as white dots
        for kp_idx in range(self.n_keypoints):
            label_depth = self.label_depth[frame_idx, kp_idx]
            # Convert depth to bin index using proper mapping
            depth_bin = np.argmin(np.abs(depth_bin_centers - label_depth))
            ax.scatter(
                kp_idx,
                depth_bin,
                color="white",
                s=self.marker_size,
                marker="o",
                edgecolors="black",
                linewidth=1,
            )

        # Add white vertical lines after each leg (every 6 keypoints)
        for leg_end in range(6, self.n_keypoints, 6):
            ax.axvline(x=leg_end - 0.5, color="white", linewidth=1)

        # Set labels and ticks
        n_legs = 6
        n_keypoints_per_leg = 5
        n_antennae = 2
        x_ticks = [
            (n_keypoints_per_leg * i) + (n_keypoints_per_leg / 2) - 0.5
            for i in range(n_legs)
        ] + [n_legs * n_keypoints_per_leg + (n_antennae / 2) - 0.5]
        k_ticklabels = [f"{side}{pos}" for side in "LR" for pos in "FMH"] + ["A"]
        ax.set_xticks(x_ticks)
        if variant_idx == 0:
            ax.set_xticklabels(k_ticklabels)
            ax.set_ylabel("depth (mm)")
            # Limit y-ticks (now for depth bins)
            n_depth_bins = depth_logits.shape[1]
            if n_depth_bins > 10:
                tick_indices = range(0, n_depth_bins, max(1, n_depth_bins // 5))
                ax.set_yticks(tick_indices)
                # Show actual depth values on y-axis
                ax.set_yticklabels(
                    [f"{depth_bin_centers[i]:.1f}" for i in tick_indices], fontsize=8
                )
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    def _get_keypoint_color(self, keypoint_name: str) -> np.ndarray:
        """Get color for a keypoint based on kchain_plotting_colors dictionary"""
        if keypoint_name in ["LPedicel", "RPedicel"]:
            # Map antenna pedicels to antenna colors
            antenna_key = "LAntenna" if keypoint_name.startswith("L") else "RAntenna"
            return kchain_plotting_colors[antenna_key]
        else:
            # Use first two characters (leg identifier)
            leg_key = keypoint_name[:2]
            return kchain_plotting_colors.get(
                leg_key, np.array([0.5, 0.5, 0.5])
            )  # Default gray

    def _plot_3d_skeleton(
        self,
        ax,
        frame_idx: int,
        variant_idx: int | None,
        show_ground_truth: bool = True,
        camera_elevation: float = 30.0,
        max_abs_azimuth: float = 30.0,
        azimuth_rotation_period: float = 300.0,
    ):
        """Plot 3D skeleton with connections between keypoints"""
        # Define skeleton connections for legs only (excluding last 2 antennae)
        skeleton_connections = self._get_skeleton_connections()

        # Plot ground truth first (so predictions appear on top when they overlap)
        if show_ground_truth:
            # Plot ground truth skeleton
            gt_points = self.label_world_xyz[frame_idx]  # (n_keypoints, 3)

            # Plot leg connections with colors from kchain_plotting_colors
            for connection in skeleton_connections:
                if (
                    connection[0] < self.n_keypoints
                    and connection[1] < self.n_keypoints
                ):
                    start_point = gt_points[connection[0]]
                    end_point = gt_points[connection[1]]

                    # Get color for this connection based on the keypoint
                    keypoint_name = self.keypoints_order[connection[0]]

                    ax.plot3D(
                        [start_point[0], end_point[0]],
                        [start_point[1], end_point[1]],
                        [start_point[2], end_point[2]],
                        color="gray",
                        linewidth=3,
                    )

            # Plot only antennae keypoints as scatter points (last 2 keypoints)
            if self.n_keypoints >= 2:
                for i in range(2):  # Last 2 keypoints
                    antenna_idx = self.n_keypoints - 2 + i
                    antenna_point = gt_points[antenna_idx]
                    keypoint_name = self.keypoints_order[antenna_idx]

                    ax.scatter(
                        antenna_point[0],
                        antenna_point[1],
                        antenna_point[2],
                        color="gray",
                        s=self.marker_size,
                    )

        # Plot predictions on top (so they appear over ground truth when overlapping)
        if variant_idx is not None:
            # Plot predicted skeleton
            pred_points = self.pred_world_xyz[
                variant_idx, frame_idx
            ]  # (n_keypoints, 3)

            # Plot leg connections with colors from kchain_plotting_colors
            for connection in skeleton_connections:
                if (
                    connection[0] < self.n_keypoints
                    and connection[1] < self.n_keypoints
                ):
                    start_point = pred_points[connection[0]]
                    end_point = pred_points[connection[1]]

                    # Get color for this connection based on the keypoint
                    keypoint_name = self.keypoints_order[connection[0]]
                    line_color = self._get_keypoint_color(keypoint_name)

                    ax.plot3D(
                        [start_point[0], end_point[0]],
                        [start_point[1], end_point[1]],
                        [start_point[2], end_point[2]],
                        color=line_color,
                        linewidth=2,
                    )

            # Plot only antennae keypoints as scatter points (last 2 keypoints)
            if self.n_keypoints >= 2:
                for i in range(2):  # Last 2 keypoints
                    antenna_idx = self.n_keypoints - 2 + i
                    antenna_point = pred_points[antenna_idx]
                    keypoint_name = self.keypoints_order[antenna_idx]
                    point_color = self._get_keypoint_color(keypoint_name)

                    ax.scatter(
                        antenna_point[0],
                        antenna_point[1],
                        antenna_point[2],
                        color=point_color,
                        s=self.marker_size,
                        label=(
                            f"Prediction {variant_idx}" if i == 0 else ""
                        ),  # Only label once
                    )

        # Set labels and limits
        if variant_idx == 0:
            ax.set_xlabel("ant.-post. (mm)")
            ax.set_ylabel("lateral (mm)")
            ax.set_zlabel("dors.-vent. (mm)", rotation=90)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

        # Set equal aspect ratio and reasonable limits
        all_points = self.label_world_xyz[frame_idx]
        if variant_idx is not None:
            all_points = np.vstack(
                [all_points, self.pred_world_xyz[variant_idx, frame_idx]]
            )

        # Calculate ranges for each dimension
        x_range = np.ptp(all_points[:, 0])
        y_range = np.ptp(all_points[:, 1])
        z_range = np.ptp(all_points[:, 2])

        # Use the maximum range to ensure equal scaling
        max_range = max(x_range, y_range, z_range) / 2.0

        # Center points for each dimension
        mid_x = np.mean(all_points[:, 0])
        mid_y = np.mean(all_points[:, 1])
        mid_z = np.mean(all_points[:, 2])

        # Set equal limits for all axes
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Ensure equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Set viewing angle - rotate 90 degrees clockwise around z-axis
        azimuth = (
            np.cos(2 * np.pi * frame_idx / azimuth_rotation_period) * max_abs_azimuth
        )
        ax.view_init(elev=camera_elevation, azim=azimuth)

    def _get_skeleton_connections(self):
        """Define skeleton connections between keypoints on the same leg (excluding antennae)"""
        # Connect keypoints within the same leg based on first two characters of name
        # The keypoints are ordered from proximal to distal within each leg
        # Exclude the last 2 keypoints (antennae)

        connections = []

        # Group keypoints by leg (first two characters) - exclude last 2 antennae
        leg_groups = {}
        leg_keypoints = self.keypoints_order[:-2]  # Exclude last 2 antennae keypoints

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
            try:
                frame = imageio.imread(str(frame_file))
                size = frame.shape[:2]  # (height, width)
                frame_sizes[i] = size
                if target_size is None:
                    target_size = size
                elif size != target_size:
                    logger.warning(
                        f"Frame size inconsistency detected: frame {i} has size {size}, expected {target_size}"
                    )
            except Exception as e:
                logger.warning(f"Could not read frame {i} for size check: {e}")

        logger.info(f"Target frame size: {target_size}")

        try:
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
                    try:
                        # Read frame using imageio (supports more formats than cv2)
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

                    except Exception as e:
                        logger.warning(f"Failed to write frame {i} ({frame_file}): {e}")

            logger.info(
                f"Successfully wrote {successful_frames}/{len(frame_files)} frames to video"
            )

            if successful_frames == 0:
                raise RuntimeError("Failed to write any frames to video")

        except Exception as e:
            logger.error(f"Video writing failed: {e}")
            raise
