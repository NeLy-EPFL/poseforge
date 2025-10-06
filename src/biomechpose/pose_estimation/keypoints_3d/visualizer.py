import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import tempfile
import shutil

from biomechpose.util import configure_matplotlib_style


configure_matplotlib_style()


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
        # video for each style transfer model can be found at
        # synthetic_videos_dir / f"translated_{model_name}.mp4"

        n_cols = (
            1 + self.n_variants
        )  # 1 for original + n_variants for style transfer models

        # Pre-load all video frames for efficiency
        print("Loading video frames...")
        all_video_frames = self._load_all_video_frames(sim_rendering_path, synthetic_videos_dir)
        print(f"Loaded {len(all_video_frames)} video streams")

        # Set up figure and subplots (4 rows now) with equal row heights
        fig = plt.figure(figsize=(n_cols * 4, 4 * 4))
        
        # Set equal row heights using gridspec
        gs = fig.add_gridspec(4, n_cols, height_ratios=[1, 1, 1, 1], hspace=0.3)

        # Create frames directory for debugging (no temp folder)
        frames_dir = Path("plot_frames")
        frames_dir.mkdir(exist_ok=True)
        print(f"DEBUG: Frames will be saved to: {frames_dir.absolute()}")

        try:
            # Generate frames
            for frame_idx in range(self.n_frames):
                fig.clear()
                
                # Recreate gridspec after clearing
                gs = fig.add_gridspec(4, n_cols, height_ratios=[1, 1, 1, 1], hspace=0.3)

                # Row 0: Videos (original + synthetic)
                for col_idx in range(n_cols):
                    ax = fig.add_subplot(gs[0, col_idx])
                    if col_idx == 0:
                        # Original simulation video
                        if 'original' in all_video_frames and frame_idx < len(all_video_frames['original']):
                            video_frame = all_video_frames['original'][frame_idx]
                            if video_frame is not None:
                                ax.imshow(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))
                                ax.set_title("Original Simulation")
                            else:
                                ax.text(0.5, 0.5, f"Frame {frame_idx}\nNot Available", 
                                       ha="center", va="center", transform=ax.transAxes)
                        else:
                            ax.text(0.5, 0.5, f"Original Video\nNot Found", 
                                   ha="center", va="center", transform=ax.transAxes)
                    else:
                        # Synthetic video from style transfer model
                        model_idx = col_idx - 1
                        model_key = f'synthetic_{model_idx}'
                        if model_key in all_video_frames and frame_idx < len(all_video_frames[model_key]):
                            video_frame = all_video_frames[model_key][frame_idx]
                            if video_frame is not None:
                                ax.imshow(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))
                                ax.set_title(f"Style Transfer {model_idx}")
                            else:
                                ax.text(0.5, 0.5, f"Frame {frame_idx}\nNot Available", 
                                       ha="center", va="center", transform=ax.transAxes)
                        else:
                            ax.text(0.5, 0.5, f"Synthetic Video\nModel {model_idx}\nNot Found", 
                                   ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Row 1: Ground truth heatmap + predicted heatmaps
                for col_idx in range(n_cols):
                    ax = fig.add_subplot(gs[1, col_idx])
                    if col_idx == 0:
                        # Show ground truth heatmap in first column
                        self._plot_ground_truth_heatmap(ax, frame_idx)
                        ax.set_title("Ground Truth Heatmap")
                    else:
                        # Show merged heatmaps for all keypoints
                        variant_idx = col_idx - 1
                        self._plot_merged_heatmaps(ax, frame_idx, variant_idx)
                        ax.set_title(f"Pred Heatmaps - Model {variant_idx}")

                # Row 2: Depth distributions
                for col_idx in range(n_cols):
                    ax = fig.add_subplot(gs[2, col_idx])
                    if col_idx == 0:
                        # Show label depth distribution in first column
                        self._plot_label_depth_distribution(ax, frame_idx)
                        ax.set_title("Label Depth Distribution")
                    else:
                        # Show depth distributions as heatmaps
                        variant_idx = col_idx - 1
                        self._plot_depth_distributions(ax, frame_idx, variant_idx)
                        ax.set_title(f"Depth Dist - Model {variant_idx}")

                # Row 3: 3D body skeleton
                for col_idx in range(n_cols):
                    ax = fig.add_subplot(gs[3, col_idx], projection="3d")
                    if col_idx == 0:
                        # Show only ground truth skeleton
                        self._plot_3d_skeleton(
                            ax, frame_idx, None, show_ground_truth=True
                        )
                        ax.set_title("Ground Truth 3D")
                    else:
                        # Show prediction + ground truth skeleton
                        variant_idx = col_idx - 1
                        self._plot_3d_skeleton(
                            ax, frame_idx, variant_idx, show_ground_truth=True
                        )
                        ax.set_title(f"3D Skeleton - Model {variant_idx}")

                plt.tight_layout()

                # Save frame
                frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
                fig.savefig(frame_path, dpi=100, bbox_inches="tight")

                if frame_idx % 10 == 0:
                    print(f"Generated frame {frame_idx}/{self.n_frames}")

            # Convert frames to video using OpenCV
            self._frames_to_video(frames_dir, output_path)

        except Exception as e:
            print(f"Error generating video: {e}")
            print(f"Frames saved in: {frames_dir.absolute()}")
            raise
        
        print(f"DEBUG: Frames preserved in: {frames_dir.absolute()}")
        plt.close(fig)
        print(f"Video saved to: {output_path}")

    def _load_all_video_frames(self, sim_rendering_path: Path, synthetic_videos_dir: Path) -> dict:
        """Efficiently load all video frames into memory for fast access"""
        all_frames = {}
        
        # Load original simulation video
        if sim_rendering_path.exists():
            print(f"Loading original video: {sim_rendering_path}")
            all_frames['original'] = self._load_video_frames(sim_rendering_path)
        else:
            print(f"Warning: Original video not found at {sim_rendering_path}")
            all_frames['original'] = []
        
        # Load synthetic videos for each style transfer model
        for model_idx, model_name in enumerate(self.style_transfer_models):
            synthetic_video_path = synthetic_videos_dir / f"translated_{model_name}.mp4"
            model_key = f'synthetic_{model_idx}'
            
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
        
        cap.release()
        print(f"Loaded {len(frames)} frames from {video_path}")
        return frames

    def _plot_merged_heatmaps(self, ax, frame_idx: int, variant_idx: int):
        """Plot merged heatmaps for all keypoints with label dots"""
        # Get heatmaps for this frame and variant
        heatmaps = self.pred_xy_heatmaps[variant_idx, frame_idx]  # (n_keypoints, H, W)

        # Merge all keypoint heatmaps (take maximum across keypoints)
        merged_heatmap = np.max(heatmaps, axis=0)  # (H, W)

        # Display merged heatmap with dynamic vmax using viridis colormap
        vmax = merged_heatmap.max()  # Use dynamic vmax (removed the *2.0 multiplier)
        im = ax.imshow(merged_heatmap, cmap="viridis", alpha=0.7, vmax=vmax)

        # Plot label points as prominent dots
        for kp_idx in range(self.n_keypoints):
            label_x = self.label_xy[frame_idx, kp_idx, 0] / self.stride_x
            label_y = self.label_xy[frame_idx, kp_idx, 1] / self.stride_y
            ax.scatter(
                label_x, label_y, color="cyan", s=50, edgecolors="black", linewidth=2
            )

        ax.set_xlim(0, merged_heatmap.shape[1])
        ax.set_ylim(merged_heatmap.shape[0], 0)  # Flip y-axis for image coordinates
        ax.set_aspect("equal")
        
        # Add colorbar
        try:
            cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        except:
            pass  # Skip colorbar if it causes issues

    def _plot_ground_truth_heatmap(self, ax, frame_idx: int):
        """Plot ground truth heatmap generated from label coordinates"""
        # Get the heatmap dimensions from predictions (assuming same size)
        heatmap_shape = self.pred_xy_heatmaps.shape[-2:]  # (H, W)
        
        # Create ground truth heatmap by placing Gaussians at label positions
        gt_heatmap = np.zeros(heatmap_shape)
        sigma = 2.0  # Standard deviation for Gaussian blobs
        
        for kp_idx in range(self.n_keypoints):
            # Get label position in heatmap coordinates
            label_x = self.label_xy[frame_idx, kp_idx, 0] / self.stride_x
            label_y = self.label_xy[frame_idx, kp_idx, 1] / self.stride_y
            
            # Skip if coordinates are out of bounds
            if (label_x < 0 or label_x >= heatmap_shape[1] or 
                label_y < 0 or label_y >= heatmap_shape[0]):
                continue
                
            # Create meshgrid for Gaussian
            y_coords, x_coords = np.mgrid[0:heatmap_shape[0], 0:heatmap_shape[1]]
            
            # Calculate Gaussian centered at label position
            gaussian = np.exp(-((x_coords - label_x)**2 + (y_coords - label_y)**2) / (2 * sigma**2))
            
            # Add to ground truth heatmap (take maximum to handle overlapping keypoints)
            gt_heatmap = np.maximum(gt_heatmap, gaussian)
        
        # Display ground truth heatmap with dynamic vmax using viridis colormap
        vmax = gt_heatmap.max()  # Use dynamic vmax (removed the *2.0 multiplier)
        im = ax.imshow(gt_heatmap, cmap="viridis", alpha=0.7, vmax=vmax)
        
        # Plot label points as cyan dots for reference
        for kp_idx in range(self.n_keypoints):
            label_x = self.label_xy[frame_idx, kp_idx, 0] / self.stride_x
            label_y = self.label_xy[frame_idx, kp_idx, 1] / self.stride_y
            # Only plot if within bounds
            if (0 <= label_x < heatmap_shape[1] and 0 <= label_y < heatmap_shape[0]):
                ax.scatter(
                    label_x, label_y, color="cyan", s=50, edgecolors="black", linewidth=2
                )

        ax.set_xlim(0, heatmap_shape[1])
        ax.set_ylim(heatmap_shape[0], 0)  # Flip y-axis for image coordinates
        ax.set_aspect("equal")
        
        # Add colorbar
        try:
            cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        except:
            pass  # Skip colorbar if it causes issues

    def _plot_depth_distributions(self, ax, frame_idx: int, variant_idx: int):
        """Plot depth predictions as a simple heatmap showing distribution across keypoints for this variant"""
        # Get depth logits for this frame and variant
        depth_logits = self.pred_depth_logits[
            variant_idx, frame_idx
        ]  # (n_keypoints, depth_bins)

        # Convert logits to probabilities using more stable computation
        depth_logits_shifted = depth_logits - np.max(depth_logits, axis=1, keepdims=True)
        depth_probs = np.exp(depth_logits_shifted) / np.sum(
            np.exp(depth_logits_shifted), axis=1, keepdims=True
        )

        # # Debug: Print some statistics
        # print(f"Frame {frame_idx}, Variant {variant_idx}:")
        # print(f"  Depth logits range: [{depth_logits.min():.2f}, {depth_logits.max():.2f}]")
        # print(f"  Depth probs range: [{depth_probs.min():.6f}, {depth_probs.max():.6f}]")
        # print(f"  Keypoints with max prob < 0.1: {np.sum(depth_probs.max(axis=1) < 0.1)}/{self.n_keypoints}")

        # Use a better colormap scaling - set vmin to a small value to highlight differences
        vmin = max(depth_probs.min(), 1e-6)  # Avoid log(0) issues
        vmax = depth_probs.max()
        
        # Show as heatmap: depth bins (y-axis) vs keypoints (x-axis) - SWAPPED AXES
        im = ax.imshow(depth_probs.T, cmap='viridis', aspect='auto', alpha=0.8, 
                      vmin=vmin, vmax=vmax)
        
        # Compute depth bin centers for proper mapping
        depth_bin_centers = np.linspace(self.depth_min, self.depth_max, self.depth_n_bins)
        
        # Add ground truth depth as red dots (coordinates swapped)
        for kp_idx in range(self.n_keypoints):
            label_depth = self.label_depth[frame_idx, kp_idx]
            # Convert depth to bin index using proper mapping
            depth_bin = np.argmin(np.abs(depth_bin_centers - label_depth))
            ax.scatter(kp_idx, depth_bin, color='red', s=30, marker='o', 
                      edgecolors='white', linewidth=1)

        # Add white vertical lines after each leg (every 6 keypoints)
        for leg_end in range(6, self.n_keypoints, 6):
            ax.axvline(x=leg_end - 0.5, color='white', linewidth=2, alpha=0.8)

        # Set labels and ticks - swapped axes
        ax.set_xlabel("Keypoint Index")
        ax.set_ylabel("Depth Bins")
        ax.set_title(f"Depth Distribution - Variant {variant_idx}")
        
        # Show all keypoints as tick marks but no labels
        ax.set_xticks(range(self.n_keypoints))
        ax.set_xticklabels([])  # No keypoint names to avoid crowding
        
        # Limit y-ticks (now for depth bins)
        n_depth_bins = depth_logits.shape[1]
        if n_depth_bins > 10:
            tick_indices = range(0, n_depth_bins, max(1, n_depth_bins // 5))
            ax.set_yticks(tick_indices)
            # Show actual depth values on y-axis
            ax.set_yticklabels([f"{depth_bin_centers[i]:.1f}" for i in tick_indices], 
                              fontsize=8)
        
        # Add colorbar if there's space
        try:
            cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label='Probability')
            cbar.ax.tick_params(labelsize=8)
        except:
            pass  # Skip colorbar if it causes issues

    def _plot_label_depth_distribution(self, ax, frame_idx: int):
        """Plot label depth distribution in same format as prediction depth distributions"""
        # Get the label depths for this frame
        label_depths = self.label_depth[frame_idx]  # (n_keypoints,)
        
        # Create a depth distribution matrix similar to predictions
        # Initialize with zeros: (n_keypoints, n_depth_bins)
        label_depth_probs = np.zeros((self.n_keypoints, self.depth_n_bins))
        
        # Create depth bin centers for mapping
        depth_bin_centers = np.linspace(self.depth_min, self.depth_max, self.depth_n_bins)
        
        # For each keypoint, put probability 1.0 at the correct depth bin
        for kp_idx in range(self.n_keypoints):
            label_depth = label_depths[kp_idx]
            # Find nearest depth bin
            depth_bin = np.argmin(np.abs(depth_bin_centers - label_depth))
            label_depth_probs[kp_idx, depth_bin] = 1.0
        
        # Use same format as prediction plots - show as heatmap with swapped axes
        im = ax.imshow(label_depth_probs.T, cmap='viridis', aspect='auto', alpha=0.8, 
                      vmin=0, vmax=1.0)
        
        # Add the same ground truth depth as red dots (which are the same as the data)
        for kp_idx in range(self.n_keypoints):
            label_depth = label_depths[kp_idx]
            # Convert depth to bin index using proper mapping
            depth_bin = np.argmin(np.abs(depth_bin_centers - label_depth))
            ax.scatter(kp_idx, depth_bin, color='red', s=30, marker='o', 
                      edgecolors='white', linewidth=1)

        # Add white vertical lines after each leg (every 6 keypoints)
        for leg_end in range(6, self.n_keypoints, 6):
            ax.axvline(x=leg_end - 0.5, color='white', linewidth=2, alpha=0.8)

        # Set labels and ticks - same format as prediction plots
        ax.set_xlabel("Keypoint Index")
        ax.set_ylabel("Depth Bins")
        ax.set_title("Label Depth Distribution")
        
        # Show all keypoints as tick marks but no labels
        ax.set_xticks(range(self.n_keypoints))
        ax.set_xticklabels([])  # No keypoint names to avoid crowding
        
        # Limit y-ticks (for depth bins)
        if self.depth_n_bins > 10:
            tick_indices = range(0, self.depth_n_bins, max(1, self.depth_n_bins // 5))
            ax.set_yticks(tick_indices)
            # Show actual depth values on y-axis
            ax.set_yticklabels([f"{depth_bin_centers[i]:.1f}" for i in tick_indices], 
                              fontsize=8)
        
        # Add colorbar if there's space
        try:
            cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label='Probability')
            cbar.ax.tick_params(labelsize=8)
        except:
            pass  # Skip colorbar if it causes issues

    def _plot_3d_skeleton(
        self,
        ax,
        frame_idx: int,
        variant_idx: int | None,
        show_ground_truth: bool = True,
    ):
        """Plot 3D skeleton with connections between keypoints"""
        # Define skeleton connections for legs only (excluding last 2 antennae)
        skeleton_connections = self._get_skeleton_connections()

        # Plot ground truth first (so predictions appear on top when they overlap)
        if show_ground_truth:
            # Plot ground truth skeleton
            gt_points = self.label_world_xyz[frame_idx]  # (n_keypoints, 3)
            
            # Always use black for labels (changed from red for column 0)
            line_color = "black"
            point_color = "black"
            
            # Plot leg connections (lines only, no scatter points for legs)
            for connection in skeleton_connections:
                if (
                    connection[0] < self.n_keypoints
                    and connection[1] < self.n_keypoints
                ):
                    start_point = gt_points[connection[0]]
                    end_point = gt_points[connection[1]]
                    ax.plot3D(
                        [start_point[0], end_point[0]],
                        [start_point[1], end_point[1]],
                        [start_point[2], end_point[2]],
                        color=line_color,
                        linewidth=3,
                        alpha=0.8,
                    )

            # Plot only antennae keypoints as scatter points (last 2 keypoints)
            if self.n_keypoints >= 2:
                antennae_points = gt_points[-2:]  # Last 2 keypoints
                ax.scatter(
                    antennae_points[:, 0],
                    antennae_points[:, 1],
                    antennae_points[:, 2],
                    color=point_color,
                    s=30,
                    alpha=0.8,
                )

        # Plot predictions on top (so they appear over ground truth when overlapping)
        if variant_idx is not None:
            # Plot predicted skeleton
            pred_points = self.pred_world_xyz[
                variant_idx, frame_idx
            ]  # (n_keypoints, 3)

            # Plot leg connections (lines only, no scatter points for legs)
            for connection in skeleton_connections:
                if (
                    connection[0] < self.n_keypoints
                    and connection[1] < self.n_keypoints
                ):
                    start_point = pred_points[connection[0]]
                    end_point = pred_points[connection[1]]
                    ax.plot3D(
                        [start_point[0], end_point[0]],
                        [start_point[1], end_point[1]],
                        [start_point[2], end_point[2]],
                        color="tab:blue",
                        linewidth=2,
                    )

            # Plot only antennae keypoints as scatter points (last 2 keypoints)
            if self.n_keypoints >= 2:
                antennae_points = pred_points[-2:]  # Last 2 keypoints
                ax.scatter(
                    antennae_points[:, 0],
                    antennae_points[:, 1],
                    antennae_points[:, 2],
                    color="tab:blue",
                    s=30,
                    label=f"Prediction {variant_idx}",
                )

        # Set labels and limits
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")

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
        ax.set_box_aspect([1,1,1])
        
        # Set viewing angle - rotate 90 degrees clockwise around z-axis
        # Default elev=30, azim=45. Clockwise rotation around z-axis means azim -= 90
        ax.view_init(elev=30, azim=45-90)  # azim=-45 for 90° clockwise rotation

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
        """Convert frame images to video using OpenCV"""
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if not frame_files:
            raise ValueError("No frame files found")

        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        height, width, channels = first_frame.shape

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            out.write(frame)

        out.release()
        cv2.destroyAllWindows()
