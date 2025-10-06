import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
    ):
        self.preds = preds
        self.labels = labels
        self.keypoints_order = keypoints_order
        self.output_path = output_path
        self.data_freq = data_freq

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

    def make_keypoint_pos_timeseries_plot(self, output_path: Path) -> None:
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
        pass  # TODO