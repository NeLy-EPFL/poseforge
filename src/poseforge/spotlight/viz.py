import matplotlib
import seaborn as sns
import numpy as np
import h5py
import cv2
import cmasher as cmr
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec

from poseforge.util.plot import configure_matplotlib_style


configure_matplotlib_style()


def draw_mask_contours(
    image: np.ndarray,
    masks: np.ndarray,
    color: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
    muscle_vrange: tuple[int, int] = (200, 1000),
    output_path: Path | None = None,
) -> np.ndarray:
    """
    Draw contours of binary masks on an image.

    Args:
        image: Input image, must be of shape (H, W) or (H, W, 3).
            If (H, W), the image will be rendered in grayscale from the
            range [vmin, vmax]. If (H, W, 3), the image is assumed to be in
            RGB and have pixel value in the range [0, 1].
        masks: Binary mask(s) as a HxW numpy array, must be of shape
            (n_masks, H, W) or (H, W).
        color: Color for the contours as (B, G, R) tuples. The range (0-1
            vs. 0-255) should be consistent with the input image.
            If None, defaults to red. If a single color is given, all masks
            will be drawn in that color. If a list of colors is given, it
            must match the number of masks.
        muscle_vrange: Min and max values for rendering grayscale images.
            Only used if image is (H, W).
        output_path: If provided, saves the output image to this path.

    Returns:
        Image with contours drawn.
    """
    masks = (masks > 0).astype(np.uint8)  # make sure masks are 0/1 in uint8

    # If image is grayscale, apply vmin/vmax and convert to 3-channel
    if image.ndim == 2:
        display_img = np.repeat(image[:, :, None], 3, axis=2).astype(np.float32)
        muscle_vmin, muscle_vmax = muscle_vrange
        muscle_norm = (display_img - muscle_vmin) / (muscle_vmax - muscle_vmin)
        display_img = np.clip(muscle_norm, 0, 1)
    else:
        if image.min() < 0 or image.max() > 1:
            raise ValueError("3-channel image must have pixel values in [0, 1]")
        display_img = image.copy().astype(np.float32)

    # Handle flexible number of masks and colors
    if len(masks.shape) == 2:
        masks = masks[None, ...]  # add channel dimension
    elif len(masks.shape) != 3:
        raise ValueError("masks must be of shape (H, W) or (n_masks, H, W)")

    n_masks = masks.shape[0]
    if color is None:
        colors = [(1, 0, 0)] * n_masks  # default to red
    elif isinstance(color, tuple):
        colors = [color] * n_masks  # make it a list
    else:
        colors = color
    assert len(colors) == n_masks

    # Draw contours for each mask
    display_img = np.ascontiguousarray(display_img[:, :, ::-1])  # RGB to BGR for OpenCV
    for i in range(n_masks):
        mask = masks[i]
        color_bgr = colors[i][::-1]  # convert RGB to BGR
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            display_img,
            contours,
            -1,
            color=color_bgr,
            thickness=1,
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), (display_img * 255).astype(np.uint8))

    return display_img[:, :, ::-1]  # convert back to RGB


def draw_template_matching_viz(
    muscle_image,
    foreground_mask,
    x_shift,
    y_shift,
    corr_matrix,
    search_limit,
    output_path,
    muscle_vrange: tuple[int, int] = (200, 1000),
):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), tight_layout=True)
    vmin, vmax = muscle_vrange

    axs[0].imshow(muscle_image, cmap=cmr.cm.nuclear, vmin=vmin, vmax=vmax)
    foreground_mask_nobg = foreground_mask.astype(np.float32)
    foreground_mask_nobg[foreground_mask == 0] = np.nan
    axs[0].imshow(foreground_mask_nobg, cmap=cmr.cm.amber, vmin=0, vmax=1, alpha=0.3)
    axs[0].set_title("Before alignment")
    axs[0].axis("off")

    foreground_mask_nobg_shifted = ndimage.shift(
        foreground_mask_nobg,
        shift=(y_shift, x_shift),
        order=0,
        mode="nearest",
        cval=np.nan,
    )
    axs[1].imshow(muscle_image, cmap=cmr.cm.nuclear, vmin=vmin, vmax=vmax)
    axs[1].imshow(foreground_mask_nobg_shifted, vmin=0, vmax=1, alpha=0.3)
    axs[1].set_title(f"After alignment")
    axs[1].axis("off")

    extent = [
        -search_limit,
        -search_limit + corr_matrix.shape[0] - 1,
        -search_limit + corr_matrix.shape[1] - 1,
        -search_limit,
    ]
    axs[2].imshow(corr_matrix, cmap="viridis", extent=extent, origin="upper")
    axs[2].set_title("Correlation matrix")
    legend = f"Best match: ({x_shift}, {y_shift})"
    axs[2].plot(x_shift, y_shift, "go", markersize=10, label=legend)
    axs[2].legend(loc="upper left")
    axs[2].set_xticks([-search_limit, 0, search_limit])
    axs[2].set_yticks([-search_limit, 0, search_limit])
    axs[2].set_xlabel("X shift (pixels)")
    axs[2].set_ylabel("Y shift (pixels)")

    fig.savefig(output_path)
    plt.close(fig)


def plot_muscle_traces_with_kinematics(
    muscle_segmentation_data_path: Path,
    keypoints3d_data_path: Path,
    inverse_kinematics_data_path: Path,
    output_path: Path,
    muscle_segments_to_plot: list[str] = ["Femur"],
    title: str = "",
    behavior_fps: float = 300.0,
    f0_percentile: float = 10.0,
    trange: tuple[float, float] | None = None,
    display: bool = False,
):
    if not muscle_segmentation_data_path.is_file():
        raise FileNotFoundError(
            f"Muscle segmentation data file not found: {muscle_segmentation_data_path}"
        )
    if not keypoints3d_data_path.is_file():
        raise FileNotFoundError(
            f"3D keypoints data file not found: {keypoints3d_data_path}"
        )
    if not inverse_kinematics_data_path.is_file():
        raise FileNotFoundError(
            f"Inverse kinematics data file not found: {inverse_kinematics_data_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # * Load muscle traces
    _traces_data_li = []
    with h5py.File(muscle_segmentation_data_path, "r") as f:
        muscle_roi_names = list(f.attrs["muscle_traces_segments"])
        for key in f.keys():
            grp = f[key]
            behavior_frame_id = grp.attrs["behavior_frame_id"]
            time = behavior_frame_id / behavior_fps
            if trange is not None and not trange[0] <= time <= trange[1]:
                continue
            muscle_traces = grp["muscle_traces"][:]

            # Load ROI sizes (we might want to filter out small ROIs later)
            roi_sizes = [
                grp.attrs[f"n_pixels_pre_dilation_{label}"]
                for label in muscle_roi_names
            ]
            roi_sizes = np.array(roi_sizes)
            data = {"muscle_traces": muscle_traces, "roi_sizes": roi_sizes}
            _traces_data_li.append((behavior_frame_id, data))

    _traces_data_li.sort(key=lambda x: x[0])  # sort by behavior frame id
    traces_data_lookup = {frame_id: data for frame_id, data in _traces_data_li}
    behavior_frame_ids_muscle_available = np.array(list(traces_data_lookup.keys()))

    # * Load 3D keypoints predictions
    _keypoints3d_data_li = []
    with h5py.File(keypoints3d_data_path, "r") as f:
        behavior_frame_ids_all_premask = f["frame_ids"][:]
        times = behavior_frame_ids_all_premask / behavior_fps
        if trange is None:
            within_trange_mask = np.ones_like(times, dtype=bool)
        else:
            within_trange_mask = (times >= trange[0]) & (times <= trange[1])
        behavior_frame_ids_all = behavior_frame_ids_all_premask[within_trange_mask]
        keypoints_xyz_all = f["keypoints_world_xyz"][:]
        keypoints_xyz_all = keypoints_xyz_all[within_trange_mask]
        keypoint_names_all = list(f["keypoints_world_xyz"].attrs["keypoints"])
        assert len(behavior_frame_ids_all) == keypoints_xyz_all.shape[0]
        assert len(keypoint_names_all) == keypoints_xyz_all.shape[1]

        # Include end effectors (claws) only
        is_claw_mask = np.array([name.endswith("Claw") for name in keypoint_names_all])
        keypoint_names_claws = [
            name for name, is_claw in zip(keypoint_names_all, is_claw_mask) if is_claw
        ]
        keypoint_xyz_claws = keypoints_xyz_all[:, is_claw_mask, :]

        # Add data by behavior frame id
        for i, frame_id in enumerate(behavior_frame_ids_all):
            _keypoints3d_data_li.append((frame_id, keypoint_xyz_claws[i, ...]))

    _keypoints3d_data_li.sort(key=lambda x: x[0])
    keypoints3d_data_lookup = {frame_id: kp for frame_id, kp in _keypoints3d_data_li}
    if trange is None:
        trange = (
            behavior_frame_ids_all[0] / behavior_fps,
            behavior_frame_ids_all[-1] / behavior_fps,
        )

    # * Load inverse kinematics results
    _joint_angles_data_li = []
    with h5py.File(inverse_kinematics_data_path, "r") as f:
        assert (f["frame_ids"][:] == behavior_frame_ids_all_premask).all()
        joint_angles_all = f["joint_angles"][:]
        joint_angles_all = joint_angles_all[within_trange_mask]
        dof_names_per_leg = f["joint_angles"].attrs["dof_names_per_leg"]
        legs = f["joint_angles"].attrs["legs"]
        for i, frame_id in enumerate(behavior_frame_ids_all):
            _joint_angles_data_li.append((frame_id, joint_angles_all[i, ...]))
    _joint_angles_data_li.sort(key=lambda x: x[0])
    joint_angles_data_lookup = {
        frame_id: angles for frame_id, angles in _joint_angles_data_li
    }

    # * Check shape consistency
    _zeroth_behavior_frame_id_for_muscle = list(traces_data_lookup.keys())[0]
    _zeroth_muscle_data = traces_data_lookup[_zeroth_behavior_frame_id_for_muscle]
    n_muscle_segments = len(_zeroth_muscle_data["muscle_traces"])
    assert _zeroth_muscle_data["muscle_traces"].shape == (n_muscle_segments,)
    assert _zeroth_muscle_data["roi_sizes"].shape == (n_muscle_segments,)
    _zeroth_behavior_frame_id = list(keypoints3d_data_lookup.keys())[0]
    _zeroth_keypoints3d_data = keypoints3d_data_lookup[_zeroth_behavior_frame_id]
    n_claws = _zeroth_keypoints3d_data.shape[0]
    assert _zeroth_keypoints3d_data.shape == (n_claws, 3)
    assert len(keypoint_names_claws) == n_claws
    _zeroth_invkin_data = joint_angles_data_lookup[_zeroth_behavior_frame_id]
    assert _zeroth_invkin_data.shape == (len(legs), len(dof_names_per_leg))

    # * Fixed plotting parameters
    n_rows_per_leg = 3  # claw pos, joint angles, muscle traces
    height_per_time_series_keypoint_pos = 0.3
    height_per_time_series_joint_angles = 0.3
    height_per_time_series_muscle_traces = 0.3
    hspace_within_grid = 0.1
    height_per_grid = (
        3 * height_per_time_series_keypoint_pos
        + len(dof_names_per_leg) * height_per_time_series_joint_angles
        + len(muscle_segments_to_plot) * height_per_time_series_muscle_traces
        + (n_rows_per_leg - 3) * hspace_within_grid
    )
    height_ratio_per_grid = [
        3 * height_per_time_series_keypoint_pos,
        len(dof_names_per_leg) * height_per_time_series_joint_angles,
        len(muscle_segments_to_plot) * height_per_time_series_muscle_traces,
    ]
    width_per_grid = 6
    hspace_between_grids = 0.06
    wspace_between_grids = 0.09
    fig_width_total = width_per_grid * 2 + hspace_between_grids
    fig_height_total = height_per_grid * 3 + wspace_between_grids * 2
    left_bound_pos = 0.13
    right_bound_pos = 0.93
    top_bound_pos = 0.94
    bottom_bound_pos = 0.05
    width_per_grid = (right_bound_pos - left_bound_pos - wspace_between_grids) / 2
    height_per_grid = (top_bound_pos - bottom_bound_pos - hspace_between_grids * 2) / 3
    xyz_amp = 2  # mm
    joint_angle_amp = 135  # deg
    muscle_trace_amp = 0.7  # DF/F0
    ylabel_offset = 0.18
    scale_bar_padding_frac_x = 0.02
    scale_bar_padding_frac_y = 0.05
    xyz_scale_bar_size = 1  # mm
    joint_angle_scale_bar_size = 90  # deg
    muscle_trace_scale_bar_size = 0.5  # DF/F0
    y_topbottom_padding_pct_of_amp = 0.3
    # fmt: off
    # xyz_colors = ["#000000", "#333333", "#666666"]
    # joint_angle_colors = ["midnightblue", "mediumblue", "slateblue", "darkviolet", "mediumvioletred", "indianred", "firebrick"]
    # muscle_colors = ["darkgreen", "seagreen", "mediumaquamarine", "darkturquoise"]
    # fmt: on
    xyz_colors = sns.color_palette("Greys_r", n_colors=6).as_hex()[:3]
    joint_angle_colors = sns.color_palette(
        "Blues_r", n_colors=len(dof_names_per_leg) * 2
    ).as_hex()[: len(dof_names_per_leg)]
    muscle_colors = sns.color_palette(
        "Greens_r", n_colors=len(muscle_segments_to_plot) * 2
    ).as_hex()[: len(muscle_segments_to_plot)]

    # * Set up plotting
    fig = plt.figure(figsize=(fig_width_total, fig_height_total))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    axes_all = {}
    for i, leg_name in enumerate(legs):
        # Figure out left-right span
        col = i // 3
        left = left_bound_pos + col * (width_per_grid + wspace_between_grids)
        right = left + width_per_grid

        # Figure out top-bottom span
        row = 2 - i % 3  # 2 - *: draw legs from top (gridspecs starts from bottom)
        bottom = bottom_bound_pos + row * (height_per_grid + hspace_between_grids)
        top = bottom + height_per_grid

        gs = GridSpec(
            n_rows_per_leg,
            1,
            height_ratios=height_ratio_per_grid,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            hspace=hspace_within_grid,
        )
        axes_this_leg = {}
        axes_this_leg["claw_position"] = fig.add_subplot(gs[0, 0])
        axes_this_leg["joint_angles"] = fig.add_subplot(gs[1, 0])
        axes_this_leg["muscle_traces"] = fig.add_subplot(gs[2, 0])
        axes_all[leg_name] = axes_this_leg

        side_full = {"L": "left", "R": "right"}[leg_name[0]]
        pos_full = {"F": "front", "M": "middle", "H": "hind"}[leg_name[1]]
        fig.text(
            (left + right) / 2,
            top,
            f"{side_full} {pos_full} leg".capitalize(),
            ha="center",
            va="bottom",
            fontsize=12,
            # fontweight="bold",
        )

    timepts_behavior = np.array(behavior_frame_ids_all) / behavior_fps
    timepts_muscle = np.array(behavior_frame_ids_muscle_available) / behavior_fps
    if trange is None:
        trange = (timepts_behavior[0], timepts_behavior[-1])

    # * Plot data for each leg
    for leg_idx, leg_name in enumerate(legs):
        axes_this_leg = axes_all[leg_name]

        # Plot claw positions
        claw_pos_ts = [
            keypoints3d_data_lookup[frame_id][leg_idx, :]
            for frame_id in behavior_frame_ids_all
        ]
        claw_pos_ts = np.array(claw_pos_ts)  # (n_timepoints, 3)
        ax = axes_this_leg["claw_position"]
        for i, dim in enumerate("xyz"):
            ts = claw_pos_ts[:, i]
            ts -= ts.mean()
            y_offset = i * xyz_amp
            color = xyz_colors[i]
            ax.plot(timepts_behavior, ts + y_offset, color=color, clip_on=False)
        padding = xyz_amp * y_topbottom_padding_pct_of_amp
        ax.set_ylim(xyz_amp * 2.5 + padding, -xyz_amp * 0.5 - padding)  # invert y-axis
        ax.set_yticks([i * xyz_amp for i in range(3)])
        ax.set_yticklabels(["front/back", "lateral", "height"])
        if leg_name[0] == "L":
            ax.set_ylabel("Claw pos.\n(mm)", labelpad=0)
            ax.yaxis.set_label_coords(x=-ylabel_offset, y=0.5, transform=ax.transAxes)
        for i, label in enumerate(ax.get_yticklabels()):
            color = xyz_colors[i]
            label.set_color(color)
        ax.set_xticklabels([])
        ax.tick_params(axis="y", which="both", length=0)
        ax.set_xlim(trange)
        if leg_name[0] == "R":
            _draw_scale_bar(
                ax,
                xyz_scale_bar_size,
                f"{xyz_scale_bar_size} mm",
                padding_frac_x=scale_bar_padding_frac_x,
                padding_frac_y=scale_bar_padding_frac_y,
            )
        sns.despine(ax=ax)

        # Plot joint angles
        ax = axes_this_leg["joint_angles"]
        for i, dof_name in enumerate(dof_names_per_leg):
            ts = np.array(
                [
                    np.rad2deg(joint_angles_data_lookup[frame_id][leg_idx, i])
                    for frame_id in behavior_frame_ids_all
                ]
            )
            ts -= ts.mean()
            y_offset = i * joint_angle_amp
            color = joint_angle_colors[i]
            ax.plot(timepts_behavior, ts + y_offset, color=color, clip_on=False)
        padding = joint_angle_amp * y_topbottom_padding_pct_of_amp
        ax.set_ylim(
            joint_angle_amp * (len(dof_names_per_leg) - 0.5) + padding,
            -joint_angle_amp * 0.5 - padding,
        )  # invert y-axis
        ax.set_yticks([i * joint_angle_amp for i in range(len(dof_names_per_leg))])
        ax.set_yticklabels([x.replace("_", " ") for x in dof_names_per_leg])
        for i, label in enumerate(ax.get_yticklabels()):
            color = joint_angle_colors[i]
            label.set_color(color)
        ax.tick_params(axis="y", which="both", length=0)
        ax.set_xlim(trange)
        if leg_name[0] == "L":
            ax.set_ylabel("Joint angles\n(°)", labelpad=0)
            ax.yaxis.set_label_coords(x=-ylabel_offset, y=0.5, transform=ax.transAxes)
        ax.set_xticklabels([])
        if leg_name[0] == "R":
            _draw_scale_bar(
                ax,
                joint_angle_scale_bar_size,
                f"{joint_angle_scale_bar_size}°",
                padding_frac_x=scale_bar_padding_frac_x,
                padding_frac_y=scale_bar_padding_frac_y,
            )
        sns.despine(ax=ax)

        # Plot muscle traces (DF/F)
        ax = axes_this_leg["muscle_traces"]
        for i, muscle_segment in enumerate(muscle_segments_to_plot):
            seg_idx = muscle_roi_names.index(f"{leg_name}{muscle_segment}")
            muscle_traces_ts = np.array(
                [
                    traces_data_lookup[frame_id]["muscle_traces"][seg_idx]
                    for frame_id in behavior_frame_ids_muscle_available
                ]
            )
            f0 = np.nanpercentile(muscle_traces_ts, f0_percentile)
            dff = (muscle_traces_ts - f0) / f0
            ts = dff - np.nanmean(dff)
            y_offset = i * muscle_trace_amp
            color = muscle_colors[i]
            ax.plot(timepts_muscle, ts + y_offset, color=color, clip_on=False)
        padding = muscle_trace_amp * y_topbottom_padding_pct_of_amp
        ax.set_ylim(
            muscle_trace_amp * (len(muscle_segments_to_plot) - 0.5) + padding,
            -muscle_trace_amp * 0.5 - padding,
        )  # invert y-axis
        ax.set_yticks(
            [i * muscle_trace_amp for i in range(len(muscle_segments_to_plot))]
        )
        ax.set_yticklabels([f"{x.lower()} ROI" for x in muscle_segments_to_plot])
        for i, label in enumerate(ax.get_yticklabels()):
            color = muscle_colors[i]
            label.set_color(color)
        ax.tick_params(axis="y", which="both", length=0)
        ax.set_xlim(trange)
        xticks = ax.get_xticks()
        # matplotlib wants xticklables to be set only after xticks are set
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x - trange[0]) for x in ax.get_xticks()])
        ax.set_xlabel("Time (s)")
        if leg_name[0] == "L":
            ax.set_ylabel("Muscles\n(ΔF/F)", labelpad=0)
            ax.yaxis.set_label_coords(x=-ylabel_offset, y=0.5, transform=ax.transAxes)
        if leg_name[0] == "R":
            _draw_scale_bar(
                ax,
                muscle_trace_scale_bar_size,
                f"{muscle_trace_scale_bar_size}",
                padding_frac_x=scale_bar_padding_frac_x,
                padding_frac_y=scale_bar_padding_frac_y,
            )
        sns.despine(ax=ax)

    fig.savefig(output_path)
    if display:
        plt.show(block=True)
    plt.close(fig)


def _draw_scale_bar(ax, scale_bar_size, text, *, padding_frac_x, padding_frac_y):
    tmin, tmax = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    tspan = tmax - tmin
    yspan = ybottom - ytop  # because y-axis is inverted
    scalebar_x = tmin + tspan * (1 + padding_frac_x)
    scalebar_bottom = ybottom - padding_frac_y * yspan
    ax.plot(
        [scalebar_x, scalebar_x],
        [scalebar_bottom, scalebar_bottom - scale_bar_size],
        color="black",
        linewidth=2,
        clip_on=False,
    )
    ax.text(
        scalebar_x,
        scalebar_bottom - scale_bar_size / 2,
        f"  {text}",
        ha="left",
        va="center",
    )
