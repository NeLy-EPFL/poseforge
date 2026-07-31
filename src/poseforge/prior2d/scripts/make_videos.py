#!/usr/bin/env python
"""Render annotated overview videos for a sample of continuous good periods.

Works on both a plain periods `.h5` (see `extract_continuous_periods_from_h5.py`)
and an IK/FK-augmented periods `.h5` (see `solve_ik.py`). For each trial,
renders its `--periods-per-trial` longest periods as short videos.

Without `--with-ik`, the raw predictions (`pred_2d_px`) are drawn with the
`kchain_plotting_colors` per-leg color convention (non-leg nodes in gray).
With `--with-ik`, raw predictions are instead drawn as a thin white line with
small white dots (thorax omitted, since the IK layer already draws it), the
IK forward-kinematics result (`fk_2d_px`) is drawn on the same frame with
`kchain_plotting_colors`, and the output gains a second panel to the right: a
synthetic 3D view of the IK reconstruction (`fk_3d_mm`, `kchain_plotting_colors`
with a gray thorax dot) plus a 1mm-spaced ground-plane grid, drawn on a black
canvas with an orthographic camera that recenters on the thorax and tracks
the fly's heading (yaw only, not pitch/roll) every frame -- the fly stays
visually fixed on screen while the grid rotates underneath it as the fly
turns. Drawing is OpenCV only (no matplotlib); videos are read/written via
`pvio`, encoded on the GPU (NVENC) with a few in parallel via joblib.

Usage:
    python make_videos.py \
        lm_ported_v000_trained000_filtered_periods_ikfk.h5 \
        bulk_data/prior-2dinvkin/sleap/lm_ported/period_videos \
        --with-ik
"""

import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import pvio
import tyro
from joblib import Parallel, delayed
from loguru import logger

from poseforge.neuromechfly.constants import kchain_plotting_colors
from poseforge.prior2d.skeleton_viz import (
    LINE_THICKNESS,
    OTHER_COLOR,
    POINT_RADIUS,
    build_monochrome_skeleton,
    build_skeleton,
    draw_pose,
)

DATA_ROOT = Path("/mnt/upramdya_data/VAS/poseforge_paper_data")
ALIGNED_VIDEO_RELPATH = Path("processed/aligned_behavior_video.mkv")

# `kchain_plotting_colors` values are RGB floats in [0, 1]; `pvio` reads/writes
# frames in RGB too (unlike OpenCV's usual BGR), so no channel reordering is
# needed here -- these tuples are used directly as OpenCV draw colors against
# `pvio`-native RGB frames.
KCHAIN_COLORS = {
    leg: tuple(round(c * 255) for c in rgb.tolist())
    for leg, rgb in kchain_plotting_colors.items()
}
WHITE = (255, 255, 255)
RAW_LINE_THICKNESS = 1
RAW_POINT_RADIUS = 3  # smaller than skeleton_viz.POINT_RADIUS's 5
POINT_RADIUS_3D = 3  # smaller than skeleton_viz.POINT_RADIUS's 5, for the 3D panel

# mm from thorax the 3D panel should comfortably fit, same for every video
# (a fixed camera distance -- "equidistant from the fly's center" -- rather
# than a per-period adaptive scale) so relative fly/pose size is directly
# comparable across periods and trials.
MM_RADIUS_TO_FIT = 2.8
SCALE_MARGIN = 0.9  # fraction of half-panel-size that MM_RADIUS_TO_FIT maps to

# The free-floating root's Z never moves away from its neutral 0 (XYView
# observations give it no Z constraint), so a fixed world Z offset below it
# is as good a ground-plane approximation as any per-frame contact estimate.
FLOOR_Z_OFFSET_MM = -2.0
GRID_SPACING_MM = 1.0
# The grid is world-XY-aligned but the camera's right/up axes are
# fly-heading-dependent (see `compute_camera_orientation`), and the oblique
# (pitched) view foreshortens a horizontal plane's extent along the screen's
# "up" direction much more than along "right" -- so a grid sized to just the
# panel's own visible radius would often run out of the panel along one axis
# well before the other, depending on heading. A large fixed multiple of the
# panel's visible half-extent keeps the grid comfortably covering the panel
# (and visibly beyond it) at any heading.
FLOOR_HALF_SIZE_MM = 3.5 * MM_RADIUS_TO_FIT / SCALE_MARGIN
GRID_COLOR = (45, 45, 45)  # dark gray; no fill, only grid lines are drawn
GRID_LINE_THICKNESS = 1


def compute_camera_orientation(
    fk_3d_mm_frame: np.ndarray, thc_idxs: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the orthographic camera's orientation for one frame.

    The camera is a "45-deg right-front, -30-deg pitch" oblique view relative
    to the fly's heading, derived from the front-leg vs. hind-leg ThC
    (leg-base) centroid, since the non-leg nodes (N, A, LA, RA, LW, RW) have
    no `fk_3d_mm` value to use instead. Recomputed every frame from that
    frame's own heading (not just the period's first frame), so the camera's
    azimuth tracks the fly's current yaw: the fly stays visually fixed on
    screen while the ground-plane grid (world-XY-fixed) appears to rotate
    underneath it as it turns. Pitch (elevation) is always a fixed 30
    degrees, so only yaw is tracked -- this method only ever reads the
    heading's XY projection, so it never captures the fly's own pitch/roll
    in the first place.

    Args:
        fk_3d_mm_frame: `(n_nodes, 3)` fk_3d_mm for one frame.
        thc_idxs: `(lf_idx, rf_idx, lh_idx, rh_idx)` indices of the LF/RF/LH/RH
            ThC (leg-base) nodes, into `fk_3d_mm_frame`'s node axis.

    Returns:
        right_axis: `(3,)` unit vector spanning the image plane's x axis.
        up_axis: `(3,)` unit vector spanning the image plane's y axis.
        view_dir: `(3,)` unit vector pointing from the fly toward the
            camera; used only for depth-sorting, not projection, since this
            is an orthographic camera.
    """
    lf_idx, rf_idx, lh_idx, rh_idx = thc_idxs
    front = (fk_3d_mm_frame[lf_idx] + fk_3d_mm_frame[rf_idx]) / 2
    hind = (fk_3d_mm_frame[lh_idx] + fk_3d_mm_frame[rh_idx]) / 2
    forward_xy = (front - hind)[:2]
    forward_xy = forward_xy / np.linalg.norm(forward_xy)
    forward = np.array([forward_xy[0], forward_xy[1], 0.0])

    z_hat = np.array([0.0, 0.0, 1.0])
    # The fly's "right" is `forward` rotated -90 deg about Z (viewed from
    # above, i.e. (x, y) -> (y, -x)); an arbitrary but fixed handedness
    # choice for this synthetic view, since there's no ground-truth answer.
    right_body = np.array([forward[1], -forward[0], 0.0])

    azimuth_dir = forward + right_body
    azimuth_dir = azimuth_dir / np.linalg.norm(azimuth_dir)  # 45-deg right-front

    pitch = np.radians(30.0)
    # Camera elevated, looking down (-30-deg pitch).
    view_dir = np.cos(pitch) * azimuth_dir + np.sin(pitch) * z_hat
    view_dir = view_dir / np.linalg.norm(view_dir)

    right_axis = np.cross(view_dir, z_hat)
    right_axis = right_axis / np.linalg.norm(right_axis)
    up_axis = np.cross(right_axis, view_dir)  # already unit: inputs unit + orthogonal

    return right_axis, up_axis, view_dir


def project_relative_to_panel(
    centered: np.ndarray,
    right_axis: np.ndarray,
    up_axis: np.ndarray,
    panel_size: int,
) -> np.ndarray:
    """Orthographically project points already relative to the camera origin.

    Args:
        centered: `(..., 3)`, already offset by whatever origin the camera is
            centered on this frame (see `compute_camera_orientation`), may contain
            NaN.
        right_axis: See `compute_camera_orientation`.
        up_axis: See `compute_camera_orientation`.
        panel_size: Panel width/height in pixels (square panel);
            `MM_RADIUS_TO_FIT` mm from the origin maps to `SCALE_MARGIN` of
            the panel's half-size, the same for every call regardless of
            period or frame.

    Returns:
        `(..., 2)` pixel (x, y) coordinates, NaN preserved where the input
        was NaN.
    """
    screen_x = centered @ right_axis
    screen_y = centered @ up_axis
    scale = SCALE_MARGIN * (panel_size / 2) / MM_RADIUS_TO_FIT
    px_x = panel_size / 2 + screen_x * scale
    px_y = panel_size / 2 - screen_y * scale  # flip: image row grows downward
    return np.stack([px_x, px_y], axis=-1)


def compute_grid_lines(
    right_axis: np.ndarray, up_axis: np.ndarray, panel_size: int
) -> np.ndarray:
    """Project the ground-plane grid's line segments into panel pixels.

    The grid is centered under whatever origin the camera follows each frame
    (the current thorax position), at a fixed world Z offset below it (see
    `FLOOR_Z_OFFSET_MM`), spanning `FLOOR_HALF_SIZE_MM` along world X/Y with
    lines every `GRID_SPACING_MM`. Since the camera's orientation now tracks
    the fly's yaw every frame (see `compute_camera_orientation`), this must
    be recomputed every frame too, not just once per period.

    Args:
        right_axis: See `compute_camera_orientation`.
        up_axis: See `compute_camera_orientation`.
        panel_size: Panel width/height in pixels (square panel).

    Returns:
        `(n_lines, 2, 2)` int32 pixel coordinates: one `(start, end)` pair of
        `(x, y)` points per line segment.
    """
    h = FLOOR_HALF_SIZE_MM
    offsets = np.arange(-h, h + GRID_SPACING_MM / 2, GRID_SPACING_MM)
    segments_world = [
        seg
        for offset in offsets
        for seg in (
            [[-h, offset, FLOOR_Z_OFFSET_MM], [h, offset, FLOOR_Z_OFFSET_MM]],
            [[offset, -h, FLOOR_Z_OFFSET_MM], [offset, h, FLOOR_Z_OFFSET_MM]],
        )
    ]
    screen = project_relative_to_panel(
        np.array(segments_world), right_axis, up_axis, panel_size
    )
    return np.round(screen).astype(np.int32)


def draw_grid_floor(canvas: np.ndarray, grid_lines: np.ndarray) -> None:
    """Draw the ground-plane grid lines onto canvas, in place (no fill).

    Args:
        canvas: Panel image to draw onto.
        grid_lines: `(n_lines, 2, 2)` int32 pixel coordinates (see
            `compute_grid_lines`).
    """
    for (x1, y1), (x2, y2) in grid_lines:
        cv2.line(
            canvas, (x1, y1), (x2, y2), GRID_COLOR, GRID_LINE_THICKNESS, cv2.LINE_AA
        )


def draw_fk_3d_panel(
    canvas: np.ndarray,
    points: np.ndarray,
    depths: np.ndarray,
    edges: list,
    point_colors: list,
) -> None:
    """Draw the IK 3D panel's skeleton onto canvas, in place, depth-sorted.

    Unlike `draw_pose` (which always draws all edges, then all points),
    edges and points are interleaved and drawn in ascending
    depth-away-from-camera order (farthest first), so nearer parts of the
    fly draw on top of farther ones where legs cross in this oblique view.

    Args:
        canvas: Panel image to draw onto.
        points: `(n_nodes, 2)` pixel coordinates (see
            `project_relative_to_panel`), may contain NaN.
        depths: `(n_nodes,)` depth-away-from-camera per node (see
            `compute_camera_orientation`'s `view_dir`), most negative farthest.
        edges: List of `(idx_a, idx_b, color)` tuples, as built by
            `build_skeleton`.
        point_colors: List of colors, one per node.
    """
    primitives = [
        (min(depths[a], depths[b]), "edge", (a, b, color)) for a, b, color in edges
    ]
    primitives += [(depths[idx], "point", idx) for idx in range(len(points))]
    primitives.sort(key=lambda primitive: primitive[0])

    for _, kind, payload in primitives:
        if kind == "edge":
            a, b, color = payload
            pa, pb = points[a], points[b]
            if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
                continue
            cv2.line(
                canvas,
                (round(pa[0]), round(pa[1])),
                (round(pb[0]), round(pb[1])),
                color,
                LINE_THICKNESS,
                cv2.LINE_AA,
            )
        else:
            pt = points[payload]
            if np.any(np.isnan(pt)):
                continue
            cv2.circle(
                canvas,
                (round(pt[0]), round(pt[1])),
                POINT_RADIUS_3D,
                point_colors[payload],
                -1,
                cv2.LINE_AA,
            )


def select_longest_periods(
    trial_group: h5py.Group, periods_per_trial: int
) -> list[str]:
    """Pick the `periods_per_trial` longest period ids in one trial's group.

    Args:
        trial_group: The `<genotype>/<fly_trial>` group.
        periods_per_trial: Maximum number of periods to select.

    Returns:
        Period ids (group keys), longest first.
    """
    period_ids = list(trial_group.keys())
    period_ids.sort(
        key=lambda pid: (
            trial_group[pid].attrs["end_idx"] - trial_group[pid].attrs["start_idx"]
        ),
        reverse=True,
    )
    return period_ids[:periods_per_trial]


def build_period_frames(
    item: dict, video_height: int
) -> tuple[list[np.ndarray], float]:
    """Build one period's annotated 2D(+3D) frames, without writing them out.

    Args:
        item: Dict with `video_path`, `start_idx`, `end_idx`, `layers` (list
            of `(points, edges, point_colors, line_thickness, point_radius)`
            drawn on the left panel, each as accepted by `draw_pose`;
            `points` is `(period_length, n_nodes, 2)`), and `panel_3d`:
            `None`, or `(fk_3d_mm, thc_idxs, edges, point_colors, th_idx)`
            for a second panel showing a synthetic 3D view of the IK
            reconstruction plus a ground-plane grid. `fk_3d_mm` is
            `(period_length, n_nodes, 3)`; `thc_idxs` is passed to
            `compute_camera_orientation` each frame; `th_idx` is the
            thorax's index into `fk_3d_mm`'s node axis, recentered on every
            frame so the view follows the fly.
        video_height: Output height in pixels; the left panel is the source
            video resized to this height (aspect ratio preserved), and, if
            `panel_3d` is given, the right panel is a `video_height` square.

    Returns:
        out_frames: One combined (left, or left+right) frame per period frame.
        fps: The source video's own frame rate (or 30.0 if unavailable).
    """
    frame_indices = list(range(item["start_idx"], item["end_idx"]))
    frames, fps = pvio.read_frames_from_video(item["video_path"], frame_indices)
    fps = fps or 30.0
    period_length = len(frames)
    src_height, src_width = frames[0].shape[:2]
    left_width = round(src_width * video_height / src_height)

    panel_3d = item["panel_3d"]
    if panel_3d is not None:
        fk_3d_mm, thc_idxs, panel_3d_edges, panel_3d_colors, th_idx = panel_3d

    out_frames = []
    for local_idx in range(period_length):
        frame = frames[local_idx]
        for points, edges, point_colors, line_thickness, point_radius in item["layers"]:
            draw_pose(
                frame,
                points[local_idx],
                edges,
                point_colors,
                line_thickness,
                point_radius,
            )
        left_panel = cv2.resize(frame, (left_width, video_height))

        if panel_3d is None:
            out_frame = left_panel
        else:
            fk_3d_mm_frame = fk_3d_mm[local_idx]
            # Orientation tracks yaw every frame; grid must be recomputed
            # alongside it (see `compute_camera_orientation`/`compute_grid_lines`).
            right_axis, up_axis, view_dir = compute_camera_orientation(
                fk_3d_mm_frame, thc_idxs
            )
            grid_lines = compute_grid_lines(right_axis, up_axis, video_height)

            right_panel = np.zeros((video_height, video_height, 3), dtype=np.uint8)
            draw_grid_floor(right_panel, grid_lines)
            origin = fk_3d_mm_frame[th_idx]  # follow: recenter every frame
            centered = fk_3d_mm_frame - origin
            points_2d = project_relative_to_panel(
                centered, right_axis, up_axis, video_height
            )
            depths = centered @ view_dir
            draw_fk_3d_panel(
                right_panel, points_2d, depths, panel_3d_edges, panel_3d_colors
            )
            out_frame = np.hstack([left_panel, right_panel])

        out_frames.append(out_frame)

    return out_frames, fps


def compute_style_context(node_names: list[str]) -> dict:
    """Precompute the skeleton edge/color styles shared by every period.

    Args:
        node_names: SLEAP node names, matching `pred_2d_px`/`fk_2d_px`'s node
            axis (a periods `.h5`'s root `node_names` attr).

    Returns:
        Dict with `th_idx`, `thc_idxs`, `kchain_edges`, `kchain_colors`,
        `kchain_edges_3d`, `kchain_colors_3d`, `raw_edges`, `raw_colors` --
        see `build_work_item`.
    """
    th_idx = node_names.index("Th")
    thc_idxs = (
        node_names.index("LF_ThC"),
        node_names.index("RF_ThC"),
        node_names.index("LH_ThC"),
        node_names.index("RH_ThC"),
    )
    kchain_edges, kchain_colors = build_skeleton(node_names, leg_colors=KCHAIN_COLORS)
    # 3D panel only: thorax drawn gray rather than the default white hub,
    # matching the other non-leg keypoints (that panel has no separate raw
    # layer to distinguish it from).
    kchain_edges_3d, kchain_colors_3d = build_skeleton(
        node_names, leg_colors=KCHAIN_COLORS, hub_color=OTHER_COLOR
    )
    raw_edges, raw_colors = build_monochrome_skeleton(node_names, WHITE)
    return {
        "th_idx": th_idx,
        "thc_idxs": thc_idxs,
        "kchain_edges": kchain_edges,
        "kchain_colors": kchain_colors,
        "kchain_edges_3d": kchain_edges_3d,
        "kchain_colors_3d": kchain_colors_3d,
        "raw_edges": raw_edges,
        "raw_colors": raw_colors,
    }


def build_work_item(
    group: h5py.Group,
    video_path: Path,
    style: dict,
    with_ik: bool,
) -> dict:
    """Build one period's `layers`/`panel_3d` for `build_period_frames`.

    Args:
        group: One period's `<genotype>/<fly_trial>/<period_id>` group.
        video_path: Trial's aligned video path.
        style: See `compute_style_context`.
        with_ik: See `main`.

    Returns:
        Dict with `video_path`, `start_idx`, `end_idx`, `layers`, `panel_3d`
        (see `build_period_frames`).
    """
    start_idx = int(group.attrs["start_idx"])
    end_idx = int(group.attrs["end_idx"])

    panel_3d = None
    if with_ik:
        missing = [name for name in ("fk_2d_px", "fk_3d_mm") if name not in group]
        if missing:
            raise SystemExit(
                f"Period group {group.name} is missing {missing}; run solve_ik.py "
                "first, or drop --with-ik."
            )
        # Thorax omitted from the raw layer: the IK layer already draws it,
        # and overlapping the two dots at (near-)identical positions read as
        # one oversized dot.
        raw_points = group["pred_2d_px"][:]
        raw_points[:, style["th_idx"]] = np.nan
        layers = [
            (
                raw_points,
                style["raw_edges"],
                style["raw_colors"],
                RAW_LINE_THICKNESS,
                RAW_POINT_RADIUS,
            ),
            (
                group["fk_2d_px"][:],
                style["kchain_edges"],
                style["kchain_colors"],
                LINE_THICKNESS,
                POINT_RADIUS,
            ),
        ]
        fk_3d_mm = group["fk_3d_mm"][:]
        panel_3d = (
            fk_3d_mm,
            style["thc_idxs"],
            style["kchain_edges_3d"],
            style["kchain_colors_3d"],
            style["th_idx"],
        )
    else:
        layers = [
            (
                group["pred_2d_px"][:],
                style["kchain_edges"],
                style["kchain_colors"],
                LINE_THICKNESS,
                POINT_RADIUS,
            )
        ]

    return {
        "video_path": video_path,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "layers": layers,
        "panel_3d": panel_3d,
    }


def render_one_video(item: dict, crf: int, video_height: int) -> None:
    """Render one period's annotated video and write it to `item["output_path"]`.

    Args:
        item: See `build_period_frames`, plus `output_path`.
        crf: x264/NVENC-scale quality passed to `pvio.write_frames_to_video`.
        video_height: See `build_period_frames`.
    """
    out_frames, fps = build_period_frames(item, video_height)
    pvio.write_frames_to_video(
        item["output_path"], out_frames, fps, mode="gpu", quality=crf, quiet=True
    )


def main(
    periods_path: tyro.conf.Positional[Path],
    output_dir: tyro.conf.Positional[Path],
    with_ik: bool = False,
    data_root: Path = DATA_ROOT,
    periods_per_trial: int = 1,
    video_height: int = 448,
    crf: int = 23,
    n_jobs: int = 3,
) -> None:
    """Render annotated overview videos for a sample of continuous good periods.

    Args:
        periods_path: Periods `.h5` file, with or without IK/FK results (see
            `extract_continuous_periods_from_h5.py` and `solve_ik.py`).
        output_dir: Directory to save the rendered period videos to. Cleared
            first if it already exists, so stale videos from an earlier run
            (e.g. with different period ids after `solve_ik.py`
            re-segments) aren't left behind alongside the new ones.
        with_ik: If True, raw predictions are drawn as a thin white line with
            small white dots (thorax omitted, since the IK layer already
            draws it), the IK forward-kinematics result (`fk_2d_px`) is drawn
            with `kchain_plotting_colors` on the same left panel, and a right
            panel is added with a synthetic 3D view of the IK reconstruction
            (`fk_3d_mm`, also `kchain_plotting_colors`, but with a gray
            thorax dot) plus a ground-plane grid; requires `periods_path` to
            have IK/FK datasets. If False, raw predictions alone are drawn
            with `kchain_plotting_colors`.
        data_root: Root directory containing the aligned videos (read-only).
        periods_per_trial: Number of (longest) periods to render per trial.
        video_height: Output video height in pixels; see `render_one_video`.
            A multiple of 16 avoids `pvio`'s encoder silently padding/resizing
            the output (our square 900x900 source videos keep both panels
            exact multiples of 16 too, so nothing gets resized).
        crf: x264/NVENC-scale quality for the output videos; lower is higher
            quality (see `pvio.write_frames_to_video`).
        n_jobs: Number of videos encoded in parallel via joblib, each on the
            GPU (NVENC); kept modest since concurrent NVENC sessions are
            limited (a handful at once is typically the practical ceiling).
    """
    if not periods_path.is_file():
        raise SystemExit(f"Input file does not exist: {periods_path}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with h5py.File(periods_path, "r") as f:
        node_names = list(f.attrs["node_names"])
        style = compute_style_context(node_names)

        work_items = []
        for genotype in f:
            for fly_trial in f[genotype]:
                video_path = data_root / genotype / fly_trial / ALIGNED_VIDEO_RELPATH
                if not video_path.is_file():
                    logger.warning(
                        f"Skipping {genotype}/{fly_trial}: video not found at {video_path}"
                    )
                    continue

                trial_group = f[genotype][fly_trial]
                period_ids = select_longest_periods(trial_group, periods_per_trial)
                for period_id in period_ids:
                    item = build_work_item(
                        trial_group[period_id], video_path, style, with_ik
                    )
                    item["output_path"] = (
                        output_dir / f"{genotype}__{fly_trial}__period{period_id}_"
                        f"f{item['start_idx']}-{item['end_idx']}.mp4"
                    )
                    work_items.append(item)
                logger.info(
                    f"{genotype}/{fly_trial}: queued {len(period_ids)} period videos"
                )

    logger.info(f"Rendering {len(work_items)} period videos (n_jobs={n_jobs})")
    Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(render_one_video)(item, crf, video_height) for item in work_items
    )
    logger.info(f"Rendered {len(work_items)} period videos to {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
