"""Interactive keypoint annotation GUI for validating pose predictions.

Load a video + a prediction h5, overlay all predicted keypoints on each frame,
navigate through frames, drag points to their true location, toggle occluded
keypoints, and save only the hand-corrected frames to a new h5 for later
ground-truth-vs-prediction comparison.

The tool is standalone and uses only existing dependencies (matplotlib with the
TkAgg backend, opencv, h5py, numpy). It must NOT import any poseforge module that
forces matplotlib's Agg backend (e.g. keypoints3d/visualizer.py), so the canonical
keypoint names are defined locally as a fallback.

Coordinates
-----------
Frames are shown at full (native) resolution. Predictions are stored in inference
pixel space (e.g. 256x256); that size is read from the h5 ``attrs["image_size"]``
or passed via ``--pred-resolution``. Predictions are scaled UP to full resolution
for display/editing, and all saved coordinates are in full-resolution pixel space.

Controls
--------
    Left-drag a point   : move it to the correct location
    Right-click a point : toggle occluded / visible (faded = occluded)
    v                   : toggle occluded/visible for the point under the cursor
    Mouse scroll        : zoom in/out, centered on the cursor
    Middle-drag         : pan the view
    f                   : fit / reset the view to the full frame
    l                   : show / hide keypoint labels
    Left / Right arrow  : previous / next frame (also p / n)
    r                   : reset the current frame to predictions
    s                   : save corrected frames to the output h5
    q                   : quit

Usage
-----
    python annotate_keypoints_gui.py VIDEO --h5 PREDS.h5 [--pred-resolution W H]

Output
------
A new h5 (``<h5_stem>_corrected.h5`` by default) containing only the modified
frames: ``frame_indices``, ``keypoints_xy`` (corrected GT, full-res),
``pred_keypoints_xy`` (original predictions, full-res), ``visible`` flags, and
metadata attributes.
"""

import argparse
import logging
from collections import OrderedDict
from pathlib import Path

# IMPORTANT: select an interactive backend BEFORE importing pyplot, and do not
# import anything that forces the Agg backend.
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402

import cv2  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Local fallback for keypoint names (32 = 6 legs x 5 + 2 pedicels). Defined here
# to avoid importing poseforge.neuromechfly.constants, which pulls in flygym.
_LEGS = [f"{side}{pos}" for side in "LR" for pos in "FMH"]
_LEG_KEYPOINTS = ["ThC", "CTr", "FTi", "TiTa", "Claw"]
KEYPOINT_NAMES_CANONICAL = [
    f"{leg}{link}" for leg in _LEGS for link in _LEG_KEYPOINTS
] + ["LPedicel", "RPedicel"]


# Per-frame state colors (RGBA).
COLOR_UNMODIFIED = np.array([1.0, 0.1, 0.1, 1.0])  # red
COLOR_MODIFIED = np.array([0.15, 1.0, 0.15, 1.0])  # lime
OCCLUDED_ALPHA = 0.15  # faded fill for occluded points


# --------------------------------------------------------------------------- #
# Data loading helpers
# --------------------------------------------------------------------------- #
def _decode_names(raw) -> list[str] | None:
    """Decode an h5 attribute holding keypoint names into a list[str]."""
    if raw is None:
        return None
    try:
        names = []
        for item in list(raw):
            if isinstance(item, bytes):
                names.append(item.decode("utf-8"))
            else:
                names.append(str(item))
        return names
    except TypeError:
        return None


def _read_predictions(
    h5_path: Path, xy_key: str
) -> tuple[np.ndarray, np.ndarray | None, list[str] | None, tuple[int, int] | None]:
    """Read predicted xy, frame ids, keypoint names, and inference size.

    Returns:
        pred_xy: (n, n_kp, 2) float32 in inference pixel space.
        frame_ids: (n,) int or None.
        kp_names: list[str] or None.
        infer_size: (W, H) or None.
    """
    with h5py.File(h5_path, "r") as f:
        # Resolve the xy dataset, with a fallback to the atomic-batch label key.
        if xy_key in f:
            dset = f[xy_key]
            pred_xy = np.asarray(dset[:], dtype=np.float32)
        
        else:
            raise KeyError(
                f"Dataset key '{xy_key}' not found in {h5_path}. Available keys: "
                f"{list(f.keys())}"
            )

        if pred_xy.ndim != 3 or pred_xy.shape[-1] != 2:
            raise ValueError(
                f"Expected predicted xy of shape (n, n_kp, 2), got {pred_xy.shape}"
            )

        frame_ids = None
        if "frame_ids" in f:
            frame_ids = np.asarray(f["frame_ids"][:]).astype(np.int64)

        kp_names = _decode_names(dset.attrs.get("keypoints"))

        infer_size = None
        raw_size = dset.attrs.get("image_size")
        if raw_size is not None:
            size = tuple(int(v) for v in np.asarray(raw_size).ravel()[:2])
            if len(size) == 2:
                infer_size = size  # (W, H)

    return pred_xy, frame_ids, kp_names, infer_size


class VideoFrames:
    """Random-access to RGB video frames, lazy by default.

    Lazy mode keeps memory bounded via an LRU cache and seeks with
    cv2.VideoCapture. Eager mode loads every frame into memory.
    """

    def __init__(self, video_path: Path, lazy: bool = True, cache_size: int = 64):
        self.path = str(video_path)
        self.lazy = lazy
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        if lazy:
            self._cap = cap
            self._next_expected = 0
            self._frames = None
        else:
            self._cap = None
            self._frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self._frames.append(frame[:, :, ::-1].copy())  # BGR -> RGB
            cap.release()
            self.n_frames = len(self._frames)

    def __len__(self) -> int:
        return self.n_frames

    def get(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.n_frames:
            raise IndexError(idx)
        if not self.lazy:
            return self._frames[idx]

        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        # Avoid an expensive seek when reading sequentially.
        if idx != self._next_expected:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        self._next_expected = idx + 1
        if not ret:
            raise RuntimeError(f"Failed to read frame {idx} from {self.path}")
        rgb = frame[:, :, ::-1].copy()  # BGR -> RGB

        self._cache[idx] = rgb
        self._cache.move_to_end(idx)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return rgb

    def close(self):
        if self._cap is not None:
            self._cap.release()


# --------------------------------------------------------------------------- #
# Annotation GUI
# --------------------------------------------------------------------------- #
class KeypointAnnotator:
    def __init__(
        self,
        frames: VideoFrames,
        pred_xy_full: np.ndarray,
        frame_ids: np.ndarray | None,
        kp_names: list[str],
        orig_size: tuple[int, int],
        infer_size: tuple[int, int],
        marker_radius: float,
        output_path: Path,
        source_video: Path,
        source_h5: Path,
        xy_key: str,
        blit: bool = False,
    ):
        self.frames = frames
        self.pred_xy_full = pred_xy_full  # (n, n_kp, 2), immutable
        self.corrected_xy = pred_xy_full.copy()
        self.n_frames, self.n_kp, _ = pred_xy_full.shape
        self.visible = np.ones((self.n_frames, self.n_kp), dtype=bool)
        self.modified = np.zeros(self.n_frames, dtype=bool)
        self.frame_ids = frame_ids
        self.kp_names = kp_names
        self.orig_size = orig_size  # (W, H)
        self.infer_size = infer_size  # (W, H)
        self.marker_radius = float(marker_radius)
        self.output_path = output_path
        self.source_video = source_video
        self.source_h5 = source_h5
        self.xy_key = xy_key
        self.blit = blit

        self.cur = 0
        self._dragging_idx: int | None = None

        # Free up keys that matplotlib binds by default so our handlers own them
        # (s=save, p=pan, o=zoom, r/h=home, left/right/v/c/backspace=nav stack,
        # l/k=axis scale, f=fullscreen). 'q' is intentionally left bound to quit.
        for key in (
            "keymap.save", "keymap.pan", "keymap.zoom", "keymap.home",
            "keymap.back", "keymap.forward", "keymap.grid", "keymap.fullscreen",
            "keymap.yscale", "keymap.xscale",
        ):
            plt.rcParams[key] = []

        # View / labels interaction state
        self._labels_visible = True
        self._panning = False

        # Figure / artists
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title("Keypoint annotator")
        self._im = self.ax.imshow(self.frames.get(0))
        xy = self.corrected_xy[0]
        marker_area = (1.6 * self.marker_radius) ** 2
        self._scatter = self.ax.scatter(
            xy[:, 0], xy[:, 1],
            s=marker_area,
            edgecolors=self._compute_colors(0),
            linewidths=1.0,
            zorder=3,
            marker="o",
            facecolors='none'
        )
        # Small text label next to each keypoint (constant pixel offset so it
        # stays readable at any zoom level).
        colors0 = self._compute_colors(0)
        self._labels = []
        for i in range(self.n_kp):
            ann = self.ax.annotate(
                self.kp_names[i],
                xy=(xy[i, 0], xy[i, 1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6,
                color=colors0[i],
                zorder=4,
                clip_on=True,
            )
            self._labels.append(ann)

        self.ax.set_xlim(0, self.orig_size[0])
        self.ax.set_ylim(self.orig_size[1], 0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self._instructions()

        # Events
        c = self.fig.canvas
        c.mpl_connect("button_press_event", self._on_press)
        c.mpl_connect("motion_notify_event", self._on_motion)
        c.mpl_connect("button_release_event", self._on_release)
        c.mpl_connect("key_press_event", self._on_key)
        c.mpl_connect("scroll_event", self._on_scroll)

        self._update_title()

    # ----- rendering ----- #
    def _instructions(self):
        self.fig.text(
            0.5, 0.005,
            "L-drag: move  |  R-click/'v': occlude  |  scroll: zoom  |  "
            "mid-drag: pan  |  f: fit  |  l: labels  |  ←/→: nav  |  "
            "r: reset  |  s: save  |  q: quit",
            ha="center", va="bottom", fontsize=8, color="0.3",
        )

    def _compute_colors(self, idx: int) -> np.ndarray:
        state = COLOR_MODIFIED if self.modified[idx] else COLOR_UNMODIFIED
        colors = np.tile(state, (self.n_kp, 1))
        occluded = ~self.visible[idx]
        colors[occluded, 3] = OCCLUDED_ALPHA
        return colors

    def _refresh_points(self):
        pts = self.corrected_xy[self.cur]
        colors = self._compute_colors(self.cur)
        self._scatter.set_offsets(pts)
        self._scatter.set_edgecolor(colors)
        for i, ann in enumerate(self._labels):
            ann.xy = (pts[i, 0], pts[i, 1])
            ann.set_color(colors[i])
            ann.set_visible(self._labels_visible)
        self.fig.canvas.draw_idle()

    def _update_title(self):
        status = "MODIFIED" if self.modified[self.cur] else "original"
        n_occ = int((~self.visible[self.cur]).sum())
        self.ax.set_title(
            f"Frame {self.cur + 1}/{self.n_frames}  |  {status}"
            f"  |  occluded: {n_occ}/{self.n_kp}"
            f"  |  {int(self.modified.sum())} frames annotated"
        )
        self.fig.canvas.draw_idle()

    # ----- navigation ----- #
    def _goto(self, idx: int):
        idx = max(0, min(self.n_frames - 1, idx))
        if idx == self.cur:
            return
        self.cur = idx
        self._im.set_data(self.frames.get(idx))
        self._refresh_points()
        self._update_title()

    def _reset_current(self):
        self.corrected_xy[self.cur] = self.pred_xy_full[self.cur]
        self.visible[self.cur] = True
        self.modified[self.cur] = False
        self._refresh_points()
        self._update_title()

    # ----- hit testing ----- #
    def _nearest_point(self, x: float, y: float) -> int | None:
        pts = self.corrected_xy[self.cur]
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        j = int(np.argmin(d))
        if d[j] <= self.marker_radius:
            return j
        return None

    def _mark_modified(self):
        if not self.modified[self.cur]:
            self.modified[self.cur] = True

    # ----- event handlers ----- #
    def _on_press(self, event):
        if event.button == 2:  # middle button -> start panning
            if event.x is None:
                return
            self._panning = True
            self._pan_px = (event.x, event.y)
            self._pan_xlim = self.ax.get_xlim()
            self._pan_ylim = self.ax.get_ylim()
            self._pan_inv = self.ax.transData.inverted()
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        j = self._nearest_point(event.xdata, event.ydata)
        if j is None:
            return
        if event.button == 3:  # right click -> toggle visibility
            self.visible[self.cur, j] = not self.visible[self.cur, j]
            self._mark_modified()
            self._refresh_points()
            self._update_title()
        elif event.button == 1:  # left click -> start drag
            self._dragging_idx = j

    def _on_motion(self, event):
        if self._panning:
            self._do_pan(event)
            return
        if self._dragging_idx is None:
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        self.corrected_xy[self.cur, self._dragging_idx] = (event.xdata, event.ydata)
        self._scatter.set_offsets(self.corrected_xy[self.cur])
        self._labels[self._dragging_idx].xy = (event.xdata, event.ydata)
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self._panning:
            self._panning = False
            return
        if self._dragging_idx is None:
            return
        self._dragging_idx = None
        self._mark_modified()
        self._refresh_points()
        self._update_title()

    # ----- zoom / pan ----- #
    def _on_scroll(self, event):
        """Cursor-centered zoom: scroll up = zoom in, down = zoom out."""
        if event.inaxes is not self.ax or event.xdata is None:
            return
        base = 1.2
        scale = 1.0 / base if event.button == "up" else base
        x, y = event.xdata, event.ydata
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(x - (x - x0) * scale, x + (x1 - x) * scale)
        # y limits are inverted (y0 > y1); the same arithmetic preserves that.
        self.ax.set_ylim(y - (y - y0) * scale, y + (y1 - y) * scale)
        self.fig.canvas.draw_idle()

    def _do_pan(self, event):
        if event.x is None:
            return
        # Map both pixel positions through the press-time transform so the
        # shift stays consistent for the whole gesture.
        x0d, y0d = self._pan_inv.transform(self._pan_px)
        x1d, y1d = self._pan_inv.transform((event.x, event.y))
        ddx, ddy = x1d - x0d, y1d - y0d
        self.ax.set_xlim(self._pan_xlim[0] - ddx, self._pan_xlim[1] - ddx)
        self.ax.set_ylim(self._pan_ylim[0] - ddy, self._pan_ylim[1] - ddy)
        self.fig.canvas.draw_idle()

    def _fit_view(self):
        self.ax.set_xlim(0, self.orig_size[0])
        self.ax.set_ylim(self.orig_size[1], 0)
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in ("right", "n"):
            self._goto(self.cur + 1)
        elif event.key in ("left", "p"):
            self._goto(self.cur - 1)
        elif event.key == "r":
            self._reset_current()
        elif event.key == "s":
            self._save()
        elif event.key == "f":
            self._fit_view()
        elif event.key == "l":
            self._labels_visible = not self._labels_visible
            for ann in self._labels:
                ann.set_visible(self._labels_visible)
            self.fig.canvas.draw_idle()
        elif event.key == "v":
            if event.inaxes is self.ax and event.xdata is not None:
                j = self._nearest_point(event.xdata, event.ydata)
                if j is not None:
                    self.visible[self.cur, j] = not self.visible[self.cur, j]
                    self._mark_modified()
                    self._refresh_points()
                    self._update_title()

    # ----- saving ----- #
    def _save(self):
        sel = np.nonzero(self.modified)[0]
        if sel.size == 0:
            logger.warning("No frames modified yet; nothing to save.")
            return

        with h5py.File(self.output_path, "w") as f:
            f.create_dataset(
                "frame_indices", data=sel.astype(np.int32), compression="gzip"
            )
            if self.frame_ids is not None:
                f.create_dataset(
                    "frame_ids",
                    data=self.frame_ids[sel].astype(np.int32),
                    compression="gzip",
                )
            f.create_dataset(
                "keypoints_xy",
                data=self.corrected_xy[sel].astype(np.float32),
                compression="gzip",
            )
            f.create_dataset(
                "pred_keypoints_xy",
                data=self.pred_xy_full[sel].astype(np.float32),
                compression="gzip",
            )
            f.create_dataset(
                "visible", data=self.visible[sel], compression="gzip"
            )
            f.attrs["keypoints"] = list(self.kp_names)
            f.attrs["full_resolution"] = list(self.orig_size)
            f.attrs["inference_size"] = list(self.infer_size)
            f.attrs["source_video"] = str(self.source_video)
            f.attrs["source_h5"] = str(self.source_h5)
            f.attrs["xy_key"] = self.xy_key
            f.attrs["n_modified"] = int(sel.size)
            f.attrs["n_total_frames"] = int(self.n_frames)

        logger.info(
            f"Saved {sel.size} corrected frame(s) to {self.output_path}"
        )

    def show(self):
        plt.show()
        self.frames.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _resolve_infer_size(
    cli_pred_resolution: tuple[int, int] | None,
    h5_infer_size: tuple[int, int] | None,
) -> tuple[int, int]:
    if cli_pred_resolution is not None:
        return cli_pred_resolution
    if h5_infer_size is not None:
        return h5_infer_size
    raise ValueError(
        "Inference resolution unknown: the h5 has no 'image_size' attribute. "
        "Please pass --pred-resolution W H (the size predictions were computed at)."
    )


def run(
    video_path: Path,
    h5_path: Path,
    xy_key: str = "pred_xy",
    pred_resolution: tuple[int, int] | None = None,
    output_path: Path | None = None,
    marker_radius: float = 2.0,
    eager: bool = False,
    blit: bool = False,
):
    video_path = Path(video_path).expanduser().resolve()
    h5_path = Path(h5_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not h5_path.is_file():
        raise FileNotFoundError(f"h5 not found: {h5_path}")

    pred_xy, frame_ids, kp_names_attr, h5_infer_size = _read_predictions(h5_path, xy_key)
    n_frames, n_kp, _ = pred_xy.shape

    # Resolve keypoint names: h5 attr -> canonical fallback -> generic.
    if kp_names_attr is not None and len(kp_names_attr) == n_kp:
        kp_names = kp_names_attr
    elif len(KEYPOINT_NAMES_CANONICAL) == n_kp:
        kp_names = KEYPOINT_NAMES_CANONICAL
    else:
        kp_names = [f"kp_{i}" for i in range(n_kp)]
        logger.warning(
            f"Could not match keypoint names to n_kp={n_kp}; using generic names."
        )

    # Load video.
    frames = VideoFrames(video_path, lazy=not eager)
    orig_size = (frames.width, frames.height)  # (W, H)
    if len(frames) != n_frames:
        logger.warning(
            f"Frame count mismatch: video has {len(frames)} frames, h5 has "
            f"{n_frames}. Assuming index alignment over the first "
            f"{min(len(frames), n_frames)}."
        )
        n_use = min(len(frames), n_frames)
        pred_xy = pred_xy[:n_use]
        if frame_ids is not None:
            frame_ids = frame_ids[:n_use]
        n_frames = n_use

    # Scale predictions from inference space up to full resolution.
    infer_size = _resolve_infer_size(pred_resolution, h5_infer_size)
    scale_x = orig_size[0] / infer_size[0]
    scale_y = orig_size[1] / infer_size[1]
    pred_xy_full = pred_xy.copy()
    pred_xy_full[..., 0] *= scale_x
    pred_xy_full[..., 1] *= scale_y
    logger.info(
        f"Video {orig_size[0]}x{orig_size[1]}, inference {infer_size[0]}x"
        f"{infer_size[1]}, scale ({scale_x:.3f}, {scale_y:.3f}), "
        f"{n_frames} frames, {n_kp} keypoints."
    )

    if output_path is None:
        output_path = h5_path.with_name(h5_path.stem + "_corrected.h5")
    output_path = Path(output_path)

    annotator = KeypointAnnotator(
        frames=frames,
        pred_xy_full=pred_xy_full,
        frame_ids=frame_ids,
        kp_names=kp_names,
        orig_size=orig_size,
        infer_size=infer_size,
        marker_radius=marker_radius,
        output_path=output_path,
        source_video=video_path,
        source_h5=h5_path,
        xy_key=xy_key,
        blit=blit,
    )
    logger.info("Window opening. Press 's' to save, 'q' to quit.")
    annotator.show()


def start():
    parser = argparse.ArgumentParser(
        description="Interactive GUI to correct predicted keypoints for validation."
    )
    parser.add_argument("video_path", type=Path, help="Input video file.")
    parser.add_argument(
        "--h5", type=Path, required=True, dest="h5_path",
        help="Prediction h5 file.",
    )
    parser.add_argument(
        "--xy-key", type=str, default="pred_xy",
        help="Dataset key for predicted xy (fallback: keypoint_pos[...,:2]).",
    )
    parser.add_argument(
        "--pred-resolution", type=int, nargs=2, default=None, metavar=("W", "H"),
        help="Inference resolution predictions were computed at. Overrides the "
        "h5 'image_size' attribute.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, dest="output_path",
        help="Output h5 path (default: <h5_stem>_corrected.h5).",
    )
    parser.add_argument(
        "--marker-radius", type=float, default=5.0,
        help="Marker size and pick radius in full-res pixels (default: 15).",
    )
    parser.add_argument(
        "--eager", action="store_true",
        help="Load all frames into memory (default: lazy VideoCapture).",
    )
    parser.add_argument(
        "--blit", action="store_true", help="Enable blitting (smoother drag)."
    )
    args = parser.parse_args()

    run(
        video_path=args.video_path,
        h5_path=args.h5_path,
        xy_key=args.xy_key,
        pred_resolution=tuple(args.pred_resolution) if args.pred_resolution else None,
        output_path=args.output_path,
        marker_radius=args.marker_radius,
        eager=args.eager,
        blit=args.blit,
    )


if __name__ == "__main__":
    start()
