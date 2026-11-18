"""Geometry prior buffers derived from MuJoCo renders.

PoseForge renders several geometry buffers alongside the RGB color codings to
use as richer conditioning for the sim-to-real translation model. MuJoCo's
offscreen renderer only produces three things natively: color, metric depth
(``enable_depth_rendering``) and body-part segmentation
(``enable_segmentation_rendering``). This module derives the remaining buffers
from those:

- **Surface normals** are back-projected from the metric depth buffer using the
  camera's vertical field of view, then estimated from the spatial gradients of
  the resulting camera-space point cloud.
- **Shading** is produced separately, as an ordinary RGB render of the fly with
  a uniform matte albedo (see ``visuals/shading.yaml``), so the rendered
  intensity reflects only the lighting and geometry, not the texture.

The module also provides colorization helpers that turn the raw float depth and
integer segmentation buffers into uint8 RGB frames suitable for writing to video
and for visual inspection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import imageio.v3 as imageio_v3
except ImportError:  # pragma: no cover - imageio is a hard dependency at runtime
    imageio_v3 = None


def focal_length_in_pixels(field_of_view_y_degrees: float, image_height: int) -> float:
    """Vertical focal length in pixels for a MuJoCo camera.

    This matches the focal scaling used by flygym's ``Renderer.get_camera_matrix``
    (``(1 / tan(fovy / 2)) * height / 2``). MuJoCo uses the same vertical focal
    length for both image axes (square pixels), so this single value is used for
    the horizontal and vertical directions.
    """
    half_field_of_view = np.deg2rad(field_of_view_y_degrees) / 2.0
    return (1.0 / np.tan(half_field_of_view)) * (image_height / 2.0)


def backproject_depth_to_points(
    depth: np.ndarray, field_of_view_y_degrees: float
) -> np.ndarray:
    """Back-project a metric depth image to a camera-space point cloud.

    Args:
        depth: ``(height, width)`` float array of metric depth (distance along the
            camera's optical axis), as returned by MuJoCo depth rendering.
        field_of_view_y_degrees: Vertical field of view of the camera, in degrees.

    Returns:
        ``(height, width, 3)`` float array of camera-space ``(x, y, z)`` points.
        ``x`` increases to the right, ``y`` increases downward, ``z`` increases
        away from the camera into the scene.
    """
    image_height, image_width = depth.shape
    focal_length = focal_length_in_pixels(field_of_view_y_degrees, image_height)
    principal_point_x = (image_width - 1) / 2.0
    principal_point_y = (image_height - 1) / 2.0

    column_indices, row_indices = np.meshgrid(
        np.arange(image_width), np.arange(image_height)
    )
    z = depth.astype(np.float64)
    x = (column_indices - principal_point_x) * z / focal_length
    y = (row_indices - principal_point_y) * z / focal_length
    return np.stack([x, y, z], axis=-1)


def depth_to_surface_normals(
    depth: np.ndarray,
    field_of_view_y_degrees: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate camera-space surface normals from a metric depth image.

    Normals are estimated as the normalized cross product of the spatial
    gradients of the back-projected point cloud, then oriented to face the
    camera. Pixels outside ``valid_mask`` (typically the background) are set to
    zero.

    Args:
        depth: ``(height, width)`` metric depth image.
        field_of_view_y_degrees: Vertical field of view of the camera, in degrees.
        valid_mask: Optional ``(height, width)`` boolean mask of foreground
            pixels. Normals are zeroed outside it.

    Returns:
        ``(height, width, 3)`` float array of unit normals in ``[-1, 1]``, with
        ``+z`` pointing toward the camera.
    """
    points = backproject_depth_to_points(depth, field_of_view_y_degrees)
    gradient_along_rows = np.gradient(points, axis=0)  # d/dy (downward)
    gradient_along_columns = np.gradient(points, axis=1)  # d/dx (rightward)

    normals = np.cross(gradient_along_columns, gradient_along_rows)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    lengths[lengths == 0] = 1.0
    normals = normals / lengths

    # Orient every normal toward the camera. The camera looks along +z (into the
    # scene), so a normal facing the camera has a negative z component; flip the
    # ones that don't so that +z consistently points back toward the camera.
    facing_away = normals[..., 2] > 0
    normals[facing_away] *= -1.0

    if valid_mask is not None:
        normals[~valid_mask] = 0.0
    return normals


def encode_normals_to_rgb(
    normals: np.ndarray, valid_mask: np.ndarray | None = None
) -> np.ndarray:
    """Encode signed unit normals as a uint8 RGB normal map.

    Uses the common tangent-space convention: red = x (right), green = up,
    blue = toward the camera, each mapped from ``[-1, 1]`` to ``[0, 255]``. A
    camera-facing surface therefore renders as the familiar light blue. Pixels
    outside ``valid_mask`` are set to black.
    """
    display_normals = np.stack(
        [normals[..., 0], -normals[..., 1], normals[..., 2]], axis=-1
    )
    encoded = ((display_normals * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    if valid_mask is not None:
        encoded[~valid_mask] = 0
    return encoded


def colorize_depth(
    depth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    colormap_name: str = "turbo",
) -> np.ndarray:
    """Colorize a metric depth image to a uint8 RGB frame.

    Depth is normalized within the foreground (``valid_mask``) to ``[0, 1]`` and
    inverted so that near surfaces are bright, then passed through a matplotlib
    colormap. Background pixels are black.
    """
    import matplotlib

    if valid_mask is None:
        valid_mask = np.ones_like(depth, dtype=bool)

    colorized = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid_mask.any():
        return colorized

    foreground_depth = depth[valid_mask]
    minimum = float(foreground_depth.min())
    maximum = float(foreground_depth.max())
    if maximum <= minimum:
        maximum = minimum + 1e-6

    normalized = np.clip((depth - minimum) / (maximum - minimum), 0.0, 1.0)
    normalized = 1.0 - normalized  # near surfaces bright

    colormap = matplotlib.colormaps[colormap_name]
    rgb = (colormap(normalized)[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid_mask] = 0
    return rgb


def build_label_palette(num_labels: int, seed: int = 0) -> np.ndarray:
    """Build a stable ``(num_labels, 3)`` uint8 color palette for labels.

    Label 0 is reserved for the background and mapped to black. Colors for the
    remaining labels are drawn from a fixed-seed generator so the palette is
    identical across frames.
    """
    generator = np.random.default_rng(seed)
    palette = (generator.random((num_labels, 3)) * 255.0).astype(np.uint8)
    if num_labels > 0:
        palette[0] = 0
    return palette


def colorize_segmentation(
    segmentation_labels: np.ndarray, palette: np.ndarray | None = None, seed: int = 0
) -> np.ndarray:
    """Colorize an integer label image (background == 0) to uint8 RGB."""
    maximum_label = int(segmentation_labels.max()) if segmentation_labels.size else 0
    if palette is None:
        palette = build_label_palette(maximum_label + 1, seed=seed)
    return palette[segmentation_labels]


def save_video(frames: np.ndarray, path: str | Path, fps: float = 25.0) -> None:
    """Write a stack of uint8 RGB frames ``(n, height, width, 3)`` to an mp4."""
    if imageio_v3 is None:
        raise ImportError("imageio is required to save videos.")
    frames = np.asarray(frames, dtype=np.uint8)
    imageio_v3.imwrite(
        str(path), frames, fps=fps, codec="libx264", pixelformat="yuv420p"
    )
