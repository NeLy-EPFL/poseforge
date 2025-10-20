import numpy as np
import cv2
import cmasher as cmr
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from pathlib import Path

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
