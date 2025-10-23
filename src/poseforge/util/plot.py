import matplotlib
import logging
from matplotlib import pyplot as plt
from cycler import cycler  # this is already a dependency of matplotlib
from distinctipy import get_colors


def configure_matplotlib_style():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    # suppress matplotlib font manager warnings
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def get_segmentation_color_palette(
    n_classes: int, fix_background_body_thorax: bool = True
):
    if fix_background_body_thorax:
        fixed_colors = [
            (0.0, 0.0, 0.0),  # Background
            (0.5, 0.5, 0.5),  # OtherSegments
            (0.75, 0.75, 0.75),  # Thorax
        ]
        additional_colors = get_colors(
            n_classes - 3,
            exclude_colors=fixed_colors,
            colorblind_type="Deuteranomaly",
            rng=42,
        )
        color_palette = fixed_colors + additional_colors
    else:
        color_palette = get_colors(
            n_classes,
            colorblind_type="Deuteranomaly",
            rng=42,
        )
    return color_palette


def text_height_in_data_units(ax, fontsize_pt):
    """
    Convert a text height (in points) to data units along the y-axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis the text will be drawn on (used to get figure height, DPI,
        and ylim).
    fontsize_pt : float
        Font size in points (1 point = 1/72 inch).

    Returns
    -------
    text_height_data : float
        Height of the text in data units along the y-axis.
    """
    # Get the current y-limits
    y0, y1 = ax.get_ylim()

    # Get axis height in inches
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_height_inch = bbox.height  # axis height in inches

    # Convert font size (pt) → inches → fraction of axis height
    text_height_inch = fontsize_pt / 72  # 1 pt = 1/72 inch
    text_height_fraction = text_height_inch / ax_height_inch

    # Convert fraction of axis height → data units
    text_height_data = text_height_fraction * abs(y1 - y0)

    return text_height_data
