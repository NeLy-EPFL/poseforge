from matplotlib import pyplot as plt
from distinctipy import get_colors


def configure_matplotlib_style():
    import matplotlib
    import logging

    matplotlib.style.use("fast")
    plt.rcParams["font.family"] = "Arial"
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
