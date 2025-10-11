from matplotlib import pyplot as plt


def configure_matplotlib_style():
    import matplotlib
    import logging

    matplotlib.style.use("fast")
    plt.rcParams["font.family"] = "Arial"
    # suppress matplotlib font manager warnings
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
