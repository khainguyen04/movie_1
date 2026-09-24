"""Consistent plot style: technical figures (report) and simple one-message figures (slides)."""

import os
import sys

import matplotlib
if "ipykernel" not in sys.modules:      # headless when run from terminal
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

PALETTE = {"Standalone": "#4C72B0", "Franchise": "#DD8452", "Sequel": "#C44E52",
           "Franchise starter": "#55A868", "Observed": "#4C72B0", "Imputed": "#DD8452",
           "Blockbuster": "#C44E52", "Not blockbuster": "#8C8C8C"}
MAIN = "#4C72B0"
ACCENT = "#DD8452"


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150,
                         "axes.titleweight": "bold", "axes.titlesize": 12})


def slide_style(ax, title: str = None) -> None:
    """Clean look for non-technical slides: no top/right spines, larger fonts."""
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=11)
    if title:
        ax.set_title(title, fontsize=15, loc="left", pad=12)


def money_fmt(x, pos=None) -> str:
    if abs(x) >= 1e9:
        return f"${x / 1e9:.1f}B"
    if abs(x) >= 1e6:
        return f"${x / 1e6:.0f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:.0f}"


def money_axis(ax, axis: str = "y") -> None:
    fmt = mticker.FuncFormatter(money_fmt)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def save_fig(fig, subdir: str, name: str, cfg: dict) -> str:
    path = os.path.join(cfg["paths"]["figures"], subdir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path