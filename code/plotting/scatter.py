"""Partial-residual scatter of an ROI-averaged spindle metric against log HI (given age, sex)."""
from __future__ import annotations

import numpy as np
from matplotlib.transforms import ScaledTranslation

from .style import (
    AXES_LW,
    FONT_LABEL,
    FONT_SUBSCRIPTED,
    LOG10,
    SCATTER_COLOR,
    minus,
    thin_spines,
)

YLABEL_MM = 7.9  # from the axes' left edge to the y label's near side: clears "−0.25" at 7 pt
# r sits this far below the axes' top-right corner in every scatter
R_LABEL_DROP_MM = 2.2


def plot_scatter(ax, series: dict | None, ylabel: str, color: str = SCATTER_COLOR) -> None:
    """Draw a precomputed series ``{x, y, slope, intercept, r}`` (nothing is refitted); ids ``observations`` and ``fit``."""
    if series is None:
        ax.text(0.5, 0.5, "No ROI", transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_LABEL, color="grey", gid="no-roi")
        return
    x_resid, y_resid = series["x"], series["y"]
    ax.scatter(x_resid, y_resid, s=12, alpha=0.55, color=color, edgecolors="white",
               linewidths=0.3, zorder=5, gid="observations")
    x_fit = np.linspace(x_resid.min(), x_resid.max(), 100)
    ax.plot(x_fit, series["slope"] * x_fit + series["intercept"], color=color, linewidth=1.0,
            zorder=6, gid="fit")
    drop = ScaledTranslation(0, -R_LABEL_DROP_MM / 25.4, ax.figure.dpi_scale_trans)
    ax.text(0.95, 0.95, "$r$ = " + minus(f"{series['r']:.2f}"), transform=ax.transAxes + drop, ha="right",
            va="top", fontsize=FONT_LABEL, color="#555555", gid="r-value")
    ax.set_xlabel(f"Residual {LOG10}(HI + 1)", fontsize=FONT_SUBSCRIPTED)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    # fixed y-label position so a column's panels line up whatever the tick width
    ax.yaxis.set_label_coords(-YLABEL_MM / (ax.get_position().width * ax.figure.get_figwidth() * 25.4), 0.5)
    ax.tick_params(axis="both", length=2, width=0.4)
    thin_spines(ax)
    ax.spines["left"].set_linewidth(AXES_LW)
    ax.spines["bottom"].set_linewidth(AXES_LW)
