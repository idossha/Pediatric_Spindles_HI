"""Grids of small scalp maps, one per test, on one colour scale (fig-4 and fig-5).

Every test is drawn at full colour; every map dots its largest cluster's channels and prints its p_cluster or q
beneath it, black and bold when the cluster survives, grey and regular when it does not, as on every map in the
paper. Geometry is in mm from the panel's bottom-left, as ``plotting.style.axes_at`` takes it.
"""
from __future__ import annotations

import numpy as np

from .style import FONT_LABEL, axes_at
from .topo import NOT_SIGNIFICANT, mark_cluster, plot_topomap

CMAP = "RdBu_r"
METRIC_TITLE = {"Density": "Density", "Duration": "Duration", "Amplitude": "Amplitude", "Frequency": "Frequency"}


def small_map(fig, rect_mm, t: np.ndarray, cluster: np.ndarray, survives: bool, label: str, *, limit: float,
              name: str):
    """One test's t-map at ``rect_mm``, its largest cluster's channels dotted and its p beneath. Returns the image."""
    import idofig as idf

    ax = idf.name(axes_at(fig, rect_mm), name)
    im = plot_topomap(ax, t, cmap=CMAP, vlim=(-limit, limit), gid=f"{name}.field")
    if cluster.any():
        mark_cluster(ax, np.flatnonzero(cluster), significant=survives, gid=f"{name}.cluster")
    cluster_label(ax, label, survives, gid=f"{name}.label")
    return im


def cluster_label(ax, label: str, survives: bool, *, gid: str):
    """The p_cluster (or q) beneath a map: bold black when its cluster survives, regular grey when not."""
    ax.text(0.5, -0.02, label, transform=ax.transAxes, ha="center", va="top", fontsize=FONT_LABEL,
            fontweight="bold" if survives else "normal", color="black" if survives else NOT_SIGNIFICANT, gid=gid)


#: the cluster-permutation p, as the manuscript writes it (q is reserved for Benjamini-Hochberg)
P_CLUSTER = r"$p_{\mathrm{cluster}}$"


def test_label(n_channels: int, value: float, symbol: str) -> str:
    """'p = 0.011' (or q), or 'no cluster' when no channel crossed the cluster-forming threshold."""
    return "no cluster" if n_channels == 0 or not np.isfinite(value) else f"{symbol} = {value:.3f}"


def horizontal_bar(fig, im, rect_mm, label: str, *, name: str) -> None:
    """A horizontal colour bar of t: whole-number ticks at the limits and 0, the label below."""
    import idofig as idf

    cb = idf.colorbar(im, cax=axes_at(fig, rect_mm), name=name, orientation="horizontal")
    limit = int(np.floor(im.norm.vmax))
    cb.set_ticks([-limit, 0, limit])
    cb.ax.tick_params(labelsize=FONT_LABEL, width=0.4, length=2, pad=1.0)
    cb.outline.set_linewidth(0.4)
    cb.set_label(label, fontsize=FONT_LABEL, labelpad=1.5)
