"""Topographic maps on the 172-channel inside172 montage; ``load_channels()`` is ``stats.io.load_channel_info``.

Channel indices here are 0-based positions in the montage.
"""
from __future__ import annotations

from functools import lru_cache

import mne
import numpy as np
from matplotlib.patches import Circle
from stats.io import SPHERE, load_channel_info

from . import PROJECT  # noqa: F401  (puts code/ on sys.path)

mne.set_log_level("ERROR")

# Marker styles are defined for a head ORIGINAL_HEAD_PT across and scaled to the head as drawn.
ORIGINAL_HEAD_PT = 80.0
CLUSTER_MARKER = {"s": 6.5, "zorder": 9}  # filled dots
NOT_SIGNIFICANT = "#777777"  # a largest cluster that does not survive correction: its dots and its p
OUTLINE_COLOR, OUTLINE_LW = "#8a8a8a", 0.6  # the region of a reference cluster outlined on another map
OUTLINE_GRID, OUTLINE_SMOOTH, OUTLINE_LEVEL = 241, 0.014, 0.35  # grid points across the head; kernel SD (head
# radius SPHERE = 0.095) and level chosen so one smooth region holds all 39 channels of fig-4's HI cluster (and 14 others)


@lru_cache(maxsize=1)
def load_channels():
    """Channel labels, x, y, MNE info and the inter-electrode distance matrix."""
    return load_channel_info()


def symmetric_limit(values: np.ndarray, fallback: float) -> float:
    """Largest |value| over finite channels, for a colour scale centred on zero."""
    finite = values[np.isfinite(values)]
    return float(np.nanmax(np.abs(finite))) if finite.size else fallback


def plot_topomap(ax, values, *, cmap: str, vlim, gid: str = "field"):
    """Interpolated scalp map without sensors or contours. Returns the image."""
    channels = load_channels()
    im, _ = mne.viz.plot_topomap(values, channels.info, axes=ax, show=False, cmap=cmap,
                                 sphere=SPHERE, contours=0, vlim=vlim, sensors=False)
    # mne clips the field 1 % outside the head; clip at the head itself so no colour shows past the outline
    im.set_clip_path(Circle((0.0, 0.0), SPHERE, transform=ax.transData))
    im.set_gid(gid)
    return im


def head_scale(ax) -> float:
    """The head's drawn diameter over ``ORIGINAL_HEAD_PT``."""
    ax.apply_aspect()
    (x0, _), (x1, _) = ax.transData.transform([(-SPHERE, 0.0), (SPHERE, 0.0)])
    return (x1 - x0) * 72 / ax.figure.dpi / ORIGINAL_HEAD_PT


def mark_cluster(ax, channel_indices, *, significant: bool = True, gid: str = "cluster"):
    """Filled dots on a map's largest permutation cluster, the only channels a map marks: black when it survives
    correction, grey when it does not."""
    channels = load_channels()
    idx = np.asarray(list(channel_indices), dtype=int)
    scale = head_scale(ax)
    return ax.scatter(channels.x[idx], channels.y[idx], s=CLUSTER_MARKER["s"] * scale ** 2, edgecolors="none",
                      facecolors="black" if significant else NOT_SIGNIFICANT, zorder=CLUSTER_MARKER["zorder"], gid=gid)


def outline_region(ax, member: np.ndarray, gid: str):
    """A thin gray line round the region of the channels in ``member``: the Gaussian-kernel-weighted share of
    member channels at each point of the head, drawn at one level. It shows where the cluster lies, not which
    channel belongs to it."""
    ch = load_channels()
    g = np.linspace(-SPHERE, SPHERE, OUTLINE_GRID)
    gx, gy = np.meshgrid(g, g)
    w = np.exp(-((gx[..., None] - ch.x) ** 2 + (gy[..., None] - ch.y) ** 2) / (2 * OUTLINE_SMOOTH ** 2))
    field = (w * member).sum(-1) / w.sum(-1)
    field[np.hypot(gx, gy) > SPHERE] = 0.0
    cs = ax.contour(gx, gy, field, levels=[OUTLINE_LEVEL], colors=OUTLINE_COLOR, linewidths=OUTLINE_LW, zorder=8)
    cs.set_gid(gid)
    return cs
