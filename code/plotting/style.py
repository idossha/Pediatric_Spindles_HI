"""The manuscript's figure style on top of idofig's (library).

``idofig.panels()`` makes each panel at its printed size with the journal rc (Liberation Sans); ``house_style()``
then applies 7 pt labels, 8 pt titles, 0.6 pt axes and matplotlib's default tick, line and legend sizes. Call it right
after ``idofig.panels()``. Panel letters come from figure.json, not from here.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import PROJECT  # noqa: F401  (puts code/ on sys.path, redirects caches)

DPI = 600  # journal resolution: the embedded rasters (topomap fields) are this fine
FONT_LABEL = 7  # axis labels: nothing prints below 7 pt
FONT_TITLE = 8
FONT_LETTER = 10  # panel letters a script draws itself, the size of idofig's own (figure.json "letters")
# a mathtext subscript is 0.7 of its line: 7.2 pt prints it at 5.04 pt
FONT_SUBSCRIPTED = 7.2
LOG10 = "log$_{10}$"
AXES_LW = 0.6
MM = 25.4

# Colours shared by more than one panel.
SCATTER_COLOR = "#1a5276"  # partial-residual observations and fit
N2_COLOR, N3_COLOR = "#6baed6", "#2171b5"

HOUSE_RC = {
    "font.size": FONT_LABEL,
    "axes.labelsize": FONT_LABEL,
    "axes.titlesize": FONT_TITLE,
    "axes.titleweight": "bold",
    "xtick.labelsize": FONT_LABEL,
    "ytick.labelsize": FONT_LABEL,
    "legend.fontsize": FONT_LABEL,
    "legend.title_fontsize": FONT_LABEL,
    "axes.linewidth": AXES_LW,
    "xtick.major.width": AXES_LW,
    "ytick.major.width": AXES_LW,
    # matplotlib's defaults (idofig's are finer)
    "xtick.minor.width": AXES_LW,
    "ytick.minor.width": AXES_LW,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "lines.linewidth": 1.5,
    "lines.markersize": 6.0,
    "patch.linewidth": 1.0,
    "legend.frameon": True,
    # one square light-grey frame for every legend over data; a legend in free space is frameon=False
    "legend.fancybox": False,
    "legend.edgecolor": "#cccccc",
    "legend.framealpha": 0.9,
    # no mathtext.fontset (idofig's keeps math in Liberation Sans); idofig lacks bold italic, whose default is DejaVu
    "mathtext.bfit": "Liberation Sans:italic:bold",
    "savefig.dpi": DPI,
}


def house_style() -> None:
    """The manuscript's rcParams, over idofig's journal defaults (call after idofig.panels)."""
    plt.rcParams.update(HOUSE_RC)


def axes_at(fig, rect_mm, **kwargs):
    """An axes at ``[left, bottom, width, height]`` millimetres from the panel's bottom-left."""
    left, bottom, width, height = rect_mm
    w, h = fig.get_figwidth() * MM, fig.get_figheight() * MM
    return fig.add_axes((left / w, bottom / h, width / w, height / h), **kwargs)


def minus(text: str) -> str:
    """Replace the hyphen of a formatted number with the typographic minus (U+2212), as tick labels have."""
    return text.replace("-", "−")


def colorbar_unit(cb, unit: str) -> None:
    """A colorbar's unit, above the bar and left-aligned to it."""
    cb.ax.set_title(unit, loc="left", fontsize=FONT_LABEL, fontweight="normal", pad=2.5,
                    gid="colorbar-label")


def thin_spines(ax, hide=("top", "right")) -> None:
    """Hide the given spines and give the rest the manuscript line width."""
    for spine in hide:
        ax.spines[spine].set_visible(False)
    for name, spine in ax.spines.items():
        if name not in hide:
            spine.set_linewidth(AXES_LW)


def panel_letter(fig, x_mm: float, top_mm: float, text: str) -> None:
    """A panel letter ``x_mm`` from the left and ``top_mm`` from the top of ``fig``, drawn as idofig draws its letters:
    for the parts of a figure that is one idofig panel (a grid of maps, a schematic)."""
    width, height = fig.get_figwidth() * 25.4, fig.get_figheight() * 25.4
    fig.text(x_mm / width, 1.0 - top_mm / height, text, ha="left", va="top", fontsize=FONT_LETTER, fontweight="bold",
             gid=f"letter-{text}")
