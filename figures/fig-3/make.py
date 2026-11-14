"""Figure 3: Normative topography of slow and fast sleep spindles

Draws only source_data/fig-3/ (summarized data; nothing is fitted here).
Build: ``idofig build fig-3`` (or ./reproduce.sh).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

import idofig as idf
import numpy as np
from plotting import topo
from plotting.tables import read_table
from project import SOURCE_DATA


def grid_map(metric: str, band: str):
    """One spindle metric's group-mean map over the 172 channels."""
    return read_table(SOURCE_DATA / "fig-3" / "grid_maps.csv")[f"{metric}_{band}"]


from plotting.gridfig import (
    NARROW_FAST,
    NARROW_SLOW,
    add_row_colorbar,
    add_row_label,
    add_title,
    grid_axes,
)
from plotting.style import house_style

# The row label names the quantity and its unit, as an axis label would: above the slow
# column's bar, "/min" ran into that column's title, and rotated beside it "s" read sideways.
# A rate is "per <unit>" in every figure (fig-s3: "per h"): "(events/min)" is wider than
# this column and was cut at the figure's left edge.
ROWS = [  # metric, row label (column 0)
    ("density", "Density\n(per min)"),
    ("duration", "Duration\n(s)"),
    ("amplitude", "Amplitude\n(µV)"),
    ("frequency", "Frequency\n(Hz)"),
]
BANDS = [("fast", "Fast (12–16 Hz)", NARROW_FAST), ("slow", "Slow (10–12 Hz)", NARROW_SLOW)]

p = idf.panels(*(f"{metric}-{band}" for metric, *_ in ROWS for band, *_ in BANDS), layout=None)
house_style()

for row, (metric, label) in enumerate(ROWS):
    maps = {band: grid_map(metric, band) for band, *_ in BANDS}
    both = np.concatenate(list(maps.values()))
    equal_span = metric == "frequency"
    half = max(float(np.nanmax(np.abs(m - np.nanmean(m)))) for m in maps.values())
    for col, (band, title, cell) in enumerate(BANDS):
        fig = p[f"{metric}-{band}"]
        ax = idf.name(grid_axes(fig, cell), "map")
        centre = float(np.nanmean(maps[band]))
        vlim = (centre - half, centre + half) if equal_span else (float(np.nanmin(both)), float(np.nanmax(both)))
        # RdBu_r is diverging although these are unsigned means: its white marks the middle of
        # the row's range (for frequency, the band's mean), not zero.
        cmap = "Greens" if metric == "density" else "RdBu_r"
        im = topo.plot_topomap(ax, maps[band], cmap=cmap, vlim=vlim)
        if col == 0:
            add_row_label(fig, label, cell=cell)
        if col == 1 or equal_span:
            add_row_colorbar(fig, im, None, cell=cell)
        if row == 0:
            add_title(fig, title, cell=cell)

idf.save(p)
