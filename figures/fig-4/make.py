"""Figure 4: Association between the hypopnea index and each spindle metric across the scalp

Draws only source_data/fig-4/ (summarized data; nothing is fitted here).
Build: ``idofig build fig-4`` (or ./reproduce.sh).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

import idofig as idf
import numpy as np
from plotting.mapgrid import (
    P_CLUSTER,
    METRIC_TITLE,
    horizontal_bar,
    small_map,
    test_label,
)
from plotting.style import FONT_TITLE, LOG10, house_style, panel_letter
from plotting.tables import read_table
from project import SOURCE_DATA

TABLE_DIR = SOURCE_DATA / "fig-4"  # this figure's source data

BANDS = ("fast", "slow")
METRICS = ("Density", "Duration", "Amplitude", "Frequency")
LEFT, MAP, PITCH = 11.0, 24.0, 28.5          # mm: band-label gutter, map size, column pitch
TOP_GAP, ROW = 6.0, 30.0                       # the metric titles, then map + label + gap per row
BAR = (LEFT + 1.5 * PITCH + MAP / 2 - 25.0, 6.5, 50.0, 1.6)  # centred under the grid

maps = read_table(TABLE_DIR / "cluster_tmaps.csv")
clusters = read_table(TABLE_DIR / "cluster_permutation_results.csv")


def test(band: str, metric: str):
    rows = np.flatnonzero((maps["band"] == band) & (maps["metric"] == metric))
    rows = rows[np.argsort(maps["channel"][rows])]
    k = int(np.flatnonzero((clusters["band"] == band) & (clusters["metric"] == metric))[0])
    return maps["t"][rows], maps["in_cluster"][rows].astype(bool), int(clusters["n_channels"][k]), float(clusters["p_value"][k])


p = idf.panels("maps", layout=None)
house_style()
fig = p["maps"]
height = fig.get_figheight() * 25.4
limit = float(np.ceil(np.nanmax(np.abs(maps["t"])) * 2) / 2)
for j, metric in enumerate(METRICS):
    fig.text((LEFT + j * PITCH + MAP / 2) / (fig.get_figwidth() * 25.4), (height - 1.0) / height, METRIC_TITLE[metric],
             ha="center", va="top", fontsize=FONT_TITLE, fontweight="bold", gid=f"title-{metric.lower()}")
for i, band in enumerate(BANDS):
    bottom = height - TOP_GAP - (i + 1) * ROW + (ROW - MAP) - 1.0
    fig.text(3.0 / (fig.get_figwidth() * 25.4), (bottom + MAP / 2) / height, band.capitalize(), rotation=90,
             ha="center", va="center", fontsize=FONT_TITLE, fontweight="bold", gid=f"band-{band}")
    for j, metric in enumerate(METRICS):
        t, cluster, n, pval = test(band, metric)
        im = small_map(fig, (LEFT + j * PITCH, bottom, MAP, MAP), t, cluster, pval < 0.05, test_label(n, pval, P_CLUSTER),
                       limit=limit, name=f"{band}-{metric.lower()}")
    panel_letter(fig, LEFT - 1.5, height - bottom - MAP - 1.0, "ab"[i])  # a fast, b slow: at the head of each row
horizontal_bar(fig, im, BAR, f"$t$ of {LOG10}(HI + 1), adjusted for age and sex", name="t")
idf.save(p)
