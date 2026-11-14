"""Figure 5: Spindle expression and attention across the scalp

Draws only source_data/fig-5/ (summarized data; nothing is fitted here).
Build: ``idofig build fig-5`` (or ./reproduce.sh).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

import idofig as idf
import numpy as np
from matplotlib.lines import Line2D
from plotting.mapgrid import (
    METRIC_TITLE,
    horizontal_bar,
    small_map,
    test_label,
)
from plotting.style import FONT_LABEL, FONT_TITLE, house_style, panel_letter
from plotting.tables import read_table
from project import SOURCE_DATA

FIGURE_DATA = TABLE_DIR = SOURCE_DATA / "fig-5"  # this figure's source data

BANDS = ("fast", "slow")
METRICS = ("Density", "Duration", "Amplitude", "Frequency")
OUTCOMES = ("d-prime", "Omission errors", "Commission errors", "Reaction time")
ROW_LABEL = {"d-prime": "$\\mathbfit{d}$′", "Omission errors": "Omission\nerrors", "Commission errors": "Commission\nerrors",
             "Reaction time": "Reaction\ntime"}
SIGN = {"d-prime": 1.0, "Omission errors": -1.0, "Commission errors": -1.0, "Reaction time": -1.0}  # + = better
LEFT, MAP, PITCH, BAND_GAP = 18.0, 17.5, 19.6, 4.0   # mm
TOP_GAP, ROW = 13.0, 23.0                              # band headers and metric titles; map + label + gap
BAR_W = 60.0

maps = read_table(FIGURE_DATA / "spindle_tova_tmaps.csv")
tests = read_table(TABLE_DIR / "spindle_tova_clusters.csv")


def test(band: str, metric: str, outcome: str):
    rows = np.flatnonzero((maps["band"] == band) & (maps["metric"] == metric) & (maps["outcome"] == outcome))
    rows = rows[np.argsort(maps["channel"][rows])]
    k = int(np.flatnonzero((tests["band"] == band) & (tests["metric"] == metric) & (tests["outcome"] == outcome))[0])
    survives = str(tests["survives"][k]) == "True"
    # the largest cluster's 1-based channels, listed for every test (the t-map's in_cluster marks survivors only)
    members = [int(c) for c in str(tests["channels"][k]).split(";") if c not in ("", "nan")]
    return (SIGN[outcome] * maps["t"][rows], np.isin(maps["channel"][rows], members), survives,
            int(tests["n_channels"][k]), float(tests["q_fdr32"][k]))


def column_x(band: str, j: int) -> float:
    return LEFT + (4 * BANDS.index(band) + j) * PITCH + (BAND_GAP if band == "slow" else 0.0)


p = idf.panels("maps", layout=None)
house_style()
fig = p["maps"]
width, height = fig.get_figwidth() * 25.4, fig.get_figheight() * 25.4
limit = float(np.ceil(np.nanmax(np.abs(maps["t"])) * 2) / 2)

for band in BANDS:
    x0, x1 = column_x(band, 0), column_x(band, 3) + MAP
    fig.text((x0 + x1) / 2 / width, (height - 0.5) / height, f"{band.capitalize()} spindles", ha="center", va="top",
             fontsize=FONT_TITLE, fontweight="bold", gid=f"band-{band}")
    fig.add_artist(Line2D([(x0 + 1) / width, (x1 - 1) / width], [(height - 5.2) / height] * 2, color="black", lw=0.6,
                          gid=f"band-rule-{band}"))
    panel_letter(fig, x0 - 1.0, 0.0, "ab"[BANDS.index(band)])
    for j, metric in enumerate(METRICS):
        fig.text((column_x(band, j) + MAP / 2) / width, (height - 6.8) / height, METRIC_TITLE[metric], ha="center",
                 va="top", fontsize=FONT_LABEL, fontweight="bold", gid=f"title-{band}-{metric.lower()}")

for i, outcome in enumerate(OUTCOMES):
    bottom = height - TOP_GAP - (i + 1) * ROW + (ROW - MAP) - 1.0
    fig.text((LEFT - 2.0) / width, (bottom + MAP / 2) / height, ROW_LABEL[outcome], ha="right", va="center",
             fontsize=FONT_LABEL, fontweight="bold", gid=f"outcome-{i}")
    for band in BANDS:
        for j, metric in enumerate(METRICS):
            t, cluster, survives, n, q = test(band, metric, outcome)
            im = small_map(fig, (column_x(band, j), bottom, MAP, MAP), t, cluster, survives, test_label(n, q, "$q$"),
                           limit=limit, name=f"{band}-{metric.lower()}-{i}")

mid = (column_x("fast", 0) + column_x("slow", 3) + MAP) / 2
horizontal_bar(fig, im, (mid - BAR_W / 2, 7.5, BAR_W, 1.6), "$t$ of the TOVA term, signed so red = better attention",
               name="t")
idf.save(p)
