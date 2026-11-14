"""The 32 spindle metric x TOVA cluster tests (``stats.spindle_tova``) as the graphical abstract quotes them.

Reads results/tables/spindle_tova_clusters.csv (the 32 tests: sample size, cluster p and the
Benjamini-Hochberg q across the 32). Nothing is fitted here.
"""
from __future__ import annotations

import numpy as np

from .data import TABLES
from .tables import read_table

CLUSTERS = TABLES / "spindle_tova_clusters.csv"


def q_value(band: str, metric: str, outcome: str) -> float:
    """One test's Benjamini-Hochberg q across the 32 (``spindle_tova_clusters.csv``)."""
    table = read_table(CLUSTERS)
    i = np.flatnonzero((table["band"] == band) & (table["metric"] == metric) & (table["outcome"] == outcome))[0]
    return float(table["q_fdr32"][i])
