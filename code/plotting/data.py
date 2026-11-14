"""Precomputed artifacts the panels read from results/figure_data/ (no statistics, no pandas).

``main_figure_data.pkl`` comes from ``stats/figure_data.py``; the group-mean maps, cluster masks, model predictions and
cluster residuals from ``stats/figure_tables.py``; the respiratory-index t-maps from ``stats/resp_cluster_sensitivity.py``.
"""
from __future__ import annotations

import pickle
from functools import lru_cache

import numpy as np
from project import COV_PATH, FIGURE_DATA, TABLE_DIR, sample_ids

from .tables import read_table

TABLES = TABLE_DIR
MAIN_FIGURE_DATA = FIGURE_DATA / "main_figure_data.pkl"
GRID_MAPS = FIGURE_DATA / "grid_maps.csv"
CLUSTER_RESIDUALS = FIGURE_DATA / "fast_duration_cluster_residuals.csv"
RESP_TMAPS = FIGURE_DATA / "resp_tmaps_duration_fast.csv"
RESP_COLUMNS = ("t_AHI", "t_oAHI")


@lru_cache(maxsize=1)
def load_figure_data() -> dict:
    """``main_figure_data.pkl`` (trusted local file): ``reg[(band, metric)] -> {'beta', 't', 'p'}`` channel maps,
    ``scatter[(band, metric)]`` partial-residual series, ``roi`` channel index lists."""
    with MAIN_FIGURE_DATA.open("rb") as stream:
        return pickle.load(stream)


def grid_map(metric: str, band: str) -> np.ndarray:
    """fig-3's group-mean map of one metric (density, duration, amplitude, frequency)."""
    return read_table(GRID_MAPS)[f"{metric}_{band}"]


def resp_limit() -> float:
    """One colour scale for fig-s4's respiratory-index t-maps (``stats.resp_cluster_sensitivity``)."""
    maps = read_table(RESP_TMAPS)
    return float(np.nanmax(np.abs(np.concatenate([maps[c] for c in RESP_COLUMNS]))))


def resp_map(column: str) -> tuple[np.ndarray, np.ndarray]:
    """A respiratory-index t-map (column ``t_<model>``) and its largest cluster."""
    maps = read_table(RESP_TMAPS)
    return maps[column], maps[column.replace("t_", "cluster_", 1)].astype(bool)


def cluster_residuals() -> dict:
    """fig-4 b: corrected fast-duration cluster mean duration vs log HI, both residualized on age and sex; ``{x, y, slope, intercept, r}``."""
    table = read_table(CLUSTER_RESIDUALS)
    return {"x": table["loghi_residual"], "y": table["duration_residual"], "slope": float(table["fit_slope"][0]),
            "intercept": float(table["fit_intercept"][0]), "r": float(table["r"][0])}


def covariates() -> dict[str, np.ndarray]:
    """The analysed sample's covariate columns (``project.sample_ids``), in file order."""
    table = read_table(COV_PATH)
    ids = sample_ids()
    keep = np.isin(table["id"], ids)
    if int(keep.sum()) != len(ids):
        raise ValueError(f"covariates for {int(keep.sum())} of the {len(ids)} children of the sample")
    return {name: column[keep] for name, column in table.items()}
