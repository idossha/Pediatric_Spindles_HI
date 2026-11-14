"""Inputs of the statistics (library; fits nothing, paths come from ``project``).

* ``load_channel_info``      -- 172-channel inside172 montage (``CHAN_PATH``): x/y on a sphere of radius
                               ``SPHERE``, MNE ``Info``, distance matrix.
* ``load_covariates``        -- the analysed sample (``COV_PATH`` rows of ``project.sample_ids``) with
                               ``logHI_c``, ``age_decimal``, ``age_c`` (centred), ``gender_bin``.
* ``load_spindle_matrices``  -- subjects x channels Count/Amplitude/Duration(/Frequency) per band from
                               ``METRICS_LONG`` (``stats.metrics``); matrix columns are 0-based, the
                               table's ``channel`` is 1-based.
* ``data_matrix_for_metric`` -- Density = Count / cleaned N2+N3 minutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.io as sio
from project import (  # noqa: F401  (re-exported: the pipeline imports paths from here)
    CHAN_PATH,
    COV_PATH,
    FIGURE_DATA,
    METRICS_LONG,
    PARTICIPANTS_PATH,
    PROJECT,
    ROI_PATH,
    SPINDLE_DIR,
    TABLE_DIR,
    ensure_dir,
    sample_ids,
)

# pandas is imported inside the table readers: the figure scripts use only the montage and their environment has no pandas.
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    import pandas as pd

import mne

SPHERE = 0.095
SLOW_RANGE = (10.0, 12.0)  # [10, 12)
FAST_RANGE = (12.0, 16.0)  # [12, 16]
PREDICTORS = ["logHI_c", "age_c", "gender_bin"]


@dataclass(frozen=True)
class ChannelInfo:
    labels: list[str]
    x: np.ndarray
    y: np.ndarray
    info: mne.Info
    distance_matrix: np.ndarray


@dataclass(frozen=True)
class SpindleMatrices:
    slow: dict[str, np.ndarray]
    fast: dict[str, np.ndarray]
    sleep_mins: np.ndarray
    missing_subjects: list[int | float | str]



def load_channel_info(chan_path: str | Path = CHAN_PATH, sphere: float = SPHERE) -> ChannelInfo:
    mat = sio.loadmat(chan_path)
    chanlocs = mat["inside172"].flatten()

    theta_deg, radius_raw, labels = [], [], []
    for ch in chanlocs:
        theta_deg.append(float(ch["theta"].item()))
        radius_raw.append(float(ch["radius"].item()))
        lbl = ch["labels"]
        if isinstance(lbl, np.ndarray):
            lbl = "".join(str(x).strip() for x in lbl.flatten())
        else:
            lbl = str(lbl).strip()
        labels.append(lbl)

    theta_deg = np.asarray(theta_deg)
    radius_raw = np.asarray(radius_raw)
    radius_scaled = radius_raw * (sphere / np.max(radius_raw))
    x = radius_scaled * np.sin(np.radians(theta_deg))
    y = radius_scaled * np.cos(np.radians(theta_deg))

    ch_pos = {name: [float(px), float(py), 0.0] for name, px, py in zip(labels, x, y)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info = mne.create_info(ch_names=labels, sfreq=100, ch_types="eeg")
    info.set_montage(montage)
    distance_matrix = cdist(np.column_stack([x, y]), np.column_stack([x, y]))
    return ChannelInfo(labels=labels, x=x, y=y, info=info, distance_matrix=distance_matrix)


def decimal_age(df: pd.DataFrame) -> pd.Series:
    """Decimal years from the age_years / age_months / age_days columns (``age_years`` alone is completed years)."""
    return df["age_years"] + df["age_months"] / 12.0 + df["age_days"] / 365.25


def load_covariates(cov_path: str | Path = COV_PATH) -> pd.DataFrame:
    """The analysed sample with covariates and ``clean_min`` (cleaned N2+N3 minutes, from the detections' manifest); sleep efficiency 0 % is an entry error, set missing."""
    import pandas as pd
    cov = pd.read_csv(cov_path)
    ids = sample_ids()
    cov = cov[cov["id"].isin(ids)].reset_index(drop=True)
    assert len(cov) == len(ids), f"covariates for {len(cov)} of the {len(ids)} children of the sample"
    cleaned_s = pd.read_csv(SPINDLE_DIR / "manifest.csv").set_index("id")["cleaned_s"]
    cov = cov.drop(columns="cleanN2N3_min").assign(clean_min=cov["id"].map(cleaned_s / 60.0))
    cov["se"] = cov["se"].where(cov["se"] > 0)
    cov["logHI"] = np.log10(cov["overall_hi"] + 1)
    cov["logHI_c"] = cov["logHI"] - cov["logHI"].mean()
    cov["age_decimal"] = decimal_age(cov)
    cov["age_c"] = cov["age_decimal"] - cov["age_decimal"].mean()
    cov["gender_bin"] = cov["gender"]
    return cov


def load_spindle_matrices(cov: pd.DataFrame, n_channels: int, include_frequency: bool = False) -> SpindleMatrices:
    """Subjects (rows of ``cov``) x channels matrices per band from ``METRICS_LONG``; ``sleep_mins`` is the density denominator."""
    import pandas as pd
    table = pd.read_csv(METRICS_LONG)
    ids = cov["id"].to_numpy()
    metrics = ["Count", "Amplitude", "Duration"] + (["Frequency"] if include_frequency else [])
    bands = {}
    for band in ("slow", "fast"):
        sub = table[table["band"] == band]
        bands[band] = {m: sub.pivot(index="id", columns="channel", values=m.lower())
                       .reindex(index=ids, columns=range(1, n_channels + 1)).to_numpy(float) for m in metrics}
    minutes = table.groupby("id")["clean_min"].first().reindex(ids).to_numpy(float)[:, np.newaxis]
    return SpindleMatrices(slow=bands["slow"], fast=bands["fast"], sleep_mins=minutes,
                           missing_subjects=[i for i in ids if i not in set(table["id"])])


def data_matrix_for_metric(
    data_dict: dict[str, np.ndarray],
    metric: str,
    sleep_mins: np.ndarray,
) -> np.ndarray:
    if metric == "Density":
        return data_dict["Count"] / sleep_mins
    return data_dict[metric]


RESP_INDICES = {"hi": "overall_hi", "ahi": "overall_ahi", "oai": "overall_oai", "cai": "overall_cai"}
OAHI_COL = "overall_oahi"  # obstructive AHI (pediatric convention) = total AHI - central apnea index
RESP_INDICES_OAHI = {"hi": "overall_hi", "ahi": "overall_ahi", "oahi": OAHI_COL, "oai": "overall_oai", "cai": "overall_cai"}
RESP_LABELS = {"hi": "HI", "ahi": "AHI", "oahi": "oAHI", "oai": "OAI", "cai": "CAI"}  # as written in tables and text


def matrix_inputs() -> list[Path]:
    """Provenance inputs of a script that reads the spindle matrices."""
    return [COV_PATH, PARTICIPANTS_PATH, CHAN_PATH, METRICS_LONG, SPINDLE_DIR / "manifest.csv"]


def spindle_matrices(cov: pd.DataFrame, n_ch: int) -> tuple[dict, dict, np.ndarray]:
    """(slow, fast, sleep_mins) of ``load_spindle_matrices`` with spindle frequency."""
    mats = load_spindle_matrices(cov, n_ch, include_frequency=True)
    return mats.slow, mats.fast, mats.sleep_mins


def add_obstructive_ahi(cov: pd.DataFrame) -> pd.DataFrame:
    """Add ``overall_oahi`` = ``overall_ahi`` - ``overall_cai`` (float noise rounded away)."""
    out = cov.copy()
    out[OAHI_COL] = (out["overall_ahi"].astype(float) - out["overall_cai"].astype(float)).round(4)
    return out


def add_respiratory_predictors(cov: pd.DataFrame, indices: dict[str, str] = RESP_INDICES) -> pd.DataFrame:
    """Add ``log<IDX>_c`` (log10(x + 1), mean-centred, as HI) per index, keeping existing columns such as ``logHI_c``.

    Pass ``RESP_INDICES_OAHI`` (after ``add_obstructive_ahi``) to also get ``logOAHI_c``.
    """
    out = cov.copy()
    for key, col in indices.items():
        name = f"log{key.upper()}_c"
        if name in out.columns:
            continue
        v = np.log10(out[col].astype(float) + 1.0)
        out[name] = v - v.mean()
    return out
