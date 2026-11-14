"""The TOVA sample (children of the sample with TOVA, less |z| >= 3 outliers), its outcomes, and the ROI columns (one per surviving HI cluster, ``stats.rois``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .io import ROI_PATH, add_respiratory_predictors, load_covariates
from .regression import zscore

TOVA_OUTCOMES = [
    ("d-prime", "DPRIMEQ1_z"),
    ("Omission errors", "OM_sqrt_z"),
    ("Commission errors", "COM_sqrt_z"),
    ("Reaction time", "RT_z"),
]


def roi_columns() -> list[str]:
    """The ROI columns of ROI_PATH, e.g. ``fast_duration_cluster``."""
    return [c for c in pd.read_csv(ROI_PATH, nrows=0).columns if c != "id"]



def load_behavioral_dataset() -> pd.DataFrame:
    """Children with d' and sqrt omissions / mean RT within |z| < 3; z-scored outcomes, logHI_c and age_c centred on this sample."""
    data = load_covariates().merge(pd.read_csv(ROI_PATH), on="id", how="left")

    data = data.dropna(subset=["DPRIMEQ1"]).copy()
    print(f"After TOVA filter: N = {len(data)}")

    data["OM_sqrt"] = np.sqrt(data["OMPERQ1"])
    data["COM_sqrt"] = np.sqrt(data["COMPERQ1"])
    data["RT"] = data["RTMEANQ1"].astype(float)

    data["OM_sqrt_z_temp"] = np.abs(stats.zscore(data["OM_sqrt"]))
    data["RT_z_temp"] = np.abs(stats.zscore(data["RT"]))
    n_before = len(data)
    data = data[(data["OM_sqrt_z_temp"] < 3) & (data["RT_z_temp"] < 3)].copy()
    data.drop(columns=["OM_sqrt_z_temp", "RT_z_temp"], inplace=True)
    print(f"After outlier removal: N = {len(data)} (removed {n_before - len(data)})")

    for col in ["DPRIMEQ1", "OM_sqrt", "COM_sqrt", "RT"]:
        data[f"{col}_z"] = zscore(data[col])

    data["logHI_c"] = np.log10(data["overall_hi"] + 1)
    data["logHI_c"] = data["logHI_c"] - data["logHI_c"].mean()
    data["age_c"] = data["age_decimal"] - data["age_decimal"].mean()

    return data


def load_behavioral_frame() -> pd.DataFrame:
    """``load_behavioral_dataset`` plus the respiratory predictors (``io.add_respiratory_predictors``)."""
    return add_respiratory_predictors(load_behavioral_dataset())
