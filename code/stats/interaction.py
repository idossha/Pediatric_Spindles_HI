"""The ROI-level Age x HI model ``roi ~ age_c + logHI_c + age_c:logHI_c + sex`` on the analysed sample
(fig-s2 b predictions, ``stats.age_hi_roi_interaction``)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .io import COV_PATH, load_covariates

FORMULA = "roi ~ age_c + logHI_c + ageXHI + sex"
COLUMNS = ["age_decimal", "age_c", "logHI", "logHI_c", "sex"]


def prepare_covariates() -> pd.DataFrame:
    """Canonical covariates with ``sex`` and the centred interaction term added."""
    cov = load_covariates(COV_PATH)
    cov["sex"] = cov["gender_bin"]
    cov["ageXHI"] = cov["age_c"] * cov["logHI_c"]
    return cov


def fit_interaction(roi_values: np.ndarray, cov: pd.DataFrame):
    """OLS interaction model for one ROI; returns ``(fit, df)`` on the complete cases."""
    df = cov[COLUMNS].copy()
    df["roi"] = np.asarray(roi_values, dtype=float)
    df = df.dropna()
    df["ageXHI"] = df["age_c"] * df["logHI_c"]
    return sm.OLS.from_formula(FORMULA, df).fit(), df

