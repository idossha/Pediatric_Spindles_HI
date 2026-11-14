"""Regression helpers (library): channel-wise OLS of metric ~ logHI_c + age_c + gender_bin, one term of a fit, BH-FDR,
partial correlation.

``channel_regression`` takes a subjects x channels matrix (``io.data_matrix_for_metric``) and the covariate frame
(``io.load_covariates``) and returns beta, t, p per channel for the first predictor (logHI_c).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

from .io import PREDICTORS


def zscore(series: pd.Series) -> pd.Series:
    """Population z-score (ddof=0)."""
    return (series - series.mean()) / series.std(ddof=0)


def channel_regression(
    data_matrix: np.ndarray,
    cov: pd.DataFrame,
    predictors: list[str] | None = None,
    min_n: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(betas, t_stats, p_vals) for ``predictors[0]``; channels with < ``min_n`` valid subjects stay t = NaN, p = 1."""
    predictors = predictors or PREDICTORS
    valid_rows = cov[predictors].notna().all(axis=1).to_numpy()
    x_arr = sm.add_constant(cov.loc[valid_rows, predictors].to_numpy(dtype=float))
    y_all = data_matrix[valid_rows, :]

    n_channels = data_matrix.shape[1]
    betas = np.full(n_channels, np.nan)
    t_stats = np.full(n_channels, np.nan)
    p_vals = np.ones(n_channels)

    for ch in range(n_channels):
        y = y_all[:, ch]
        ok = ~np.isnan(y)
        if ok.sum() < min_n:
            continue
        model = sm.OLS(y[ok], x_arr[ok]).fit()
        betas[ch] = model.params[1]
        t_stats[ch] = model.tvalues[1]
        p_vals[ch] = model.pvalues[1]

    return betas, t_stats, p_vals


def ols_row(fit, term: str) -> dict:
    """beta, SE, t, p and 95 % CI of one term of a statsmodels fit."""
    ci_low, ci_high = fit.conf_int().loc[term]
    return {
        "beta": float(fit.params[term]),
        "SE": float(fit.bse[term]),
        "t": float(fit.tvalues[term]),
        "p": float(fit.pvalues[term]),
        "CI_lower": float(ci_low),
        "CI_upper": float(ci_high),
    }


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg q-values (monotone step-up), NaN-safe."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def partial_corr(x, y, covars: np.ndarray) -> tuple[float, float, int]:
    """(r, p, n) partial correlation of x and y given ``covars`` (n x k), listwise complete."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(covars).all(axis=1)
    X = sm.add_constant(covars[ok])
    rx = x[ok] - X @ np.linalg.lstsq(X, x[ok], rcond=None)[0]
    ry = y[ok] - X @ np.linalg.lstsq(X, y[ok], rcond=None)[0]
    n = int(ok.sum())
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = n - 2 - covars.shape[1]
    t = r * np.sqrt(dof / max(1e-12, 1 - r**2))
    return r, float(2 * sps.t.sf(abs(t), dof)), n
