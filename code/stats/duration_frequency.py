"""Correlation of fast-spindle duration with slow-spindle frequency, both averaged over the corrected
fast-duration HI cluster (``stats.clusters.largest_cluster_channels``), per participant.

Pearson and Spearman r; partial r for age and sex, and additionally for the child's overall spindle frequency
(detection-weighted mean over both bands and all channels).

Inputs : COV_PATH, CHAN_PATH, METRICS_LONG.
Output : results/tables/duration_frequency_correlation.csv
Run    : cd code && python -m stats.duration_frequency
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from project import TABLE_DIR, ensure_dir
from provenance import record
from scipy import stats as sps
from stats.clusters import adjacency_matrix, largest_cluster_channels
from stats.io import (
    PREDICTORS,
    data_matrix_for_metric,
    load_channel_info,
    load_covariates,
    matrix_inputs,
    spindle_matrices,
)
from stats.regression import partial_corr


def overall_frequency(slow: dict, fast: dict) -> np.ndarray:
    """Per child: detection-weighted mean Frequency over both bands and all channels."""
    count = slow["Count"] + fast["Count"]
    weighted = np.nan_to_num(slow["Count"] * slow["Frequency"]) + np.nan_to_num(fast["Count"] * fast["Frequency"])
    return weighted.sum(axis=1) / count.sum(axis=1)


def main() -> None:
    channels = load_channel_info()
    cov = load_covariates()
    slow, fast, sleep_mins = spindle_matrices(cov, len(channels.labels))
    dur = data_matrix_for_metric(fast, "Duration", sleep_mins)
    cluster = largest_cluster_channels(dur, cov, adjacency_matrix(channels), PREDICTORS)
    x = np.nanmean(dur[:, cluster], axis=1)
    y = np.nanmean(data_matrix_for_metric(slow, "Frequency", sleep_mins)[:, cluster], axis=1)
    covars = cov[["age_c", "gender_bin"]].to_numpy(dtype=float)
    overall = overall_frequency(slow, fast)

    ok = np.isfinite(x) & np.isfinite(y)
    pr, pp = sps.pearsonr(x[ok], y[ok])
    sr, sp = sps.spearmanr(x[ok], y[ok])
    par, par_p, npar = partial_corr(x, y, covars)
    paro, paro_p, naro = partial_corr(x, y, np.column_stack([covars, overall]))
    df = pd.DataFrame([{
        "region": "fast-duration corrected HI cluster (both metrics averaged over it)",
        "n_channels": len(cluster),
        "n": int(ok.sum()),
        "pearson_r": round(float(pr), 4),
        "pearson_p": float(pp),
        "spearman_rho": round(float(sr), 4),
        "spearman_p": float(sp),
        "partial_r_age_sex": round(par, 4),
        "partial_p_age_sex": float(par_p),
        "n_partial": npar,
        "partial_r_age_sex_overall_freq": round(paro, 4),
        "partial_p_age_sex_overall_freq": float(paro_p),
        "n_partial_overall_freq": naro,
    }])
    ensure_dir(TABLE_DIR)
    out = TABLE_DIR / "duration_frequency_correlation.csv"
    df.to_csv(out, index=False)
    record(out, script=__file__, inputs=matrix_inputs())
    print(df.T.to_string())
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
