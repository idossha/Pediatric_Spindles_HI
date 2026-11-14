"""Robustness of the fast-duration HI cluster (fig-s3 c-d).

LOSO: ``stats.clusters.cluster_permutation_test`` refit once per child, each time without that child.
Bootstrap: 10,000 subject resamples (seed 20260615) of the standardized log HI coefficient on the participant
mean fast duration over the corrected cluster, adjusted for age and sex (z-scored within each resample).

Inputs : COV_PATH, CHAN_PATH, METRICS_LONG.
Output : results/tables/loo_fast_duration_sensitivity.csv, bootstrap_fast_duration_effect.csv,
         bootstrap_fast_duration_draws.npy (read by fig-s3 and ``density_audit``)
Run    : cd code && python -m stats.fast_duration_robustness   # a few minutes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from project import TABLE_DIR, ensure_dir
from provenance import record
from stats.clusters import (
    adjacency_matrix,
    cluster_permutation_test,
    largest_cluster_channels,
)
from stats.io import (
    PREDICTORS,
    data_matrix_for_metric,
    load_channel_info,
    load_covariates,
    matrix_inputs,
    spindle_matrices,
)

N_BOOT = 10_000
BOOT_SEED = 20260615
LOO = TABLE_DIR / "loo_fast_duration_sensitivity.csv"
BOOT = TABLE_DIR / "bootstrap_fast_duration_effect.csv"
DRAWS = TABLE_DIR / "bootstrap_fast_duration_draws.npy"


def std_beta(y: np.ndarray, cov: np.ndarray) -> float:
    """Coefficient of cov[:, 0] with y and every covariate z-scored (ddof = 0)."""
    valid = np.isfinite(y) & np.isfinite(cov).all(axis=1)
    if valid.sum() < 10 or y[valid].std() == 0:
        return np.nan
    yv, xv = y[valid], cov[valid]
    yz = (yv - yv.mean()) / yv.std()
    cols = [(c - c.mean()) / c.std() if c.std() > 0 else c - c.mean() for c in xv.T]
    return float(np.linalg.lstsq(np.column_stack([np.ones(valid.sum()), *cols]), yz, rcond=None)[0][1])


def loso(data: np.ndarray, cov: pd.DataFrame, adjacency: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, sid in enumerate(cov["id"]):
        keep = np.arange(len(cov)) != i
        top = (cluster_permutation_test(data[keep], cov[keep].reset_index(drop=True), adjacency) or [None])[0]
        rows.append({"dropped_id": sid, "dropped_hi": float(cov.at[i, "overall_hi"]),
                     "top_cluster_p": top["p"] if top else np.nan, "top_cluster_nch": top["n"] if top else 0,
                     "peak_t": top["peak_t"] if top else np.nan})
    return pd.DataFrame(rows)


def bootstrap(data: np.ndarray, cov: pd.DataFrame, adjacency: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    cluster = largest_cluster_channels(data, cov, adjacency, PREDICTORS)
    y = np.nanmean(data[:, cluster], axis=1)
    x = cov[PREDICTORS].to_numpy(dtype=float)
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(y), size=(N_BOOT, len(y)))
    draws = np.array([std_beta(y[ix], x[ix]) for ix in idx])
    draws = draws[np.isfinite(draws)]
    summary = pd.DataFrame([{
        "n_cluster_channels": len(cluster), "std_beta_observed": std_beta(y, x),
        "boot_median": float(np.median(draws)), "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)), "prop_negative": float(np.mean(draws < 0)), "n_boot": len(draws),
    }])
    return summary, draws


def main() -> None:
    channels = load_channel_info()
    cov = load_covariates()
    adjacency = adjacency_matrix(channels)
    _, fast, sleep_mins = spindle_matrices(cov, len(channels.labels))
    data = data_matrix_for_metric(fast, "Duration", sleep_mins)

    ensure_dir(TABLE_DIR)
    summary, draws = bootstrap(data, cov, adjacency)
    summary.to_csv(BOOT, index=False)
    np.save(DRAWS, draws)
    for path in (BOOT, DRAWS):
        record(path, script=__file__, inputs=matrix_inputs())
    print(summary.to_string(index=False))

    loo = loso(data, cov, adjacency)
    loo.to_csv(LOO, index=False)
    record(LOO, script=__file__, inputs=matrix_inputs())
    sig = loo["top_cluster_p"] < 0.05
    print(f"LOSO: p < .05 in {int(sig.sum())}/{len(loo)} folds; p range "
          f"{loo['top_cluster_p'].min():.4f}-{loo['top_cluster_p'].max():.4f}")


if __name__ == "__main__":
    main()
