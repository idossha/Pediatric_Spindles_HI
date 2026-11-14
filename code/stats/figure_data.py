"""The precomputed inputs of the topographic figures: channel-wise HI regression maps, the ROI channel lists (one per
surviving HI cluster, keyed by ``clusters.roi_column``) and normative topographies, pickled for the figure producers.

Inputs : COV_PATH, CHAN_PATH, METRICS_LONG, results/tables/cluster_permutation_results.csv.
Output : results/figure_data/main_figure_data.pkl
Run    : python -m stats.figure_data
"""
from __future__ import annotations

import pickle

import mne
import numpy as np
import statsmodels.api as sm
from provenance import record
from scipy import stats

from .clusters import RESULTS_PATH, largest_cluster_channels, roi_column, surviving
from .io import (
    CHAN_PATH,
    COV_PATH,
    FIGURE_DATA,
    METRICS_LONG,
    PREDICTORS,
    data_matrix_for_metric,
    ensure_dir,
    load_channel_info,
    load_covariates,
    load_spindle_matrices,
)
from .regression import channel_regression as run_channel_regression

CACHE_DIR = FIGURE_DATA
CACHE_PATH = CACHE_DIR / "main_figure_data.pkl"


def load() -> dict:
    """Load the precomputed figure data; raise if the analysis step has not run."""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"{CACHE_PATH} not found. Run `python -m stats.figure_data` "
            "first — figures are generated from this artifact, not from raw analysis."
        )
    with open(CACHE_PATH, "rb") as fh:
        return pickle.load(fh)


def _scatter_series(data_matrix, roi_channels, cov):
    """Partial-residual scatter (ROI metric vs logHI | age, sex); ``stats.figure_tables`` writes fig-4 b's."""
    roi_avg = np.nanmean(data_matrix[:, roi_channels], axis=1)
    hi = cov["logHI"].values
    age = cov["age_c"].values
    sex = cov["gender_bin"].values
    mask = ~(np.isnan(roi_avg) | np.isnan(hi) | np.isnan(age) | np.isnan(sex))
    cov_mat = sm.add_constant(np.column_stack([age[mask], sex[mask]]))
    y_resid = roi_avg[mask] - sm.OLS(roi_avg[mask], cov_mat).fit().predict(cov_mat)
    x_resid = hi[mask] - sm.OLS(hi[mask], cov_mat).fit().predict(cov_mat)
    slope, intercept, r_val, _p, _se = stats.linregress(x_resid, y_resid)
    return {"x": x_resid, "y": y_resid, "slope": slope, "intercept": intercept, "r": r_val}


def compute() -> dict:
    mne.set_log_level("ERROR")
    channels = load_channel_info()
    info = channels.info
    nCh = len(channels.labels)

    cov = load_covariates(COV_PATH)
    matrices = load_spindle_matrices(cov, nCh, include_frequency=True)
    slow, fast, sleep_mins = matrices.slow, matrices.fast, matrices.sleep_mins
    band_dict = {"slow": slow, "fast": fast}

    adj = mne.channels.find_ch_adjacency(info, ch_type="eeg")[0].toarray().astype(bool)

    # Figure 2: normative per-channel topographies
    topo = {
        "den_slow": np.nanmean(slow["Count"] / sleep_mins, axis=0),
        "den_fast": np.nanmean(fast["Count"] / sleep_mins, axis=0),
        "dur_slow": np.nanmean(slow["Duration"], axis=0),
        "dur_fast": np.nanmean(fast["Duration"], axis=0),
    }

    # ROIs: the HI clusters that survive correction
    roi = {roi_column(band, metric): largest_cluster_channels(
        data_matrix_for_metric(band_dict[band], metric, sleep_mins), cov, adj, PREDICTORS)
        for _, band, metric in surviving()}

    # Channel-wise HI regression maps (fig-4 a, the graphical abstract, the uncorrected channel counts)
    reg = {}
    for band, metric in [("fast", "Duration"), ("slow", "Frequency"), ("slow", "Amplitude"),
                         ("slow", "Duration"), ("slow", "Density"), ("fast", "Density")]:
        beta, t, p = run_channel_regression(data_matrix_for_metric(band_dict[band], metric, sleep_mins), cov, PREDICTORS)
        reg[(band, metric)] = {"beta": beta, "t": t, "p": p}

    return {"topo": topo, "roi": roi, "reg": reg, "channel_labels": channels.labels}


def main() -> None:
    import argparse

    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    data = compute()
    ensure_dir(CACHE_DIR)
    with open(CACHE_PATH, "wb") as fh:
        pickle.dump(data, fh)
    record(CACHE_PATH, script=__file__, inputs=[COV_PATH, CHAN_PATH, METRICS_LONG, RESULTS_PATH])
    print(f"[SAVED] {CACHE_PATH}")
    print("  ROIs: " + ", ".join(f"{k}={len(v)}" for k, v in data["roi"].items()))


if __name__ == "__main__":
    main()
