"""Primary attention analysis: channel-wise cluster tests of every spindle metric against every TOVA outcome.

8 metrics (fast, slow x density, amplitude, duration, frequency) x 4 outcomes (``stats.behavioral.TOVA_OUTCOMES``)
= 32 tests on the TOVA frame (``stats.behavioral``). Per channel: metric ~ TOVA_z + age_c + gender + logHI_c; clusters of channels at two-sided p < .05,
Freedman-Lane permutation (``common.cluster_permutation_general``: 5,000 permutations, seed 20260607). Per test, the
family-wise corrected p of its largest cluster; Benjamini-Hochberg q across the 32 tests (a test without a
suprathreshold cluster enters with p = 1).

For each test surviving q < .05, DESCRIPTIVE effect sizes (conditional on the cluster, selected on the same data): the
child's metric averaged over the cluster; the standardized beta (95% CI) of TOVA_z ~ metric_z + age_c + gender +
logHI_c; and both variables residualized on age, sex and log HI with the line through them.

Inputs : COV_PATH, PARTICIPANTS_PATH, CHAN_PATH, METRICS_LONG, ROI_PATH (the TOVA frame).
Output : results/tables/spindle_tova_clusters.csv (32 tests),
         results/tables/spindle_tova_cluster_effects_descriptive.csv (surviving tests),
         results/figure_data/spindle_tova_tmaps.csv (t and uncorrected p per channel and test, surviving-cluster
         mask),
         results/figure_data/spindle_tova_cluster_values.csv (per child, surviving tests)
Run    : cd code && python -m stats.spindle_tova
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from project import FIGURE_DATA, ROI_PATH, TABLE_DIR, ensure_dir
from provenance import record
from scipy import stats
from stats.behavioral import TOVA_OUTCOMES, load_behavioral_frame
from stats.behavioral_family import RAW_OUTCOME
from stats.clusters import adjacency_matrix, cluster_permutation_general, fmt_channels
from stats.io import (
    data_matrix_for_metric,
    load_channel_info,
    load_covariates,
    matrix_inputs,
    spindle_matrices,
)
from stats.regression import bh_fdr, ols_row, zscore

BANDS = ("fast", "slow")
METRICS = ("Density", "Amplitude", "Duration", "Frequency")
COVARIATES = ["age_c", "gender", "logHI_c"]
Q_ALPHA = 0.05
TEST_NOTE = ("metric ~ TOVA_z + age_c + gender + logHI_c per channel, TOVA frame; channels at two-sided p < .05, cluster mass, 5,000 Freedman-Lane "
             "permutations, seed 20260607; p = corrected p of the largest cluster (blank: no suprathreshold cluster); "
             "q_fdr32 = Benjamini-Hochberg over the 32 tests with p = 1 for a test without a cluster.")
DESCRIPTIVE_NOTE = ("DESCRIPTIVE: the cluster is selected on the same data (conditional on selection; p not an independent "
                    "test). TOVA_z ~ metric_z + age_c + gender + logHI_c on the child's metric averaged over the cluster; "
                    "residuals on age_c, gender, logHI_c; fit_slope = the raw-unit coefficient of the cluster mean.")


def residual(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    return y - design @ np.linalg.lstsq(design, y, rcond=None)[0]


def main() -> None:
    channels = load_channel_info()
    n_ch = len(channels.labels)
    adjacency = adjacency_matrix(channels)
    cov62 = load_covariates()
    slow, fast, sleep_mins = spindle_matrices(cov62, n_ch)
    band_data = {"slow": slow, "fast": fast}
    beh = load_behavioral_frame()
    order = {sid: i for i, sid in enumerate(cov62["id"])}
    rows_56 = np.array([order[s] for s in beh["id"]])

    tests, maps, data = [], [], []  # data[i]: (cluster channels, metric matrix, TOVA column) of test i
    for band in BANDS:
        for metric in METRICS:
            dm = data_matrix_for_metric(band_data[band], metric, sleep_mins)[rows_56, :]
            for outcome, col in TOVA_OUTCOMES:
                clusters, tmap, _ = cluster_permutation_general(dm, beh, adjacency, [col, *COVARIATES])
                top = clusters[0] if clusters else None
                tests.append({"band": band, "metric": metric, "outcome": outcome, "N": len(beh),
                              "n_channels": top["n"] if top else 0,
                              "cluster_mass": top["mass"] if top else np.nan,
                              "peak_t": top["peak_t"] if top else np.nan,
                              "p": top["p"] if top else np.nan,
                              "channels": fmt_channels(top["channels"]) if top else ""})
                data.append((top["channels"] if top else [], dm, col))
                # the channel's own residual df (subjects with the metric and every predictor, minus 5 parameters)
                n_valid = (np.isfinite(dm) & np.isfinite(beh[[col, *COVARIATES]].to_numpy(float)).all(axis=1)[:, None]).sum(axis=0)
                p_channel = 2 * stats.t.sf(np.abs(tmap), n_valid - (len(COVARIATES) + 2))
                maps.append(pd.DataFrame({"band": band, "metric": metric, "outcome": outcome,
                                          "channel": np.arange(1, n_ch + 1), "t": tmap, "p": p_channel}))
                print(f"{band:<5}{metric:<10}{outcome:<18}n={tests[-1]['n_channels']:>3}  p={tests[-1]['p']:.4f}", flush=True)

    table = pd.DataFrame(tests)
    table["q_fdr32"] = bh_fdr(table["p"].fillna(1.0))
    table["survives"] = table["q_fdr32"] < Q_ALPHA

    effects, values = [], []
    design_cols = beh[COVARIATES].to_numpy(dtype=float)
    for t in table[table["survives"]].itertuples():
        chans, dm, col = data[t.Index]
        x = np.nanmean(dm[:, chans], axis=1)
        y_z, y_raw = beh[col].to_numpy(float), beh[RAW_OUTCOME[col]].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y_z) & np.isfinite(design_cols).all(axis=1)
        sub = beh.loc[ok, COVARIATES].assign(metric_z=zscore(pd.Series(x[ok])).to_numpy(), y=y_z[ok])
        fit = sm.OLS.from_formula("y ~ metric_z + " + " + ".join(COVARIATES), sub).fit()
        design = sm.add_constant(design_cols[ok])
        xr, yr = residual(x[ok], design), residual(y_raw[ok], design)
        slope, intercept = np.polyfit(xr, yr, 1)
        effects.append({"band": t.band, "metric": t.metric, "outcome": t.outcome, "n_channels": t.n_channels,
                        "N": int(fit.nobs), "q_fdr32": t.q_fdr32, **ols_row(fit, "metric_z"),
                        "raw_outcome_scale": RAW_OUTCOME[col], "fit_slope_raw": slope})
        values.append(pd.DataFrame({"band": t.band, "metric": t.metric, "outcome": t.outcome, "id": beh.loc[ok, "id"],
                                    "cluster_mean": x[ok], "outcome_raw": y_raw[ok], "raw_outcome_scale": RAW_OUTCOME[col],
                                    "cluster_mean_residual": xr, "outcome_residual": yr,
                                    "fit_slope": slope, "fit_intercept": intercept}))

    masks = pd.concat(maps, ignore_index=True)
    masks["in_cluster"] = 0
    for t in table[table["survives"]].itertuples():
        m = (masks["band"] == t.band) & (masks["metric"] == t.metric) & (masks["outcome"] == t.outcome)
        masks.loc[m & masks["channel"].isin(np.asarray(data[t.Index][0]) + 1), "in_cluster"] = 1

    ensure_dir(TABLE_DIR)
    ensure_dir(FIGURE_DATA)
    inputs = [*matrix_inputs(), ROI_PATH]
    cols = ["band", "metric", "outcome", "N", "n_channels", "cluster_mass", "peak_t", "p", "q_fdr32", "survives", "channels"]
    outputs = [
        (TABLE_DIR / "spindle_tova_clusters.csv", table[cols], TEST_NOTE),
        (TABLE_DIR / "spindle_tova_cluster_effects_descriptive.csv", pd.DataFrame(effects), DESCRIPTIVE_NOTE),
        (FIGURE_DATA / "spindle_tova_tmaps.csv", masks, TEST_NOTE + " p = uncorrected two-sided p of the channel's TOVA t "
         "(residual df = its complete cases - 5); in_cluster = the largest cluster of a test with q < .05."),
        (FIGURE_DATA / "spindle_tova_cluster_values.csv", pd.concat(values, ignore_index=True), DESCRIPTIVE_NOTE),
    ]
    for path, df, note in outputs:
        df.to_csv(path, index=False, float_format="%.10g")
        record(path, script=__file__, inputs=inputs, note=note)
        print(f"[SAVED] {path}")
    print(table.loc[table["survives"], ["band", "metric", "outcome", "n_channels", "peak_t", "p", "q_fdr32"]].to_string(index=False))


if __name__ == "__main__":
    main()
