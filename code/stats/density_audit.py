"""Spindle yield per child (detections per band, channels without detections, cleaned minutes, channel-averaged
density), its minima and the lowest-yield child, and the primary results without that child; also the number of
leave-one-out folds keeping the fast-duration cluster significant.

Inputs : COV_PATH, CHAN_PATH, ROI_PATH, METRICS_LONG, results/tables/loo_fast_duration_sensitivity.csv.
Output : results/tables/density_audit.csv, density_audit_summary.csv, density_audit_robustness.csv
Run    : cd code && python -m stats.density_audit
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from project import ROI_PATH, TABLE_DIR, ensure_dir
from provenance import record
from stats.behavioral import TOVA_OUTCOMES, load_behavioral_frame
from stats.clusters import (
    adjacency_matrix,
    cluster_permutation_general,
    largest_cluster_channels,
)
from stats.io import (
    PREDICTORS,
    load_channel_info,
    load_covariates,
    matrix_inputs,
    spindle_matrices,
)
from stats.regression import ols_row, zscore

LOO_PATH = TABLE_DIR / "loo_fast_duration_sensitivity.csv"
PRIMARY_TOVA = [(label, col) for label, col in TOVA_OUTCOMES if col in ("DPRIMEQ1_z", "OM_sqrt_z")]


def roi_mean(matrix: np.ndarray, channels: list[int]) -> np.ndarray:
    return np.nanmean(matrix[:, channels], axis=1)


def recentre(df: pd.DataFrame) -> pd.DataFrame:
    """logHI_c and age_c centred on the sample at hand (does not change any tested slope)."""
    log_hi = np.log10(df["overall_hi"] + 1)
    return df.assign(logHI_c=log_hi - log_hi.mean(), age_c=df["age_decimal"] - df["age_decimal"].mean())


def participant_table(cov: pd.DataFrame, slow: dict, fast: dict, sleep_mins: np.ndarray) -> pd.DataFrame:
    cs, cf = slow["Count"], fast["Count"]
    ds, df_ = cs / sleep_mins, cf / sleep_mins
    return pd.DataFrame(
        {
            "id": cov["id"].to_numpy(),
            "age_decimal": cov["age_decimal"].to_numpy(),
            "overall_hi": cov["overall_hi"].to_numpy(),
            "clean_min": sleep_mins[:, 0],
            "detections_total": (cs + cf).sum(axis=1).astype(int),
            "detections_slow": cs.sum(axis=1).astype(int),
            "detections_fast": cf.sum(axis=1).astype(int),
            "channels_without_fast": (cf == 0).sum(axis=1),
            "channels_without_slow": (cs == 0).sum(axis=1),
            "channels_without_any": ((cs + cf) == 0).sum(axis=1),
            "density_fast": df_.mean(axis=1),
            "density_slow": ds.mean(axis=1),
            "density_fast_channel_median": np.median(df_, axis=1),
            "density_slow_channel_median": np.median(ds, axis=1),
            "density_fast_channel_max": df_.max(axis=1),
            "density_slow_channel_max": ds.max(axis=1),
        }
    ).sort_values("detections_total", ignore_index=True)


def summary_rows(per: pd.DataFrame, loo: pd.DataFrame) -> list[dict]:
    rows = []
    for band in ("fast", "slow"):
        col = f"density_{band}"
        i = per[col].idxmin()
        rows += [
            {"quantity": f"{band} density: minimum (events/min)", "value": per.at[i, col], "id": per.at[i, "id"]},
            {"quantity": f"{band} density: children rounding to 0.0 at one decimal", "value": int((per[col].round(1) == 0).sum())},
            {"quantity": f"{band} density: children rounding to 0.00 at two decimals", "value": int((per[col].round(2) == 0).sum())},
            {"quantity": f"children with at least one channel without {band} detections", "value": int((per[f"channels_without_{band}"] > 0).sum())},
        ]
    rows.append({"quantity": "children with at least one channel without any detection",
                 "value": int((per["channels_without_any"] > 0).sum())})
    lid = int(per.at[0, "id"])
    for col in ("detections_total", "detections_slow", "detections_fast", "channels_without_fast",
                "channels_without_slow", "channels_without_any", "clean_min", "density_fast",
                "density_slow", "age_decimal", "overall_hi"):
        rows.append({"quantity": f"lowest-yield participant: {col}", "value": per.at[0, col], "id": lid})
    rows.append({"quantity": "next-lowest total detections", "value": per.at[1, "detections_total"], "id": per.at[1, "id"]})

    sig = loo["top_cluster_p"] < 0.05
    rows += [
        {"quantity": "leave-one-out folds with fast-duration cluster p < .05", "value": int(sig.sum()), "n": len(loo)},
        {"quantity": "leave-one-out folds with p >= .05", "value": int((~sig).sum()), "n": len(loo)},
        {"quantity": "leave-one-out: smallest p among non-significant folds", "value": float(loo.loc[~sig, "top_cluster_p"].min())},
        {"quantity": "leave-one-out: largest p across folds", "value": float(loo["top_cluster_p"].max()),
         "id": int(loo.loc[loo["top_cluster_p"].idxmax(), "dropped_id"])},
    ]
    return rows


def robustness_rows(cov: pd.DataFrame, fast: dict, adjacency: np.ndarray, drop_id: int, loo: pd.DataFrame) -> list[dict]:
    dur = fast["Duration"]
    roi = largest_cluster_channels(dur, cov, adjacency, PREDICTORS)
    roi_vals = pd.Series(roi_mean(dur, roi), index=cov["id"])
    keep = (cov["id"] != drop_id).to_numpy()
    rows = []

    for sample, frame, mat in (("full", cov, dur), (f"without id {drop_id}", recentre(cov[keep].reset_index(drop=True)), dur[keep])):
        d = frame.assign(roi=frame["id"].map(roi_vals).to_numpy())
        fit = sm.OLS.from_formula("roi ~ logHI_c + age_c + gender_bin", d).fit()
        rows.append({"analysis": "corrected-cluster ROI fast duration ~ logHI (age, sex)", "sample": sample,
                     "term": "logHI_c", **ols_row(fit, "logHI_c"), "n": int(fit.nobs), "n_channels": len(roi)})

        clusters, _, _ = cluster_permutation_general(mat, frame, adjacency, PREDICTORS)
        top = clusters[0]
        rows.append({"analysis": "fast-duration cluster permutation (p < .05 channels, 5,000 permutations)", "sample": sample,
                     "term": "cluster", "t": top["peak_t"], "p": top["p"], "cluster_mass": top["mass"],
                     "n": len(frame), "n_channels": top["n"]})

    loo_row = loo.loc[loo["dropped_id"] == drop_id]
    if len(loo_row):
        mine = rows[-1]
        assert abs(float(loo_row["top_cluster_p"].iloc[0]) - mine["p"]) < 1e-9, "LOO table disagrees with the rerun"
        assert int(loo_row["top_cluster_nch"].iloc[0]) == mine["n_channels"]
        print(f"[CHECK] cluster without id {drop_id} equals its leave-one-out row (p = {mine['p']:.6f})")

    beh = load_behavioral_frame()
    beh = beh.assign(roi_cluster=beh["id"].map(roi_vals).to_numpy())
    raw = {"DPRIMEQ1_z": "DPRIMEQ1", "OM_sqrt_z": "OM_sqrt"}
    for sample, frame in (("full", beh), (f"without id {drop_id}", recentre(beh[beh["id"] != drop_id]))):
        for label, col in PRIMARY_TOVA:
            d = frame.assign(roi_z=zscore(frame["roi_cluster"]), y=zscore(frame[raw[col]]))
            fit = sm.OLS.from_formula("y ~ roi_z + age_c + gender + logHI_c", d).fit()
            rows.append({"analysis": f"TOVA {label} ~ corrected-cluster ROI (age, sex, logHI)", "sample": sample,
                         "term": "roi_z (standardized)", **ols_row(fit, "roi_z"), "n": int(fit.nobs), "n_channels": len(roi)})
    return rows


def main() -> None:
    channels = load_channel_info()
    cov = load_covariates()
    adjacency = adjacency_matrix(channels)
    slow, fast, sleep_mins = spindle_matrices(cov, len(channels.labels))
    loo = pd.read_csv(LOO_PATH)
    assert len(loo) == len(cov) and set(loo["dropped_id"]) == set(cov["id"])

    per = participant_table(cov, slow, fast, sleep_mins)
    lowest = int(per.at[0, "id"])
    summary = pd.DataFrame(summary_rows(per, loo), dtype=object)
    robust = pd.DataFrame(robustness_rows(cov, fast, adjacency, lowest, loo))

    ensure_dir(TABLE_DIR)
    inputs = matrix_inputs()
    outs = {
        "density_audit.csv": (per, inputs),
        "density_audit_summary.csv": (summary, [*inputs, LOO_PATH]),
        "density_audit_robustness.csv": (robust, [*inputs, ROI_PATH, LOO_PATH]),
    }
    for name, (df, ins) in outs.items():
        out = TABLE_DIR / name
        df.to_csv(out, index=False)
        record(out, script=__file__, inputs=ins)
        print(f"[SAVED] {out}")

    pd.set_option("display.width", 220)
    print(per.head(6).to_string(index=False))
    print(summary.to_string(index=False))
    print(robust.to_string(index=False))


if __name__ == "__main__":
    main()
