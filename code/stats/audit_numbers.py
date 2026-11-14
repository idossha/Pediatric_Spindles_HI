"""Numbers the manuscript prose quotes that no other table carries; outputs in results/tables/.

(a) tova_exclusion_rule.csv        -- sample -> with TOVA -> TOVA frame: |z| >= 3 on sqrt omissions or mean RT (as ``stats.behavioral``).
(b) channel_integrity.csv          -- valid children per channel per band x metric; residual df = min - 4.
(c) normative_topography_peaks.csv -- topographic min / max / whole-head mean per band (density, duration from
                                      ``main_figure_data.pkl``; amplitude, count from METRICS_LONG).
(d) behavioral_covariate_terms.csv -- age, sex, log HI coefficients of the ROI x TOVA models (``behavioral_family``).
(e) roi_channel_membership.csv     -- the ROIs (surviving HI clusters) as 1-based inside-172 positions and EGI labels.
(f) uncorrected_channel_counts.csv -- channels at uncorrected p < .05 (negative / positive) per channel-wise HI map.
(g) null_metric_clusters.csv       -- cluster test for fast / slow count and fast amplitude.

Inputs : COV_PATH, ROI_PATH, CHAN_PATH, METRICS_LONG, FIGURE_DATA/main_figure_data.pkl.
Run    : cd code && python -m stats.audit_numbers
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from project import (
    CHAN_PATH,
    FIGURE_DATA,
    METRICS_LONG,
    ROI_PATH,
    TABLE_DIR,
    ensure_dir,
)
from provenance import record
from scipy import stats as sps
from stats.behavioral import TOVA_OUTCOMES, load_behavioral_frame, roi_columns
from stats.clusters import adjacency_matrix, cluster_permutation_test
from stats.io import (
    data_matrix_for_metric,
    load_channel_info,
    load_covariates,
    matrix_inputs,
    spindle_matrices,
)

PKL = FIGURE_DATA / "main_figure_data.pkl"


def _maps() -> dict:
    with open(PKL, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------- (a) exclusion rule
def exclusion_rule() -> pd.DataFrame:
    cov = load_covariates()
    roi = pd.read_csv(ROI_PATH)
    data = cov.merge(roi, on="id", how="left").dropna(subset=["DPRIMEQ1"]).copy()
    data["OM_sqrt"] = np.sqrt(data["OMPERQ1"])
    data["COM_sqrt"] = np.sqrt(data["COMPERQ1"])
    data["RT"] = data["RTMEANQ1"].astype(float)
    data["RT_raw"] = data["RTMEANQ1"]

    rows = []
    flagged = pd.Series(False, index=data.index)
    for label, col, in_rule in [
        ("sqrt(omission error %)", "OM_sqrt", True),
        ("mean reaction time", "RT", True),
        ("mean reaction time (raw, ms)", "RT_raw", False),
        ("sqrt(commission error %)", "COM_sqrt", False),
    ]:
        z = np.abs(sps.zscore(data[col]))
        hit = z >= 3
        if in_rule:
            flagged |= hit
        rows.append(
            {
                "variable": label,
                "column": col,
                "in_applied_rule": in_rule,
                "n_complete_tova": len(data),
                "max_abs_z": round(float(z.max()), 3),
                "n_excluded_by_this_variable": int(hit.sum()),
                "excluded_ids": ";".join(str(int(i)) for i in data.loc[hit, "id"]),
            }
        )

    excl = data.loc[flagged]
    rows.append(
        {
            "variable": "APPLIED RULE (|z| >= 3 on sqrt omissions OR mean RT)",
            "column": "OM_sqrt|RT",
            "in_applied_rule": True,
            "n_complete_tova": len(data),
            "max_abs_z": np.nan,
            "n_excluded_by_this_variable": int(flagged.sum()),
            "excluded_ids": ";".join(str(int(i)) for i in excl["id"]),
        }
    )
    return pd.DataFrame(rows)


# ------------------------------------------------------------- (b) channel integrity
def _analysed_matrices():
    cov = load_covariates()
    channels = load_channel_info()
    slow, fast, sleep_mins = spindle_matrices(cov, len(channels.labels))
    return cov, channels, {"slow": slow, "fast": fast}, sleep_mins


def channel_integrity() -> pd.DataFrame:
    cov, _, bands, _ = _analysed_matrices()
    rows = []
    for band in ("slow", "fast"):
        for metric in ("Amplitude", "Duration", "Frequency"):
            counts = np.isfinite(bands[band][metric]).sum(axis=0)
            rows.append(
                {
                    "band": band,
                    "metric": metric,
                    "n_channels": int(counts.size),
                    "min_valid_of_62": int(counts.min()),
                    "max_valid_of_62": int(counts.max()),
                    "mean_pct_valid": round(float(counts.mean() / len(cov) * 100), 1),
                    "min_residual_df": int(counts.min()) - 4,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------- (f) uncorrected channel counts
def uncorrected_channel_counts() -> pd.DataFrame:
    rows = []
    for (band, metric), reg in _maps()["reg"].items():
        p = np.asarray(reg["p"], dtype=float)
        t = np.asarray(reg["t"], dtype=float)
        sig = p < 0.05
        rows.append(
            {
                "band": band,
                "metric": metric,
                "n_channels_p05": int(sig.sum()),
                "n_negative_p05": int((sig & (t < 0)).sum()),
                "n_positive_p05": int((sig & (t > 0)).sum()),
                "max_abs_t": round(float(np.nanmax(np.abs(t))), 4),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------- (g) null-metric cluster tests
def null_metric_clusters() -> pd.DataFrame:
    cov, channels, bands, sleep_mins = _analysed_matrices()
    adjacency = adjacency_matrix(channels)
    rows = []
    for band, metric in (("fast", "Count"), ("slow", "Count"), ("fast", "Amplitude")):
        clusters = cluster_permutation_test(data_matrix_for_metric(bands[band], metric, sleep_mins), cov, adjacency)
        top = clusters[0] if clusters else None
        rows.append(
            {
                "band": band,
                "metric": metric,
                "n_channels": top["n"] if top else 0,
                "cluster_mass": round(top["mass"], 4) if top else np.nan,
                "peak_t": round(top["peak_t"], 4) if top else np.nan,
                "p_value": round(top["p"], 6) if top else np.nan,
                "survives_p05": bool(top and top["p"] < 0.05),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------- (c) normative topography peaks
def topography_peaks() -> pd.DataFrame:
    topo = _maps()["topo"]
    rows = []
    label = {
        "den_slow": ("slow", "Density", "events/min"),
        "den_fast": ("fast", "Density", "events/min"),
        "dur_slow": ("slow", "Duration", "s"),
        "dur_fast": ("fast", "Duration", "s"),
    }
    for k, (band, metric, unit) in label.items():
        v = np.asarray(topo[k], dtype=float)
        rows.append(
            {
                "band": band,
                "metric": metric,
                "unit": unit,
                "source": "results/figure_data/main_figure_data.pkl:topo/" + k,
                "whole_head_mean": round(float(np.nanmean(v)), 4),
                "topographic_min": round(float(np.nanmin(v)), 4),
                "topographic_max": round(float(np.nanmax(v)), 4),
            }
        )

    # amplitude and count: channel group means of METRICS_LONG (count zero-filled)
    long = pd.read_csv(METRICS_LONG)
    key = {c.lower(): c for c in long.columns}
    ch_col = key.get("channel") or key.get("idxchannel")
    for metric, col, unit in [("Amplitude", key["amplitude"], "uV"), ("Count", key["count"], "events")]:
        for band, sub in long.groupby(key["band"]):
            m = sub.groupby(ch_col)[col].mean()
            rows.append(
                {
                    "band": str(band).lower(),
                    "metric": metric,
                    "unit": unit,
                    "source": "results/tables/spindle_metrics_long.csv (channel group means)",
                    "whole_head_mean": round(float(m.mean()), 4),
                    "topographic_min": round(float(m.min()), 4),
                    "topographic_max": round(float(m.max()), 4),
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------ (d) behavioral covariate terms
def covariate_terms() -> pd.DataFrame:
    data = load_behavioral_frame()
    rows = []
    for tova_label, tova_col in TOVA_OUTCOMES:
        for roi_col in roi_columns():
            sub = data.dropna(subset=[tova_col, roi_col, "age_c", "gender", "logHI_c"]).copy()
            sub["roi_z"] = (sub[roi_col] - sub[roi_col].mean()) / sub[roi_col].std(ddof=1)
            fit = sm.OLS.from_formula(f"{tova_col} ~ roi_z + age_c + gender + logHI_c", sub).fit()
            for term, name in [("age_c", "age"), ("gender", "sex"), ("logHI_c", "log HI")]:
                rows.append(
                    {
                        "roi": roi_col,
                        "TOVA_outcome": tova_label,
                        "covariate": name,
                        "term": term,
                        "beta": float(fit.params[term]),
                        "SE": float(fit.bse[term]),
                        "t": float(fit.tvalues[term]),
                        "p": float(fit.pvalues[term]),
                        "N": len(sub),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------- (e) ROI channel membership
def roi_membership() -> pd.DataFrame:
    labels = load_channel_info(CHAN_PATH).labels
    rows = []
    for name, channels in _maps()["roi"].items():
        idx = sorted(int(i) for i in channels)
        rows.append(
            {
                "roi": name,
                "n_channels": len(idx),
                "inside172_positions": " ".join(str(i + 1) for i in idx),
                "egi_labels": " ".join(labels[i] for i in idx),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dir(TABLE_DIR)
    inputs_all = matrix_inputs() + [ROI_PATH, PKL]
    outputs = {
        "tova_exclusion_rule.csv": (exclusion_rule(), inputs_all),
        "channel_integrity.csv": (channel_integrity(), matrix_inputs()),
        "normative_topography_peaks.csv": (topography_peaks(), inputs_all),
        "behavioral_covariate_terms.csv": (covariate_terms(), inputs_all),
        "roi_channel_membership.csv": (roi_membership(), inputs_all),
        "uncorrected_channel_counts.csv": (uncorrected_channel_counts(), [PKL]),
        "null_metric_clusters.csv": (null_metric_clusters(), matrix_inputs()),
    }
    for name, (df, inputs) in outputs.items():
        path = TABLE_DIR / name
        df.to_csv(path, index=False)
        record(path, script=__file__, inputs=inputs)
        print(f"[SAVED] {path}")
        print(df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
