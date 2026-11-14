"""Do the corrected HI clusters (``stats.clusters.surviving``) survive one added sleep or oxygenation covariate (one at a time)?

Covariates: ari, se (sleep efficiency; a recorded 0.0 % is missing), tst, waso, n2 / n3 (% of TST), clean_min,
odi (>= 3 % desaturations per h, from ``stats.oxygenation``).

covariate_sensitivity.csv -- ``stats.clusters`` test of each surviving cluster's metric with the covariate in the
    design; same permutation stream for every row. ``base_same_n_*`` = the unadjusted model on the same children;
    ``n_primary_suprathreshold`` / ``mean_t_in_primary`` = how much of the primary cluster survives; ``selcluster_*``
    = OLS on the mean over the primary cluster (selected on the same t-map: conditional on selection, p not corrected).
behavioral_covariate_sensitivity.csv -- the TOVA ~ ROI model (each ROI of ROI_PATH) with the covariate added (the TOVA frame).

``cluster_mean_series`` (corrected fast-duration cluster ROI) is imported by ``figure_tables``.

Inputs : COV_PATH, PARTICIPANTS_PATH, CHAN_PATH, METRICS_LONG, ROI_PATH, results/tables/oxygenation_derived.csv.
Output : results/tables/covariate_sensitivity.csv, behavioral_covariate_sensitivity.csv
Run    : cd code && python -m stats.covariate_sensitivity      (after stats.oxygenation)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from project import ROI_PATH, TABLE_DIR, ensure_dir
from provenance import record
from stats.behavioral import TOVA_OUTCOMES, load_behavioral_frame, roi_columns
from stats.clusters import (
    RESULTS_PATH,
    adjacency_matrix,
    cluster_permutation_general,
    fmt_channels,
    largest_cluster_channels,
    surviving,
)
from stats.io import (
    PREDICTORS,
    data_matrix_for_metric,
    load_channel_info,
    load_covariates,
    matrix_inputs,
    spindle_matrices,
)
from stats.regression import ols_row, zscore

COVARIATES = [
    ("ari", "Arousal index (per h)"),
    ("se", "Sleep efficiency (%)"),
    ("tst", "Total sleep time (min)"),
    ("waso", "Wake after sleep onset (min)"),
    ("n2", "N2 (% of TST)"),
    ("n3", "N3 (% of TST)"),
    ("clean_min", "Artifact-free N2+N3 (min)"),
    ("odi", "Oxygen desaturation index, >=3% (per h; derived from scored events)"),
]
OXY_PATH = TABLE_DIR / "oxygenation_derived.csv"
NOTE = (
    "Channel-wise HI cluster permutation with one covariate added (channels at two-sided p < .05, cluster mass, 5,000 Freedman-Lane "
    "permutations, seed 20260607). base_same_n_* = HI + age + sex on the same children. selcluster_* = OLS on the "
    "mean over the primary cluster selected on the same HI t-map: "
    "conditional on that selection, p not corrected for it. se: one child's 0.0% set missing. odi from oxygenation_derived.csv."
)


def attach_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Merge ODI from ``oxygenation_derived.csv``; sleep efficiency of 0 % becomes missing."""
    oxy = pd.read_csv(OXY_PATH)[["id", "odi"]]
    out = df.drop(columns=["odi"], errors="ignore").merge(oxy, on="id", how="left")
    out["se"] = out["se"].where(out["se"] > 0)
    return out


def selected_cluster_mean(dm: np.ndarray, cov: pd.DataFrame, adjacency: np.ndarray) -> tuple[pd.Series, list[int]]:
    """Participant mean of ``dm`` over the largest observed HI cluster (channels at p < .05)."""
    cluster = largest_cluster_channels(dm, cov, adjacency, PREDICTORS)
    sub = dm[:, cluster]
    vals = np.full(len(cov), np.nan)
    ok = ~np.all(np.isnan(sub), axis=1)
    vals[ok] = np.nanmean(sub[ok], axis=1)
    return pd.Series(vals, index=cov.index), cluster


def cluster_mean_series(cov: pd.DataFrame) -> tuple[pd.Series, list[int]]:
    """Participant mean fast-spindle duration over the corrected fast-duration HI cluster."""
    channels = load_channel_info()
    _, fast, sleep_mins = spindle_matrices(cov, len(channels.labels))
    vals, cluster = selected_cluster_mean(data_matrix_for_metric(fast, "Duration", sleep_mins), cov, adjacency_matrix(channels))
    return vals.rename("cluster_fast_dur"), cluster


def cluster_summary(clusters: list[dict], tmap: np.ndarray, crit: np.ndarray, primary: list[int]) -> dict:
    top = clusters[0] if clusters else None
    return {
        "n_channels": top["n"] if top else 0,
        "cluster_mass": round(top["mass"], 4) if top else np.nan,
        "peak_t": round(top["peak_t"], 4) if top else np.nan,
        "p_corrected": round(top["p"], 6) if top else np.nan,
        "survives_p05": bool(top and top["p"] < 0.05),
        "n_clusters_formed": len(clusters),
        "n_primary_suprathreshold": int(np.sum(np.abs(tmap[primary]) > crit[primary])),
        "mean_t_in_primary": round(float(np.nanmean(tmap[primary])), 4),
        "channels_1based": fmt_channels(top["channels"]) if top else "",
    }


def ols(df: pd.DataFrame, y: str, terms: list[str]):
    return sm.OLS.from_formula(f"{y} ~ " + " + ".join(terms), df).fit()


def topographic_rows(cov: pd.DataFrame) -> pd.DataFrame:
    channels = load_channel_info()
    adjacency = adjacency_matrix(channels)
    slow, fast, sleep_mins = spindle_matrices(cov, len(channels.labels))
    band_data = {"slow": slow, "fast": fast}
    rows = []
    for label, band, metric in surviving():
        dm = data_matrix_for_metric(band_data[band], metric, sleep_mins)
        sel, primary = selected_cluster_mean(dm, cov, adjacency)
        df = cov.assign(sel=sel)
        base_all = cluster_summary(*cluster_permutation_general(dm, cov, adjacency, PREDICTORS), primary)
        for extra, extra_label in [(None, "(none: HI + age + sex)"), *COVARIATES]:
            ok = df[extra].notna() if extra else pd.Series(True, index=df.index)
            if extra is None:
                adjusted, base = base_all, base_all
            else:
                adjusted = cluster_summary(*cluster_permutation_general(dm, cov, adjacency, PREDICTORS + [extra]), primary)
                if ok.all():
                    base = base_all
                else:  # unadjusted model on the children who have the covariate
                    restricted = cov.copy()
                    restricted.loc[~ok, PREDICTORS] = np.nan
                    base = cluster_summary(*cluster_permutation_general(dm, restricted, adjacency, PREDICTORS), primary)
            sub = df[ok].dropna(subset=["sel", *PREDICTORS])
            fit = ols(sub, "sel", PREDICTORS + ([extra] if extra else []))
            fit0 = ols(sub, "sel", PREDICTORS)
            hi = ols_row(fit, "logHI_c")
            rows.append(
                {
                    "metric_label": label,
                    "band": band,
                    "metric": metric,
                    "added_covariate": extra or "",
                    "covariate_label": extra_label,
                    "n": int(ok.sum()),
                    "covariate_r_logHI": round(float(df.loc[ok, extra].corr(df.loc[ok, "logHI_c"])), 4) if extra else np.nan,
                    "covariate_rho_logHI": round(float(df.loc[ok, extra].corr(df.loc[ok, "logHI_c"], method="spearman")), 4) if extra else np.nan,
                    **adjusted,
                    "base_same_n_n_channels": base["n_channels"],
                    "base_same_n_cluster_mass": base["cluster_mass"],
                    "base_same_n_p_corrected": base["p_corrected"],
                    "selcluster_definition": f"mean over the {len(primary)}-channel HI cluster selected on the same data (conditional on selection)",
                    "selcluster_n_channels": len(primary),
                    "selcluster_n": int(fit.nobs),
                    "selcluster_logHI_beta": round(hi["beta"], 4),
                    "selcluster_logHI_SE": round(hi["SE"], 4),
                    "selcluster_logHI_p": round(hi["p"], 6),
                    "selcluster_covariate_beta": round(float(fit.params[extra]), 6) if extra else np.nan,
                    "selcluster_covariate_p": round(float(fit.pvalues[extra]), 6) if extra else np.nan,
                    "selcluster_base_same_n_logHI_beta": round(float(fit0.params["logHI_c"]), 4),
                    "selcluster_base_same_n_logHI_p": round(float(fit0.pvalues["logHI_c"]), 6),
                }
            )
            r = rows[-1]
            print(
                f"{label:<20} {extra or 'none':<14} n={r['n']:>2}  cluster {r['n_channels']:>2} ch p={r['p_corrected']}"
                f"  (same-n base {r['base_same_n_n_channels']} ch p={r['base_same_n_p_corrected']})"
                f"  primary supra={r['n_primary_suprathreshold']}/{len(primary)}  selcluster b={r['selcluster_logHI_beta']} p={r['selcluster_logHI_p']}"
            )
    return pd.DataFrame(rows)


def behavioral_rows(roi: str) -> pd.DataFrame:
    data = attach_covariates(load_behavioral_frame())
    data = data.assign(roi_cluster=data[roi])
    base_terms = ["roi_cluster_z", "age_c", "gender", "logHI_c"]
    rows = []
    for tova_label, tova_col in TOVA_OUTCOMES:
        for extra, extra_label in [(None, "(none: ROI + age + sex + log HI)"), *COVARIATES]:
            sub = data.dropna(subset=[tova_col, "roi_cluster", "age_c", "gender", "logHI_c", *([extra] if extra else [])]).copy()
            sub["roi_cluster_z"] = zscore(sub["roi_cluster"])
            fit = ols(sub, tova_col, base_terms + ([extra] if extra else []))
            fit0 = ols(sub, tova_col, base_terms)
            rows.append(
                {
                    "roi": roi,
                    "TOVA_outcome": tova_label,
                    "added_covariate": extra or "",
                    "covariate_label": extra_label,
                    "term": "roi_cluster_z",
                    "n": int(fit.nobs),
                    **{k: round(v, 6 if k == "p" else 4) for k, v in ols_row(fit, "roi_cluster_z").items()},
                    "base_same_n_beta": round(float(fit0.params["roi_cluster_z"]), 4),
                    "base_same_n_SE": round(float(fit0.bse["roi_cluster_z"]), 4),
                    "base_same_n_p": round(float(fit0.pvalues["roi_cluster_z"]), 6),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    cov = attach_covariates(load_covariates())
    print("[INFO] covariate n: " + ", ".join(f"{c}={int(cov[c].notna().sum())}" for c, _ in COVARIATES))
    topo = topographic_rows(cov)

    beh = pd.concat([behavioral_rows(roi) for roi in roi_columns()], ignore_index=True)

    ensure_dir(TABLE_DIR)
    out = TABLE_DIR / "covariate_sensitivity.csv"
    topo.to_csv(out, index=False)
    record(out, script=__file__, inputs=[*matrix_inputs(), RESULTS_PATH, OXY_PATH], note=NOTE)
    out2 = TABLE_DIR / "behavioral_covariate_sensitivity.csv"
    beh.to_csv(out2, index=False)
    record(
        out2,
        script=__file__,
        inputs=[*matrix_inputs(), ROI_PATH, OXY_PATH],
        note="TOVA_z ~ roi_cluster_z + age_c + gender + logHI_c [+ one covariate], TOVA frame of stats.behavioral; "
        "roi_cluster = the ROI column of roi_values.csv (mean over a corrected HI cluster), z-scored within each model's sample; "
        "base_same_n_* = the model without the covariate on the same children. se: one child's 0.0% set missing.",
    )
    print(beh.to_string(index=False))
    print(f"\n[SAVED] {out}\n[SAVED] {out2}")


if __name__ == "__main__":
    main()
