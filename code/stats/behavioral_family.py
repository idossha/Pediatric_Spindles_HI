"""Behavioral models of the ROIs (one per surviving HI cluster, ``stats.rois``) on the ``stats.behavioral`` TOVA frame.

DESCRIPTIVE effect sizes: the primary attention analysis is ``spindle_tova`` (channel-wise, FDR across 32 tests).

behavioral_roi_models.csv       -- TOVA_z ~ ROI_z + age_c + gender + logHI_c per ROI x outcome: beta, SE, 95% CI, p.
behavioral_adjustment_levels.csv -- each ROI x 4 outcomes, unadjusted / age+sex / age+sex+logHI: standardized and
                                   raw beta with 95% CI, R2, R2_adj, F, model_p, df.
hi_only_tova_models.csv         -- TOVA_z ~ logHI_c + age_c + gender.

Inputs : COV_PATH, ROI_PATH (``stats.rois``), METRICS_LONG.
Run    : cd code && python -m stats.behavioral_family
"""
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from project import COV_PATH, ROI_PATH, TABLE_DIR, ensure_dir
from provenance import record
from stats.behavioral import TOVA_OUTCOMES, load_behavioral_frame, roi_columns
from stats.io import matrix_inputs
from stats.regression import bh_fdr, ols_row, zscore

#: raw (pre-z) outcome column of each z-scored outcome
RAW_OUTCOME = {"DPRIMEQ1_z": "DPRIMEQ1", "OM_sqrt_z": "OM_sqrt", "COM_sqrt_z": "COM_sqrt", "RT_z": "RT"}
NOTE = "Descriptive: the ROI is the HI cluster (selected on HI, not on TOVA); no multiplicity correction."


def fit_stats(fit) -> dict:
    """Model-level fit statistics."""
    return {
        "R2": float(fit.rsquared),
        "R2_adj": float(fit.rsquared_adj),
        "F": float(fit.fvalue),
        "model_p": float(fit.f_pvalue),
        "df_model": int(fit.df_model),
        "df_resid": int(fit.df_resid),
    }


def roi_models(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for roi in roi_columns():
        for tova_label, tova_col in TOVA_OUTCOMES:
            sub = data.dropna(subset=[tova_col, roi, "age_c", "gender", "logHI_c"]).copy()
            sub["roi_z"] = zscore(sub[roi])
            fit = sm.OLS.from_formula(f"{tova_col} ~ roi_z + age_c + gender + logHI_c", sub).fit()
            rows.append({"roi": roi, "TOVA_outcome": tova_label, **ols_row(fit, "roi_z"), **fit_stats(fit), "N": int(fit.nobs)})
    return pd.DataFrame(rows)


def adjustment_levels(data: pd.DataFrame, roi: str) -> list[dict]:
    levels = [
        ("unadjusted", []),
        ("age+sex", ["age_c", "gender"]),
        ("age+sex+logHI", ["age_c", "gender", "logHI_c"]),
    ]
    rows = []
    for tova_label, tova_col in TOVA_OUTCOMES:
        raw_col = RAW_OUTCOME[tova_col]
        for level, extra in levels:
            sub = data.dropna(subset=[tova_col, roi, "age_c", "gender", "logHI_c"]).copy()
            sub["roi_z"] = zscore(sub[roi])
            fit_std = sm.OLS.from_formula(f"{tova_col} ~ " + " + ".join(["roi_z", *extra]), sub).fit()
            fit_raw = sm.OLS.from_formula(f"{raw_col} ~ " + " + ".join([roi, *extra]), sub).fit()
            std = ols_row(fit_std, "roi_z")
            raw = ols_row(fit_raw, roi)
            rows.append(
                {
                    "roi": roi,
                    "TOVA_outcome": tova_label,
                    "adjustment": level,
                    "beta_std": std["beta"],
                    "SE_std": std["SE"],
                    "CI_lower_std": std["CI_lower"],
                    "CI_upper_std": std["CI_upper"],
                    "p_std": std["p"],
                    "beta_raw": raw["beta"],
                    "SE_raw": raw["SE"],
                    "CI_lower_raw": raw["CI_lower"],
                    "CI_upper_raw": raw["CI_upper"],
                    "raw_outcome_scale": raw_col,
                    **fit_stats(fit_std),
                    "N": int(fit_std.nobs),
                }
            )
    return rows


def main() -> None:
    data = load_behavioral_frame()
    inputs = [COV_PATH, ROI_PATH, *matrix_inputs()]
    ensure_dir(TABLE_DIR)

    models = roi_models(data)
    out = TABLE_DIR / "behavioral_roi_models.csv"
    models.to_csv(out, index=False)
    record(out, script=__file__, inputs=inputs, note=NOTE)

    adj = pd.DataFrame([r for roi in roi_columns() for r in adjustment_levels(data, roi)])
    out2 = TABLE_DIR / "behavioral_adjustment_levels.csv"
    adj.to_csv(out2, index=False)
    record(out2, script=__file__, inputs=inputs, note=NOTE)

    hi_rows = []
    for tova_label, tova_col in TOVA_OUTCOMES:
        sub = data.dropna(subset=[tova_col, "age_c", "gender", "logHI_c"])
        fit = sm.OLS.from_formula(f"{tova_col} ~ logHI_c + age_c + gender", sub).fit()
        hi_rows.append({"TOVA_outcome": tova_label, "term": "logHI_c", **ols_row(fit, "logHI_c"), **fit_stats(fit), "N": int(fit.nobs)})
    hi = pd.DataFrame(hi_rows)
    hi["q_family4"] = bh_fdr(hi["p"])
    out3 = TABLE_DIR / "hi_only_tova_models.csv"
    hi.to_csv(out3, index=False)
    record(out3, script=__file__, inputs=[COV_PATH])

    pd.set_option("display.width", 200)
    print(models[["roi", "TOVA_outcome", "beta", "CI_lower", "CI_upper", "p", "N"]].to_string(index=False))
    print()
    print(adj[["roi", "TOVA_outcome", "adjustment", "beta_std", "CI_lower_std", "CI_upper_std", "p_std"]].to_string(index=False))
    print()
    print(hi.to_string(index=False))
    print(f"\n[SAVED] {out}\n[SAVED] {out2}\n[SAVED] {out3}")


if __name__ == "__main__":
    main()
