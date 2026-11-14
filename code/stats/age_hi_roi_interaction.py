"""ROI-level Age x HI interaction for every ROI column of ROI_PATH (one per surviving HI cluster, ``stats.rois``).

roi ~ age_c + logHI_c + age_c:logHI_c + sex (``stats.interaction``); simple slopes of log10(HI + 1) at the mean age
and +/- 1 SD (decimal years).

Inputs : COV_PATH, ROI_PATH
Output : results/tables/age_hi_roi_interaction.csv
Run    : cd code && python -m stats.age_hi_roi_interaction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from project import COV_PATH, ROI_PATH, TABLE_DIR, ensure_dir
from provenance import record
from scipy import stats as sps
from stats.behavioral import roi_columns
from stats.interaction import fit_interaction, prepare_covariates

#: unit of each ROI column's metric (``<band>_<metric>_cluster``)
UNITS = {"duration": "s", "frequency": "Hz", "amplitude": "µV", "density": "per min"}
PROBES = ("lo", "mid", "hi")


def simple_slope(fit, age_c_value: float) -> tuple[float, float]:
    """Slope of logHI_c at one centred age, and its two-sided p."""
    b, v = fit.params, fit.cov_params()
    slope = b["logHI_c"] + b["ageXHI"] * age_c_value
    var = v.loc["logHI_c", "logHI_c"] + age_c_value**2 * v.loc["ageXHI", "ageXHI"] + 2 * age_c_value * v.loc["logHI_c", "ageXHI"]
    t = slope / np.sqrt(var)
    return float(slope), float(2 * sps.t.sf(abs(t), df=int(fit.df_resid)))


def main() -> None:
    cov = prepare_covariates()
    cov = cov.merge(pd.read_csv(ROI_PATH), on="id", how="left")

    age_mean, age_sd = cov["age_decimal"].mean(), cov["age_decimal"].std(ddof=1)
    probe_c = {"lo": -age_sd, "mid": 0.0, "hi": age_sd}
    rows = []
    for col in roi_columns():
        fit, _ = fit_interaction(cov[col].to_numpy(dtype=float), cov)
        slopes = {tag: simple_slope(fit, probe_c[tag]) for tag in PROBES}
        rows.append(
            {
                "roi": col,
                "unit": UNITS[col.split("_")[1]],
                "n": int(fit.nobs),
                "age_mean": round(age_mean, 2),
                "age_sd": round(age_sd, 2),
                "interaction_beta": round(float(fit.params["ageXHI"]), 5),
                "interaction_t": round(float(fit.tvalues["ageXHI"]), 3),
                "interaction_p": round(float(fit.pvalues["ageXHI"]), 4),
                **{f"slope_{tag}": round(slopes[tag][0], 5) for tag in PROBES},
                **{f"slope_p_{tag}": round(slopes[tag][1], 4) for tag in PROBES},
            }
        )
    out = pd.DataFrame(rows)

    ensure_dir(TABLE_DIR)
    path = TABLE_DIR / "age_hi_roi_interaction.csv"
    out.to_csv(path, index=False)
    record(path, script=__file__, inputs=[COV_PATH, ROI_PATH])
    print(out.to_string(index=False))
    print(f"\n[SAVED] {path}")


if __name__ == "__main__":
    main()
