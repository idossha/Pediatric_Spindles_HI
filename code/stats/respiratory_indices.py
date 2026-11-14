"""Respiratory indices against one another: Spearman and Pearson among HI, AHI, oAHI (= AHI - CAI), OAI and CAI,
raw and log10(x + 1) coded (``log<IDX>`` rows), plus a distribution summary per index (``skew`` is scipy's
biased default; ``skew_log10`` is for log10(x + 1)).

Inputs : COV_PATH.
Output : results/tables/respiratory_indices_{correlations,distribution}.csv
Run    : cd code && python -m stats.respiratory_indices
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from project import COV_PATH, TABLE_DIR, ensure_dir
from provenance import record
from scipy import stats as sps
from stats.io import (
    RESP_INDICES_OAHI,
    RESP_LABELS,
    add_obstructive_ahi,
    load_covariates,
)


def main() -> None:
    cov = add_obstructive_ahi(load_covariates())
    raw = {RESP_LABELS[k]: cov[c].astype(float).to_numpy() for k, c in RESP_INDICES_OAHI.items()}
    log = {f"log{k}": np.log10(v + 1.0) for k, v in raw.items()}
    series = {**raw, **log}

    rows = []
    for a, b in itertools.combinations(series, 2):
        x, y = series[a], series[b]
        ok = np.isfinite(x) & np.isfinite(y)
        pr, pp = sps.pearsonr(x[ok], y[ok])
        sr, sp = sps.spearmanr(x[ok], y[ok])
        rows.append(
            {
                "index_a": a,
                "index_b": b,
                "n": int(ok.sum()),
                "pearson_r": round(float(pr), 4),
                "pearson_p": float(pp),
                "spearman_rho": round(float(sr), 4),
                "spearman_p": float(sp),
            }
        )
    corr = pd.DataFrame(rows)
    ensure_dir(TABLE_DIR)
    out = TABLE_DIR / "respiratory_indices_correlations.csv"
    corr.to_csv(out, index=False)
    record(out, script=__file__, inputs=[COV_PATH])

    drows = []
    for name, v in raw.items():
        q = np.percentile(v, [25, 50, 75])
        drows.append(
            {
                "index": name,
                "n": v.size,
                "mean": round(float(v.mean()), 4),
                "sd": round(float(v.std(ddof=1)), 4),
                "median": round(float(q[1]), 4),
                "q1": round(float(q[0]), 4),
                "q3": round(float(q[2]), 4),
                "min": round(float(v.min()), 4),
                "max": round(float(v.max()), 4),
                "n_zero": int(np.sum(v == 0)),
                "pct_zero": round(100.0 * np.mean(v == 0), 1),
                "n_le_0p5": int(np.sum(v <= 0.5)),
                "iqr": round(float(q[2] - q[0]), 4),
                "skew": round(float(sps.skew(v)), 3),
                "skew_log10": round(float(sps.skew(np.log10(v + 1.0))), 3),
            }
        )
    dist = pd.DataFrame(drows)
    out2 = TABLE_DIR / "respiratory_indices_distribution.csv"
    dist.to_csv(out2, index=False)
    record(out2, script=__file__, inputs=[COV_PATH])

    print(corr.to_string(index=False))
    print()
    print(dist.to_string(index=False))
    print(f"\n[SAVED] {out}\n[SAVED] {out2}")


if __name__ == "__main__":
    main()
