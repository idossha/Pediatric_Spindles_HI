"""Cluster-forming threshold sensitivity: the surviving HI cluster tests (``stats.clusters.surviving``) at the primary
threshold (each channel's two-sided p < .05 critical t) and at fixed |t| > 2.5 and 3.0.

Inputs : COV_PATH, CHAN_PATH, METRICS_LONG, results/tables/cluster_permutation_results.csv.
Output : results/tables/cluster_threshold_sensitivity.csv
Run    : cd code && python -m stats.threshold_sensitivity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from project import TABLE_DIR, ensure_dir
from provenance import record
from stats.clusters import (
    RESULTS_PATH,
    adjacency_matrix,
    cluster_permutation_test,
    fmt_channels,
    surviving,
)
from stats.io import (
    data_matrix_for_metric,
    load_channel_info,
    load_covariates,
    matrix_inputs,
    spindle_matrices,
)

THRESHOLDS = (None, 2.5, 3.0)  # None: the primary per-channel critical t


def main() -> None:
    channels = load_channel_info()
    cov = load_covariates()
    adjacency = adjacency_matrix(channels)
    slow, fast, sleep_mins = spindle_matrices(cov, len(channels.labels))
    band_data = {"slow": slow, "fast": fast}

    rows = []
    for label, band, metric in surviving():
        data_matrix = data_matrix_for_metric(band_data[band], metric, sleep_mins)
        for thr in THRESHOLDS:
            clusters = cluster_permutation_test(data_matrix, cov, adjacency, t_thresh=thr)
            top = clusters[0] if clusters else None
            rows.append(
                {
                    "metric_label": label,
                    "band": band,
                    "metric": metric,
                    "threshold": "p < .05" if thr is None else f"|t| > {thr}",
                    "n_channels": top["n"] if top else 0,
                    "cluster_mass": round(top["mass"], 4) if top else np.nan,
                    "peak_t": round(top["peak_t"], 4) if top else np.nan,
                    "p_value": round(top["p"], 6) if top else np.nan,
                    "survives_p05": bool(top and top["p"] < 0.05),
                    "channels_1based": fmt_channels(top["channels"]) if top else "",
                }
            )
            print(f"{label:<22}{rows[-1]['threshold']:<10}  n={rows[-1]['n_channels']:>3}  p={rows[-1]['p_value']}")

    ensure_dir(TABLE_DIR)
    out = TABLE_DIR / "cluster_threshold_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    record(out, script=__file__, inputs=[*matrix_inputs(), RESULTS_PATH])
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
