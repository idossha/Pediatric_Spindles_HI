"""Per-participant ROI values: one column per HI cluster that survives correction (``clusters.surviving``), the
mean of the metric over that cluster's channels (``clusters.largest_cluster_channels``).

Inputs : COV_PATH, CHAN_PATH, METRICS_LONG, results/tables/cluster_permutation_results.csv.
Output : results/tables/roi_values.csv (ROI_PATH, + provenance sidecar), read by the behavioral models.
Run    : python -m stats.rois
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from provenance import record

from .clusters import RESULTS_PATH, largest_cluster_channels, roi_column, surviving
from .io import (
    CHAN_PATH,
    COV_PATH,
    METRICS_LONG,
    PREDICTORS,
    ROI_PATH,
    data_matrix_for_metric,
    ensure_dir,
    load_channel_info,
    load_covariates,
    load_spindle_matrices,
)


def main() -> None:
    import mne

    mne.set_log_level("ERROR")
    channels = load_channel_info(CHAN_PATH)
    cov = load_covariates(COV_PATH)
    matrices = load_spindle_matrices(cov, len(channels.labels), include_frequency=True)
    bands = {"slow": matrices.slow, "fast": matrices.fast}
    adjacency = mne.channels.find_ch_adjacency(channels.info, ch_type="eeg")[0].toarray().astype(bool)

    results = pd.DataFrame({"id": cov["id"]})
    for label, band, metric in surviving():
        dm = data_matrix_for_metric(bands[band], metric, matrices.sleep_mins)
        cluster = largest_cluster_channels(dm, cov, adjacency, PREDICTORS)
        results[roi_column(band, metric)] = np.nanmean(dm[:, cluster], axis=1)
        print(f"  {roi_column(band, metric)}: {len(cluster)} channels ({label})")

    ensure_dir(ROI_PATH.parent)
    results.to_csv(ROI_PATH, index=False)
    record(ROI_PATH, script=__file__, inputs=[COV_PATH, CHAN_PATH, METRICS_LONG, RESULTS_PATH])
    print(f"[SAVED] {ROI_PATH} (N = {len(results)})")


if __name__ == "__main__":
    main()
