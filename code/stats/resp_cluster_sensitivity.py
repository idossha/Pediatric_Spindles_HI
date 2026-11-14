"""The cluster test of each surviving HI cluster's metric (``stats.clusters.surviving``) with other respiratory indices in place of HI.

Six models (metric ~ index + age_c + gender_bin; indices log10(x + 1), centred): HI, AHI (total), oAHI (AHI - CAI),
OAI, HI|OAI (effect HI, both in the model) and OAI|HI. Settings and permutation seed as ``stats.clusters``.

Inputs : COV_PATH, PARTICIPANTS_PATH, CHAN_PATH, METRICS_LONG, results/tables/cluster_permutation_results.csv.
Output : results/tables/resp_cluster_sensitivity.csv, resp_cluster_overlap.csv (overlap with the HI cluster);
         results/figure_data/resp_tmaps_<metric>_<band>.csv (t per channel and model, and the
         model's largest cluster as a 0/1 ``cluster_<model>`` column)
Run    : cd code && python -m stats.resp_cluster_sensitivity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from project import FIGURE_DATA, TABLE_DIR, ensure_dir
from provenance import record
from stats.clusters import (
    RESULTS_PATH,
    adjacency_matrix,
    cluster_permutation_general,
    fmt_channels,
    surviving,
)
from stats.io import (
    RESP_INDICES_OAHI,
    add_obstructive_ahi,
    add_respiratory_predictors,
    data_matrix_for_metric,
    load_channel_info,
    load_covariates,
    matrix_inputs,
    spindle_matrices,
)

NOTE = (
    "Channel-wise OLS t for the first predictor, two-sided p < .05 cluster-forming threshold, cluster mass = sum |t|, "
    "5,000 Freedman-Lane permutations with seed 20260607; AHI is total AHI, oAHI = AHI - CAI; all indices log10(x+1), centred."
)

MODELS = [
    ("HI", ["logHI_c", "age_c", "gender_bin"]),
    ("AHI", ["logAHI_c", "age_c", "gender_bin"]),
    ("oAHI", ["logOAHI_c", "age_c", "gender_bin"]),
    ("OAI", ["logOAI_c", "age_c", "gender_bin"]),
    ("HI|OAI", ["logHI_c", "logOAI_c", "age_c", "gender_bin"]),
    ("OAI|HI", ["logOAI_c", "logHI_c", "age_c", "gender_bin"]),
]


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def main() -> None:
    channels = load_channel_info()
    cov = add_respiratory_predictors(add_obstructive_ahi(load_covariates()), RESP_INDICES_OAHI)
    adjacency = adjacency_matrix(channels)
    slow, fast, sleep_mins = spindle_matrices(cov, len(channels.labels))
    band_data = {"slow": slow, "fast": fast}

    rows, overlap_rows = [], []
    for label, band, metric in surviving():
        data_matrix = data_matrix_for_metric(band_data[band], metric, sleep_mins)
        tmaps = {"channel_1based": np.arange(1, len(channels.labels) + 1), "label": channels.labels}
        ref: set[int] = set()
        ref_t = np.full(len(channels.labels), np.nan)
        for model, predictors in MODELS:
            clusters, tmap, _ = cluster_permutation_general(data_matrix, cov, adjacency, predictors)
            tmaps[f"t_{model}"] = tmap
            top = clusters[0] if clusters else None
            tmaps[f"cluster_{model}"] = np.isin(np.arange(len(tmap)), top["channels"] if top else []).astype(int)
            rows.append(
                {
                    "metric_label": label,
                    "band": band,
                    "metric": metric,
                    "model": model,
                    "effect": predictors[0],
                    "formula": f"{metric} ~ " + " + ".join(predictors),
                    "n_channels": top["n"] if top else 0,
                    "cluster_mass": round(top["mass"], 4) if top else np.nan,
                    "peak_t": round(top["peak_t"], 4) if top else np.nan,
                    "p_value": round(top["p"], 6) if top else np.nan,
                    "survives_p05": bool(top and top["p"] < 0.05),
                    "n_clusters_formed": len(clusters),
                    "channels_1based": fmt_channels(top["channels"]) if top else "",
                }
            )
            chans = set(top["channels"]) if top else set()
            if model == "HI":
                ref, ref_t = chans, tmap
            else:
                both = np.isfinite(tmap) & np.isfinite(ref_t)
                in_ref = sorted(ref)
                overlap_rows.append(
                    {
                        "metric_label": label,
                        "model": model,
                        "n_HI_cluster": len(ref),
                        "n_model_cluster": len(chans),
                        "n_shared": len(ref & chans),
                        "jaccard": round(jaccard(ref, chans), 4),
                        "prop_of_HI_cluster_covered": round(len(ref & chans) / len(ref), 4) if ref else np.nan,
                        "tmap_r_with_HI": round(float(np.corrcoef(tmap[both], ref_t[both])[0, 1]), 4),
                        "mean_t_in_HI_cluster": round(float(np.nanmean(tmap[in_ref])), 4) if in_ref else np.nan,
                        "mean_t_HI_in_HI_cluster": round(float(np.nanmean(ref_t[in_ref])), 4) if in_ref else np.nan,
                    }
                )
            print(f"{label:<22}{model:<8}n={rows[-1]['n_channels']:>3}  mass={rows[-1]['cluster_mass']}  p={rows[-1]['p_value']}")

        ensure_dir(FIGURE_DATA)
        fd = FIGURE_DATA / f"resp_tmaps_{metric.lower()}_{band}.csv"
        pd.DataFrame(tmaps).to_csv(fd, index=False)
        record(fd, script=__file__, inputs=[*matrix_inputs(), RESULTS_PATH], note=NOTE)

    ensure_dir(TABLE_DIR)
    out = TABLE_DIR / "resp_cluster_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    record(out, script=__file__, inputs=[*matrix_inputs(), RESULTS_PATH], note=NOTE)
    out2 = TABLE_DIR / "resp_cluster_overlap.csv"
    pd.DataFrame(overlap_rows).to_csv(out2, index=False)
    record(out2, script=__file__, inputs=[*matrix_inputs(), RESULTS_PATH], note=NOTE)
    print(f"\n[SAVED] {out}\n[SAVED] {out2}")


if __name__ == "__main__":
    main()
