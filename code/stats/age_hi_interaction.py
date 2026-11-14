"""Channel-wise Age x HI interaction with cluster correction (fig-s2).

Per channel (>= 40 valid children): metric ~ age_c + logHI_c + age_c:logHI_c + sex; clusters of channels at two-sided p < .05 for the
interaction over MNE Delaunay adjacency (mass = sum |t|). Null: Freedman-Lane (reduced model without the interaction),
5,000 permutations, seed 20260512 per metric; p = (#{null max mass >= observed} + 1) / 5,001.

Inputs : COV_PATH, CHAN_PATH, METRICS_LONG.
Output : results/tables/age_hi_interaction_clusters.csv; results/figure_data/age_hi_interaction_tmap.csv (long:
         one t-map per FIGURE_METRICS entry, keyed by ``metric`` = "<band>_<metric>", with its largest cluster) and
         age_hi_interaction_roi.csv (one column of child means over each largest cluster), read by
         ``stats.figure_tables interaction`` and fig-s2.
Run    : cd code && python -m stats.age_hi_interaction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from project import FIGURE_DATA, TABLE_DIR, ensure_dir
from provenance import record
from scipy import stats as sps
from stats.clusters import adjacency_matrix, cluster_components, critical_t
from stats.interaction import prepare_covariates
from stats.io import (
    data_matrix_for_metric,
    load_channel_info,
    matrix_inputs,
    spindle_matrices,
)

N_PERM = 5000
SEED = 20260512
MIN_N = 40
FULL = ["age_c", "logHI_c", "ageXHI", "sex"]  # effect of interest: ageXHI, design column 3
REDUCED = ["age_c", "logHI_c", "sex"]
METRICS = [
    ("slow", "Density", "Slow density"),
    ("slow", "Count", "Slow count"),
    ("slow", "Duration", "Slow duration"),
    ("slow", "Amplitude", "Slow amplitude"),
    ("fast", "Density", "Fast density"),
    ("fast", "Count", "Fast count"),
    ("fast", "Duration", "Fast duration"),
    ("fast", "Amplitude", "Fast amplitude"),
    ("slow", "Frequency", "Slow frequency"),
    ("fast", "Frequency", "Fast frequency"),
]
FIGURE_METRICS = [("slow", "Amplitude"), ("fast", "Frequency")]  # the two corrected maps
CLUSTER_TABLE = TABLE_DIR / "age_hi_interaction_clusters.csv"
TMAP = FIGURE_DATA / "age_hi_interaction_tmap.csv"
ROI_TABLE = FIGURE_DATA / "age_hi_interaction_roi.csv"


def interaction_maps(y_mat: np.ndarray, cov: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Observed interaction t and p per channel, the Freedman-Lane null t-maps (n_ch x N_PERM) and each channel's
    critical t."""
    rng = np.random.default_rng(SEED)
    perms = np.array([rng.permutation(len(cov)) for _ in range(N_PERM)])
    xf_all = np.column_stack([np.ones(len(cov)), cov[FULL].to_numpy(dtype=float)])
    xr_all = np.column_stack([np.ones(len(cov)), cov[REDUCED].to_numpy(dtype=float)])
    n_ch = y_mat.shape[1]
    obs_t, obs_p = np.full(n_ch, np.nan), np.full(n_ch, np.nan)
    null_t = np.full((n_ch, N_PERM), np.nan)
    crit = np.full(n_ch, np.nan)
    for ch in range(n_ch):
        y = y_mat[:, ch].astype(float)
        valid = ~np.isnan(y) & np.isfinite(xf_all).all(axis=1)
        nv = int(valid.sum())
        if nv < MIN_N:
            continue
        yv, xf, xr = y[valid], xf_all[valid], xr_all[valid]
        a = np.linalg.inv(xf.T @ xf)
        dof = nv - xf.shape[1]
        crit[ch] = critical_t(dof)

        def t_stat(ys: np.ndarray, a=a, xf=xf, dof=dof) -> np.ndarray:  # ys: (..., nv)
            b = ys @ (a @ xf.T).T
            rss = np.einsum("...i,...i->...", ys, ys) - np.einsum("...j,...j->...", b, ys @ xf)
            return b[..., 3] / np.sqrt(rss / dof * a[3, 3])

        obs_t[ch] = t_stat(yv)
        obs_p[ch] = 2 * sps.t.sf(abs(obs_t[ch]), dof)
        fitted = xr @ np.linalg.lstsq(xr, yv, rcond=None)[0]
        resid = yv - fitted
        order = np.argsort(np.argsort(perms[:, valid], axis=1), axis=1)
        null_t[ch] = t_stat(fitted[None, :] + resid[order])
    return obs_t, obs_p, null_t, crit


def clusters_with_p(obs_t: np.ndarray, null_t: np.ndarray, crit: np.ndarray, adjacency: np.ndarray) -> list[dict]:
    def masses(t):
        return [float(np.nansum(np.abs(t[c]))) for c in cluster_components(np.isfinite(t) & (np.abs(t) > crit), adjacency)]

    max_null = np.array([max(masses(null_t[:, p]) or [0.0]) for p in range(N_PERM)])
    out = []
    for comp in cluster_components(np.isfinite(obs_t) & (np.abs(obs_t) > crit), adjacency):
        mass = float(np.nansum(np.abs(obs_t[comp])))
        out.append({"channels": comp, "n": len(comp), "mass": mass, "mean_t": float(np.nanmean(obs_t[comp])),
                    "p": float((np.sum(max_null >= mass) + 1) / (N_PERM + 1))})
    return sorted(out, key=lambda c: c["mass"], reverse=True)


def main() -> None:
    channels = load_channel_info()
    adjacency = adjacency_matrix(channels)
    cov = prepare_covariates()
    slow, fast, sleep_mins = spindle_matrices(cov, len(channels.labels))
    bands = {"slow": slow, "fast": fast}

    rows, tmaps = [], []
    roi = pd.DataFrame({"id": cov["id"]})
    for band, metric, label in METRICS:
        y_mat = data_matrix_for_metric(bands[band], metric, sleep_mins)
        obs_t, obs_p, null_t, crit = interaction_maps(y_mat, cov)
        clusters = clusters_with_p(obs_t, null_t, crit, adjacency)
        best = clusters[0] if clusters else None
        rows.append({
            "band": band, "metric": metric, "label": label,
            "n_channels_p05": int(np.sum(np.isfinite(obs_p) & (obs_p < 0.05))),
            "best_cluster_n": best["n"] if best else 0,
            "best_cluster_mass": round(best["mass"], 3) if best else np.nan,
            "best_cluster_mean_t": round(best["mean_t"], 3) if best else np.nan,
            "best_cluster_p": best["p"] if best else np.nan,
            "cluster_corrected": any(c["p"] < 0.05 for c in clusters),
        })
        print(rows[-1])
        if (band, metric) in FIGURE_METRICS:
            key = f"{band}_{metric.lower()}"
            members = best["channels"] if best else []
            tmaps.append(pd.DataFrame({"metric": key, "channel": np.arange(1, len(obs_t) + 1), "label": channels.labels,
                                       "t": obs_t, "in_largest_cluster": np.isin(np.arange(len(obs_t)), members)}))
            roi[f"{key}_cluster_mean"] = np.nanmean(y_mat[:, members], axis=1)

    ensure_dir(TABLE_DIR)
    ensure_dir(FIGURE_DATA)
    pd.DataFrame(rows).to_csv(CLUSTER_TABLE, index=False)
    pd.concat(tmaps).to_csv(TMAP, index=False)
    roi.to_csv(ROI_TABLE, index=False)
    for path in (CLUSTER_TABLE, TMAP, ROI_TABLE):
        record(path, script=__file__, inputs=matrix_inputs())
    print(f"[SAVED] {CLUSTER_TABLE}\n[SAVED] {TMAP}\n[SAVED] {ROI_TABLE}")


if __name__ == "__main__":
    main()
