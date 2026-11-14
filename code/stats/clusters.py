"""Cluster-based permutation test of the covariate-adjusted logHI_c effect (pipeline step 2).

Model per channel: metric ~ logHI_c + age_c + gender_bin (OLS t for logHI_c); clusters are connected
channels (MNE Delaunay adjacency) with two-sided p < .05 (|t| above the critical t at each channel's residual
degrees of freedom), mass = sum of |t|. Null: Freedman-Lane residual
permutation (one shared subject relabeling across channels), max cluster mass per permutation;
p = (#{null_max >= observed} + 1) / (N + 1).

Inputs : COV_PATH, CHAN_PATH, METRICS_LONG.
Output : results/tables/cluster_permutation_results.csv (the largest cluster per metric) and
         results/tables/cluster_tmaps.csv (per channel: t and p of logHI_c, and whether the channel is in that
         cluster; the maps of the HI figure), each with a provenance sidecar.
Run    : python -m stats.clusters          (5,000 permutations x 8 metrics; slow)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from provenance import record
from scipy import stats as sps

from .io import (
    CHAN_PATH,
    COV_PATH,
    METRICS_LONG,
    PREDICTORS,
    TABLE_DIR,
    ChannelInfo,
    data_matrix_for_metric,
    ensure_dir,
    load_channel_info,
    load_covariates,
    load_spindle_matrices,
)
from .regression import channel_regression

ALPHA = 0.05  # cluster-forming threshold: two-sided p < ALPHA per channel


def critical_t(dof):
    """|t| a channel must exceed to join a cluster: the two-sided ``ALPHA`` critical t at ``dof``."""
    return sps.t.ppf(1 - ALPHA / 2, dof)
N_PERM = 5000
SEED = 20260607
MIN_N = 30


def cluster_components(supra: np.ndarray, adjacency: np.ndarray) -> list[list[int]]:
    """Connected components of suprathreshold channels over the adjacency graph."""
    nodes = set(np.where(supra)[0].tolist())
    visited: set[int] = set()
    out: list[list[int]] = []
    for s in sorted(nodes):
        if s in visited:
            continue
        stack = [s]
        visited.add(s)
        comp: list[int] = []
        while stack:
            n = stack.pop()
            comp.append(n)
            for nb in np.where(adjacency[n])[0]:
                nb = int(nb)
                if nb in nodes and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        out.append(sorted(comp))
    return out


def largest_cluster_channels(
    data_matrix: np.ndarray,
    cov_df: pd.DataFrame,
    adjacency: np.ndarray,
    predictors: list[str] | None = None,
) -> list[int]:
    """Channels of the top observed logHI_c cluster (no permutation); same rule as ``cluster_permutation_test``."""
    predictors = predictors or PREDICTORS
    _, t, p = channel_regression(data_matrix, cov_df, predictors)
    supra = np.isfinite(t) & (p < ALPHA)
    comps = cluster_components(supra, adjacency)
    if not comps:
        return []
    comps.sort(key=lambda c: float(np.nansum(np.abs(t[c]))), reverse=True)
    return comps[0]


def channel_t_maps(
    data_matrix: np.ndarray,
    cov: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
    min_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed t-map (nCh,), Freedman-Lane null t-maps (nCh x n_perm) and each channel's critical t;
    ``cov`` columns are [logHI_c, age_c, gender_bin]."""
    n_sub, n_ch = data_matrix.shape
    obs_t = np.full(n_ch, np.nan)
    null_t = np.full((n_ch, n_perm), np.nan)
    crit = np.full(n_ch, np.nan)
    base_ok = ~np.isnan(cov).any(axis=1)

    # One subject relabeling per permutation shared by all channels (else null t-maps are spatially white).
    global_orders = np.argsort(rng.random((n_perm, n_sub)), axis=1)
    ranks = np.empty((n_perm, n_sub), dtype=np.int64)
    np.put_along_axis(
        ranks, global_orders, np.broadcast_to(np.arange(n_sub), (n_perm, n_sub)), axis=1
    )

    for ch in range(n_ch):
        y = data_matrix[:, ch]
        valid = base_ok & ~np.isnan(y)
        nv = int(valid.sum())
        if nv < min_n:
            continue
        yv = y[valid]
        Xf = np.column_stack([np.ones(nv), cov[valid, 0], cov[valid, 1], cov[valid, 2]])
        # Reduced design (Freedman-Lane) drops logHI_c
        Xr = np.column_stack([np.ones(nv), cov[valid, 1], cov[valid, 2]])
        XtX = Xf.T @ Xf
        try:
            A = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            continue
        P = A @ Xf.T               # (4 x nv) pseudo-inverse
        dof = nv - 4
        if dof <= 0:
            continue
        crit[ch] = critical_t(dof)
        A11 = A[1, 1]

        beta = P @ yv
        rss = float(yv @ yv - beta @ (Xf.T @ yv))
        se1 = np.sqrt(rss / dof * A11)
        obs_t[ch] = beta[1] / se1 if se1 > 0 else np.nan

        Ar = np.linalg.inv(Xr.T @ Xr)
        beta_r = Ar @ (Xr.T @ yv)
        fitted_r = Xr @ beta_r
        resid_r = yv - fitted_r

        # Permute reduced residuals with the shared relabeling restricted to this channel's valid subjects.
        order = np.argsort(ranks[:, valid], axis=1)        # (n_perm x nv)
        Yp = fitted_r[None, :] + resid_r[order]            # (n_perm x nv)
        beta1 = Yp @ P[1]                                  # (n_perm,)
        XtY = Yp @ Xf                                       # (n_perm x 4)
        betas = XtY @ A.T                                   # (n_perm x 4)
        yty = np.einsum("pi,pi->p", Yp, Yp)
        rss_p = yty - np.einsum("pj,pj->p", betas, XtY)
        se1_p = np.sqrt(np.maximum(rss_p, 1e-12) / dof * A11)
        null_t[ch] = beta1 / se1_p

    return obs_t, null_t, crit


def clusters_with_p(obs_t: np.ndarray, null_t: np.ndarray, thresh, adjacency: np.ndarray) -> list[dict]:
    """Suprathreshold clusters of ``obs_t``, mass descending, each with its max-mass permutation p over ``null_t``'s columns."""
    clusters: list[dict] = []
    for comp in cluster_components(np.isfinite(obs_t) & (np.abs(obs_t) > thresh), adjacency):
        mass = float(np.nansum(np.abs(obs_t[comp])))
        peak_t = float(obs_t[comp][np.nanargmax(np.abs(obs_t[comp]))])
        clusters.append({"channels": comp, "n": len(comp), "mass": mass, "peak_t": peak_t})

    n_perm = null_t.shape[1]
    max_masses = np.zeros(n_perm)
    for p in range(n_perm):
        tp = null_t[:, p]
        cc = cluster_components(np.isfinite(tp) & (np.abs(tp) > thresh), adjacency)
        max_masses[p] = max([float(np.nansum(np.abs(tp[c]))) for c in cc] or [0.0])

    for cl in clusters:
        cl["p"] = float((np.sum(max_masses >= cl["mass"]) + 1) / (n_perm + 1))
    clusters.sort(key=lambda d: d["mass"], reverse=True)
    return clusters


def cluster_permutation_test(
    data_matrix: np.ndarray,
    cov_df: pd.DataFrame,
    adjacency: np.ndarray,
    predictors: list[str] | None = None,
    t_thresh: float | None = None,
    n_perm: int = N_PERM,
    seed: int = SEED,
    min_n: int = MIN_N,
) -> list[dict]:
    """Observed clusters, mass descending, as dicts with ``channels``, ``n``, ``mass``, ``peak_t``, ``p``.

    ``data_matrix`` is nSub x nCh (``data_matrix_for_metric``); the first predictor is the effect of interest.
    ``t_thresh`` replaces the per-channel critical t with one fixed |t| (the threshold sensitivity only).
    """
    predictors = predictors or PREDICTORS
    cov = cov_df[predictors].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    obs_t, null_t, crit = channel_t_maps(data_matrix, cov, rng, n_perm, min_n)
    thresh = crit if t_thresh is None else t_thresh

    return clusters_with_p(obs_t, null_t, thresh, adjacency)


def _channel_t_maps_general(
    data_matrix: np.ndarray,
    cov: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
    min_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``channel_t_maps`` for K predictors: design ``[const, cov[:, 0..K-1]]``, effect = column 1."""
    n_sub, n_ch = data_matrix.shape
    k = cov.shape[1]
    obs_t = np.full(n_ch, np.nan)
    null_t = np.full((n_ch, n_perm), np.nan)
    crit = np.full(n_ch, np.nan)
    base_ok = ~np.isnan(cov).any(axis=1)

    global_orders = np.argsort(rng.random((n_perm, n_sub)), axis=1)
    ranks = np.empty((n_perm, n_sub), dtype=np.int64)
    np.put_along_axis(ranks, global_orders, np.broadcast_to(np.arange(n_sub), (n_perm, n_sub)), axis=1)

    for ch in range(n_ch):
        y = data_matrix[:, ch]
        valid = base_ok & ~np.isnan(y)
        nv = int(valid.sum())
        if nv < min_n:
            continue
        yv = y[valid]
        Xf = np.column_stack([np.ones(nv), cov[valid, :]])
        Xr = np.column_stack([np.ones(nv), cov[valid, 1:]])  # Freedman-Lane reduced design
        dof = nv - (k + 1)
        if dof <= 0:
            continue
        crit[ch] = critical_t(dof)
        try:
            A = np.linalg.inv(Xf.T @ Xf)
        except np.linalg.LinAlgError:
            continue
        P = A @ Xf.T
        A11 = A[1, 1]

        beta = P @ yv
        rss = float(yv @ yv - beta @ (Xf.T @ yv))
        se1 = np.sqrt(rss / dof * A11)
        obs_t[ch] = beta[1] / se1 if se1 > 0 else np.nan

        Ar = np.linalg.inv(Xr.T @ Xr)
        fitted_r = Xr @ (Ar @ (Xr.T @ yv))
        resid_r = yv - fitted_r

        order = np.argsort(ranks[:, valid], axis=1)
        Yp = fitted_r[None, :] + resid_r[order]
        beta1 = Yp @ P[1]
        XtY = Yp @ Xf
        betas = XtY @ A.T
        yty = np.einsum("pi,pi->p", Yp, Yp)
        rss_p = yty - np.einsum("pj,pj->p", betas, XtY)
        se1_p = np.sqrt(np.maximum(rss_p, 1e-12) / dof * A11)
        null_t[ch] = beta1 / se1_p

    return obs_t, null_t, crit


def cluster_permutation_general(
    data_matrix: np.ndarray,
    cov_df: pd.DataFrame,
    adjacency: np.ndarray,
    predictors: list[str],
    n_perm: int = N_PERM,
    seed: int = SEED,
    min_n: int = MIN_N,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """``cluster_permutation_test`` for K predictors (``predictors[0]`` = effect); returns (clusters,
    observed t-map, each channel's critical t)."""
    cov = cov_df[predictors].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    obs_t, null_t, crit = _channel_t_maps_general(data_matrix, cov, rng, n_perm, min_n)

    return clusters_with_p(obs_t, null_t, crit, adjacency), obs_t, crit


def adjacency_matrix(channels: ChannelInfo) -> np.ndarray:
    """MNE Delaunay channel adjacency of the montage (bool matrix)."""
    import mne

    mne.set_log_level("ERROR")
    return mne.channels.find_ch_adjacency(channels.info, ch_type="eeg")[0].toarray().astype(bool)


def fmt_channels(channels) -> str:
    """0-based channel indices as 1-based numbers, semicolon separated."""
    return ";".join(str(int(c) + 1) for c in sorted(channels))


RESULTS_PATH = TABLE_DIR / "cluster_permutation_results.csv"
TMAPS_PATH = TABLE_DIR / "cluster_tmaps.csv"


def surviving() -> list[tuple[str, str, str]]:
    """(metric_label, band, metric) of the HI tests in RESULTS_PATH whose largest cluster survives correction
    (p < .05); by the Methods rule these clusters are the ROIs."""
    import pandas as pd

    t = pd.read_csv(RESULTS_PATH)
    t = t[t["p_value"] < 0.05]
    return list(zip(t["metric_label"], t["band"], t["metric"]))


def roi_column(band: str, metric: str) -> str:
    """The ROI's name in roi_values.csv and the figure data, e.g. ``fast_duration_cluster``."""
    return f"{band}_{metric.lower()}_cluster"


# (metric_label, band, metric)
METRICS = [
    ("Fast duration", "fast", "Duration"),       # HEADLINE
    ("Slow frequency", "slow", "Frequency"),
    ("Slow amplitude", "slow", "Amplitude"),
    ("Slow duration", "slow", "Duration"),
    ("Slow density", "slow", "Density"),
    ("Fast density", "fast", "Density"),
    ("Fast amplitude", "fast", "Amplitude"),
    ("Fast frequency", "fast", "Frequency"),
]


def main() -> None:
    import argparse

    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    channels = load_channel_info()
    n_ch = len(channels.labels)
    adjacency = adjacency_matrix(channels)

    cov = load_covariates()
    matrices = load_spindle_matrices(cov, n_ch, include_frequency=True)
    sleep_mins = matrices.sleep_mins
    band_data = {"slow": matrices.slow, "fast": matrices.fast}

    rows, maps = [], []
    print("\n" + "=" * 86)
    print(f"{'Metric':<22}{'band':<6}{'metric':<11}{'n_ch':>5}{'mass':>11}"
          f"{'peak_t':>9}{'p':>10}")
    print("=" * 86)
    for metric_label, band, metric in METRICS:
        data_matrix = data_matrix_for_metric(band_data[band], metric, sleep_mins)
        clusters = cluster_permutation_test(data_matrix, cov, adjacency)
        top = clusters[0] if clusters else None
        _, t_map, p_map = channel_regression(data_matrix, cov)
        in_cluster = np.zeros(n_ch, dtype=int)
        if top is not None:
            in_cluster[np.asarray(top["channels"], dtype=int)] = 1
        maps.append(pd.DataFrame({"band": band, "metric": metric, "channel": np.arange(1, n_ch + 1),
                                  "t": t_map, "p": p_map, "in_cluster": in_cluster}))
        if top is None:
            rows.append(
                {
                    "metric_label": metric_label,
                    "band": band,
                    "metric": metric,
                    "n_channels": 0,
                    "cluster_mass": np.nan,
                    "peak_t": np.nan,
                    "p_value": np.nan,
                }
            )
            print(f"{metric_label:<22}{band:<6}{metric:<11}{'-':>5}"
                  f"{'no suprathreshold cluster':>40}")
            continue
        star = "*" if top["p"] < 0.05 else " "
        rows.append(
            {
                "metric_label": metric_label,
                "band": band,
                "metric": metric,
                "n_channels": top["n"],
                "cluster_mass": round(top["mass"], 4),
                "peak_t": round(top["peak_t"], 4),
                "p_value": round(top["p"], 6),
            }
        )
        print(f"{metric_label:<22}{band:<6}{metric:<11}{top['n']:>5}"
              f"{top['mass']:>11.2f}{top['peak_t']:>9.2f}"
              f"{top['p']:>9.4f}{star}")
    print("=" * 86)

    ensure_dir(TABLE_DIR)
    out_path = RESULTS_PATH
    pd.DataFrame(
        rows,
        columns=[
            "metric_label",
            "band",
            "metric",
            "n_channels",
            "cluster_mass",
            "peak_t",
            "p_value",
        ],
    ).to_csv(out_path, index=False)
    record(out_path, script=__file__, inputs=[COV_PATH, CHAN_PATH, METRICS_LONG])
    pd.concat(maps, ignore_index=True).to_csv(TMAPS_PATH, index=False, float_format="%.10g")
    record(TMAPS_PATH, script=__file__, inputs=[COV_PATH, CHAN_PATH, METRICS_LONG],
           note="in_cluster marks the largest suprathreshold cluster of each map, whatever its corrected p.")
    print(f"\n[SAVED] {out_path}, {TMAPS_PATH}")


if __name__ == "__main__":
    main()
