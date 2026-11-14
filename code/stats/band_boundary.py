"""Slow/fast boundary sensitivity of the spindle-duration HI cluster tests.

Mean spindle duration per child x channel is recomputed from the detections (as ``stats.metrics.child_metrics``)
for other frequency bands, and each map gets the HI cluster test of ``stats.clusters.cluster_permutation_test``
(default predictors, N_PERM, SEED). Fast: [b, 16] Hz for each boundary b in BOUNDARIES; slow mirror: [10, b);
also fast [12, 12.5) only, every spindle [10, 16], and a per-child shifted boundary b_i = 12 + (child i's median
spindle frequency - the sample median of those medians), fast [b_i, 16] and slow [10, b_i). Fast [12.5, 16] only
is the b = 12.5 row. The 12-Hz rows equal the primary fast and slow duration maps (asserted).

n_overlap_canonical: channels shared with the primary cluster of the same band's duration map (cluster_tmaps.csv
in_cluster; fast for the fast and 10-16 rows). median_spindles: median spindle count per child-channel in the band.

Inputs : COV_PATH, CHAN_PATH, METRICS_LONG, SPINDLE_DIR detections, results/tables/cluster_tmaps.csv.
Output : results/tables/band_boundary_sensitivity.csv
Run    : cd code && uv run python -m stats.band_boundary
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from project import SPINDLE_DIR, TABLE_DIR, ensure_dir
from provenance import record
from stats.clusters import TMAPS_PATH, adjacency_matrix, cluster_permutation_test
from stats.io import load_channel_info, load_covariates, matrix_inputs, spindle_matrices

BOUNDARIES = (11.5, 11.75, 12.0, 12.25, 12.5, 13.0)
OUT = TABLE_DIR / "band_boundary_sensitivity.csv"


def band_maps(det: list[pd.DataFrame], lo: np.ndarray, hi: np.ndarray, hi_closed: bool,
              n_ch: int) -> tuple[np.ndarray, np.ndarray]:
    """(mean duration, count) children x channels of the detections with lo_i <= frequency < hi_i (<= if hi_closed)."""
    dur, count = np.full((len(det), n_ch), np.nan), np.zeros((len(det), n_ch))
    for i, sp in enumerate(det):
        f = sp["Frequency"]
        g = sp[(f >= lo[i]) & ((f <= hi[i]) if hi_closed else (f < hi[i]))].groupby("IdxChannel")["Duration"]
        dur[i] = g.mean().reindex(range(n_ch)).to_numpy(float)
        count[i] = g.size().reindex(range(n_ch), fill_value=0).to_numpy(float)
    return dur, count


def main() -> None:
    channels = load_channel_info()
    n_ch = len(channels.labels)
    cov = load_covariates()
    adjacency = adjacency_matrix(channels)
    slow, fast, _ = spindle_matrices(cov, n_ch)
    det = [pd.read_csv(SPINDLE_DIR / f"{sid:03d}_all_spindles_detailed.csv", usecols=["IdxChannel", "Duration", "Frequency"])
           for sid in cov["id"]]
    tm = pd.read_csv(TMAPS_PATH)
    canonical = {b: set(tm.loc[(tm["band"] == b) & (tm["metric"] == "Duration") & (tm["in_cluster"] == 1), "channel"] - 1)
                 for b in ("slow", "fast")}

    n = len(det)
    med = np.array([np.median(sp["Frequency"]) for sp in det])
    shifted = 12.0 + med - np.median(med)
    const = partial(np.full, n)
    cases = [*((f"fast >= {b:g}", "fast", const(b), const(16.0), True) for b in BOUNDARIES),
             ("fast 12-12.5 only", "fast", const(12.0), const(12.5), False),
             ("fast >= per-child shift", "fast", shifted, const(16.0), True),
             ("all 10-16", "fast", const(10.0), const(16.0), True),
             *((f"slow < {b:g}", "slow", const(10.0), const(b), False) for b in BOUNDARIES),
             ("slow < per-child shift", "slow", const(10.0), shifted, False)]

    rows = []
    for definition, band, lo, hi, hi_closed in cases:
        dur, count = band_maps(det, lo, hi, hi_closed, n_ch)
        if definition in ("fast >= 12", "slow < 12"):
            primary = (fast if band == "fast" else slow)["Duration"]
            assert np.allclose(dur, primary, equal_nan=True, atol=1e-9), f"{definition} differs from METRICS_LONG"
        top = (cluster_permutation_test(dur, cov, adjacency) or [None])[0]
        rows.append({"definition": definition, "band": band,
                     "lower_hz": lo[0] if np.ptp(lo) == 0 else np.nan, "upper_hz": hi[0] if np.ptp(hi) == 0 else np.nan,
                     "n_channels": top["n"] if top else 0, "peak_t": top["peak_t"] if top else np.nan,
                     "p_cluster": top["p"] if top else np.nan,
                     "n_overlap_canonical": len(set(top["channels"]) & canonical[band]) if top else 0,
                     "median_spindles": float(np.median(count))})
        print(f"{definition:<26} n={rows[-1]['n_channels']:>3}  p={rows[-1]['p_cluster']:.6f}  "
              f"overlap={rows[-1]['n_overlap_canonical']}", flush=True)

    ensure_dir(TABLE_DIR)
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.6g")
    record(OUT, script=__file__, inputs=[*matrix_inputs(), SPINDLE_DIR / "*_all_spindles_detailed.csv", TMAPS_PATH],
           note=f"Per-child shifted boundary: median {np.median(shifted):.2f} Hz, range {shifted.min():.2f}-"
                f"{shifted.max():.2f} Hz (lower_hz/upper_hz NaN on those rows).")


if __name__ == "__main__":
    main()
