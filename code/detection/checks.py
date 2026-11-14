"""Per-file checks that detections belong to their recording and cover it.

Alignment: YASA's stored ``Amplitude`` (1-30 Hz, detrended peak-to-peak) is re-measured from the .fdt for a fixed
random sample of detections; the same measurement 10 s later is the null. Grid: Start/End on the 100 Hz grid.
Coverage: the last detection ends near the end of the cleaned file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from detection.files import open_fdt
from detection.settings import (
    ALIGN_N,
    ALIGN_PAD,
    ALIGN_SEED,
    DETECT_SF,
    MAX_END_GAP_S,
    MIN_ALIGN_RHO,
    MIN_WITHIN_5PCT,
)

COLUMNS = ["align_n", "align_spearman", "align_spearman_shift10", "align_frac_within_5pct", "on_grid",
           "last_end_s", "end_gap_s"]


def _ptp(mm: np.memmap, sf: float, k0: int, k1: int, ch: int) -> float:
    from mne.filter import filter_data

    lo, hi = max(k0 - ALIGN_PAD, 0), min(k1 + ALIGN_PAD + 1, mm.shape[0])
    seg = np.asarray(mm[lo:hi, ch], dtype=np.float64)
    if len(seg) < 3 * ALIGN_PAD // 2:
        return np.nan
    y = filter_data(seg, sf, 1.0, 30.0, method="fir", verbose=False)[k0 - lo:k1 - lo + 1]
    x = np.arange(len(y), dtype=float)
    return float(np.ptp(y - np.polyval(np.polyfit(x, y, 1), x)))


def run(sid: int, sp: pd.DataFrame, set_path: Path) -> dict:
    from scipy.stats import spearmanr

    mm, sf = open_fdt(set_path)
    n, shift = mm.shape[0], int(10 * sf)
    d = sp[(sp["Start"] * sf > ALIGN_PAD) & (sp["End"] * sf < n - ALIGN_PAD - 2000)]
    rng = np.random.default_rng([ALIGN_SEED, sid])
    q = d["Start"].quantile([0, 0.5, 1.0]).to_numpy()
    rows = []
    for lo, hi in ((q[0], q[1]), (q[1], q[2])):  # half of the sample from each half of the night
        s = d[(d["Start"] >= lo) & (d["Start"] <= hi)]
        for r in s.sample(min(ALIGN_N // 2, len(s)), random_state=int(rng.integers(10**9))).itertuples():
            k0, k1, ch = round(r.Start * sf), round(r.End * sf), int(r.IdxChannel)
            rows.append((r.Amplitude, _ptp(mm, sf, k0, k1, ch), _ptp(mm, sf, k0 + shift, k1 + shift, ch)))
    m = pd.DataFrame(rows, columns=["stored", "here", "shift10"]).dropna()
    grid = np.concatenate([sp["Start"].to_numpy(), sp["End"].to_numpy()]) * DETECT_SF
    last = float(sp["End"].max())
    return {"align_n": len(m), "align_spearman": spearmanr(m["stored"], m["here"]).statistic,
            "align_spearman_shift10": spearmanr(m["stored"], m["shift10"]).statistic,
            "align_frac_within_5pct": float(np.mean(np.abs(m["here"] / m["stored"] - 1) < 0.05)),
            "on_grid": bool(np.all(np.abs(grid - np.round(grid)) < 1e-6)), "last_end_s": last,
            "end_gap_s": n / sf - last}


def passes(row: dict | pd.Series) -> bool:
    return bool(row["align_spearman"] >= MIN_ALIGN_RHO and row["align_frac_within_5pct"] >= MIN_WITHIN_5PCT
                and row["on_grid"] and row["end_gap_s"] <= MAX_END_GAP_S)
