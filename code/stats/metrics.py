"""Channel x participant spindle metrics from the detections (pipeline step 0).

Per child of the sample (``project.sample_ids``), channel of the 172-channel montage and band (``all`` detections;
``slow`` 10 to < 12 Hz; ``fast`` 12 to 16 Hz, by spindle frequency): count, density = count / minutes of
the child's cleaned N2+N3 file (samples / srate, ``cleaned_s`` in the detections' manifest), and the
mean amplitude, duration and mean spindle frequency of the channel's detections. As Methods state, a channel
without a detection has count and density 0 and no amplitude, duration or frequency. ``clean_min``
repeats the denominator on every row. Every analysis reads its spindle matrices from this table
(``stats.io.load_spindle_matrices``, ``load_metric_matrix``).

Inputs : SPINDLE_DIR (the per-subject detections and manifest.csv), PARTICIPANTS_PATH, CHAN_PATH.
Output : results/tables/spindle_metrics_long.csv (METRICS_LONG, + provenance sidecar).
Run    : python -m stats.metrics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from project import (
    CHAN_PATH,
    METRICS_LONG,
    PARTICIPANTS_PATH,
    SPINDLE_DIR,
    ensure_dir,
    sample_ids,
)
from provenance import record

from .io import FAST_RANGE, SLOW_RANGE, load_channel_info

BANDS = {
    "all": lambda f: np.ones(len(f), bool),
    "slow": lambda f: (f >= SLOW_RANGE[0]) & (f < SLOW_RANGE[1]),
    "fast": lambda f: (f >= FAST_RANGE[0]) & (f <= FAST_RANGE[1]),
}


def child_metrics(sid: int, clean_min: float, n_channels: int) -> pd.DataFrame:
    sp = pd.read_csv(SPINDLE_DIR / f"{sid:03d}_all_spindles_detailed.csv",
                     usecols=["IdxChannel", "Amplitude", "Duration", "Frequency"])
    out = []
    for band, keep in BANDS.items():
        g = sp[keep(sp["Frequency"].to_numpy())].groupby("IdxChannel")
        m = g[["Amplitude", "Duration", "Frequency"]].mean().reindex(range(n_channels))
        count = g.size().reindex(range(n_channels), fill_value=0)
        out.append(pd.DataFrame({
            "id": sid, "channel": np.arange(1, n_channels + 1), "band": band, "count": count.to_numpy(),
            "density": count.to_numpy() / clean_min, "amplitude": m["Amplitude"].to_numpy(),
            "duration": m["Duration"].to_numpy(), "frequency": m["Frequency"].to_numpy(), "clean_min": clean_min,
        }))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ids = sample_ids()
    manifest = pd.read_csv(SPINDLE_DIR / "manifest.csv").set_index("id")
    missing = sorted(set(ids) - set(manifest.index))
    if missing:
        raise SystemExit(f"no detections for {missing} in {SPINDLE_DIR}")
    n_channels = len(load_channel_info(CHAN_PATH).labels)
    table = pd.concat([child_metrics(sid, manifest.loc[sid, "cleaned_s"] / 60.0, n_channels) for sid in ids],
                      ignore_index=True)
    ensure_dir(METRICS_LONG.parent)
    table.to_csv(METRICS_LONG, index=False)
    record(METRICS_LONG, script=__file__, inputs=[PARTICIPANTS_PATH, CHAN_PATH, SPINDLE_DIR / "manifest.csv",
                                                  SPINDLE_DIR / "*_all_spindles_detailed.csv"],
           note="Density = count / (cleaned_s / 60) of the detections' manifest; zero-filled; morphology missing "
                "where count = 0. Bands: slow [10, 12), fast [12, 16] Hz spindle frequency; all = every detection.")
    print(f"[SAVED] {METRICS_LONG}: {len(ids)} children x {n_channels} channels x {len(BANDS)} bands")


if __name__ == "__main__":
    main()
