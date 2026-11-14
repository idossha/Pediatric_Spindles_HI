"""Run the detector on one child: load, resample to 100 Hz, YASA on channel blocks in parallel."""
from __future__ import annotations

import gc
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from detection.files import set_path
from detection.settings import BLOCK, DETECT_SF, YASA_KWARGS


def _block(npy: str, first: int, sf: float, names: list[str]) -> pd.DataFrame:
    import yasa

    data = np.array(np.load(npy, mmap_mode="r")[first:first + len(names)])
    sp = yasa.spindles_detect(data, sf=sf, ch_names=names, **YASA_KWARGS)
    return pd.DataFrame() if sp is None else sp.summary().assign(IdxChannel=lambda d: d["IdxChannel"] + first)


def detect(sid: int, workers: int, tmp: Path) -> pd.DataFrame:
    """All detections of one child, in channel order (identical to a single YASA call)."""
    import mne

    raw = mne.io.read_raw_eeglab(str(set_path(sid)), preload=True, verbose=False)
    raw.resample(DETECT_SF, n_jobs=workers, verbose=False)
    data, sf, names = raw.get_data(units="uV"), raw.info["sfreq"], raw.ch_names
    del raw
    npy = tmp / f"detect_{sid:03d}_{os.getpid()}.npy"
    np.save(npy, data)
    del data
    gc.collect()
    try:
        with ProcessPoolExecutor(workers) as pool:
            jobs = [pool.submit(_block, str(npy), b, sf, names[b:b + BLOCK]) for b in range(0, len(names), BLOCK)]
            return pd.concat([j.result() for j in jobs], ignore_index=True)
    finally:
        npy.unlink(missing_ok=True)
