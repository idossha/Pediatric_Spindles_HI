"""Where a child's cleaned recording and detections live, and how to read the recording."""
from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
from project import DATA_ROOT, SPINDLE_DIR


def set_path(sid: int) -> Path:
    """The child's cleaned N2+N3 EEGLAB file (``_clean`` or ``_duplicateclean``)."""
    hits = sorted(p for p in (DATA_ROOT / f"sub-{sid:03d}" / "eeg").glob("*N2N3*clean.set") if not p.name.startswith("._"))
    if not hits:
        raise FileNotFoundError(f"no cleaned N2N3 .set for {sid:03d} under {DATA_ROOT}")
    return hits[0]


def detections_path(sid: int, folder: Path = SPINDLE_DIR) -> Path:
    return folder / f"{sid:03d}_all_spindles_detailed.csv"


def open_fdt(path: Path) -> tuple[np.memmap, float]:
    """Memory-mapped EEG (samples x channels, float32) and its sampling rate."""
    with h5py.File(path, "r") as f:
        nb, n, sf = (np.array(f[k]).ravel()[0] for k in ("nbchan", "pnts", "srate"))
    return np.memmap(path.with_suffix(".fdt"), dtype="<f4", mode="r", shape=(int(n), int(nb))), float(sf)


def sha256(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def describe(sid: int) -> dict:
    """Input record for the manifest: paths, checksums, cleaned duration (the density denominator)."""
    path = set_path(sid)
    mm, sf = open_fdt(path)
    return {"set": str(path), "set_sha256": sha256(path), "fdt": str(path.with_suffix(".fdt")),
            "fdt_sha256": sha256(path.with_suffix(".fdt")), "cleaned_s": mm.shape[0] / sf, "file_sf": sf,
            "n_channels": mm.shape[1]}
