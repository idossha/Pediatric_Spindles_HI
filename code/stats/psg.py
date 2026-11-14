"""Read-only PSG readers for one child: cleaned .set header and timeline, spindle detections, hypnogram, scored events.

Inputs come from ``project.DATA_ROOT``; per-spindle detections from ``project.SPINDLE_DIR``. Used by
``stats.figure_tables`` (fig-1's example night, the graphical abstract's night) and ``run_all.py --dry-run``.

Timing conventions (checked for all 62 recordings):

* The 1-based original-sample index at 200 Hz (seconds from the EDF start = (index - 1) / 200) is rebuilt from
  the EEGLAB boundary events; it equals ``scoringchans[:, 2]`` wherever that is intact (61 of 62; one child's was lost).
* Cleaned sample k (0-based) is at cleaned time k / srate; a detection's Start/End (cleaned s) maps to sample round(x * srate).
* EDF start = ``recording_start_date_num`` (MATLAB datenum) ROUNDED to the second (the stored value sits microseconds
  below it; truncating puts the scored-event clock 1 s late).
* Scored-event clock time: t = (tod - start_tod) mod 86400, minus one day if above recording length + 30 min.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from detection.files import detections_path as spindles_path
from project import DATA_ROOT

EXAMPLES_PATH = DATA_ROOT / "sourcedata/tables/figure_examples.json"  # {figure id: child id}; raw data only
HYP_CODE = {"W": 0, "U": 0, "N1": 1, "N2": 2, "N3": 3, "R": 5, "?": -1}  # anything else -9


@dataclass(frozen=True)
class SubjectFiles:
    set: Path | None
    events: Path | None
    hypnogram: Path | None
    spindles: Path | None


def _glob1(folder: Path, pattern: str) -> Path | None:
    hits = sorted(p for p in folder.glob(pattern) if not p.name.startswith("._")) if folder.is_dir() else []
    return hits[0] if hits else None


def example_child(figure: str) -> int:
    """The child whose recording ``figure`` draws (EXAMPLES_PATH, so no id is typed in the code)."""
    return int(json.loads(EXAMPLES_PATH.read_text())[figure])


def paths(sid: int) -> SubjectFiles:
    """One subject's inputs; a missing file is None (two children have no PSG folder)."""
    pad = f"{int(sid):03d}"
    eeg, psg = DATA_ROOT / f"sub-{pad}" / "eeg", DATA_ROOT / f"sub-{pad}" / "psg"
    hyp, sp = psg / f"{pad}_hypnogram.txt", spindles_path(sid)
    return SubjectFiles(
        set=_glob1(eeg, "*N2N3*clean.set"),
        events=_glob1(psg, f"{pad}_scored*events.txt"),
        hypnogram=hyp if hyp.exists() else None,
        spindles=sp if sp.exists() else None,
    )


# --------------------------------------------------------------------------------------- .set


def _datenum_to_datetime(datenum: float) -> dt.datetime:
    days = datenum - 366.0
    return dt.datetime.fromordinal(int(days)) + dt.timedelta(seconds=round((days - int(days)) * 86400.0))


def _deref(f: h5py.File, ref):
    obj = f[ref]
    a = np.array(obj)
    return "".join(map(chr, a.ravel())) if obj.attrs.get("MATLAB_class") == b"char" else a.ravel()


def read_header(path: Path) -> tuple[dt.datetime, float]:
    """(EDF start rounded to the second, recording length in s)."""
    with h5py.File(path, "r") as f:
        props = f["edf"]["properties"]
        start = _datenum_to_datetime(float(np.array(props["recording_start_date_num"]).ravel()[0]))
        return start, float(np.array(props["recording_length"]).ravel()[0])


@dataclass
class SetInfo:
    srate: float
    n: int                    # cleaned samples
    idx: np.ndarray           # int64 original-sample index (1-based, 200 Hz) per cleaned sample, boundary rebuild
    start: dt.datetime
    rec_len_s: float
    boundary_latency: np.ndarray


def read_set(path: Path) -> SetInfo:
    start, rec_len = read_header(path)
    with h5py.File(path, "r") as f:
        srate = float(np.array(f["srate"]).ravel()[0])
        shape = f["scoringchans"].shape
        n = shape[0] if shape[0] != 3 or shape[1] == 3 else shape[1]
        ev = f["event"]
        m = ev["type"].shape[0]
        types = [_deref(f, ev["type"][i, 0]) for i in range(m)]
        lats = np.array([float(_deref(f, ev["latency"][i, 0])[0]) for i in range(m)])
        durs = []
        for i in range(m):
            d = _deref(f, ev["duration"][i, 0])
            durs.append(float(d[0]) if isinstance(d, np.ndarray) and d.size else np.nan)
    evdf = pd.DataFrame({"type": types, "latency": lats, "duration": durs})
    bnd = evdf[evdf["type"] == "boundary"].sort_values("latency")
    # cleaned sample k (1-based) = original sample k + samples deleted before it
    k = np.arange(1, n + 1, dtype=np.int64)
    lat = bnd["latency"].to_numpy()
    deleted = np.concatenate([[0], np.cumsum(bnd["duration"].to_numpy())]).astype(np.int64)
    idx = k + deleted[np.searchsorted(lat, k, side="left")]
    return SetInfo(srate=srate, n=n, idx=idx, start=start, rec_len_s=rec_len, boundary_latency=lat)


# --------------------------------------------------------------------------------- detections


def read_spindles(path: Path) -> pd.DataFrame:
    """Per-channel detections (columns by name)."""
    keep = ["Start", "End", "Duration", "Amplitude", "Frequency", "IdxChannel"]
    df = pd.read_csv(path, usecols=keep)
    return df.apply(pd.to_numeric, errors="coerce")


# ------------------------------------------------------------------------------------ scoring


def read_hypnogram(path: Path) -> np.ndarray:
    """Per-30-s-epoch stage codes (W/U 0, N1 1, N2 2, N3 3, R 5, '?' -1, other -9)."""
    toks = [ln.strip().upper() for ln in path.read_text(encoding="latin-1").splitlines() if ln.strip()]
    return np.array([HYP_CODE.get(t, -9) for t in toks], dtype=np.int16)


def classify(label: str) -> str:
    lo = label.lower()
    for kind, key in (("hypopnea", "hypopnea"), ("apnea", "apnea"), ("rera", "rera"), ("arousal", "arousal"),
                      ("desat", "desat")):
        if key in lo:
            return kind
    return "other"


_CLOCK = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")
_DUR = re.compile(r"^\s*(\d+):(\d+(?:\.\d+)?)\s*$")


def _num(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return np.nan


def read_events(path: Path, start: dt.datetime, rec_len_s: float) -> pd.DataFrame:
    """Scored-events text: one row per event line, times in s from the EDF start. Fields 6 and 7 of a
    respiratory or desaturation line are the SpO2 nadir and drop (%); '-' elsewhere (NaN here)."""
    start_tod = start.hour * 3600 + start.minute * 60 + start.second
    rows = []
    with open(path, encoding="latin-1") as fh:
        for ln in fh:
            p = [x.strip() for x in ln.rstrip("\r\n").split(",")]
            if len(p) < 5 or not _CLOCK.match(p[0]):
                continue
            h, mi, s = (int(x) for x in p[0].split(":"))
            t = float((h * 3600 + mi * 60 + s - start_tod) % 86400)
            if t > rec_len_s + 1800:
                t -= 86400.0
            dm = _DUR.match(p[4])
            rows.append({"epoch_file": int(p[1]) if p[1].isdigit() else -1, "stage_file": p[2], "label": p[3],
                         "dur_s": int(dm.group(1)) * 60 + float(dm.group(2)) if dm else np.nan, "t_start": t,
                         "spo2_nadir": _num(p[5]) if len(p) > 5 else np.nan, "spo2_drop": _num(p[6]) if len(p) > 6 else np.nan})
    df = pd.DataFrame(rows, columns=["epoch_file", "stage_file", "label", "dur_s", "t_start", "spo2_nadir", "spo2_drop"])
    df["t_end"] = df["t_start"] + df["dur_s"].fillna(0.0)
    df["kind"] = df["label"].map(classify)
    df["epoch_computed"] = (np.floor(df["t_start"] / 30.0) + 1).astype(int)
    return df
