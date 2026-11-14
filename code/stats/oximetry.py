"""The raw pulse-oximeter signal of one night (library): readers, validity, desaturation detection, hypopnea classes.

Used by ``stats.hypopnea_type`` (the signal-based hypopnea classes, primary) and ``stats.oximetry_qc`` (quality,
agreement, nadir).

Signal: the SpO2 (%, 10 Hz), Pulse and Pleth (mV, 50 Hz) channels of each child's full-night EDF, extracted unfiltered
to ``DATA_ROOT/sub-XXX/psg/sub-XXX_task-sleep_recording-{spo2,pleth}_physio.tsv.gz`` by
``DATA_ROOT/sourcedata/scripts/extract_oximetry.py`` (60 of the 62 children; two have no EDF and no scored
events). Time 0 is the EDF start; ``load`` asserts that the sidecar's EDF start clock time equals the cleaned .set's
(``psg.read_header``), the clock every scored event and hypnogram epoch is placed on, and that the record count
equals the .set's recording length.

Validity (per 1-s second of the recording; a second is invalid if any of its ten SpO2 samples is flagged, and
takes the first reason in this order for the stacked minutes):
  out_of_range     SpO2 < 50 % or > 100 % (0 = no reading), widened by PAD_S = 5 s each side (sensor on/off ramps).
  pleth_flat       pleth peak-to-peak within the second < PLETH_FLAT_MV = 1 mV (~33 digital steps; with the sensor on
                   it is ~1,000 mV): no pulsatile signal, sensor off or disconnected; widened by PAD_S.
  flat             SpO2 exactly constant (same digital value) for >= FLAT_S = 10 s at 50 <= value < FLAT_CEILING
                   = 98 %: a frozen output. In these nights every such run sits at 0 (out_of_range) or at the
                   oximeter's top steps (98.67 %, 99.0 %, held while pleth and pulse stay live), which are not flagged.
  step             SpO2 changes by > STEP_PCT = 10 % within 1 s; after such a fall, everything up to an equally
                   implausible rise within STEP_PLATEAU_S = 60 s (one child holds 73 % for 7 s between a 97 -> 73 % fall in
                   0.2 s and the jump back). The literature's 4 %/s rule is not used: this
                   oximeter's output updates in held steps, and a 4 %/s rule flags the onset of 30-60 % of the
                   scorer-marked desaturations (checked on 8 children while building this stage).
  scored_artifact  inside a scorer-marked 'SpO2 Artifact' interval.
Reported but NOT invalidating:
  low_perfusion    pleth peak-to-peak < LOW_PERF = 10 % of the child's median over in-range sleep seconds. Arousals
                   constrict the finger vessels and shrink the pleth (the basis of peripheral arterial tonometry), and
                   while building the stage this flag fell inside the windows of 34 of the 61 scored arousal-only
                   hypopneas whose window had any flag; invalidating on it would remove arousal windows selectively.
The pleth rules (pleth_flat, low_perfusion) apply only to a night with a live pleth channel (that median >=
PLETH_LIVE_MV = 10 mV; ``pleth_live``): one child's pleth is flat (~0.3 mV) all night while its SpO2 is physiological, so a
missing pleth recording is not read as a missing SpO2. Sleep = hypnogram N1/N2/N3/REM epochs.

Desaturations (``detect``, on valid seconds only): s1 = the second's mean SpO2, rounded to the integer % the oximeter
reports (its analogue output sits at fixed non-integer levels, e.g. 97.78 for 98 and 96.6 for 97, so rounding
recovers the displayed value); sm = 3-s centred median of s1 (NaN if any of the 3 is invalid); pre-event baseline =
the highest sm in the preceding BASELINE_S = 20 s (all 20 valid, else the second is not assessable); an event is a
run of >= MIN_DESAT_S = 2 s with baseline - sm >= the drop (DROP = 3 %). Onset = first second of the run (when the
drop reaches it, so after the fall starts); nadir = lowest s1 in the run. The 20-s window was chosen among 10, 20,
30, 60 and 120 s (max) and 30 / 120 s (mean, median) on 19 children: the 120-s max doubled the event count against
the scored events, the means missed most of them; 10-30 s maxima gave the closest balance.

Hypopnea classes (``classify``), over the window [onset, end + after_s]: desaturating = a signal desaturation with
onset in the window; unknown = otherwise, if any second of the window is invalid or not assessable; arousal-only = a
scored arousal starting in the window; otherwise unclassified.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from project import DATA_ROOT

from stats import psg

PAD_S = 5
PLETH_FLAT_MV = 1.0
FLAT_S = 10
FLAT_CEILING = 98.0
STEP_PCT = 10.0
STEP_PLATEAU_S = 60
LOW_PERF = 0.10
BASELINE_S, DROP, MIN_DESAT_S = 20, 3.0, 2
PLETH_LIVE_MV = 10.0
REASONS = ("out_of_range", "pleth_flat", "flat", "step", "scored_artifact")  # make a second invalid
FLAGS = ("low_perfusion",)  # reported, not invalidating
SLEEP = (1, 2, 3, 5)
FS, PLETH_FS = 10, 50
CLASSES = ("desaturating", "arousal_only", "unclassified", "unknown")
SIGNAL_INPUTS = [DATA_ROOT / "sub-*/psg/sub-*_task-sleep_recording-spo2_physio.tsv.gz",
                 DATA_ROOT / "sub-*/psg/sub-*_task-sleep_recording-pleth_physio.tsv.gz",
                 DATA_ROOT / "sourcedata/tables/oximetry_manifest.tsv"]


def physio(sid: int, kind: str) -> tuple[np.ndarray, dict]:
    base = DATA_ROOT / f"sub-{sid:03d}" / "psg" / f"sub-{sid:03d}_task-sleep_recording-{kind}_physio"
    side = json.loads(base.with_suffix(".json").read_text())
    return pd.read_csv(base.with_suffix(".tsv.gz"), sep="\t", header=None).to_numpy(float), side


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) of each run of True."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def widen(mask: np.ndarray, k: int) -> np.ndarray:
    return np.convolve(mask.astype(float), np.ones(2 * k + 1), mode="same") > 0 if k else mask


def interval_seconds(n: int, starts, ends) -> np.ndarray:
    """Seconds [floor(start), ceil(end)) covered by any interval."""
    m = np.zeros(n, bool)
    for a, b in zip(starts, ends):
        m[max(int(np.floor(a)), 0):min(int(np.ceil(b)), n)] = True
    return m


def overlaps(a0, a1, b0, b1) -> np.ndarray:
    """(len a, len b) bool: interval a meets interval b."""
    return (np.asarray(a0)[:, None] <= np.asarray(b1)[None, :]) & (np.asarray(a1)[:, None] >= np.asarray(b0)[None, :])


def validity(spo2: np.ndarray, pleth: np.ndarray, stage: np.ndarray,
             artifact: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """(per-second mask of each reason (``REASONS`` + ``FLAGS``), 1-s pleth peak-to-peak, its sleep median)."""
    n = len(stage)
    S = spo2[: n * FS].reshape(n, FS)
    p2p = np.ptp(pleth[: n * PLETH_FS].reshape(n, PLETH_FS), axis=1)
    out_of_range = ((S < 50) | (S > 100)).any(axis=1)
    # frozen output: exact constant runs >= FLAT_S in [50, FLAT_CEILING) (0 is out_of_range; the top steps are not)
    flat10 = np.zeros(len(spo2), bool)
    for a, b in runs(np.r_[False, np.diff(spo2) == 0]):
        if b - a + 1 >= FLAT_S * FS and 50 <= spo2[a] < FLAT_CEILING:
            flat10[a - 1:b] = True
    step10 = np.zeros(len(spo2), bool)
    d = spo2[FS:] - spo2[:-FS]
    for i in np.flatnonzero(np.abs(d) > STEP_PCT):
        step10[i:i + FS + 1] = True
    rises = np.flatnonzero(d > STEP_PCT)
    for i in np.flatnonzero(d < -STEP_PCT):  # an implausible fall and the implausible rise ending its plateau
        j = rises[np.searchsorted(rises, i):][:1]
        if len(j) and j[0] - i <= STEP_PLATEAU_S * FS:
            step10[i:j[0] + FS + 1] = True
    ok_sleep = ~out_of_range & np.isin(stage, SLEEP)
    med = float(np.median(p2p[ok_sleep]) if ok_sleep.any() else np.median(p2p))
    live = med >= PLETH_LIVE_MV  # no pleth recorded this night: the pleth rules cannot apply
    return {"out_of_range": widen(out_of_range, PAD_S),
            "pleth_flat": widen(p2p < PLETH_FLAT_MV, PAD_S) & live,
            "flat": flat10[: n * FS].reshape(n, FS).any(axis=1),
            "step": step10[: n * FS].reshape(n, FS).any(axis=1),
            "low_perfusion": (p2p < LOW_PERF * med) & live,
            "scored_artifact": artifact}, p2p, med


def detect(s1: np.ndarray, drop_pct: float = DROP) -> tuple[pd.DataFrame, np.ndarray]:
    """Desaturations on the 1-s valid series ``s1`` (NaN = invalid); returns (events, assessable seconds)."""
    s1 = np.round(s1)
    sm = pd.Series(s1).rolling(3, center=True).median()
    base = sm.rolling(BASELINE_S, min_periods=BASELINE_S).max().shift(1)
    drop = (base - sm).to_numpy()
    rows = []
    for a, b in runs(np.nan_to_num(drop, nan=-1) >= drop_pct):
        if b - a >= MIN_DESAT_S:
            rows.append({"onset_s": float(a), "end_s": float(b), "baseline": float(base.iloc[a]),
                         "nadir": float(np.nanmin(s1[a:b])), "drop": float(base.iloc[a] - np.nanmin(s1[a:b]))})
    return pd.DataFrame(rows, columns=["onset_s", "end_s", "baseline", "nadir", "drop"]), np.isfinite(drop)


def load(sid: int, ev: pd.DataFrame) -> dict:
    """One child's night on the 1-s grid of the recording clock. ``ev``: its scored events with hypnogram ``stage``
    (``stats.hypopnea_type.child``), for the scored-artifact intervals."""
    f = psg.paths(sid)
    start, rec_len = psg.read_header(f.set)
    sp, side = physio(sid, "spo2")
    pl, side_p = physio(sid, "pleth")
    edf_start = f"{side['RecordingStartDate']}T{side['RecordingStartClockTime']}"
    assert edf_start == start.isoformat() and side_p["RecordingStartClockTime"] == side["RecordingStartClockTime"], \
        f"{sid}: EDF start {edf_start} != .set start {start.isoformat()}"
    assert side["EDFRecords"] == round(rec_len), f"{sid}: EDF records {side['EDFRecords']} != .set length {rec_len}"
    assert side["SamplingFrequency"] == FS and side_p["SamplingFrequency"] == PLETH_FS
    spo2, pulse, pleth = sp[:, 0], sp[:, 1], pl[:, 0]
    n = len(spo2) // FS
    hyp = psg.read_hypnogram(f.hypnogram)
    stage = np.full(n, -9, np.int16)
    m = min(n, 30 * len(hyp))
    stage[:m] = np.repeat(hyp, 30)[:m]
    art = ev[ev["label"].str.lower().str.contains("spo2 artifact")]
    masks, p2p, p2p_med = validity(spo2, pleth, stage, interval_seconds(n, art["t_start"], art["t_end"]))
    invalid = np.logical_or.reduce([masks[r] for r in REASONS])
    S = spo2[: n * FS].reshape(n, FS)
    return {"id": sid, "stage": stage, "masks": masks, "invalid": invalid, "S": S, "p2p": p2p,
            "s1": np.where(invalid, np.nan, S.mean(axis=1)), "pulse": pulse[: n * FS].reshape(n, FS).mean(axis=1),
            "edf_start": edf_start, "n": n, "p2p_med": p2p_med,
            "valid_n2n3_h": (np.isin(stage, (2, 3)) & ~invalid).sum() / 3600}


def classify(on, off, arousals, sig: pd.DataFrame, assess: np.ndarray, desat_s: float, arousal_s: float) -> np.ndarray:
    """Signal class (``CLASSES``) of each hypopnea [on, off] (s): a desaturation of ``sig`` with onset in
    [on, off + desat_s]; else unknown if a second of that window is not ``assess``-able; else an ``arousals`` onset in
    [on, off + arousal_s]; else unclassified."""
    ar, onset = np.asarray(arousals), sig["onset_s"].to_numpy()
    out = []
    for a, e in zip(on, off):
        b = e + desat_s
        if ((onset >= np.floor(a)) & (onset <= b)).any():
            out.append("desaturating")
        elif not assess[max(int(np.floor(a)), 0):int(np.ceil(b)) + 1].all():
            out.append("unknown")
        elif ((ar >= a) & (ar <= e + arousal_s)).any():
            out.append("arousal_only")
        else:
            out.append("unclassified")
    return np.array(out, dtype="<U12")
