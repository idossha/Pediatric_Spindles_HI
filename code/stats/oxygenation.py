"""Oxygenation per child: ODI (sleep-stage 'SpO2 Desat' events with drop >= 3 % per hour of TST) and the
lowest scored desaturation SpO2 from the scored-events export; the physician-read SpO2 nadir and any stated
obstructive AHI from the clinical PSG report (.rtf, read with macOS ``textutil``).

Inputs : DATA_ROOT/sub-*/psg (scored events, reports), COV_PATH.
Output : results/tables/oxygenation_derived.csv, oxygenation_reported.csv
Run    : cd code && python -m stats.oxygenation
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from project import COV_PATH, DATA_ROOT, PARTICIPANTS_PATH, TABLE_DIR, ensure_dir
from provenance import record
from stats.io import load_covariates

SLEEP_STAGES = {"N1", "N2", "N3", "REM"}
DESAT_LABEL = "SpO2 Desat"
MIN_DROP = 3.0

NADIR_PATTERNS = [
    ("SpO2 nadir", re.compile(r"SpO2 nadir\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)),
    ("lowest oxygen saturation", re.compile(r"lowest oxygen saturation\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)),
]
OAHI_PATTERN = re.compile(r"obstructive AHI\D{0,30}?(\d+(?:\.\d+)?)", re.IGNORECASE)


def psg_files(sid: int, pattern: str) -> list[Path]:
    folder = DATA_ROOT / f"sub-{sid:03d}" / "psg"
    if not folder.exists():
        return []
    return sorted(f for f in folder.glob(pattern) if not f.name.startswith("._"))


def parse_events(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="latin-1").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        rows.append({"stage": parts[2].upper(), "event": parts[3], "spo2": parts[5], "drop": parts[6]})
    ev = pd.DataFrame(rows, columns=["stage", "event", "spo2", "drop"])
    ev["spo2"] = pd.to_numeric(ev["spo2"], errors="coerce")
    ev["drop"] = pd.to_numeric(ev["drop"], errors="coerce")
    return ev


def derived_row(sid: int, tst_min: float) -> tuple[dict, Path | None]:
    files = psg_files(sid, "*scored*event*.txt")
    if not files:
        return {"id": sid, "n_desat_sleep": np.nan, "odi": np.nan, "spo2_nadir_sleep": np.nan, "source_file": ""}, None
    ev = parse_events(files[0])
    desat_sleep = (ev["event"] == DESAT_LABEL) & ev["stage"].isin(SLEEP_STAGES)
    n = int((desat_sleep & (ev["drop"] >= MIN_DROP)).sum())
    nadir = ev.loc[desat_sleep, "spo2"].min()
    return {
        "id": sid,
        "n_desat_sleep": n,
        "odi": round(n / (tst_min / 60.0), 4),
        "spo2_nadir_sleep": float(nadir) if pd.notna(nadir) else np.nan,
        "source_file": files[0].name,
    }, files[0]


def report_text(path: Path) -> str:
    return subprocess.run(["textutil", "-convert", "txt", "-stdout", str(path)], capture_output=True, text=True, check=True).stdout


def read_report(text: str) -> tuple[float, str, float]:
    nadir, phrase = np.nan, ""
    for name, pattern in NADIR_PATTERNS:
        m = pattern.search(text)
        if m:
            nadir, phrase = float(m.group(1)), name
            break
    m = OAHI_PATTERN.search(text)
    return nadir, phrase, float(m.group(1)) if m else np.nan


def reported_row(sid: int) -> tuple[dict, list[Path]]:
    files = psg_files(sid, "*Polysomnography*.rtf")
    if not files:
        return {"id": sid, "nadir_pct": np.nan, "nadir_phrase": "", "oahi_reported": np.nan, "source_file": ""}, []
    reads = [read_report(report_text(f)) for f in files]
    if not all(np.isclose(r[0], reads[0][0], equal_nan=True) for r in reads):
        raise ValueError(f"sub-{sid:03d}: report copies disagree: {reads}")
    nadir, phrase, oahi = reads[0]
    return {"id": sid, "nadir_pct": nadir, "nadir_phrase": phrase, "oahi_reported": oahi, "source_file": files[0].name}, files


def main() -> None:
    cov = load_covariates()
    derived, reported, event_files, report_files = [], [], [], []
    for sid, tst in zip(cov["id"].astype(int), cov["tst"].astype(float)):
        row, f = derived_row(sid, tst)
        derived.append(row)
        event_files += [f] if f else []
        row, fs = reported_row(sid)
        reported.append(row)
        report_files += fs

    ensure_dir(TABLE_DIR)
    der = pd.DataFrame(derived)
    out = TABLE_DIR / "oxygenation_derived.csv"
    der.to_csv(out, index=False)
    record(
        out,
        script=__file__,
        inputs=[COV_PATH, PARTICIPANTS_PATH, *event_files],
        note=(
            "odi = sleep-stage (N1/N2/N3/REM) events labelled exactly 'SpO2 Desat' with drop >= 3%, per hour of tst; "
            "spo2_nadir_sleep = lowest SpO2 of a sleep-stage 'SpO2 Desat' event; 'SpO2 Artifact' markers excluded. "
            f"Coverage {der['odi'].notna().sum()}/{len(der)} (no event export for ids "
            + ", ".join(str(i) for i in der.loc[der['odi'].isna(), 'id'])
            + ")."
        ),
    )

    rep = pd.DataFrame(reported)
    out2 = TABLE_DIR / "oxygenation_reported.csv"
    rep.to_csv(out2, index=False)
    no_report = rep.loc[rep["source_file"] == "", "id"].tolist()
    no_nadir = rep.loc[(rep["source_file"] != "") & rep["nadir_pct"].isna(), "id"].tolist()
    record(
        out2,
        script=__file__,
        inputs=[COV_PATH, PARTICIPANTS_PATH, *report_files],
        note=(
            "nadir_pct = the interpreting physician's 'SpO2 nadir X%' (or 'lowest oxygen saturation of X%') from the "
            "clinical PSG report, read with textutil -convert txt. "
            f"Coverage {rep['nadir_pct'].notna().sum()}/{len(rep)}; no report on the drive for ids {no_report}; "
            f"report without a stated nadir for ids {no_nadir}. oahi_reported = physician-stated obstructive AHI where given."
        ),
    )

    both = der.merge(rep, on="id").merge(cov[["id", "overall_ahi", "overall_cai"]], on="id")
    both["oahi"] = (both["overall_ahi"] - both["overall_cai"]).round(2)
    stated = both["oahi_reported"].notna()
    print(der.describe().round(2).to_string())
    print(f"\nODI coverage {der['odi'].notna().sum()}/{len(der)}; physician nadir coverage {rep['nadir_pct'].notna().sum()}/{len(rep)} "
          f"(no report: {no_report}; no nadir stated: {no_nadir})")
    print(rep["nadir_pct"].describe().round(2).to_string())
    diff = (both["spo2_nadir_sleep"] - both["nadir_pct"]).abs()
    print(f"derived vs physician nadir: |diff| > 3 points in {(diff > 3).sum()} of {diff.notna().sum()}")
    print(f"physician-stated obstructive AHI == AHI - CAI in {(both.loc[stated, 'oahi_reported'] == both.loc[stated, 'oahi']).sum()} of {stated.sum()} reports")
    print(f"\n[SAVED] {out}\n[SAVED] {out2}")


if __name__ == "__main__":
    main()
