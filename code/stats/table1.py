"""Table 1 (managed id ``table1``): sample, respiratory indices, sleep architecture, oxygenation,
fast and slow spindle metrics (channel averages; zero count and density where a channel has no detection) and
TOVA Q1. Continuous rows mean (min, max), the SpO2 nadir median (IQR); categorical rows n (%).

Inputs : COV_PATH, ROI_PATH, METRICS_LONG, results/tables/oxygenation_{derived,reported}.csv, oximetry_nadir.csv.
Output : results/tables/table1.csv, table1.json (rows with full statistics, footnote, audit block)
Run    : cd code && python -m stats.table1
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from project import ROI_PATH, TABLE_DIR, ensure_dir, sample_ids
from provenance import record
from stats.behavioral import load_behavioral_frame
from stats.io import (
    OAHI_COL,
    add_obstructive_ahi,
    load_covariates,
    matrix_inputs,
    spindle_matrices,
)

ODI_PATH = TABLE_DIR / "oxygenation_derived.csv"
REPORT_PATH = TABLE_DIR / "oxygenation_reported.csv"  # the stated obstructive AHI (audit)
NADIR_PATH = TABLE_DIR / "oximetry_nadir.csv"

BANDS = (("fast", "Fast spindles (12–16 Hz)"), ("slow", "Slow spindles (10–12 Hz)"))

FOOTNOTE = [
    "Continuous variables are mean (min, max); categorical variables n (%).",
    "Obstructive AHI is the apnea–hypopnea index minus the central apnea index; total AHI includes central apneas.",
    "Sleep stages are percentages of total sleep time. Sleep efficiency was unavailable or implausible for three children.",
    "The oxygen desaturation index is the number of scored desaturations of at least 3% during sleep per hour of total sleep time; the SpO2 nadir is the lowest 1-s SpO2 during sleep over the valid oximeter signal (children with an oximeter recording), given as median (interquartile range).",
    "Spindle metrics are averaged across the 172 scalp channels: a channel without a detection in a band contributes zero to count and density, and amplitude, duration and mean spindle frequency are averaged over channels with at least one detection. Density is the number of detections per minute of artifact-free N2 + N3 sleep.",
    "TOVA = Test of Variables of Attention, first quarter (Q1). Omission errors are the percentage of the 36 targets missed; commission errors the percentage of the 126 non-targets responded to.",
]


def fmt(v, d: int = 1) -> str:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return f"{v.mean():.{d}f} ({v.min():.{d}f}, {v.max():.{d}f})"


def fmt_median(v, d: int = 1) -> str:
    q1, med, q3 = np.percentile(np.asarray(v, dtype=float), [25, 50, 75])
    return f"{med:.{d}f} ({q1:.{d}f}–{q3:.{d}f})"


def stats_of(v) -> dict:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "median": float(med), "q1": float(q1),
            "q3": float(q3), "min": float(v.min()), "max": float(v.max())}


class Table:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.section = ""

    def start(self, section: str) -> None:
        self.section = section

    def cont(self, label: str, v, d: int = 1, median: bool = False) -> None:
        v = np.asarray(v, dtype=float)
        ok = v[np.isfinite(v)]
        value = fmt_median(ok, d) if median else fmt(ok, d)
        self.rows.append({"section": self.section, "row": label, "value": value, "n": int(ok.size), "stats": stats_of(ok)})

    def count(self, label: str, k: int, n: int) -> None:
        self.rows.append({"section": self.section, "row": label, "value": f"{k} ({100 * k / n:.0f}%)", "n": n,
                          "stats": {"count": k}})

    def sex(self, label: str, gender: pd.Series) -> None:
        n, boys, girls = len(gender), int((gender == 0).sum()), int((gender == 1).sum())
        assert boys + girls == n
        self.rows.append({"section": self.section, "row": label,
                          "value": f"{boys} ({100 * boys / n:.0f}%) / {girls} ({100 * girls / n:.0f}%)", "n": n,
                          "stats": {"boys": boys, "girls": girls}})


def spindle_summaries(cov: pd.DataFrame) -> dict[str, np.ndarray]:
    """Participant-level channel averages per band and metric from the analysed detections."""
    slow, fast, sleep_mins = spindle_matrices(cov, 172)
    out = {}
    for band, m in (("slow", slow), ("fast", fast)):
        assert np.isfinite(m["Count"]).all(), "every channel carries a count (zero when empty)"
        out[f"{band}_count"] = m["Count"].mean(axis=1)
        out[f"{band}_density"] = (m["Count"] / sleep_mins).mean(axis=1)
        for metric in ("Amplitude", "Duration", "Frequency"):
            out[f"{band}_{metric.lower()}"] = np.nanmean(m[metric], axis=1)
    return out


def off_grid_tova(tova: pd.DataFrame) -> dict[str, list[int]]:
    """Participants whose Q1 omission / commission % is not a multiple of 100/36 / 100/126.

    A standard Q1 presents 36 targets and 126 non-targets; a value off that grid means a
    different number of trials was scored (e.g. a test stopped early). Reported, not altered.
    """
    out = {}
    for col, n_trials in (("OMPERQ1", 36), ("COMPERQ1", 126)):
        k = tova[col].to_numpy() * n_trials / 100.0
        bad = np.abs(k - np.round(k)) > 0.02
        out[col] = [int(i) for i in tova.loc[bad, "id"]]
        if bad.any():
            print(f"[WARN] {col} off the {n_trials}-trial grid for ids {out[col]}")
    return out


def oxygenation_rows(t: Table, ids: pd.Series) -> None:
    t.cont("Oxygen desaturation index (≥ 3%, events/h)", pd.read_csv(ODI_PATH).set_index("id")["odi"].reindex(ids).to_numpy())
    nadir = pd.read_csv(NADIR_PATH).set_index("id")["signal_nadir_valid"].reindex(ids)
    t.cont("SpO2 nadir, valid signal (%), median (IQR)", nadir.to_numpy(), median=True)


def check_oahi(cov: pd.DataFrame, oahi: pd.Series) -> dict:
    """Compare AHI - CAI with the obstructive AHI the interpreting physician states, where stated."""
    rep = pd.read_csv(REPORT_PATH).set_index("id")["oahi_reported"].reindex(cov["id"]).to_numpy()
    ok = np.isfinite(rep)
    diff = np.abs(oahi.to_numpy()[ok] - rep[ok])
    return {"checked": int(ok.sum()), "max_abs_difference": float(diff.max()) if ok.any() else None}


def main() -> None:
    cov = load_covariates()
    assert set(cov["id"]) == set(sample_ids())
    tova = load_behavioral_frame()
    spindles = spindle_summaries(cov)
    off_grid = off_grid_tova(tova)

    oahi = add_obstructive_ahi(cov)[OAHI_COL]
    assert (oahi >= 0).all()
    oahi_check = check_oahi(cov, oahi)

    t = Table()
    t.start("Sample")
    t.cont("Age (years), full sample", cov["age_decimal"])
    t.sex("Boys / Girls, full sample", cov["gender"])
    t.cont("Age (years), TOVA sample", tova["age_decimal"])
    t.sex("Boys / Girls, TOVA sample", tova["gender"])

    t.start("Respiratory indices (events/h)")
    t.cont("Hypopnea index", cov["overall_hi"])
    t.cont("Obstructive apnea–hypopnea index", oahi)
    t.cont("Total apnea–hypopnea index", cov["overall_ahi"])
    t.cont("Obstructive apnea index", cov["overall_oai"])

    t.start("Sleep architecture")
    t.cont("Total sleep time (min)", cov["tst"])
    t.cont("Sleep efficiency (%)", cov["se"])
    t.cont("Wake after sleep onset (min)", cov["waso"])
    t.cont("N1 (% of TST)", cov["n1"])
    t.cont("N2 (% of TST)", cov["n2"])
    t.cont("N3 (% of TST)", cov["n3"])
    t.cont("REM (% of TST)", cov["rem"])
    t.cont("Arousal index (events/h)", cov["ari"])

    t.start("Oxygenation")
    oxygenation_rows(t, cov["id"])

    for band, title in BANDS:
        t.start(title)
        t.cont("Count (detections per channel)", spindles[f"{band}_count"])
        t.cont("Density (events/min)", spindles[f"{band}_density"], 2)
        t.cont("Amplitude (µV)", spindles[f"{band}_amplitude"])
        t.cont("Duration (s)", spindles[f"{band}_duration"], 2)
        t.cont("Mean spindle frequency (Hz)", spindles[f"{band}_frequency"])

    t.start("Attention (TOVA Q1)")
    t.cont("d′", tova["DPRIMEQ1"])
    t.cont("Omission errors (%)", tova["OMPERQ1"])
    t.cont("Commission errors (%)", tova["COMPERQ1"])
    t.cont("Mean reaction time (ms)", tova["RTMEANQ1"])

    ensure_dir(TABLE_DIR)
    csv = pd.DataFrame([{k: r[k] for k in ("section", "row", "value", "n")} for r in t.rows])
    csv["n"] = csv["n"].astype("Int64")
    out_csv = TABLE_DIR / "table1.csv"
    csv.to_csv(out_csv, index=False)
    out_json = TABLE_DIR / "table1.json"
    audit = {
        "tova_q1_off_trial_grid": off_grid,
        "obstructive_ahi_vs_reports": oahi_check,
        # children at or above the obstructive-AHI cutoffs of mild (1/h), moderate (5/h) and severe (10/h) pediatric OSA
        "obstructive_ahi_at_or_above": {"n": len(oahi), **{str(c): int((oahi >= c).sum()) for c in (1, 5, 10)}},
    }
    out_json.write_text(json.dumps({"columns": ["section", "row", "value", "n"], "rows": t.rows,
                                    "footnote": FOOTNOTE, "audit": audit}, indent=2, ensure_ascii=False) + "\n")

    inputs = [*matrix_inputs(), ROI_PATH, ODI_PATH, REPORT_PATH, NADIR_PATH]
    for out in (out_csv, out_json):
        record(out, script=__file__, inputs=inputs)

    pd.set_option("display.width", 200)
    print(csv.to_string(index=False))
    print(f"\n[SAVED] {out_csv}\n[SAVED] {out_json}")


if __name__ == "__main__":
    main()
