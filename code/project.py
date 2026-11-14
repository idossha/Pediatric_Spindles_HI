"""Every path the code reads or writes; nothing else types one.

The main figures read ``SOURCE_DATA`` (source_data/, in this repository). The analysis pipeline reads the study
data under ``DATA_ROOT`` (source tables, cleaned N2+N3 recordings, PSG scoring, oximeter signals and the spindle
detections; available on request): set ``SPINDLEPEDS_DATA`` to where they are. Its outputs go to ``results/``.
Matplotlib and MNE caches go to ``code/.cache``.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CODE = PROJECT / "code"
CACHE = CODE / ".cache"

SOURCE_DATA = PROJECT / "source_data"  # summarized data of the main figures, one folder per figure

DATA_ROOT = Path(os.environ.get("SPINDLEPEDS_DATA", PROJECT / "data"))  # the study data (not included)
SOURCE_TABLES = DATA_ROOT / "sourcedata/tables"
COV_PATH = SOURCE_TABLES / "analysis_sample_complete_N62.csv"
PARTICIPANTS_PATH = SOURCE_TABLES / "participant_notes.csv"  # every child of the parent log: sample, reasons, notes
TANNER_PATH = SOURCE_TABLES / "osa_peds_tanner.xlsx"  # self-rated Tanner stage, one sheet per sex
CHAN_PATH = SOURCE_DATA / "inside172.mat"  # the 172-channel montage
SPINDLE_DIR = DATA_ROOT / "derivatives/spindles-yasa-100hz"

RESULTS = PROJECT / "results"
TABLE_DIR = RESULTS / "tables"
FIGURE_DATA = RESULTS / "figure_data"
METRICS_LONG = TABLE_DIR / "spindle_metrics_long.csv"
ROI_PATH = TABLE_DIR / "roi_values.csv"


def sample_ids() -> list[int]:
    """The analysed children: ``in_spindle_sample`` = 1 in ``PARTICIPANTS_PATH``, where each exclusion has its reason."""
    import csv

    with open(PARTICIPANTS_PATH, newline="", encoding="utf-8") as fh:
        return sorted(int(r["id"]) for r in csv.DictReader(fh) if r["in_spindle_sample"] == "1")


def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def redirect_caches() -> None:
    """Point MPLCONFIGDIR, XDG_CACHE_HOME and MNE_HOME into code/.cache (idempotent)."""
    for var, folder in (("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "xdg"), ("MNE_HOME", "mne")):
        os.environ.setdefault(var, str(CACHE / folder))
        Path(os.environ[var]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")


redirect_caches()
