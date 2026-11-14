"""``manifest.csv`` beside the detections: one row per child with inputs, versions, output checksum and checks."""
from __future__ import annotations

import pandas as pd
from project import SPINDLE_DIR, sample_ids

from detection import checks
from detection.files import sha256

PATH = SPINDLE_DIR / "manifest.csv"


def read() -> pd.DataFrame:
    return pd.read_csv(PATH) if PATH.exists() else pd.DataFrame(columns=["id"])


def write_row(entry: dict) -> None:
    m = read()
    m = pd.concat([m[m["id"] != entry["id"]], pd.DataFrame([entry])], ignore_index=True).sort_values("id")
    m.to_csv(PATH, index=False)


def verify() -> pd.DataFrame:
    """Every child of the sample present, each file unchanged since it was recorded, every check passing."""
    m = read()
    ids = sample_ids()
    missing = sorted(set(ids) - set(m["id"].astype(int)))
    if missing:
        raise SystemExit(f"no detections for {missing}: run `python -m detection detect {' '.join(map(str, missing))}`")
    for r in m.itertuples():
        if sha256(SPINDLE_DIR / r.output) != r.output_sha256:
            raise SystemExit(f"{r.output} differs from its manifest row")
    bad = m[~m.apply(checks.passes, axis=1)]
    if len(bad):
        raise SystemExit(f"checks fail:\n{bad[['id', *checks.COLUMNS]].to_string(index=False)}")
    print(f"{len(m)} children verified; alignment rho >= {m['align_spearman'].min():.3f}, "
          f"end gap <= {m['end_gap_s'].max():.0f} s")
    return m
