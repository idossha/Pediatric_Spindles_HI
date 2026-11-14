"""``python -m detection detect ID ... [--workers N --tmp DIR]`` or ``python -m detection verify``."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from detection import checks, manifest
from detection.files import describe, detections_path, sha256
from detection.settings import DETECT_SF, YASA_KWARGS


def run_detect(sid: int, workers: int, tmp: Path) -> None:
    import mne
    import scipy
    import yasa

    from detection.detect import detect

    t0 = time.time()
    df = detect(sid, workers, tmp)
    out = detections_path(sid)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=True)
    info = describe(sid)
    manifest.write_row({"id": sid, "source": "detected", **info, "detect_sf": DETECT_SF, "detect_params": str(YASA_KWARGS),
                        "yasa": yasa.__version__, "mne": mne.__version__, "numpy": np.__version__,
                        "scipy": scipy.__version__, "pandas": pd.__version__, "python": sys.version.split()[0],
                        "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "output": out.name,
                        "output_sha256": sha256(out), "n_detections": len(df),
                        **checks.run(sid, pd.read_csv(out), Path(info["set"]))})
    print(f"{sid:03d}: {len(df):,} detections in {time.time() - t0:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m detection")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="re-detect children and update the manifest")
    d.add_argument("ids", nargs="+", type=int)
    d.add_argument("--workers", type=int, default=8)
    d.add_argument("--tmp", type=Path, default=Path(tempfile.gettempdir()))
    sub.add_parser("verify", help="check the detection set against its manifest")
    a = ap.parse_args()
    if a.cmd == "verify":
        manifest.verify()
        return
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"  # one thread per worker process
    for sid in a.ids:
        run_detect(sid, a.workers, a.tmp)


if __name__ == "__main__":
    main()
