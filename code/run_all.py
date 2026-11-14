#!/usr/bin/env python3
"""Regenerate every table and figure from ``project.DATA_ROOT``, stage by stage, in dependency order.

From code/ (environment: code/pyproject.toml + uv.lock, synced into code/.venv by ``uv sync``):
  uv run python run_all.py               every stage, then results/RUN_MANIFEST.json
  uv run python run_all.py --from rois   that stage and every later one;  --only a b  just those stages
  uv run python run_all.py --dry-run     check inputs, detections, imports and the environment; compute nothing
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import tomllib

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from project import (
    CHAN_PATH,
    COV_PATH,
    DATA_ROOT,
    PARTICIPANTS_PATH,
    PROJECT,
    RESULTS,
    SPINDLE_DIR,
    sample_ids,
)

PY = [sys.executable]
IDOFIG = Path(sys.executable).parent / "idofig"

# name, command (Python stages run from code/, idofig from the project root), producer whose sidecars name its outputs
STAGES = [
    ("detect", PY + ["-m", "detection", "verify"], None),
    ("metrics", PY + ["-m", "stats.metrics"], "stats/metrics.py"),
    ("oxygenation", PY + ["-m", "stats.oxygenation"], "stats/oxygenation.py"),
    ("respiratory_indices", PY + ["-m", "stats.respiratory_indices"], "stats/respiratory_indices.py"),
    ("clusters", PY + ["-m", "stats.clusters"], "stats/clusters.py"),
    ("rois", PY + ["-m", "stats.rois"], "stats/rois.py"),
    ("figure_data", PY + ["-m", "stats.figure_data"], "stats/figure_data.py"),
    ("audit_numbers", PY + ["-m", "stats.audit_numbers"], "stats/audit_numbers.py"),
    ("fast_duration_robustness", PY + ["-m", "stats.fast_duration_robustness"],
     "stats/fast_duration_robustness.py"),
    ("density_audit", PY + ["-m", "stats.density_audit"], "stats/density_audit.py"),
    ("age_hi_interaction", PY + ["-m", "stats.age_hi_interaction"], "stats/age_hi_interaction.py"),
    ("age_hi_roi_interaction", PY + ["-m", "stats.age_hi_roi_interaction"],
     "stats/age_hi_roi_interaction.py"),
    ("behavioral_family", PY + ["-m", "stats.behavioral_family"], "stats/behavioral_family.py"),
    ("covariate_sensitivity", PY + ["-m", "stats.covariate_sensitivity"], "stats/covariate_sensitivity.py"),
    ("resp_cluster_sensitivity", PY + ["-m", "stats.resp_cluster_sensitivity"],
     "stats/resp_cluster_sensitivity.py"),
    ("spindle_tova", PY + ["-m", "stats.spindle_tova"], "stats/spindle_tova.py"),
    ("duration_frequency", PY + ["-m", "stats.duration_frequency"], "stats/duration_frequency.py"),
    ("threshold_sensitivity", PY + ["-m", "stats.threshold_sensitivity"], "stats/threshold_sensitivity.py"),
    ("band_boundary", PY + ["-m", "stats.band_boundary"], "stats/band_boundary.py"),
    ("pubertal_sensitivity", PY + ["-m", "stats.pubertal_sensitivity"], "stats/pubertal_sensitivity.py"),
    ("hypopnea_type", PY + ["-m", "stats.hypopnea_type"], "stats/hypopnea_type.py"),
    ("oximetry_qc", PY + ["-m", "stats.oximetry_qc"], "stats/oximetry_qc.py"),
    ("table1", PY + ["-m", "stats.table1"], "stats/table1.py"),
    ("figure_tables", PY + ["-m", "stats.figure_tables"], "stats/figure_tables.py"),
]
NAMES = [s[0] for s in STAGES]


def sha256(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def sidecars() -> list[dict]:
    out = []
    for p in sorted(RESULTS.rglob("*.provenance.json")):
        d = json.loads(p.read_text())
        d["_artifact"] = PROJECT / d["artifact"]
        out.append(d)
    return out


def outputs_by_script() -> dict[str, list[Path]]:
    by: dict[str, list[Path]] = {}
    for d in sidecars():
        by.setdefault(d["producer"]["script"], []).append(d["_artifact"])
    return by


def pins() -> dict[str, str]:
    deps = tomllib.loads((CODE / "pyproject.toml").read_text())["project"]["dependencies"]
    return dict(d.split("==") for d in deps if "==" in d)


# ------------------------------------------------------------------------------------ dry run


def dry_run() -> int:
    problems: list[str] = []

    def check(ok: bool, what: str) -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {what}")
        if not ok:
            problems.append(what)

    import pandas as pd

    print(f"DATA_ROOT = {DATA_ROOT}")
    for p in (COV_PATH, PARTICIPANTS_PATH, CHAN_PATH, SPINDLE_DIR / "manifest.csv"):
        check(p.is_file() and p.stat().st_size > 0, f"input {p}")
    if problems:
        return finish(problems)

    print("detections")
    ids = sample_ids()
    m = pd.read_csv(SPINDLE_DIR / "manifest.csv").set_index("id")
    check(set(ids) <= set(m.index), f"a manifest row for each of the {len(ids)} children of the sample")
    bad = [f"{i:03d}" for i in ids if i in m.index
           and not ((SPINDLE_DIR / m.loc[i, "output"]).is_file() and sha256(SPINDLE_DIR / m.loc[i, "output"]) == m.loc[i, "output_sha256"])]
    check(not bad, f"detection files present with the manifest's sha256 {bad or ''}".rstrip())
    missing = [f"{i:03d}" for i in m.index if not (Path(m.loc[i, "set"]).is_file() and Path(m.loc[i, "fdt"]).is_file())]
    check(not missing, f"cleaned .set/.fdt present for every child {missing or ''}".rstrip())
    from detection import checks
    check(m[checks.COLUMNS].notna().all().all() and bool(m.apply(checks.passes, axis=1).all()),
          "every manifest row carries passing checks (alignment, 0.01-s grid, end gap)")
    print(f"  sources: {m['source'].value_counts().to_dict()}")

    print("scored events and reports")
    from stats import psg
    files = {i: psg.paths(i) for i in ids}
    with_events = [i for i, f in files.items() if f.events]
    print(f"  scored-event files for {len(with_events)} of {len(ids)} children")
    check(all(files[i].hypnogram for i in with_events), "a hypnogram for every child with scored events")
    start, rec_len = psg.read_header(files[ids[0]].set)
    check(rec_len > 0, f"smoke: {ids[0]:03d}'s .set header reads (start {start}, {rec_len / 3600:.1f} h)")
    sp = psg.read_spindles(files[ids[0]].spindles)
    check(len(sp) == int(m.loc[ids[0], "n_detections"]), f"smoke: {ids[0]:03d}'s detections read ({len(sp):,} rows)")
    check(shutil.which("textutil") is not None, "macOS textutil (oxygenation reads the clinical reports)")
    try:
        from stats.io import load_covariates
        from stats.metrics import child_metrics
        cov = load_covariates()
        check(len(cov) == len(ids) and cov["clean_min"].notna().all(),
              f"smoke: covariates of {len(ids)} children with cleaned minutes")
        cm = child_metrics(ids[0], float(cov.loc[cov["id"] == ids[0], "clean_min"].iloc[0]), 172)
        check(len(cm) == 3 * 172 and int(cm.loc[cm["band"] == "all", "count"].sum()) == len(sp),
              f"smoke: {ids[0]:03d}'s channel metrics (3 bands x 172 channels, counts sum to its detections)")
    except Exception as e:  # noqa: BLE001  a smoke failure is a finding, not a crash
        check(False, f"smoke: covariates / metrics -> {type(e).__name__}: {e}")

    print("stage modules import")
    for name, cmd, _ in STAGES:
        if cmd[0] != sys.executable:
            continue
        mod = cmd[2] if cmd[1] == "-m" else cmd[1].removesuffix(".py").replace("/", ".")
        try:
            importlib.import_module(mod)
            check(True, f"{name}: import {mod}")
        except Exception as e:  # noqa: BLE001  report every failing import, not only the first
            check(False, f"{name}: import {mod} -> {type(e).__name__}: {e}")

    print("environment (code/pyproject.toml, uv.lock, code/.venv)")
    check(Path(sys.prefix).resolve() == (CODE / ".venv").resolve(), f"running in code/.venv ({sys.prefix})")
    for cmd in (["uv", "lock", "--check"], ["uv", "sync", "--check"]):
        r = subprocess.run(cmd, cwd=CODE, capture_output=True, text=True, check=False)
        check(r.returncode == 0, f"{' '.join(cmd)}" + ("" if r.returncode == 0 else f": {(r.stderr or r.stdout).strip()}"))
    for pkg, want in pins().items():
        try:
            got = version(pkg)
        except PackageNotFoundError:
            got = None
        check(got == want, f"{pkg} {got} (pinned {want})")
    check(IDOFIG.is_file(), f"idofig {version('idofig')} at {IDOFIG}")

    print("outputs per stage (from the current provenance sidecars)")
    by = outputs_by_script()
    for name, _, script in STAGES:
        if script:
            outs = by.pop(f"code/{script}", [])
            print(f"  {name}: " + (", ".join(p.relative_to(RESULTS).as_posix() for p in outs) or "(none yet; written on first run)"))
    orphans = [p.relative_to(RESULTS).as_posix() for ps in by.values() for p in ps]
    if orphans:
        print(f"  not written by any stage (stale; delete or give them a stage): {', '.join(orphans)}")
    return finish(problems)


def finish(problems: list[str]) -> int:
    print(f"\n{'dry run clean' if not problems else f'{len(problems)} problem(s):'}")
    for p in problems:
        print(f"  - {p}")
    return 1 if problems else 0


# ------------------------------------------------------------------------------------- run


def run(selected: list[str]) -> int:
    started = dt.datetime.now(dt.timezone.utc)
    timings = {}
    for name, cmd, script in STAGES:
        if name not in selected:
            continue
        print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
        t0, before = time.time(), dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        if subprocess.run(cmd, cwd=CODE if cmd[0] == sys.executable else PROJECT, check=False).returncode:
            print(f"stage {name} failed")
            return 1
        timings[name] = round(time.time() - t0, 1)
        if script:
            stale = [d["artifact"] for d in sidecars() if d["producer"]["script"] == f"code/{script}" and d["producedAt"] < before]
            if stale:
                print(f"stage {name} did not rewrite: {stale}")
                return 1
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True, check=False).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "code"], cwd=PROJECT, capture_output=True, text=True, check=False).stdout)
    manifest = {
        "started": started.isoformat(timespec="seconds"),
        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "stages_seconds": timings,
        "git": {"commit": commit, "code_dirty": dirty},
        "environment": {"python": sys.version.split()[0],
                        "packages": {p: version(p) for p in [*pins(), "idofig"]}},
        "inputs": {p.name: sha256(p) for p in (COV_PATH, PARTICIPANTS_PATH, CHAN_PATH, SPINDLE_DIR / "manifest.csv")},
        "data_root": str(DATA_ROOT),
        "outputs": {d["artifact"]: {"sha256": d["sha256"], "producer": d["producer"]["script"], "producedAt": d["producedAt"]}
                    for d in sidecars()},
    }
    (RESULTS / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {RESULTS / 'RUN_MANIFEST.json'}; {sum(timings.values()) / 60:.1f} min")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from", dest="start", choices=NAMES)
    ap.add_argument("--only", nargs="+", choices=NAMES)
    a = ap.parse_args()
    if a.dry_run:
        sys.exit(dry_run())
    selected = a.only or (NAMES[NAMES.index(a.start):] if a.start else NAMES)
    sys.exit(run(selected))


if __name__ == "__main__":
    main()
