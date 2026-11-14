"""Write the `<artifact>.provenance.json` sidecar beside a result (results/README.md).

    from provenance import record  # code/ is on sys.path
    df.to_csv(OUT, index=False)
    record(OUT, script=__file__, inputs=[COV_PATH, ROI_PATH])

The sidecar says what produced the file, from which inputs (with hashes), when,
on which machine and in which environment, so a table in results/ can always be
traced back to its inputs on the data root (project.DATA_ROOT) and to the commit that
produced it. Nothing derived is
stored anywhere else.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: os.PathLike[str] | str) -> str:
    """Project-relative where it can be; an input outside the project (the data root) keeps
    its absolute path, so the sidecar still says where the data came from."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "statsmodels", "mne", "matplotlib"):
        with contextlib.suppress(ImportError, AttributeError):
            packages[name] = __import__(name).__version__
    return {"python": sys.version.split()[0], "executable": sys.executable, "packages": packages}


def record(artifact: os.PathLike[str] | str, *, script: os.PathLike[str] | str, inputs: list[os.PathLike[str] | str], note: str | None = None) -> Path:
    artifact = Path(artifact).resolve()
    sidecar = artifact.with_name(artifact.name + ".provenance.json")
    entry = {
        "schemaVersion": 1,
        "artifact": _rel(artifact),
        "sha256": _sha256(artifact),
        "bytes": artifact.stat().st_size,
        "producer": {"script": _rel(script), "command": f"python {_rel(script)}", "commit": _git_commit()},
        "inputs": [{"path": _rel(p), "sha256": _sha256(Path(p)) if Path(p).exists() else None} for p in inputs],
        "producedAt": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "machine": {"hostname": platform.node(), "platform": platform.platform()},
        "environment": _environment(),
        "recordedAt": _dt.date.today().isoformat(),  # noqa: DTZ011 (local date, as every existing sidecar)
        "note": note,
    }
    sidecar.write_text(json.dumps(entry, indent=2) + "\n")
    return sidecar
