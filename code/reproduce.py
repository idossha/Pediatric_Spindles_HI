"""Reproduce the paper's Figures 1-6 as output/fig-1.png ... output/fig-6.png (300 dpi).

Everything is built in a temporary folder that is deleted afterwards: each figure is drawn by its make.py from
source_data/<id>/ and composed from its figure.json by idofig, and the composed figure is rasterized with resvg.
Only the PNGs are kept.

Usage: ./reproduce.sh [fig-1 fig-4 ...]   (no argument: every figure)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import idofig
import resvg_py

ROOT = Path(__file__).resolve().parents[1]
FIGURES = sorted(p.name for p in (ROOT / "figures").iterdir() if (p / "figure.json").exists())
DPI = 300
FONTS = Path(idofig.__file__).parent / "fonts"  # the font every figure is drawn with


def build(figures: list[str], work: Path) -> None:
    """Copy the sources to ``work`` and build the figures there with idofig (every script reads the copy)."""
    skip = shutil.ignore_patterns("__pycache__", ".cache", "*.svg", "manifest.json", "figure.lock.json")
    for name in ("pyproject.toml", "figures", "source_data", "code"):
        src = ROOT / name
        shutil.copytree(src, work / name, ignore=skip) if src.is_dir() else shutil.copy2(src, work / name)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, "-m", "idofig", "build", "--force", *figures], cwd=work, env=env, check=True)


def main(figures: list[str]) -> None:
    unknown = sorted(set(figures) - set(FIGURES))
    if unknown:
        sys.exit(f"unknown figure(s) {', '.join(unknown)}; choose from {', '.join(FIGURES)}")
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        build(figures, Path(tmp))
        for fig in figures:
            png = resvg_py.svg_to_bytes(svg_path=str(Path(tmp) / "figures" / fig / "figure.svg"), dpi=float(DPI),
                                        background="#ffffff", skip_system_fonts=True, font_dirs=[str(FONTS)])
            (out / f"{fig}.png").write_bytes(bytes(png))
            print(f"wrote output/{fig}.png")


if __name__ == "__main__":
    main(sys.argv[1:] or FIGURES)
