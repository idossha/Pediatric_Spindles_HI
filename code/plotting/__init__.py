"""Shared plotting helpers for the figure scripts under figures/<id>/ (library; puts ``code/`` on ``sys.path``).

Runs in the project's one uv environment (``code/.venv``) but fits nothing: model fits are precomputed by
``stats/figure_tables.py`` and ``stats/figure_data.py`` into ``results/figure_data/`` and only read here.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
_CODE = str(PROJECT / "code")
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

import project  # noqa: F401  (redirects the matplotlib/MNE caches into code/.cache)
