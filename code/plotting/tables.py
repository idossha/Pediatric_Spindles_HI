"""Read the small CSV tables the figures plot, without pandas.

``read_table(path)`` returns ``{column: numpy array}``: bool where every value is True/False, float where every value
parses (blank or NA is NaN), object (text) otherwise. ``row(path, **match)`` is the one row matching the given values.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_MISSING = {"", "na", "nan", "none", "null"}


def _column(values: list[str]) -> np.ndarray:
    lowered = {v.strip().lower() for v in values}
    if lowered and lowered <= {"true", "false"}:
        return np.array([v.strip().lower() == "true" for v in values], dtype=bool)
    try:
        return np.array([np.nan if v.strip().lower() in _MISSING else float(v) for v in values], dtype=float)
    except ValueError:
        return np.array(values, dtype=object)


def read_table(path: str | Path) -> dict[str, np.ndarray]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    header, body = rows[0], rows[1:]
    return {name: _column([r[i] if i < len(r) else "" for r in body]) for i, name in enumerate(header)}


def row(path: str | Path, **match) -> dict[str, object]:
    table = read_table(path)
    n = len(next(iter(table.values())))
    hits = [i for i in range(n) if all(table[k][i] == v for k, v in match.items())]
    if len(hits) != 1:
        raise ValueError(f"{path}: {len(hits)} rows match {match}")
    return {k: v[hits[0]] for k, v in table.items()}
