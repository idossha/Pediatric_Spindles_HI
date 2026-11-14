#!/usr/bin/env bash
# Reproduce Figures 1-6 as output/fig-N.png (or only the ones named, e.g. ./reproduce.sh fig-4).
set -euo pipefail
cd "$(dirname "$0")"
if command -v uv >/dev/null; then uv run python code/reproduce.py "$@"; else .venv/bin/python code/reproduce.py "$@"; fi
