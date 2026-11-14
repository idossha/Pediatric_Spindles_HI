# Changelog

All notable changes to this repository are listed here. Versions follow the release tags; the format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0] - Unreleased

The revised manuscript.

### Changed

- The analysis pipeline is one package under `code/` run by `code/run_all.py`: spindles re-detected with the same detector and settings, decimal age in every model, the channel-wise spindle–attention analysis with false-discovery-rate correction across its 32 maps, the sensitivity analyses, and the exploratory hypopnea-type analysis from the oximeter signal.
- Each main figure is one folder under `figures/` (drawing script, layout, style adjustments) built by idofig.

### Added

- `source_data/<id>/`: the summarized data each main figure draws, with a README listing the files.
- `tables/`: Tables 1–3 as CSV.
- `./reproduce.sh`: re-renders Figures 1–6 from the source data.
- `CITATION.cff`, `.zenodo.json` and this changelog.

## [1.0.0] - 2026-07

### Added

- Analysis code of the originally submitted manuscript.
