# code/

Two things live here: the drawing modules the main figures use (`plotting/`, run by `../reproduce.sh` on `../source_data/`), and the analysis pipeline that produced every number, table and figure input of the paper.

## The pipeline

```
uv sync
SPINDLEPEDS_DATA=/path/to/study/data uv run python code/run_all.py            # every stage
SPINDLEPEDS_DATA=/path/to/study/data uv run python code/run_all.py --dry-run  # check inputs; compute nothing
```

`run_all.py` runs the stages in dependency order (`--from STAGE`, `--only STAGE ...`), writes each output under `results/` with a `.provenance.json` sidecar (`provenance.py`) and finishes with `results/RUN_MANIFEST.json`. It reads the study data (source tables, cleaned N2 + N3 recordings, PSG scoring, oximeter signals and spindle detections), which are not in this repository and are available from the corresponding author on reasonable request.

| Path | Role |
|---|---|
| `project.py` | Every path; nothing else types one. |
| `detection/` | Spindle detection with YASA at 100 Hz and the paper's parameters (`settings.py`), its checks and the detections manifest. |
| `stats/metrics.py` | Channel × child spindle metrics (count, density, amplitude, duration, frequency; slow and fast). |
| `stats/io.py`, `regression.py` | Loaders (montage, covariates, spindle matrices), the respiratory-index coding, OLS helpers and the Benjamini–Hochberg rate. |
| `stats/clusters.py`, `rois.py` | The channel-wise models of the hypopnea index and their cluster-based permutation tests (Freedman–Lane); one region value per surviving cluster. |
| `stats/spindle_tova.py`, `behavioral.py`, `behavioral_family.py` | The 32 spindle × attention maps with their false-discovery-rate correction, and the region-level attention models. |
| `stats/*_sensitivity.py`, `fast_duration_robustness.py`, `*_interaction.py`, `density_audit.py`, `duration_frequency.py`, `respiratory_indices.py`, `band_boundary.py`, `audit_numbers.py` | Sensitivity and secondary analyses. |
| `stats/oxygenation.py`, `oximetry.py`, `oximetry_qc.py`, `hypopnea_type.py`, `psg.py` | Oxygenation measures, the oximeter signal and its validity, and the hypopnea-type analysis. |
| `stats/table1.py`, `figure_data.py`, `figure_tables.py` | The descriptive tables and what the figure scripts read. |
| `plotting/` | Helpers of the figure scripts under `../figures/<id>/make.py`; they read tables and fit nothing. |
| `reproduce.py` | What `../reproduce.sh` runs. |
