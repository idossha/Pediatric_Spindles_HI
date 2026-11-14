# Figure 1 source data

Night architecture and spindle detection.

Summarized from the study data by the analysis pipeline (`code/run_all.py`); every file is drawn in the figure. Children are numbered 1..N.

| File | Contents |
|---|---|
| `example_night/hypnogram.csv` | 1162 rows; columns `t_h`, `stage`, `kind` |
| `example_night/raster.csv` | 39904 rows; columns `band`, `order`, `channel`, `bin_start_h`, `bin_end_h`, `density` |
| `example_night/traces.csv` | 3000 rows; columns `channel`, `t`, `uV` |
| `example_night/detections.csv` | 4 rows; columns `channel`, `band`, `start`, `end`, `Frequency` |
| `example_night/criteria.csv` | 600 rows; columns `t`, `eeg`, `sigma`, `rms`, `rel_pow`, `corr` |
| `example_night/meta.csv` | 1 rows; columns `hi`, `window_start_cleaned_s`, `window_start_h`, `window_s`, `thresh_rms`, `thresh_rel_pow`, `thresh_corr` |
| `example_night/montage.csv` | 172 rows; columns `egi`, `x`, `y`, `shown` |
