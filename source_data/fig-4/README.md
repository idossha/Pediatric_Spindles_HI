# Figure 4 source data

Association between the hypopnea index and each spindle metric across the scalp.

Summarized from the study data by the analysis pipeline (`code/run_all.py`); every file is drawn in the figure. Children are numbered 1..N.

| File | Contents |
|---|---|
| `cluster_tmaps.csv` | 1376 rows; columns `band`, `metric`, `channel`, `t`, `p`, `in_cluster` |
| `cluster_permutation_results.csv` | 8 rows; columns `metric_label`, `band`, `metric`, `n_channels`, `cluster_mass`, `peak_t`, `p_value` |
