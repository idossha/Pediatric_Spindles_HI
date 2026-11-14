# Figure 5 source data

Spindle expression and attention across the scalp.

Summarized from the study data by the analysis pipeline (`code/run_all.py`); every file is drawn in the figure. Children are numbered 1..N.

| File | Contents |
|---|---|
| `spindle_tova_tmaps.csv` | 5504 rows; columns `band`, `metric`, `outcome`, `channel`, `t`, `p`, `in_cluster` |
| `spindle_tova_clusters.csv` | 32 rows; columns `band`, `metric`, `outcome`, `N`, `n_channels`, `cluster_mass`, `peak_t`, `p`, `q_fdr32`, `survives`, `channels` |
