# Figure 6 source data

Hypopnea type and fast spindle duration (exploratory).

Summarized from the study data by the analysis pipeline (`code/run_all.py`); every file is drawn in the figure. Children are numbered 1..N.

| File | Contents |
|---|---|
| `hypopnea_type_example_hypnogram.csv` | 1414 rows; columns `t_s`, `code` |
| `hypopnea_type_example_events.csv` | 283 rows; columns `kind`, `cls`, `label`, `stage`, `t_start`, `t_end`, `spo2_drop` |
| `hypopnea_type_example_spindles.csv` | 142 rows; columns `t_s`, `n_fast_cluster`, `child_mean_duration_s`, `id` |
| `hypopnea_type_tmaps.csv` | 172 rows; columns `channel`, `label`, `t_desaturating`, `cluster_desaturating`, `t_arousal_only`, `cluster_arousal_only`, `hi_cluster`, `t_joint_desaturating`, `cluster_joint_desaturating`, `t_joint_arousal_only`, `cluster_joint_arousal_only` |
| `hypopnea_type_children.csv` | 60 rows; columns `id`, `hi`, `desaturating_per_h`, `arousal_only_per_h` |
| `hypopnea_type_windows.csv` | 12 rows; columns `classification`, `desat_after_end_s`, `arousal_after_end_s`, `min_drop_pct`, `primary`, `index`, `n_class`, `n_unclassified`, `n_unknown`, `n_hypopneas`, `n_children`, `r_log_indices`, `beta_alone`, `CI_lower_alone`, `CI_upper_alone`, `p_alone`, `beta_joint`, `CI_lower_joint`, `CI_upper_joint`, `p_joint`, `diff_joint`, `SE_diff_joint`, `p_diff_joint`, `cluster_n_channels`, `cluster_p`, `joint_cluster_n_channels`, `joint_cluster_p` |
| `hypopnea_type_clusters.csv` | 4 rows; columns `index`, `model`, `effect`, `n_children`, `n_channels`, `cluster_mass`, `peak_t`, `p_value`, `n_clusters_formed`, `n_shared_with_hi_cluster`, `channels_1based` |
| `hypopnea_type_models.csv` | 20 rows; columns `outcome`, `model`, `index`, `term`, `beta`, `SE`, `t`, `p`, `CI_lower`, `CI_upper`, `n` |
| `cluster_tmaps.csv` | 1376 rows; columns `band`, `metric`, `channel`, `t`, `p`, `in_cluster` |
