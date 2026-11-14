"""Tanner-stage distribution and the fast-duration HI cluster test without the mid/late pubertal children.

Tanner stage (``tanner_final``, 1-5) comes from TANNER_PATH: one sheet per sex, so a child listed on both sheets
(two children) takes the row of the sheet matching ``gender`` (1 = female). Stage 1 is pre-, 2 early, 3 mid and
>= 4 late pubertal. The distribution is reported for the spindle sample (N = 62) and the TOVA sample
(``stats.behavioral.load_behavioral_dataset``, N = 56). The children at stage >= 3 are then omitted from the
primary fast-duration cluster test (``stats.clusters.cluster_permutation_test``: default predictors, N_PERM,
SEED); centring is irrelevant to the HI slope's t, so the covariates are used as loaded.
n_overlap_canonical: channels shared with the primary 39-channel cluster (cluster_tmaps.csv in_cluster).

Inputs : TANNER_PATH, COV_PATH, CHAN_PATH, METRICS_LONG, results/tables/cluster_tmaps.csv.
Output : results/tables/pubertal_stage_distribution.csv, results/tables/pubertal_sensitivity.csv
Run    : cd code && uv run python -m stats.pubertal_sensitivity
"""

from __future__ import annotations

import pandas as pd
from project import TABLE_DIR, TANNER_PATH, ensure_dir
from provenance import record
from stats.behavioral import load_behavioral_dataset
from stats.clusters import TMAPS_PATH, adjacency_matrix, cluster_permutation_test
from stats.io import load_channel_info, load_covariates, matrix_inputs, spindle_matrices

SHEETS = {"female": ("pptid", 1), "male": ("ppt_number", 0)}  # sheet: (id column, gender code)
STAGES = {1: "pre-pubertal", 2: "early pubertal", 3: "mid-pubertal", 4: "late pubertal", 5: "late pubertal"}
DIST = TABLE_DIR / "pubertal_stage_distribution.csv"
OUT = TABLE_DIR / "pubertal_sensitivity.csv"


def tanner(cov: pd.DataFrame) -> pd.Series:
    """Tanner stage per child of ``cov`` (index id; NaN = not rated), from the sheet of the child's sex."""
    rows = [pd.read_excel(TANNER_PATH, sheet_name=s).rename(columns={col: "id"})[["id", "tanner_final"]].assign(gender=g)
            for s, (col, g) in SHEETS.items()]
    t = pd.concat(rows).dropna(subset=["id"]).astype({"id": int})
    m = cov[["id", "gender"]].merge(t, on=["id", "gender"], how="left")
    assert not m["id"].duplicated().any(), "a child has two Tanner rows of the same sex"
    return m.set_index("id")["tanner_final"]


def distribution(stage: pd.Series, sample: str) -> pd.DataFrame:
    cat = stage.map(STAGES).fillna("not rated")
    order = ["pre-pubertal", "early pubertal", "mid-pubertal", "late pubertal", "not rated"]
    counts = cat.value_counts().reindex(order, fill_value=0)
    return pd.DataFrame({"sample": sample, "n_sample": len(stage), "stage": order, "n": counts.to_numpy()})


def main() -> None:
    channels = load_channel_info()
    cov = load_covariates()
    stage = tanner(cov)
    tova_ids = load_behavioral_dataset()["id"]
    dist = pd.concat([distribution(stage, "spindle (N = 62)"), distribution(stage[tova_ids], "TOVA (N = 56)")])
    print(dist.to_string(index=False))

    pubertal = sorted(stage.index[stage >= 3])
    adjacency = adjacency_matrix(channels)
    _, fast, _ = spindle_matrices(cov, len(channels.labels))
    keep = ~cov["id"].isin(pubertal).to_numpy()
    top = cluster_permutation_test(fast["Duration"][keep], cov[keep].reset_index(drop=True), adjacency)[0]
    tm = pd.read_csv(TMAPS_PATH)
    canonical = set(tm.loc[(tm["band"] == "fast") & (tm["metric"] == "Duration") & (tm["in_cluster"] == 1), "channel"] - 1)
    row = {"excluded_ids": ";".join(f"{i:03d}" for i in pubertal), "n": int(keep.sum()), "n_channels": top["n"],
           "cluster_mass": top["mass"], "peak_t": top["peak_t"], "p_cluster": top["p"],
           "n_overlap_canonical": len(set(top["channels"]) & canonical), "n_canonical": len(canonical)}
    print(row)

    ensure_dir(TABLE_DIR)
    dist.to_csv(DIST, index=False)
    record(DIST, script=__file__, inputs=[TANNER_PATH],
           note="Stage 1 pre, 2 early, 3 mid, >= 4 late pubertal; 'not rated' = no Tanner row or no rating.")
    pd.DataFrame([row]).to_csv(OUT, index=False, float_format="%.6g")
    record(OUT, script=__file__, inputs=[TANNER_PATH, *matrix_inputs(), TMAPS_PATH],
           note="Children at Tanner stage >= 3 omitted.")


if __name__ == "__main__":
    main()
