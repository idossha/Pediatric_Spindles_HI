"""Exploratory: which hypopneas carry the HI association with shorter anterior fast spindle duration.

Children: the sample (``project.sample_ids``) with a scored-event file and an oximeter recording (two children have
neither). Only hypopneas scored in N2 or N3 are counted: a hypopnea's stage is the hypnogram's 30-s epoch containing
its onset (floor of onset s / 30 on the recording clock, the clock the detections are staged on); the scorer's stage
column of the event line agrees for every hypopnea, which the stage asserts.

Classes (primary: the raw oximeter signal, ``stats.oximetry``): over the window [onset, end + 20 s] a counted
hypopnea is desaturating if the valid signal shows a desaturation (>= 3 %) with onset in the window, unknown if
otherwise a second of the window is invalid or not assessable, arousal-only if otherwise a scored arousal (any stage
or label) starts in the window, else unclassified. Unknown hypopneas enter neither rate. Each class's count per hour
of valid N2 + N3 SpO2 (hypnogram N2/N3 seconds whose signal is valid) enters as log10(rate + 1), centred, with age_c
and gender_bin, as HI does.

Comparison ('scored events'): the same hypopneas classed from the scored-event file alone, on its one clock: a
scored desaturation with drop >= 3 % starting in the window, else a scored arousal, else unclassified; rates per hour
of N2 + N3 sleep (``tst`` min x (``n2`` + ``n3``) / 100 / 60, the covariate table's).

Outcome: fast spindle duration, channel-wise (cluster test as ``stats.clusters``: same threshold, 5,000
Freedman-Lane permutations, seed 20260607) and as the mean over the corrected 39-channel HI cluster
(``roi_values.csv`` ``fast_duration_cluster``, reproduced here from the detections and checked equal); and the
same cluster mean from N2 spindles only and N3 spindles only, and their within-child difference (N2 minus N3), whose
coefficients are the stage differences of the two associations. A detection's stage: its Start (s on the cleaned
N2+N3 file) x the .set's srate is its cleaned sample, whose original 200-Hz index (``psg.read_set``) gives the
recording time and so the 30-s hypnogram epoch. REM never enters: the detections are N2+N3 only.

Per rule (``evaluate``): each rate alone and both in one model on the cluster mean (with a Wald test of
beta_desaturating - beta_arousal_only in the joint model), and the channel-wise cluster test of each rate alone and
in the joint model (each type in turn the effect, the other a covariate). Rules (s after end for the desaturation,
s after end for the arousal, minimum drop %): the primary (20, 20, 3) and four alternatives on the signal classes,
and the scored-event classes at the primary rule.

Inputs : COV_PATH, PARTICIPANTS_PATH, CHAN_PATH, METRICS_LONG, ROI_PATH, cluster_permutation_results.csv, and per
         child DATA_ROOT/sub-*/psg/*_scored*events.txt, *_hypnogram.txt, the oximeter files (``stats.oximetry``),
         the cleaned .set and the detections.
Output : results/tables/hypopnea_type_{children,models,clusters,windows}.csv (children, models, clusters: the primary
         signal classes; windows: every rule, ``classification`` signal or scored events);
         results/figure_data/hypopnea_type_tmaps.csv (t and largest cluster of each type alone and, ``joint``, with both
         in one model; ``hi_cluster``: 0/1, the HI cluster's channels) and
         hypopnea_type_example_{hypnogram,events}.csv
         (fig-6 a's example child, ``psg.example_child``: its N2/N3 hypopneas with class, the signal desaturations and all arousals) and hypopnea_type_example_spindles.csv (its N2/N3 fast detections on the cluster
         channels per SPINDLE_BIN_S of the recording clock, its id and its cluster-mean fast duration, the value the
         analysis uses).
Run    : cd code && python -m stats.hypopnea_type
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from project import DATA_ROOT, FIGURE_DATA, ROI_PATH, TABLE_DIR, ensure_dir, sample_ids
from provenance import record
from stats import oximetry as ox
from stats import psg
from stats.clusters import (
    RESULTS_PATH,
    adjacency_matrix,
    cluster_permutation_general,
    fmt_channels,
    largest_cluster_channels,
)
from stats.io import (
    FAST_RANGE,
    PREDICTORS,
    data_matrix_for_metric,
    load_channel_info,
    load_covariates,
    matrix_inputs,
    spindle_matrices,
)
from stats.regression import ols_row

WINDOWS = [(20.0, 20.0, 3.0), (10.0, 10.0, 3.0), (15.0, 15.0, 3.0), (30.0, 30.0, 3.0), (20.0, 20.0, 4.0)]
PRIMARY = (20.0, 20.0, 3.0)
SIGNAL, SCORED = "signal", "scored events"  # the windows table's ``classification``
CLASSES = ox.CLASSES  # desaturating, arousal_only, unclassified, unknown (never from the scored events)
RATED = CLASSES[:3]  # unknown enters no rate
INDEX = {"desaturating": "logDESAT_c", "arousal_only": "logAROUSAL_c"}  # the two modelled types
COVARS = PREDICTORS[1:]  # age_c, gender_bin
STAGES = {"N2": 2, "N3": 3}
SCORER_STAGE = {"AWAKE": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 5}  # the event line's stage column
ORIG_SR = 200.0  # the original-sample index's rate
SPINDLE_BIN_S = 300.0  # fig-6 a's spindle strip
NOTE = ("Exploratory. N2/N3 hypopneas (hypnogram epoch of onset) classed from the oximeter signal (stats.oximetry; "
        "unknown = invalid or unassessable signal in the window, in neither rate), rates per hour of valid N2 + N3 "
        "SpO2; 'scored events': classed from the scored-event file, rates per hour of N2 + N3 sleep (tst x (n2 + n3) "
        "/ 100). log10(rate + 1) centred over the children analysed; OLS with age_c and gender_bin; cluster tests as "
        "stats.clusters (two-sided p < .05 cluster-forming threshold, 5,000 Freedman-Lane permutations, seed "
        "20260607). Joint model: both rates; 'difference' = Wald test of beta_desaturating - beta_arousal_only.")
EVENT_INPUTS = [DATA_ROOT / "sub-*/psg/*_scored*events.txt", DATA_ROOT / "sub-*/psg/*_hypnogram.txt",
                DATA_ROOT / "sub-*/eeg/*N2N3*clean.set"]


def counted(ev: pd.DataFrame) -> pd.Series:
    """The hypopneas the analysis counts: those whose onset epoch is N2 or N3."""
    return (ev["kind"] == "hypopnea") & ev["stage"].isin(STAGES.values())


def classify_scored(ev: pd.DataFrame, desat_s: float, arousal_s: float, drop: float) -> np.ndarray:
    """Scored-event class of each counted hypopnea row of ``ev`` (in order)."""
    h = ev[counted(ev)]
    ds = ev.loc[(ev["kind"] == "desat") & (ev["spo2_drop"] >= drop), "t_start"].to_numpy()
    ar = ev.loc[ev["kind"] == "arousal", "t_start"].to_numpy()
    on, off = h["t_start"].to_numpy()[:, None], h["t_end"].to_numpy()[:, None]
    d = ((ds >= on) & (ds <= off + desat_s)).any(axis=1)
    a = ((ar >= on) & (ar <= off + arousal_s)).any(axis=1)
    return np.where(d, "desaturating", np.where(a, "arousal_only", "unclassified"))


def classify_signal(ev: pd.DataFrame, sig: pd.DataFrame, assess: np.ndarray, desat_s: float,
                    arousal_s: float) -> np.ndarray:
    """Signal class of each counted hypopnea row of ``ev`` (in order); ``sig``, ``assess`` from ``oximetry.detect``."""
    h = ev[counted(ev)]
    return ox.classify(h["t_start"], h["t_end"], ev.loc[ev["kind"] == "arousal", "t_start"], sig, assess,
                       desat_s, arousal_s)


def child(sid: int, n_ch: int) -> tuple[pd.DataFrame, np.ndarray, dict, dict[str, np.ndarray], pd.DataFrame]:
    """(scored events, hypnogram, validation row, per-channel mean fast duration for all / N2 / N3 detections,
    the fast detections with their recording time ``t_s`` and hypnogram ``stage``)."""
    f = psg.paths(sid)
    info = psg.read_set(f.set)
    hyp = psg.read_hypnogram(f.hypnogram)
    ev = psg.read_events(f.events, info.start, info.rec_len_s)
    e = np.floor(ev["t_start"].to_numpy() / 30).astype(np.int64)
    ev["stage"] = np.where((e >= 0) & (e < len(hyp)), hyp[np.clip(e, 0, len(hyp) - 1)], -9)
    h = ev["kind"] == "hypopnea"
    assert (ev.loc[h, "stage"] == ev.loc[h, "stage_file"].map(SCORER_STAGE)).all(), f"{sid}: hypopnea stages disagree"
    sp = psg.read_spindles(f.spindles)
    k = np.clip(np.round(sp["Start"].to_numpy() * info.srate).astype(np.int64), 0, info.n - 1)
    t_rec = (info.idx[k] - 1) / ORIG_SR  # s from the recording start
    epoch = (t_rec // 30).astype(np.int64)
    stage = np.where(epoch < len(hyp), hyp[np.clip(epoch, 0, len(hyp) - 1)], -9)
    fq = sp["Frequency"].to_numpy()
    fast = (fq >= FAST_RANGE[0]) & (fq <= FAST_RANGE[1])
    dur = {}
    for name, keep in (("all", fast), ("N2", fast & (stage == 2)), ("N3", fast & (stage == 3))):
        dur[name] = sp[keep].groupby("IdxChannel")["Duration"].mean().reindex(range(n_ch)).to_numpy(float)
    check = {"id": sid, "set_srate": info.srate, "n_detections": len(sp),
             "pct_detections_n2n3": 100 * float(np.isin(stage, (2, 3)).mean()),
             "n_fast": int(fast.sum()), "n_fast_n2": int((fast & (stage == 2)).sum()),
             "n_fast_n3": int((fast & (stage == 3)).sum()), "end_gap_s": info.n / info.srate - float(sp["End"].max()),
             "pct_events_epoch_agrees": 100 * float((ev["epoch_file"] == ev["epoch_computed"]).mean())}
    return ev, hyp, check, dur, sp[fast].assign(t_s=t_rec[fast], stage=stage[fast])


def add_indices(cov: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    out = cov.copy()
    for c, col in INDEX.items():
        x = np.log10(rates[f"{c}_per_h"].to_numpy(float) + 1.0)
        out[col] = x - x.mean()
    return out


def class_rates(classes: dict[int, np.ndarray], hours: pd.Series) -> pd.DataFrame:
    """Per child: counted hypopneas, the count of each class and each rated class per hour of ``hours``."""
    out = pd.DataFrame([{"id": sid, "n_hypopneas": len(cls), **{f"n_{c}": int((cls == c).sum()) for c in CLASSES}}
                        for sid, cls in classes.items()])
    for c in RATED:
        out[f"{c}_per_h"] = out[f"n_{c}"] / hours.loc[out["id"]].to_numpy()
    return out


def roi_models(y: np.ndarray, cov: pd.DataFrame, outcome: str) -> list[dict]:
    """Each index alone, then both in one model, and the joint model's beta_desaturating - beta_arousal_only, on the
    cluster mean ``y``."""
    import statsmodels.api as sm

    rows = []
    for model, terms in [*((f"{c} alone", [INDEX[c]]) for c in INDEX), ("joint", list(INDEX.values()))]:
        X = sm.add_constant(cov[terms + COVARS].astype(float))
        ok = np.isfinite(y)
        fit = sm.OLS(y[ok], X[ok]).fit()
        for c, term in INDEX.items():
            if term in terms:
                rows.append({"outcome": outcome, "model": "alone" if model != "joint" else "joint", "index": c,
                             "term": term, **ols_row(fit, term), "n": int(fit.nobs)})
    term = " - ".join(INDEX.values())
    w = fit.t_test(f"{term} = 0")  # the joint fit
    lo, hi = w.conf_int()[0]
    rows.append({"outcome": outcome, "model": "joint", "index": "difference", "term": term,
                 "beta": float(w.effect[0]), "SE": float(w.sd[0, 0]), "t": float(w.tvalue[0, 0]),
                 "p": float(w.pvalue), "CI_lower": float(lo), "CI_upper": float(hi), "n": int(fit.nobs)})
    return rows


def cluster_tests(dm: np.ndarray, cov: pd.DataFrame, adjacency: np.ndarray,
                  joint: bool = False) -> dict[str, tuple[dict | None, np.ndarray, int]]:
    """Each type alone, or (``joint``) both in one model with each in turn the effect and the other a covariate."""
    out = {}
    for c, col in INDEX.items():
        other = [v for v in INDEX.values() if v != col] if joint else []
        clusters, tmap, _ = cluster_permutation_general(dm, cov, adjacency, [col, *other, *COVARS])
        out[c] = (clusters[0] if clusters else None, tmap, len(clusters))
    return out


def cluster_row(c: str, model: str, top: dict | None, n_formed: int, n_children: int, hi_cluster) -> dict:
    return {"index": c, "model": model, "effect": INDEX[c], "n_children": n_children,
            "n_channels": top["n"] if top else 0,
            "cluster_mass": round(top["mass"], 4) if top else np.nan,
            "peak_t": round(top["peak_t"], 4) if top else np.nan,
            "p_value": round(top["p"], 6) if top else np.nan, "n_clusters_formed": n_formed,
            "n_shared_with_hi_cluster": len(set(top["channels"]) & set(hi_cluster)) if top else 0,
            "channels_1based": fmt_channels(top["channels"]) if top else ""}


def evaluate(rates: pd.DataFrame, cov: pd.DataFrame, roi: np.ndarray, dm: np.ndarray, adjacency: np.ndarray) -> dict:
    """One rule's analyses on the children of ``rates`` (rows aligned with ``cov``, ``roi`` and ``dm``)."""
    cv = add_indices(cov, rates)
    return {"cv": cv, "r": float(np.corrcoef(cv[INDEX["desaturating"]], cv[INDEX["arousal_only"]])[0, 1]),
            "roi": roi_models(roi, cv, "cluster mean, N2+N3"), "alone": cluster_tests(dm, cv, adjacency),
            "joint": cluster_tests(dm, cv, adjacency, joint=True)}


def rule_rows(classification: str, window, primary: bool, rates: pd.DataFrame, res: dict) -> list[dict]:
    """The windows table's two rows (one per modelled type) of one rule."""
    def roi(index, model):
        return next(x for x in res["roi"] if x["index"] == index and x["model"] == model)

    d = roi("difference", "joint")
    rows = []
    for c in INDEX:
        alone, joint = roi(c, "alone"), roi(c, "joint")
        top, jtop = res["alone"][c][0], res["joint"][c][0]
        rows.append({"classification": classification, "desat_after_end_s": window[0],
                     "arousal_after_end_s": window[1], "min_drop_pct": window[2], "primary": primary, "index": c,
                     "n_class": int(rates[f"n_{c}"].sum()), "n_unclassified": int(rates["n_unclassified"].sum()),
                     "n_unknown": int(rates["n_unknown"].sum()), "n_hypopneas": int(rates["n_hypopneas"].sum()),
                     "n_children": len(rates), "r_log_indices": res["r"],
                     "beta_alone": alone["beta"], "CI_lower_alone": alone["CI_lower"],
                     "CI_upper_alone": alone["CI_upper"], "p_alone": alone["p"],
                     "beta_joint": joint["beta"], "CI_lower_joint": joint["CI_lower"],
                     "CI_upper_joint": joint["CI_upper"], "p_joint": joint["p"],
                     "diff_joint": d["beta"], "SE_diff_joint": d["SE"], "p_diff_joint": d["p"],
                     "cluster_n_channels": top["n"] if top else 0, "cluster_p": top["p"] if top else np.nan,
                     "joint_cluster_n_channels": jtop["n"] if jtop else 0,
                     "joint_cluster_p": jtop["p"] if jtop else np.nan})
    return rows


def main() -> None:
    channels = load_channel_info()
    n_ch = len(channels.labels)
    adjacency = adjacency_matrix(channels)
    cov_all = load_covariates()

    # the corrected fast-duration HI cluster, as stats.rois derives it (all 62 children)
    _, fast62, mins62 = spindle_matrices(cov_all, n_ch)
    cluster = largest_cluster_channels(data_matrix_for_metric(fast62, "Duration", mins62), cov_all, adjacency, PREDICTORS)

    ids = [i for i in sample_ids() if psg.paths(i).events]
    example = psg.example_child("fig-6")
    print(f"{len(ids)} of {len(sample_ids())} children have a scored-event file; cluster {len(cluster)} channels")
    cov = cov_all.set_index("id").loc[ids].reset_index()
    by_id = cov.set_index("id")
    n2n3_h = by_id["tst"].astype(float) * (by_id["n2"] + by_id["n3"]) / 100.0 / 60.0  # hours of N2 + N3 sleep

    drops = sorted({w[2] for w in WINDOWS})
    events, signal, valid_h, checks, dur = {}, {}, {}, [], {s: [] for s in ("all", "N2", "N3")}
    for sid in ids:
        ev, hyp, check, d, fast_sp = child(sid, n_ch)
        night = ox.load(sid, ev)
        events[sid], valid_h[sid] = ev, night["valid_n2n3_h"]
        signal[sid] = {drop: ox.detect(night["s1"], drop) for drop in drops}
        checks.append(check)
        for s, values in dur.items():
            values.append(d[s])
        if sid == example:
            hyp_example, sp_example = hyp, fast_sp
    valid_h = pd.Series(valid_h)
    checks = pd.DataFrame(checks)
    with warnings.catch_warnings():  # a child without an N3 fast spindle in the cluster has no N3 mean
        warnings.simplefilter("ignore", RuntimeWarning)
        cluster_mean = {s: np.nanmean(np.vstack(dur[s])[:, cluster], axis=1) for s in dur}

    # the detections reproduce roi_values.csv and the metrics table
    roi = pd.read_csv(ROI_PATH).set_index("id").loc[ids, "fast_duration_cluster"].to_numpy(float)
    assert np.allclose(cluster_mean["all"], roi, rtol=0, atol=1e-9), "cluster mean differs from roi_values.csv"
    _, fast, mins = spindle_matrices(cov, n_ch)
    dm = data_matrix_for_metric(fast, "Duration", mins)
    assert np.allclose(np.vstack(dur["all"]), dm, equal_nan=True, atol=1e-9), "channel durations differ from METRICS_LONG"

    window_rows, model_rows, cluster_rows = [], [], []
    for classification, window in [*((SIGNAL, w) for w in WINDOWS), (SCORED, PRIMARY)]:
        if classification == SIGNAL:
            classes = {sid: classify_signal(ev, *signal[sid][window[2]], *window[:2]) for sid, ev in events.items()}
            rates = class_rates(classes, valid_h)
        else:
            rates = class_rates({sid: classify_scored(ev, *window) for sid, ev in events.items()}, n2n3_h)
        res = evaluate(rates, cov, roi, dm, adjacency)
        primary = classification == SIGNAL and window == PRIMARY
        window_rows += rule_rows(classification, window, primary, rates, res)
        if primary:
            primary_classes = classes
            children = rates.merge(cov[["id", "overall_hi", "tst"]].rename(columns={"overall_hi": "hi", "tst": "tst_min"}),
                                   on="id").assign(n2n3_h=lambda d: n2n3_h.loc[d["id"]].to_numpy(),
                                                   valid_n2n3_h=lambda d: valid_h.loc[d["id"]].to_numpy(),
                                                   n_hypopneas_night=lambda d: [int((events[i]["kind"] == "hypopnea").sum())
                                                                                for i in d["id"]]).merge(checks, on="id")
            cv = res["cv"]
            model_rows = res["roi"] + roi_models(cluster_mean["N2"], cv, "cluster mean, N2 spindles") \
                + roi_models(cluster_mean["N3"], cv, "cluster mean, N3 spindles") \
                + roi_models(cluster_mean["N2"] - cluster_mean["N3"], cv, "cluster mean, N2 minus N3 spindles")
            tmaps = {"channel": np.arange(1, n_ch + 1), "label": channels.labels}
            for c, (top, tmap, n_formed) in res["alone"].items():
                tmaps[f"t_{c}"] = tmap
                tmaps[f"cluster_{c}"] = np.isin(np.arange(n_ch), top["channels"] if top else []).astype(int)
                cluster_rows.append(cluster_row(c, "alone", top, n_formed, len(ids), cluster))
            tmaps["hi_cluster"] = np.isin(np.arange(n_ch), cluster).astype(int)  # fig-4's fast-duration HI cluster
            for c, (top, tmap, n_formed) in res["joint"].items():  # fig-6 c draws these: each type adjusted for the other
                tmaps[f"t_joint_{c}"] = tmap
                tmaps[f"cluster_joint_{c}"] = np.isin(np.arange(n_ch), top["channels"] if top else []).astype(int)
                cluster_rows.append(cluster_row(c, "joint", top, n_formed, len(ids), cluster))
        w = window_rows[-2:]
        print(f"{classification} {window}: {w[0]['n_class']}/{w[1]['n_class']}/{w[0]['n_unclassified']}/"
              f"{w[0]['n_unknown']} (desat/arousal/unclassified/unknown); r = {res['r']:.2f}; diff p {w[0]['p_diff_joint']:.3f}; "
              + "; ".join(f"{x['index']} alone {x['beta_alone']:+.3f} (p {x['p_alone']:.3f}) joint {x['beta_joint']:+.3f} "
                          f"(p {x['p_joint']:.3f}) cluster {x['cluster_n_channels']} ch p {x['cluster_p']:.3f}, joint "
                          f"{x['joint_cluster_n_channels']} ch p {x['joint_cluster_p']:.3f}" for x in w))

    low = checks[checks["pct_detections_n2n3"] < 99]
    print(f"stage mapping: min {checks['pct_detections_n2n3'].min():.2f}% of detections in N2/N3"
          + (f"; below 99%: {low['id'].tolist()}" if len(low) else "; every child >= 99%"))
    print(f"cleaned-file end minus last detection end: {checks['end_gap_s'].min():.2f} to {checks['end_gap_s'].max():.2f} s")
    print(f"fast detections in N2 {checks['n_fast_n2'].sum():,}, in N3 {checks['n_fast_n3'].sum():,}")

    ensure_dir(TABLE_DIR)
    ensure_dir(FIGURE_DATA)
    inputs = [*matrix_inputs(), ROI_PATH, RESULTS_PATH, *EVENT_INPUTS, *ox.SIGNAL_INPUTS]
    outputs = {TABLE_DIR / "hypopnea_type_children.csv": children,
               TABLE_DIR / "hypopnea_type_models.csv": pd.DataFrame(model_rows),
               TABLE_DIR / "hypopnea_type_clusters.csv": pd.DataFrame(cluster_rows),
               TABLE_DIR / "hypopnea_type_windows.csv": pd.DataFrame(window_rows),
               FIGURE_DATA / "hypopnea_type_tmaps.csv": pd.DataFrame(tmaps)}
    for path, table in outputs.items():
        table.to_csv(path, index=False, float_format="%.10g")
        record(path, script=__file__, inputs=inputs, note=NOTE)

    # the example child's night for fig-6 a: hypnogram and the events that classify, on the recording clock
    ev = events[example]
    hyp_at = lambda t: hyp_example[np.clip((np.asarray(t) // 30).astype(np.int64), 0, len(hyp_example) - 1)]  # noqa: E731
    sig = signal[example][PRIMARY[2]][0]
    shown = pd.concat([ev[counted(ev)].assign(cls=primary_classes[example]),
                       pd.DataFrame({"kind": "desat", "label": "signal desaturation", "stage": hyp_at(sig["onset_s"]),
                                     "t_start": sig["onset_s"], "t_end": sig["end_s"], "spo2_drop": sig["drop"]}),
                       ev[ev["kind"] == "arousal"]])
    shown = shown[["kind", "cls", "label", "stage", "t_start", "t_end", "spo2_drop"]].sort_values("t_start", kind="stable")
    f = psg.paths(example)
    sp = sp_example[sp_example["IdxChannel"].isin(cluster) & sp_example["stage"].isin(STAGES.values())]
    edges = np.arange(0.0, 30.0 * len(hyp_example) + SPINDLE_BIN_S, SPINDLE_BIN_S)
    spindles = pd.DataFrame({"t_s": edges[:-1], "n_fast_cluster": np.histogram(sp["t_s"], edges)[0],
                             "child_mean_duration_s": cluster_mean["all"][ids.index(example)], "id": example})
    for name, table in (("hypnogram", pd.DataFrame({"t_s": 30.0 * np.arange(len(hyp_example)), "code": hyp_example})),
                        ("events", shown), ("spindles", spindles)):
        path = FIGURE_DATA / f"hypopnea_type_example_{name}.csv"
        table.to_csv(path, index=False, float_format="%.6g")
        record(path, script=__file__, inputs=[f.events, f.hypnogram, f.set, f.spindles, *ox.SIGNAL_INPUTS],
               note="fig-6's example child (stats.psg.example_child); times in s from the recording start; N2/N3 "
                    "hypopneas only, signal classes at the primary rule; desat = signal desaturations (>= 3 %; "
                    "spo2_drop = drop from baseline); "
                    "stage = hypnogram code of the onset epoch. spindles: "
                    f"fast detections in N2/N3 on the {len(cluster)} cluster channels, counted (over channels) per "
                    f"{SPINDLE_BIN_S:g}-s bin starting at t_s, timed by cleaned sample -> original-sample index; "
                    "child_mean_duration_s is the child's cluster-mean fast duration (roi_values.csv "
                    "fast_duration_cluster), id its id.")
    print("[SAVED] " + ", ".join(p.name for p in outputs) + ", hypopnea_type_example_{hypnogram,events,spindles}.csv")


if __name__ == "__main__":
    main()
