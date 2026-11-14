"""Pulse-oximetry quality from the raw oximeter signal, and whether the oxygenation measures survive it.

The signal, its validity rules and the desaturation detector are ``stats.oximetry``'s (its docstring); the hypopnea
classes ``stats.hypopnea_type``'s. Sleep = hypnogram N1/N2/N3/REM epochs; TST and N2+N3 hours are counted from the
hypnogram. Invalid segments >= 30 s (``SEGMENT_S``; runs of consecutive invalid seconds touching sleep) are counted as
sensor-off episodes.

Agreement with the scored 'SpO2 Desat' events (drop >= 3 %, onset epoch in sleep): a scored event is detected when a
signal desaturation overlaps [start - TOL_S, end + TOL_S], TOL_S = 10 s; a signal event in sleep is confirmed when it
overlaps a scored desaturation the same way. Signal ODI = signal events with onset in sleep per hour of valid sleep;
the scored ODI is ``oxygenation_derived.csv`` ``odi`` (per hour of TST, as the paper uses it).

Nadir: lowest valid 1-s SpO2 in sleep and the 1st percentile of valid 10-Hz samples in sleep, vs the clinical
report nadir (``oxygenation_reported.csv``) and the lowest scored desaturation (``oxygenation_derived.csv``). For the
report nadir, the seconds whose lowest raw sample (> 0) reads as it (rounded to the integer) are located (sleep first)
and their validity and reasons reported; ``suspect`` (suspected artifact): a report nadir more than SUSPECT_GAP
points below the lowest valid 1-s SpO2 in sleep.

Hypopnea classes: each counted hypopnea (N2/N3 onset) at ``stats.hypopnea_type.PRIMARY``, scored-event class by
signal class (the primary), cross-tabulated.

Models: the primary signal analysis of ``stats.hypopnea_type`` (``evaluate``, the windows table's columns) without
the children whose N2+N3 SpO2 is < 80 % valid; the full-sample signal rates are asserted equal to
hypopnea_type_children.csv. ODI sensitivity: the fast-duration HI cluster test (HI + age + sex) with the signal ODI
added, beside the same-n base and the scored ODI (asserted equal to covariate_sensitivity.csv), as
``stats.covariate_sensitivity`` does.

Inputs : the oximetry files (``stats.oximetry``), as stats.hypopnea_type (scored events, hypnograms, cleaned .set,
         detections, COV_PATH, ROI_PATH, METRICS_LONG), oxygenation_derived.csv, oxygenation_reported.csv,
         hypopnea_type_children.csv, covariate_sensitivity.csv.
Output : results/tables/oximetry_quality_children.csv, oximetry_desat_agreement.csv, oximetry_nadir.csv,
         oximetry_hypopnea_reclass.csv, oximetry_hypopnea_models.csv, oximetry_odi_clusters.csv;
         results/figure_data/oximetry_example_trace.csv (10-s bins) and oximetry_example_intervals.csv.
Run    : cd code && uv run python -m stats.oximetry_qc      (after hypopnea_type and covariate_sensitivity)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from project import FIGURE_DATA, ROI_PATH, TABLE_DIR, ensure_dir, sample_ids
from provenance import record
from stats import hypopnea_type as ht
from stats import oximetry as ox
from stats import psg
from stats.clusters import adjacency_matrix, cluster_permutation_general, fmt_channels
from stats.covariate_sensitivity import attach_covariates
from stats.io import PREDICTORS, data_matrix_for_metric, load_channel_info, load_covariates, matrix_inputs, spindle_matrices
from stats.oximetry import (
    BASELINE_S,
    DROP,
    FLAGS,
    FLAT_CEILING,
    FLAT_S,
    FS,
    LOW_PERF,
    MIN_DESAT_S,
    PAD_S,
    PLETH_FLAT_MV,
    PLETH_LIVE_MV,
    REASONS,
    SIGNAL_INPUTS,
    SLEEP,
    STEP_PCT,
    STEP_PLATEAU_S,
    overlaps,
    runs,
)

SEGMENT_S = 30
TOL_S = 10
VALID_MIN_PCT = 80.0
SUSPECT_GAP = 5.0  # % points; the other report nadirs lie within 3 of the valid-signal nadir
BIN_S = 10  # example traces
LT80 = "signal, excluding < 80% valid"

OXY_DERIVED = TABLE_DIR / "oxygenation_derived.csv"
OXY_REPORTED = TABLE_DIR / "oxygenation_reported.csv"
HT_CHILDREN = TABLE_DIR / "hypopnea_type_children.csv"
COVSENS = TABLE_DIR / "covariate_sensitivity.csv"
OUT = {k: TABLE_DIR / f"oximetry_{k}.csv" for k in ("quality_children", "desat_agreement", "nadir", "hypopnea_reclass",
                                                     "hypopnea_models", "odi_clusters")}
TRACE = FIGURE_DATA / "oximetry_example_trace.csv"
INTERVALS = FIGURE_DATA / "oximetry_example_intervals.csv"


def child(sid: int, n_ch: int) -> dict:
    ev = ht.child(sid, n_ch)[0]
    c = ox.load(sid, ev)
    c["sig"], c["assess"] = ox.detect(c["s1"])
    return c | {"ev": ev}


def quality_row(c: dict) -> dict:
    st, inv, masks = c["stage"], c["invalid"], c["masks"]
    sleep, n2n3 = np.isin(st, SLEEP), np.isin(st, (2, 3))
    row = {"id": c["id"], "edf_start": c["edf_start"], "rec_h": c["n"] / 3600, "tst_h": sleep.sum() / 3600,
           "n2n3_h": n2n3.sum() / 3600, "valid_tst_h": (sleep & ~inv).sum() / 3600,
           "valid_n2n3_h": (n2n3 & ~inv).sum() / 3600}
    row["valid_tst_pct"] = 100 * row["valid_tst_h"] / row["tst_h"]
    row["valid_n2n3_pct"] = 100 * row["valid_n2n3_h"] / row["n2n3_h"]
    left = n2n3.copy()
    for r in REASONS:  # exclusive minutes, first reason wins (stack order)
        row[f"n2n3_invalid_min_{r}"] = (left & masks[r]).sum() / 60
        left &= ~masks[r]
    for r in REASONS + FLAGS:  # any-reason minutes (overlap)
        row[f"n2n3_any_min_{r}"] = (n2n3 & masks[r]).sum() / 60
    row["n2n3_invalid_min"] = (n2n3 & inv).sum() / 60
    row["tst_invalid_min"] = (sleep & inv).sum() / 60
    segs = [(a, b) for a, b in runs(inv) if b - a >= SEGMENT_S and sleep[a:b].any()]
    row["n_invalid_segments_30s"] = len(segs)
    row["invalid_segments_30s_min"] = sum(b - a for a, b in segs) / 60
    row["longest_invalid_segment_min"] = max((b - a for a, b in segs), default=0) / 60
    row["below_80pct_valid_n2n3"] = row["valid_n2n3_pct"] < VALID_MIN_PCT
    row["pleth_p2p_median_mV"] = c["p2p_med"]
    row["pleth_live"] = c["p2p_med"] >= PLETH_LIVE_MV
    sig = c["sig"]
    on = sig["onset_s"].to_numpy(int)
    row["n_signal_desats_sleep"] = int(sleep[on].sum()) if len(on) else 0
    row["odi_signal"] = row["n_signal_desats_sleep"] / row["valid_tst_h"]
    return row


def agreement_row(c: dict) -> dict:
    ev, sig, st, inv = c["ev"], c["sig"], c["stage"], c["invalid"]
    sleep = np.isin(st, SLEEP)
    sc = ev[(ev["kind"] == "desat") & (ev["spo2_drop"] >= DROP) & ev["stage"].isin(SLEEP)]
    all_sc = ev[ev["kind"] == "desat"]
    on = sig["onset_s"].to_numpy(int)
    sig_sleep = sig[sleep[on]] if len(on) else sig
    hit = overlaps(sc["t_start"] - TOL_S, sc["t_end"] + TOL_S, sig["onset_s"], sig["end_s"]).any(axis=1) if len(sig) else np.zeros(len(sc), bool)
    valid_sc = np.array([not inv[max(int(a), 0):int(np.ceil(b)) + 1].any() for a, b in zip(sc["t_start"], sc["t_end"])], bool)
    conf = overlaps(sig_sleep["onset_s"], sig_sleep["end_s"], all_sc["t_start"] - TOL_S, all_sc["t_end"] + TOL_S).any(axis=1) \
        if len(all_sc) else np.zeros(len(sig_sleep), bool)
    # matched pairs: onset lag and nadir difference for scored events in valid signal
    lags, dnadir = [], []
    for (_, e), h, v in zip(sc.iterrows(), hit, valid_sc):
        if h and v:
            k = np.flatnonzero(overlaps([e["t_start"] - TOL_S], [e["t_end"] + TOL_S], sig["onset_s"], sig["end_s"])[0])[0]
            lags.append(sig["onset_s"].iloc[k] - e["t_start"])
            dnadir.append(sig["nadir"].iloc[k] - e["spo2_nadir"])
    n, nv = len(sc), int(valid_sc.sum())
    return {"id": c["id"], "n_scored": n, "n_scored_valid": nv, "n_scored_detected": int(hit.sum()),
            "n_scored_valid_detected": int((hit & valid_sc).sum()),
            "sensitivity": hit.mean() if n else np.nan, "sensitivity_valid": (hit & valid_sc).sum() / nv if nv else np.nan,
            "n_signal": len(sig_sleep), "n_signal_confirmed": int(conf.sum()),
            "ppv": conf.mean() if len(sig_sleep) else np.nan,
            "median_onset_lag_s": float(np.median(lags)) if lags else np.nan,
            "median_nadir_diff_pct": float(np.median(dnadir)) if dnadir else np.nan,
            "pct_nadir_within_1": 100 * float(np.mean(np.abs(dnadir) <= 1)) if dnadir else np.nan}


def nadir_row(c: dict, report: float, scored: float) -> dict:
    st, inv, S, s1, masks = c["stage"], c["invalid"], c["S"], c["s1"], c["masks"]
    sleep = np.isin(st, SLEEP)
    vs = sleep & ~inv
    signal_nadir = float(np.round(np.nanmin(s1[vs]))) if vs.any() else np.nan  # integer readout
    row = {"id": c["id"], "suspect": bool(report < signal_nadir - SUSPECT_GAP), "report_nadir": report,
           "scored_nadir": scored, "signal_nadir_valid": signal_nadir,
           "signal_nadir_valid_t_s": float(np.flatnonzero(vs)[np.nanargmin(s1[vs])]) if vs.any() else np.nan,
           "signal_p1_valid": float(np.percentile(S[vs], 1)) if vs.any() else np.nan,
           "signal_min_raw_sleep": float(np.min(np.where(S[sleep] > 0, S[sleep], np.inf))) if sleep.any() else np.nan}
    if np.isfinite(report):
        low = np.where(S > 0, S, np.inf).min(axis=1)
        near = np.round(low) == report  # the integer readout
        cand = np.flatnonzero(near & sleep) if (near & sleep).any() else np.flatnonzero(near)
        row["n_s_at_report_nadir"] = int(near.sum())
        row["n_s_at_report_nadir_sleep"] = int((near & sleep).sum())
        row["n_s_at_report_nadir_valid"] = int((near & ~inv).sum())
        if len(cand):
            t = int(cand[0])
            w = slice(max(t - PAD_S, 0), t + PAD_S + 1)
            row |= {"report_nadir_t_s": float(t), "report_nadir_stage": int(st[t]), "valid_at_report_nadir": not inv[t],
                    "reasons_at_report_nadir": ";".join(r for r in REASONS if masks[r][w].any())}
        row["n_s_below_report_nadir_valid_sleep"] = int((vs & (np.round(s1) < report)).sum())
    return row


def reclass(c: dict) -> pd.DataFrame:
    """Each counted hypopnea's scored-event and signal class at the primary rule."""
    ev = c["ev"]
    h = ev[ht.counted(ev)]
    return pd.DataFrame({"id": c["id"], "t_start": h["t_start"], "t_end": h["t_end"],
                         "scored_class": ht.classify_scored(ev, *ht.PRIMARY),
                         "signal_class": ht.classify_signal(ev, c["sig"], c["assess"], *ht.PRIMARY[:2])})


def cluster_fields(top: dict | None, n_formed: int) -> dict:
    return {"n_channels": top["n"] if top else 0, "cluster_mass": top["mass"] if top else np.nan,
            "peak_t": top["peak_t"] if top else np.nan, "p_cluster": top["p"] if top else np.nan,
            "n_clusters_formed": n_formed, "channels_1based": fmt_channels(top["channels"]) if top else ""}


def main() -> None:
    channels = load_channel_info()
    n_ch = len(channels.labels)
    adjacency = adjacency_matrix(channels)
    ids = [i for i in sample_ids() if psg.paths(i).events]
    cov_all = load_covariates()
    cov = cov_all.set_index("id").loc[ids].reset_index()
    der = pd.read_csv(OXY_DERIVED).set_index("id")
    rep = pd.read_csv(OXY_REPORTED).set_index("id")

    quality, agree, nadir, recl, kept, classes = [], [], [], [], {}, {}
    for sid in ids:
        c = child(sid, n_ch)
        q = quality_row(c) | {"odi_scored": der.loc[sid, "odi"], "n_scored_desats_sleep": der.loc[sid, "n_desat_sleep"]}
        quality.append(q)
        agree.append(agreement_row(c))
        nadir.append(nadir_row(c, rep.loc[sid, "nadir_pct"], der.loc[sid, "spo2_nadir_sleep"]))
        recl.append(reclass(c))
        classes[sid] = recl[-1]["signal_class"].to_numpy()
        kept[sid] = {k: c[k] for k in ("ev", "stage", "masks", "invalid", "S", "p2p", "pulse", "sig")}
        print(f"{sid:03d}: valid N2+N3 {q['valid_n2n3_pct']:.1f}%, ODI signal {q['odi_signal']:.2f} vs scored "
              f"{q['odi_scored']:.2f}; desat sens {agree[-1]['sensitivity']:.2f} ppv {agree[-1]['ppv']:.2f}", flush=True)
    quality, agree, nadir, recl = pd.DataFrame(quality), pd.DataFrame(agree), pd.DataFrame(nadir), pd.concat(recl)

    # cross-tab scored vs signal class
    xt = recl.groupby(["scored_class", "signal_class"]).size().rename("n").reset_index()
    xt["pct_of_scored_class"] = 100 * xt["n"] / xt.groupby("scored_class")["n"].transform("sum")
    xt = xt.sort_values(["scored_class", "signal_class"])

    # ------------------------------------------------------------------ the primary signal analysis without < 80 % valid
    q = quality.set_index("id").loc[ids]
    rates = ht.class_rates(classes, q["valid_n2n3_h"])
    published = pd.read_csv(HT_CHILDREN).set_index("id").loc[ids]
    cols = [f"n_{k}" for k in ht.CLASSES] + [f"{k}_per_h" for k in ht.RATED]
    assert np.allclose(rates[cols].to_numpy(float), published[cols].to_numpy(float)), "signal rates != hypopnea_type_children"
    keep = ~q["below_80pct_valid_n2n3"].to_numpy(bool)
    roi = pd.read_csv(ROI_PATH).set_index("id").loc[ids, "fast_duration_cluster"].to_numpy(float)
    _, fast, mins = spindle_matrices(cov, n_ch)
    dm = data_matrix_for_metric(fast, "Duration", mins)
    kept_rates = rates[keep].reset_index(drop=True)
    res = ht.evaluate(kept_rates, cov[keep].reset_index(drop=True), roi[keep], dm[keep], adjacency)
    models = pd.DataFrame(ht.rule_rows(LT80, ht.PRIMARY, False, kept_rates, res)).assign(
        n_excluded=int((~keep).sum()), excluded_ids=";".join(f"{i:03d}" for i in np.array(ids)[~keep]))

    # ODI sensitivity: the fast-duration HI cluster test, all 62 children as stats.covariate_sensitivity
    cv62 = attach_covariates(cov_all).merge(quality[["id", "odi_signal"]], on="id", how="left")
    _, fast62, mins62 = spindle_matrices(cv62, n_ch)
    dm62 = data_matrix_for_metric(fast62, "Duration", mins62)
    rows = []
    for name, extra_cov in (("hi_base_same_n", None), ("hi_plus_odi_scored", "odi"), ("hi_plus_odi_signal", "odi_signal")):
        if extra_cov:
            preds, c62 = PREDICTORS + [extra_cov], cv62
        else:  # unadjusted, on the children with a signal ODI
            preds, c62 = PREDICTORS, cv62.copy()
            c62.loc[c62["odi_signal"].isna(), PREDICTORS] = np.nan
        clusters, _, _ = cluster_permutation_general(dm62, c62, adjacency, preds)
        n_used = int(c62[preds].notna().all(axis=1).sum())
        rows.append({"variant": name, "term": "logHI_c", "n": n_used, "added_covariate": extra_cov or "",
                     **cluster_fields(clusters[0] if clusters else None, len(clusters))})
    odi = pd.DataFrame(rows)

    # the scored-ODI row reproduces covariate_sensitivity.csv
    cs = pd.read_csv(COVSENS).query("metric_label == 'Fast duration' and added_covariate == 'odi'").iloc[0]
    o = odi.query("variant == 'hi_plus_odi_scored'").iloc[0]
    assert o["n_channels"] == cs["n_channels"] and np.isclose(o["p_cluster"], cs["p_corrected"], atol=1e-6), \
        "scored-ODI row != covariate_sensitivity"

    # ------------------------------------------------------------------ example nights: best, median, worst
    order = quality.sort_values("valid_n2n3_pct", ascending=False)["id"].tolist()
    examples = {"good": order[0], "mediocre": order[len(order) // 2], "worst": order[-1]}
    traces, intervals = [], []
    for role, sid in examples.items():
        k = kept[sid]
        nb = len(k["stage"]) // BIN_S
        sl = slice(0, nb * BIN_S)
        inv = k["invalid"][sl].reshape(nb, BIN_S)
        S = k["S"][sl].reshape(nb, BIN_S * FS)
        Sv = np.where(np.repeat(inv, FS, axis=1), np.nan, S)
        with warnings.catch_warnings():  # an all-invalid bin has no valid mean
            warnings.simplefilter("ignore", RuntimeWarning)
            traces.append(pd.DataFrame({
                "id": sid, "role": role, "t_s": BIN_S * np.arange(nb), "stage": k["stage"][sl][::BIN_S],
                "spo2_mean_valid": np.nanmean(Sv, axis=1), "spo2_min_valid": np.nanmin(Sv, axis=1),
                "spo2_mean_raw": S.mean(axis=1), "spo2_min_raw": S.min(axis=1), "valid_frac": 1 - inv.mean(axis=1),
                "pleth_p2p_median_mV": np.median(k["p2p"][sl].reshape(nb, BIN_S), axis=1),
                "pulse_mean_bpm": k["pulse"][sl].reshape(nb, BIN_S).mean(axis=1)}))
        ev = k["ev"]
        for kind, sel in (("scored_desat", (ev["kind"] == "desat") & (ev["spo2_drop"] >= DROP)),
                          ("scored_artifact", ev["label"].str.lower().str.contains("spo2 artifact")),
                          ("hypopnea_n2n3", ht.counted(ev))):
            e = ev[sel]
            intervals.append(pd.DataFrame({"id": sid, "role": role, "kind": kind, "t_start": e["t_start"],
                                           "t_end": e["t_end"], "value": e["spo2_drop"], "reason": ""}))
        s = k["sig"]
        intervals.append(pd.DataFrame({"id": sid, "role": role, "kind": "signal_desat", "t_start": s["onset_s"],
                                       "t_end": s["end_s"], "value": s["drop"], "reason": ""}))
        for a, b in runs(k["invalid"]):
            if b - a >= SEGMENT_S:
                why = ";".join(r for r in REASONS if k["masks"][r][a:b].any())
                intervals.append(pd.DataFrame([{"id": sid, "role": role, "kind": "invalid_segment_30s", "t_start": float(a),
                                                "t_end": float(b), "value": np.nan, "reason": why}]))
    traces, intervals = pd.concat(traces), pd.concat(intervals).sort_values(["id", "t_start"])

    # ------------------------------------------------------------------ summary
    v = quality["valid_n2n3_pct"]
    print(f"\nvalid N2+N3 SpO2: median {v.median():.1f}% (IQR {v.quantile(.25):.1f}-{v.quantile(.75):.1f}, min {v.min():.1f}); "
          f"< 80 %: {quality.loc[quality['below_80pct_valid_n2n3'], 'id'].tolist()}")
    print(f"desat agreement: sensitivity median {agree['sensitivity'].median():.2f} (valid {agree['sensitivity_valid'].median():.2f}), "
          f"PPV median {agree['ppv'].median():.2f}; pooled sens {agree['n_scored_detected'].sum() / agree['n_scored'].sum():.2f}, "
          f"pooled PPV {agree['n_signal_confirmed'].sum() / agree['n_signal'].sum():.2f}; onset lag median "
          f"{agree['median_onset_lag_s'].median():.1f} s; nadir diff median {agree['median_nadir_diff_pct'].median():.2f}")
    print(f"ODI signal vs scored: r = {quality[['odi_signal', 'odi_scored']].corr().iloc[0, 1]:.3f}")
    print(nadir[nadir["suspect"]].to_string(index=False))
    print(xt.to_string(index=False))
    print(models[["classification", "index", "n_children", "beta_joint", "p_joint", "p_diff_joint", "cluster_n_channels",
                  "cluster_p", "joint_cluster_n_channels", "joint_cluster_p"]].to_string(index=False))
    print(odi[["variant", "n", "n_channels", "p_cluster"]].to_string(index=False))

    ensure_dir(TABLE_DIR)
    ensure_dir(FIGURE_DATA)
    inputs = [*SIGNAL_INPUTS, *ht.EVENT_INPUTS, *matrix_inputs(), ROI_PATH, OXY_DERIVED, OXY_REPORTED, HT_CHILDREN,
              COVSENS]
    rules = (f"Validity per 1-s second: SpO2 <50 or >100 % (+-{PAD_S} s), pleth p2p < {PLETH_FLAT_MV} mV (+-{PAD_S} s), "
             f"exact-constant SpO2 >= {FLAT_S} s in [50, {FLAT_CEILING:g}) %, > {STEP_PCT} % change within 1 s (and the plateau between such a fall and rise <= {STEP_PLATEAU_S} s), scored SpO2 Artifact (pleth p2p < "
             f"{LOW_PERF:.0%} of the child's sleep median reported as low_perfusion, not invalidating); sleep from the hypnogram. Signal "
             f"desaturation: integer-rounded 3-s median >= {DROP} % below the max of the preceding {BASELINE_S} s for >= {MIN_DESAT_S} s "
             "(stats/oximetry.py docstring).")
    notes = {"quality_children": rules + " n2n3_invalid_min_* exclusive (first reason wins, column order); "
                                         "n2n3_any_min_* overlapping. odi_signal per valid sleep hour; odi_scored from "
                                         "oxygenation_derived.csv (per TST hour).",
             "desat_agreement": rules + f" Scored = 'SpO2 Desat' drop >= {DROP} % with onset epoch in sleep; matched at "
                                        f"+-{TOL_S} s overlap. sensitivity_valid: scored events whose interval is valid signal.",
             "nadir": rules + " report_nadir from oxygenation_reported.csv; scored_nadir = lowest sleep 'SpO2 Desat' SpO2. "
                              "*_at_report_nadir: seconds whose lowest raw sample (> 0), rounded, equals the report nadir "
                              f"(first in sleep); reasons over +-{PAD_S} s.",
             "hypopnea_reclass": rules + " Counted N2/N3 hypopneas, primary window; signal_class: desaturating "
                                         "(signal desaturation onset in the window) > unknown (any invalid or "
                                         "unassessable second) > arousal_only > unclassified.",
             "hypopnea_models": "stats.hypopnea_type's primary signal analysis (its windows table's columns: "
                                "each rate alone and both together on the cluster-mean fast duration, the Wald test "
                                "of their difference, and the channel-wise cluster tests alone and joint) without the "
                                f"children < {VALID_MIN_PCT:.0f} % valid N2+N3 SpO2 (excluded_ids).",
             "odi_clusters": "Fast-duration HI cluster test (HI + age + sex, 5,000 permutations, seed 20260607) with "
                             "the scored ODI (oxygenation_derived.csv) or the signal ODI added, and the same-n base "
                             "(children with a signal ODI)."}
    tables = {"quality_children": quality, "desat_agreement": agree, "nadir": nadir, "hypopnea_reclass": xt,
              "hypopnea_models": models, "odi_clusters": odi}
    for key, table in tables.items():
        table.to_csv(OUT[key], index=False, float_format="%.10g")
        record(OUT[key], script=__file__, inputs=inputs, note=notes[key])
    ex = ", ".join(f"{r} = {s:03d}" for r, s in examples.items())
    traces.to_csv(TRACE, index=False, float_format="%.6g")
    record(TRACE, script=__file__, inputs=SIGNAL_INPUTS + ht.EVENT_INPUTS,
           note=f"Example nights ({ex}; highest, median and lowest % valid N2+N3). {BIN_S}-s bins from t_s (s from the "
                "EDF start): hypnogram stage code at bin start (0 W, 1 N1, 2 N2, 3 N3, 5 REM, -9 none), SpO2 mean/min "
                "over valid samples and raw, fraction of valid seconds, median of the 1-s pleth peak-to-peak (mV), "
                "mean pulse (bpm). " + rules)
    intervals.to_csv(INTERVALS, index=False, float_format="%.6g")
    record(INTERVALS, script=__file__, inputs=SIGNAL_INPUTS + ht.EVENT_INPUTS,
           note=f"Example nights ({ex}): scored desaturations (drop >= {DROP} %; value = drop), scored SpO2 artifacts, "
                "counted N2/N3 hypopneas, signal desaturations (value = drop from baseline) and invalid segments "
                f">= {SEGMENT_S} s (reason = every rule flagging any of its seconds); s from the EDF start.")
    print("[SAVED] " + ", ".join(p.name for p in [*OUT.values(), TRACE, INTERVALS]))


if __name__ == "__main__":
    main()
