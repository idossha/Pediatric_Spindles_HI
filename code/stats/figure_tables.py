"""The small tables the figure producers read (figures/<id>/make.py never fits a model).

Parts (``python -m stats.figure_tables [part ...]``, all when none is named), each writing one table under
results/figure_data/ with a provenance sidecar: grid (fig-3 maps), interaction
(fig-s2 b, d, e), scatter (fig-s3 e), night (graphical abstract), example (fig-1: one child's night, a 6 s
window at five midline channels and the detector's criteria at Pz; reads the cleaned recording on DATA_ROOT).
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd
from provenance import record

from .interaction import fit_interaction, prepare_covariates
from .io import (
    CHAN_PATH,
    COV_PATH,
    FIGURE_DATA,
    METRICS_LONG,
    ROI_PATH,
    data_matrix_for_metric,
    ensure_dir,
    load_channel_info,
    load_covariates,
    load_spindle_matrices,
)

GRID_MAPS = FIGURE_DATA / "grid_maps.csv"
ROI_TABLE = FIGURE_DATA / "age_hi_interaction_roi.csv"
PREDICTIONS = FIGURE_DATA / "age_hi_interaction_predictions.csv"
CLUSTER_RESIDUALS = FIGURE_DATA / "fast_duration_cluster_residuals.csv"
GA_NIGHT = FIGURE_DATA / "graphical_abstract_night.csv"

# the children whose nights the graphical abstract and fig-1 draw: psg.example_child("fig-ga-story"), and
# psg.example_child("fig-1"), the highest fast density at Pz together with the highest slow density at Fz
EXAMPLE_DIR = FIGURE_DATA / "example_night"
# fig-1's midline channels, front to back, as EGI labels (display name, label). Cz is the recording
# reference and not among the 172 analysed channels, so the central row is CPz (E81), just behind it.
EXAMPLE_CHANNELS = (("Fz", "21"), ("FCz", "15"), ("CPz", "81"), ("Pz", "101"), ("Oz", "126"))
EXAMPLE_HALF_S = 3.0  # half the trace window
RASTER_BIN_MIN = 5.0


def grid_maps() -> pd.DataFrame:
    """fig-3's group means: one column per metric and band, one row per channel.

    Every map comes from the analysed per-subject detections, as Methods and Table 1 define
    the metrics: a channel without a detection has density zero (it enters the mean) and no
    duration, amplitude or frequency (it is skipped).
    """
    cov = load_covariates()
    n_channels = len(load_channel_info().labels)
    matrices = load_spindle_matrices(cov, n_channels, include_frequency=True)
    if matrices.missing_subjects:
        raise SystemExit(f"no spindle metrics for {matrices.missing_subjects} in {METRICS_LONG}")
    bands = {"fast": matrices.fast, "slow": matrices.slow}
    maps = {"channel": np.arange(1, n_channels + 1)}
    for metrics, band_order in ((("Density", "Duration", "Amplitude"), ("fast", "slow")), (("Frequency",), ("fast", "slow"))):
        for band in band_order:
            for metric in metrics:
                maps[f"{metric.lower()}_{band}"] = np.nanmean(data_matrix_for_metric(bands[band], metric, matrices.sleep_mins), axis=0)
    return pd.DataFrame(maps)


def interaction_predictions() -> pd.DataFrame:
    """Predicted cluster-mean slow-spindle amplitude and fast-spindle frequency (each over its largest Age x HI
    cluster) and fast-spindle duration (over the primary HI cluster, ROI_PATH) across log HI at mean age and
    +/- 1 SD; ``metric`` keys the three, ``hi_slope`` is the HI slope per probe."""
    cov = prepare_covariates()
    table = pd.read_csv(ROI_TABLE).set_index("id").loc[cov["id"]]
    duration = pd.read_csv(ROI_PATH).set_index("id").loc[cov["id"], "fast_duration_cluster"].to_numpy()
    return pd.concat([*(_predictions(m, table[f"{m}_cluster_mean"].to_numpy(), cov)
                        for m in ("slow_amplitude", "fast_frequency")),
                      _predictions("fast_duration", duration, cov)], ignore_index=True)


def _predictions(metric: str, roi: np.ndarray, cov: pd.DataFrame) -> pd.DataFrame:
    fit, df = fit_interaction(roi, cov)
    age_mean, age_sd = df["age_decimal"].mean(), df["age_decimal"].std(ddof=1)
    loghi_mean = df["logHI"].mean()
    loghi = np.linspace(df["logHI"].min(), df["logHI"].max(), 100)
    rows = []
    for probe, age_value in (("minus-1-sd", age_mean - age_sd), ("mean", age_mean), ("plus-1-sd", age_mean + age_sd)):
        pred = pd.DataFrame({"age_c": age_value - age_mean, "logHI_c": loghi - loghi_mean, "sex": df["sex"].mean()},
                            index=np.arange(len(loghi)))
        pred["ageXHI"] = pred["age_c"] * pred["logHI_c"]
        frame = fit.get_prediction(pred).summary_frame(alpha=0.05)
        for i in range(len(loghi)):
            rows.append({"metric": metric, "probe": probe, "age_years": age_value, "logHI": loghi[i], "mean": frame["mean"].iloc[i],
                         "ci_low": frame["mean_ci_lower"].iloc[i], "ci_high": frame["mean_ci_upper"].iloc[i],
                         "hi_slope": fit.params["logHI_c"] + fit.params["ageXHI"] * (age_value - age_mean)})
    return pd.DataFrame(rows)


def cluster_residuals() -> pd.DataFrame:
    """fig-4 b: the corrected fast-duration cluster's mean fast-spindle duration against log HI.

    The cluster is the one the behavioral models average (``cluster_mean_series``, the corrected
    fast-duration HI cluster, outlined green in fig-4 a); the partial residuals and their fit
    are computed exactly as ``stats.figure_data`` computes every other scatter series (both
    variables residualized on age and sex, then a least-squares line and Pearson r).
    """
    from .covariate_sensitivity import cluster_mean_series
    from .figure_data import _scatter_series

    cov = load_covariates()
    values, cluster = cluster_mean_series(cov)
    series = _scatter_series(values.to_numpy()[:, None], [0], cov)
    keep = ~(np.isnan(values.to_numpy()) | cov[["logHI", "age_c", "gender_bin"]].isna().any(axis=1).to_numpy())
    return pd.DataFrame({"id": cov["id"].to_numpy()[keep], "loghi_residual": series["x"],
                         "duration_residual": series["y"], "fit_slope": series["slope"],
                         "fit_intercept": series["intercept"], "r": series["r"], "n_channels": len(cluster)})


def abstract_night() -> pd.DataFrame:
    """The graphical abstract child's night per 30 s epoch: the display stage (W 0, REM 1, N1 2, N2 3, N3 4, as in
    fig-1 a) and how many scored hypopneas the events file places in the epoch."""
    from .psg import classify, example_child, paths, read_hypnogram

    psg = paths(example_child("fig-ga-story"))

    # read_hypnogram gives the pipeline's Compumedics codes (W/U 0, N1 1, N2 2, N3 3, R 5; '?' -1,
    # anything else -9); remap to the display order so W is on top and N3 at the bottom, and
    # leave unscored epochs empty.
    display = {0: 0, 1: 2, 2: 3, 3: 4, 5: 1}
    stages = [display.get(int(c), np.nan) for c in read_hypnogram(psg.hypnogram)]
    # Field 2 of each event row is the 1-based 30 s epoch, the grid the hypnogram is written on.
    rows = []
    with psg.events.open("r", encoding="latin-1") as stream:
        for line in stream:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 5 or not re.match(r"^\d{1,2}:\d{2}:\d{2}$", parts[0].strip()):
                continue
            rows.append({"label": parts[3].strip(), "epoch": int(parts[1])})
    events = pd.DataFrame(rows)
    hypopnea_epochs = events.loc[events["label"].map(classify) == "hypopnea", "epoch"].to_numpy()
    counts = np.bincount(hypopnea_epochs, minlength=len(stages) + 1)[1:len(stages) + 1]
    return pd.DataFrame({"epoch": np.arange(1, len(stages) + 1), "stage": stages, "hypopneas": counts})


def example_night() -> dict[str, pd.DataFrame]:
    """fig-1: the example child's night and one detection window, on the detector's own data.

    * ``hypnogram``: 30 s epochs from the recording start (display stage W 0, REM 1, N1 2, N2 3, N3 4) and the
      scored hypopneas' onsets.
    * ``raster``: detections per minute of artifact-free N2+N3 in 5 min bins of the recording timeline, per
      channel (``order`` 0 = most anterior) and band; bins with under a minute of retained data are empty.
    * ``traces``: the window's EEG (resampled to 100 Hz as the detector saw it, then 0.5-30 Hz for display) at
      the five midline channels, and ``detections``: that window's detections at those channels.
    * ``montage``: the 172 channels' projected positions, with the display name of the five traced ones.
    * ``criteria``: YASA 0.6.5's three criteria at Pz across the window (10-16 Hz sigma signal, its moving RMS,
      relative sigma power and moving correlation), and ``meta``: their thresholds (the RMS one from the whole
      channel, as the detector computes it), the child's age and HI, and where the window sits.

    The window is the N2 stretch, at least 8 s from any deletion junction, holding the strongest pair of a
    fast Pz detection (>= 12.8 Hz, 0.8-1.4 s) and a slow Fz detection (<= 11.5 Hz) 1.5-3 s apart.
    """
    import mne
    from detection.files import open_fdt
    from detection.settings import DETECT_SF, YASA_KWARGS
    from yasa.others import moving_transform, trimbothstd
    from yasa.spectral import stft_power

    from . import psg

    mne.set_log_level("ERROR")
    sid = psg.example_child("fig-1")
    files = psg.paths(sid)
    info = psg.read_set(files.set)
    orig_s = (info.idx - 1) / 200.0  # recording time of each cleaned 200 Hz sample
    hyp = psg.read_hypnogram(files.hypnogram)
    events = psg.read_events(files.events, info.start, info.rec_len_s)
    sp = psg.read_spindles(files.spindles)
    sp["band"] = np.where(sp["Frequency"] < 12, "slow", "fast")
    sp["t_rec"] = orig_s[np.clip(np.round(sp["Start"] * info.srate).astype(int), 0, info.n - 1)]
    channels = load_channel_info()
    labels = list(channels.labels)
    idx = {name: labels.index(label) for name, label in EXAMPLE_CHANNELS}

    display = {0: 0, 1: 2, 2: 3, 3: 4, 5: 1}
    hypnogram = pd.DataFrame({"t_h": 30.0 * np.arange(len(hyp)) / 3600,
                              "stage": [display.get(int(c), np.nan) for c in hyp]})
    hypnogram = pd.concat([hypnogram.assign(kind="stage"),
                           pd.DataFrame({"t_h": events.loc[events["kind"] == "hypopnea", "t_start"] / 3600,
                                         "stage": np.nan, "kind": "hypopnea"})], ignore_index=True)

    edges = np.arange(0.0, 30.0 * len(hyp) + RASTER_BIN_MIN * 60, RASTER_BIN_MIN * 60)
    clean_min = np.histogram(orig_s, edges)[0] / info.srate / 60.0
    order = np.argsort(-np.asarray(channels.y))
    rows = []
    for band in ("fast", "slow"):
        g = sp[sp["band"] == band]
        for rank, ch in enumerate(order):
            counts = np.histogram(g.loc[g["IdxChannel"] == ch, "t_rec"], edges)[0]
            density = np.where(clean_min >= 1.0, counts / np.maximum(clean_min, 1e-9), np.nan)
            rows.append(pd.DataFrame({"band": band, "order": rank, "channel": ch + 1, "bin_start_h": edges[:-1] / 3600,
                                      "bin_end_h": edges[1:] / 3600, "density": density}))
    raster = pd.concat(rows, ignore_index=True)

    # the window
    junctions = info.boundary_latency / info.srate
    fz = sp[(sp["IdxChannel"] == idx["Fz"]) & (sp["band"] == "slow") & (sp["Frequency"] <= 11.5)]
    best = None
    for r in sp[(sp["IdxChannel"] == idx["Pz"]) & (sp["Frequency"] >= 12.8) & sp["Duration"].between(0.8, 1.4)].itertuples():
        near = fz[(fz["Start"] - r.Start).abs().between(1.5, 3.0)]
        if near.empty:
            continue
        centre = (r.Start + near["Start"].iloc[0]) / 2
        if np.any(np.abs(junctions - centre) < 8) or hyp[int(orig_s[int(centre * info.srate)] // 30)] != 2:
            continue
        score = r.Amplitude + near["Amplitude"].iloc[0]
        if best is None or score > best[0]:
            best = (score, centre)
    t0, t1 = best[1] - EXAMPLE_HALF_S, best[1] + EXAMPLE_HALF_S

    mm, _ = open_fdt(files.set)
    pad = 10.0  # s of context around the window for resampling and filtering
    a, b = int((t0 - pad) * info.srate), int((t1 + pad) * info.srate)
    seg = mne.filter.resample(np.asarray(mm[a:b, [idx[n] for n, _ in EXAMPLE_CHANNELS]], dtype=np.float64).T,
                              down=info.srate / DETECT_SF, npad="auto")
    shown = mne.filter.filter_data(seg, DETECT_SF, 0.5, 30.0)
    k0, k1 = int(pad * DETECT_SF), int((pad + t1 - t0) * DETECT_SF)
    tt = np.arange(k1 - k0) / DETECT_SF
    traces = pd.concat([pd.DataFrame({"channel": name, "t": tt, "uV": shown[i, k0:k1]})
                        for i, (name, _) in enumerate(EXAMPLE_CHANNELS)], ignore_index=True)
    det = sp[sp["IdxChannel"].isin(idx.values()) & (sp["End"] > t0) & (sp["Start"] < t1)].copy()
    det["channel"] = det["IdxChannel"].map({v: k for k, v in idx.items()})
    detections = det.assign(start=det["Start"] - t0, end=det["End"] - t0)[["channel", "band", "start", "end", "Frequency"]]

    # the criteria at Pz; the RMS threshold, as YASA computes it, from the whole channel (this resampling
    # reproduces all of the child's Pz detections exactly)
    pz_full = mne.filter.resample(np.asarray(mm[:, idx["Pz"]], dtype=np.float64), down=info.srate / DETECT_SF, npad="auto")
    lo_b, hi_b = YASA_KWARGS["freq_sp"]
    broad_full = mne.filter.filter_data(pz_full, DETECT_SF, 1, 30, method="fir")
    sigma_full = mne.filter.filter_data(broad_full, DETECT_SF, lo_b, hi_b, l_trans_bandwidth=1.5, h_trans_bandwidth=1.5,
                                        method="fir")
    _, mrms = moving_transform(x=sigma_full, sf=DETECT_SF, window=0.3, step=0.1, method="rms", interp=True)
    thr_rms = min(mrms.mean() + YASA_KWARGS["thresh"]["rms"] * trimbothstd(mrms, cut=0.10), 10)
    c0, c1 = int(t0 * DETECT_SF), int(t1 * DETECT_SF)
    w0, w1 = c0 - int(pad * DETECT_SF), c1 + int(pad * DETECT_SF)
    f, t_stft, sxx = stft_power(broad_full[w0:w1], DETECT_SF, window=2, step=0.2, band=(1, 30), interp=False, norm=True)
    rel = np.interp(np.arange(w1 - w0) / DETECT_SF, t_stft, sxx[(f >= lo_b) & (f <= hi_b)].sum(0))
    _, corr = moving_transform(x=sigma_full[w0:w1], y=broad_full[w0:w1], sf=DETECT_SF, window=0.3, step=0.1,
                               method="corr", interp=True)
    k = slice(c0 - w0, c0 - w0 + (c1 - c0))
    criteria = pd.DataFrame({"t": np.arange(c1 - c0) / DETECT_SF, "eeg": broad_full[c0:c1], "sigma": sigma_full[c0:c1],
                             "rms": mrms[c0:c1], "rel_pow": rel[k], "corr": corr[k]})

    cov = load_covariates().set_index("id")
    meta = pd.DataFrame([{"id": sid, "age_years": cov.loc[sid, "age_decimal"],
                          "hi": cov.loc[sid, "overall_hi"], "window_start_cleaned_s": t0,
                          "window_start_h": orig_s[int(t0 * info.srate)] / 3600, "window_s": t1 - t0,
                          "thresh_rms": thr_rms, "thresh_rel_pow": YASA_KWARGS["thresh"]["rel_pow"],
                          "thresh_corr": YASA_KWARGS["thresh"]["corr"]}])
    shown = {label: name for name, label in EXAMPLE_CHANNELS}
    montage = pd.DataFrame({"egi": labels, "x": np.asarray(channels.x), "y": np.asarray(channels.y),
                            "shown": [shown.get(label, "") for label in labels]})
    return {"hypnogram": hypnogram, "raster": raster, "traces": traces, "detections": detections,
            "criteria": criteria, "meta": meta, "montage": montage}


def _write_grid() -> None:
    grid_maps().to_csv(GRID_MAPS, index=False, float_format="%.10g")
    record(GRID_MAPS, script=__file__, inputs=[COV_PATH, CHAN_PATH, METRICS_LONG],
           note="Density zero-filled where a channel has no detection.")


def _write_interaction() -> None:
    interaction_predictions().to_csv(PREDICTIONS, index=False, float_format="%.10g")
    record(PREDICTIONS, script=__file__, inputs=[COV_PATH, ROI_TABLE, ROI_PATH])


def _write_scatter() -> None:
    cluster_residuals().to_csv(CLUSTER_RESIDUALS, index=False, float_format="%.10g")
    record(CLUSTER_RESIDUALS, script=__file__, inputs=[COV_PATH, CHAN_PATH, METRICS_LONG],
           note="The corrected fast-duration HI cluster (stats.covariate_sensitivity.cluster_mean_series).")


def _write_night() -> None:
    abstract_night().to_csv(GA_NIGHT, index=False)
    from .psg import example_child, paths

    files = paths(example_child("fig-ga-story"))
    record(GA_NIGHT, script=__file__, inputs=[files.hypnogram, files.events])


def _write_example() -> None:
    from .psg import example_child, paths

    files = paths(example_child("fig-1"))
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in example_night().items():
        out = EXAMPLE_DIR / f"{name}.csv"
        table.to_csv(out, index=False, float_format="%.6g")
        record(out, script=__file__, inputs=[files.set, files.set.with_suffix(".fdt"), files.hypnogram, files.events,
                                             files.spindles, COV_PATH, CHAN_PATH],
               note="fig-1's example child (stats.psg.example_child).")


PARTS = {"grid": [_write_grid], "interaction": [_write_interaction],
         "scatter": [_write_scatter], "night": [_write_night], "example": [_write_example]}


def main(which: list[str]) -> None:
    unknown = [w for w in which if w not in PARTS]
    if unknown:
        raise SystemExit(f"unknown part(s) {unknown}; choose from {sorted(PARTS)}")
    which = which or list(PARTS)
    ensure_dir(FIGURE_DATA)
    done = []
    for part in which:
        for write in PARTS[part]:
            if write not in done:
                write()
                done.append(write)
    print("wrote " + ", ".join(which))


if __name__ == "__main__":
    main(sys.argv[1:])
