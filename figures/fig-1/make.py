"""Figure 1: Night architecture and spindle detection

Draws only source_data/fig-1/ (summarized data; nothing is fitted here).
Build: ``idofig build fig-1`` (or ./reproduce.sh).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

import idofig as idf
import matplotlib as mpl
import numpy as np
from plotting.style import (
    FONT_LABEL,
    N2_COLOR,
    N3_COLOR,
    axes_at,
    house_style,
    thin_spines,
)
from plotting.tables import read_table
from project import SOURCE_DATA

FIGURE_DATA = SOURCE_DATA / "fig-1"  # this figure's source data

DATA = FIGURE_DATA / "example_night"
SLOW_COLOR, FAST_COLOR = "#d95f02", "#5e3c99"
HYPOPNEA_COLOR, THRESHOLD_COLOR = "#b2182b", "#b2182b"
STAGES = ["W", "R", "N1", "N2", "N3"]  # display codes 0..4, top to bottom
ORDER_CMAP = mpl.colormaps["Greys_r"]  # channel order: dark = front, light = back
ORDER_RANGE = (0.05, 0.85)

# a (mm from the panel's bottom-left)
NIGHT_LEFT, NIGHT_W = 22.0, 143.0
HYPNO_AXES = (NIGHT_LEFT, 60.0, NIGHT_W, 20.5)
RASTER_AXES = {"slow": (NIGHT_LEFT, 34.0, NIGHT_W, 22.0), "fast": (NIGHT_LEFT, 9.5, NIGHT_W, 22.0)}
STRIP_W = 1.4
CBAR_X, CBAR_W = NIGHT_LEFT + NIGHT_W + 2.0, 1.6
ROW_H, ROW_GAP = 8.6, 1.8
CRIT_LEFT, CRIT_W = 19.0, 44.0
TRACE_LEFT, TRACE_W = 10.0, 78.0
TRACE_LABEL_GAP = 2.6  # mm above each trace row for its detections' frequency labels
DETAIL_TOP = 54.0  # b and c: the top of the first row
MONTAGE_AXES = (90.7, 41.2, 18.0, 18.3)  # the montage inset, in the margin right of the two frontal rows
TRACE_YLIM = (-60.0, 76.0)  # µV, shared by every channel so one scale bar serves all; holds every trace unclipped
WINDOW_MARK = 4.0  # pt, the triangle over the hypnogram at the window shown in b and c


def order_color(rank: np.ndarray, n: int):
    lo, hi = ORDER_RANGE
    return ORDER_CMAP(lo + (hi - lo) * rank / (n - 1))


def night(fig) -> None:
    hyp = read_table(DATA / "hypnogram.csv")
    raster = read_table(DATA / "raster.csv")
    meta = read_table(DATA / "meta.csv")
    stage = hyp["kind"] == "stage"
    t, s = hyp["t_h"][stage], hyp["stage"][stage]
    end = t.max() + 30 / 3600

    ax = idf.name(axes_at(fig, HYPNO_AXES), "hypnogram")
    for code, color in ((3, N2_COLOR), (4, N3_COLOR)):  # one span per run of epochs, lightened behind the line
        on = np.flatnonzero(np.diff(np.concatenate([[0], (s == code).astype(int), [0]])))
        for a, b in zip(on[::2], on[1::2]):
            ax.axvspan(t[a], t[b - 1] + 30 / 3600, color=color, alpha=0.35, lw=0, zorder=0, label=f"stage{code}-{a}")
    ok = np.isfinite(s)
    ax.step(np.append(t[ok], end), np.append(s[ok], s[ok][-1]), where="post", color="black", lw=0.6, label="stage")
    ax.vlines(hyp["t_h"][~stage], 4.55, 4.95, color=HYPOPNEA_COLOR, lw=0.6, label="hypopneas")
    w0 = float(meta["window_start_h"][0])
    ax.plot([w0], [1.0], marker="v", ms=WINDOW_MARK, color="black", mew=0, transform=ax.get_xaxis_transform(),
            clip_on=False, ls="none", label="window")
    ax.annotate("b, c", (w0, 1.0), xycoords=ax.get_xaxis_transform(), xytext=(0, WINDOW_MARK * 0.5 + 0.5),
                textcoords="offset points", ha="center", va="bottom", fontsize=FONT_LABEL, fontweight="bold",
                annotation_clip=False, gid="window-label")
    ax.set_yticks(range(5), STAGES)
    ax.set_ylim(5.0, -0.2)
    ax.set_xlim(0, end)
    ax.tick_params(labelbottom=False)
    thin_spines(ax)

    n = int(raster["order"].max()) + 1
    for band, cmap in (("slow", "Oranges"), ("fast", "Purples")):
        m = raster["band"] == band
        edges = np.append(np.unique(raster["bin_start_h"][m]), raster["bin_end_h"][m].max())
        grid = np.full((n, len(edges) - 1), np.nan)
        cols = np.searchsorted(edges, raster["bin_start_h"][m])
        grid[raster["order"][m].astype(int), cols] = raster["density"][m]
        rx = idf.name(axes_at(fig, RASTER_AXES[band], sharex=ax), f"raster-{band}")
        im = rx.pcolormesh(edges, np.arange(n + 1), grid, cmap=cmap, vmin=0, vmax=np.nanpercentile(grid, 98),
                           shading="flat", rasterized=True, label=f"{band}-rate")
        rx.set_ylim(n, 0)
        rx.set_yticks([])
        rx.set_ylabel(f"{band.capitalize()} spindles\nper channel", labelpad=15)
        beside = -(STRIP_W + 0.9) / NIGHT_W  # left of the order strip, in axes fraction
        rx.text(beside, 1.0, "front", transform=rx.transAxes, ha="right", va="top", fontsize=FONT_LABEL,
                gid=f"{band}-front")
        rx.text(beside, 0.0, "back", transform=rx.transAxes, ha="right", va="bottom", fontsize=FONT_LABEL,
                gid=f"{band}-back")
        strip = axes_at(fig, (RASTER_AXES[band][0] - STRIP_W - 0.4, RASTER_AXES[band][1], STRIP_W, RASTER_AXES[band][3]))
        strip.imshow(order_color(np.arange(n), n)[:, None, :], aspect="auto", interpolation="nearest")
        strip.axis("off")
        idf.name(strip, f"order-{band}")
        thin_spines(rx, hide=("top", "right"))
        if band == "fast":
            rx.set_xlabel("Time from recording start (h)")
        else:
            rx.tick_params(labelbottom=False)
        cax = axes_at(fig, (CBAR_X, RASTER_AXES[band][1] + 2.0, CBAR_W, RASTER_AXES[band][3] - 6.0))
        cb = idf.colorbar(im, cax=cax, name=f"{band}-rate")
        cb.ax.tick_params(labelsize=FONT_LABEL, width=0.4, length=2)
        cb.outline.set_linewidth(0.4)
        cb.ax.set_title("/min", loc="left", fontsize=FONT_LABEL, pad=2.5)


def scale_bar(ax, seconds: float, microvolts: float | None, y: float, label_uv: str | None = None) -> None:
    ax.plot([0, seconds], [y, y], color="black", lw=1.0, clip_on=False, solid_capstyle="butt")
    ax.text(seconds / 2, y, f"{seconds:g} s", ha="center", va="top", fontsize=FONT_LABEL)


def shade(ax, detections, channel: str) -> None:
    for i in np.flatnonzero(detections["channel"] == channel):
        color = FAST_COLOR if detections["band"][i] == "fast" else SLOW_COLOR
        ax.axvspan(detections["start"][i], detections["end"][i], color=color, alpha=0.12, lw=0,
                   label=f"detection-{i}")


def criteria(fig) -> None:
    c = read_table(DATA / "criteria.csv")
    meta = read_table(DATA / "meta.csv")
    det = read_table(DATA / "detections.csv")
    rows = [("eeg", "EEG", None), ("sigma", "σ 10–16 Hz", None), ("rms", "Moving RMS", "thresh_rms"),
            ("rel_pow", "Rel. σ power", "thresh_rel_pow"), ("corr", "Correlation", "thresh_corr")]
    top = DETAIL_TOP
    axes = []
    for k, (col, label, thresh) in enumerate(rows):
        ax = idf.name(axes_at(fig, (CRIT_LEFT, top - (k + 1) * ROW_H - k * ROW_GAP, CRIT_W, ROW_H)), f"criterion-{col}")
        v = c[col]
        ax.plot(c["t"], v, color="black", lw=0.5, label=col)
        if thresh:
            th = float(meta[thresh][0])
            ax.axhline(th, color=THRESHOLD_COLOR, lw=0.6, ls=(0, (3, 2)), label=f"{col}-threshold")
            ax.fill_between(c["t"], th, v, where=v >= th, color=THRESHOLD_COLOR, alpha=0.2, lw=0,
                            label=f"{col}-above")
        shade(ax, det, "Pz")
        ax.set_xlim(0, c["t"].max())
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=2)
        thin_spines(ax, hide=("top", "right", "left", "bottom"))
        ax.tick_params(bottom=False, labelbottom=False)
        axes.append(ax)
    axes[0].set_title("Detection logic", fontsize=FONT_LABEL + 1, fontweight="bold", pad=3)
    last = axes[-1]
    lo = last.get_ylim()[0]
    scale_bar(last, 1.0, None, lo - 0.15 * np.ptp(last.get_ylim()))


def montage(fig) -> None:
    """The 172-channel montage, nose up, with the five traced channels filled."""
    m = read_table(DATA / "montage.csv")
    x, y = np.asarray(m["x"], dtype=float), np.asarray(m["y"], dtype=float)
    shown = np.array([str(s) not in ("", "nan") for s in m["shown"]])
    r = 1.12 * float(np.max(np.hypot(x, y)))
    ax = idf.name(axes_at(fig, MONTAGE_AXES), "montage")
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(t), r * np.sin(t), color="black", lw=0.6, gid="head")
    base = np.sqrt(1 - 0.09 ** 2) * r
    ax.plot([-0.09 * r, 0.0, 0.09 * r], [base, 1.15 * r, base], color="black", lw=0.6, solid_capstyle="butt", gid="nose")
    te = np.linspace(-np.pi / 2, np.pi / 2, 60)
    for side, gid in ((1, "ear-right"), (-1, "ear-left")):
        ax.plot(side * r * (1 + 0.07 * np.cos(te)), 0.17 * r * np.sin(te), color="black", lw=0.6, gid=gid)
    ax.scatter(x[~shown], y[~shown], s=1.8, color="0.65", linewidths=0, gid="electrodes")
    ax.scatter(x[shown], y[shown], s=9.0, color="black", linewidths=0, zorder=3, gid="shown")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.1 * r, 1.1 * r)
    ax.set_ylim(-1.04 * r, 1.19 * r)


def traces(fig) -> None:
    tr = read_table(DATA / "traces.csv")
    det = read_table(DATA / "detections.csv")
    names = ["Fz", "FCz", "CPz", "Pz", "Oz"]
    h = fig.get_figheight() * 25.4
    top = DETAIL_TOP
    span = 5 * ROW_H + 4 * ROW_GAP
    pitch = span / len(names)
    lo, hi = TRACE_YLIM
    axes = []
    for k, name in enumerate(names):
        rect = (TRACE_LEFT, top - (k + 1) * pitch, TRACE_W, pitch - TRACE_LABEL_GAP)
        ax = idf.name(axes_at(fig, rect), f"trace-{name.lower()}")
        m = tr["channel"] == name
        t, v = tr["t"][m], tr["uV"][m]
        ax.plot(t, v, color="black", lw=0.5, label=name.lower())
        for i in np.flatnonzero(det["channel"] == name):
            color = FAST_COLOR if det["band"][i] == "fast" else SLOW_COLOR
            s = (t >= det["start"][i]) & (t <= det["end"][i])
            ax.plot(t[s], v[s], color=color, lw=0.7, label=f"{name.lower()}-spindle-{i}")
            ax.text((det["start"][i] + det["end"][i]) / 2, v[s].max() + 0.03 * (hi - lo),  # just above the spindle
                    f"{det['Frequency'][i]:.1f} Hz", color=color, ha="center", va="bottom", fontsize=FONT_LABEL,
                    gid=f"{name.lower()}-frequency-{i}")
        ax.set_ylim(lo, hi)
        ax.set_xlim(0, t.max())
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=2)
        thin_spines(ax, hide=("top", "right", "left", "bottom"))
        ax.tick_params(bottom=False, labelbottom=False)
        axes.append(ax)
    axes[0].set_title("Detection examples", fontsize=FONT_LABEL + 1, fontweight="bold", pad=3 + TRACE_LABEL_GAP / 25.4 * 72)
    last = axes[-1]  # the amplitude scale beside the last row, under the montage
    x_end = last.get_xlim()[1]
    last.plot([x_end + 0.08] * 2, [-25, 25], color="black", lw=1.0, clip_on=False, gid="scale-uv")
    last.text(x_end + 0.15, 0, "50 µV", va="center", fontsize=FONT_LABEL, gid="scale-uv-label")
    scale_bar(axes[-1], 1.0, None, lo - 0.05 * (hi - lo))
    for band, color, x in (("slow 10–12 Hz", SLOW_COLOR, 0.0), ("fast 12–16 Hz", FAST_COLOR, 0.32)):
        fig.text((TRACE_LEFT + TRACE_W * (0.36 + x)) / (fig.get_figwidth() * 25.4), 0.6 / h, band, color=color,
                 fontsize=FONT_LABEL, va="bottom", gid=f"key-{band.split()[0]}")


p = idf.panels("night", "criteria", "traces", layout=None)
house_style()
night(p["night"])
criteria(p["criteria"])
traces(p["traces"])
montage(p["traces"])
idf.save(p)
