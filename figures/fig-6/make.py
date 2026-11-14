"""Figure 6: Hypopnea type and fast spindle duration (exploratory)

Draws only source_data/fig-6/ (summarized data; nothing is fitted here).
Build: ``idofig build fig-6`` (or ./reproduce.sh).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

import idofig as idf
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from plotting.mapgrid import P_CLUSTER, small_map, test_label
from plotting.oximetry import CLASS_COLOR, SPO2
from plotting.style import (
    FONT_LABEL,
    FONT_SUBSCRIPTED,
    FONT_TITLE,
    LOG10,
    N2_COLOR,
    N3_COLOR,
    axes_at,
    house_style,
    minus,
    thin_spines,
)
from plotting.tables import read_table
from plotting.topo import OUTLINE_COLOR, OUTLINE_LW, outline_region
from project import SOURCE_DATA

FIGURE_DATA = TABLE_DIR = SOURCE_DATA / "fig-6"  # this figure's source data
from stats.io import SPHERE

FIGDATA, TABLES = FIGURE_DATA, TABLE_DIR

# the two hypopnea classes the analysis counts, in plotting.oximetry's palette (unclassified and unknown are not drawn)
CLASS_NAME = {"desaturating": "Desaturating", "arousal_only": "Arousal-only"}
DESAT_COLOR = "#7b3294"
AROUSAL_COLOR = "#1a1a1a"
SPINDLE_COLOR = "#666666"
EXCLUDED_COLOR = "#e4e4e4"  # wake, N1, REM: shaded across every track of the night, which shows N2/N3 events only
WINDOW_S = {"desaturating": 20.0, "arousal_only": 20.0}  # classification window: onset → end + this

STAGES = ["W", "R", "N1", "N2", "N3"]           # display rows 0..4, top to bottom (as fig-1 a)
STAGE_ROW = {0: 0, 5: 1, 1: 2, 2: 3, 3: 4}       # file code → display row; negative codes are not stages

# a (mm from the panel's bottom-left)
NIGHT_LEFT, NIGHT_W = 22.0, 88.0
HYPNO_AXES = (NIGHT_LEFT, 36.5, NIGHT_W, 18.0)
TRACK_H, TRACK_PITCH, TRACK_TOP = 3.6, 5.0, 29.0  # event tracks under the hypnogram
SPINDLE_AXES = (NIGHT_LEFT, 12.0, NIGHT_W, 4.6)  # the fast-spindle strip under the event tracks
ZOOM_LEFT, ZOOM_W, ZOOM_H = 128.0, 46.0, 16.0
ZOOM_BOTTOMS = (39.5, 19.0)
ZOOM_TITLE_X = -12.5  # mm from the zoom axes' left: titles start over the row labels
ZOOM_SPAN = (-15.0, 50.0)  # s around the hypopnea's onset
ZOOM_EXAMPLES = (("desaturating", 14073.0), ("arousal_only", 4228.0))  # onsets of two N2 hypopneas of that child
MARK_SIZE = 4.0  # pt: the numbered triangle over the hypnogram at an enlarged hypopnea's onset
ZOOM_ROWS = {"hypopnea": 2.0, "desat": 1.0, "arousal": 0.0}  # y of each zoom row, top to bottom
ZOOM_ROW_NAME = {"hypopnea": "Hypopnea", "desat": f"{SPO2} desat.", "arousal": "Arousal"}

# b, d
COMP_AXES = (14.0, 7.0, 46.0, 42.0)
BAR_WIDTH = 0.8  # of one child's column
FOREST_AXES = (33.0, 8.5, 40.0, 8.0)  # under the two maps of c

# c: two maps side by side, the colour bar upright at their right
MAP, PITCH, MAP_LEFT = 27.0, 42.0, 16.0
MAP_BOTTOM = 5.0
BAR = (MAP_LEFT + PITCH + MAP + 9.0, MAP_BOTTOM + 3.0, 1.8, MAP - 6.0)
MAPS = ("desaturating", "arousal_only")


def spans(mask: np.ndarray):
    """(first, last) index of each run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return zip(edges[::2], edges[1::2] - 1)


# ---------------------------------------------------------------------------------------------------------- a
def night(fig) -> None:
    hyp = read_table(FIGDATA / "hypopnea_type_example_hypnogram.csv")
    ev = read_table(FIGDATA / "hypopnea_type_example_events.csv")
    t = hyp["t_s"] / 3600
    s = np.array([STAGE_ROW.get(int(c), np.nan) for c in hyp["code"]], dtype=float)
    end = t.max() + 30 / 3600
    width, height = fig.get_figwidth() * 25.4, fig.get_figheight() * 25.4

    ax = idf.name(axes_at(fig, HYPNO_AXES), "hypnogram")
    for row, color in ((3, N2_COLOR), (4, N3_COLOR)):
        for a, b in spans(s == row):
            ax.axvspan(t[a], t[b] + 30 / 3600, color=color, alpha=0.35, lw=0, zorder=0, label=f"stage{row}-{a}")
    ok = np.isfinite(s)
    ax.step(np.append(t[ok], end), np.append(s[ok], s[ok][-1]), where="post", color="black", lw=0.6, label="stage")
    ax.set_yticks(range(5), STAGES)
    ax.set_ylim(4.6, -0.6)
    ax.set_xlim(0, end)
    ax.tick_params(labelbottom=False)
    thin_spines(ax)

    kind, cls, t0 = ev["kind"], ev["cls"], ev["t_start"]
    tracks = [("hypopneas", "N2/N3 hypopneas"), ("desats", f"{SPO2} desat. ≥ 3%"), ("arousals", "Arousals")]
    rects = [(NIGHT_LEFT, TRACK_TOP - k * TRACK_PITCH, NIGHT_W, TRACK_H) for k in range(len(tracks))]
    axes = {}
    for (key, text), rect in zip([*tracks, ("spindles", "Fast spindles\n(cluster)")], [*rects, SPINDLE_AXES]):
        tx = idf.name(axes_at(fig, rect, sharex=ax), f"track-{key}")
        tx.set_ylim(0, 1)
        tx.set_yticks([])
        tx.set_ylabel(text, rotation=0, ha="right", va="center", labelpad=2)
        thin_spines(tx, hide=("top", "right", "left", "bottom"))
        tx.tick_params(bottom=False, labelbottom=False)
        axes[key] = tx
    for c in ("arousal_only", "desaturating"):  # red drawn last, on top
        m = (kind == "hypopnea") & (cls == c)
        axes["hypopneas"].vlines(t0[m] / 3600, 0, 1, color=CLASS_COLOR[c], lw=0.6, label=f"hypopneas-{c}")
    in_n2n3 = np.isin(ev["stage"], (2, 3))
    m = (kind == "desat") & in_n2n3
    axes["desats"].vlines(t0[m] / 3600, 0, 1, color=DESAT_COLOR, lw=0.5, label="desaturations")
    m = (kind == "arousal") & in_n2n3
    axes["arousals"].vlines(t0[m] / 3600, 0, 1, color=AROUSAL_COLOR, lw=0.4, label="arousals")
    spindles(axes["spindles"], width, height)
    bottom = axes["spindles"]
    bottom.spines["bottom"].set_visible(True)
    bottom.spines["bottom"].set_position(("outward", 1.5))
    bottom.tick_params(bottom=True, labelbottom=True)
    bottom.set_xlabel("Time from recording start (h)")
    mark_y = HYPNO_AXES[1] + HYPNO_AXES[3] + 1.0
    for a in (ax, *axes.values()):  # the excluded epochs' shading runs behind every track, through the gaps
        a.patch.set_visible(False)
    y0, y1 = SPINDLE_AXES[1] / height, (HYPNO_AXES[1] + HYPNO_AXES[3]) / height
    for a, b in spans(~np.isin(hyp["code"], (2, 3))):
        x0, x1 = (NIGHT_LEFT + NIGHT_W * np.array([t[a], t[b] + 30 / 3600]) / end) / width
        fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure, fc=EXCLUDED_COLOR, lw=0,
                                 zorder=-1, gid=f"excluded-{a}"))
    for k, (c, onset) in enumerate(ZOOM_EXAMPLES, start=1):  # where the hypopneas enlarged at right fall: a point, no width
        x = (NIGHT_LEFT + NIGHT_W * onset / 3600 / end) / width
        fig.add_artist(Line2D([x], [mark_y / height], transform=fig.transFigure, marker="v", ms=MARK_SIZE, mew=0,
                              color=CLASS_COLOR[c], zorder=6, gid=f"example-{k}-marker"))
        fig.text(x, (mark_y + 0.9) / height, str(k), ha="center", va="bottom", fontsize=FONT_LABEL,
                 fontweight="bold", color=CLASS_COLOR[c], gid=f"example-{k}")

    first = None
    for k, ((c, onset), bottom_mm) in enumerate(zip(ZOOM_EXAMPLES, ZOOM_BOTTOMS), start=1):
        zx = idf.name(axes_at(fig, (ZOOM_LEFT, bottom_mm, ZOOM_W, ZOOM_H), sharex=first), f"zoom-{k}")
        first = first or zx
        zoom(zx, ev, hyp, c, onset, k, last=k == len(ZOOM_EXAMPLES))


def spindles(ax, width, height) -> None:
    """The child's N2/N3 fast spindles on the cluster channels per bin, and the one mean duration it contributes."""
    sp = read_table(FIGDATA / "hypopnea_type_example_spindles.csv")
    step = (sp["t_s"][1] - sp["t_s"][0]) / 3600
    ax.bar(sp["t_s"] / 3600, sp["n_fast_cluster"], width=step, align="edge", color=SPINDLE_COLOR, lw=0,
           label="spindle-counts")
    ax.set_ylim(0, sp["n_fast_cluster"].max())
    right = (NIGHT_LEFT + NIGHT_W + 1.5) / width
    mid = (SPINDLE_AXES[1] + SPINDLE_AXES[3] / 2) / height
    fig = ax.figure
    fig.text(right, mid, f"Mean duration\n{sp['child_mean_duration_s'][0]:.2f} s", ha="left", va="center",
             fontsize=FONT_LABEL, color="#333333", gid="spindle-mean")


def zoom(ax, ev, hyp, c, onset, k, last) -> None:
    kind, cls, t0, t1 = ev["kind"], ev["cls"], ev["t_start"], ev["t_end"]
    lo, hi = onset + ZOOM_SPAN[0], onset + ZOOM_SPAN[1]
    shown = (t1 > lo) & (t0 < hi)
    i = int(np.flatnonzero((kind == "hypopnea") & (t0 == onset))[0])
    assert cls[i] == c, f"the example at {onset} s is {cls[i]}, not {c}"
    epoch_code = int(hyp["code"][int(np.flatnonzero(hyp["t_s"] == 30.0 * (onset // 30))[0])])
    assert epoch_code in (2, 3) and int(ev["stage"][i]) == epoch_code, f"the example at {onset} s is not in N2/N3"
    w_end = t1[i] - onset + WINDOW_S[c]
    rows = ZOOM_ROWS
    bar_h = 0.56
    yb = rows["hypopnea"] + 0.75  # the window's bracket and its rule, above the rows

    # the window shades the rows that classify
    ax.add_patch(Rectangle((0, rows["arousal"] - 0.5), w_end, yb - 0.25 - (rows["arousal"] - 0.5), fc=CLASS_COLOR[c],
                           alpha=0.13, lw=0, zorder=0, label="window"))
    for j in np.flatnonzero(shown):
        x0, w = max(t0[j], lo) - onset, min(t1[j], hi) - max(t0[j], lo)
        if kind[j] == "hypopnea":
            color = CLASS_COLOR[cls[j]]
        elif kind[j] == "desat":
            color = DESAT_COLOR
        else:
            color = AROUSAL_COLOR
        ax.broken_barh([(x0, w)], (rows[kind[j]] - bar_h / 2, bar_h), color=color, lw=0, zorder=3,
                       label=f"{kind[j]}-{j}")
    ax.plot([0, 0, w_end, w_end], [yb - 0.15, yb, yb, yb - 0.15], color=CLASS_COLOR[c], lw=0.6,
            solid_capstyle="butt", label="bracket")
    ax.text(w_end / 2, yb + 0.25, f"onset to end + {WINDOW_S[c]:g} s", ha="center", va="bottom", fontsize=FONT_LABEL,
            color=CLASS_COLOR[c], gid=f"zoom-{k}-window-label")
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="none", ec=CLASS_COLOR[c], lw=0.5,
                           clip_on=False, zorder=1, gid=f"zoom-{k}-frame"))  # matches example k's marker on the night
    ax.set_xlim(*ZOOM_SPAN)
    ax.set_ylim(-0.5, yb + 1.35)
    ax.set_yticks(list(rows.values()), [ZOOM_ROW_NAME[r] for r in rows])
    ax.tick_params(axis="y", length=0, pad=2)
    thin_spines(ax, hide=("top", "right", "left"))
    ax.text(ZOOM_TITLE_X / ZOOM_W, 1.0 + 0.8 / ZOOM_H, f"{k}  {CLASS_NAME[c]}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=FONT_LABEL, fontweight="bold", color=CLASS_COLOR[c], gid=f"zoom-{k}-title")
    ax.tick_params(axis="x", length=2, width=0.4, labelbottom=last)
    ax.xaxis.set_major_formatter(lambda v, _: minus(f"{v:g}"))
    if last:
        ax.set_xlabel("Time from hypopnea onset (s)")


# ---------------------------------------------------------------------------------------------------------- b
def composition(fig) -> None:
    comp = read_table(TABLES / "hypopnea_type_children.csv")
    windows = read_table(TABLES / "hypopnea_type_windows.csv")
    order = np.argsort(comp["hi"], kind="stable")  # lowest HI at left
    up, down = (np.log10(comp[f"{c}_per_h"][order] + 1.0) for c in MAPS)  # the uncentred values the models take
    top = float(np.ceil(max(up.max(), down.max()) * 2) / 2)
    xs = np.arange(len(up))

    ax = idf.name(axes_at(fig, COMP_AXES), "bars")
    ax.bar(xs, up, width=BAR_WIDTH, color=CLASS_COLOR["desaturating"], lw=0, label="desaturating")
    ax.bar(xs, -down, width=BAR_WIDTH, color=CLASS_COLOR["arousal_only"], lw=0, label="arousal_only")
    ax.axhline(0, color="black", lw=0.6, zorder=3, label="zero")
    ax.set_xlim(-0.8, len(xs) - 0.2)
    ax.set_ylim(-top, top)
    ticks = np.arange(-top, top + 0.25, 0.5)
    ax.set_yticks(ticks, [f"{abs(v):g}" for v in ticks])  # magnitudes on both sides of zero
    for v, tick in zip(ticks, ax.yaxis.get_major_ticks()):  # each magnitude twice: name each side
        side = "up" if v > 0 else "down" if v < 0 else "zero"
        tick.label1.set_gid(f"ytick-{side}-{abs(v):g}".replace(".", "-"))
    ax.set_xticks([])
    ax.set_xlabel("Children, sorted by HI")
    ax.set_ylabel(f"{LOG10}(hypopneas per hour of\nvalid N2 + N3 {SPO2} + 1)", fontsize=FONT_SUBSCRIPTED)
    thin_spines(ax, hide=("top", "right", "bottom"))
    ax.tick_params(axis="y", length=2, width=0.4)
    ax.text(0.03, 0.97, "Desaturating ↑", transform=ax.transAxes, ha="left", va="top", fontsize=FONT_LABEL,
            color=CLASS_COLOR["desaturating"], gid="label-desaturating")
    ax.text(0.03, 0.03, "Arousal-only ↓", transform=ax.transAxes, ha="left", va="bottom", fontsize=FONT_LABEL,
            color=CLASS_COLOR["arousal_only"], gid="label-arousal_only")
    r = float(windows["r_log_indices"][windows["primary"]][0])
    ax.text(0.03, 0.85, f"$r$ = {r:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=FONT_LABEL,
            color="#333333", gid="r-value")


# ---------------------------------------------------------------------------------------------------------- c
def maps(fig) -> None:
    tm = read_table(FIGDATA / "hypopnea_type_tmaps.csv")
    cl = read_table(TABLES / "hypopnea_type_clusters.csv")
    fig3 = read_table(TABLE_DIR / "cluster_tmaps.csv")
    order = np.argsort(tm["channel"])
    hi_cluster = tm["hi_cluster"][order].astype(bool)
    width, height = fig.get_figwidth() * 25.4, fig.get_figheight() * 25.4
    limit = float(np.ceil(max(np.nanmax(np.abs(fig3["t"])), *(np.nanmax(np.abs(tm[f"t_joint_{k}"])) for k in MAPS)) * 2) / 2)
    im = None
    for j, key in enumerate(MAPS):
        i = int(np.flatnonzero((cl["index"] == key) & (cl["model"] == "joint"))[0])
        n, pval = int(cl["n_channels"][i]), float(cl["p_value"][i])
        fig.text((MAP_LEFT + j * PITCH + MAP / 2) / width, (height - 1.0) / height, CLASS_NAME[key], ha="center",
                 va="top", fontsize=FONT_TITLE, fontweight="bold", color=CLASS_COLOR[key], gid=f"title-{key}")
        im = small_map(fig, (MAP_LEFT + j * PITCH, MAP_BOTTOM, MAP, MAP), tm[f"t_joint_{key}"][order],
                       tm[f"cluster_joint_{key}"][order].astype(bool), pval < 0.05, test_label(n, pval, P_CLUSTER),
                       limit=limit, name=f"map-{key}")
        outline_region(im.axes, hi_cluster, f"map-{key}.hi-cluster")
    key_y = (MAP_BOTTOM + MAP / 2) / height  # between the two maps
    key_x = (MAP_LEFT + PITCH / 2 + MAP / 2) / width
    fig.add_artist(Line2D([key_x - 1.75 / width, key_x + 1.75 / width], [key_y + 1.6 / height] * 2,
                          transform=fig.transFigure, color=OUTLINE_COLOR, lw=OUTLINE_LW, gid="hi-cluster-key"))
    fig.text(key_x, key_y, "HI\ncluster", ha="center", va="top", fontsize=FONT_LABEL, color="#555555",
             linespacing=1.0, gid="hi-cluster-label")
    cb = idf.colorbar(im, cax=axes_at(fig, BAR), name="t", orientation="vertical")
    whole = int(np.floor(im.norm.vmax))
    cb.set_ticks([-whole, 0, whole])
    cb.ax.tick_params(labelsize=FONT_LABEL, width=0.4, length=2, pad=1.0)
    cb.ax.yaxis.set_major_formatter(lambda v, _: minus(f"{v:g}"))
    cb.outline.set_linewidth(0.4)
    cb.set_label(f"$t$ of {LOG10}(rate + 1)", fontsize=FONT_LABEL, labelpad=5.0)


# ---------------------------------------------------------------------------------------------------------- d
def forest(fig) -> None:
    """Both rates in one model of the cluster-mean duration: each coefficient, adjusted for the other."""
    co = read_table(TABLES / "hypopnea_type_models.csv")
    joint = (co["outcome"] == "cluster mean, N2+N3") & (co["model"] == "joint")
    rows = [("desaturating", CLASS_NAME["desaturating"], CLASS_COLOR["desaturating"], 0.0),
            ("arousal_only", CLASS_NAME["arousal_only"], CLASS_COLOR["arousal_only"], 1.0)]
    ax = idf.name(axes_at(fig, FOREST_AXES), "forest")
    to_axes = 1.0 / FOREST_AXES[2]  # mm → axes fraction
    for index, name, color, y in rows:
        i = int(np.flatnonzero(joint & (co["index"] == index))[0])
        est, lo, hi, p = co["beta"][i], co["CI_lower"][i], co["CI_upper"][i], co["p"][i]
        ax.plot([lo, hi], [y, y], color=color, lw=1.0, solid_capstyle="butt", label=f"ci-{index}")
        ax.plot([est], [y], marker="o", ms=3.0, color=color, mew=0, ls="none", label=f"estimate-{index}")
        ax.text(1.0 + 1.5 * to_axes, y, minus(f"$p$ = {p:.3f}"), transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=FONT_LABEL, fontweight="bold" if p < 0.05 else "normal", color=color,
                gid=f"p-{index}")
        ax.text(-1.0 * to_axes, y, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=FONT_LABEL, color=color, gid=f"label-{index}")
    ax.axvline(0, color="0.5", lw=0.6, ls=(0, (3, 2)), zorder=0, label="zero")
    ax.set_yticks([])
    ax.tick_params(axis="x", length=2, width=0.4)
    ax.set_ylim(1.6, -0.6)
    ax.set_xlim(-0.29, 0.19)
    ax.set_xticks([-0.2, -0.1, 0.0, 0.1])
    ax.xaxis.set_major_formatter(lambda v, _: minus(f"{v:g}"))
    ax.set_xlabel(f"Coefficient (s per {LOG10} unit)", fontsize=FONT_SUBSCRIPTED)
    thin_spines(ax, hide=("top", "right", "left"))


p = idf.panels("night", "composition", "maps", "forest", layout=None)
house_style()
night(p["night"])
composition(p["composition"])
maps(p["maps"])
forest(p["forest"])
idf.save(p)
