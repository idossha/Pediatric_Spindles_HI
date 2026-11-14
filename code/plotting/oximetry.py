"""Pulse-oximetry quality panels shared by fig-s5 and the internal QC figures fig-ox-quality, fig-ox-agreement and
fig-ox-hypopnea (library). Each function draws one panel into an idofig panel figure from the tables of
``stats.oximetry_qc``; the hypopnea-class palette is fig-6's.
"""
from __future__ import annotations

import idofig as idf
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Patch
from project import TABLE_DIR

from .style import FONT_LABEL, FONT_TITLE, SCATTER_COLOR, axes_at, thin_spines
from .tables import read_table

SPO2, SPO2_BOLD = r"SpO$_2$", r"SpO$\mathbf{_2}$"  # plain and bold-title forms
THRESHOLD = 80.0  # % of N2 + N3 valid (``below_80pct_valid_n2n3``)
REASONS = {  # first-reason order of stats.oximetry
    "out_of_range": (f"{SPO2} out of range", "#3b5b92"),
    "pleth_flat": ("No pleth pulse", "#e08214"),
    "flat": (f"Frozen {SPO2}", "#b35806"),
    "step": ("Implausible step", "#8073ac"),
    "scored_artifact": ("Scored artifact", "#a6a6a6"),
}
# the hypopnea classes in every figure (fig-6 too); unknown's text is darker than its light-grey mark
CLASS_COLOR = {"desaturating": "#c0392b", "arousal_only": "#1f8a99", "unclassified": "#9a9a9a", "unknown": "#cfcfcf"}
CLASS_TEXT = CLASS_COLOR | {"unknown": "#555555"}
CLASS_NAME = {"desaturating": "Desaturating", "arousal_only": "Arousal-only", "unclassified": "Unclassified",
              "unknown": "Unknown"}
SCORED = ("desaturating", "arousal_only", "unclassified")
SIGNAL = (*SCORED, "unknown")
REPORT_COLOR = "#d95f02"
IDENTITY = dict(color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=0)
AXES = (13.0, 11.0, 42.0, 40.0)  # a square-ish scatter axes
NADIR_AXES = (13.0, 11.0, 42.0, 36.0)  # room above for the legend row under the title
DETECTION_AXES = (13.0, 11.0, 42.0, 35.0)  # room above for the pooled values and the title
BARS_AXES = (18.0, 8.0, 156.0, 30.0)
HEAT_AXES = (18.0, 17.0, 38.0, 30.0)
LOG_TICKS = [0, 1, 3, 10, 30, 100]
LABEL_OFFSET = {6: (12, -12)}  # pt: a labelled ODI dot whose label would cover its neighbours
OUTLIER = 10.0  # % below the signal nadir: label that child's scored nadir
CMAP = mpl.colormaps["Greys"]


def pad(i) -> str:
    return f"{int(i):03d}"


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def quality_bars(fig, ids: bool = False) -> None:
    """One bar per child (sorted by % of N2 + N3 valid, worst at left): % of N2 + N3 invalid, stacked by first
    reason; dashed line = 80 % valid, the children below it labelled with their id when ``ids`` (internal figures
    only). Reasons with no minutes are not drawn."""
    q = read_table(TABLE_DIR / "oximetry_quality_children.csv")
    order = np.argsort(q["valid_n2n3_pct"], kind="stable")
    total = q["n2n3_h"][order] * 60.0
    shown = [r for r in REASONS if q[f"n2n3_invalid_min_{r}"].sum() > 0]
    ax = idf.name(axes_at(fig, BARS_AXES), "bars")
    xs = np.arange(len(order))
    base = np.zeros(len(order))
    for r in shown:
        pct = 100.0 * q[f"n2n3_invalid_min_{r}"][order] / total
        ax.bar(xs, pct, bottom=base, width=0.8, color=REASONS[r][1], lw=0, label=r)
        base += pct
    assert np.allclose(base, 100.0 - q["valid_n2n3_pct"][order], atol=1e-3), "reasons do not sum to the invalid share"
    ax.axhline(100.0 - THRESHOLD, color="0.3", lw=0.6, ls=(0, (3, 2)), label="threshold")
    ax.text(len(xs) - 0.5, 100.0 - THRESHOLD + 0.6, f"{THRESHOLD:g}% valid", ha="right", va="bottom",
            fontsize=FONT_LABEL, color="0.3", gid="threshold-label")
    below = q["below_80pct_valid_n2n3"][order]
    for x, i, b, top in zip(xs, q["id"][order], below, base):
        if b and ids:
            ax.text(x - 0.4, top + 0.6, pad(i), ha="left", va="bottom", fontsize=FONT_LABEL, gid=f"id-{pad(i)}")
    ax.set_xlim(-0.8, len(xs) - 0.2)
    ax.set_ylim(0, float(np.ceil((base.max() + 4.0) / 5.0) * 5.0))
    ax.set_xticks([])
    ax.set_xlabel(f"Children (n = {len(xs)}), sorted by % of N2 + N3 valid")
    ax.set_ylabel("N2 + N3 invalid (%)")
    thin_spines(ax, hide=("top", "right", "bottom"))
    ax.tick_params(axis="y", length=2)
    ax.legend(handles=[Patch(color=REASONS[r][1], label=REASONS[r][0]) for r in shown], loc="upper right",
              ncol=len(shown), frameon=False, handlelength=1.0, columnspacing=1.2, borderaxespad=0.0)
    ax.set_title(f"Invalid {SPO2_BOLD} per child, by first reason", loc="left", fontsize=FONT_TITLE)


def nadir(fig, ids: bool = False) -> None:
    """Lowest valid 1-s SpO2 in sleep against the lowest scored desaturation (dots) and the clinical-report nadir
    (triangles), identity line; the report nadirs suspected artifact are filled and any scored nadir more than
    OUTLIER % below the signal nadir is ringed, and both are labelled with the child's id when ``ids`` (internal
    figures only)."""
    n = read_table(TABLE_DIR / "oximetry_nadir.csv")
    x = n["signal_nadir_valid"]
    ax = idf.name(axes_at(fig, NADIR_AXES), "nadir")
    ax.plot([40, 100], [40, 100], label="identity", **IDENTITY)
    ok = np.isfinite(n["scored_nadir"])
    ax.scatter(x[ok], n["scored_nadir"][ok], s=9, color=SCATTER_COLOR, lw=0, label="scored", zorder=3)
    ok = np.isfinite(n["report_nadir"])
    ax.scatter(x[ok], n["report_nadir"][ok], s=12, marker="^", facecolor="none", edgecolor=REPORT_COLOR, lw=0.6,
               label="report", zorder=3)
    sus = n["suspect"]
    ax.scatter(x[sus], n["report_nadir"][sus], s=12, marker="^", color=REPORT_COLOR, lw=0.6, label="report-suspect",
               zorder=3)
    far = n["scored_nadir"] < x - OUTLIER  # a scored desaturation far below anything the valid signal reaches
    ax.scatter(x[far], n["scored_nadir"][far], s=40, facecolor="none", edgecolor=SCATTER_COLOR, lw=0.6,
               label="scored-far", zorder=3)
    if ids:
        for mask, y, color, gid in ((sus, n["report_nadir"], REPORT_COLOR, "id"),
                                    (far, n["scored_nadir"], SCATTER_COLOR, "id-scored")):
            for i, a, b in zip(n["id"][mask], x[mask], y[mask]):
                ax.annotate(pad(i), (a, b), xytext=(3, 0), textcoords="offset points", fontsize=FONT_LABEL,
                            va="center", color=color, gid=f"{gid}-{pad(i)}")
    ax.set_xlim(80, 96)
    ax.set_ylim(35, 100)
    ax.set_xlabel("Signal nadir, valid seconds (%)")
    ax.set_ylabel("Nadir (%)")
    ax.legend(handles=ax.collections[:2],  # one row under the title, over the axes, clear of every point
              labels=[f"Scored (n = {int(np.isfinite(n['scored_nadir']).sum())})", f"Report (n = {int(ok.sum())})"],
              loc="lower center", bbox_to_anchor=(0.57, ax.get_position().y1 + 0.01), bbox_transform=fig.transFigure,
              ncol=2, frameon=False, handlelength=1.0, handletextpad=0.1, columnspacing=1.2, borderaxespad=0.0)
    thin_spines(ax)
    ax.set_title(f"{SPO2_BOLD} nadir in sleep", fontsize=FONT_TITLE, pad=15)


def _log1p_axis(ax, axis: str) -> None:
    fwd, inv = (lambda v: np.log10(np.maximum(v, 0) + 1)), (lambda v: 10 ** v - 1)
    getattr(ax, f"set_{axis}scale")("function", functions=(fwd, inv))
    getattr(ax, f"set_{axis}ticks")(LOG_TICKS, [f"{v:g}" for v in LOG_TICKS])
    getattr(ax, f"set_{axis}lim")(0, 100)


def odi(fig, ids: bool = False) -> None:
    """Signal ODI (per hour of valid sleep) against the scored ODI (per hour of TST), one dot per child on
    log(1 + ODI) axes, identity line and Pearson r of log10(1 + ODI); open dots: < 80 % of N2 + N3 valid, labelled
    with the child's id when ``ids`` (internal figures only)."""
    q = read_table(TABLE_DIR / "oximetry_quality_children.csv")
    x, y, low = q["odi_scored"], q["odi_signal"], q["below_80pct_valid_n2n3"]
    ax = idf.name(axes_at(fig, AXES), "odi")
    ax.plot([0, 100], [0, 100], label="identity", **IDENTITY)
    ax.scatter(x[~low], y[~low], s=9, color=SCATTER_COLOR, lw=0, label="children")
    ax.scatter(x[low], y[low], s=9, facecolor="none", edgecolor=SCATTER_COLOR, lw=0.6, label="below-80")
    if ids:
        for i, a, b in zip(q["id"][low], x[low], y[low]):  # 006 sits in the cloud: led out to the empty lower right
            dx, dy = LABEL_OFFSET.get(int(i), (0, -3))
            ax.annotate(pad(i), (a, b), xytext=(dx, dy), textcoords="offset points", fontsize=FONT_LABEL, va="top",
                        ha="center" if dx == 0 else "left", gid=f"id-{pad(i)}",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4", shrinkA=1, shrinkB=2.5) if dx else None)
    _log1p_axis(ax, "x")
    _log1p_axis(ax, "y")
    r = float(np.corrcoef(np.log10(x + 1), np.log10(y + 1))[0, 1])
    ax.text(0.04, 0.96, f"$r$ = {r:.2f}\nn = {len(x)}", transform=ax.transAxes, ha="left", va="top",
            fontsize=FONT_LABEL, gid="r-value")
    ax.set_xlabel("Scored ODI (/h TST)")
    ax.set_ylabel("Signal ODI (/h valid sleep)")
    thin_spines(ax)
    ax.set_title("Oxygen desaturation index", fontsize=FONT_TITLE, pad=4)


def detection(fig) -> None:
    """Per child: the share of scored desaturations a signal desaturation overlaps (all, and those on valid seconds)
    and the share of signal desaturations a scored one overlaps (PPV); bar = median, above = pooled counts."""
    a = read_table(TABLE_DIR / "oximetry_desat_agreement.csv")
    cols = [("Sensitivity", a["sensitivity"], a["n_scored_detected"].sum() / a["n_scored"].sum()),
            ("Sensitivity,\nvalid", a["sensitivity_valid"],
             a["n_scored_valid_detected"].sum() / a["n_scored_valid"].sum()),
            ("PPV", a["ppv"], a["n_signal_confirmed"].sum() / a["n_signal"].sum())]
    ax = idf.name(axes_at(fig, DETECTION_AXES), "detection")
    rng = np.random.default_rng(0)  # jitter only
    for k, (name, v, pooled) in enumerate(cols):
        v = v[np.isfinite(v)] * 100
        ax.scatter(k + rng.uniform(-0.22, 0.22, len(v)), v, s=6, color=SCATTER_COLOR, alpha=0.7, lw=0,
                   label=f"children-{k}")
        ax.plot([k - 0.32, k + 0.32], [np.median(v)] * 2, color="black", lw=1.0, solid_capstyle="butt",
                label=f"median-{k}")
        ax.text(k, 104, f"{pooled * 100:.0f}%", ha="center", va="bottom", fontsize=FONT_LABEL, gid=f"pooled-{k}")
    ax.text(-0.6, 104, "Pooled", ha="right", va="bottom", fontsize=FONT_LABEL, color="0.35", gid="pooled-label")
    ax.set_xticks(range(len(cols)), [c[0] for c in cols])
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(0, 102)
    ax.set_ylabel("Per child (%)")
    thin_spines(ax)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Desaturation agreement", fontsize=FONT_TITLE, pad=12)


def reclass(fig) -> None:
    """Counted (N2/N3) hypopneas, scored-event class (rows) by signal class (columns): n and % of the row, shaded by
    the row %."""
    rc = read_table(TABLE_DIR / "oximetry_hypopnea_reclass.csv")
    n = np.zeros((len(SCORED), len(SIGNAL)))
    for s, g, k in zip(rc["scored_class"], rc["signal_class"], rc["n"]):
        n[SCORED.index(s), SIGNAL.index(g)] = k
    pct = 100 * n / n.sum(1, keepdims=True)
    ax = idf.name(axes_at(fig, HEAT_AXES), "heat")
    ax.imshow(pct, cmap=CMAP, vmin=0, vmax=110, aspect="auto", label="cells")
    for i in range(len(SCORED)):
        for j in range(len(SIGNAL)):
            dark = pct[i, j] > 55
            ax.text(j, i, f"{int(n[i, j])}\n{pct[i, j]:.0f}%", ha="center", va="center", fontsize=FONT_LABEL,
                    linespacing=1.0, color="white" if dark else "black", gid=f"cell-{SCORED[i]}-{SIGNAL[j]}")
        ax.text(len(SIGNAL) - 0.4, i, f"{int(n[i].sum())}", ha="left", va="center", fontsize=FONT_LABEL,
                color="0.35", gid=f"total-{SCORED[i]}")
    ax.text(len(SIGNAL) - 0.4, -0.75, "n", ha="left", va="bottom", fontsize=FONT_LABEL, color="0.35", gid="total-n")
    ax.set_yticks(range(len(SCORED)), [CLASS_NAME[c] for c in SCORED])
    ax.set_xticks(range(len(SIGNAL)), ["Desat.", "Arousal-\nonly", "Unclas-\nsified", "Unknown"])
    for t, c in zip(ax.get_yticklabels(), SCORED):
        t.set_color(CLASS_TEXT[c])
    ax.tick_params(length=0, pad=2)
    ax.xaxis.set_ticks_position("top")
    for tick, c in zip(ax.xaxis.get_major_ticks(), SIGNAL):  # the top labels are label2
        tick.label2.set_color(CLASS_TEXT[c])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-0.55, -0.6, "Scored ↓\nSignal →", ha="right", va="bottom", fontsize=FONT_LABEL, color="0.35",
            linespacing=1.0, gid="corner")
    height = fig.get_figheight() * 25.4
    fig.text(0.5, 1.0 - 1.0 / height, f"Hypopneas reclassified (n = {int(n.sum())})", ha="center", va="top",
             fontsize=FONT_TITLE, fontweight="bold", gid="title")
