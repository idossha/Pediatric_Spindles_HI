"""Figure 2: Cluster-based permutation test

Draws only source_data/fig-2/ (summarized data; nothing is fitted here).
Build: ``idofig build fig-2`` (or ./reproduce.sh).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

import idofig as idf
import matplotlib as mpl
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from plotting.style import (
    FONT_LABEL,
    FONT_LETTER,
    FONT_SUBSCRIPTED,
    LOG10,
    N2_COLOR,
    SCATTER_COLOR,
    axes_at,
    colorbar_unit,
    house_style,
)

SCHEMATIC_LEFT = 8.25  # mm from the panel's left edge to the artwork's origin
ENVELOPE_COLOR = "#d94801"  # the permutation loop's arrow and count


# Laid out in the millimetres of the artwork it redraws: x from SCHEMATIC_LEFT, y down from the artwork's top (the panel starts 5.2 mm above it: room for the letter
# and the headings' first line). The marks are illustrative, read off that artwork, not results.
SCHEMATIC_FONT = FONT_LABEL
BLUE, GREY, FILL, TAIL, HIST = "#1a5276", "#555555", "#f4f6f7", "#800000", N2_COLOR
# Column 1: points and fit line on the unit square of the scatter's axes.
MODEL_POINTS = [(132, 376), (127, 438), (108, 518), (210, 516), (248, 498), (254, 541), (289, 519),
                (273, 588), (306, 576), (304, 653), (400, 631), (440, 641), (416, 676), (474, 661),
                (448, 713), (518, 618), (512, 720), (520, 822), (634, 697), (620, 747), (632, 747),
                (651, 803), (686, 805), (702, 817), (767, 888), (763, 903)]  # artwork px, box 70..805 x 350..930
MODEL_FIT = [(100, 463), (775, 828)]
# Column 2: electrodes as (x, y) in head radii from the centre (y down), their t, and whether
# the cluster marks them (a black dot, as on the paper's maps): every marked one past the two-sided
# p < .05 critical t at the HI models' 58 residual df (the caption's threshold), none else.
HEAD_DOTS = [(-0.29, -0.74, -2.6, True), (-0.02, -0.80, -3.3, True), (0.25, -0.75, -2.5, True),
             (0.01, -0.57, -2.4, True), (0.26, -0.50, -2.2, True), (-0.23, -0.50, -1.6, False),
             (-0.53, -0.59, -1.2, False), (0.49, -0.63, -1.0, False), (-0.69, -0.38, -0.4, False),
             (-0.43, -0.36, -0.8, False), (-0.06, -0.29, -1.4, False), (0.15, -0.25, -1.0, False),
             (0.44, -0.34, -0.7, False), (0.67, -0.42, -0.3, False), (-0.78, -0.12, -0.3, False),
             (-0.54, -0.14, -0.4, False), (-0.24, -0.17, -0.5, False), (0.28, -0.07, -0.8, False),
             (0.54, -0.12, -0.2, False), (0.77, -0.17, 0.3, False), (-0.28, 0.05, -0.2, False),
             (-0.00, -0.01, -0.4, False), (-0.54, 0.10, 0.3, False), (-0.78, 0.15, -0.2, False),
             (0.24, 0.15, -0.2, False), (0.54, 0.13, -0.2, False), (0.78, 0.10, 0.3, False),
             (-0.16, 0.23, 0.4, False), (0.06, 0.27, -0.2, False), (-0.45, 0.33, -0.1, False),
             (-0.67, 0.40, -0.1, False), (0.43, 0.35, 0.2, False), (0.69, 0.37, -0.1, False),
             (-0.26, 0.48, -0.3, False), (-0.01, 0.55, -0.2, False), (0.23, 0.50, -0.1, False),
             (-0.49, 0.61, 0.3, False), (0.52, 0.58, 0.4, False), (-0.25, 0.74, 0.3, False),
             (0.02, 0.78, -0.2, False), (0.29, 0.72, -0.1, False)]
CRITICAL_T = 2.0017  # scipy.stats.t.ppf(0.975, 58)
assert all((abs(t) > CRITICAL_T) == marked for _, _, t, marked in HEAD_DOTS)
# Column 4: the null distribution's bar heights, read off the artwork. The threshold is the
# drawn null's 95th percentile, the first bin edge with at most 5 % of the mass beyond it, as
# "sig. if p_cluster < 0.05" says: the artwork's 26.5 left 1.1 % beyond, and the red tail
# there was under 0.8 mm tall.
NULL_COUNTS = [73, 301, 438, 578, 635, 665, 618, 635, 605, 518, 478, 428, 368, 318, 258, 238, 213,
               183, 148, 118, 98, 80, 68, 55, 48, 35, 33, 26, 20, 13, 8, 7, 5, 4, 4, 3, 2, 1]
THRESHOLD = next(i for i in range(len(NULL_COUNTS))
                 if sum(NULL_COUNTS[i:]) <= 0.05 * sum(NULL_COUNTS)) - 0.5
OBSERVED = 33.8


def schematic(fig) -> None:
    w, h = fig.get_figwidth() * 25.4, fig.get_figheight() * 25.4
    left, top = SCHEMATIC_LEFT, h - 5.2   # the artwork's origin, in panel mm from bottom-left

    def at(x, y, width, height, **kw):  # an axes at artwork mm (x, y the top-left corner)
        return axes_at(fig, (left + x, top - y - height, width, height), **kw)

    ax = idf.name(fig.add_axes((0, 0, 1, 1)), "schematic")
    ax.set_xlim(-left, w - left)
    ax.set_ylim(top, top - h)  # y down, in mm
    ax.axis("off")
    font = {"fontsize": SCHEMATIC_FONT}
    # Black, as every other title in the paper; blue stays on the marks it labels.
    heading = {"fontsize": 8, "fontweight": "bold", "ha": "center", "va": "center", "linespacing": 1.2}
    # Regular, as the fourth column's formula and rule are: the 8 pt bold headings are the emphasis.
    caption = {"fontsize": SCHEMATIC_FONT, "ha": "center", "va": "center", "linespacing": 1.35}

    def arrow(x0, x1, y, gid):
        ax.annotate("", xy=(x1, y), xytext=(x0, y), gid=gid,
                    arrowprops={"arrowstyle": "-|>,head_length=0.5,head_width=0.3", "color": GREY, "lw": 1.6,
                                    "shrinkA": 0, "shrinkB": 0})

    for letter, x in zip("abcd", (0.5, 39.5, 80.0, 117.0)):  # one letter per step, left of its heading
        ax.text(x, -4.4, letter, ha="left", va="top", fontsize=FONT_LETTER, fontweight="bold", gid=f"letter-{letter}")

    # 1  the channel-wise model at one electrode
    ax.text(18.0, -1.6, "Channel-wise\nmodel", gid="model-heading", **heading)
    ax.text(18.0, 5.2, f"$y$ = $\\beta_0$ + $\\beta_1$ {LOG10}(HI + 1)\n+ $\\beta_2$ Age + $\\beta_3$ Sex + $\\varepsilon$",
            ha="center", va="center", linespacing=1.4, gid="model-formula", fontsize=FONT_SUBSCRIPTED)
    sc = idf.name(at(4.0, 10.0, 28.0, 24.5), "model")
    u = [(x - 70) / 735 for x, _ in MODEL_POINTS]
    v = [1 - (y - 350) / 580 for _, y in MODEL_POINTS]
    sc.scatter(u, v, s=10, color=SCATTER_COLOR, alpha=0.7, edgecolors="white", linewidths=0.4, zorder=3,
               gid="model-points")
    sc.plot([(x - 70) / 735 for x, _ in MODEL_FIT], [1 - (y - 350) / 580 for _, y in MODEL_FIT],
            color=ENVELOPE_COLOR, lw=1.6, zorder=4, gid="model-fit")
    sc.set_xlim(0, 1)
    sc.set_ylim(0, 1)
    sc.set_xticks([])
    sc.set_yticks([])
    for spine in sc.spines.values():
        spine.set_color(GREY)
        spine.set_linewidth(0.9)
    sc.set_xlabel(f"{LOG10}(HI + 1)", labelpad=2, fontsize=FONT_SUBSCRIPTED)
    sc.set_ylabel("Spindle metric", labelpad=2, **font)
    sc.text(0.96, 0.95, "one electrode", transform=sc.transAxes, ha="right", va="top", style="italic",
            fontweight="bold", color=GREY, gid="one-electrode", **font)
    ax.text(18.0, 45.2, "extract $t$($\\beta_1$) at each\nof 172 channels", gid="model-caption",
            **{**caption, "fontsize": FONT_SUBSCRIPTED})
    arrow(35.0, 38.6, 22.0, "arrow-1")

    # 2  the observed t-map and its cluster
    ax.text(56.0, -1.6, "Observed $\\mathbfit{t}$-map\n→ cluster", gid="map-heading", **heading)
    cx, cy, r = 55.0, 23.5, 12.0
    ax.add_patch(Polygon([(cx - 0.16 * r, cy - 0.985 * r), (cx, cy - 1.19 * r), (cx + 0.16 * r, cy - 0.985 * r)],
                         closed=False, fill=False, ec="#333333", lw=1.0, gid="map-nose"))
    for side in (-1, 1):
        ear = [(cx + side * r * (1.0 + dx), cy + r * dy) for dx, dy in
               ((-0.005, -0.14), (0.06, -0.12), (0.10, -0.05), (0.10, 0.05), (0.06, 0.12), (-0.005, 0.14))]
        ax.add_patch(Polygon(ear, closed=False, fill=False, ec="#333333", lw=1.0,
                             gid=f"map-ear-{'left' if side < 0 else 'right'}"))
    ax.add_patch(Circle((cx, cy), r, fill=False, ec="#333333", lw=1.0, gid="map-head"))
    norm = mpl.colors.Normalize(-3.5, 3.5)
    cmap = mpl.colormaps["RdBu_r"]
    for k, (dx, dy, t, marked) in enumerate(HEAD_DOTS):
        ax.add_patch(Circle((cx + dx * r, cy + dy * r), 0.62, fc=cmap(norm(t)), ec="none", zorder=3,
                            gid=f"map-electrode-{k + 1}"))
        if marked:
            ax.add_patch(Circle((cx + dx * r, cy + dy * r), 0.34, fc="black", ec="none", zorder=4,
                                gid=f"map-cluster-{k + 1}"))
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), cax=at(69.6, 16.5, 1.6, 14.0), ticks=[-3, 0, 3])
    bar.ax.tick_params(labelsize=SCHEMATIC_FONT, width=0.4, length=1.5, pad=1.0)
    bar.outline.set_linewidth(0.6)
    bar.ax.set_gid("map-scale")
    colorbar_unit(bar, "$t$")
    # Σ (U+03A3), not the n-ary ∑ (U+2211) Arial lacks, which fell back to another face.
    ax.text(cx, 45.2, "cluster mass = Σ|$t$|\n(Delaunay adjacency)", gid="map-caption", **caption)
    arrow(76.2, 79.6, 22.0, "arrow-2")

    # 3  the Freedman-Lane permutation loop
    bx, bw, bh = 85.2, 27.0, 7.4
    tops = [4.4, 14.4, 24.4, 34.4]
    ax.text(bx + bw / 2, -1.6, "Freedman–Lane\npermutation", gid="loop-heading", **heading)
    for k, (y, text) in enumerate(zip(tops, ("Fit reduced model\n(Age + Sex)", "Permute residuals\nacross participants",
                                              "Add back → refit full;\nrecompute", "Store max\ncluster mass"))):
        ax.add_patch(FancyBboxPatch((bx, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.8", fc=FILL, ec=BLUE,
                                    lw=0.9, gid=f"loop-step-{k + 1}"))
        ax.text(bx + bw / 2, y + bh / 2, text, ha="center", va="center", linespacing=1.2,
                gid=f"loop-step-{k + 1}-text", **font)
        if k:
            ax.annotate("", xy=(bx + bw / 2, y), xytext=(bx + bw / 2, tops[k - 1] + bh), gid=f"loop-arrow-{k}",
                        arrowprops={"arrowstyle": "-|>,head_length=0.4,head_width=0.22", "color": GREY, "lw": 1.1,
                                        "shrinkA": 0, "shrinkB": 0})
    # Back to step 2, not step 1: Freedman-Lane fits the reduced model once, and only the
    # permute / refit / store steps repeat. Into step 1, the loop said it was refitted 5,000 times.
    loop_x = bx - 2.0
    ax.plot([bx, loop_x, loop_x], [tops[3] + bh / 2, tops[3] + bh / 2, tops[1] + bh / 2], color=ENVELOPE_COLOR,
            lw=1.3, solid_joinstyle="miter", gid="loop-back")
    ax.annotate("", xy=(bx, tops[1] + bh / 2), xytext=(loop_x - 0.33, tops[1] + bh / 2), gid="loop-back-head",
                arrowprops={"arrowstyle": "-|>,head_length=0.45,head_width=0.25", "color": ENVELOPE_COLOR, "lw": 1.3,
                                "shrinkA": 0, "shrinkB": 0})
    # 1.9 mm left of the line: at 1.3 the comma's tail came within 0.06 mm of it.
    ax.text(loop_x - 1.9, (tops[1] + tops[3] + bh) / 2, "× 5,000", rotation=90, ha="center", va="center",
            color=ENVELOPE_COLOR, fontweight="bold", gid="loop-count", **font)
    arrow(114.6, 118.0, 22.0, "arrow-3")

    # 4  the null distribution of the maximum cluster mass, and p_cluster
    null_x, null_w = 122.5, 20.0
    null_c = null_x + null_w / 2  # heading, formula and rule centred on the histogram, as columns 1 and 3 are
    ax.text(null_c, -1.6, "Null distribution\n→ $\\mathbfit{p}_{\\mathbf{cluster}}$", gid="null-heading", **heading)
    hx = idf.name(at(null_x, 5.0, null_w, 29.5), "null")
    bins = range(len(NULL_COUNTS))
    for i, bar in enumerate(hx.bar(bins, NULL_COUNTS, width=0.9, color=[HIST if i < THRESHOLD else TAIL for i in bins], lw=0)):
        bar.set_gid(f"null-bar-{i + 1}")
    # The dotted line stops under its label, centred over it: at the 95th percentile the label,
    # right-aligned left of the line as the artwork had it, ran 1.1 mm across the y axis.
    hx.axvline(THRESHOLD, ymax=0.8, color=BLUE, lw=0.9, ls=(0, (1.5, 1.5)), gid="null-threshold")
    hx.axvline(OBSERVED, color=BLUE, lw=1.8, gid="null-observed")
    hx.set_xlim(-1.5, 40.5)
    hx.set_ylim(0, max(NULL_COUNTS) * 1.45)  # the top band holds the two labels
    hx.set_xticks([])
    hx.set_yticks([])
    for side in ("top", "right"):
        hx.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        hx.spines[side].set_color(GREY)
        hx.spines[side].set_linewidth(0.9)
    hx.set_xlabel("Max cluster mass (permuted)", labelpad=2, **font)
    hx.set_ylabel("Frequency", labelpad=2, **font)
    top_y = max(NULL_COUNTS) * 1.45
    hx.text(THRESHOLD, top_y * 0.9, "sig.\nthreshold", ha="center", va="center", color=BLUE, linespacing=1.2,
            gid="null-threshold-label", **font)
    hx.annotate("observed\ncluster mass", xy=(OBSERVED + 0.4, top_y * 0.55), xytext=(OBSERVED + 2.0, top_y * 0.88),
                ha="left", va="center", color=BLUE, linespacing=1.2, gid="null-observed-label",
                arrowprops={"arrowstyle": "-|>,head_length=0.35,head_width=0.18", "color": BLUE, "lw": 0.9,
                                "shrinkA": 2, "shrinkB": 0}, **font)
    # The rule on the baseline of columns 1 and 2's second caption line, the formula 1.4 mm
    # above it: at 42.6 and 49.2 the rule hung 2.2 mm below every other caption.
    ax.text(null_c, 41.6, "$p_{\\mathrm{cluster}}$ = $\\dfrac{\\#\\{\\mathrm{perm\\ max} \\geq \\mathrm{obs}\\} + 1}{5{,}000 + 1}$",
            ha="center", va="center", gid="null-formula", **font)
    # "sig. if p_corr = < 0.05" in the artwork: "= <" was a typo for "<".
    ax.text(null_c, 47.0, "sig. if $p_{\\mathrm{cluster}}$ < 0.05", ha="center", va="center", gid="null-rule", **font)


p = idf.panels("schematic", layout=None)
house_style()
schematic(p["schematic"])
idf.save(p)
