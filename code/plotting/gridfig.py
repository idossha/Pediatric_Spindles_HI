"""Cell layout for the grid topomap figures (fig-3, fig-s4); library.

Every head is the same size and each column's cells share one head-axes rectangle, so heads align down columns and
along rows. Reserves are in mm and only where a column draws beside the head: the row label left of column 0, the
colorbar right of the last column, the column title and panel letters on top. figure.json places each cell at its size
(``place.w``/``place.h``); ``idofig.panels()`` hands the script a panel of exactly that size.

Layouts: ``NARROW_FAST``/``NARROW_SLOW`` (fig-3, two columns across 85 mm), ``COMPACT_*`` (fig-s4, four columns
across 178 mm).
Title and row label are figure-level text pinned to the head's box; panel letters come from figure.json.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .style import AXES_LW, FONT_LABEL, FONT_TITLE, colorbar_unit

MM_PER_IN = 25.4


@dataclass(frozen=True)
class Cell:
    """One grid cell's mm geometry and the head-axes fractions derived from it."""

    width_mm: float
    height_mm: float
    left_mm: float    # row label (col 0 only); reserved, empty elsewhere
    right_mm: float   # colorbar + its tick numbers (last col only); reserved, empty elsewhere
    top_mm: float     # column title (row 0 only)
    bottom_mm: float

    @property
    def width_in(self) -> float:
        return self.width_mm / MM_PER_IN

    @property
    def height_in(self) -> float:
        return self.height_mm / MM_PER_IN

    @property
    def left(self) -> float:
        return self.left_mm / self.width_mm

    @property
    def right(self) -> float:
        return 1 - self.right_mm / self.width_mm

    @property
    def bottom(self) -> float:
        return self.bottom_mm / self.height_mm

    @property
    def top(self) -> float:
        return 1 - self.top_mm / self.height_mm

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def head_diam_mm(self) -> float:
        # mne keeps an equal aspect, so the head is limited by the smaller box dimension
        return min((self.right - self.left) * self.width_mm, (self.top - self.bottom) * self.height_mm)


GUTTER = 0.889  # between cells, and from the canvas edge
TOP = 4.7  # title row: a 10 pt letter at the cell's top edge shares its baseline with an 8 pt title


def _column_cells(total_mm: float, reserves: list[tuple[float, float]], bottom_mm: float) -> list[Cell]:
    """Cells of one row across ``total_mm``, one per (left, right) reserve, with equal square heads."""
    head = (total_mm - GUTTER * (len(reserves) + 1) - sum(l + r for l, r in reserves)) / len(reserves)
    return [Cell(width_mm=round(l + head + r, 3), height_mm=round(TOP + head + bottom_mm, 3), left_mm=l,
                 right_mm=r, top_mm=TOP, bottom_mm=bottom_mm) for l, r in reserves]


# fig-3: the fast column's right reserve is the gap between columns; the slow column has a 1.2 mm left margin.
# A bar reserve is what the bar draws (0.6 mm gap, 1.4 mm bar, "12.5" with tick and pad: 8.1 mm) plus 0.5 mm spare.
# Each letter stands 5.7 mm left of its head (figure.json letter_offset_mm), clear of its column's title.
NARROW_FAST, NARROW_SLOW = _column_cells(85.0, [(13.0, 8.6), (1.2, 8.6)], bottom_mm=1.5)
# fig-s4: 4 mm between a head and the next cell; 4.5 mm beneath the head for its p_cluster.
COMPACT_FIRST, COMPACT_MID, _, COMPACT_LAST = _column_cells(
    178.0, [(13.0, 4.0), (4.0, 4.0), (4.0, 4.0), (4.0, 9.5)], bottom_mm=4.5)


TITLE_FONTSIZE = FONT_TITLE  # the size of every other panel title


def fit(fig, cell: Cell) -> Cell:
    """The cell at the panel's figure.json size; a resized panel keeps the reserves in mm and the head takes the rest."""
    w, h = fig.get_figwidth() * MM_PER_IN, fig.get_figheight() * MM_PER_IN
    if abs(w - cell.width_mm) <= 0.05 and abs(h - cell.height_mm) <= 0.05:
        return cell
    return replace(cell, width_mm=w, height_mm=h)


def grid_axes(fig, cell: Cell):
    """The head axes of a cell, in the panel ``fig`` idofig made at its figure.json size."""
    cell = fit(fig, cell)
    return fig.add_axes([cell.left, cell.bottom, cell.right - cell.left, cell.top - cell.bottom])


def add_row_label(fig, text: str, cell: Cell) -> None:
    """Row label in the left reserve (column 0 only), anchored 0.5 mm left of the head."""
    cell = fit(fig, cell)
    x = cell.left - 0.5 / cell.width_mm
    y = (cell.bottom + cell.top) / 2
    fig.text(x, y, text, transform=fig.transFigure, fontsize=FONT_LABEL, fontweight="bold",
             ha="right", va="center", linespacing=1.2, gid="row-label")


def add_title(fig, text: str, cell: Cell) -> None:
    """Column title (row 0 only), centred 1.5 mm above the head."""
    cell = fit(fig, cell)
    gap = 1.5 / cell.height_mm
    fig.text(cell.center_x, cell.top + gap, text, transform=fig.transFigure,
             fontsize=TITLE_FONTSIZE, fontweight="bold", ha="center", va="bottom", gid="col-title")


def add_row_colorbar(fig, im, label: str | None, cell: Cell):
    """One colorbar per row in the right reserve, as tall as the head as drawn (not the reserve); the unit sits above (``colorbar_unit``), or in the row label when ``label`` is None."""
    cell = fit(fig, cell)
    head = im.axes
    head.apply_aspect()
    box = head.get_position()
    cax = fig.add_axes([cell.right + 0.6 / cell.width_mm, box.y0, 1.4 / cell.width_mm, box.height])
    import idofig

    cb = idofig.colorbar(im, cax=cax, name="scale")
    cb.ax.tick_params(labelsize=FONT_LABEL, width=0.4, length=2, pad=1.5)
    cb.outline.set_linewidth(AXES_LW)
    if label:
        colorbar_unit(cb, label)
    return cb
