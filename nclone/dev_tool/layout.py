"""Dependency-free geometry for the dev panel.

Kept free of pygame draw calls (like :mod:`nclone.tas_tool.layout`) so the panel
renderer and any tests agree on the same constants. Per-row hit-testing is done
in :mod:`nclone.dev_tool.panel` because it depends on the live (dynamic) row
list; this module only fixes the panel's overall geometry and the scroll maths
that keep every control reachable when the rows outgrow the window height.
"""

from typing import Iterable, Tuple

from .controls import HEADER, STATUS, Row

# Width (px) of the reserved strip. Passed to ``RenderConfig(panel_width=...)``
# so the renderer left-aligns the game frame and leaves this column free.
PANEL_WIDTH = 320

# Inner layout metrics (screen px). Rows are deliberately compact: the row list
# grows with the map source (generator seed field, benchmark buckets, recording)
# and the window is only SRCHEIGHT (600) tall, so every px saved here is one
# fewer px of scrolling.
PAD = 8           # gap from the panel edges
ROW_H = 20        # height of a button / field row
HEADER_H = 18     # height of a section header row
LINE_H = 15       # height of a plain status text line
GAP = 2           # vertical gap between rows

# Scroll indicator drawn against the panel's right edge (px).
SCROLLBAR_W = 4
# Pixels scrolled per mouse-wheel notch (~3 rows).
SCROLL_STEP = 3 * (ROW_H + GAP)


def panel_rect(screen_w: int, screen_h: int) -> Tuple[int, int, int, int]:
    """Return (x, y, w, h) of the panel strip in screen coordinates."""
    x = max(0, screen_w - PANEL_WIDTH)
    return x, 0, min(PANEL_WIDTH, screen_w), screen_h


def content_x(screen_w: int) -> Tuple[int, int]:
    """Return (left_x, width) of the panel's inner content column.

    The scrollbar gutter is always reserved so row widths do not jump when the
    content becomes scrollable.
    """
    px, _, pw, _ = panel_rect(screen_w, 0)
    return px + PAD, pw - 2 * PAD - SCROLLBAR_W


def row_height(row: Row) -> int:
    """Vertical space (px) one row consumes, including its trailing gap."""
    if row.kind == HEADER:
        return HEADER_H
    if row.kind == STATUS:
        return LINE_H
    return ROW_H + GAP


def content_height(rows: Iterable[Row]) -> int:
    """Total height (px) of the row list, including top and bottom padding."""
    return sum(row_height(row) for row in rows) + 2 * PAD


def max_scroll(rows: Iterable[Row], view_h: int) -> int:
    """Largest valid scroll offset (0 when everything already fits)."""
    return max(0, content_height(rows) - int(view_h))
