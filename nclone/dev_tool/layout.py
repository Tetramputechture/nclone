"""Dependency-free geometry for the dev panel.

Kept free of pygame draw calls (like :mod:`nclone.tas_tool.layout`) so the panel
renderer and any tests agree on the same constants. Per-row hit-testing is done
in :mod:`nclone.dev_tool.panel` because it depends on the live (dynamic) row
list; this module only fixes the panel's overall geometry.
"""

from typing import Tuple

# Width (px) of the reserved strip. Passed to ``RenderConfig(panel_width=...)``
# so the renderer left-aligns the game frame and leaves this column free.
PANEL_WIDTH = 320

# Inner layout metrics (screen px).
PAD = 8           # gap from the panel edges
ROW_H = 22        # height of a button / field row
HEADER_H = 20     # height of a section header row
LINE_H = 16       # height of a plain status text line
GAP = 3           # vertical gap between rows


def panel_rect(screen_w: int, screen_h: int) -> Tuple[int, int, int, int]:
    """Return (x, y, w, h) of the panel strip in screen coordinates."""
    x = max(0, screen_w - PANEL_WIDTH)
    return x, 0, min(PANEL_WIDTH, screen_w), screen_h


def content_x(screen_w: int) -> Tuple[int, int]:
    """Return (left_x, width) of the panel's inner content column."""
    px, _, pw, _ = panel_rect(screen_w, 0)
    return px + PAD, pw - 2 * PAD
