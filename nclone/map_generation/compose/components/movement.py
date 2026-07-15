"""Movement-expansion components: wide running-jump gaps and ceiling mines.

These extend the Phase-1/3 vocabulary (``flat_floor``, ``single_gap``,
``mine_field``, slopes, curves, ``tunnel``) with the two on-path movement
mechanics it lacked:

* ``long_gap``    — a gap too wide for a standing jump, so the ninja must build
  speed on the runway and *long-jump* (``single_gap`` is only 2-3 wide).
* ``ceiling_mine``— mines hung from a low roof over a walkable floor: harmless
  to walk under but they **cap jump height**, forcing low, controlled movement
  (the ``hazard:mine.ceiling`` tag had no component before this).

Both obey the composable-map contract: the floor returns to the shared baseline
at BOTH horizontal edges, so the canvas seam-carve never has to bridge a height
mismatch, and the playable corridor stays connected by construction. Only
baseline-legal entities are placed (toggle mines), so generated maps still pass
``subset_ok``.
"""

from __future__ import annotations

from ..component import Component, ComponentContext, IntRange, SOLID, EMPTY
from ..registry import register_component


@register_component(
    name="long_gap",
    tags={"terrain:gap", "move:jump.gap", "role:traverse"},
    params={"gap_width": IntRange(4, 6)},
    difficulty={"move": 3},
    min_w=12,
)
class LongGap(Component):
    """A wide floorless pit centred in the cell with a runway on each side.

    Wider than ``single_gap`` so a standing jump can't clear it — the ninja
    needs a running start. Kept interior (runway >= 3 tiles each side) so the
    seam-carve's boundary-column floor doesn't fill the pit.
    """

    def stamp(self, ctx: ComponentContext):
        ctx.floor()
        # leave at least a 3-tile runway on each side of the pit
        max_gap = max(2, ctx.width - 6)
        gw = min(int(ctx.p.gap_width), max_gap)
        gap_start = ctx.x0 + (ctx.width - gw) // 2
        for gx in range(gap_start, gap_start + gw):
            ctx.clear_column(gx, ctx.floor_row, ctx.base_row)
        ctx.set_edge("W", ctx.floor_row)
        ctx.set_edge("E", ctx.floor_row)
        return ctx.result()


@register_component(
    name="ceiling_mine",
    tags={"hazard:mine.ceiling", "move:walk", "role:traverse"},
    params={"mine_count": IntRange(1, 3), "clearance": IntRange(3, 4)},
    difficulty={"hazard": 3, "move": 1},
    min_w=8,
)
class CeilingMine(Component):
    """A walkable floor with mines hanging from a low solid roof.

    The floor stays continuous (walk straight through), but the low roof + the
    mines hung just beneath it cap the ninja's jump height, forcing controlled
    low movement. Mines are placed interior (margin from the cell edges) so the
    boundary-column head-clearance carve doesn't delete them.
    """

    def stamp(self, ctx: ComponentContext):
        ctx.floor()
        clearance = max(3, int(ctx.p.clearance))
        ceil_row = ctx.floor_row - clearance
        roof_top = ctx.region.y
        if ceil_row <= roof_top:
            ceil_row = roof_top + 1
        # solid roof across the band, filled up to the region top
        for gx in range(ctx.x0, ctx.x1 + 1):
            for gy in range(roof_top, ceil_row + 1):
                if ctx.get_tile(gx, gy) == EMPTY:
                    ctx.set_tile(gx, gy, SOLID)
        # mines hanging one tile below the roof into the passage
        for gx in ctx.spread_x(int(ctx.p.mine_count), margin=2):
            ctx.add_mine(gx, ceil_row + 1)
        ctx.set_edge("W", ctx.floor_row)
        ctx.set_edge("E", ctx.floor_row)
        return ctx.result()
