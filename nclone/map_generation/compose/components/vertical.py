"""Vertical-descent components: sheer drops and wall-jump slot climbs.

The Phase-1/3 vocabulary (``flat_floor``, ``single_gap``, ``long_gap``,
``mine_field``, slopes, curves) is purely horizontal -- no component demands a
controlled FALL or an enclosed wall-jump climb. These add that missing axis:

* ``drop_shaft``  — a 45-ramp climb to a crest that ends in a SHEER cliff back
  down to the baseline: the ninja must fall ``drop_h`` tiles and land in
  control (the crest's east face is a clean vertical wall to brake against).
* ``wall_chain``  — two solid pillars enclosing a 2-tile-wide slot entered
  through a ground-level doorway: the ninja wall-jump-chains up the slot, tops
  out on a pillar, and drops down the far side (the validated wall-jumpable
  shaft width -- see the loop_corridor corridor_width=2 precedent).

Both are strictly interior: the floor returns to the shared baseline at BOTH
horizontal edges, so the canvas seam-carve never has to bridge a height
mismatch. Both declare ``min_h`` high enough that they only fit FULL-HEIGHT
cells (single-row templates), so the existing 2x3 templates can never pick
them even via a permissive filter. No entities are placed, so ``subset_ok``
is preserved.

PHYSICS: an uncontrolled walk-off free-fall survives up to 10 tiles (death
from 11, see ``tests/test_precision_route_generators.py``); ``drop_h`` is
capped well below that so an unbraked drop stays survivable -- the
*braking-mandatory* class lives in the dedicated ``drop_landing`` generator.
"""

from __future__ import annotations

from ..component import Component, ComponentContext, IntRange, SOLID
from ..registry import register_component


@register_component(
    name="drop_shaft",
    tags={"move:fall.drop", "move:fall.control", "role:traverse"},
    params={"drop_h": IntRange(3, 6)},
    difficulty={"move": 2},
    min_w=12,
    min_h=12,
)
class DropShaft(Component):
    """A ramp-up to a crest ending in a sheer cliff down to the baseline.

    West -> east: flat entry, a 45-degree ramp climbing ``drop_h`` tiles, a
    2-tile crest, then the sheer east face straight back down to the baseline
    floor (the landing zone). The route *must* fall the full ``drop_h``.
    """

    def stamp(self, ctx: ComponentContext):
        # Baseline floor across the cell (entry + landing zone).
        ctx.floor()
        # Clamp so the ramp + 2-tile crest + >= 3-tile landing all fit and the
        # crest keeps >= 2 rows of headroom inside the cell.
        drop_h = max(
            2,
            min(
                int(ctx.p.drop_h),
                ctx.width - 8,
                ctx.floor_row - (ctx.region.y + 2),
            ),
        )
        crest_row = ctx.floor_row - drop_h
        # 45-ramp up from the baseline (also fills its own support).
        nx = ctx.slope_up(ctx.x0 + 2, ctx.floor_row, drop_h, "45")
        # 2-tile crest, solid to the base: its east face is the sheer cliff.
        for gx in range(nx, min(nx + 2, ctx.x1 - 2)):
            ctx.set_tile(gx, crest_row, SOLID)
            ctx.fill_support(gx, crest_row + 1)
        ctx.set_edge("W", ctx.floor_row)
        ctx.set_edge("E", ctx.floor_row)  # slope_up moved E; reset to baseline
        return ctx.result()


@register_component(
    name="wall_chain",
    tags={"move:walljump.chain", "terrain:shaft", "role:traverse"},
    params={"h": IntRange(4, 6)},
    difficulty={"move": 4},
    min_w=12,
    min_h=12,
)
class WallChain(Component):
    """Two pillars enclosing a 2-wide wall-jump slot, entered at ground level.

    Layout (interior to the cell; both edges stay at the baseline floor)::

        AA..BB      A = near pillar (2 wide) with a 2-tall doorway at the
        AA..BB          bottom; the ninja walks under it into the slot.
        ....BB      . = the 2-wide slot: wall-jump chain up between the
        AA..BB          pillar faces, top out on a pillar, walk over B and
        --------        drop down its far side back to the floor.
    """

    def stamp(self, ctx: ComponentContext):
        ctx.floor()
        h = max(3, min(int(ctx.p.h), ctx.floor_row - (ctx.region.y + 2)))
        top_row = ctx.floor_row - h
        cx = ctx.x0 + ctx.width // 2
        near = (cx - 2, cx - 1)  # pillar A columns (doorway at the bottom)
        far = (cx + 2, cx + 3)  # pillar B columns (solid to the floor)
        # Pillar A: solid from the top down to 3 above the floor -- rows
        # floor-1 / floor-2 stay empty as the ground-level doorway.
        for gx in near:
            for gy in range(top_row, ctx.floor_row - 2):
                ctx.set_tile(gx, gy, SOLID)
        # Pillar B: solid from the top all the way to the floor.
        for gx in far:
            for gy in range(top_row, ctx.floor_row):
                ctx.set_tile(gx, gy, SOLID)
        ctx.set_edge("W", ctx.floor_row)
        ctx.set_edge("E", ctx.floor_row)
        return ctx.result()
