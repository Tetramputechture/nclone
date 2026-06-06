"""Phase-1 components: flat floor, single gap, mine field, and a filler.

Each is a single ``@register_component`` decorator + one ``stamp`` method that
paints through the region-clamped :class:`ComponentContext`.
"""

from __future__ import annotations

from ..component import Component, ComponentContext, IntRange, SOLID, EMPTY
from ..registry import register_component


@register_component(
    name="flat_floor",
    tags={"terrain:flat", "move:walk", "role:traverse"},
    difficulty={"move": 0},
)
class FlatFloor(Component):
    def stamp(self, ctx: ComponentContext):
        ctx.floor()
        ctx.set_edge("W", ctx.floor_row)
        ctx.set_edge("E", ctx.floor_row)
        return ctx.result()


@register_component(
    name="single_gap",
    tags={"terrain:gap", "move:jump.gap", "role:traverse"},
    params={"gap_width": IntRange(2, 3)},
    difficulty={"move": 2},
    min_w=6,
)
class SingleGap(Component):
    def stamp(self, ctx: ComponentContext):
        ctx.floor()
        gw = min(int(ctx.p.gap_width), max(2, ctx.width - 4))
        gap_start = ctx.x0 + (ctx.width - gw) // 2
        for gx in range(gap_start, gap_start + gw):
            ctx.clear_column(gx, ctx.floor_row, ctx.base_row)
        ctx.set_edge("W", ctx.floor_row)
        ctx.set_edge("E", ctx.floor_row)
        return ctx.result()


@register_component(
    name="mine_field",
    tags={"hazard:mine.field", "move:walk", "role:traverse"},
    params={"mine_count": IntRange(1, 3)},
    difficulty={"hazard": 3, "move": 0},
    min_w=5,
)
class MineField(Component):
    def stamp(self, ctx: ComponentContext):
        ctx.floor()
        for gx in ctx.spread_x(int(ctx.p.mine_count), margin=2):
            ctx.add_mine(gx, ctx.floor_row - 1)
        ctx.set_edge("W", ctx.floor_row)
        ctx.set_edge("E", ctx.floor_row)
        return ctx.result()


@register_component(
    name="filler_solid",
    tags={"role:filler"},
    difficulty={},
    min_w=1,
    min_h=1,
)
class FillerSolid(Component):
    """Off-path filler: solid block within the cell (never required)."""

    def stamp(self, ctx: ComponentContext):
        for gx in range(ctx.x0, ctx.x1 + 1):
            for gy in range(ctx.region.y, ctx.region.y1 + 1):
                ctx.set_tile(gx, gy, SOLID)
        return ctx.result()
