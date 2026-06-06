"""Phase-3 terrain components: varying-degree slopes, walkable curves, and
ceiling/wall connectors (tunnels).

Design rule for robustness: every path component returns its surface to the
shared baseline at BOTH horizontal edges (a slope is a symmetric hill, a curve
returns to baseline, a tunnel only adds a roof). That keeps consecutive cells
trivially connected so the carve never has to bridge a height mismatch on these.
"""

from __future__ import annotations

from ..component import Component, ComponentContext, IntRange, Choice, SOLID, EMPTY
from ..registry import register_component
from .. import connectors
from ... import terrain_utils


# --- step / ledge -------------------------------------------------------------
@register_component(
    name="step_up",
    tags={"terrain:step", "move:jump.stepup", "move:fall.drop", "role:traverse"},
    params={"height": IntRange(1, 2)},
    difficulty={"move": 2},
    min_w=8,
)
class StepUp(Component):
    """A raised platform (step up, flat, step down) -- returns to baseline."""

    def stamp(self, ctx: ComponentContext):
        ctx.floor()
        h = int(ctx.p.height)
        top = ctx.floor_row - h
        x0 = ctx.x0 + ctx.width // 3
        x1 = ctx.x1 - ctx.width // 3
        for gx in range(x0, x1 + 1):
            ctx.set_tile(gx, top, SOLID)
            ctx.fill_support(gx, top + 1)
            for gy in range(max(1, top - 3), top):
                ctx.set_tile(gx, gy, EMPTY)
        ctx.set_edge("W", ctx.floor_row)
        ctx.set_edge("E", ctx.floor_row)
        return ctx.result()


# --- varying-degree slopes (symmetric hills) ----------------------------------
class _Hill(Component):
    KIND = "mild"

    def stamp(self, ctx: ComponentContext):
        ctx.floor()
        h = int(ctx.p.height)
        sx = ctx.x0 + 2
        mx = ctx.x1 - 1
        if self.KIND == "45":
            terrain_utils.create_45_degree_hill(ctx.map, sx, ctx.floor_row, h, max_x=mx)
        else:
            terrain_utils.create_mixed_hill(
                ctx.map, sx, ctx.floor_row, h,
                ascent_type=self.KIND, descent_type=self.KIND, max_x=mx,
            )
        ctx.set_edge("W", ctx.floor_row)
        ctx.set_edge("E", ctx.floor_row)
        return ctx.result()


@register_component(
    name="slope_mild",
    tags={"terrain:slope.mild", "move:slope.ascend", "move:slope.descend", "role:traverse"},
    params={"height": IntRange(1, 2)},
    difficulty={"move": 1},
    min_w=10,
)
class SlopeMild(_Hill):
    KIND = "mild"


@register_component(
    name="slope_steep",
    tags={"terrain:slope.steep", "move:slope.ascend", "move:slope.descend", "role:traverse"},
    params={"height": IntRange(2, 3)},
    difficulty={"move": 2},
    min_w=8,
)
class SlopeSteep(_Hill):
    KIND = "steep"


@register_component(
    name="slope_45",
    tags={"terrain:slope.45", "move:slope.ascend", "move:slope.descend", "role:traverse"},
    params={"height": IntRange(1, 2)},
    difficulty={"move": 2},
    min_w=8,
)
class Slope45(_Hill):
    KIND = "45"


# --- walkable curves ----------------------------------------------------------
@register_component(
    name="convex_hump",
    tags={"terrain:curve.convex", "move:curve.ride", "role:traverse"},
    params={"top_width": IntRange(0, 2)},
    difficulty={"move": 2},
    min_w=6,
)
class ConvexHump(Component):
    def stamp(self, ctx: ComponentContext):
        ctx.floor()
        ctx.hump(ctx.x0 + 2, top_width=int(ctx.p.top_width))
        ctx.set_edge("W", ctx.floor_row)
        ctx.set_edge("E", ctx.floor_row)
        return ctx.result()


@register_component(
    name="concave_bowl",
    tags={"terrain:curve.concave", "move:curve.ride", "role:traverse"},
    params={"bottom_width": IntRange(0, 2)},
    difficulty={"move": 2},
    min_w=6,
)
class ConcaveBowl(Component):
    def stamp(self, ctx: ComponentContext):
        ctx.floor()
        ctx.bowl(ctx.x0 + 2, bottom_width=int(ctx.p.bottom_width))
        ctx.set_edge("W", ctx.floor_row)
        ctx.set_edge("E", ctx.floor_row)
        return ctx.result()


# --- tunnel: a low roof using ceiling/wall connectors -------------------------
@register_component(
    name="tunnel",
    tags={"terrain:tunnel", "move:walk", "role:traverse"},
    params={"roof": Choice(["slope", "hump", "dip", "flat"])},
    difficulty={"move": 2},
    min_w=8,
)
class Tunnel(Component):
    """Flat floor with a low solid roof carrying a sloped / curved ceiling."""

    def stamp(self, ctx: ComponentContext):
        ctx.floor()
        ceil_row = ctx.floor_row - 3  # ~2-tile passage under the roof
        if ceil_row < ctx.region.y + 1:
            ceil_row = ctx.region.y + 1
        roof_top = ctx.region.y
        x = ctx.x0 + 1
        kind = ctx.p.roof
        if kind == "hump":
            connectors.ceiling_hump(ctx.map, x, ceil_row, width=2, roof_top=roof_top)
            ctx.tag("terrain:curve.convex")
        elif kind == "dip":
            connectors.ceiling_dip(ctx.map, x, ceil_row, width=2, roof_top=roof_top)
            ctx.tag("terrain:curve.concave")
        elif kind == "slope":
            connectors.ceiling_slope(ctx.map, x, ceil_row, height=2, descend=True,
                                     roof_top=roof_top)
            ctx.tag("terrain:slope.45")
        # fill the rest of the roof flat across the band
        for gx in range(ctx.x0, ctx.x1 + 1):
            if ctx.get_tile(gx, ceil_row) == EMPTY:
                ctx.set_tile(gx, ceil_row, SOLID)
            for gy in range(roof_top, ceil_row):
                if ctx.get_tile(gx, gy) == EMPTY:
                    ctx.set_tile(gx, gy, SOLID)
        ctx.set_edge("W", ctx.floor_row)
        ctx.set_edge("E", ctx.floor_row)
        return ctx.result()


# --- off-path decoration ------------------------------------------------------
@register_component(
    name="decoration_mines",
    tags={"role:decoration", "hazard:mine.static"},
    params={"mine_count": IntRange(1, 2)},
    difficulty={},
    min_w=4,
    min_h=3,
)
class DecorationMines(Component):
    """Off-path distractor: a solid shelf with a couple of (unreachable) mines."""

    def stamp(self, ctx: ComponentContext):
        shelf = ctx.region.y1
        for gx in range(ctx.x0, ctx.x1 + 1):
            for gy in range(shelf, ctx.region.y1 + 1):
                ctx.set_tile(gx, gy, SOLID)
        for gx in ctx.spread_x(int(ctx.p.mine_count), margin=1):
            ctx.add_mine(gx, shelf - 1)
        return ctx.result()
