"""Geometry connector emitters: slopes, curves, and ceiling/wall joins.

Every function emits VERIFIED tile ids and returns the new x cursor (and the
implied fine tag, so a component auto-tags itself). Walkable orientations are
reused from the already-verified ``curve_utils`` / ``terrain_utils`` helpers;
the ceiling/overhang pieces (diagonals 6/7, convex 10/11, concave 14/15 -- all
"solid above") are emitted here with the support filled UPWARD toward the roof,
and are guarded by ``tests/test_connectors.py`` which samples each emitted tile
against ``graph_builder._check_subnode_validity_simple`` (the authoritative
collision predicate) so an orientation regression is caught immediately.

Tile-id cheat-sheet (floor = walkable / solid-below, ceiling = solid-above):
  diagonals : floor 8 "/" ascend, 9 "\\" descend ; ceiling 6, 7
  convex    : floor 12 (rise) 13 (fall)           ; ceiling 10, 11
  concave   : floor 17 (descend) 16 (ascend)      ; ceiling 14, 15
"""

from __future__ import annotations

from typing import Tuple

from ...constants import MAP_TILE_HEIGHT
from .. import terrain_utils, curve_utils

SOLID = 1
EMPTY = 0

# ceiling/overhang tile ids (solid above) -- not placed as floor terrain.
CEIL_DIAG_SOLID_UL = 7   # "/" traversable lower-left, solid upper-right
CEIL_DIAG_SOLID_UR = 6   # "\" traversable lower-right, solid upper-left
CEIL_CONVEX_L = 11
CEIL_CONVEX_R = 10
CEIL_CONCAVE_L = 14
CEIL_CONCAVE_R = 15


def _fill_support(m, gx: int, top: int, base: int) -> None:
    for gy in range(int(top), int(base) + 1):
        if m.get_tile(gx, gy) == EMPTY:
            m.set_tile(gx, gy, SOLID)


def _fill_ceiling(m, gx: int, bottom: int, top: int = 1) -> None:
    """Fill solid UPWARD from ``bottom`` to ``top`` (toward the roof)."""
    for gy in range(int(top), int(bottom) + 1):
        if m.get_tile(gx, gy) == EMPTY:
            m.set_tile(gx, gy, SOLID)


# --- floor slopes (varying degree) -------------------------------------------
def slope_up(m, gx: int, gy: int, height: int, kind: str = "45",
             base_row: int = MAP_TILE_HEIGHT - 2, x_max: int = None) -> Tuple[int, int, str]:
    """Ascending floor slope. Returns (new_x, surface_row, tag)."""
    height = max(1, int(height))
    if kind == "45":
        nx = terrain_utils.create_45_degree_slope_up(m, gx, gy, height, max_x=x_max)
        tag = "terrain:slope.45"
    elif kind == "steep":
        nx = terrain_utils.create_slope_up(m, gx, gy, height, use_mild=False, max_x=x_max)
        tag = "terrain:slope.steep"
    else:  # mild
        nx = terrain_utils.create_slope_up(m, gx, gy, height, use_mild=True, max_x=x_max)
        tag = "terrain:slope.mild"
    surf = gy - height
    for cx in range(gx, max(gx + 1, nx)):
        _fill_support(m, cx, gy + 1, base_row)
    return nx, surf, tag


def slope_down(m, gx: int, gy: int, height: int, kind: str = "45",
               base_row: int = MAP_TILE_HEIGHT - 2, x_max: int = None) -> Tuple[int, int, str]:
    """Descending floor slope. Returns (new_x, surface_row, tag)."""
    height = max(1, int(height))
    if kind == "45":
        nx = terrain_utils.create_45_degree_slope_down(m, gx, gy, height, max_x=x_max)
        tag = "terrain:slope.45"
    elif kind == "steep":
        nx = terrain_utils.create_slope_down(m, gx, gy, height, use_mild=False, max_x=x_max)
        tag = "terrain:slope.steep"
    else:
        nx = terrain_utils.create_slope_down(m, gx, gy, height, use_mild=True, max_x=x_max)
        tag = "terrain:slope.mild"
    surf = gy + height
    for cx in range(gx, max(gx + 1, nx)):
        _fill_support(m, cx, surf + 1, base_row)
    return nx, surf, tag


# --- walkable curves (reuse verified curve_utils) -----------------------------
def convex_hump(m, gx: int, floor_row: int, base_row: int, top_width: int = 0,
                x_max: int = None) -> Tuple[int, str]:
    nx = curve_utils.create_convex_hump(m, gx, floor_row, base_row, top_width=top_width)
    return nx, "terrain:curve.convex"


def concave_bowl(m, gx: int, floor_row: int, base_row: int, bottom_width: int = 0,
                 x_max: int = None) -> Tuple[int, str]:
    nx = curve_utils.create_concave_bowl(m, gx, floor_row, base_row, bottom_width=bottom_width)
    return nx, "terrain:curve.concave"


# --- ceiling / wall connectors (solid-above tiles; orientation test-guarded) --
def ceiling_slope(m, gx: int, ceil_row: int, height: int, descend: bool = True,
                  roof_top: int = 1) -> Tuple[int, str]:
    """A sloped roof that tapers ``height`` tiles over its run (solid above).

    ``descend`` True => roof drops left->right (uses tile 6, solid upper-left so
    the open face is lower-right). Returns (new_x, tag).
    """
    height = max(1, int(height))
    cur_x, cur_row = int(gx), int(ceil_row)
    tile = CEIL_DIAG_SOLID_UR if descend else CEIL_DIAG_SOLID_UL
    for _ in range(height):
        m.set_tile(cur_x, cur_row, tile)
        _fill_ceiling(m, cur_x, cur_row - 1, roof_top)
        cur_x += 1
        cur_row += 1 if descend else -1
    return cur_x, "terrain:tunnel"


def ceiling_hump(m, gx: int, ceil_row: int, width: int = 0,
                 roof_top: int = 1) -> Tuple[int, str]:
    """Convex roof bulge dipping DOWN into the tunnel (solid above). Tiles 11,10."""
    m.set_tile(gx, ceil_row, CEIL_CONVEX_L)
    cur = gx + 1
    for _ in range(max(0, width)):
        m.set_tile(cur, ceil_row, SOLID)
        cur += 1
    m.set_tile(cur, ceil_row, CEIL_CONVEX_R)
    cur += 1
    for cx in range(gx, cur):
        _fill_ceiling(m, cx, ceil_row - 1, roof_top)
    return cur, "terrain:curve.convex"


def ceiling_dip(m, gx: int, ceil_row: int, width: int = 0,
                roof_top: int = 1) -> Tuple[int, str]:
    """Concave roof curving UP away from the tunnel (solid above). Tiles 14,15."""
    m.set_tile(gx, ceil_row, CEIL_CONCAVE_L)
    cur = gx + 1
    for _ in range(max(0, width)):
        m.set_tile(cur, ceil_row, SOLID)
        cur += 1
    m.set_tile(cur, ceil_row, CEIL_CONCAVE_R)
    cur += 1
    for cx in range(gx, cur):
        _fill_ceiling(m, cx, ceil_row - 1, roof_top)
    return cur, "terrain:curve.concave"


def wall_column(m, gx: int, top: int, bottom: int) -> None:
    """Solid vertical wall in column ``gx`` from ``top`` to ``bottom`` (for shafts)."""
    for gy in range(int(top), int(bottom) + 1):
        m.set_tile(gx, gy, SOLID)
