"""Curved-terrain authoring utilities for N++ map generation.

This module is the curved-tile analogue of :mod:`terrain_utils` (which only
authors straight slopes). It places the curved tiles -- convex quarter-circles
(ids 10-13) and concave quarter-pipes (ids 14-17) -- as *walkable* terrain so
generators can build features the ninja rides over (convex humps) or through
(concave bowls), training momentum control on curved surfaces.

Tile-orientation reference (VERIFIED against the authoritative collision
predicate ``graph.reachability.graph_builder._check_subnode_validity_simple``,
which is the same geometry the physics simulator and the surface/lattice
planner use -- the matplotlib renderer is y-flipped relative to physics and was
NOT trusted). Local tile coords are (0,0)=top-left, (24,24)=bottom-right, y
DOWN; "solid below, empty above" = a floor you stand on:

Convex quarter-circles (10-13): solid is the quarter-disc around the arc
centre; the exposed arc is the walkable surface.
  - 12 (centre bottom-right): surface ASCENDS left->right, peak at top-right.
  - 13 (centre bottom-left):  surface DESCENDS left->right, peak at top-left.
  => a convex HUMP (∩) the ninja rides over is ``[12][13]`` (12 left, 13 right),
     peaking at their shared edge, with solid filled beneath.

Concave quarter-pipes (14-17): solid is the corner opposite the arc centre;
the concave arc is the walkable surface.
  - 17 (centre top-right): surface DESCENDS left->right into the dip.
  - 16 (centre top-left):  surface ASCENDS left->right out of the dip.
  => a concave BOWL (∪) the ninja rides through is ``[17][16]`` (17 left, 16
     right), with the low point at their shared edge; widen by inserting a
     lowered flat floor between them.

The other four curved tiles (10, 11 = convex with solid on top; 14, 15 =
concave with solid on top) are ceiling/overhang pieces (solid above, empty
below) and are intentionally NOT used as floor terrain here.
"""

from typing import Optional

from ..constants import MAP_TILE_HEIGHT

# --- Verified walkable-curve tile constants -------------------------------
# Convex hump halves (ride over the bulge):
CONVEX_RISE = 12   # left half: surface ascends L->R, peak at top-right
CONVEX_FALL = 13   # right half: surface descends L->R, peak at top-left
# Concave bowl halves (ride through the dip):
CONCAVE_DESCEND = 17  # left half: surface descends L->R into the dip
CONCAVE_ASCEND = 16   # right half: surface ascends L->R out of the dip

SOLID_TILE = 1

# Set of curved tile ids this module can place, for tests / purity checks.
WALKABLE_CURVE_TILES = frozenset(
    {CONVEX_RISE, CONVEX_FALL, CONCAVE_DESCEND, CONCAVE_ASCEND}
)


def _fill_solid_column(map_instance, x: int, y_from: int, y_to: int) -> None:
    """Fill solid tiles in column ``x`` from row ``y_from``..``y_to`` inclusive.

    Only overwrites empty (type-0) cells so existing curve tiles are preserved.
    """
    if y_from > y_to:
        return
    for y in range(y_from, min(y_to, MAP_TILE_HEIGHT - 1) + 1):
        if map_instance.get_tile(x, y) == 0:
            map_instance.set_tile(x, y, SOLID_TILE)


def create_convex_hump(
    map_instance,
    x: int,
    floor_row: int,
    base_row: int,
    top_width: int = 0,
) -> int:
    """Place a convex hump (∩) the ninja must ride over.

    The surrounding flat floor is assumed to be a solid tile at ``floor_row``
    (ninja walks on its top edge). The hump rises one tile above that surface:
    the curved caps sit at ``floor_row - 1`` and the cells from ``floor_row``
    down to ``base_row`` are filled solid for support.

    Args:
        map_instance: Map being authored.
        x: Left column of the hump.
        floor_row: Row of the surrounding flat floor (solid).
        base_row: Lowest row to fill solid beneath the hump.
        top_width: Number of flat (solid) tiles between the rise and fall caps,
            giving a rounded-mesa shape. ``0`` => a minimal 2-wide hump.

    Returns:
        The x column immediately after the hump.
    """
    cap_row = floor_row - 1
    map_instance.set_tile(x, cap_row, CONVEX_RISE)
    cur = x + 1
    for _ in range(max(0, top_width)):
        map_instance.set_tile(cur, cap_row, SOLID_TILE)
        cur += 1
    map_instance.set_tile(cur, cap_row, CONVEX_FALL)
    cur += 1
    # Solid support beneath every column of the hump.
    for xx in range(x, cur):
        _fill_solid_column(map_instance, xx, floor_row, base_row)
    return cur


def create_concave_bowl(
    map_instance,
    x: int,
    floor_row: int,
    base_row: int,
    bottom_width: int = 0,
) -> int:
    """Place a concave bowl (∪) the ninja must ride through.

    The surrounding flat floor is a solid tile at ``floor_row``. The bowl dips
    one tile below that surface: the descend/ascend quarter-pipes sit at
    ``floor_row``, an optional lowered flat floor at ``floor_row + 1`` widens
    the basin, and everything beneath (down to ``base_row``) is solid.

    Args:
        map_instance: Map being authored.
        x: Left column of the bowl.
        floor_row: Row of the surrounding flat floor (solid).
        base_row: Lowest row to fill solid beneath the bowl.
        bottom_width: Number of lowered flat-floor tiles between the descend and
            ascend pipes. ``0`` => a minimal 2-wide bowl.

    Returns:
        The x column immediately after the bowl.
    """
    bottom_row = floor_row + 1
    map_instance.set_tile(x, floor_row, CONCAVE_DESCEND)
    cur = x + 1
    for _ in range(max(0, bottom_width)):
        map_instance.set_tile(cur, bottom_row, SOLID_TILE)
        cur += 1
    map_instance.set_tile(cur, floor_row, CONCAVE_ASCEND)
    cur += 1
    # Solid support beneath the basin (rows below the curve/floor tiles).
    for xx in range(x, cur):
        _fill_solid_column(map_instance, xx, bottom_row, base_row)
    return cur


def create_s_curve(
    map_instance,
    x: int,
    floor_row: int,
    base_row: int,
    start_with_bowl: bool = True,
    bottom_width: int = 0,
    top_width: int = 0,
) -> int:
    """Place a bowl immediately followed by a hump (or vice-versa).

    Produces an S-shaped run of curved terrain that forces the ninja to manage
    momentum down-then-up (or up-then-down) across adjacent curves.

    Returns the x column immediately after the composite feature.
    """
    if start_with_bowl:
        cur = create_concave_bowl(map_instance, x, floor_row, base_row, bottom_width)
        return create_convex_hump(map_instance, cur, floor_row, base_row, top_width)
    cur = create_convex_hump(map_instance, x, floor_row, base_row, top_width)
    return create_concave_bowl(map_instance, cur, floor_row, base_row, bottom_width)


def create_undulation(
    map_instance,
    x: int,
    floor_row: int,
    base_row: int,
    n_features: int,
    rng,
    max_x: Optional[int] = None,
) -> int:
    """Lay a sequence of alternating bowls and humps (rolling curved terrain).

    Each feature is a minimal 2-3 wide bowl or hump; flat floor between them is
    the caller's responsibility (this only places the curved features and their
    solid support). Stops early if it would exceed ``max_x``.

    Returns the x column immediately after the last placed feature.
    """
    cur = x
    use_bowl = rng.choice([True, False])
    for _ in range(max(0, n_features)):
        width = 2 + rng.randint(0, 1)  # 2 or 3 wide
        if max_x is not None and cur + width >= max_x:
            break
        if use_bowl:
            cur = create_concave_bowl(
                map_instance, cur, floor_row, base_row, bottom_width=width - 2
            )
        else:
            cur = create_convex_hump(
                map_instance, cur, floor_row, base_row, top_width=width - 2
            )
        use_bowl = not use_bowl
    return cur
