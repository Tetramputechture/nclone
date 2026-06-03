"""Tests for curve_utils: verified walkable-curve orientations.

These lock the empirically-verified tile-id -> orientation mapping (convex
hump = [12][13], concave bowl = [17][16]) by asserting both the tile ids used
and that the resulting surface is actually traversable per the reachability
graph (the same geometry the physics/PBRS stack uses).
"""

from __future__ import annotations

import pytest

from nclone.constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from nclone.map_generation.map import Map
from nclone.map_generation import curve_utils as cu


def _flat_canvas() -> Map:
    m = Map(seed=1)
    m.reset()
    m.set_tiles_bulk([0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))
    return m


def _grid(m: Map):
    return [m.get_tile(x, y) for y in range(MAP_TILE_HEIGHT) for x in range(MAP_TILE_WIDTH)]


def test_curve_tile_constants_are_in_curved_range():
    # 12/13 convex quarter-circles, 16/17 concave quarter-pipes.
    assert cu.CONVEX_RISE == 12 and cu.CONVEX_FALL == 13
    assert cu.CONCAVE_DESCEND == 17 and cu.CONCAVE_ASCEND == 16
    assert cu.WALKABLE_CURVE_TILES == {12, 13, 16, 17}


def test_hump_places_convex_tiles():
    m = _flat_canvas()
    floor_row, base_row = 15, 18
    for x in range(MAP_TILE_WIDTH):
        m.set_tile(x, floor_row, 1)
    cu.create_convex_hump(m, 18, floor_row, base_row, top_width=1)
    grid = _grid(m)
    assert cu.CONVEX_RISE in grid and cu.CONVEX_FALL in grid


def test_bowl_places_concave_tiles():
    m = _flat_canvas()
    floor_row, base_row = 15, 18
    for x in range(MAP_TILE_WIDTH):
        m.set_tile(x, floor_row, 1)
    cu.create_concave_bowl(m, 18, floor_row, base_row, bottom_width=1)
    grid = _grid(m)
    assert cu.CONCAVE_DESCEND in grid and cu.CONCAVE_ASCEND in grid


@pytest.mark.parametrize("feature", ["bowl", "hump", "scurve"])
def test_curve_features_are_traversable(feature):
    """A curve feature with spawn on one side, switch+door on the other, is
    solvable per the reachability graph (orientation regression guard)."""
    from nclone.map_generation.solvability import is_map_solvable

    m = _flat_canvas()
    floor_row, base_row = 15, 18
    for x in range(MAP_TILE_WIDTH):
        for y in range(floor_row, base_row + 1):
            m.set_tile(x, y, 1)
    if feature == "bowl":
        cu.create_concave_bowl(m, 18, floor_row, base_row, bottom_width=2)
    elif feature == "hump":
        cu.create_convex_hump(m, 18, floor_row, base_row, top_width=2)
    else:
        cu.create_s_curve(m, 14, floor_row, base_row, bottom_width=1, top_width=1)
    sy = floor_row - 1
    m.set_ninja_spawn(3, sy, 1)
    m.add_entity(3, 38, sy, 0, 0, 34, sy)
    assert is_map_solvable(m.map_data())
