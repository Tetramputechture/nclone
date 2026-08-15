"""Authoritative coordinate + glyph conversions for the composable map system.

Every producer/consumer of map geometry should route through THIS module so the
grid<->units<->pixels<->tile relationship lives in exactly one place. The current
``map_visualizer.py`` does ``units // 6`` and never subtracts the per-entity
store-time offset, which mis-places the ninja (and would mis-place any
offset-bearing entity). The functions here are the fix.

The relationship (verified against ``map.Map._to_map_data_units`` and
``ninja.py``)::

    grid  -> units : units = grid * GRID_SIZE_FACTOR (4) + offset
    units -> pixels: px    = units * 6
    => 1 tile = 24 px = 4 units; the ninja-spawn offset of 6 units = 1.5 tiles.

Entity store-time offsets (applied inside ``Map.add_entity`` /
``Map.set_ninja_spawn``) are PER ENTITY TYPE: ninja / exit door / exit switch /
locked door / gold carry +6, and a toggle mine carries 0. Recovering a grid
coordinate therefore requires subtracting the *type-appropriate* offset before
dividing -- a single global offset would be wrong for mines.

The +6 decomposes as 4 units of border (map_data tile ``(i,j)`` is sim tile
``(i+1,j+1)``) plus 2 units of half-tile, so an integer grid coordinate names the
CENTRE of that tile. Gold used to carry -6 and therefore landed three tiles
up-and-left of everything else placed at the same coordinate.
"""

from __future__ import annotations

from typing import Dict, Tuple

from ..constants import GRID_SIZE_FACTOR  # 4.0
from ...constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT

UNITS_PER_TILE = int(GRID_SIZE_FACTOR)  # 4
PX_PER_UNIT = 6
PX_PER_TILE = UNITS_PER_TILE * PX_PER_UNIT  # 24

# --- map_data header indices (mirrors map.Map.map_data / from_map_data) -------
TILE_DATA_START = 184
TILE_DATA_END = 1150  # exclusive; 966 = 42*23 tiles
EXIT_DOOR_COUNT = 1156
DEATH_BALL_COUNT = 1200
NINJA_X = 1231
NINJA_Y = 1232
NINJA_ORIENTATION = 1233
ENTITY_START = 1235
ENTITY_STRIDE = 5  # ints per standard entity record

# --- per-entity-type store-time offsets in map-data units ---------------------
# Matches the offsets applied in map.Map (NINJA_SPAWN_OFFSET_UNITS etc.).
ENTITY_OFFSET_UNITS: Dict[str, int] = {
    "ninja": 6,
    "exit_door": 6,
    "exit_switch": 6,
    "gold": 6,
    "toggle_mine": 0,
    "locked_door": 6,
    "locked_door_switch": 6,
}

# Map raw entity-type ids (in the serialized record) to the offset key above.
ENTITY_TYPE_TO_NAME: Dict[int, str] = {
    0: "ninja",
    1: "toggle_mine",
    21: "toggle_mine",  # toggled variant, same (zero) offset
    2: "gold",
    3: "exit_door",
    4: "exit_switch",
    6: "locked_door",
    7: "locked_door_switch",
    8: "locked_door",  # trap door, same +6 offset
}


def grid_to_units(gx: float, gy: float, offset: int = 0) -> Tuple[float, float]:
    """Grid (tile) coordinate -> map-data units."""
    return gx * UNITS_PER_TILE + offset, gy * UNITS_PER_TILE + offset


def units_to_grid(ux: float, uy: float, offset: int = 0) -> Tuple[float, float]:
    """Map-data units -> grid (tile) coordinate, subtracting ``offset`` first."""
    return (ux - offset) / UNITS_PER_TILE, (uy - offset) / UNITS_PER_TILE


def units_to_px(u: float) -> float:
    """Map-data units -> world pixels."""
    return u * PX_PER_UNIT


def entity_units_to_grid(entity_type: int, ux: float, uy: float) -> Tuple[float, float]:
    """Recover the grid coordinate of a stored entity, offset-aware.

    ``entity_type`` is the raw id from the serialized record (1=mine, 3=door,
    4=switch, ...). Unknown types are treated as zero-offset.
    """
    name = ENTITY_TYPE_TO_NAME.get(int(entity_type), "toggle_mine")
    offset = ENTITY_OFFSET_UNITS.get(name, 0)
    return units_to_grid(ux, uy, offset)


def tile_index(gx: int, gy: int) -> int:
    """Flat index into the tile array for grid (gx, gy)."""
    return gx + gy * MAP_TILE_WIDTH


# --- 1:1 tile glyph map (all 34 real tile ids distinct, for recoverable ASCII) -
# The structured ``describe_level`` carries exact tile ids; these glyphs only
# need to be UNIQUE so a reader can recover the id from the legend. Floor
# (walkable) pieces get the visually-suggestive characters; ceiling/overhang
# variants get a distinct (lowercase / alternate) glyph so floor 8 "/" is never
# confused with ceiling 7 "/".
TILE_GLYPH: Dict[int, str] = {
    0: ".",  # empty
    1: "#",  # full solid
    2: '"',  # top-half solid
    3: "]",  # right-half solid
    4: "_",  # bottom-half solid
    5: "[",  # left-half solid
    # 45-degree diagonals -- 8/9 are walkable floor, 6/7 are ceiling/overhang.
    6: "r",  # ceiling "\" (solid upper-right)
    7: "l",  # ceiling "/" (solid upper-left)
    8: "/",  # floor ascend
    9: "\\",  # floor descend
    # convex quarter-circles -- 12/13 walkable hump, 10/11 ceiling bulge.
    10: "a",  # ceiling convex (open bottom-right)
    11: "b",  # ceiling convex (open bottom-left)
    12: "c",  # floor convex rise (hump left half)
    13: "C",  # floor convex fall (hump right half)
    # concave quarter-pipes -- 16/17 walkable bowl, 14/15 ceiling pipe.
    14: "f",  # ceiling concave (solid top-left)
    15: "g",  # ceiling concave (solid top-right)
    16: "h",  # floor concave ascend (bowl right half)
    17: "H",  # floor concave descend (bowl left half)
    # mild slopes (18-25)
    18: "i", 19: "j", 20: "k", 21: "m", 22: "n", 23: "o", 24: "p", 25: "q",
    # steep slopes (26-33)
    26: "s", 27: "t", 28: "u", 29: "w", 30: "x", 31: "y", 32: "z", 33: "@",
}


def tile_glyph(tile_id: int) -> str:
    """1:1 glyph for a tile id ('?' for the unused/glitched 34-37)."""
    return TILE_GLYPH.get(int(tile_id), "?")
