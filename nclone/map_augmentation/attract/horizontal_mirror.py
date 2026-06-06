"""Horizontal-mirror augmentation for N++ attract files.

N++ physics are perfectly symmetric about the vertical centre-line of the
level.  Reflecting every tile, entity and the ninja spawn left-to-right, and
simultaneously swapping Left/Right input bits, produces a demonstration that
is byte-for-byte equivalent to a new level that the simulator will complete
successfully.

Coordinate conventions (raw map-data units)
-------------------------------------------
Tile grid   : 42 wide x 23 tall; index = col + row * 42.  col 0 = left edge.
Entity / spawn x : map-data units.  Multiply by 6 to get full-map pixels.
Full map width    : 44 tiles * 4 map-data-units/tile = 176 map-data units.

Tile-type mirror map
--------------------
Derived from nclone/map_augmentation/mirror_map.py.  Tile types that have
left/right chirality are swapped; symmetric types (0,1,2,4) are left alone.
"""

from __future__ import annotations

import copy
from typing import Dict, List

from .base_augmentation import AttractData, BaseAugmentation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAP_TILE_WIDTH: int = 42
_MAP_TILE_HEIGHT: int = 23
_TILE_COUNT: int = _MAP_TILE_WIDTH * _MAP_TILE_HEIGHT  # 966

# Full map width in map-data units (44 tiles * 4 units/tile).
# Entity and spawn x-coordinates use this space.
_FULL_MAP_WIDTH_UNITS: int = 176

# Tile type to mirrored tile type.  Missing entries are symmetric (self-map).
_TILE_MIRROR: Dict[int, int] = {
    # Half tiles – vertical orientation flips
    3: 5,
    5: 3,
    # 45-degree slopes
    6: 7,
    7: 6,
    8: 9,
    9: 8,
    # Quarter-moon (concave curves)
    10: 11,
    11: 10,
    12: 13,
    13: 12,
    # Quarter-pipes (convex curves)
    14: 16,
    16: 14,
    15: 17,
    17: 15,
    # Short mild slopes
    18: 19,
    19: 18,
    20: 21,
    21: 20,
    # Raised mild slopes
    22: 23,
    23: 22,
    24: 25,
    25: 24,
    # Short steep slopes
    26: 28,
    28: 26,
    27: 29,
    29: 27,
    # Raised steep slopes
    30: 32,
    32: 30,
    31: 33,
    33: 31,
}

# Input byte mirror: swap bit1 (right=0x02) and bit2 (left=0x04).
# bit0 = jump (unchanged); conflicting inputs 6/7 map to themselves.
_INPUT_MIRROR: Dict[int, int] = {
    0: 0,  # noop   -> noop
    1: 1,  # jump   -> jump
    2: 4,  # right  -> left
    3: 5,  # R+jump -> L+jump
    4: 2,  # left   -> right
    5: 3,  # L+jump -> R+jump
    6: 6,  # invalid -> invalid (unchanged)
    7: 7,  # invalid -> invalid (unchanged)
}


# ---------------------------------------------------------------------------
# Public augmentation class
# ---------------------------------------------------------------------------


class HorizontalMirror(BaseAugmentation):
    """Reflect the level and inputs horizontally.

    The augmentation is an exact involution (applying it twice gives back the
    original), which means the mirrored demonstration is guaranteed to succeed
    whenever the original does.
    """

    @property
    def name(self) -> str:
        return "hmirror"

    def augment(self, data: AttractData) -> AttractData:
        """Return a horizontally mirrored copy of *data*.

        Args:
            data: Decoded attract-file data as returned by
                  ``NppAttractDecoder.decode_npp_attract_file()``.

        Returns:
            New AttractData with mirrored tiles, entities, spawn and inputs.

        Raises:
            ValueError: If ``data['tiles']`` does not contain exactly 966 values.
        """
        tiles: List[int] = data["tiles"]
        if len(tiles) != _TILE_COUNT:
            raise ValueError(
                f"Expected {_TILE_COUNT} tiles, got {len(tiles)}"
            )

        return {
            "tiles": _mirror_tiles(tiles),
            "entities": _mirror_entities(data["entities"]),
            "ninja_spawn": _mirror_spawn(data["ninja_spawn"]),
            # Header bytes are kept verbatim (contains level metadata that the
            # encoder only copies; it has no effect on simulation).
            "header": data["header"],
            "demo_data": _mirror_inputs(data["demo_data"]),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mirror_tiles(tiles: List[int]) -> List[int]:
    """Return a 966-element list with columns reflected and types swapped."""
    mirrored = [0] * _TILE_COUNT
    for row in range(_MAP_TILE_HEIGHT):
        for col in range(_MAP_TILE_WIDTH):
            src_idx = col + row * _MAP_TILE_WIDTH
            tile_type = tiles[src_idx]
            mirrored_type = _TILE_MIRROR.get(tile_type, tile_type)
            dst_col = _MAP_TILE_WIDTH - 1 - col
            dst_idx = dst_col + row * _MAP_TILE_WIDTH
            mirrored[dst_idx] = mirrored_type
    return mirrored


def _mirror_entity(entity: dict) -> dict:
    """Return a new entity dict with the x-coordinate reflected.

    For all entity types present in filtered attract files (toggle mines,
    gold, exit door, exit switch) the entity is circular so only the
    x-coordinate needs to be updated.
    """
    mirrored = dict(entity)
    # Reflect x around the centre of the full 44-tile-wide map.
    mirrored["x"] = _FULL_MAP_WIDTH_UNITS - entity["x"]
    return mirrored


def _mirror_entities(entities: List[dict]) -> List[dict]:
    """Return a new list of mirrored entity dicts (order preserved)."""
    return [_mirror_entity(e) for e in entities]


def _mirror_spawn(spawn: tuple) -> tuple:
    """Return (mirrored_x, y) for the ninja spawn position."""
    x, y = spawn
    return (_FULL_MAP_WIDTH_UNITS - x, y)


def _mirror_inputs(inputs: List[int]) -> List[int]:
    """Return a new input sequence with L/R swapped each frame."""
    return [_INPUT_MIRROR.get(inp, inp) for inp in inputs]


# ---------------------------------------------------------------------------
# Module-level convenience: expose the mapping tables for testing / inspection
# ---------------------------------------------------------------------------

TILE_MIRROR_MAP: Dict[int, int] = _TILE_MIRROR
INPUT_MIRROR_MAP: Dict[int, int] = _INPUT_MIRROR
FULL_MAP_WIDTH_UNITS: int = _FULL_MAP_WIDTH_UNITS
