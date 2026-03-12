"""Augmentation that translates the reachable area to a new map position.

N++ physics are position-invariant: gravity, collision response, and all game
mechanics depend only on the ninja's position *relative* to the surrounding
geometry, not on absolute coordinates.  Shifting the entire playable region
(tiles + entities + spawn) by an integer number of tiles therefore produces a
level where the original input sequence is guaranteed to succeed.

The vacated tiles left behind after translation are filled with random tile
types (0–33) because the ninja can never reach them.

Translation limits
------------------
The reachable+buffer bounding box is computed in tile coordinates.  The
maximum translation in each axis direction is constrained so the bounding box
stays within the playable 42×23 grid.  Both horizontal and vertical offsets are
chosen uniformly at random from the feasible set, excluding the identity (0, 0).

Usage
-----
>>> aug = ReachableAreaTranslator(seed=0)
>>> new_data = aug.augment(attract_data)
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from .base_augmentation import AttractData, BaseAugmentation
from .reachability_augment_utils import (
    _FULL_MAP_HEIGHT_UNITS,
    _FULL_MAP_WIDTH_UNITS,
    _MAP_TILE_HEIGHT,
    _MAP_TILE_WIDTH,
    _MAP_UNITS_PER_TILE,
    _TILE_COUNT,
    get_protected_tiles,
    get_reachable_bounding_box,
    get_reachable_tiles,
)

_VALID_TILE_TYPES: List[int] = list(range(34))


class ReachableAreaTranslator(BaseAugmentation):
    """Translate the reachable region to a random valid position in the map.

    The augmentation:

    1. Computes the reachable+buffer bounding box.
    2. Picks a random non-identity (dx, dy) offset in tile units such that the
       bounding box remains inside the 42×23 playable grid after translation.
    3. Shifts every tile inside the bounding box to its new location.
    4. Fills all tiles outside the new bounding box position with random types.
    5. Translates entity and ninja-spawn coordinates by the same offset.
    6. Copies demo_data unchanged.

    If the bounding box already spans the entire playable grid in both axes
    (no valid non-identity translation exists), the original data is returned
    unchanged as a no-op.

    Args:
        seed:   Optional RNG seed for reproducible outputs.
        buffer: Safety buffer in tiles around the reachable area (default 1).
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        buffer: int = 1,
    ) -> None:
        self._rng = random.Random(seed)
        self._buffer = buffer

    @property
    def name(self) -> str:
        return "translate"

    def augment(self, data: AttractData) -> AttractData:
        """Return a copy of *data* with the reachable region translated.

        Args:
            data: Decoded attract-file data.

        Returns:
            New AttractData with shifted tiles, entities, and ninja_spawn.

        Raises:
            ValueError: If ``data['tiles']`` does not contain exactly 966 values.
        """
        tiles: List[int] = data["tiles"]
        if len(tiles) != _TILE_COUNT:
            raise ValueError(
                f"ReachableAreaTranslator: expected {_TILE_COUNT} tiles, "
                f"got {len(tiles)}"
            )

        reachable = get_reachable_tiles(data)
        protected = get_protected_tiles(reachable, buffer=self._buffer)
        min_col, min_row, max_col, max_row = get_reachable_bounding_box(protected)

        # Maximum tile offset in each direction before the bounding box would
        # exceed the playable grid boundaries.
        max_left = min_col
        max_right = (_MAP_TILE_WIDTH - 1) - max_col
        max_up = min_row
        max_down = (_MAP_TILE_HEIGHT - 1) - max_row

        if max_left == 0 and max_right == 0 and max_up == 0 and max_down == 0:
            # Bounding box already fills the entire grid; nothing to translate.
            return self._copy_unchanged(data)

        dx, dy = self._pick_translation(max_left, max_right, max_up, max_down)

        if dx == 0 and dy == 0:
            return self._copy_unchanged(data)

        new_tiles = self._translate_tiles(tiles, min_col, min_row, max_col, max_row, dx, dy)
        new_entities = self._translate_entities(data["entities"], dx, dy)
        new_spawn = self._translate_spawn(data["ninja_spawn"], dx, dy)

        return {
            "tiles": new_tiles,
            "entities": new_entities,
            "ninja_spawn": new_spawn,
            "header": data["header"],
            "demo_data": list(data["demo_data"]),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_translation(
        self,
        max_left: int,
        max_right: int,
        max_up: int,
        max_down: int,
    ) -> Tuple[int, int]:
        """Sample a non-identity (dx, dy) from the feasible rectangle."""
        # Rejection-sample until we get a non-zero offset.  The probability
        # of drawing (0, 0) is 1 / ((max_left+max_right+1) * (max_up+max_down+1)),
        # which is at most 1/2 given that at least one axis has room to move.
        while True:
            dx = self._rng.randint(-max_left, max_right)
            dy = self._rng.randint(-max_up, max_down)
            if dx != 0 or dy != 0:
                return dx, dy

    def _translate_tiles(
        self,
        tiles: List[int],
        min_col: int,
        min_row: int,
        max_col: int,
        max_row: int,
        dx: int,
        dy: int,
    ) -> List[int]:
        """Build a new tile list with the bounding-box region shifted by (dx, dy).

        All tiles outside the new bounding box position are filled with random
        tile types (0–33).  Tiles inside the source bounding box are copied to
        their shifted destinations; any source tile that maps outside the grid
        after shifting is silently dropped (this should not occur when dx/dy
        respect the translation limits, but is guarded for robustness).
        """
        new_tiles = [self._rng.choice(_VALID_TILE_TYPES)] * _TILE_COUNT

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                new_col = col + dx
                new_row = row + dy
                if 0 <= new_col < _MAP_TILE_WIDTH and 0 <= new_row < _MAP_TILE_HEIGHT:
                    src_idx = col + row * _MAP_TILE_WIDTH
                    dst_idx = new_col + new_row * _MAP_TILE_WIDTH
                    new_tiles[dst_idx] = tiles[src_idx]

        return new_tiles

    def _translate_entities(
        self, entities: List[dict], dx: int, dy: int
    ) -> List[dict]:
        """Shift entity x/y coordinates by (dx, dy) tiles in map-data units.

        1 tile = 4 map-data units, so:
            entity_x_new = entity_x + dx * 4
            entity_y_new = entity_y + dy * 4
        """
        x_shift = dx * _MAP_UNITS_PER_TILE
        y_shift = dy * _MAP_UNITS_PER_TILE
        translated = []
        for ent in entities:
            new_ent = dict(ent)
            new_ent["x"] = ent["x"] + x_shift
            new_ent["y"] = ent["y"] + y_shift
            # Guard: drop entities that land outside the full map bounds.
            if 0 <= new_ent["x"] <= _FULL_MAP_WIDTH_UNITS and 0 <= new_ent["y"] <= _FULL_MAP_HEIGHT_UNITS:
                translated.append(new_ent)
        return translated

    def _translate_spawn(
        self, spawn: Tuple[int, int], dx: int, dy: int
    ) -> Tuple[int, int]:
        """Shift ninja spawn by (dx, dy) tiles in map-data units."""
        x_shift = dx * _MAP_UNITS_PER_TILE
        y_shift = dy * _MAP_UNITS_PER_TILE
        return (spawn[0] + x_shift, spawn[1] + y_shift)

    @staticmethod
    def _copy_unchanged(data: AttractData) -> AttractData:
        """Return a shallow copy of *data* without modification."""
        return {
            "tiles": list(data["tiles"]),
            "entities": list(data["entities"]),
            "ninja_spawn": data["ninja_spawn"],
            "header": data["header"],
            "demo_data": list(data["demo_data"]),
        }
