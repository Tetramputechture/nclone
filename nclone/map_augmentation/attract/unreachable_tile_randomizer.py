"""Augmentation that randomizes tiles in areas unreachable from the ninja spawn.

Tiles the ninja can never reach have no effect on physics or gameplay.
Randomizing them creates visually diverse training samples while guaranteeing
the original demo_data remains valid.

Safety guarantee
----------------
The reachable tile set is expanded by a 1-tile buffer before any modification.
A ninja with a 10 px radius can only interact with geometry within roughly
0.42 tiles of its centre, so one full tile of margin is conservative.

Usage
-----
>>> aug = UnreachableTileRandomizer(seed=42)
>>> new_data = aug.augment(attract_data)
"""

from __future__ import annotations

import random
from typing import List, Optional

from .base_augmentation import AttractData, BaseAugmentation
from .reachability_augment_utils import (
    _MAP_TILE_HEIGHT,
    _MAP_TILE_WIDTH,
    _TILE_COUNT,
    get_protected_tiles,
    get_reachable_tiles,
)

# Valid tile type identifiers (0 = empty, 1 = full block, 2–33 = shaped tiles).
_VALID_TILE_TYPES: List[int] = list(range(34))


class UnreachableTileRandomizer(BaseAugmentation):
    """Replace every tile outside the reachable+buffer zone with a random type.

    The augmentation does not mutate the input and does not modify any
    gameplay-affecting data (entities, spawn position, input sequence, header).

    Args:
        seed:   Optional integer seed for the internal random number generator.
                Pass a fixed seed to obtain reproducible outputs in tests.
                If None, the RNG is seeded from the system entropy source.
        buffer: Number of tiles to preserve around each reachable tile as a
                safety margin (default 1).
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
        return "tile_rand"

    def augment(self, data: AttractData) -> AttractData:
        """Return a copy of *data* with unreachable tiles replaced by random types.

        Args:
            data: Decoded attract-file data.

        Returns:
            New AttractData with modified tile list; all other fields unchanged.

        Raises:
            ValueError: If ``data['tiles']`` does not contain exactly 966 values.
        """
        tiles: List[int] = data["tiles"]
        if len(tiles) != _TILE_COUNT:
            raise ValueError(
                f"UnreachableTileRandomizer: expected {_TILE_COUNT} tiles, "
                f"got {len(tiles)}"
            )

        reachable = get_reachable_tiles(data)
        protected = get_protected_tiles(reachable, buffer=self._buffer)

        new_tiles = list(tiles)
        for row in range(_MAP_TILE_HEIGHT):
            for col in range(_MAP_TILE_WIDTH):
                if (col, row) not in protected:
                    new_tiles[col + row * _MAP_TILE_WIDTH] = self._rng.choice(
                        _VALID_TILE_TYPES
                    )

        return {
            "tiles": new_tiles,
            "entities": list(data["entities"]),
            "ninja_spawn": data["ninja_spawn"],
            "header": data["header"],
            "demo_data": list(data["demo_data"]),
        }
