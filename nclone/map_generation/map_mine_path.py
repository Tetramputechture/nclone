"""Dedicated minimal generator for the MINE_NAVIGATION mechanic bucket.

Produces a flat horizontal corridor tall enough to jump (>= 3 tiles
interior), seeded with a small number of toggle mines on the floor at a
parameterized minimum spacing. The ninja must jump over or weave past
each mine to reach the switch and exit on the far side. No walls, no
slopes, no vertical mechanics, no wall-jump: the only non-locomotion
mechanic is the mines themselves, so these levels form a clean
baseline for the MINE_NAVIGATION bucket.
"""

from typing import Optional, Tuple

from .map import Map
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class MapMinePath(Map):
    """Minimal MINE_NAVIGATION generator: flat path with N toggle mines."""

    MIN_WIDTH = 10
    MAX_WIDTH = 24
    MIN_HEIGHT = 3
    MAX_HEIGHT = 5

    MIN_MINE_COUNT = 1
    MAX_MINE_COUNT = 4

    # Minimum horizontal tile spacing between mine centers. Enforced even
    # when the requested mine count is large relative to corridor width.
    MIN_MINE_SPACING = 2

    # Minimum tile distance between any mine and the ninja spawn. Mirrors
    # the convention used by other generators to avoid "impossible to
    # avoid" mines at spawn.
    SPAWN_BUFFER_TILES = 2
    # Mixed tier 3+4: one-tile step-up (halves the floor height)
    ADD_STEP = False

    def _pick_corridor(self) -> Tuple[int, int, int, int]:
        width = self.rng.randint(self.MIN_WIDTH, self.MAX_WIDTH)
        height = self.rng.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)
        max_start_x = MAP_TILE_WIDTH - width - 2
        max_start_y = MAP_TILE_HEIGHT - height - 2
        x1 = self.rng.randint(2, max(2, max_start_x))
        y1 = self.rng.randint(2, max(2, max_start_y))
        x2 = x1 + width - 1
        y2 = y1 + height - 1
        return x1, y1, x2, y2

    def _carve_corridor(self, x1: int, y1: int, x2: int, y2: int) -> None:
        total = MAP_TILE_WIDTH * MAP_TILE_HEIGHT
        self.set_tiles_bulk([1] * total)
        self.set_empty_rectangle(x1, y1, x2, y2)

    def _pick_entity_columns(
        self, x1: int, x2: int
    ) -> Tuple[int, int, int, bool]:
        """Return ``(ninja_x, switch_x, door_x, ninja_on_left)``."""
        ninja_on_left = self.rng.choice([True, False])
        if ninja_on_left:
            ninja_x = x1
            switch_x = x2
            door_x = x2 - 1 if x2 - 1 > ninja_x else x2
        else:
            ninja_x = x2
            switch_x = x1
            door_x = x1 + 1 if x1 + 1 < ninja_x else x1
        return ninja_x, switch_x, door_x, ninja_on_left

    def _sample_mine_positions(
        self, x1: int, x2: int, ninja_x: int, reserved: set
    ) -> list:
        """Pick mine x-columns honoring ``MIN_MINE_SPACING`` and spawn buffer."""
        candidates = []
        for x in range(x1 + 1, x2):
            if x in reserved:
                continue
            if abs(x - ninja_x) < self.SPAWN_BUFFER_TILES:
                continue
            candidates.append(x)
        self.rng.shuffle(candidates)

        requested = self.rng.randint(self.MIN_MINE_COUNT, self.MAX_MINE_COUNT)
        picked: list = []
        for x in candidates:
            if len(picked) >= requested:
                break
            if all(abs(x - px) >= self.MIN_MINE_SPACING for px in picked):
                picked.append(x)
        picked.sort()
        return picked

    def _apply_step(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Raise *either* the left *or* right half of the floor by 1 tile."""
        edge = (x1 + x2) // 2
        height = 1
        raise_right = self.rng.choice([True, False])
        raised_top_y = y2 - height + 1
        if raise_right:
            fill_range = range(edge, x2 + 1)
        else:
            fill_range = range(x1, edge)
        for x in fill_range:
            for y in range(raised_top_y, y2 + 1):
                if y1 <= y <= y2:
                    self.set_tile(x, y, 1)

    def _ground_y_for(
        self, x: int, x1: int, x2: int, y1: int, y2: int
    ) -> int:
        """Topmost empty cell in the column (ninja / switch / door / mine)."""
        if not (x1 <= x <= x2):
            return y2
        for y in range(y2, y1 - 1, -1):
            if self.get_tile(x, y) == 0:
                return y
        return y2

    def generate(self, seed: Optional[int] = None) -> "MapMinePath":
        """Generate one MINE_NAVIGATION level."""
        if seed is not None:
            self.rng.seed(seed)
        self.reset()

        x1, y1, x2, y2 = self._pick_corridor()
        self._carve_corridor(x1, y1, x2, y2)
        if self.ADD_STEP:
            self._apply_step(x1, y1, x2, y2)

        ninja_x, switch_x, door_x, ninja_on_left = self._pick_entity_columns(x1, x2)
        ninja_y = self._ground_y_for(ninja_x, x1, x2, y1, y2)
        switch_y = self._ground_y_for(switch_x, x1, x2, y1, y2)
        door_y = self._ground_y_for(door_x, x1, x2, y1, y2)

        orientation = 1 if ninja_on_left else -1
        self.set_ninja_spawn(ninja_x, ninja_y, orientation=orientation)
        self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)

        reserved = {ninja_x, switch_x, door_x}
        mine_columns = self._sample_mine_positions(x1, x2, ninja_x, reserved)
        for mx in mine_columns:
            my = self._ground_y_for(mx, x1, x2, y1, y2)
            self.add_entity(1, mx, my)

        return self
