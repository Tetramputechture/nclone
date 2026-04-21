"""Dedicated minimal generator for the WALL_JUMP mechanic bucket.

Produces a clean narrow vertical shaft with parallel smooth walls and no
mines. The ninja must wall-jump up (or fall down and wall-jump partway
back up) to reach the switch and then the exit door. All features that
would contaminate the wall-jump signal (chaotic wall tiles, mine
gauntlets, platform scatter, corridors) are intentionally excluded.

Optional ``ADD_MIDDLE_LEDGE`` inserts a single small platform roughly
halfway up the shaft to break a pure wall-jump streak into a two-stage
ascent. This is used in higher-tier presets without introducing any new
mechanics.
"""

from typing import Optional, Tuple

from .map import Map
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class MapWallJump(Map):
    """Minimal WALL_JUMP generator: clean vertical shaft, no mines."""

    # Single-column shaft: matches :func:`nclone.planning.optimal_route.geometry.classify`
    # (``TOPOLOGY_WALL_JUMP_SHAFT`` requires ``chamber_tile_width == 1``).
    MIN_WIDTH = 1
    MAX_WIDTH = 1
    MIN_HEIGHT = 8
    MAX_HEIGHT = 18

    # When True, a single 1-tile ledge is placed roughly halfway up the
    # shaft's interior to break the wall-jump streak. Still mine-free.
    ADD_MIDDLE_LEDGE = False

    # Placement of switch/exit relative to shaft ends. ``top`` means
    # switch near the top of the shaft (ninja starts at the bottom and
    # must ascend); ``bottom`` reverses the arrangement. ``random``
    # picks per-level.
    GOAL_POSITION = "top"

    def _pick_shaft(self) -> Tuple[int, int, int, int]:
        """Choose shaft interior ``(x1, y1, x2, y2)``.

        Shaft interior is fully empty. Surrounding solid tiles form the
        walls, floor, and ceiling.
        """
        width = self.rng.randint(self.MIN_WIDTH, self.MAX_WIDTH)
        height = self.rng.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)

        max_start_x = MAP_TILE_WIDTH - width - 2
        max_start_y = MAP_TILE_HEIGHT - height - 2
        x1 = self.rng.randint(2, max(2, max_start_x))
        y1 = self.rng.randint(2, max(2, max_start_y))
        x2 = x1 + width - 1
        y2 = y1 + height - 1
        return x1, y1, x2, y2

    def _carve_shaft(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Fill solid; carve the empty shaft interior."""
        total = MAP_TILE_WIDTH * MAP_TILE_HEIGHT
        self.set_tiles_bulk([1] * total)
        self.set_empty_rectangle(x1, y1, x2, y2)

    def _maybe_add_middle_ledge(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> Tuple[int, int]:
        """Place one solid tile roughly halfway up the shaft.

        Returns the ``(ledge_x, ledge_y)`` placed, or ``(-1, -1)`` if no
        ledge was placed (shaft too short, or feature disabled).
        """
        if not self.ADD_MIDDLE_LEDGE:
            return -1, -1

        height = y2 - y1 + 1
        if height < 8:
            return -1, -1

        mid_y = y1 + height // 2 + self.rng.randint(-1, 1)
        ledge_x = self.rng.randint(x1, x2)
        self.set_tile(ledge_x, mid_y, 1)
        return ledge_x, mid_y

    def _choose_goal_position(self) -> str:
        if self.GOAL_POSITION == "random":
            return self.rng.choice(["top", "bottom"])
        return self.GOAL_POSITION

    def generate(self, seed: Optional[int] = None) -> "MapWallJump":
        """Generate one WALL_JUMP level."""
        if seed is not None:
            self.rng.seed(seed)
        self.reset()

        x1, y1, x2, y2 = self._pick_shaft()
        self._carve_shaft(x1, y1, x2, y2)
        ledge_x, ledge_y = self._maybe_add_middle_ledge(x1, y1, x2, y2)

        goal_position = self._choose_goal_position()

        if goal_position == "top":
            ninja_y = y2
            goal_y = y1
            ninja_x = self.rng.randint(x1, x2)
            switch_x = self.rng.randint(x1, x2)
            door_x = self.rng.randint(x1, x2)
        else:
            ninja_y = y1
            goal_y = y2
            ninja_x = self.rng.randint(x1, x2)
            switch_x = self.rng.randint(x1, x2)
            door_x = self.rng.randint(x1, x2)

        if (ledge_x, ledge_y) != (-1, -1):
            if ninja_x == ledge_x and ninja_y == ledge_y:
                ninja_x = ledge_x + (1 if ledge_x < x2 else -1)
            if switch_x == ledge_x and goal_y == ledge_y:
                switch_x = ledge_x + (1 if ledge_x < x2 else -1)
            if door_x == ledge_x and goal_y == ledge_y:
                door_x = ledge_x + (1 if ledge_x < x2 else -1)

        orientation = self.rng.choice([1, -1])
        self.set_ninja_spawn(ninja_x, ninja_y, orientation=orientation)
        self.add_entity(3, door_x, goal_y, 0, 0, switch_x, goal_y)
        return self
