"""PRECISION bucket generator (recommended extra): narrow mined ledges.

Combines tight footing with hazards: a chain of short ledges separated by gaps,
with toggle mines sitting on the ledges. The agent must land precisely on a
narrow platform AND avoid the mine -- training speed control and landing
accuracy under a no-overshoot constraint. This is the mine + jump precision
mechanic that pure MINE_NAVIGATION (flat) and MULTI_JUMP (mine-free) don't
cover on their own.

Entity scope: player, one exit switch + door, toggle mines (always >=1).

Config-driven attributes (from ``PrecisionConfig``): MIN/MAX_WIDTH,
MIN/MAX_HEIGHT, MIN/MAX_JUMPS, MAX_MINES, LEDGE_WIDTH.
"""

from typing import Optional

from .map import Map
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from . import composition as comp


class MapPrecision(Map):
    MIN_WIDTH = 22
    MAX_WIDTH = 38
    MIN_HEIGHT = 8
    MAX_HEIGHT = 12
    MIN_JUMPS = 2
    MAX_JUMPS = 4
    MAX_MINES = 2
    LEDGE_WIDTH = 3

    def generate(self, seed: Optional[int] = None) -> "MapPrecision":
        if seed is not None:
            self.rng.seed(seed)
        self.reset()
        self.set_tiles_bulk([0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))

        height = self.rng.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)
        base_row = MAP_TILE_HEIGHT - 2
        floor_row = min(base_row - 1, max(6, height))
        ledge_w = max(3, int(self.LEDGE_WIDTH))

        n_jumps = self.rng.randint(self.MIN_JUMPS, self.MAX_JUMPS)

        # spawn flat, then [narrow ledge, gap] chain, then exit flat.
        plan = [("flat", 5)]
        for _ in range(n_jumps):
            plan.append(("gap", 6))
            plan.append(("flat", ledge_w))
        plan.append(("flat", 5))

        results = comp.assemble_segments(self, plan, floor_row, base_row, self.rng)
        # PRECISION must always carry at least one mine; clamp the requested
        # budget to >=1 (within the available ledge cells).
        n_mines = max(1, int(self.MAX_MINES))
        if not comp.finalize_floor_level(self, results, n_mines, self.rng):
            self.set_tiles_bulk([0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))
            results = comp.assemble_segments(
                self,
                [("flat", 5), ("gap", 6), ("flat", 4), ("gap", 6), ("flat", 6)],
                floor_row,
                base_row,
                self.rng,
            )
            comp.finalize_floor_level(self, results, 1, self.rng)
        return self
