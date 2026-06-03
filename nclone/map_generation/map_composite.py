"""COMPOSITE bucket generator: large levels chaining multiple mechanics.

Composes a long left-to-right course that strings together several different
segment types -- gaps, slopes, bowls, humps, staircases -- separated by short
flat runs, spanning most of the 42-wide canvas. This trains the agent to switch
between mechanics within a single episode and to carry/shed momentum across
transitions (e.g. a slope into a gap, a bowl into a hump).

Entity scope: player, one exit switch + door, optional toggle mines. The
segment engine guarantees a continuous walkable surface; the reachability
solvability gate (``solvability.is_map_solvable``) is the safety net for the
rare layout that strands a goal.

Config-driven attributes (from ``CompositeConfig``): MIN/MAX_WIDTH,
MIN/MAX_HEIGHT, MIN/MAX_SEGMENTS, MAX_MINES, VARIANT
(``"mixed"`` | ``"undulation"`` | ``"terrain"``).
"""

from typing import List, Optional, Tuple

from .map import Map
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from . import composition as comp


class MapComposite(Map):
    MIN_WIDTH = 30
    MAX_WIDTH = 40
    MIN_HEIGHT = 10
    MAX_HEIGHT = 16
    MIN_SEGMENTS = 2
    MAX_SEGMENTS = 5
    MAX_MINES = 0
    VARIANT = "mixed"

    def _segment_pool(self) -> Tuple[str, ...]:
        if self.VARIANT == "undulation":
            return ("bowl", "hump")
        if self.VARIANT == "terrain":
            return ("slope_up", "slope_down", "bowl", "hump")
        return comp.MECHANIC_SEGMENTS  # mixed: gaps, slopes, curves, staircases

    def generate(self, seed: Optional[int] = None) -> "MapComposite":
        if seed is not None:
            self.rng.seed(seed)
        self.reset()
        self.set_tiles_bulk([0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))

        height = self.rng.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)
        base_row = MAP_TILE_HEIGHT - 2
        floor_row = min(base_row - 1, max(9, height))

        n_segments = self.rng.randint(self.MIN_SEGMENTS, self.MAX_SEGMENTS)
        pool = self._segment_pool()

        plan: List[Tuple[str, int]] = [("flat", 4)]
        for _ in range(n_segments):
            seg = self.rng.choice(pool)
            width = 6 if seg == "gap" else self.rng.randint(3, 5)
            plan.append((seg, width))
            plan.append(("flat", self.rng.randint(2, 3)))
        plan.append(("flat", 5))

        results = comp.assemble_segments(self, plan, floor_row, base_row, self.rng)
        if not comp.finalize_floor_level(self, results, int(self.MAX_MINES), self.rng):
            self.set_tiles_bulk([0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))
            results = comp.assemble_segments(
                self,
                [("flat", 5), ("slope_up", 3), ("flat", 3), ("gap", 6), ("flat", 6)],
                floor_row,
                base_row,
                self.rng,
            )
            comp.finalize_floor_level(self, results, 0, self.rng)
        return self
