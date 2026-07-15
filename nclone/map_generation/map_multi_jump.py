"""MULTI_JUMP bucket generator: levels requiring >=2 sequenced jumps.

Where ``BASIC_JUMP`` isolates a single jump, this isolates *sequencing*: a run
of gaps and/or step-up staircases that must be cleared one after another. The
jump count scales with the tier via MIN/MAX_JUMPS.

Entity scope: player, one exit switch + door, optional toggle mines (higher
tiers only). No slopes/curves/wall-jump segments.

Config-driven attributes (from ``MultiJumpConfig``): MIN/MAX_WIDTH,
MIN/MAX_HEIGHT, MIN/MAX_JUMPS, MAX_MINES, VARIANT
(``"gaps"`` | ``"staircase"`` | ``"mixed"``).
"""

from typing import Optional

from .map import Map
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from . import composition as comp


class MapMultiJump(Map):
    MIN_WIDTH = 24
    MAX_WIDTH = 40
    MIN_HEIGHT = 8
    MAX_HEIGHT = 14
    MIN_JUMPS = 2
    MAX_JUMPS = 3
    MAX_MINES = 0
    VARIANT = "gaps"
    # Carved-gap width cap (seg_gap honours this) and the initial run-up length.
    # A 2-tile gap + a long flat run-up = an easy hop; the default keeps the
    # original 3-tile gap for the harder tiers.
    MAX_GAP_W = 3
    RUNUP = 5

    def generate(self, seed: Optional[int] = None) -> "MapMultiJump":
        if seed is not None:
            self.rng.seed(seed)
        self.reset()
        self.set_tiles_bulk([0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))

        height = self.rng.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)
        base_row = MAP_TILE_HEIGHT - 2
        # keep the floor low enough that staircases have headroom to climb
        floor_row = min(base_row - 1, max(8, height))

        n_jumps = self.rng.randint(self.MIN_JUMPS, self.MAX_JUMPS)

        plan = [("flat", int(self.RUNUP))]
        for i in range(n_jumps):
            if self.VARIANT == "gaps":
                seg = "gap"
            elif self.VARIANT == "staircase":
                seg = "staircase"
            else:  # mixed
                seg = "gap" if i % 2 == 0 else "staircase"
            if seg == "gap":
                plan.append(("gap", self.rng.randint(6, 7)))
                plan.append(("flat", self.rng.randint(2, 3)))
            else:
                plan.append(("staircase", 5))
                plan.append(("flat", self.rng.randint(2, 3)))
        plan.append(("flat", 6))

        results = comp.assemble_segments(self, plan, floor_row, base_row, self.rng)
        if not comp.finalize_floor_level(self, results, int(self.MAX_MINES), self.rng):
            self.set_tiles_bulk([0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))
            results = comp.assemble_segments(
                self,
                [("flat", 5), ("gap", 6), ("flat", 3), ("gap", 6), ("flat", 6)],
                floor_row,
                base_row,
                self.rng,
            )
            comp.finalize_floor_level(self, results, 0, self.rng)
        return self
