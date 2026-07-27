"""Dedicated generator for the open flat-run archetype (``flat-run-open``).

Produces a fully open arena -- no interior tiles, just the implicit solid
border the simulator always wraps the play field in -- across which the ninja
makes a single flat run along the bottom floor. The exit switch and exit door
are spread out along that same floor so the only mechanic is horizontal running:
reach the switch, then the door.

This is the "open space" sibling of the FLAT_RUN bucket's corridor/chamber
presets (``horizontal_corridor`` / ``single_chamber``): instead of a tight
floor-and-ceiling corridor, the run happens in wide-open space with a high
ceiling, which generalises the hand-built ``flat-run-open`` reference map.

Mine-free by construction: FLAT_RUN is a no-mine bucket, so this generator
never places hazards.

Config-driven class attributes (set by ``GeneratorFactory`` from
``OpenFieldConfig``): MIN/MAX_WIDTH (the run length the entities are spread
across -- the size curriculum's difficulty knob), MIN/MAX_HEIGHT (nominal arena
height, kept for size-curriculum parity), MIN/MAX_SEPARATION (tiles between
consecutive spawn/switch/door anchors).
"""

from typing import Optional

from .map import Map
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class MapOpenField(Map):
    """Open-arena flat run: spawn -> switch -> door along the bottom floor."""

    MIN_WIDTH = 18
    MAX_WIDTH = 38
    MIN_HEIGHT = 10
    MAX_HEIGHT = 20

    MIN_SEPARATION = 3
    MAX_SEPARATION = 8

    # Gold count range (SPREAD regime); overwritten by OpenFieldConfig via the
    # factory. 0/0 disables gold. See docs/GOLD_AND_SCORE_OPTIMIZATION.md.
    MIN_GOLD_COUNT = 5
    MAX_GOLD_COUNT = 16

    def generate(self, seed: Optional[int] = None) -> "MapOpenField":
        if seed is not None:
            self.rng.seed(seed)
        self.reset()

        # Open arena above an authored ground strip: every interior cell is
        # empty except a solid floor across the bottom row that the ninja runs
        # along (the wide-open space above is the "open field"). An explicit
        # floor is more robust than relying on the implicit border the loader
        # wraps the field in.
        self.set_tiles_bulk([0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))
        ground_row = MAP_TILE_HEIGHT - 1
        for x in range(MAP_TILE_WIDTH):
            self.set_tile(x, ground_row, 1)

        floor_row = ground_row - 1  # walking surface (atop the ground strip)

        # The run length is the difficulty knob the size curriculum interpolates.
        run = self.rng.randint(self.MIN_WIDTH, self.MAX_WIDTH)
        left = 1
        right = min(left + run, MAP_TILE_WIDTH - 2)

        gap1 = self.rng.randint(self.MIN_SEPARATION, self.MAX_SEPARATION)
        gap2 = self.rng.randint(self.MIN_SEPARATION, self.MAX_SEPARATION)

        # Lay spawn -> switch -> door left to right so the run hits the switch
        # before the door; mirror the whole layout half the time for variety.
        ninja_x = left
        switch_x = min(ninja_x + gap1, right - 1)
        door_x = min(switch_x + gap2, right)

        if self.rng.choice([True, False]):
            # Mirror about the populated span so the run goes right -> left.
            span_l, span_r = ninja_x, door_x
            ninja_x = span_l + span_r - ninja_x
            switch_x = span_l + span_r - switch_x
            door_x = span_l + span_r - door_x

        orientation = 1 if switch_x >= ninja_x else -1
        self.set_ninja_spawn(ninja_x, floor_row, orientation)
        self.add_entity(3, door_x, floor_row, 0, 0, switch_x, floor_row)

        # SPREAD gold: scatter across the full run width at/just above the floor so
        # the score objective rewards sweeping the whole open arena (the large-open
        # case). Reachable by construction — on or a hop above the continuous floor.
        self._place_spread_gold(left, right, floor_row, {ninja_x, switch_x, door_x})

        return self

    def _place_spread_gold(self, left, right, floor_row, occupied):
        n_gold = self.rng.randint(self.MIN_GOLD_COUNT, self.MAX_GOLD_COUNT)
        if n_gold <= 0:
            return
        cand_x = [x for x in range(left, right + 1) if x not in occupied]
        self.rng.shuffle(cand_x)
        for gx in cand_x[:n_gold]:
            # Floor level or one/two tiles above — all jump-reachable in open space.
            gy = floor_row - self.rng.randint(0, 2)
            gx2, gy2 = self._find_closest_valid_tile(gx, gy, 0)
            self.add_entity(2, gx2, gy2)
