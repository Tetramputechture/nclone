"""Dedicated generator for the rounded-corner wall-ascent archetype
(``open-round-corner``).

Produces an open arena whose floor curves up into a side wall through a concave
quarter-pipe (rounded corner). The exit door floats up that side wall; the
ninja runs across the open floor, hits the switch, then rides the rounded corner
into a wall-jump ascent to reach the door. The single mechanic is *curve-assisted
wall ascent*: the curved corner eases the transition from running to climbing.

It belongs to the CURVES bucket (the concave corner is built from the verified
``curve_utils`` concave tiles) but layers a vertical wall-jump objective on top,
generalising the hand-built ``open-round-corner`` reference map.

Mine-free by construction (CURVES is a no-mine bucket).

Config-driven class attributes (set by ``GeneratorFactory`` from
``RoundCornerConfig``): MIN/MAX_WIDTH + MIN/MAX_HEIGHT (arena extent, the size
curriculum's knobs), MIN/MAX_DOOR_HEIGHT (tiles the door sits up the wall -- the
height of the required ascent), MIN/MAX_CORNER_RADIUS (size of the rounded
quarter-pipe), SIDE (``"left"`` | ``"right"`` | ``"random"``).
"""

from typing import Optional

from .map import Map
from .curve_utils import CONCAVE_DESCEND, CONCAVE_ASCEND, SOLID_TILE
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class MapRoundCorner(Map):
    """Open arena: run -> switch -> ride a rounded corner up a wall to the door."""

    MIN_WIDTH = 16
    MAX_WIDTH = 34
    MIN_HEIGHT = 12
    MAX_HEIGHT = 20

    MIN_DOOR_HEIGHT = 4
    MAX_DOOR_HEIGHT = 10

    MIN_CORNER_RADIUS = 1
    MAX_CORNER_RADIUS = 3

    SIDE = "random"

    def _build_corner(self, side: str, floor_row: int, radius: int) -> None:
        """Stamp a concave quarter-pipe easing the floor up into ``side`` wall.

        Nearest the wall the curve is highest; it descends to floor level over
        ``radius`` columns. Each curved cap sits atop a solid support column so
        the geometry is collision-correct.
        """
        for i in range(radius):
            height = radius - i  # tallest column hugs the wall
            top_row = floor_row - (height - 1)
            if side == "left":
                col = i
                cap = CONCAVE_DESCEND  # surface high on the (left) wall side
            else:
                col = (MAP_TILE_WIDTH - 1) - i
                cap = CONCAVE_ASCEND  # surface high on the (right) wall side
            self.set_tile(col, top_row, cap)
            for yy in range(top_row + 1, floor_row + 1):
                self.set_tile(col, yy, SOLID_TILE)

    def generate(self, seed: Optional[int] = None) -> "MapRoundCorner":
        if seed is not None:
            self.rng.seed(seed)
        self.reset()

        # Open arena; the ninja stands on the implicit solid border.
        self.set_tiles_bulk([0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))

        floor_row = MAP_TILE_HEIGHT - 1

        side = self.SIDE
        if side not in ("left", "right"):
            side = self.rng.choice(["left", "right"])

        radius = self.rng.randint(self.MIN_CORNER_RADIUS, self.MAX_CORNER_RADIUS)
        door_height = self.rng.randint(self.MIN_DOOR_HEIGHT, self.MAX_DOOR_HEIGHT)
        door_row = max(1, floor_row - door_height)

        self._build_corner(side, floor_row, radius)

        run = self.rng.randint(self.MIN_WIDTH, self.MAX_WIDTH)

        if side == "left":
            door_x = 0
            switch_x = min(radius + 2, MAP_TILE_WIDTH - 3)
            ninja_x = min(switch_x + run, MAP_TILE_WIDTH - 2)
            orientation = -1  # face the corner (to the left)
        else:
            door_x = MAP_TILE_WIDTH - 1
            switch_x = max(MAP_TILE_WIDTH - 3 - radius, 2)
            ninja_x = max(switch_x - run, 1)
            orientation = 1  # face the corner (to the right)

        self.set_ninja_spawn(ninja_x, floor_row, orientation)
        # Door floats up the wall; switch sits on the open floor en route.
        self.add_entity(3, door_x, door_row, 0, 0, switch_x, floor_row)

        return self
