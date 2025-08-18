import math
import array
import struct

from ..entities import Entity, GridSegmentLinear, GridSegmentCircular
from ..physics import *
from ..ninja import NINJA_RADIUS
from .entity_drone_base import EntityDroneBase

class EntityMiniDrone(EntityDroneBase):
    ENTITY_TYPE = 26
    MAX_COUNT_PER_LEVEL = 512

    def __init__(self, type, sim, xcoord, ycoord, orientation, mode):
        super().__init__(type, sim, xcoord, ycoord, orientation, mode, 1.3)
        self.is_logical_collidable = True
        self.RADIUS = 4
        self.GRID_WIDTH = 12

    def logical_collision(self):
        """Kill the ninja if it touches the mini drone."""
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                        ninja.xpos, ninja.ypos, NINJA_RADIUS):
                ninja.kill(0, 0, 0, 0, 0)
