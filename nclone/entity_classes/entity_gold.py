import math
import array
import struct

from ..entities import Entity, GridSegmentLinear, GridSegmentCircular
from ..physics import *
from ..ninja import NINJA_RADIUS

class EntityGold(Entity):
    ENTITY_TYPE = 2
    RADIUS = 6
    MAX_COUNT_PER_LEVEL = 8192

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True

    def logical_collision(self):
        """The gold is collected if touches by a ninja that is not in winning state."""
        ninja = self.sim.ninja
        if ninja.state != 8:
            if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                        ninja.xpos, ninja.ypos, NINJA_RADIUS):
                ninja.gold_collected += 1
                self.active = False
                self.log_collision()
