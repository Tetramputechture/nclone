import math
import array
import struct

from ..entities import Entity, GridSegmentLinear, GridSegmentCircular
from ..physics import *
from ..ninja import NINJA_RADIUS

class EntityExit(Entity):
    ENTITY_TYPE = 3
    RADIUS = 12
    MAX_COUNT_PER_LEVEL = 16

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        self.switch_hit = False

    def logical_collision(self):
        """The ninja wins if it touches the exit door. The door is not interactable from the entity
        grid before the exit switch is collected.
        """
        ninja = self.sim.ninja
        if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                    ninja.xpos, ninja.ypos, NINJA_RADIUS):
            ninja.win()
