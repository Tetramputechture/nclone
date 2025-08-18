import math
import array
import struct

from ..entities import Entity, GridSegmentLinear, GridSegmentCircular
from ..physics import *
from ..ninja import NINJA_RADIUS

class EntityExitSwitch(Entity):
    ENTITY_TYPE = 4
    RADIUS = 6

    def __init__(self, type, sim, xcoord, ycoord, parent):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        self.parent = parent

    def logical_collision(self):
        """If the ninja is colliding with the switch, open its associated door. This is done in practice
        by adding the parent door entity to the entity grid.
        """
        ninja = self.sim.ninja
        if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                    ninja.xpos, ninja.ypos, NINJA_RADIUS):
            self.active = False
            # Add door to the entity grid so the ninja can touch it
            self.sim.grid_entity[self.parent.cell].append(self.parent)
            self.parent.switch_hit = True  # Mark the switch as hit on the parent Exit door
            self.log_collision()
