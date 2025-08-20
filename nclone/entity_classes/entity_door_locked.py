
from ..physics import *
from ..ninja import NINJA_RADIUS
from .entity_door_base import EntityDoorBase

class EntityDoorLocked(EntityDoorBase):
    ENTITY_TYPE = 6
    RADIUS = 5
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord):
        super().__init__(type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord)

    def logical_collision(self):
        """If the ninja collects the associated open switch, open the door."""
        ninja = self.sim.ninja
        if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                    ninja.xpos, ninja.ypos, NINJA_RADIUS):
            ninja.doors_opened += 1
            self.change_state(closed=False)
            self.active = False
