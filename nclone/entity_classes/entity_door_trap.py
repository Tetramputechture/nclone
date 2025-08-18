from ..physics import *
from ..ninja import NINJA_RADIUS
from .entity_door_base import EntityDoorBase

class EntityDoorTrap(EntityDoorBase):
    ENTITY_TYPE = 8
    RADIUS = 5
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord):
        super().__init__(type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord)
        self.change_state(closed=False)

    def logical_collision(self):
        """If the ninja collects the associated close switch, close the door."""
        ninja = self.sim.ninja
        if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                    ninja.xpos, ninja.ypos, NINJA_RADIUS):
            self.change_state(closed=True)
            self.active = False
