
from ..physics import *
from ..ninja import NINJA_RADIUS
from .entity_drone_base import EntityDroneBase

class EntityDroneZap(EntityDroneBase):
    ENTITY_TYPE = 14
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation, mode):
        super().__init__(type, sim, xcoord, ycoord, orientation, mode, 8/7)
        self.is_logical_collidable = True

    def logical_collision(self):
        """Kill the ninja if it touches the regular drone."""
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                        ninja.xpos, ninja.ypos, NINJA_RADIUS):
                ninja.kill(0, 0, 0, 0, 0)
