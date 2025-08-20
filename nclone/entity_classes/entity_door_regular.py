
from ..physics import *
from ..ninja import NINJA_RADIUS
from .entity_door_base import EntityDoorBase

class EntityDoorRegular(EntityDoorBase):
    ENTITY_TYPE = 5
    RADIUS = 10
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord):
        super().__init__(type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord)
        self.is_thinkable = True
        self.open_timer = 0

    def think(self):
        """If the door has been opened for more than 5 frames without being touched by the ninja, 
        close it.
        """
        if not self.closed:
            self.open_timer += 1
            if self.open_timer > 5:
                self.change_state(closed=True)

    def logical_collision(self):
        """If the ninja touches the activation region of the door (circle with a radius of 10 at the
        door's center), open it."""
        ninja = self.sim.ninja
        if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                    ninja.xpos, ninja.ypos, NINJA_RADIUS):
            self.change_state(closed=False)
            self.open_timer = 0
