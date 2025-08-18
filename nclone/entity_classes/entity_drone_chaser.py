import math
import array
import struct

from ..entities import Entity, GridSegmentLinear, GridSegmentCircular
from ..physics import *
from ..ninja import NINJA_RADIUS
from .entity_drone_zap import EntityDroneZap

class EntityDroneChaser(EntityDroneZap):
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation, mode):
        super().__init__(type, sim, xcoord, ycoord, orientation, mode)
        self.is_thinkable = True
        self.speed_slow = self.speed
        self.speed_chase = 2 * self.speed
        self.chasing = False

    def think(self):
        # TODO
        if not self.chasing:
            ninja = self.sim.ninja
            if ninja.is_valid_target():
                for i in range(-1, 2):
                    dir = (self.dir + i) % 4
                    xdir, ydir = self.DIR_TO_VEC[dir]
                    if xdir*(ninja.xpos - self.xpos) + ydir*(ninja.ypos - self.ypos) > 0:
                        if abs(ydir*(ninja.xpos - self.xpos) - xdir*(ninja.ypos - self.ypos)) <= 12:
                            pass

    def choose_next_direction_and_goal(self):
        # TODO
        super().choose_next_direction_and_goal()
