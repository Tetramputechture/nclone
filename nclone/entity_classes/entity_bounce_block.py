import math
import array
import struct

from ..entities import Entity, GridSegmentLinear, GridSegmentCircular
from ..physics import *
from ..ninja import NINJA_RADIUS

class EntityBounceBlock(Entity):
    ENTITY_TYPE = 17
    SEMI_SIDE = 9
    STIFFNESS = 0.02222222222222222
    DAMPENING = 0.98
    STRENGTH = 0.2
    MAX_COUNT_PER_LEVEL = 512

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_physical_collidable = True
        self.is_logical_collidable = True
        self.is_movable = True
        self.xspeed, self.yspeed = 0, 0
        self.xorigin, self.yorigin = self.xpos, self.ypos

    def move(self):
        """Update the position and speed of the bounce block by applying the spring force and dampening."""
        self.xspeed *= self.DAMPENING
        self.yspeed *= self.DAMPENING
        self.xpos += self.xspeed
        self.ypos += self.yspeed
        xforce = self.STIFFNESS * (self.xorigin - self.xpos)
        yforce = self.STIFFNESS * (self.yorigin - self.ypos)
        self.xpos += xforce
        self.ypos += yforce
        self.xspeed += xforce
        self.yspeed += yforce
        self.grid_move()

    def physical_collision(self):
        """Apply 80% of the depenetration to the bounce block and 20% to the ninja."""
        ninja = self.sim.ninja
        depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                            self.SEMI_SIDE + NINJA_RADIUS)
        if depen:
            depen_x, depen_y = depen[0]
            depen_len = depen[1][0]
            self.xpos -= depen_x * depen_len * (1-self.STRENGTH)
            self.ypos -= depen_y * depen_len * (1-self.STRENGTH)
            self.xspeed -= depen_x * depen_len * (1-self.STRENGTH)
            self.yspeed -= depen_y * depen_len * (1-self.STRENGTH)
            return (depen_x, depen_y), (depen_len * self.STRENGTH, depen[1][1])

    def logical_collision(self):
        """Check if the ninja can interact with the wall of the bounce block"""
        ninja = self.sim.ninja
        depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                            self.SEMI_SIDE + NINJA_RADIUS + 0.1)
        if depen:
            return depen[0][0]
