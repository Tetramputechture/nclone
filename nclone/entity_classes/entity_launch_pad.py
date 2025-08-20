
from ..entities import Entity
from ..physics import *
from ..ninja import NINJA_RADIUS

class EntityLaunchPad(Entity):
    ENTITY_TYPE = 10
    RADIUS = 6
    BOOST = 36/7
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        self.orientation = orientation
        self.normal_x, self.normal_y = map_orientation_to_vector(orientation)

    def logical_collision(self):
        """If the ninja is colliding with the launch pad (semi circle hitbox), return boost."""
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                        ninja.xpos, ninja.ypos, NINJA_RADIUS):
                if ((self.xpos - (ninja.xpos - NINJA_RADIUS*self.normal_x))*self.normal_x
                        + (self.ypos - (ninja.ypos - NINJA_RADIUS*self.normal_y))*self.normal_y) >= -0.1:
                    yboost_scale = 1
                    if self.normal_y < 0:
                        yboost_scale = 1 - self.normal_y
                    xboost = self.normal_x * self.BOOST
                    yboost = self.normal_y * self.BOOST * yboost_scale
                    return (xboost, yboost)
