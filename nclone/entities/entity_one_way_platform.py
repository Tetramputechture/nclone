import math
import array
import struct

from ..entities import Entity, GridSegmentLinear, GridSegmentCircular
from ..physics import *
from ..ninja import NINJA_RADIUS

class EntityOneWayPlatform(Entity):
    ENTITY_TYPE = 11
    SEMI_SIDE = 12
    MAX_COUNT_PER_LEVEL = 512

    def __init__(self, type, sim, xcoord, ycoord, orientation):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        self.is_physical_collidable = True
        self.orientation = orientation
        self.normal_x, self.normal_y = map_orientation_to_vector(orientation)

    def calculate_depenetration(self, ninja):
        """Return the depenetration vector between the ninja and the one way. Return nothing if no
        penetration.
        """
        dx = ninja.xpos - self.xpos
        dy = ninja.ypos - self.ypos
        lateral_dist = dy * self.normal_x - dx * self.normal_y
        direction = (ninja.yspeed * self.normal_x -
                     ninja.xspeed * self.normal_y) * lateral_dist
        # The platform has a bigger width if the ninja is moving towards its center.
        radius_scalar = 0.91 if direction < 0 else 0.51
        if abs(lateral_dist) < radius_scalar * NINJA_RADIUS + self.SEMI_SIDE:
            normal_dist = dx * self.normal_x + dy * self.normal_y
            if 0 < normal_dist <= NINJA_RADIUS:
                normal_proj = ninja.xspeed * self.normal_x + ninja.yspeed * self.normal_y
                if normal_proj <= 0:
                    dx_old = ninja.xpos_old - self.xpos
                    dy_old = ninja.ypos_old - self.ypos
                    normal_dist_old = dx_old * self.normal_x + dy_old * self.normal_y
                    if NINJA_RADIUS - normal_dist_old <= 1.1:
                        return (self.normal_x, self.normal_y), (NINJA_RADIUS - normal_dist, 0)

    def physical_collision(self):
        """Return depenetration between ninja and one way (None if no penetration)."""
        return self.calculate_depenetration(self.sim.ninja)

    def logical_collision(self):
        """Return wall normal if the ninja enters walled state from entity, else return None."""
        collision_result = self.calculate_depenetration(self.sim.ninja)
        if collision_result:
            if abs(self.normal_x) == 1:
                return self.normal_x
