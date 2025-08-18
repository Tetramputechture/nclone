import math
import array
import struct

from ..entities import Entity, GridSegmentLinear, GridSegmentCircular
from ..physics import *
from ..ninja import NINJA_RADIUS

class EntityLaser(Entity):
    RADIUS = 5.9
    SPIN_SPEED = 0.010471975  # roughly 2pi/600
    SURFACE_FLAT_SPEED = 0.1
    SURFACE_CORNER_SPEED = 0.005524805665672641  # roughly 0.1/(5.9*pi)

    def __init__(self, type, sim, xcoord, ycoord, orientation, mode):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_thinkable = True
        # Find out what is the laser mode : spinner or surface. Surface mode if segment close enough.
        result, closest_point = get_single_closest_point(
            self.sim, self.xpos, self.ypos, 12)
        if result == -1:
            self.mode = 1
        else:
            if closest_point is None:
                self.mode = 1
            else:
                dist = math.sqrt(
                    (closest_point[0] - self.xpos)**2 + (closest_point[1] - self.ypos)**2)
                self.mode = 1 if dist < 7 else 0
        if self.mode == 0:  # Spinner mode
            self.xend, self.yend = self.xpos, self.ypos
            dx, dy = map_orientation_to_vector(orientation)
            self.angle = math.atan2(dy, dx)
            self.dir = -1 if mode == 1 else 1
        elif self.mode == 1:  # Surface mode
            self.xvec, self.yvec = 0, 0
            self.angle = 0
            self.dir = -1 if mode == 1 else 1
            self.sx, self.sy = 0, 0

    def think(self):
        # TODO
        if self.mode == 0:
            self.think_spinner()

    def think_spinner(self):
        # TODO
        angle_new = (self.angle + self.SPIN_SPEED*self.dir) % (2*math.pi)
        dx = math.cos(self.angle)
        dy = math.sin(self.angle)
        self.len = get_raycast_distance(self.sim, self.xpos, self.ypos, dx, dy)
        if self.len:
            self.xend = self.xpos + dx*self.len
            self.yend = self.ypos + dy*self.len
        else:
            self.xend = self.xpos
            self.yend = self.ypos
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            if raycast_vs_player(self.sim, self.xpos, self.ypos, ninja.xpos, ninja.ypos, NINJA_RADIUS):
                ninja_angle = math.atan2(
                    ninja.ypos-self.ypos, ninja.xpos-self.xpos)
                angle_diff = abs(self.angle - ninja_angle) % (2*math.pi)
                if angle_diff <= 0.0052359875:
                    ninja.kill(0, 0, 0, 0, 0)
                else:
                    if check_lineseg_vs_ninja(self.xpos, self.ypos, self.xend, self.yend, ninja):
                        ninja.kill(0, 0, 0, 0, 0)
        self.angle = angle_new

    def think_surface(self):
        segments = gather_segments_from_region(self.sim, self.xpos-12, self.ypos-12,
                                               self.xpos+12, self.ypos+12)
        if not segments:
            return
        while True:
            xspeed = -self.dir*self.yvec*self.SURFACE_FLAT_SPEED
            yspeed = self.dir*self.xvec*self.SURFACE_FLAT_SPEED
            xpos_new = self.xpos + xspeed
            ypos_new = self.ypos + yspeed
            shortest_distance = 9999999
            result = 0
            closest_point = (0, 0)
            for segment in segments:
                is_back_facing, a, b = segment.get_closest_point(
                    xpos_new, ypos_new)
                distance_sq = (xpos_new - a)**2 + (ypos_new - b)**2
                if distance_sq < shortest_distance:
                    shortest_distance = distance_sq
                    closest_point = (a, b)
                    result = -1 if is_back_facing else 1
            dx = xpos_new - closest_point[0]
            dy = ypos_new - closest_point[1]
            if ((self.xpos - self.sx)*dx + (self.ypos - self.sy)*dy) > 0.01 and segment.oriented:
                dist = math.sqrt(
                    (closest_point[0] - self.sx)**2 + (closest_point[1] - self.sy)**2)
                if dist >= 0.0000001:
                    pass
                else:
                    angle = math.atan2(self.yvec, self.xvec)
                    angle += self.dir*self.SURFACE_CORNER_SPEED
                    self.xvec = math.cos(angle)
                    self.yvec = math.sin(angle)

    def get_state(self):
        state = super().get_state()
        # # Normalize angle
        # state[5] = max(0.0, min(1.0, self.angle / (2 * math.pi)))
        # # Normalize direction (-1 or 1)
        # state[6] = max(0.0, min(1.0, (float(self.dir) + 1) / 2))
        return state
