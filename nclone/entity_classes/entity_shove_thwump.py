import math

from ..entities import Entity
from ..physics import *
from ..ninja import NINJA_RADIUS


class EntityShoveThwump(Entity):
    """Shove Thwump Entity (Type 28)

    A variant of the thwump that activates on contact rather than line of sight, and
    launches in the direction of initial contact. Features a deadly core and complex
    state-based collision behavior, creating unique positioning challenges.

    Physical Properties:
        - Outer Size: 12*12 pixel square (24*24 total)
        - Inner Core: 8-pixel radius (deadly)
        - Max Per Level: 128 instances
        - Movement Speeds:
            * Launch: 4 pixels/frame
            * Retreat: 1 pixel/frame

    Behavior:
        - Movement States:
            * 0: Immobile (at rest)
            * 1: Activated (contact triggered)
            * 2: Launching (moving away)
            * 3: Retreating (returning)
        - Activation Mechanics:
            * Triggers on significant contact (>0.2 penetration)
            * Launches away from contact point
            * Direction set by initial contact
            * One-frame activation window
        - Collision Types:
            * Deadly inner core
            * State-dependent outer collision
            * Directional collision in activated state
            * Wall detection for movement
        - Movement Pattern:
            * Launches after activation
            * Continues until wall hit
            * Retreats to origin
            * Resets for next activation

    AI Strategy Notes:
        - Contact-based hazard
        - Consider:
            * Safe contact angles
            * Launch direction control
            * Movement timing
            * Retreat patterns
        - Can be used for:
            * Forced positioning
            * Direction control
            * Area denial
            * Timing challenges

    Technical Implementation:
        - Complex State System:
            * Multi-state management
            * Direction tracking
            * Origin position memory
            * Activation logic
        - Collision Handling:
            * Dual collision radii
            * State-based collision response
            * Direction-dependent interaction
            * Grid-aligned movement
        - Movement Control:
            * Independent axis movement
            * Wall collision detection
            * Position restoration
            * Grid cell transitions
    """
    SEMI_SIDE = 12
    RADIUS = 8  # for the projectile inside
    MAX_COUNT_PER_LEVEL = 128

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_thinkable = True
        self.is_logical_collidable = True
        self.is_physical_collidable = True
        self.xorigin, self.yorigin = self.xpos, self.ypos
        self.xdir, self.ydir = 0, 0
        self.set_state(0)  # 0:immobile, 1:activated, 2:launching, 3:retreating
        self.activated = False

    def set_state(self, state):
        """Changes the state of the shwump. 0:immobile, 1:activated, 2:launching, 3:retreating
        Also logs it, combined with the direction information into a single integer."""
        self.state = state
        if self.sim.sim_config.full_export:
            dir = map_vector_to_orientation(self.xdir, self.ydir)
            self.log_collision(4 * state + dir // 2)

    def think(self):
        """Update the state of the shwump and move it if possible."""
        if self.state == 1:
            if self.activated:
                self.activated = False
                return
            self.set_state(2)
        if self.state == 3:
            origin_dist = abs(self.xpos - self.xorigin) + \
                abs(self.ypos - self.yorigin)
            if origin_dist >= 1:
                self.move_if_possible(self.xdir, self.ydir, 1)
            else:
                self.xpos = self.xorigin
                self.ypos = self.yorigin
                self.set_state(0)
        elif self.state == 2:
            self.move_if_possible(-self.xdir, -self.ydir, 4)

    def move_if_possible(self, xdir, ydir, speed):
        """Move the shwump depending of state and orientation.
        Not called in Simulator.tick like other entity move functions.
        """
        if self.ydir == 0:
            xpos_new = self.xpos + xdir * speed
            cell_x = math.floor(self.xpos / 12)
            cell_x_new = math.floor(xpos_new / 12)
            if cell_x != cell_x_new:
                cell_y1 = math.floor((self.ypos - 8) / 12)
                cell_y2 = math.floor((self.ypos + 8) / 12)
                if not is_empty_column(self.sim, cell_x, cell_y1, cell_y2, xdir):
                    self.set_state(3)
                    return
            self.xpos = xpos_new
        else:
            ypos_new = self.ypos + ydir * speed
            cell_y = math.floor(self.ypos / 12)
            cell_y_new = math.floor(ypos_new / 12)
            if cell_y != cell_y_new:
                cell_x1 = math.floor((self.xpos - 8) / 12)
                cell_x2 = math.floor((self.xpos + 8) / 12)
                if not is_empty_row(self.sim, cell_x1, cell_x2, cell_y, ydir):
                    self.set_state(3)
                    return
            self.ypos = ypos_new
        self.grid_move()

    def physical_collision(self):
        """Return the depenetration vector for the ninja if it collides with the shwump.
        Note that if the shwump is in activated state, only one of its sides is collidable.
        """
        ninja = self.sim.ninja
        if self.state <= 1:
            depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                                self.SEMI_SIDE + NINJA_RADIUS)
            if depen:
                depen_x, depen_y = depen[0]
                if self.state == 0 or self.xdir * depen_x + self.ydir * depen_y >= 0.01:
                    return depen

    def logical_collision(self):
        """Return the wall normal if the ninja interacts with an active vertical side.
        Kill the ninja if it touches the lethal core of the shwump.
        """
        ninja = self.sim.ninja
        depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                            self.SEMI_SIDE + NINJA_RADIUS + 0.1)
        if depen and self.state <= 1:
            depen_x, depen_y = depen[0]
            if self.state == 0:
                self.activated = True
                if depen[1][1] > 0.2:
                    self.xdir = depen_x
                    self.ydir = depen_y
                    self.set_state(1)
            elif self.state == 1:
                if self.xdir * depen_x + self.ydir * depen_y >= 0.01:
                    self.activated = True
                else:
                    return
            return depen_x
        if overlap_circle_vs_circle(ninja.xpos, ninja.ypos, NINJA_RADIUS,
                                    self.xpos, self.ypos, self.RADIUS):
            ninja.kill(0, 0, 0, 0, 0)
