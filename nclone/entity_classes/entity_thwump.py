import math

from ..entities import Entity
from ..physics import (
    is_empty_column,
    is_empty_row,
    penetration_square_vs_point,
    overlap_circle_vs_segment,
)
from ..ninja import NINJA_RADIUS
from ..constants.physics_constants import (
    THWUMP_SEMI_SIDE,
    THWUMP_FORWARD_SPEED,
    THWUMP_BACKWARD_SPEED,
)


class EntityThwump(Entity):
    """Thwump Entity (Type 20)

    A deadly hazard that charges at high speed when the ninja enters its line of sight.
    Thwumps feature directional lethality and can be used as moving platforms, creating
    complex timing and positioning challenges.

    Physical Properties:
        - Size: 9*9 pixel square (18*18 total)
        - Movement Speeds:
            * Forward (Charge): 20/7 pixels/frame (~2.86)
            * Backward (Retreat): 8/7 pixels/frame (~1.14)
        - Max Per Level: 128 instances
        - Orientation: 4 directions (0: right, 2: down, 4: left, 6: up)

    Behavior:
        - Movement States:
            * 0: Immobile (at rest)
            * 1: Forward (charging)
            * -1: Backward (retreating)
        - Detection:
            * Line of sight based
            * 38-pixel activation range (2 * (9 + 10))
            * Checks for clear path to ninja
            * Only triggers when ninja is ahead
        - Collision Types:
            * Deadly face in charge direction
            * Safe sides and opposite face
            * Can be stood on (horizontal movement)
            * Supports wall interactions
        - Movement Pattern:
            * Charges when triggered
            * Continues until wall hit
            * Retreats to origin
            * Resets for next charge

    AI Strategy Notes:
        - Critical timing hazard
        - Consider:
            * Safe approach angles
            * Platform usage opportunities
            * Charge timing and patterns
            * Retreat windows
        - Can be used for:
            * Forced movement timing
            * Dynamic platforms
            * Area denial
            * Speed boosts (riding)

    Technical Implementation:
        - Complex State Management:
            * Tracks origin position
            * Handles state transitions
            * Manages movement speeds
            * Supports logging
        - Collision Systems:
            * Physical collision for basic contact
            * Logical collision for special interactions
            * Directional lethality checks
            * Wall detection for movement
        - Movement Logic:
            * Grid-aligned movement
            * Wall collision detection
            * Origin position tracking
            * Direction-based behavior
    """

    SEMI_SIDE = THWUMP_SEMI_SIDE
    FORWARD_SPEED = THWUMP_FORWARD_SPEED
    BACKWARD_SPEED = THWUMP_BACKWARD_SPEED
    MAX_COUNT_PER_LEVEL = 128

    def __init__(self, type, sim, xcoord, ycoord, orientation):
        super().__init__(type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_movable = True
        self.is_thinkable = True
        self.is_logical_collidable = True
        self.is_physical_collidable = True
        self.orientation = orientation
        self.is_horizontal = orientation in (0, 4)
        self.direction = 1 if orientation in (0, 2) else -1
        self.xorigin, self.yorigin = self.xpos, self.ypos
        self.set_state(0)  # 0:immobile, 1:forward, -1:backward

    def set_state(self, state):
        """Set the thwump's state and log it. 0:immobile, 1:forward, -1:backward"""
        self.state = state
        if self.sim.sim_config.full_export:
            self.log_collision(state % 3)  # The logged value goes from 0 to 2

    def move(self):
        """Update the position of the thwump only if it is already moving. If the thwump retracts past
        its origin, it will stop moving.
        """
        if self.state:  # If not immobile.
            speed = self.FORWARD_SPEED if self.state == 1 else self.BACKWARD_SPEED
            speed_dir = self.direction * self.state
            if not self.is_horizontal:
                ypos_new = self.ypos + speed * speed_dir
                # If the thwump as retreated past its starting point, set its position to the origin.
                if (
                    self.state == -1
                    and (ypos_new - self.yorigin) * (self.ypos - self.yorigin) < 0
                ):
                    self.ypos = self.yorigin
                    self.set_state(0)
                    return
                cell_y = math.floor((self.ypos + speed_dir * 11) / 12)
                cell_y_new = math.floor((ypos_new + speed_dir * 11) / 12)
                if cell_y != cell_y_new:
                    cell_x1 = math.floor((self.xpos - 11) / 12)
                    cell_x2 = math.floor((self.xpos + 11) / 12)
                    if not is_empty_row(self.sim, cell_x1, cell_x2, cell_y, speed_dir):
                        self.set_state(-1)
                        return
                self.ypos = ypos_new
            else:
                xpos_new = self.xpos + speed * speed_dir
                # If the thwump as retreated past its starting point, set its position to the origin.
                if (
                    self.state == -1
                    and (xpos_new - self.xorigin) * (self.xpos - self.xorigin) < 0
                ):
                    self.xpos = self.xorigin
                    self.set_state(0)
                    return
                cell_x = math.floor((self.xpos + speed_dir * 11) / 12)
                cell_x_new = math.floor((xpos_new + speed_dir * 11) / 12)
                if cell_x != cell_x_new:
                    cell_y1 = math.floor((self.ypos - 11) / 12)
                    cell_y2 = math.floor((self.ypos + 11) / 12)
                    if not is_empty_column(
                        self.sim, cell_x, cell_y1, cell_y2, speed_dir
                    ):
                        self.set_state(-1)
                        return
                self.xpos = xpos_new
            self.grid_move()

    def think(self):
        """Make the thwump charge if it has sight of the ninja."""
        ninja = self.sim.ninja
        if not self.state and ninja.is_valid_target():
            activation_range = 2 * (self.SEMI_SIDE + NINJA_RADIUS)
            if not self.is_horizontal:
                # If the ninja is in the activation range
                if abs(self.xpos - ninja.xpos) < activation_range:
                    ninja_ycell = math.floor(ninja.ypos / 12)
                    thwump_ycell = math.floor((self.ypos - self.direction * 11) / 12)
                    thwump_xcell1 = math.floor((self.xpos - 11) / 12)
                    thwump_xcell2 = math.floor((self.xpos + 11) / 12)
                    dy = ninja_ycell - thwump_ycell
                    if dy * self.direction >= 0:
                        for i in range(100):
                            if not is_empty_row(
                                self.sim,
                                thwump_xcell1,
                                thwump_xcell2,
                                thwump_ycell,
                                self.direction,
                            ):
                                dy = ninja_ycell - thwump_ycell
                                break
                            thwump_ycell += self.direction
                        if i > 0 and dy * self.direction <= 0:
                            self.set_state(1)
            else:
                # If the ninja is in the activation range
                if abs(self.ypos - ninja.ypos) < activation_range:
                    ninja_xcell = math.floor(ninja.xpos / 12)
                    thwump_xcell = math.floor((self.xpos - self.direction * 11) / 12)
                    thwump_ycell1 = math.floor((self.ypos - 11) / 12)
                    thwump_ycell2 = math.floor((self.ypos + 11) / 12)
                    dx = ninja_xcell - thwump_xcell
                    if dx * self.direction >= 0:
                        for i in range(100):
                            if not is_empty_column(
                                self.sim,
                                thwump_xcell,
                                thwump_ycell1,
                                thwump_ycell2,
                                self.direction,
                            ):
                                dx = ninja_xcell - thwump_xcell
                                break
                            thwump_xcell += self.direction
                        if i > 0 and dx * self.direction <= 0:
                            self.set_state(1)

    def physical_collision(self):
        """Return the depenetration vector for the ninja if it collides with the thwump."""
        ninja = self.sim.ninja
        return penetration_square_vs_point(
            self.xpos, self.ypos, ninja.xpos, ninja.ypos, self.SEMI_SIDE + NINJA_RADIUS
        )

    def logical_collision(self):
        """Return the wall normal if the ninja can interact with a thwump's side.
        Kill the ninja if it touches the lethal region on the thwump's charging face.
        """
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            depen = penetration_square_vs_point(
                self.xpos,
                self.ypos,
                ninja.xpos,
                ninja.ypos,
                self.SEMI_SIDE + NINJA_RADIUS + 0.1,
            )
            if depen:
                if self.is_horizontal:
                    dx = (self.SEMI_SIDE + 2) * self.direction
                    dy = self.SEMI_SIDE - 2
                    px1, py1 = self.xpos + dx, self.ypos - dy
                    px2, py2 = self.xpos + dx, self.ypos + dy
                else:
                    dx = self.SEMI_SIDE - 2
                    dy = (self.SEMI_SIDE + 2) * self.direction
                    px1, py1 = self.xpos - dx, self.ypos + dy
                    px2, py2 = self.xpos + dx, self.ypos + dy
                if overlap_circle_vs_segment(
                    ninja.xpos, ninja.ypos, NINJA_RADIUS + 2, px1, py1, px2, py2
                ):
                    ninja.kill(0, 0, 0, 0, 0)
                return depen[0][0]
