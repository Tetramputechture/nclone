import math

from ..entities import Entity
from ..physics import *

class EntityDroneBase(Entity):
    """Parent class that all drone type entities inherit from."""
    RADIUS = 7.5
    GRID_WIDTH = 24
    DIR_TO_VEC = {0: [1, 0], 1: [0, 1], 2: [-1, 0], 3: [0, -1]}
    # Dictionary to choose the next direction from the patrolling mode of the drone.
    # Patrolling modes : {0:follow wall CW, 1:follow wall CCW, 2:wander CW, 3:wander CCW}
    # Directions : {0:keep forward, 1:turn right, 2:go backward, 3:turn left}
    DIR_LIST = {0: [1, 0, 3, 2], 1: [3, 0, 1, 2],
                2: [0, 1, 3, 2], 3: [0, 3, 1, 2]}

    def __init__(self, type, sim, xcoord, ycoord, orientation, mode, speed):
        super().__init__(type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_movable = True
        self.speed = speed
        self.dir = None
        self.turn(orientation // 2)
        self.orientation = orientation
        self.mode = mode
        self.xtarget, self.ytarget = self.xpos, self.ypos
        self.xpos2, self.ypos2 = self.xpos, self.ypos

    def turn(self, dir):
        """Change the drone's direction and log it."""
        self.dir_old = self.dir or dir
        self.dir = dir
        if self.sim.sim_config.full_export:
            self.log_collision(dir)

    def move(self):
        """Make the drone move along the grid. The drone will try to move towards the center of an
        adjacent cell. When at the center of that cell, it will then choose the next cell to move
        towards.
        """
        xspeed = self.speed * self.DIR_TO_VEC[self.dir][0]
        yspeed = self.speed * self.DIR_TO_VEC[self.dir][1]
        dx = self.xtarget - self.xpos
        dy = self.ytarget - self.ypos
        dist = math.sqrt(dx**2 + dy**2)
        # If the drone has reached or passed the center of the cell, choose the next cell to go to.
        if dist < 0.000001 or (dx * (self.xtarget - (self.xpos + xspeed)) + dy * (self.ytarget - (self.ypos + yspeed))) < 0:
            self.xpos, self.ypos = self.xtarget, self.ytarget
            can_move = self.choose_next_direction_and_goal()
            if can_move:
                disp = self.speed - dist
                self.xpos += disp * self.DIR_TO_VEC[self.dir][0]
                self.ypos += disp * self.DIR_TO_VEC[self.dir][1]
        # Otherwise, make the drone keep moving along its current direction.
        else:
            xspeed = self.speed * self.DIR_TO_VEC[self.dir][0]
            yspeed = self.speed * self.DIR_TO_VEC[self.dir][1]
            self.xpos += xspeed
            self.ypos += yspeed
            self.grid_move()

    def choose_next_direction_and_goal(self):
        """Return true if the drone can move in at least one of four directions.
        The directions are tested in the order according to the drone's preference depending of its mode.
        """
        for i in range(4):
            new_dir = (self.dir + self.DIR_LIST[self.mode][i]) % 4
            valid_dir = self.test_next_direction_and_goal(new_dir)
            if valid_dir:
                self.turn(new_dir)
                return True
        return False

    def test_next_direction_and_goal(self, dir):
        """Return true if the drone can move to the adjacent cell along the given direction.
        This is true if there are no walls impeding the drone's movement.
        If true, set the center of the adjacent cell as the drone's next target."""
        xdir, ydir = self.DIR_TO_VEC[dir]
        xtarget = self.xpos + self.GRID_WIDTH*xdir
        ytarget = self.ypos + self.GRID_WIDTH*ydir
        if not ydir:
            cell_x = math.floor((self.xpos + xdir*self.RADIUS) / 12)
            cell_xtarget = math.floor((xtarget + xdir*self.RADIUS) / 12)
            cell_y1 = math.floor((self.ypos - self.RADIUS) / 12)
            cell_y2 = math.floor((self.ypos + self.RADIUS) / 12)
            while cell_x != cell_xtarget:
                if not is_empty_column(self.sim, cell_x, cell_y1, cell_y2, xdir):
                    return False
                cell_x += xdir
        else:
            cell_y = math.floor((self.ypos + ydir*self.RADIUS) / 12)
            cell_ytarget = math.floor((ytarget + ydir*self.RADIUS) / 12)
            cell_x1 = math.floor((self.xpos - self.RADIUS) / 12)
            cell_x2 = math.floor((self.xpos + self.RADIUS) / 12)
            while cell_y != cell_ytarget:
                if not is_empty_row(self.sim, cell_x1, cell_x2, cell_y, ydir):
                    return False
                cell_y += ydir
        self.xtarget, self.ytarget = xtarget, ytarget
        return True

    def get_state(self):
        state = super().get_state()
        # Normalize mode (0-3)
        # state.append(max(0.0, min(1.0, float(self.mode) / 3)))
        # state[6] = max(0.0, min(1.0, (float(self.dir) + 1) /
        #                2 if self.dir is not None else 0.5))  # Normalize direction
        # Normalize orientation
        # state[7] = max(0.0, min(1.0, float(self.orientation) / 7))
        return state
