import math
import array
import struct

from ..entities import Entity, GridSegmentLinear, GridSegmentCircular
from ..physics import *
from ..ninja import NINJA_RADIUS

class EntityDoorBase(Entity):
    """Parent class that all door type entities inherit from : regular doors, locked doors, trap doors."""

    def __init__(self, type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        self.closed = True
        self.orientation = orientation
        self.sw_xpos = 6 * sw_xcoord
        self.sw_ypos = 6 * sw_ycoord
        self.is_vertical = orientation in (0, 4)
        vec = map_orientation_to_vector(orientation)
        # Find the cell that the door is in for the grid segment.
        door_xcell = math.floor((self.xpos - 12*vec[0]) / 24)
        door_ycell = math.floor((self.ypos - 12*vec[1]) / 24)
        door_cell = clamp_cell(door_xcell, door_ycell)
        # Find the half cell of the door for the grid edges.
        door_half_xcell = 2*(door_cell[0] + 1)
        door_half_ycell = 2*(door_cell[1] + 1)
        # Create the grid segment and grid edges.
        self.grid_edges = []
        if self.is_vertical:
            self.segment = GridSegmentLinear((self.xpos, self.ypos-12), (self.xpos, self.ypos+12),
                                             oriented=False)
            self.grid_edges.append((door_half_xcell, door_half_ycell-2))
            self.grid_edges.append((door_half_xcell, door_half_ycell-1))
            for grid_edge in self.grid_edges:
                sim.ver_grid_edge_dic[grid_edge] += 1
        else:
            self.segment = GridSegmentLinear((self.xpos-12, self.ypos), (self.xpos+12, self.ypos),
                                             oriented=False)
            self.grid_edges.append((door_half_xcell-2, door_half_ycell))
            self.grid_edges.append((door_half_xcell-1, door_half_ycell))
            for grid_edge in self.grid_edges:
                sim.hor_grid_edge_dic[grid_edge] += 1
        sim.segment_dic[door_cell].append(self.segment)
        # Update position and cell so it corresponds to the switch and not the door.
        self.xpos = self.sw_xpos
        self.ypos = self.sw_ypos
        self.cell = clamp_cell(math.floor(self.xpos / 24),
                               math.floor(self.ypos / 24))

    def change_state(self, closed):
        """Change the state of the door from closed to open or from open to closed."""
        self.closed = closed
        self.segment.active = closed
        self.log_collision(0 if closed else 1)
        for grid_edge in self.grid_edges:
            if self.is_vertical:
                self.sim.ver_grid_edge_dic[grid_edge] += 1 if closed else -1
            else:
                self.sim.hor_grid_edge_dic[grid_edge] += 1 if closed else -1

    def get_state(self):
        state = super().get_state()
        # state.append(float(self.closed))  # Already 0 or 1
        # state[7] = float(self.orientation) / 7  # Normalize orientation
        return state
