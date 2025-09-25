import math

from ..entities import Entity, GridSegmentLinear
from ..physics import *


class EntityDoorBase(Entity):
    """Door Base Entity - Parent Class for All Door Types

    Abstract base class that provides core door functionality for regular doors, locked doors,
    and trap doors. Doors are dynamic barriers that can be opened or closed based on specific
    conditions, creating complex navigation and timing challenges.

    Physical Properties:
        - Size: 24 pixels long (vertical or horizontal)
        - Collision Type: Linear segment with grid-based edges
        - Orientation: 8 possible directions (0-7)
            * 0,4: Vertical doors
            * Other values: Horizontal doors

    Core Behavior:
        - State Management:
            * Maintains open/closed state
            * Updates collision geometry based on state
            * Handles grid edge updates for proper navigation
        - Positioning:
            * Door segment position: Physical barrier location
            * Entity position: Switch/trigger location
            * Supports both vertical and horizontal orientations
        - Collision System:
            * Maintains separate collision segments per cell

    Technical Implementation:
        - Grid Integration:
            * Creates and manages grid segments for collision
            * Handles cell-based positioning and updates
        - State Tracking:
            * Maintains orientation and switch position data
            * Handles state change propagation to physics system

    Derived Classes:
        - Regular Door (Type 5): Temporary open on ninja proximity
        - Locked Door (Type 6): Permanent open on switch collection
        - Trap Door (Type 8): Starts open, closes on switch collection

    Note: This is an abstract base class and should not be instantiated directly.
    Use one of the derived door types instead.
    """

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
        door_xcell = math.floor((self.xpos - 12 * vec[0]) / 24)
        door_ycell = math.floor((self.ypos - 12 * vec[1]) / 24)
        door_cell = clamp_cell(door_xcell, door_ycell)
        # Find the half cell of the door for the grid edges.
        door_half_xcell = 2 * (door_cell[0] + 1)
        door_half_ycell = 2 * (door_cell[1] + 1)
        # Create the grid segment and grid edges.
        self.grid_edges = []
        if self.is_vertical:
            self.segment = GridSegmentLinear(
                (self.xpos, self.ypos - 12), (self.xpos, self.ypos + 12), oriented=False
            )
            self.grid_edges.append((door_half_xcell, door_half_ycell - 2))
            self.grid_edges.append((door_half_xcell, door_half_ycell - 1))
            for grid_edge in self.grid_edges:
                sim.ver_grid_edge_dic[grid_edge] += 1
        else:
            self.segment = GridSegmentLinear(
                (self.xpos - 12, self.ypos), (self.xpos + 12, self.ypos), oriented=False
            )
            self.grid_edges.append((door_half_xcell - 2, door_half_ycell))
            self.grid_edges.append((door_half_xcell - 1, door_half_ycell))
            for grid_edge in self.grid_edges:
                sim.hor_grid_edge_dic[grid_edge] += 1
        sim.segment_dic[door_cell].append(self.segment)
        # Update position and cell so it corresponds to the switch and not the door.
        self.xpos = self.sw_xpos
        self.ypos = self.sw_ypos
        self.cell = clamp_cell(math.floor(self.xpos / 24), math.floor(self.ypos / 24))

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
