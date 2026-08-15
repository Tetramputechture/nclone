import math
import os

#: Restore the orientation-ignoring door winding.
_DOOR_WINDING_AB = os.environ.get("NCLONE_AB_DOOR_WINDING") == "1"

from ..entities import Entity, GridSegmentLinear
from ..physics import map_orientation_to_vector, clamp_cell


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

    DOOR_RADIUS = 12

    def __init__(self, type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        self.closed = True
        #: What the grid edge counters currently reflect, so `change_state` can
        #: apply only the delta. Starts True to match the `+= 1` the segment
        #: build below applies to every one of this door's edges. Twin of
        #: `registered_closed` in `nclone-native/src/entities/switch_door.rs`.
        self._registered_closed = True
        self.orientation = orientation
        self.sw_xpos = 6 * sw_xcoord
        self.sw_ypos = 6 * sw_ycoord
        self.is_vertical = orientation in (0, 4)
        vec = map_orientation_to_vector(orientation)
        # Find the cell that the door is in for the grid segment.
        door_xcell = math.floor((self.xpos - self.DOOR_RADIUS * vec[0]) / 24)
        door_ycell = math.floor((self.ypos - self.DOOR_RADIUS * vec[1]) / 24)
        door_cell = clamp_cell(door_xcell, door_ycell)
        # Find the half cell of the door for the grid edges.
        door_half_xcell = 2 * (door_cell[0] + 1)
        door_half_ycell = 2 * (door_cell[1] + 1)
        # Create the grid segment and grid edges.
        self.grid_edges = []
        # Always wind LOW-to-HIGH, and carry the door's facing separately.
        #
        # A door is `oriented=False`, so the winding is inert for collision --
        # except numerically, and that turns out to matter more than anything it
        # decides. `GridSegmentLinear.get_closest_point` evaluates
        # `x1 + t/lensq * px`, and when `x1` is the FAR end that expression ends
        # in a cancelling subtraction: on a 24 px face wound high-to-low it
        # misses the query point by up to 2^-47 on 725 positions in 20000, where
        # the low-to-high winding is exact on all 20000.
        #
        # A ninja standing on a door floor then gets `floor_normalized_x` of
        # ~-7e-16 instead of 0, and `ninja.py`'s flat-ground test
        # `hor_input * floor_normalized_x >= 0` is a SIGN test -- so a denormal
        # tilt sends flat ground down the uphill branch and hands out
        # `GROUND_ACCEL / 2`. On `MET-S-E-11-04` that is +0.0333 px/frame of
        # free acceleration from frame 11, and it compounds into a death at 619
        # on a demo the game completes.
        #
        # The winding used to be flipped to `-vec` because the dual laser reads
        # its seating normal off `(-py, px)` -- see `10131334`, where the game
        # has `(0, -1)` and the unflipped build gives `(0, +1)`. That was two
        # jobs on one field. The laser now reads `segment.door_normal` instead,
        # which is that same `-vec` stated directly, so both sides get what they
        # need. Dropping the flip without giving the laser `door_normal` costs 30
        # demos, all of them laser deaths.
        #
        # `NCLONE_AB_DOOR_WINDING=1` no longer changes anything and is kept only
        # so old command lines do not error.
        if self.is_vertical:
            a = (self.xpos, self.ypos - self.DOOR_RADIUS)
            b = (self.xpos, self.ypos + self.DOOR_RADIUS)
            self.segment = GridSegmentLinear(a, b, oriented=False)
            self.grid_edges.append((door_half_xcell, door_half_ycell - 2))
            self.grid_edges.append((door_half_xcell, door_half_ycell - 1))
            for grid_edge in self.grid_edges:
                sim.door_ver_grid_edge_dic[grid_edge] += 1
        else:
            a = (self.xpos - self.DOOR_RADIUS, self.ypos)
            b = (self.xpos + self.DOOR_RADIUS, self.ypos)
            self.segment = GridSegmentLinear(a, b, oriented=False)
            self.grid_edges.append((door_half_xcell - 2, door_half_ycell))
            self.grid_edges.append((door_half_xcell - 1, door_half_ycell))
            for grid_edge in self.grid_edges:
                sim.door_hor_grid_edge_dic[grid_edge] += 1
        #: The door's outward facing, for consumers that used to infer it from
        #: the winding. `-vec` reproduces the flipped build's `(-py, px)` exactly
        #: on all four orientations.
        self.segment.door_normal = (-vec[0], -vec[1])

        # Add segment to segment_dic (MUST be outside if/else to work for both vertical and horizontal!)
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
        # The edge entry is a REFCOUNT of "a closed door owns this edge", so it
        # may only move on an actual TRANSITION. `EntityDoorRegular` calls this
        # with `closed=False` from `logical_collision`, i.e. once per frame for
        # as long as the ninja stands in the doorway, and every one of those used
        # to decrement: on `10146777` the two doors at x=648 and x=672 sit at
        # **-17** and **-12** while reading `closed=False`.
        #
        # Nothing tests the magnitude, but plenty tests the truthiness, and -17
        # is as blocking as +1. That is what made an OPEN door stop a floor
        # guard's line-of-sight walk, so the guard never charged and the demo was
        # won by accident. Native has never had the bug -- `switch_door.rs`
        # carries `registered_closed` and applies only the delta -- which is why
        # the two engines disagreed on 47 demos.
        if closed == self._registered_closed:
            return
        self._registered_closed = closed
        for grid_edge in self.grid_edges:
            if self.is_vertical:
                self.sim.door_ver_grid_edge_dic[grid_edge] += 1 if closed else -1
            else:
                self.sim.door_hor_grid_edge_dic[grid_edge] += 1 if closed else -1

    def get_state(self):
        state = super().get_state()
        # state.append(float(self.closed))  # Already 0 or 1
        # state[7] = float(self.orientation) / 7  # Normalize orientation
        return state
