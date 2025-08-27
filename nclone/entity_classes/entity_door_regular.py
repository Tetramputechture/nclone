
from ..physics import *
from ..ninja import NINJA_RADIUS
from .entity_door_base import EntityDoorBase


class EntityDoorRegular(EntityDoorBase):
    """Regular Door Entity (Type 5)

    A proximity-activated door that temporarily opens when the ninja is nearby and automatically
    closes after a short delay. Closed doors block entities from passing through.

    Physical Properties:
        - Activation Radius: 10 pixels
        - Door Length: 24 pixels (inherited)
        - Max Per Level: 256 instances
        - Orientation: Supports vertical and horizontal (inherited)

    Behavior:
        - Proximity Detection:
            * Opens when ninja enters 10-pixel activation radius
            * Centered on door's switch position
            * Uses circular collision detection
        - Auto-Close Timer:
            * Closes automatically after 5 frames without ninja contact
            * Timer resets on each ninja contact
            * Prevents door from staying open indefinitely

    AI Strategy Notes:
        - Can be used for temporary protection from hazards

    Technical Implementation:
        - Inherits core door functionality from EntityDoorBase
        - Maintains open timer for auto-close behavior
        - Updates state through collision and think cycles
        - Supports both physical and logical collision detection
    """
    ENTITY_TYPE = 5
    RADIUS = 10
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord):
        super().__init__(type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord)
        self.is_thinkable = True
        self.open_timer = 0

    def think(self):
        """If the door has been opened for more than 5 frames without being touched by the ninja, 
        close it.
        """
        if not self.closed:
            self.open_timer += 1
            if self.open_timer > 5:
                self.change_state(closed=True)

    def logical_collision(self):
        """If the ninja touches the activation region of the door (circle with a radius of 10 at the
        door's center), open it."""
        ninja = self.sim.ninja
        if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                    ninja.xpos, ninja.ypos, NINJA_RADIUS):
            self.change_state(closed=False)
            self.open_timer = 0
