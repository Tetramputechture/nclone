from ..physics import *
from ..ninja import NINJA_RADIUS
from .entity_door_base import EntityDoorBase


class EntityDoorTrap(EntityDoorBase):
    """Trap Door Entity (Type 8)

    A switch-activated door that starts open and permanently closes when the ninja collects its
    associated switch. Trap doors create urgency and force strategic planning by potentially
    cutting off paths or trapping the ninja in dangerous areas.

    Physical Properties:
        - Switch Radius: 5 pixels
        - Door Length: 24 pixels (inherited)
        - Max Per Level: 256 instances
        - Orientation: Supports vertical and horizontal (inherited)

    Behavior:
        - Initial State:
            * Starts in open position
            * No collision detection when open
            * Switch is active and collectible
        - Switch Activation:
            * Closes permanently when switch is collected
            * Switch becomes inactive after collection
            * Creates permanent barrier once closed
            * No reopening mechanism

    AI Strategy Notes:
        - Critical timing element for route planning
        - Can create point-of-no-return scenarios
        - May require collecting switches in specific order
        - Consider alternative routes after closure
        - Can be used to:
            * Force commitment to a path
            * Create time pressure
            * Block retreat options
            * Separate level sections

    Technical Implementation:
        - Inherits core door functionality from EntityDoorBase
        - Inverted initial state (starts open)
        - One-time state change on activation
        - Permanent state after closure
    """
    ENTITY_TYPE = 8
    RADIUS = 5
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord):
        super().__init__(type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord)
        self.change_state(closed=False)

    def logical_collision(self):
        """If the ninja collects the associated close switch, close the door."""
        ninja = self.sim.ninja
        if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                    ninja.xpos, ninja.ypos, NINJA_RADIUS):
            self.change_state(closed=True)
            self.active = False
