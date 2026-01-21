from ..physics import overlap_circle_vs_circle
from ..ninja import NINJA_RADIUS
from .entity_door_base import EntityDoorBase


class EntityDoorLocked(EntityDoorBase):
    """Locked Door Entity (Type 6)

    A switch-activated door that permanently opens when the ninja collects its associated switch.
    Locked doors are key elements in level progression and often gate access to critical areas
    or create one-way paths.

    Physical Properties:
        - Switch Radius: 5 pixels
        - Door Length: 24 pixels (inherited)
        - Max Per Level: 256 instances
        - Orientation: Supports vertical and horizontal (inherited)

    Behavior:
        - Switch Activation:
            * Opens when ninja collects associated switch
            * Opening is permanent (never closes)
            * Switch becomes inactive after collection
            * Increments ninja's doors_opened counter
        - Initial State:
            * Starts closed and locked
            * Requires specific switch collection
            * No proximity-based activation

    AI Strategy Notes:
        - Critical for level progression planning
        - Consider switch collection order for optimal routing
        - Can create permanent shortcuts or escape routes
        - May be used to section levels into distinct phases

    Technical Implementation:
        - Inherits core door functionality from EntityDoorBase
        - Maintains switch collection state
        - Updates ninja progression metrics
        - One-time state change on activation
    """

    RADIUS = 5
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord):
        super().__init__(type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord)
        
        # Store initial values for fast reset (after parent init sets positions)
        self._initial_xpos = self.xpos  # Switch position
        self._initial_ypos = self.ypos  # Switch position
        self._initial_cell = self.cell

    def logical_collision(self):
        """If the ninja collects the associated open switch, open the door."""
        ninja = self.sim.ninja
        if overlap_circle_vs_circle(
            self.xpos, self.ypos, self.RADIUS, ninja.xpos, ninja.ypos, NINJA_RADIUS
        ):
            ninja.doors_opened += 1
            self.change_state(closed=False)
            self.active = False
            
            # Invalidate gym environment caches (switch state changed)
            # PERFORMANCE: Enables efficient cache invalidation
            if hasattr(self.sim, 'gym_env') and self.sim.gym_env:
                self.sim.gym_env.invalidate_switch_cache()

    def reset_state(self):
        """Reset locked door to initial state for fast environment reset."""
        self.xpos = self._initial_xpos
        self.ypos = self._initial_ypos
        self.cell = self._initial_cell
        self.active = True  # Switch must be active (not collected)
        # Reset door to closed state
        if not self.closed:
            self.change_state(closed=True)