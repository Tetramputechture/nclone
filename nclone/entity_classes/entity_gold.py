from ..entities import Entity
from ..physics import overlap_circle_vs_circle
from ..ninja import NINJA_RADIUS


class EntityGold(Entity):
    """Gold Entity (Type 2)

    Optional entities not used by our RL agent.

    Physical Properties:
        - Radius: 6 pixels
        - Collision Type: Circle
        - Max Per Level: 8192 instances
        - Collection: Single touch, disappears after collection

    Behavior:
        - Collection Mechanics:
            * Collected on ninja contact
            * Must be alive (not in victory state)
            * Increments ninja's gold_collected counter
            * Disappears after collection
            * Logs collection event
        - State Management:
            * Initially active and visible
            * Becomes inactive on collection
            * Cannot be recollected
            * No interaction with other entities

    AI Strategy Notes:
        - Optional objectives
        - Consider:
            * Risk vs. reward for collection
            * Path efficiency with gold
            * Safety of collection route
            * Time cost of collection
        - Used for:
            * Score optimization
            * Route variations
            * Risk assessment
            * Performance metrics

    Technical Implementation:
        - Simple circular collision detection
        - Maintains collection counter
        - Handles state deactivation
        - Supports collection logging
        - No complex physics interactions
    """

    RADIUS = 6
    MAX_COUNT_PER_LEVEL = 8192

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        
        # Store initial values for fast reset
        self._initial_xpos = self.xpos
        self._initial_ypos = self.ypos
        self._initial_cell = self.cell

    def logical_collision(self):
        """The gold is collected if touches by a ninja that is not in winning state."""
        ninja = self.sim.ninja
        if ninja.state != 8:
            if overlap_circle_vs_circle(
                self.xpos, self.ypos, self.RADIUS, ninja.xpos, ninja.ypos, NINJA_RADIUS
            ):
                ninja.gold_collected += 1
                self.active = False
                self.log_collision()

    def reset_state(self):
        """Reset gold to initial state for fast environment reset."""
        self.xpos = self._initial_xpos
        self.ypos = self._initial_ypos
        self.cell = self._initial_cell
        self.active = True  # Gold must be active (not collected)
