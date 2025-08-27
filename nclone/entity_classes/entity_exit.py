
from ..entities import Entity
from ..physics import *
from ..ninja import NINJA_RADIUS


class EntityExit(Entity):
    """Exit Door Entity (Type 3)

    The primary goal object in each level. The exit door becomes interactable only after
    its associated exit switch has been collected, and touching it completes the level.
    This two-stage activation creates the core gameplay loop and progression mechanics.

    Physical Properties:
        - Radius: 12 pixels
        - Max Per Level: 16 instances
        - Collision Type: Circle
        - Activation: Requires exit switch collection

    Behavior:
        - Initial State:
            * Not interactable (not in entity grid)
            * Visually present but cannot be touched
            * Awaits switch activation
        - After Switch Collection:
            * Becomes interactable
            * Added to entity grid
            * Touching triggers level completion
        - Win Condition:
            * Must be touched after switch activation
            * Instantly triggers victory state
            * No additional requirements

    AI Strategy Notes:
        - Primary goal for path planning
        - Consider:
            * Route to switch first
            * Optimal path switch → exit
            * Alternative approaches if main path blocked
            * Risk vs. reward for optional collectibles
        - Critical for:
            * Level completion
            * Route optimization
            * Speedrun strategies
            * Risk assessment

    Technical Implementation:
        - Uses circular collision detection
        - Tracks switch activation state
        - Handles win condition triggering
        - Manages entity grid presence
    """
    ENTITY_TYPE = 3
    RADIUS = 12
    MAX_COUNT_PER_LEVEL = 16

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        self.switch_hit = False

    def logical_collision(self):
        """The ninja wins if it touches the exit door. The door is not interactable from the entity
        grid before the exit switch is collected.
        """
        ninja = self.sim.ninja
        if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                    ninja.xpos, ninja.ypos, NINJA_RADIUS):
            ninja.win()
