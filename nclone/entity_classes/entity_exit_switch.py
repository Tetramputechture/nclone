
from ..entities import Entity
from ..physics import *
from ..ninja import NINJA_RADIUS


class EntityExitSwitch(Entity):
    """Exit Switch Entity (Type 4)

    A critical progression element that activates its associated exit door when collected.
    Exit switches are the first objective in most levels, creating a two-stage completion
    requirement that influences route planning and risk assessment.

    Physical Properties:
        - Radius: 6 pixels
        - Collision Type: Circle
        - Parent Link: Associated with specific exit door
        - Collection: Single use, disappears after activation

    Behavior:
        - Initial State:
            * Active and collectible
            * Visible in entity grid
            * Parent exit door inactive
        - On Collection:
            * Becomes inactive and disappears
            * Activates parent exit door
            * Adds exit door to entity grid
            * Logs collection event
        - Parent Activation:
            * Marks switch as hit on parent
            * Makes exit door interactable
            * Creates permanent state change
            * No way to deactivate

    AI Strategy Notes:
        - Primary initial objective
        - Consider:
            * Safest path to switch
            * Post-collection route to exit
            * Hazard positioning relative to switch
            * Optional collectibles en route
        - Critical for:
            * Level progression
            * Route planning
            * Risk evaluation
            * Speedrun optimization

    Technical Implementation:
        - Maintains parent door reference
        - Handles entity grid management
        - Updates parent door state
        - Manages collection logging
    """
    RADIUS = 6

    def __init__(self, type, sim, xcoord, ycoord, parent):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        self.parent = parent

    def logical_collision(self):
        """If the ninja is colliding with the switch, open its associated door. This is done in practice
        by adding the parent door entity to the entity grid.
        """
        ninja = self.sim.ninja
        if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                    ninja.xpos, ninja.ypos, NINJA_RADIUS):
            self.active = False
            # Add door to the entity grid so the ninja can touch it
            self.sim.grid_entity[self.parent.cell].append(self.parent)
            self.parent.switch_hit = True  # Mark the switch as hit on the parent Exit door
            self.log_collision()
