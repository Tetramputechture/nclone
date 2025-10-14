from ..physics import overlap_circle_vs_circle
from ..ninja import NINJA_RADIUS
from .entity_drone_base import EntityDroneBase


class EntityDroneZap(EntityDroneBase):
    """Zap Drone Entity (Type 14)

    The standard drone type that follows a fixed patrol pattern and is lethal on contact.
    Zap drones provide predictable but challenging obstacles that require careful timing
    and route planning to avoid.

    Physical Properties:
        - Radius: 7.5 pixels (inherited)
        - Speed: 8/7 pixels/frame (~1.14)
        - Max Per Level: 256 instances
        - Grid Movement: 24-pixel cell size (inherited)

    Behavior:
        - Movement Pattern:
            * Follows inherited patrol modes (wall/wander, CW/CCW)
            * Maintains constant speed
            * Grid-aligned movement between cell centers
        - Collision:
            * Instantly lethal on ninja contact
            * Uses circular collision detection
            * No special interaction with other entities
        - Predictability:
            * Fixed patrol routes based on mode
            * Consistent timing and spacing
            * No dynamic behavior changes

    AI Strategy Notes:
        - Learn patrol patterns to time movements
        - Use grid alignment for precise positioning
        - Consider mode when planning routes
        - Watch for overlapping patrol paths
        - Can be used as:
            * Timing obstacles
            * Area denial
            * Forced movement patterns
            * Checkpoint barriers

    Technical Implementation:
        - Inherits core drone functionality from EntityDroneBase
        - Adds lethal collision detection
        - Simple and efficient movement calculations
        - Maintains grid-based positioning system
    """

    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation, mode):
        super().__init__(type, sim, xcoord, ycoord, orientation, mode, 8 / 7)
        self.is_logical_collidable = True

    def logical_collision(self):
        """Kill the ninja if it touches the regular drone."""
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            if overlap_circle_vs_circle(
                self.xpos, self.ypos, self.RADIUS, ninja.xpos, ninja.ypos, NINJA_RADIUS
            ):
                ninja.kill(0, 0, 0, 0, 0)
