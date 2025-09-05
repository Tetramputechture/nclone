
from ..physics import *
from ..ninja import NINJA_RADIUS
from .entity_drone_base import EntityDroneBase
from ..constants.physics_constants import MINI_DRONE_RADIUS, MINI_DRONE_GRID_SIZE


class EntityMiniDrone(EntityDroneBase):
    """Mini Drone Entity (Type 26)

    A smaller, faster variant of the standard drone that operates on a tighter grid.
    Mini drones create more intense challenges due to their increased speed and smaller
    detection windows, while their reduced size allows for denser patterns.

    Physical Properties:
        - Radius: 4 pixels (reduced from standard 7.5)
        - Speed: 1.3 pixels/frame (faster than standard)
        - Grid Size: 12 pixels (half of standard)
        - Max Per Level: 512 instances (double standard)

    Behavior:
        - Movement Pattern:
            * Follows inherited patrol modes (wall/wander, CW/CCW)
            * Uses smaller 12*12 grid cells
            * Faster movement between cells
            * More frequent direction changes
        - Collision:
            * Instantly lethal on ninja contact
            * Smaller collision radius
            * More precise positioning required
        - Density:
            * Can fit in tighter spaces
            * Allows for more complex patterns
            * Higher instance limit per level

    AI Strategy Notes:
        - Requires more precise timing than standard drones
        - Smaller safe zones between drones
        - Higher speed means less reaction time
        - Can create:
            * Dense patrol networks
            * Rapid alternating patterns
            * Tight corridors
            * Quick-response barriers

    Technical Implementation:
        - Inherits core drone functionality from EntityDroneBase
        - Overrides physical constants for size and grid
        - Maintains faster movement speed
        - Uses standard collision detection with reduced radius
    """
    RADIUS = MINI_DRONE_RADIUS
    GRID_WIDTH = MINI_DRONE_GRID_SIZE
    MAX_COUNT_PER_LEVEL = 512

    def __init__(self, type, sim, xcoord, ycoord, orientation, mode):
        super().__init__(type, sim, xcoord, ycoord, orientation, mode, 1.3)
        self.is_logical_collidable = True

    def logical_collision(self):
        """Kill the ninja if it touches the mini drone."""
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                        ninja.xpos, ninja.ypos, NINJA_RADIUS):
                ninja.kill(0, 0, 0, 0, 0)
