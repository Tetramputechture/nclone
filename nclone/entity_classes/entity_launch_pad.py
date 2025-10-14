from ..entities import Entity
from ..physics import map_orientation_to_vector, overlap_circle_vs_circle
from ..ninja import NINJA_RADIUS


class EntityLaunchPad(Entity):
    """Launch Pad Entity (Type 10)

    A movement aid that propels the ninja in a specific direction with high velocity.
    Launch pads are crucial for reaching distant areas and creating complex movement
    sequences, with special handling for upward launches.

    Physical Properties:
        - Radius: 6 pixels
        - Boost Strength: 36/7 pixels/frame (~5.14)
        - Max Per Level: 256 instances
        - Orientation: 8 possible directions (0-7)
        - Collision: Semi-circular hitbox in boost direction

    Behavior:
        - Boost Mechanics:
            * Direction based on orientation
            * Applies velocity (boost_x * 2/3, boost_y * 2/3)
            * Enhanced upward boost (1 - normal_y scaling)
            * Requires directional contact (-0.1 threshold)
        - Activation:
            * Instant velocity change
            * No cooldown period
            * Can chain with other movement
            * 4-frame input buffer window
        - Special Cases:
            * Upward launches get extra boost
            * Considers ninja's radius for contact
            * Requires valid approach angle
            * Only affects living ninja

    AI Strategy Notes:
        - Critical for:
            * Long-distance traversal
            * Height gain
            * Speed optimization
            * Complex routes
        - Consider:
            * Approach angle
            * Landing trajectory
            * Chain potential
            * Buffer timing
        - Can be used for:
            * Skip sections
            * Speed boosts
            * Alternative paths
            * Height access

    Technical Implementation:
        - Direction System:
            * Uses orientation vectors
            * Handles all 8 directions
            * Special Y-axis scaling
            * Precise contact checks
        - Collision Detection:
            * Circular base collision
            * Directional validation
            * Radius consideration
            * State checking
        - Boost Calculation:
            * Vector-based velocity
            * Directional scaling
            * Upward enhancement
            * Precise thresholds
    """

    RADIUS = 6
    BOOST = 36 / 7
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        self.orientation = orientation
        self.normal_x, self.normal_y = map_orientation_to_vector(orientation)

    def logical_collision(self):
        """If the ninja is colliding with the launch pad (semi circle hitbox), return boost."""
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            if overlap_circle_vs_circle(
                self.xpos, self.ypos, self.RADIUS, ninja.xpos, ninja.ypos, NINJA_RADIUS
            ):
                if (
                    (self.xpos - (ninja.xpos - NINJA_RADIUS * self.normal_x))
                    * self.normal_x
                    + (self.ypos - (ninja.ypos - NINJA_RADIUS * self.normal_y))
                    * self.normal_y
                ) >= -0.1:
                    yboost_scale = 1
                    if self.normal_y < 0:
                        yboost_scale = 1 - self.normal_y
                    xboost = self.normal_x * self.BOOST
                    yboost = self.normal_y * self.BOOST * yboost_scale
                    return (xboost, yboost)
