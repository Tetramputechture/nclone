from ..entities import Entity
from ..physics import map_orientation_to_vector
from ..ninja import NINJA_RADIUS
from ..constants.physics_constants import ONE_WAY_PLATFORM_SEMI_SIDE


class EntityOneWayPlatform(Entity):
    """One-Way Platform Entity (Type 11)

    A specialized platform that only allows collision from one direction, creating
    directional movement constraints and enabling complex level traversal mechanics.
    The platform's behavior changes based on the ninja's approach angle and velocity.

    Physical Properties:
        - Size: 12*12 pixel square (24*24 total)
        - Max Per Level: 512 instances
        - Orientation: 8 possible directions (0-7)
        - Collision Zones:
            * Wide Zone: 0.9 * NINJA_RADIUS when moving toward center
            * Narrow Zone: 0.51 * NINJA_RADIUS when moving away

    Behavior:
        - Directional Collision:
            * Only solid from one direction
            * Allows passage from other directions
            * Normal based on orientation
            * Supports both physical and logical collisions
        - Collision Detection:
            * Checks approach angle
            * Considers velocity direction
            * Uses adaptive collision width
            * Requires specific entry conditions
        - Special Conditions:
            * Must be moving toward platform
            * Must be within valid distance
            * Previous position matters
            * Velocity projection checked

    AI Strategy Notes:
        - Critical for:
            * One-way passages
            * Forced movement paths
            * Drop-through platforms
            * Vertical progression
        - Consider:
            * Approach direction
            * Entry velocity
            * Platform orientation
            * Alternative routes
        - Can be used for:
            * Shortcuts
            * Safe landing zones
            * Route restrictions
            * Vertical mobility

    Technical Implementation:
        - Complex Physics:
            * Precise depenetration calculation
            * Velocity-based collision width
            * Historical position checking
            * Normal projection tests
        - Dual Collision Types:
            * Physical: Basic collision response
            * Logical: Wall state detection
            * Separate collision zones
            * Direction-specific behavior
    """

    SEMI_SIDE = ONE_WAY_PLATFORM_SEMI_SIDE
    MAX_COUNT_PER_LEVEL = 512

    def __init__(self, type, sim, xcoord, ycoord, orientation):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        self.is_physical_collidable = True
        self.orientation = orientation
        self.normal_x, self.normal_y = map_orientation_to_vector(orientation)

    def calculate_depenetration(self, ninja):
        """Return the depenetration vector between the ninja and the one way. Return nothing if no
        penetration.
        """
        dx = ninja.xpos - self.xpos
        dy = ninja.ypos - self.ypos
        lateral_dist = dy * self.normal_x - dx * self.normal_y
        direction = (
            ninja.yspeed * self.normal_x - ninja.xspeed * self.normal_y
        ) * lateral_dist
        # The platform has a bigger width if the ninja is moving towards its center.
        # 0.9 and the inclusive `<= 0` are `DAT_10571cd8` and the comparison in
        # `docs/decomp/full/entity_1040ED60.c`. The inherited value was 0.91, and
        # that hundredth is worth a tenth of a radius of reach: on `10139391` the
        # ninja clears a platform edge by |lateral| = 21.07 against the true bound
        # of 21.0, and 0.91 turned the miss into a ceiling on frame 183. Five
        # demos of one map family left the game's trajectory right there.
        radius_scalar = 0.9 if direction <= 0 else 0.51
        if abs(lateral_dist) < radius_scalar * NINJA_RADIUS + self.SEMI_SIDE:
            normal_dist = dx * self.normal_x + dy * self.normal_y
            # `FUN_1040ED60` gates on `|normal_dist| < RADIUS + arg8` instead.
            # Ported literally with `arg8 = 0` -- what the physical caller
            # `FUN_1040EF70` passes -- it costs 599 demos, so one of the two
            # readings is wrong. The logical caller `FUN_1040EEC0` forwards a
            # real eighth argument, so the game likely runs this test at two
            # slacks and one shared helper cannot express that. Kept as the
            # inherited form until the logical caller is decompiled.
            if 0 < normal_dist <= NINJA_RADIUS:
                normal_proj = (
                    ninja.xspeed * self.normal_x + ninja.yspeed * self.normal_y
                )
                if normal_proj <= 0:
                    dx_old = ninja.xpos_old - self.xpos
                    dy_old = ninja.ypos_old - self.ypos
                    normal_dist_old = dx_old * self.normal_x + dy_old * self.normal_y
                    if NINJA_RADIUS - normal_dist_old <= 1.1:
                        return (self.normal_x, self.normal_y), (
                            NINJA_RADIUS - normal_dist,
                            0,
                        )

    def physical_collision(self):
        """Return depenetration between ninja and one way (None if no penetration)."""
        return self.calculate_depenetration(self.sim.ninja)

    def logical_collision(self):
        """Return wall normal if the ninja enters walled state from entity, else return None."""
        collision_result = self.calculate_depenetration(self.sim.ninja)
        if collision_result:
            if abs(self.normal_x) == 1:
                return self.normal_x
