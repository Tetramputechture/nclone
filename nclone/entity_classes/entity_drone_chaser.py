
from ..physics import *
from .entity_drone_zap import EntityDroneZap


class EntityDroneChaser(EntityDroneZap):
    """Chaser Drone Entity

    An advanced drone type that combines standard patrol behavior with pursuit mechanics.
    When the ninja enters its line of sight, the chaser drone can switch to a faster chase
    mode, creating dynamic and challenging encounters.

    Physical Properties:
        - Radius: 7.5 pixels (inherited)
        - Dual Speed Modes:
            * Patrol: 8/7 pixels/frame (~1.14, inherited)
            * Chase: 16/7 pixels/frame (~2.28, double patrol)
        - Max Per Level: 256 instances
        - Grid Movement: 24-pixel cell size (inherited)

    Behavior:
        - Patrol Mode:
            * Standard drone movement patterns
            * Regular speed and grid alignment
            * Follows inherited patrol logic
        - Chase Detection:
            * Checks three directions (forward, diagonal left/right)
            * 12-pixel lateral detection width
            * Requires line of sight to target
        - Chase Mode:
            * Doubles movement speed
            * Maintains grid alignment
            * Pursues detected ninja
            * Returns to patrol when line of sight broken

    AI Strategy Notes:
        - More dangerous than standard drones
        - Can force route changes when detected
        - Line of sight can be broken with obstacles
        - Consider:
            * Safe approach angles
            * Escape routes
            * Using terrain for cover
            * Speed differential during chase

    Technical Implementation:
        - Inherits from EntityDroneZap for base functionality
        - Adds chase state management
        - Implements line of sight detection
        - Handles speed mode transitions
        - Note: Implementation marked as TODO in code

    Note: This class appears to be in development, with some
    functionality marked as TODO. The chase behavior may not
    be fully implemented yet.
    """
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation, mode):
        super().__init__(type, sim, xcoord, ycoord, orientation, mode)
        self.is_thinkable = True
        self.speed_slow = self.speed
        self.speed_chase = 2 * self.speed
        self.chasing = False

    def think(self):
        # TODO
        if not self.chasing:
            ninja = self.sim.ninja
            if ninja.is_valid_target():
                for i in range(-1, 2):
                    dir = (self.dir + i) % 4
                    xdir, ydir = self.DIR_TO_VEC[dir]
                    if xdir*(ninja.xpos - self.xpos) + ydir*(ninja.ypos - self.ypos) > 0:
                        if abs(ydir*(ninja.xpos - self.xpos) - xdir*(ninja.ypos - self.ypos)) <= 12:
                            pass

    def choose_next_direction_and_goal(self):
        # TODO
        super().choose_next_direction_and_goal()
