import math

from ..entities import Entity
from ..physics import *
from ..ninja import NINJA_RADIUS


class EntityBoostPad(Entity):
    """Launch Pad Entity (Type 24)

    A launch pad is a movement aid that propels the ninja in the direction of their current velocity.
    When touched, it provides a significant boost to the ninja's momentum, making it a critical tool
    for traversing large gaps and reaching high areas.

    Physical Properties:
        - Radius: 6 pixels
        - Collision Type: Circle
        - Max Per Level: 128 instances

    Behavior:
        - Activation: Triggers when ninja makes initial contact
        - Boost Mechanics: Amplifies current velocity vector by adding 2 units in the same direction
        - Buffer Window: 4-frame window after touching to still receive boost
        - One-Shot: Only boosts on initial contact, requires breaking contact to reactivate

    AI Strategy Notes:
        - Use for reaching distant platforms or high areas
        - Plan trajectory carefully as boost direction depends on entry velocity
        - Can chain with other movement mechanics (wall jumps, slopes) for complex routes
        - Consider breaking contact and re-engaging for multiple boosts when needed

    Technical Implementation:
        - Tracks ninja contact state to prevent multiple boosts
        - Normalizes velocity vector before applying boost
        - Applies boost instantaneously on first frame of contact
    """
    ENTITY_TYPE = 24
    RADIUS = 6
    MAX_COUNT_PER_LEVEL = 128

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_movable = True
        self.is_touching_ninja = False

    def move(self):
        """If the ninja starts touching the booster, add 2 to its velocity norm."""
        ninja = self.sim.ninja
        if not ninja.is_valid_target():
            self.is_touching_ninja = False
            return
        if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                    ninja.xpos, ninja.ypos, NINJA_RADIUS):
            if not self.is_touching_ninja:
                vel_norm = math.sqrt(ninja.xspeed**2 + ninja.yspeed**2)
                if vel_norm > 0:
                    x_boost = 2 * ninja.xspeed/vel_norm
                    y_boost = 2 * ninja.yspeed/vel_norm
                    ninja.xspeed += x_boost
                    ninja.yspeed += y_boost
                self.is_touching_ninja = True
        else:
            self.is_touching_ninja = False
