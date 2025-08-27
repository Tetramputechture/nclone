
from ..entities import Entity
from ..physics import *
from ..ninja import NINJA_RADIUS


class EntityBounceBlock(Entity):
    """Bounce Block Entity (Type 17)

    A dynamic physics-based block that acts as a spring-mass system, providing momentum-preserving
    interactions with the ninja. Bounce blocks are crucial for advanced movement techniques and
    can be used for creative routing strategies.

    Physical Properties:
        - Size: 9*9 pixel square (18*18 total)
        - Collision Type: Square with spring physics
        - Max Per Level: 512 instances
        - Spring Constants:
            * Stiffness: 0.0222 (determines spring force)
            * Dampening: 0.98 (velocity decay per frame)
            * Strength: 0.2 (force distribution ratio to ninja)

    Behavior:
        - Spring Physics:
            * Maintains an origin point and current position
            * Applies spring force based on displacement from origin
            * Force proportional to distance (Hooke's Law)
            * Velocity decays through dampening
        - Collision Response:
            * 80% of collision force applied to block
            * 20% of collision force applied to ninja
            * Supports both physical and logical (wall) collisions
            * Additional 0.1 pixel buffer for wall interactions

    AI Strategy Notes:
        - Use for momentum preservation in complex jumps
        - Can create dynamic platforms for height gain
        - Chain interactions for extended movement sequences
        - Consider block's current state (compressed/extended) for timing

    Technical Implementation:
        - Updates position and velocity each frame
        - Applies spring forces relative to origin point
        - Handles both physical and logical collision types
        - Supports position logging for debugging/replay
    """
    ENTITY_TYPE = 17
    SEMI_SIDE = 9
    STIFFNESS = 0.02222222222222222
    DAMPENING = 0.98
    STRENGTH = 0.2
    MAX_COUNT_PER_LEVEL = 512

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_physical_collidable = True
        self.is_logical_collidable = True
        self.is_movable = True
        self.xspeed, self.yspeed = 0, 0
        self.xorigin, self.yorigin = self.xpos, self.ypos

    def move(self):
        """Update the position and speed of the bounce block by applying the spring force and dampening."""
        self.xspeed *= self.DAMPENING
        self.yspeed *= self.DAMPENING
        self.xpos += self.xspeed
        self.ypos += self.yspeed
        xforce = self.STIFFNESS * (self.xorigin - self.xpos)
        yforce = self.STIFFNESS * (self.yorigin - self.ypos)
        self.xpos += xforce
        self.ypos += yforce
        self.xspeed += xforce
        self.yspeed += yforce
        self.grid_move()

    def physical_collision(self):
        """Apply 80% of the depenetration to the bounce block and 20% to the ninja."""
        ninja = self.sim.ninja
        depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                            self.SEMI_SIDE + NINJA_RADIUS)
        if depen:
            depen_x, depen_y = depen[0]
            depen_len = depen[1][0]
            self.xpos -= depen_x * depen_len * (1-self.STRENGTH)
            self.ypos -= depen_y * depen_len * (1-self.STRENGTH)
            self.xspeed -= depen_x * depen_len * (1-self.STRENGTH)
            self.yspeed -= depen_y * depen_len * (1-self.STRENGTH)
            return (depen_x, depen_y), (depen_len * self.STRENGTH, depen[1][1])

    def logical_collision(self):
        """Check if the ninja can interact with the wall of the bounce block"""
        ninja = self.sim.ninja
        depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                            self.SEMI_SIDE + NINJA_RADIUS + 0.1)
        if depen:
            return depen[0][0]
