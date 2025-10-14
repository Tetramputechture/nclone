import math

from ..entities import Entity
from ..physics import (
    sweep_circle_vs_tiles,
    get_single_closest_point,
    overlap_circle_vs_circle,
)
from ..ninja import NINJA_RADIUS


class EntityDeathBall(Entity):
    """Death Ball Entity (Type 25)

    A dynamic hazard that actively seeks and pursues the ninja. Death balls feature complex physics
    interactions with the environment and other death balls, making them challenging and unpredictable
    obstacles.

    Physical Properties:
        - Dual Collision Radii:
            * Ninja Collision: 5 pixels (deadly contact)
            * Environment Collision: 8 pixels (walls and other death balls)
        - Movement Constants:
            * Acceleration: 0.04 pixels/frame²
            * Max Speed: 0.85 pixels/frame
            * Drag (Above Max Speed): 0.9 multiplier
            * Drag (No Target): 0.95 multiplier
        - Max Per Level: 64 instances

    Behavior:
        - Target Seeking:
            * Continuously accelerates toward nearest valid ninja
            * Maintains velocity with drag when exceeding max speed
            * Decelerates when no valid target exists
        - Collision Response:
            * Bounces off walls with variable strength based on impact speed
            * Repels other death balls within 16-pixel range
            * Bounces away from ninja on contact (while killing ninja)
        - Wall Interaction:
            * Uses swept circle collision detection for high speeds
            * 16 iterations of depenetration for precise wall contact
            * Projects velocity onto surface normal for realistic bounces
            * Bounce strength varies (1x or 2x) based on impact speed

    AI Strategy Notes:
        - Use walls and obstacles to redirect or slow death balls
        - Consider death ball momentum when planning routes
        - Multiple death balls can create chain reactions
        - Death balls can be temporarily avoided by breaking line of sight

    Technical Implementation:
        - Implements both physical and logical collision systems
        - Uses interpolation for high-speed wall collisions
        - Handles complex interactions with other death balls
        - Supports position logging for debugging/replay
        - Maintains separate collision radii for different interaction types
    """

    RADIUS = 5  # radius for collisions against ninjas
    RADIUS2 = 8  # radius for collisions against other balls and tiles
    ACCELERATION = 0.04
    MAX_SPEED = 0.85
    DRAG_MAX_SPEED = 0.9
    DRAG_NO_TARGET = 0.95
    MAX_COUNT_PER_LEVEL = 64

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_thinkable = True
        self.is_logical_collidable = True
        self.xspeed, self.yspeed = 0, 0

    def think(self):
        """Make the ball move towards the closest ninja. Handle collision with tiles and bounces
        against other balls and ninjas."""
        ninja = self.sim.ninja
        if (
            not ninja.is_valid_target()
        ):  # If no valid targets, decelerate ball to a stop
            self.xspeed *= self.DRAG_NO_TARGET
            self.yspeed *= self.DRAG_NO_TARGET
        else:  # Otherwise, apply acceleration towards closest ninja. Apply drag if speed exceeds 0.85.
            dx = ninja.xpos - self.xpos
            dy = ninja.ypos - self.ypos
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 0:
                dx /= dist
                dy /= dist
            self.xspeed += dx * self.ACCELERATION
            self.yspeed += dy * self.ACCELERATION
            speed = math.sqrt(self.xspeed**2 + self.yspeed**2)
            if speed > self.MAX_SPEED:
                new_speed = (speed - self.MAX_SPEED) * self.DRAG_MAX_SPEED
                if (
                    new_speed <= 0.01
                ):  # If speed exceed the cap by a tiny amount, remove the excedent
                    new_speed = 0
                new_speed += self.MAX_SPEED
                self.xspeed = self.xspeed / speed * new_speed
                self.yspeed = self.yspeed / speed * new_speed
        xpos_old = self.xpos
        ypos_old = self.ypos
        self.xpos += self.xspeed
        self.ypos += self.yspeed

        # Interpolation routine for high-speed wall collisions.
        time = sweep_circle_vs_tiles(
            self.sim, xpos_old, ypos_old, self.xspeed, self.yspeed, self.RADIUS2 * 0.5
        )
        self.xpos = xpos_old + time * self.xspeed
        self.ypos = ypos_old + time * self.yspeed

        # Depenetration routine for collision against tiles.
        xnormal, ynormal = 0, 0
        for _ in range(16):
            result, closest_point = get_single_closest_point(
                self.sim, self.xpos, self.ypos, self.RADIUS2
            )
            if result == 0:
                break
            a, b = closest_point
            dx = self.xpos - a
            dy = self.ypos - b
            dist = math.sqrt(dx**2 + dy**2)
            depen_len = self.RADIUS2 - dist * result
            if depen_len < 0.0000001:
                break
            if dist == 0:
                return
            xnorm = dx / dist
            ynorm = dy / dist
            self.xpos += xnorm * depen_len
            self.ypos += ynorm * depen_len
            xnormal += xnorm
            ynormal += ynorm

        # If there has been tile colision, project speed of deathball onto surface and add bounce if applicable.
        normal_len = math.sqrt(xnormal**2 + ynormal**2)
        if normal_len > 0:
            dx = xnormal / normal_len
            dy = ynormal / normal_len
            dot_product = self.xspeed * dx + self.yspeed * dy
            if (
                dot_product < 0
            ):  # Project velocity onto surface only if moving towards surface
                speed = math.sqrt(self.xspeed**2 + self.yspeed**2)
                bounce_strength = 1 if speed <= 1.35 else 2
                self.xspeed -= dx * dot_product * bounce_strength
                self.yspeed -= dy * dot_product * bounce_strength

        # Handle bounces with other deathballs
        db_count = self.sim.map_data[1200]
        if self.index + 1 < db_count:
            db_targets = self.sim.entity_dic[self.type][self.index + 1 :]
            for db_target in db_targets:
                dx = self.xpos - db_target.xpos
                dy = self.ypos - db_target.ypos
                dist = math.sqrt(dx**2 + dy**2)
                if dist < 16:
                    dx = dx / dist * 4
                    dy = dy / dist * 4
                    self.xspeed += dx
                    self.yspeed += dy
                    db_target.xspeed -= dx
                    db_target.yspeed -= dy
        self.grid_move()

    def logical_collision(self):
        """If the ninja touches the ball, kill it and make the ball bounce from it."""
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            if overlap_circle_vs_circle(
                self.xpos, self.ypos, self.RADIUS, ninja.xpos, ninja.ypos, NINJA_RADIUS
            ):
                dx = self.xpos - ninja.xpos
                dy = self.ypos - ninja.ypos
                dist = math.sqrt(dx**2 + dy**2)
                self.xspeed += dx / dist * 10
                self.yspeed += dy / dist * 10
                ninja.kill(0, 0, 0, 0, 0)
