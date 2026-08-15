"""Eye Bat Entity (Type 27).

Reconstructed from `npp.dll` -- see `docs/ENTITY_INTEGRATION_SPEC.md` §8.2
(`think` = 0x1040B2E0, ctor = 0x1040ABC0). This was the guide's "unknown type 27".

A flying hazard that homes on the ninja whenever it has line of sight and coasts
to a halt when it loses it. Structurally it is the death ball with a different
control law, so `entity_death_ball.py` is the template -- but three constants
differ in kind, not just value:

* **`0.85` is a per-axis multiplicative decay, not a max speed.** The death
  ball's `0.85` is its speed cap; the same number here damps each velocity
  component independently while no target is visible. Spec §8.2 flags this
  explicitly as a correction to the guide's anchor table.
* **The speed cap is applied as normalize-then-scale by `min(speed, 3.0)`**, not
  as the death ball's soft "damp only the excess" drag.
* **Two distinct radii.** `12.0` against the ninja (its own `.rdata` const
  block), `5.0` against tiles. Do not conflate them.

EyeBat is thinkable but NOT movable (spec §4), so position integration happens
inside `think`, after the velocity update and before tile depenetration.
"""

import math

from ..entities import Entity
from ..physics import (
    find_visible_ninja,
    get_single_closest_point,
    overlap_circle_vs_circle,
)
from ..ninja import NINJA_RADIUS


class EntityEyeBat(Entity):
    """Type 27. Line-of-sight homing flier."""

    #: Radius against the ninja. `EntityEyeBat::logical_collision` passes
    #: `DAT_10571CF8 = 5.0` to `FUN_1050C650`, so this is 5.0, the same value as
    #: TILE_RADIUS below -- they are not the two distinct radii round 1 read.
    #:
    #: The `12.0` that used to sit here is real but is the magnitude of the push
    #: vector (`nx*12.0, ny*12.0 - 4.0`), not a radius. The `12.0f` opening the
    #: const block is a **frame count**: the render pupil extends over 12 frames.
    RADIUS = 5.0
    #: Radius against tile geometry (`FUN_1040ADA0`'s argument). Deliberately
    #: different from RADIUS; spec §8.2 point 2.
    TILE_RADIUS = 5.0
    #: Acceleration toward the ninja, `0x10571D98`. Shared with the Turret.
    ACCELERATION = 0.1
    #: Per-axis velocity decay while no ninja is visible, `0x10572670`/`0x10572678`.
    DRAG_NO_TARGET = 0.85
    #: Speed clamp, `0x10572398`.
    MAX_SPEED = 3.0
    #: Upper bound of the chase counter (`0xC`).
    MAX_CHASE_TIMER = 12
    #: Depenetration iteration cap. The binary's loop bound was unread; 16 is the
    #: death ball's, and any surface geometry that needs more than 16 passes has
    #: already trapped the entity.
    DEPEN_ITERATIONS = 16

    MAX_COUNT_PER_LEVEL = 512

    def __init__(self, entity_type, sim, xcoord, ycoord):
        super().__init__(entity_type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_thinkable = True
        self.is_logical_collidable = True
        # The ctor zeroes +0x20..+0x38 and holds no constants of its own -- every
        # number above is read from .rdata at use time.
        self.xspeed = 0.0
        self.yspeed = 0.0
        #: `+0x30`. Render-only facing angle; kept so exports match, never read.
        self.facing = 0.0
        #: `+0x34`. Counts UP to 12 while chasing and DOWN to 0 while idle. What
        #: it gates is unread ([U]); it drives no physics here.
        self.chase_timer = 0
        #: `+0x38`. Set on any frame the bat had a target.
        self.chasing = False
        self._initial_xpos = self.xpos
        self._initial_ypos = self.ypos
        self._initial_cell = self.cell

    def think(self):
        """Seek + accelerate + clamp, with decay on target loss, then integrate."""
        ninja = find_visible_ninja(self.sim, self.xpos, self.ypos)
        if ninja is None:
            self.chase_timer = max(0, self.chase_timer - 1)
            self.xspeed *= self.DRAG_NO_TARGET
            self.yspeed *= self.DRAG_NO_TARGET
            self.chasing = False
        else:
            dx = ninja.xpos - self.xpos
            dy = ninja.ypos - self.ypos
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 0:
                dx /= dist
                dy /= dist
            self.xspeed += dx * self.ACCELERATION
            self.yspeed += dy * self.ACCELERATION
            speed = math.sqrt(self.xspeed**2 + self.yspeed**2)
            if speed > 0:
                # Normalize, then scale by min(speed, 3.0): below the cap this is
                # the identity, above it the speed is pinned at exactly 3.0.
                clamp = min(speed, self.MAX_SPEED)
                self.xspeed = clamp * self.xspeed / speed
                self.yspeed = clamp * self.yspeed / speed
            self.facing = math.atan2(self.yspeed, self.xspeed)
            self.chase_timer = min(self.chase_timer + 1, self.MAX_CHASE_TIMER)
            self.chasing = True

        self.xpos += self.xspeed
        self.ypos += self.yspeed

        # Tile depenetration. Same shape as the death ball's, at radius 5, and
        # the velocity is PROJECTED onto the surface afterwards -- the bat does
        # not bounce, but it does not keep driving into the wall either.
        xnormal = 0.0
        ynormal = 0.0
        for _ in range(self.DEPEN_ITERATIONS):
            result, closest_point = get_single_closest_point(
                self.sim, self.xpos, self.ypos, self.TILE_RADIUS
            )
            if result == 0:
                break
            a, b = closest_point
            dx = self.xpos - a
            dy = self.ypos - b
            dist = math.sqrt(dx**2 + dy**2)
            depen_len = self.TILE_RADIUS - dist * result
            if depen_len < 0.0000001:
                break
            if dist == 0:
                break
            xnorm = dx / dist
            ynorm = dy / dist
            self.xpos += xnorm * depen_len
            self.ypos += ynorm * depen_len
            xnormal += xnorm
            ynormal += ynorm

        # Cancel the velocity component pointing INTO the surface. This is the
        # death ball's projection with `bounce_strength` fixed at 1, i.e. slide,
        # never rebound.
        #
        # Without it a bat that reaches a ceiling keeps its full approach speed
        # and rams the same surface every frame forever: the depenetration puts
        # it back at exactly `TILE_RADIUS` each time, so it is pinned, and a
        # pinned bat sits wherever it stalled for the rest of the demo. On
        # `10160699` bat 0 pins under a ceiling at frame 139 holding
        # `yspeed = -2.656`; the game's `+0x28` reads **exactly 0.0** on that
        # frame and 0.084 on the next, i.e. it has been zeroed and is
        # re-accelerating cleanly. By frame 225 ours has not moved since, and it
        # is sitting 20 px from the ninja instead of the game's 50.
        normal_len = math.sqrt(xnormal**2 + ynormal**2)
        if normal_len > 0:
            nx = xnormal / normal_len
            ny = ynormal / normal_len
            dot_product = self.xspeed * nx + self.yspeed * ny
            if dot_product < 0:
                self.xspeed -= nx * dot_product
                self.yspeed -= ny * dot_product

        self.grid_move()

    def logical_collision(self):
        """Lethal on contact."""
        ninja = self.sim.ninja
        if not ninja.is_valid_target():
            return
        if overlap_circle_vs_circle(
            self.xpos, self.ypos, self.RADIUS, ninja.xpos, ninja.ypos, NINJA_RADIUS
        ):
            ninja.kill(0, 0, 0, 0, 0, cause="eye_bat")

    def reset_state(self):
        """Reset for fast environment reset (same level, no reallocation)."""
        self.xpos = self._initial_xpos
        self.ypos = self._initial_ypos
        self.cell = self._initial_cell
        self.xspeed = 0.0
        self.yspeed = 0.0
        self.facing = 0.0
        self.chase_timer = 0
        self.chasing = False
        self.active = True
