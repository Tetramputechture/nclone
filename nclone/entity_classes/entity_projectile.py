"""The shared rocket projectile integrator (`EntityRocket` / `EntityPCRocket`).

`docs/ENTITY_INTEGRATION_SPEC.md` §8.8/§8.9 establishes two things that decide the
shape of this module:

1. **No runtime entity spawning is required.** The in-flight projectile is stored
   as *fields inside the launcher struct* -- there is no allocation, no new
   entity and no lifetime management. One rocket entity owns at most one live
   projectile. This overturned the original guide's headline warning that the sim
   would need a whole projectile subsystem.
2. **Types 18 and 29 share every helper** -- grid register, cell recompute,
   sweep, the three collision-resolution variants, and the teardown. They differ
   only in what steers the thing: type 18 aims once at launch, type 29 is
   steered by player input.

So: one integrator (`ProjectileState`), two drivers.

The launcher's own body is harmless; only the projectile kills, via a circle test
at the *projectile's* position (spec §8.0.1 -- the logical collision reads
`+0x50…+0x5C`, which is projectile state, not the launcher's `+0x10`).
"""

import math

from ..physics import sweep_circle_vs_tiles


class ProjectileState:
    """One in-flight rocket: position, unit direction, speed and odometer.

    **The speed model is an accumulator with a ceiling, not a geometric series.**
    `FUN_10412780` state 2 (`docs/decomp/ROCKET_think_10412780.c`) reads:

        if (speed < MAX_SPEED)  step *= SPEED_MULTIPLIER
        speed = min(speed + step, MAX_SPEED)

    so `speed` starts at **zero**, gains a step that itself grows by 1.1^(2/3) a
    frame, and is capped. This module used to read `16/7` as the *initial speed*
    and multiply it every frame with no ceiling -- but `16/7` is `+0x20`, the
    **cap** (`DAT_10572AD0`), and the initial step is `+0x28 = 2/45`
    (`DAT_10572AD8`). The old model launched at 2.44 px/frame and passed 29
    px/frame by frame 40; the real rocket never exceeds 2.29 and takes ~50 frames
    to get there. Rockets arriving many frames early is what made type 18 the
    third-largest source of false kills in the attract corpus.
    """

    #: `+0x20` (`DAT_10572AD0`) -- the terminal speed, an `n/7` like the rest of
    #: the family (spec §6.4).
    MAX_SPEED = 16 / 7
    #: `+0x28` (`DAT_10572AD8`) -- the initial per-frame speed increment.
    INITIAL_STEP = 2 / 45
    #: `+0x30`, VERIFIED: the ctor stores `pow(1.1, 2/3)`, computed at
    #: construction from the same `2/3` at `0x10572AC0` that the ninja ctor uses --
    #: which is why a byte search for it finds nothing. A hardcoded `1.1` is the
    #: same class of bug round 3 closed for the ninja's drag/friction. It scales
    #: the *step*, not the speed.
    SPEED_MULTIPLIER = 1.1 ** (2 / 3)
    #: `+0x38` (`DAT_10572A10`) -- steering gain, applied to the signed
    #: perpendicular component of the direction to the lead point.
    TURN_RATE = 1 / 15
    #: `+0x48` (`DAT_105729E8`) -- how many frames ahead the rocket aims, for both
    #: the ninja's motion and its own.
    LEAD_TIME = 1.5
    #: Collision radius against the NINJA: `EntityRocket::logical_collision`
    #: passes a literal immediate **`0`** as `param_3` of `FUN_1050C650`, so the
    #: only radius in the test is the ninja's own 10px. The `4.0` (`0x10571CF0`)
    #: that used to sit here is the scale on the returned push vector, not a
    #: radius -- using it made the hit circle 14px instead of 10, ~+96% area.
    RADIUS = 0.0
    #: `0x10571CF0`. Scale on the raw, UN-normalized offset in the push vector.
    PUSH_SCALE = 4.0
    #: Safety bound on the odometer. Every level is walled, so the sweep
    #: terminates a rocket in practice; this only stops a rocket that somehow
    #: escapes geometry from flying forever. The real bound is unread.
    MAX_TRAVEL = 4096.0

    def __init__(self):
        self.xpos = 0.0
        self.ypos = 0.0
        self.xdir = 1.0
        self.ydir = 0.0
        self.speed = 0.0
        self.step = 0.0
        self.travel = 0.0

    def launch(self, xpos, ypos, xdir, ydir):
        self.xpos, self.ypos = xpos, ypos
        self.xdir, self.ydir = xdir, ydir
        # The LAUNCH branch zeroes speed and seeds the step from `+0x28`.
        self.speed = 0.0
        self.step = self.INITIAL_STEP
        self.travel = 0.0

    def steer_toward(self, ninja):
        """Rotate the heading toward where the ninja will be in `LEAD_TIME` frames.

        The tail of `FUN_10412780`'s state-2 branch, run after the move and only
        while the ninja is a valid target. Both points are extrapolated by
        `LEAD_TIME`: the ninja along its velocity, the rocket along its own
        heading. The heading then turns by `TURN_RATE` times the signed
        perpendicular component of the direction to that lead point, and is
        renormalised -- proportional steering, not a snap to the bearing.
        """
        lead = self.LEAD_TIME
        dx = (ninja.xspeed * lead + ninja.xpos) - (self.xdir * self.speed * lead + self.xpos)
        dy = (ninja.yspeed * lead + ninja.ypos) - (self.ydir * self.speed * lead + self.ypos)
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            return
        # Signed perpendicular component: cross(dir, unit_delta).
        cross = (dy / length) * self.xdir + (dx / length) * -self.ydir
        xdir_new = -self.ydir * cross * self.TURN_RATE + self.xdir
        ydir_new = self.xdir * cross * self.TURN_RATE + self.ydir
        self.xdir, self.ydir = xdir_new, ydir_new
        norm = math.sqrt(self.xdir * self.xdir + self.ydir * self.ydir)
        if norm != 0:
            self.xdir /= norm
            self.ydir /= norm
        else:
            self.xdir = self.ydir = 0.0

    def advance(self, sim):
        """Integrate one frame. Returns True if the projectile should detonate."""
        if self.speed < self.MAX_SPEED:
            self.step *= self.SPEED_MULTIPLIER
        self.speed = min(self.speed + self.step, self.MAX_SPEED)
        step = self.speed
        self.travel += step

        xpos_old, ypos_old = self.xpos, self.ypos
        dx = self.xdir * step
        dy = self.ydir * step
        # Sweep so a 40px/frame rocket cannot tunnel through a wall.
        time = sweep_circle_vs_tiles(sim, xpos_old, ypos_old, dx, dy, self.RADIUS)
        self.xpos = xpos_old + time * dx
        self.ypos = ypos_old + time * dy
        return time < 1.0 or self.travel > self.MAX_TRAVEL


def aim_at(from_x, from_y, to_x, to_y):
    """Unit vector toward a point, with the game's degenerate-case fallback.

    When the target direction has zero length the game falls back to `(1.0, 0.0)`.
    That branch is real and is the guard behind the truncated assertion string
    `"EntityRocket::think() found a player coincid…"` found in `.rdata`
    (spec §8.8 fact 2), so it is reproduced rather than treated as unreachable.
    """
    dx = to_x - from_x
    dy = to_y - from_y
    length = math.sqrt(dx**2 + dy**2)
    if length != 0:
        return dx / length, dy / length
    return 1.0, 0.0
