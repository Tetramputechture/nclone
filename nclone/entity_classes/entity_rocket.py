"""Rocket Entity (Type 18).

Reconstructed from `docs/ENTITY_INTEGRATION_SPEC.md` §8.8
(`think` = 0x10412780, ctor = 0x104111A0, `move` = 0x104116D0). The largest
single gate in the corpus: type 18 appears on 15% of attract maps.

**`move` is render-only and must not be ported.** It reads a frame delta and
`1/60`, the discriminator from spec §6.1. All simulation lives in `think`; the
launcher itself never moves.

The launcher is a three-state machine that owns one projectile inline (see
`entity_projectile.py`):

    0 IDLE      one frame: acquire a ninja by line of sight, then hand over
    1 CHARGING  count 15 frames, then aim, seed the projectile and hand over
    2 FLYING    integrate, sweep, steer, detonate on contact

**Only IDLE tests line of sight.** `FUN_10412780`'s state-1 arm gates on
`FUN_10513C30`, which asks whether the target is still *valid*, not whether it
is still *visible*: once a launcher has acquired, the telegraph runs to
completion even if the ninja ducks behind a wall. The port used to re-test sight
every charging frame and reset the counter, which made every launch late by
however many frames the ninja spent under cover.

`CHARGING` also ends the frame it launches -- the arm sets state 2 and returns,
and never falls into the state-2 code, so the projectile spends one frame
sitting on the tube.

### What is verified

The state machine and its transitions; `move` being render-only; the projectile
living inside the launcher struct; the degenerate coincident-player fallback;
that the logical collision tests the *projectile's* position; continuous homing
(the tail of the state-2 arm, `docs/decomp/ROCKET_think_10412780.c`).

Every constructor constant is now read straight off the game rather than
inferred -- the launcher's 176-byte stride is inside the capture, so
`tools/npptrace/rkfields.py` prints them from 41 captures and 105 launches:

    +0x20 max speed   2.2857142857142856   16/7
    +0x28 first step  0.044444444444444439 2/45
    +0x30 step gain   1.0656022367666107   pow(1.1, 2/3) -- Ghidra drops the call
    +0x38 turn rate   0.066666666666666666 1/15
    +0x40 telegraph   15                   frames
    +0x48 aim lead    1.5                  frames, steering only

Note `+0x30` is **not** the literal `1.1` the ctor loads: the ctor calls
`_libm_sse2_pow_precise` on it and Ghidra lost the call, so the stored value is
`1.1 ** (2/3)`. The launch aim itself has no lead.
"""

import math

from ..entities import Entity
from ..physics import find_visible_ninja, overlap_circle_vs_circle
from ..ninja import NINJA_RADIUS
import os

from .entity_projectile import ProjectileState, aim_at

#: Restore the fall-through that used to launch and fly on the same frame.
_FAST_LAUNCH = os.environ.get("NCLONE_AB_ROCKET_FAST_LAUNCH") == "1"



class EntityRocket(Entity):
    """Type 18. Fires one inline projectile at a time along a fixed heading."""

    IDLE = 0
    CHARGING = 1
    FLYING = 2

    MAX_COUNT_PER_LEVEL = 128

    #: `+0x40`, the telegraph in frames: `+0x88` counts up by one a frame and the
    #: rocket leaves the tube when it reaches this. No longer inferred -- the
    #: field is inside the captured 176-byte stride and reads **15** on all 41
    #: captures that contain a launcher, and all 105 captured launches fire on
    #: the frame the counter would reach exactly 15. It is also what stops the
    #: launcher re-firing the instant a rocket bursts.
    LAUNCH_DELAY = 15.0

    def __init__(self, entity_type, sim, xcoord, ycoord):
        super().__init__(entity_type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_thinkable = True
        self.is_logical_collidable = True
        #: `+0x90`.
        self.state = self.IDLE
        #: `+0x98`, the previous state. Render/audio only, kept for exports.
        self.prev_state = self.IDLE
        self.projectile = ProjectileState()
        #: `+0x88`, zeroed on entering LAUNCH. Charge toward the next launch.
        self.charge = 0.0
        self._initial_xpos = self.xpos
        self._initial_ypos = self.ypos
        self._initial_cell = self.cell

    def think(self):
        prev = self.state
        self.prev_state = prev

        if prev == self.IDLE:
            # `FUN_1051E690` -- acquire. One frame, and it does NOT charge.
            if find_visible_ninja(self.sim, self.xpos, self.ypos) is None:
                return
            self.state = self.CHARGING
            self.charge = 0.0
            return

        if prev == self.CHARGING:
            ninja = self.sim.ninja
            # `FUN_10513C30` is a target-validity test, NOT a sight test: once
            # the launcher has acquired, the telegraph runs to completion even
            # if the ninja breaks line of sight. Re-testing sight here and
            # resetting the charge made every launch late by however many
            # frames the ninja spent behind cover -- nine of them on `10134378`,
            # which is a whole rocket-length of lead.
            if ninja is None or not ninja.is_valid_target():
                self.state = self.IDLE
                return
            self.charge += 1.0
            if self.charge < self.LAUNCH_DELAY:
                return
            # Aim at where the ninja is NOW -- `FUN_10513690` reads the live
            # position, with no lead. The port used to aim a think behind, which
            # `launchdir.py` had fitted at 88.6% over 105 captured launches; that
            # was an artefact of launching late, and once the telegraph above was
            # corrected the stale aim cost a demo instead of saving one.
            xdir, ydir = aim_at(self.xpos, self.ypos, ninja.xpos, ninja.ypos)
            self.projectile.launch(self.xpos, self.ypos, xdir, ydir)
            self.state = self.FLYING
            # The launch frame seeds and RETURNS -- `FUN_10412780`'s state-1 arm
            # ends in `state = 2; return`, and it never touches the state-2 code.
            # Confirmed against the captures: on the frame the game enters state
            # 2, `+0x78` still holds the raw ctor seed `2/45`, so the integrator
            # has not run. The projectile spends one frame sitting on the tube.
            #
            # `NCLONE_AB_ROCKET_FAST_LAUNCH=1` restores the old fall-through,
            # which was a compensation for the late launch fixed above.
            if not _FAST_LAUNCH:
                return
            prev = self.FLYING

        if prev == self.FLYING:
            if self.projectile.advance(self.sim):
                self._detonate()
                return
            # Homing, and it really is continuous: the tail of `FUN_10412780`'s
            # state-2 branch steers every frame, after the move and the
            # detonation sweep. This module's header used to record straight
            # flight as the literal reading of the spec with homing "the first
            # thing to check on the next Ghidra pass" -- the pass happened, and
            # `docs/decomp/ROCKET_think_10412780.c` has it.
            ninja = self.sim.ninja
            if ninja is not None and ninja.is_valid_target():
                self.projectile.steer_toward(ninja)
            # THE KILL TEST LIVES HERE, not in `logical_collision`.
            #
            # `logical_collision` is only reached via
            # `gather_entities_from_neighbourhood(sim, ninja.x, ninja.y)`, which
            # returns entities registered in the nine cells around the NINJA.
            # A rocket registers at its LAUNCHER's cell and never moves -- the
            # projectile is a field inside the launcher, not an entity -- so the
            # dispatch only fires when the ninja walks up to the launcher
            # itself. Measured before this existed: a projectile passed within
            # **2.0 px** of a stationary ninja (radius 10) over 600 frames and
            # the ninja did not die, on a level with clear line of sight.
            #
            # Turret (19) and DualLaser (23) already solve this the same way --
            # "their bodies do not collide, and their beams kill inside think"
            # (spec §6.6). A rocket is that same shape of hazard: its lethal
            # region is not at its body. Killing here needs no change to the
            # spatial grid, which native rebuilds only at init and restore.
            self._projectile_kill_test()

    def _detonate(self):
        """Tear the projectile down and re-arm the launcher.

        `FUN_10411400` is unread ([U]); returning to IDLE is the only transition
        that keeps the machine live, and it matches the shipped game's behaviour
        of re-firing after a rocket bursts on a wall.
        """
        self.state = self.IDLE

    def _projectile_kill_test(self):
        """Discrete circle test, projectile vs ninja. Kills and detonates.

        Discrete, not swept, and deliberately: the binary can tunnel a fast
        rocket past the ninja, and a swept test would make the sim MORE lethal
        than the game and start failing demos the real engine passed.
        """
        if self.state != self.FLYING:
            return
        ninja = self.sim.ninja
        if ninja is None or not ninja.is_valid_target():
            return
        if overlap_circle_vs_circle(
            self.projectile.xpos,
            self.projectile.ypos,
            self.projectile.RADIUS,
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS,
        ):
            ninja.kill(0, 0, 0, 0, 0, cause="rocket")
            self._detonate()

    def logical_collision(self):
        """Kept for the case the ninja IS beside the launcher; the real test now
        runs every frame inside `think` (see `_projectile_kill_test`).

        The PROJECTILE kills, not the launcher.

        A discrete circle test, matching the binary: the projectile registers
        itself in the spatial grid and the ninja's collision pass calls this. That
        means the original can tunnel a fast rocket past the ninja, and this
        reproduces it -- adding a swept kill test here would make the sim MORE
        lethal than the game and start failing demos the real engine passed.
        """
        if self.state != self.FLYING:
            return
        ninja = self.sim.ninja
        if ninja is None or not ninja.is_valid_target():
            return
        if overlap_circle_vs_circle(
            self.projectile.xpos,
            self.projectile.ypos,
            self.projectile.RADIUS,
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS,
        ):
            ninja.kill(0, 0, 0, 0, 0, cause="rocket")
            self._detonate()

    def reset_state(self):
        """Reset for fast environment reset (same level, no reallocation)."""
        self.xpos = self._initial_xpos
        self.ypos = self._initial_ypos
        self.cell = self._initial_cell
        self.state = self.IDLE
        self.prev_state = self.IDLE
        self.projectile = ProjectileState()
        self.charge = 0.0
        self.active = True
