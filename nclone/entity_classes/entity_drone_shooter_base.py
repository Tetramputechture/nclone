"""Shooter Drone base class -- `EntityDroneShooterBase` in `npp.dll`.

Reconstructed from `docs/ENTITY_INTEGRATION_SPEC.md` §8.3. This is a real
intermediate class in the game's RTTI (spec §3), shared by the chaingun drone
(12) and the laser drone (13), and it is the only entity class in the binary with
more than the usual two virtual slots -- 10 instead of 2. The extra eight are
five per-type behaviour hooks, which is exactly the seam modelled here.

`think` (0x104005D0) is shared verbatim by both types; the two subclasses differ
only in their patrol speed and in those five hooks:

    aim()            vtable[+0x14]   track the target while charging
    fire()           vtable[+0x18]   discharge
    is_fire_done()   vtable[+0x1C]   has the discharge finished?
    on_abort()       vtable[+0x20]   target lost, or discharge ended
    cooldown_tick()  vtable[+0x24]   per-frame work while cooling down

**These drones have no logical collision of their own** (spec §6.6): types 12,
13, 19 and 23 all inherit the `return 0` default, so their *bodies* are harmless
and every kill comes from the projectile or beam. Do not add a kill-on-touch.

### What is verified and what is not

Verified from the binary: the class hierarchy, the four states and their
transitions, the five hook seam, the field offsets, and both patrol speeds
(4/7 and 2/7).

**Not** verified: the five hook bodies and the shared target acquisition
(`FUN_10400240`) are all tagged [U] in §8.3, and the charge/cooldown durations at
`+0x70`/`+0x74` have no known writer. Everything below the state machine is
therefore a reconstruction, marked `[I]` where a number was chosen rather than
read. Closing that gap needs another Ghidra pass over those six functions; until
then the attract-demo harness is the only check, and it can only detect a
model that kills too eagerly, never one that kills too little.
"""



import math
import os

from ..physics import (
    beam_hits_ninja,
    get_raycast_distance,
    raycast_vs_player,
)
from .entity_drone_base import EntityDroneBase


_PER_ENTITY_LIMBS = (
    os.environ.get("NCLONE_AB_PER_ENTITY_LIMBS") == "1"
)

#: The whole weapon cycle as decompiled rather than as reconstructed. ON, and
#: it is the reason this class stopped being a reconstruction: it takes the
#: chaingun's agreement with the capture corpus from 72.18% of drone-frames on
#: state and 49.55% on position to **99.82% / 98.19%**, and the attract gate from
#: 6445/6504 to 6455/6504 with no demo lost. `NCLONE_AB_DRONE_RECONSTRUCTED=1`
#: restores the old model. See `_think_decomp`.
_DECOMP = os.environ.get("NCLONE_AB_DRONE_RECONSTRUCTED") != "1"

#: `DAT_105705C0`. The binary guards every normalisation with
#: `max(|len|, 1.0) * eps < |len|` rather than a bare `len != 0`.
_EPS = 2.220446049250313e-16


class EntityDroneShooterBase(EntityDroneBase):
    """Patrolling drone with a charge / fire / cooldown weapon cycle."""

    IDLE = 0
    AIMING = 1
    FIRING = 2
    COOLDOWN = 3

    # All three were **[I]** -- no writer was ever found for `+0x70`/`+0x74` --
    # and all three were wrong. They are now MEASURED, off the capture corpus's
    # own `+0x7C` state field: 51 completed runs of each state across both
    # capture batches, and every single run is the same length.
    #
    #     state 1  53 frames x51      state 2  80 frames x51
    #     state 3  90 frames x51
    #
    #: Frames of charge before the shot leaves (`+0x70`). Entered with the timer
    #: at 0; the transition fires the frame it reaches 53, and the timer is NOT
    #: reset -- every one of the 51 state-2 entries carries a timer of exactly 53.
    #: §73: the chaingun's kill (`FUN_1050c690`) is a bare circle test.
    #: Subclasses that genuinely thread between limbs override this.
    BEAM_USES_LIMBS = False
    CHARGE_DURATION = 53
    #: Frames the discharge lasts. The `+0x78` timer does not advance during it,
    #: so this is counted separately -- see `fire_timer`.
    FIRE_DURATION = 80
    #: Frames of cooldown before re-acquisition is allowed (`+0x74`). Counts the
    #: `+0x78` timer 0 -> 90; all 34 entries into state 0 carry a timer of 90.
    COOLDOWN_DURATION = 90
    #: `+0x50` swings this fraction of the remaining angle per frame -- see `aim`.
    #: The same 20% the dual laser's beam uses.
    AIM_LERP = 0.2
    #: The discharge sweeps the gun by this much per shot. Fitted over 1404
    #: captured steps: mean |step| = 0.016667, i.e. exactly 1/60 rad.
    SWEEP_STEP = 1.0 / 60.0
    #: The burst starts this many sweep steps BEFORE the ninja's bearing, so it
    #: crosses the target around shot 9 of 20 and ends ten steps past. Measured
    #: as `first_step - (bearing - gun)` = 0.15 = 9 x (1/60) on 76 of 78 type-12
    #: discharges. The burst brackets the ninja; it does not chase it.
    SWEEP_LEAD = 9.0
    #: Frames between shots inside one discharge. `+0x88` counts shots and
    #: `+0x8C` cycles 0..3; across the capture corpus the gap between launches is
    #: **4 on all 969 of them**, giving `FIRE_DURATION / SHOT_INTERVAL` = 20
    #: shots, which is exactly what `+0x88` reaches.
    SHOT_INTERVAL = 4
    #: Only the chaingun's sweep is determined. Type 13 scores 41% on the same
    #: rule from 17 samples on one demo, and §26 says the two do not share a
    #: struct, so it keeps the old behaviour until it has captures of its own.
    SWEEP_ENABLED = False

    # -- the decompiled weapon cycle -----------------------------------------
    #
    # These four are READ OUT OF THE CTOR, not fitted. `FUN_103FE460` (type 12)
    # calls `FUN_104001B0(..., 0x34, 0x5A, 0xC)` and then writes `+0x2C`, and
    # `FUN_104001B0` stores its two ints straight into `+0x70` and `+0x74`:
    #
    #     +0x70 = 0x34 = 52     charge, compared `+0x70 < timer + 1`
    #     +0x74 = 0x5A = 90     cooldown, compared `+0x74 <= timer`
    #     +0xB0 = DAT_10572398 = 3.0    shot period, compared `tick > +0xB0`
    #
    #: Charge length. `52` here is `53` frames of aiming; see `_think_decomp`.
    CHARGE_TICKS = 52
    #: Cooldown length, in frames.
    COOLDOWN_TICKS = 90
    #: Frames between shots, less one -- the tick fires on `tick > SHOT_PERIOD`.
    SHOT_PERIOD = 3.0
    #: `if (0x12 < shots) return done`. Nineteen shots leave, numbered 1..19,
    #: and the twentieth boundary ends the discharge instead of firing.
    MAX_SHOTS = 18
    #: The discharge offsets shot `n` by `(n/18 - 0.5) * 0.3` ALONG THE
    #: PERPENDICULAR -- not by that many radians. Shot 9 is exactly on the base
    #: direction, so the burst brackets the target 8 steps before and 10 after.
    SWEEP_DIV = 18.0
    SWEEP_SPAN = 0.3
    #: Only type 12's five hooks are decompiled; type 13's are a different set
    #: (`HOOK_t13_dronelaser_*`) and keep the reconstruction until they are read.
    DECOMP_DISCHARGE = False

    def __init__(self, entity_type, sim, xcoord, ycoord, orientation, mode, speed):
        super().__init__(entity_type, sim, xcoord, ycoord, orientation, mode, speed)
        self.is_thinkable = True
        # Left False deliberately -- see the module docstring and spec §6.6.
        self.is_logical_collidable = False
        #: Patrol through the whole weapon cycle again -- see `move`.
        self._patrol_ab = os.environ.get("NCLONE_AB_DRONE_PATROL_ALWAYS") == "1"
        #: Abort a charge when the target is lost again -- see `think`.
        self._abort_ab = os.environ.get("NCLONE_AB_DRONE_ABORT") == "1"
        #: The whole verified weapon model, as one switch:
        #: `NCLONE_AB_DRONE_FULL_SHOT=1` gives line-of-sight acquisition, a gun
        #: that tracks the ninja, and a 20-shot burst along the current gun
        #: angle.
        #:
        #: **OFF, and this is the interesting measurement in this class.** With
        #: it on, the drone's observable state becomes frame-exact against the
        #: capture -- on `10131718` every state transition lands on the game's
        #: own frame (0->1 at f37, 1->2 at f90, 2->3 at f170) at the game's own
        #: position -- and the gate collapses from 174/271 to **10/271**, with
        #: `drone_shot` going 21 -> 186.
        #:
        #: That is not a reason to doubt the model. Every piece of it is scored
        #: against the capture and matches: the gun angle to the digit on every
        #: frame of a charge, the state dwells 51/51, the shot spacing 969/969,
        #: acquisition 17/19. What the collapse localises is the ONE hook none of
        #: that covers -- whether a shot kills. See §17.
        _full = os.environ.get("NCLONE_AB_DRONE_FULL_SHOT") == "1"
        # The tracking gun is only HALF the model: without the sweep it points
        # straight at the ninja for the whole discharge. Type 13's sweep is
        # undetermined (see the handoff), so it keeps the axis aim until it
        # has captures -- tracking without a sweep is strictly worse.
        self._aim_track_ab = _full and self.SWEEP_ENABLED
        self._los_acquire_ab = _full
        # Only the type whose sweep is measured gets the burst; the laser
        # drone keeps the single-shot discharge until it has captures.
        self._burst_ab = _full and self.SWEEP_ENABLED
        #: Use the measured bracketing sweep for the discharge -- see
        #: `_begin_sweep`. Only meaningful with the full model.
        self._sweep_ab = _full and self.SWEEP_ENABLED
        #: Re-arm after the cooldown without re-testing -- see think().
        self._rearm_ab = _full
        #: Suppress the shot on the frame the discharge ends (§42).
        self._no_last_shot_ab = (
            os.environ.get("NCLONE_AB_DRONE_NO_LAST_SHOT") == "1"
        )
        #: Delay only the first shot to +SHOT_INTERVAL (§40).
        self._shot_delay_ab = (
            os.environ.get("NCLONE_AB_DRONE_SHOT_DELAY") == "1"
        )
        #: The decompiled cycle, replacing the reconstruction wholesale.
        self._decomp_ab = _DECOMP and self.DECOMP_DISCHARGE
        #: `+0x88` shots fired this discharge, `+0x8C` frames since the last.
        self.shots = 0
        self.shot_tick = 0
        #: `+0x90`/`+0x98`, the discharge's frozen base direction, and
        #: `+0xA0`/`+0xA8`, its signed perpendicular. Set once by `_fire_setup`.
        self.fire_dx, self.fire_dy = 1.0, 0.0
        self.perp_x, self.perp_y = 0.0, 0.0
        #: Direction of the discharge sweep, fixed for the burst.
        self.sweep_sign = 1.0
        #: `+0x50`. Gun angle, kept UNWRAPPED. While idle it follows the facing.
        self.aim_angle = 0.0
        #: `+0x7C`.
        self.state = self.IDLE
        #: `+0x78`. Charge timer in state 1, cooldown timer in state 3. It is
        #: frozen at 53 for the whole of state 2 in every capture, so the
        #: discharge is counted by `fire_timer` instead.
        self.timer = 0
        #: Counts the discharge out. Whatever field the game uses for it is not
        #: `+0x78`; the dwell is 80 frames on all 51 captured runs.
        self.fire_timer = 0
        #: `+0x80`. The game stores a ninja index and uses -1 for "none".
        self.target = None
        #: Aim direction, refreshed by `aim()` while charging.
        self.aim_dx = 1.0
        self.aim_dy = 0.0
        #: Beam/bullet endpoint, for rendering and for the lethality test.
        self.xend = self.xpos
        self.yend = self.ypos
        self._aim_from_facing()

    def move(self):
        """A shooter drone patrols only while IDLE. It stops dead the frame it
        acquires and does not move again until the cooldown ends.

        Measured, not inferred: over both capture batches, 32861 shooter-drone
        frames, the game moves it on **all 21488** frames in state 0 and on
        **none** of the 11373 frames in states 1, 2 and 3. Not one exception.

        This port patrolled through the whole charge/fire/cooldown cycle, so by
        the time it fired it was somewhere the real drone never was -- on
        `10131718` the game's drone stops at (518.857, 300) on frame 37 and stays
        there, while this one had walked to (540, 306) by frame 156 and shot the
        ninja from a corridor the real one had already left.

        `NCLONE_AB_DRONE_PATROL_ALWAYS=1` restores the old behaviour.
        """
        if self.state != self.IDLE and not self._patrol_ab:
            return
        super().move()

    # -- the shared state machine (0x104005D0) -----------------------------

    def think(self):
        if self._decomp_ab:
            self._think_decomp()
            return
        if self.state == self.IDLE:
            self._try_acquire()
            return

        if self.state == self.AIMING:
            # **The charge is never aborted.** All 51 completed state-1 runs in
            # the capture corpus are exactly 53 frames; an abort would show up as
            # a short one and there is not a single one. This port dropped to
            # cooldown the moment `_target_visible` went false, which on
            # `10132175` cut a charge off after 9 frames that the game ran to
            # completion. `NCLONE_AB_DRONE_ABORT=1` restores it.
            if self._abort_ab and not self._target_visible():
                self.timer = 0
                self.state = self.COOLDOWN
                self.on_abort()
                return
            self.aim()
            self.timer += 1
            if self.CHARGE_DURATION <= self.timer:
                self.state = self.FIRING
                # NOT reset: the game carries the charge timer into state 2 and
                # leaves it there. All 51 captured state-2 entries read 53.
                self.fire_timer = 0
                # Shot 0 leaves along the charge's FINAL angle. `+0x50` is
                # unchanged on the transition frame and for the three frames
                # after it, and only moves on the SECOND shot -- on `10131718`
                # it holds 4.99448 from f90 to f93 and jumps at f94. Setting the
                # sweep up before this `fire()` sent shot 0 along the jumped
                # angle instead, which is one wrong shot per discharge.
                if self._shot_delay_ab:
                    # Sweep is still set up HERE -- moving it to the first shot
                    # changes the ninja state it samples and walks the whole
                    # burst off (§40). Only the shot itself is delayed.
                    if self._sweep_ab:
                        self._begin_sweep()
                else:
                    self.fire()
                    if self._sweep_ab:
                        self._begin_sweep()
                return

        elif self.state == self.FIRING:
            # The gun keeps tracking THROUGH the discharge -- `+0x50` is still
            # moving at f94 on `10131718`, four frames into the burst -- and a
            # shot leaves every `SHOT_INTERVAL` frames along wherever it points
            # at that moment.
            #
            # `+0x90`/`+0x98` is NOT the fire direction. It is a stale copy of
            # the gun angle at the FIRST shot, which is why it reads as one
            # frozen direction per discharge on all 51 captured bursts and means
            # nothing. The impact record at `+0xC0` proves it: on `10131718` its
            # bearing from the drone is -1.1503 on f94, matching `+0x50` (-1.15)
            # and not `+0x90` (-1.2837).
            # Incremented BEFORE the test: the frame the state was entered counts
            # as the first of the discharge, so the dwell is `FIRE_DURATION`
            # frames inclusive. Testing first spent 81 where the game spends 80.
            self.fire_timer += 1
            # The last frame of the discharge is also a multiple of
            # SHOT_INTERVAL, so without this the burst gets a 20th shot on the
            # very frame it ends -- and `on_abort` then wipes the beam, leaving a
            # kill with no drone in state 2 to explain it (§42).
            _ending = self._no_last_shot_ab and self.is_fire_done()
            if _ending:
                pass
            elif (self._shot_delay_ab
                    and self.fire_timer == self.SHOT_INTERVAL):
                # First shot: along the angle `_begin_sweep` already produced,
                # with NO further step -- otherwise it lands one increment past
                # where the game's first impact is recorded.
                self.fire()
            elif self._burst_ab and self.fire_timer % self.SHOT_INTERVAL == 0:
                # The gun HOLDS between shots -- 77.5% of captured firing frames
                # show no movement -- and steps by one sweep increment on each
                # shot frame.
                if self._sweep_ab:
                    self.aim_angle += self.sweep_sign * self.SWEEP_STEP
                    self.aim_dx = math.cos(self.aim_angle)
                    self.aim_dy = math.sin(self.aim_angle)
                else:
                    self.aim()
                self.fire()
            if self.is_fire_done():
                self.timer = 0
                self.state = self.COOLDOWN
                self.on_abort()

        elif self.state == self.COOLDOWN:
            self.timer += 1
            self.cooldown_tick()
            if self.COOLDOWN_DURATION <= self.timer:
                # The game RE-ARMS without re-testing. On 10132168 drone 1 the
                # cooldown ends at f389 and the game goes straight back to
                # state 1 at the same position; this port re-ran acquisition,
                # failed, and dropped to state 0 -- then acquired somewhere else
                # later and fired on frames the game never fires on. That is the
                # mechanism behind the full model firing at the wrong times.
                if (self._rearm_ab and self.target is not None
                        and self.target.is_valid_target()):
                    self.state = self.AIMING
                    self.timer = 0
                    self._aim_from_facing()
                elif not self._try_acquire():
                    self.state = self.IDLE
                    self.target = None

    # -- the decompiled cycle ----------------------------------------------

    def _think_decomp(self):
        """`FUN_104005D0`, the shared state machine, as it is actually written.

        Four differences from the reconstruction above, and between them they
        remove every fitted number from the weapon:

        * **the 1 -> 2 transition does not fire.** `vtable[+0x18]`
          (`FUN_103FE510`) only freezes the aim; the shots all come out of
          `vtable[+0x1C]`, which this port called `is_fire_done` and never read.
        * **the discharge has no duration.** It ends when the shot counter passes
          18, which lands on frame 80 because the period is 3 -- the 80 was
          right, but as a consequence rather than a constant.
        * **the charge aborts only on a dead or celebrating ninja**
          (`FUN_10513BD0`: state 6 or 9), never on losing sight of the target.
        * **the cooldown re-runs acquisition** rather than re-arming blind.
        """
        if self.state == self.IDLE:
            self._acquire_decomp()
            return

        if self.state == self.AIMING:
            ninja = self.target
            if ninja is None or ninja.state in (6, 9):
                self.timer = 0
                self.state = self.COOLDOWN
                self.on_abort()
                return
            self.aim()
            t = self.timer
            self.timer = t + 1
            if self.CHARGE_TICKS < t + 1:
                self.state = self.FIRING
                self._fire_setup(ninja.xpos, ninja.ypos,
                                 ninja.xspeed, ninja.yspeed)
            return

        if self.state == self.FIRING:
            if self._discharge_tick():
                self.timer = 0
                self.state = self.COOLDOWN
                self.on_abort()
            return

        # COOLDOWN
        self.timer += 1
        self.cooldown_tick()
        if self.COOLDOWN_TICKS <= self.timer and not self._acquire_decomp():
            self.state = self.IDLE
            self.target = None

    def _acquire_decomp(self):
        """`FUN_10400240`: nearest valid ninja first, then a slack line of sight.

        The binary scores all 8 slots by squared distance, sorts them, and walks
        them nearest-first until one passes `FUN_1050E8D0` -- which is
        `raycast_vs_player`, already an exact port. With one ninja the sort
        degenerates, so this is just the gate and the sightline.
        """
        ninja = self.sim.ninja
        if ninja is None or not ninja.is_valid_target():
            return False
        if not raycast_vs_player(
            self.sim, self.xpos, self.ypos, ninja.xpos, ninja.ypos
        ):
            return False
        self.target = ninja
        self.timer = 0
        self.state = self.AIMING
        self._aim_from_facing()
        self.on_acquire()
        return True

    def on_acquire(self):
        """vtable `+0x10`, the hook `FUN_10400240` calls once it has a target.

        A no-op for the chaingun, whose aim is frozen later, at the 1 -> 2
        transition. The laser drone freezes its beam here instead.
        """

    def _fire_setup(self, tx, ty, vx, vy):
        """`FUN_103FE510`, vtable `+0x18`. Freeze the discharge's aim.

        **No shot leaves here.** `+0x90`/`+0x98` is the unit vector to the target
        at this frame and `+0xA0`/`+0xA8` its perpendicular, signed against the
        target's velocity across the line of fire. Every shot of the discharge is
        built from those two and nothing else; the gun angle `+0x50` is written
        FROM the shot, which is why it reads as one frozen direction per
        discharge and why reading it as the fire direction went nowhere.
        """
        self.shots = 0
        self.shot_tick = 0
        dx = tx - self.xpos
        dy = ty - self.ypos
        dist = math.sqrt(dy * dy + dx * dx)
        if max(abs(dist), 1.0) * _EPS < abs(dist):
            dx /= dist
            dy /= dist
            self.fire_dx, self.fire_dy = dx, dy
            # `if (0.0 <= dir_x*vy - dir_y*vx) negate dir_y else negate dir_x`.
            if 0.0 <= dx * vy - dy * vx:
                self.perp_x, self.perp_y = -dy, dx
            else:
                self.perp_x, self.perp_y = dy, -dx
            return
        # Degenerate -- the target is on top of the drone. Fall back to its
        # velocity, then to +x, and sweep nothing.
        self.perp_x = self.perp_y = 0.0
        vlen = math.sqrt(vy * vy + vx * vx)
        if max(abs(vlen), 1.0) * _EPS < abs(vlen):
            self.fire_dx, self.fire_dy = vx / vlen, vy / vlen
        else:
            self.fire_dx, self.fire_dy = 1.0, 0.0

    def _discharge_tick(self):
        """`FUN_103FE790`, vtable `+0x1C`. Fire the burst; True when it is over.

        Shot `n` (1-based) leaves along `normalise(base + t*perp)` with
        `t = (n/18 - 0.5) * 0.3`, so shot 9 lies exactly on the base direction
        and the nineteen shots bracket it eight steps before and ten after. That
        is a sweep in DIRECTION space, not in angle: the resulting bearing is
        `atan(t)` off the base, not `t`.

        Checked against the capture on `10138394` drone 1 -- the formula
        reproduces the game's own `+0x50` (-0.4375 at shot 18, -0.4537 at shot
        19) and its impact point to the digit, and our raycast then matches the
        game's recorded endpoint to within 1e-9 on all 1951 captured shots.
        """
        self.shot_tick += 1
        if float(self.shot_tick) <= self.SHOT_PERIOD:
            return False
        n = self.shots
        self.shot_tick = 0
        if self.MAX_SHOTS < n:
            return True
        self.shots = n + 1
        t = (self.shots / self.SWEEP_DIV - 0.5) * self.SWEEP_SPAN
        vx = t * self.perp_x + self.fire_dx
        vy = t * self.perp_y + self.fire_dy
        m = math.sqrt(vy * vy + vx * vx)
        if m == 0.0:
            return False
        self.aim_dx, self.aim_dy = vx / m, vy / m
        self.aim_angle = math.atan2(self.aim_dy, self.aim_dx)
        self._trace_beam()
        self._beam_kill()
        return False

    # -- shared target acquisition (FUN_10400240, [U]) ---------------------

    def _try_acquire(self):
        """Take the ninja as a target if it is in the firing lane. True on success.

        The binary's version returns a bool and writes `+0x80`; since state 0 has
        no transition of its own in the decompiled switch, entering state 1 has to
        happen in here.

        **The lane is the drone's facing direction, not a free line of sight.** A
        drone is a grid entity whose gun points along the axis it faces, so
        acquisition reuses the drone family's own corridor test
        (`ninja_in_corridor`, shared with the chaser's verified detection) rather
        than a free raycast. Free-aiming instead makes these drones unmissable:
        measured on the attract corpus, it killed the ninja in 43 of the 43 demos
        containing a chaingun drone that the real game let through.
        """
        ninja = self.sim.ninja
        if ninja is None or not ninja.is_valid_target():
            return False
        if self._los_acquire_ab:
            # **Acquisition is line of sight, not the patrol lane.**
            #
            # Measured against the 19 frames the game moves a shooter drone into
            # state 1, casting from the GAME's own drone position so our patrol
            # cannot confound it: 13 of them happen exactly one frame after the
            # first frame the drone can see the ninja, one on the same frame and
            # three two frames later; the one-frame offset is tick ordering. On
            # the two remaining ones this port's raycast never succeeds at all,
            # over 310 px and 590 px, which is more likely a raycast problem than
            # a second rule.
            #
            # `ninja_in_corridor` is False at **all 19** of those frames, so the
            # lane cannot be it. The ninja's offset ranges from (-15.6, -265.9)
            # to (425.8, -402.0) -- nowhere near the drone's axis in any of them.
            if not raycast_vs_player(
                self.sim, self.xpos, self.ypos, ninja.xpos, ninja.ypos
            ):
                return False
        elif not self.ninja_in_corridor(self.dir, ninja):
            return False
        self.target = ninja
        self.state = self.AIMING
        self.timer = 0
        # The gun starts the charge on the drone's facing and only begins to
        # swing on the NEXT frame: `+0x50` still reads pi on `10131718`'s frame
        # 37, the frame the state changes, and first moves on 38.
        self._aim_from_facing()
        return True

    def _begin_sweep(self):
        """Set up the discharge sweep, measured in §28.

        The gun jumps to `SWEEP_LEAD` steps before the ninja's bearing and then
        steps `SWEEP_STEP` per shot in a fixed direction, so the twenty shots
        bracket the target instead of tracking it.

        The direction is **against the ninja's velocity across the line of
        fire** -- 78 of 78 captured type-12 discharges, without exception.
        """
        ninja = self.sim.ninja
        if ninja is None:
            self.sweep_sign = 1.0
            return
        cross = (ninja.xspeed * math.sin(self.aim_angle)
                 - ninja.yspeed * math.cos(self.aim_angle))
        self.sweep_sign = 1.0 if cross < 0.0 else -1.0
        bearing = math.atan2(ninja.ypos - self.ypos, ninja.xpos - self.xpos)
        diff = bearing - self.aim_angle
        while diff >= math.pi:
            diff -= 2.0 * math.pi
        while diff < -math.pi:
            diff += 2.0 * math.pi
        self.aim_angle += diff - self.sweep_sign * self.SWEEP_LEAD * self.SWEEP_STEP
        self.aim_dx = math.cos(self.aim_angle)
        self.aim_dy = math.sin(self.aim_angle)

    def _aim_from_facing(self):
        dx, dy = self.DIR_TO_VEC[self.dir]
        self.aim_angle = math.atan2(dy, dx)
        self.aim_dx, self.aim_dy = dx, dy

    def _target_visible(self):
        """Is the target still alive, still in the lane, and still in sight?"""
        ninja = self.target
        if ninja is None or not ninja.is_valid_target():
            return False
        if not self.ninja_in_corridor(self.dir, ninja):
            return False
        return raycast_vs_player(self.sim, self.xpos, self.ypos, ninja.xpos, ninja.ypos)

    # -- per-type hooks ----------------------------------------------------

    def aim(self):
        """vtable[+0x14]. Swing the gun toward the ninja, 20% of the gap a frame.

        This is `+0x50` in the captured slab, and the model below reproduces it
        **exactly on every frame of the charge**. On `10131718`'s drone the game
        runs 3.1416, 3.4429, 3.6854, 3.8807, 4.0384 ... 4.7788 over frames 37-60
        and so does this, to every digit the float32 field carries.

        Two things it settles:

        * the gun is **not** axis-aligned. This port pointed it along
          `DIR_TO_VEC[dir]`, so the shot left along the patrol lane; the game's
          leaves along a free angle -- the captured fire direction on that frame
          is (0.2839, -0.9589), which is not any of the four.
        * it tracks the ninja's CURRENT bearing, not the one latched at
          acquisition. `+0xE0` holds the latched bearing (-1.6420 for that whole
          charge) and lerping toward it drifts away from the game by frame 45.

        The gun keeps its unwrapped value -- the angle runs past 2pi rather than
        being normalised, which is what makes it rotate the long way round.

        `+0x50` is a float32 in the capture, so it cannot say whether the game's
        wrap uses a true pi or one of the hand-typed constants the dual laser
        needs; at this resolution both agree.
        """
        if not self._aim_track_ab:
            self.aim_dx, self.aim_dy = self.DIR_TO_VEC[self.dir]
            return
        ninja = self.sim.ninja
        if ninja is not None:
            bearing = math.atan2(ninja.ypos - self.ypos, ninja.xpos - self.xpos)
            diff = bearing - self.aim_angle
            while diff >= math.pi:
                diff -= 2.0 * math.pi
            while diff < -math.pi:
                diff += 2.0 * math.pi
            self.aim_angle += diff * self.AIM_LERP
        self.aim_dx = math.cos(self.aim_angle)
        self.aim_dy = math.sin(self.aim_angle)

    def fire(self):
        """vtable[+0x18]. Discharge along the aim direction. **[I]** -- body unread."""
        self._trace_beam()
        self._beam_kill()

    def is_fire_done(self):
        """vtable[+0x1C]. **[I]** -- body unread."""
        return self.fire_timer >= self.FIRE_DURATION

    def on_abort(self):
        """vtable[+0x20]. **[I]** -- body unread."""
        self.xend, self.yend = self.xpos, self.ypos

    def cooldown_tick(self):
        """vtable[+0x24]. **[I]** -- body unread."""

    # -- shared weapon helpers ---------------------------------------------

    def _trace_beam(self):
        """Cast the shot against tile geometry and record where it stops."""
        length = get_raycast_distance(
            self.sim, self.xpos, self.ypos, self.aim_dx, self.aim_dy
        )
        if length:
            self.xend = self.xpos + self.aim_dx * length
            self.yend = self.ypos + self.aim_dy * length
        else:
            self.xend, self.yend = self.xpos, self.ypos
        #: The raycast's own distance, which the hit test takes as a parameter
        #: rather than recomputing from the endpoints. See
        #: `physics.check_lineseg_vs_ninja`.
        self._beam_len = length or None

    def _beam_kill(self):
        """Kill the ninja if the traced shot crosses it."""
        ninja = self.sim.ninja
        if ninja is None or not ninja.is_valid_target():
            return
        if self._decomp_ab or _PER_ENTITY_LIMBS:
            # §73: the chaingun's kill is `FUN_1050C690`, a bare circle test
            # with no skeleton in it. Per type, not the global env var.
            use_limbs = None if self.BEAM_USES_LIMBS else False
        else:
            use_limbs = None
        if beam_hits_ninja(
            self.xpos, self.ypos, self.xend, self.yend, ninja,
            getattr(self, "_beam_len", None), use_limbs=use_limbs,
        ):
            ninja.kill(0, 0, 0, 0, 0, cause="drone_shot")

    def reset_state(self):
        super().reset_state()
        self.state = self.IDLE
        self.timer = 0
        self.fire_timer = 0
        self.shots = 0
        self.shot_tick = 0
        self.fire_dx, self.fire_dy = 1.0, 0.0
        self.perp_x, self.perp_y = 0.0, 0.0
        self.target = None
        self.aim_dx, self.aim_dy = 1.0, 0.0
        self.xend, self.yend = self.xpos, self.ypos
