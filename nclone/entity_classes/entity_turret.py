"""Turret Entity (Type 19).

Reconstructed from `docs/ENTITY_INTEGRATION_SPEC.md` §8.7
(`think` = 0x1041A150, ctor = 0x10418F60, `move` = 0x10419140). At `0x110` bytes
this is the largest entity in the game.

**`move` is render-only and must not be ported** -- it interpolates a rotation
using a frame delta and `1/60` (spec §6.1). All simulation is in `think`.

**The turret is hitscan.** It fires through the shared raycast helper
(`FUN_1050E0E0`, the same one DualLaser uses) and spawns no projectile entity,
so it needs no spawning machinery -- only raycasting. And it has **no logical
collision of its own** (spec §6.6, the `return 0` default), so its body is
harmless and the damage is applied inside `think` at the moment of the shot.

Two things drive the port and a naive turret gets both wrong:

1. **The aim point is stateful and eased.** It is not recomputed from scratch
   each frame; it converges on the predicted position at a rate chosen by
   distance band. "Aim straight at the ninja" will not reproduce demos.
2. **The shot leads the target** by `ninja_velocity * lead_time`.

`+0x98` does double duty -- charge accumulator in TRACKING, fire timer in
FIRING -- and is zeroed on the transition between them.

### What is verified and what is not

Verified: the four states, the eased/banded aim, the velocity lead, the
hitscan mechanism, `move` being render-only, the absence of a body collision,
and the *set* of nine ctor constants.

**Not** verified: which ctor constant lands in which field. The spec recovered
the constants and the field layout but not the assignment, so `BANDS`,
`AIM_CONVERGENCE`, `CHARGE_RATES`, `CHARGE_THRESHOLD` and `FIRE_DELAY` below are
an inferred binding -- every value is a real ctor constant, but the mapping is
reasoned from role (near targets track faster and charge faster) rather than
read. What *is* now read straight off the game's own struct, frame by frame, is
the state sequence: nclone reproduces `10138417`'s turret bit-exactly through
the shot on frame 164 -- state, charge, band, aim and the beam's endpoint.
"""

import math
import os

from ..constants import NINJA_RADIUS
from ..entities import Entity
from ..physics import (
    beam_hits_ninja,
    find_visible_ninja,
    get_raycast_distance,
    raycast_vs_player,
    seg_vs_circle,
)


class EntityTurret(Entity):
    """Type 19. Searches, tracks with a leading eased aim, then hitscans."""

    SEARCHING = 0
    TRACKING = 1
    FIRING = 2
    #: `+0xA4 == 3`. The state round 9 did not know about -- see `think`.
    RECOVERING = 3

    MAX_COUNT_PER_LEVEL = 128

    # ---- ctor constant binding, VERIFIED against 0x10418F60 ----
    # (docs/ENTITY_RE_ROUND2_FINDINGS.md §1). Round 1 had every one of these
    # wrong, which is what made the turret kill 145 of 187 expert attract demos.
    #
    # All three arrays are FOUR long, not three: the think's walk is
    #     i = 0; while (i < 3 && thresh[i] > d2) { i++; band = i; }
    # so `i` reaches 3 and indexes `+0x38 + i*8` / `+0x58 + i*8`, i.e. `+0x50`
    # and `+0x70`. Round 1's 3-tuples were one short, which is why one tuple had
    # to be reused for two different rows.

    #: `+0x20`/`+0x28`/`+0x30`. Squared-distance band edges, descending:
    #: 96², 42², 24². Compared against the squared distance from the AIM POINT to
    #: the predicted ninja position -- this is the aim ERROR, not the range.
    BANDS = (9216.0, 1764.0, 576.0)
    #: `+0x38`..`+0x50`. Per-band aim easing factor. Index 0 is the widest error.
    AIM_CONVERGENCE = (0.02, 7 / 300, 1 / 30, 1 / 30)
    #: `+0x58`..`+0x70`. Per-band charge accumulation per frame.
    #:
    #: **Band 0's rate is `0.0`, and that is the whole safety valve**: while the
    #: aim is more than 96px off target the turret accumulates no charge at all,
    #: so it cannot fire. The aim still eases in at 2%/frame, so a turret does
    #: eventually converge on a stationary target -- but a moving ninja keeps the
    #: error above the gate and is never shot at.
    CHARGE_RATES = (0.0, 0.5, 1.5, 3.5)
    #: `+0xC8`. Charge needed before the turret commits to a shot. Combined with
    #: the rates above: 180 frames (3.0s) at 42-96px, 60 frames at 24-42px,
    #: ~26 frames inside one cell, never beyond 96px.
    CHARGE_THRESHOLD = 90.0
    #: `+0xD0`. Frames between committing and the shot leaving.
    FIRE_DELAY = 15.0
    #: `+0xD8`. Frames spent RECOVERING after the shot before tracking resumes.
    #: The ctor loads `+0xD0` and `+0xD8` from one 16-byte block
    #: (`_DAT_10572CD0`), and the round-9 capture of `10138417` shows the turret
    #: fire on frame 164 and leave state 3 on 179 -- so both halves are `15.0`.
    RECOVER_DELAY = 15.0
    #: `+0x78`. Frames of ninja velocity used to lead the shot.
    LEAD_TIME = 1.5
    #: `+0xF0`, set to 9 by the fire path. Decremented every think and clamped at
    #: 0, but **never read by the simulation** -- state 0 has no gate on it, so it
    #: is a muzzle-flash/audio counter. The port used to gate SEARCHING on it,
    #: which was harmless only because nothing ever wrote it.
    MUZZLE_TIMER = 9

    def __init__(self, entity_type, sim, xcoord, ycoord, orientation):
        super().__init__(entity_type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_thinkable = True
        # Body is harmless -- see the module docstring and spec §6.6.
        self.is_logical_collidable = False
        self.orientation = orientation
        #: `+0xA4`.
        self.state = self.SEARCHING
        #: `+0xA0`.
        self.target = None
        #: `+0x80`/`+0x88`. The eased aim point, latched to the turret's own
        #: position on acquisition so the first frames of tracking swing out
        #: from the barrel rather than teleporting onto the ninja.
        self.aim_x = self.xpos
        self.aim_y = self.ypos
        #: `+0x90`.
        self.band = 0
        #: Restore the round-9 three-state machine: fire, then drop straight to
        #: SEARCHING and re-latch the aim onto the muzzle. Known wrong -- see
        #: `think`. `NCLONE_AB_TURRET_LEGACY=1`.
        self._legacy_ab = os.environ.get("NCLONE_AB_TURRET_LEGACY") == "1"
        #: Route the shot through the 11-limb test instead of `seg_vs_circle`.
        #: Also wrong -- the limb test belongs to the laser family -- but it is
        #: the other half of the pair, so it A/Bs separately.
        #: `NCLONE_AB_TURRET_LIMBS=1`.
        self._limbs_ab = os.environ.get("NCLONE_AB_TURRET_LIMBS") == "1"
        #: `+0x108`. Whether the last shot connected. RECOVERING reads it: a
        #: turret that hit drops its target instead of resuming the track.
        self.hit_ninja = False
        #: `+0x98`. Charge in TRACKING, fire timer in FIRING and RECOVERING.
        self.charge = 0.0
        #: `+0xF0`. Decremented every think, clamped at 0.
        self.timer = 0
        #: Last traced beam, for rendering.
        self.xend, self.yend = self.xpos, self.ypos
        self._initial_cell = self.cell

    def _target_visible(self):
        """`FUN_104190B0`: the target is alive and the path to it is clear."""
        ninja = self.target
        return (
            ninja is not None
            and ninja.is_valid_target()
            and raycast_vs_player(
                self.sim, self.xpos, self.ypos, ninja.xpos, ninja.ypos
            )
        )

    def think(self):
        """The four-state machine of `FUN_1041A150`.

        Round 9 modelled three states and had the turret drop to SEARCHING the
        instant it fired. The binary has a fourth, and the round-9 capture of
        `10138417` shows it plainly: the turret commits on frame 149, winds up
        through 163, **fires on 164** and moves to state 3, sits there for 15
        frames, and returns to TRACKING on 179 with its aim untouched.

        That last detail is the whole bug. Dropping to SEARCHING re-latches the
        aim onto the turret's own muzzle, and the muzzle is by definition close to
        anything the turret can see -- so the aim error collapses into band 3, the
        charge rate jumps to 3.5/frame, and the port re-fires roughly 26 frames
        after a shot. The game keeps the *stale* aim, which the ninja has by then
        run away from, so the error sits in band 0 where the charge rate is `0.0`
        and the turret cannot fire again until it has eased all the way back. On
        `10138417` the port is back to charge 26.5 by frame 200 while the game is
        still at 0.

        Round 9 read the same capture as "fires on 149, 164 and 179, every 15
        frames" and concluded a firing turret re-fires with no cooldown. Those are
        the 1->2, 2->3 and 3->1 transitions of one single shot.
        """
        self.timer -= 1
        if self.timer < 0:
            self.timer = 0

        if self.state == self.SEARCHING:
            ninja = find_visible_ninja(self.sim, self.xpos, self.ypos)
            if ninja is None:
                return
            self.target = ninja
            # Latch the aim origin to the turret's own position.
            self.aim_x, self.aim_y = self.xpos, self.ypos
            self.charge = 0.0
            self.state = self.TRACKING
            return

        if self.state == self.TRACKING:
            if not self._target_visible():
                self.state = self.SEARCHING
                self.target = None
                return
            ninja = self.target
            # Lead the target, then ease the aim point toward the prediction.
            dx = (ninja.xspeed * self.LEAD_TIME + ninja.xpos) - self.aim_x
            dy = (ninja.yspeed * self.LEAD_TIME + ninja.ypos) - self.aim_y
            d2 = dx * dx + dy * dy
            band = self._select_band(d2)
            self.charge += self.CHARGE_RATES[band]
            factor = self.AIM_CONVERGENCE[band]
            self.aim_x += factor * dx
            self.aim_y += factor * dy
            if self.charge <= self.CHARGE_THRESHOLD:
                return
            self.state = self.FIRING
            self.charge = 0.0
            return

        if self.state == self.FIRING:
            # The aim FREEZES while the shot winds up -- the game's `+0x80`/`+0x88`
            # hold still for the whole cycle. Converging it here diverges at frame
            # 150 instead of 165.
            self.charge += 1.0
            if self.charge < self.FIRE_DELAY:
                return
            self._fire()
            self.charge = 0.0
            if self._legacy_ab:
                self.state = self.SEARCHING
                self.target = None
                return
            self.state = self.RECOVERING
            self.timer = self.MUZZLE_TIMER
            return

        if self.state == self.RECOVERING:
            self.charge += 1.0
            if self.charge < self.RECOVER_DELAY:
                return
            # A turret that CONNECTED drops its target; one that missed resumes
            # tracking the same ninja, aim intact, from a zeroed charge.
            if not self._target_visible() or self.hit_ninja:
                self.state = self.SEARCHING
                self.target = None
                return
            self.hit_ninja = False
            self.state = self.TRACKING
            self.charge = 0.0

    def _select_band(self, d2):
        """Port of the binary's band walk.

        ``band = 0; i = 0; while (i < 3 && thresh[i] > d2) { i++; band = i; }``
        -- `i` counts how many thresholds the error is inside, so **0 is the
        widest band and 3 the tightest**. The returned index is what the rates
        and convergence factors are read with.

        `+0x90` is zeroed *before* the walk, not only written from inside it.
        Round 9 read it the other way and had the field keep the previous frame's
        value when the error fell outside every threshold; the game's own struct
        shows it drop to 0 (`10138417`, frames 179 -> 180). Nothing reads the
        field, so this is a fidelity fix, not a behaviour change.
        """
        self.band = 0
        i = 0
        while i < 3 and self.BANDS[i] > d2:
            i += 1
            self.band = i
        return i

    def _fire(self):
        """Hitscan along the eased aim direction.

        Because the aim point is eased and leads the target, the ray genuinely
        misses when the ninja changes direction mid-charge -- that is the intended
        behaviour, not a bug to correct by re-aiming at the last moment.

        The hit test is `seg_vs_circle` (`FUN_1050C690`), **not** the 11-limb
        `beam_hits_ninja`. The binary's callers split cleanly: `FUN_1050B070`
        (limbs) is reached only from the laser family, `FUN_1050C690` from the
        turret and the chaingun drone. Limbs are a strict subset of the circle, so
        the port was under-lethal per shot -- the opposite sign to the state-machine
        bug above, which is why neither showed up alone.
        """
        ninja = self.sim.ninja
        self.hit_ninja = False
        if ninja is None or not ninja.is_valid_target():
            return
        dx = self.aim_x - self.xpos
        dy = self.aim_y - self.ypos
        length = math.sqrt(dx**2 + dy**2)
        if length == 0:
            return
        dx /= length
        dy /= length
        ray_len = get_raycast_distance(self.sim, self.xpos, self.ypos, dx, dy)
        if not ray_len:
            self.xend, self.yend = self.xpos, self.ypos
            return
        self.xend = self.xpos + dx * ray_len
        self.yend = self.ypos + dy * ray_len
        if self._limbs_ab:
            hit = beam_hits_ninja(self.xpos, self.ypos, self.xend, self.yend, ninja)
        else:
            # `ray_len`, not `hypot(xend - xpos, yend - ypos)`: the game passes the
            # raycast's own return value as the segment length.
            hit = seg_vs_circle(
                ninja.xpos,
                ninja.ypos,
                NINJA_RADIUS,
                self.xpos,
                self.ypos,
                self.xend,
                self.yend,
                ray_len,
            )
        if hit:
            self.hit_ninja = True
            ninja.kill(0, 0, 0, 0, 0, cause="turret")

    def reset_state(self):
        """Reset for fast environment reset (same level, no reallocation)."""
        self.cell = self._initial_cell
        self.state = self.SEARCHING
        self.target = None
        self.aim_x, self.aim_y = self.xpos, self.ypos
        self.band = 0
        self.charge = 0.0
        self.timer = 0
        self.hit_ninja = False
        self.xend, self.yend = self.xpos, self.ypos
        self.active = True
