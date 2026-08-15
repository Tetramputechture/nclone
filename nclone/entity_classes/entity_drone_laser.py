"""Laser Drone Entity (Type 13).

Reconstructed from `docs/ENTITY_INTEGRATION_SPEC.md` §8.3. Shares the whole
charge / fire / cooldown state machine with the chaingun drone via
`EntityDroneShooterBase`.

Verified from the binary (spec §8.3):
  * ctor 0x103FF4C0 sets the patrol speed to **2/7** (`0x10572478`) -- the
    slowest drone in the game -- and writes `+0x28` and `+0x2A`;
  * `think` is the shared `0x104005D0`;
  * logical collision is the shared `return 0` default -- the body is harmless,
    only the beam kills;
  * **this type overrides `move`** with `0x103FF6B0`, where the chaingun inherits
    `EntityDroneShooterBase`'s `0x10400580`.

Not verified: the body of that `move` override is unread ([U] in §8.0), so the
inherited grid patrol is used unchanged here. If a future pass finds the override
holds back movement during the charge -- the obvious candidate, since a laser
drone visibly stops to fire -- it belongs in a `move()` on this class.
"""

import math
import os

from ..constants.physics_constants import DRONE_RADIUS
from .entity_drone_shooter_base import EntityDroneShooterBase, _EPS


class EntityDroneLaser(EntityDroneShooterBase):
    """Type 13. Very slow patrol, long charge, then a sustained beam."""

    MAX_COUNT_PER_LEVEL = 256
    RADIUS = DRONE_RADIUS

    #: `0x10572478`, the n/7 speed family (spec §6.4).
    SPEED = 2 / 7

    # **Type-specific after all.** An earlier round pooled types 12 and 13 into
    # one measurement and concluded the durations were shared. That pooling was
    # invalid: the two structs are different sizes -- 0xE8 for the chaingun,
    # 0xD8 for the laser drone -- so half the samples were reading whatever
    # happened to live at that offset in the other layout.
    #
    # Split by type, both are unanimous across every captured run:
    #
    #     type 12   state 1  53 x78    state 2  80 x78    state 3  90 x78
    #     type 13   state 1  46 x19    state 2  80 x19    state 3  60 x19
    #
    # The base class carries the chaingun's.
    #
    # **These are correct and OFF by default**, which needs saying plainly: 46/60
    # costs 4 demos (smoke 1381 -> 1377, `drone_shot` 21 -> 25). A shorter charge
    # and a shorter cooldown mean more discharges, and this port's shot is known
    # to over-kill by roughly 10x (see §17) -- so more discharges means more false
    # kills. The measurement is not in doubt; what converts it into a regression
    # is a different bug. Turn these on with the hit test, not before.
    #
    # `NCLONE_AB_DRONE_TYPE_TIMINGS=1`.
    if os.environ.get("NCLONE_AB_DRONE_TYPE_TIMINGS") == "1":
        CHARGE_DURATION = 46
        COOLDOWN_DURATION = 60

    # -- the decompiled cycle, and it is nothing like the chaingun's ---------
    #
    # Straight out of `FUN_103FF4C0`, which calls
    # `FUN_104001B0(..., 0x2D, 0x3C, 0xD)` and then writes `+0x88 <- 0x50`:
    #
    #     +0x70 = 0x2D = 45   ->  46 frames of charge
    #     +0x74 = 0x3C = 60   ->  60 frames of cooldown
    #     +0x88 = 0x50 = 80   ->  80 frames of beam
    #
    # which is exactly the 46/80/60 the capture corpus measured and this class
    # left switched off. It was off because it "cost 4 demos"; what actually
    # cost them was the rest of the model, and the rest of the model is below.
    CHARGE_TICKS = 45
    COOLDOWN_TICKS = 60
    #: `+0x88`, compared `+0x88 <= +0x8C` after the counter is bumped.
    FIRE_TICKS = 80
    DECOMP_DISCHARGE = True
    #: `FUN_103FFB00` tests the beam with `FUN_1050B070`, the ELEVEN-LIMB test --
    #: not the chaingun's bare circle. §73 guessed the laser family kept the
    #: skeleton and the call site confirms it.
    BEAM_USES_LIMBS = True

    def __init__(self, entity_type, sim, xcoord, ycoord, orientation, mode):
        super().__init__(
            entity_type, sim, xcoord, ycoord, orientation, mode, self.SPEED
        )

    def on_acquire(self):
        """Freeze the beam on the target's bearing, once, at acquisition.

        `+0x90`/`+0x98` is set on the frame the drone enters state 1 and is never
        written again: across every captured discharge it holds one direction
        through the charge, the whole 80-frame beam and the cooldown, and that
        direction is the bearing to the ninja at the acquisition frame to within
        0.003-0.008 rad -- one frame of ninja motion, i.e. the tick-order offset.

        So the beam does NOT track, and it does not point along the patrol lane
        either. This class used to do the latter.
        """
        ninja = self.target
        if ninja is None:
            return
        dx = ninja.xpos - self.xpos
        dy = ninja.ypos - self.ypos
        dist = math.sqrt(dy * dy + dx * dx)
        if max(abs(dist), 1.0) * _EPS < abs(dist):
            self.fire_dx, self.fire_dy = dx / dist, dy / dist

    def _fire_setup(self, tx, ty, vx, vy):
        """`FUN_103FF770`, vtable `+0x18`. Resets `+0x8C` and starts the sound.

        It does not aim and it does not fire -- the aim was frozen at
        acquisition and the beam is tested by the discharge tick.
        """
        self.shot_tick = 0

    def _discharge_tick(self):
        """`FUN_103FFB00`, vtable `+0x1C`. A CONTINUOUS beam, not a burst.

        Every frame of the discharge it recasts along the frozen direction and
        runs the 11-limb test against every ninja, then bumps `+0x8C` and reports
        done once it reaches `+0x88`. This class tested the ninja on the single
        transition frame, so its laser was 1/80th as lethal as the game's -- and
        the direction it tested was the patrol lane, so it was rarely the right
        line either.
        """
        self.aim_dx, self.aim_dy = self.fire_dx, self.fire_dy
        self._trace_beam()
        self._beam_kill()
        self.shot_tick += 1
        return self.FIRE_TICKS <= self.shot_tick
