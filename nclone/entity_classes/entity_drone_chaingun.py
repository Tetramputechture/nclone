"""Chaingun Drone Entity (Type 12).

Reconstructed from `docs/ENTITY_INTEGRATION_SPEC.md` §8.3. Everything
behavioural lives in `EntityDroneShooterBase`, shared with the laser drone --
per the spec, the two types differ *only* in their patrol speed and in the five
per-type hooks.

Verified from the binary (spec §8.3):
  * ctor 0x103FE460 sets the patrol speed to **4/7** (`0x10572390`) and a `3.0`,
    and writes `+0x2C` and `+0x39`;
  * `move` is `EntityDroneShooterBase`'s own `0x10400580`, which this type
    inherits unchanged;
  * `think` is the shared `0x104005D0`;
  * logical collision is the shared `return 0` default -- the body is harmless.

Not verified: what `3.0` is for. It is the same value as `0x10572398`, which §7
also lists as the EyeBat speed clamp, so it is a shared global rather than a
chaingun-specific constant. Burst size and bullet speed are both plausible and
neither is confirmed, so it is not used here.
"""

from ..constants.physics_constants import DRONE_RADIUS
import os

from .entity_drone_shooter_base import EntityDroneShooterBase


class EntityDroneChaingun(EntityDroneShooterBase):
    """Type 12. Slow patrol, charges and fires a hitscan burst."""

    MAX_COUNT_PER_LEVEL = 256
    #: The sweep is measured for this type; see the base class.
    SWEEP_ENABLED = True
    #: All five of this type's hooks are decompiled (`HOOK_t12_chaingun_*`), so
    #: `NCLONE_AB_DRONE_DECOMP=1` applies to it and not to the laser drone.
    DECOMP_DISCHARGE = True
    RADIUS = DRONE_RADIUS

    #: `0x10572390`, the n/7 speed family (spec §6.4).
    SPEED = 4 / 7

    # The chaingun's 53 / 80 / 90 live on the base class; see the note in
    # `entity_drone_laser.py` for why they are type-specific after all.

    def __init__(self, entity_type, sim, xcoord, ycoord, orientation, mode):
        super().__init__(
            entity_type, sim, xcoord, ycoord, orientation, mode, self.SPEED
        )

    def fire(self):
        """Trace the shot and test the ninja. `[I]` -- the hook body is unread.

        `think` calls this ONCE, on the 1 -> 2 transition; the docstring here used
        to claim a burst sprays over the whole discharge, but nothing ever called
        it again, so the claim was describing behaviour the class did not have.
        """
        self._trace_beam()
        if os.environ.get("NCLONE_AB_CHAINGUN_HARMLESS") != "1":
            self._beam_kill()

    def cooldown_tick(self):
        """No-op; the chaingun's barrel spin is presentation state."""
