"""Player-Controlled Rocket Entity (Type 29).

Reconstructed from `docs/ENTITY_INTEGRATION_SPEC.md` §8.9
(`think` = 0x10410870, ctor = 0x1040F560). This type is absent from every entity
table the project had before the RE pass -- it was a genuine parser gap.

**It is not map-placed, so it is deliberately NOT wired into the loader.** The
spec flagged this as an open question ([I] in §2): type 29's factory call is
`FUN_1040f560(iVar20, iVar20)`, taking no position or scale arguments unlike every
other case, which suggested it is spawned at runtime rather than placed. That is
now settled empirically -- type 29 does not occur once in the entity streams of
8000 decoded attract maps (nor do the phantom types 7 and 9), while every other
official type does. Adding it to `entity_factory` would create an entity the map
format cannot express.

It is implemented anyway, for two reasons: it completes the official type set, and
it shares its whole projectile integrator with type 18 (spec: "every helper is
shared" -- grid register, cell recompute, sweep, the three collision-resolution
variants, teardown), so it costs a driver rather than an entity.

The difference from type 18 is **steering**: this think reads the active
input/controller count and applies player input to the projectile's direction each
frame, and on termination releases the controller by writing `+0x24 = -1`.

### What is verified and what is not

Verified: the shared integrator and helper set, the state field being type 18's
`+0xB0`, the controller-index/flag pair at `+0x24`/`+0x28`, the release-on-teardown
behaviour, and the ctor constant set (which overlaps Rocket's heavily, including
the same `16/7`).

**Not** verified: the exact input-to-direction mapping ([U] in §8.9). `TURN_RATE`
below is `0.05`, a real think constant, applied as radians per frame per unit of
horizontal input -- the binding is inferred.
"""

import math

from ..entities import Entity
from .entity_projectile import ProjectileState, aim_at


class EntityPlayerControlledRocket(Entity):
    """Type 29. Runtime-spawned rocket steered by player input."""

    IDLE = 0
    LAUNCH = 1
    FLYING = 2

    #: `0x10571D90`. Radians per frame per unit of horizontal input. **[I]**.
    TURN_RATE = 0.05

    def __init__(self, entity_type, sim, xcoord, ycoord):
        super().__init__(entity_type, sim, xcoord, ycoord)
        self.is_thinkable = True
        self.is_logical_collidable = True
        #: `+0xB0` -- type 18's `+0x90` under a different name.
        self.state = self.LAUNCH
        #: `+0x24` / `+0x28`. -1 means released.
        self.controller = 0
        self.controlled = True
        self.projectile = ProjectileState()

    def think(self):
        if self.state == self.LAUNCH:
            ninja = self.sim.ninja
            tx = ninja.xpos if ninja is not None else self.xpos
            ty = ninja.ypos if ninja is not None else self.ypos
            xdir, ydir = aim_at(self.xpos, self.ypos, tx, ty)
            self.projectile.launch(self.xpos, self.ypos, xdir, ydir)
            self.state = self.FLYING
            return

        if self.state != self.FLYING:
            return

        # Steering: rotate the heading by the player's horizontal input.
        if self.controlled:
            ninja = self.sim.ninja
            hor = getattr(ninja, "hor_input", 0) if ninja is not None else 0
            if hor:
                angle = math.atan2(self.projectile.ydir, self.projectile.xdir)
                angle += hor * self.TURN_RATE
                self.projectile.xdir = math.cos(angle)
                self.projectile.ydir = math.sin(angle)

        if self.projectile.advance(self.sim):
            self._release()
            return
        # Same grid-dispatch problem as type 18 -- see `entity_rocket.py`. The
        # projectile is a field inside the launcher, so the entity never leaves
        # the launcher's cell and `logical_collision` is only reached if the
        # ninja stands beside the tube. LATENT for gamma-1 (type 29 has zero
        # instances in the phase-1 corpus), fixed here because the two share
        # every helper and a divergence between them is how the next reader gets
        # misled.
        self._projectile_kill_test()

    def _release(self):
        """`FUN_1040F7F0`: tear down and hand the controller back."""
        self.state = self.IDLE
        self.controller = -1
        self.controlled = False
        self.active = False

    def _projectile_kill_test(self):
        """Discrete circle test, projectile vs ninja; runs every frame from
        `think`. See `entity_rocket.EntityRocket._projectile_kill_test`."""
        from ..physics import overlap_circle_vs_circle
        from ..ninja import NINJA_RADIUS

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
            ninja.kill(0, 0, 0, 0, 0, cause="pc_rocket")
            self._release()

    def logical_collision(self):
        from ..physics import overlap_circle_vs_circle
        from ..ninja import NINJA_RADIUS

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
            ninja.kill(0, 0, 0, 0, 0, cause="pc_rocket")
            self._release()
