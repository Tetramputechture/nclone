"""Evil Ninja Entity (Type 22).

Reconstructed from `docs/ENTITY_INTEGRATION_SPEC.md` §8.10
(`think` = 0x10405AD0, ctor = 0x10401FB0).

**The evil ninja is not an AI. It is a position-tape replayer.** It has no ninja
state machine, no physics and no steering -- it replays *recorded ninja positions*
out of a ring buffer that the Simulator fills every frame. The original RE guide's
reading ("mirrors the player; probably reuses the whole ninja state machine; large
but mostly reuse") was wrong in mechanism and wrong in cost: the entity is
trivial, and the work is the Simulator-side recorder in `nsim.py`
(`Simulator.ninja_tape`), which nothing else in the port needs and which is
invisible from the entity side.

The recorder is 8 ninjas x 18000 records; 18000 frames is 5 minutes at 60Hz.
Only the first 16 bytes of each 0x78-byte record -- the position -- are read by
this entity; everything else in the snapshot is ignored.

### Why the cursor starts at 0 and the wait is 120 frames

The tape cursor trails the recorder's write index, and `think` derives a render
fade from `lag / 120.0`. Starting the entity in WAITING with a 120-frame timer and
a cursor of 0 makes the lag be exactly 120 on the first playback frame -- so the
evil ninja retraces your path from two seconds ago, and the fade divisor is
exactly the activation delay. That self-consistency is the argument for this
reading; the actual trigger (`+0x50`, "recording source") is unread.

### What is verified and what is not

Verified: the replayer mechanism, the ring buffer geometry and wrap rule, the four
states, the 30/120 frame timers, the 120.0 fade divisor, the radius going from
10.0 dormant to 6.0 on activation, and that only the position is copied.

**Not** verified: what moves the entity out of state 0, and `FUN_10405A80(bool)`
(called with "all ninjas dead" on tape exhaustion and with 0 after the post
delay) -- inferred to be an outcome/score hook with no simulation effect. State 3
has no outgoing transition in the decompiled body, so the re-arm back to WAITING
below is an inference; the alternative reading is that the entity parks at spawn
forever once it catches up.
"""

import math
import os

from ..entities import Entity
from ..physics import overlap_circle_vs_circle
from ..ninja import NINJA_RADIUS


class EntityEvilNinja(Entity):
    """Type 22. Retraces the ninja's own path from `WAIT_FRAMES` ago."""

    DORMANT = 0
    WAITING = 1
    PLAYBACK = 2
    POST = 3

    MAX_COUNT_PER_LEVEL = 128

    #: `+0x40`. `0x10571D08` while dormant, `0x10570620` once active.
    RADIUS_DORMANT = 10.0
    RADIUS_ACTIVE = 6.0
    #: The two frame timers the binary uses, and the fade divisor.
    WAIT_FRAMES = 120
    POST_FRAMES = 30
    FADE_DIVISOR = 120.0

    def __init__(self, entity_type, sim, xcoord, ycoord):
        super().__init__(entity_type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_thinkable = True
        self.is_logical_collidable = True
        #: `+0x20`. Spawn position, the restore target on reset and on POST.
        self.spawn_xpos = self.xpos
        self.spawn_ypos = self.ypos
        #: `+0x48`. **DORMANT, not WAITING.**
        #
        # Read straight off the captures: all **151** evil ninjas across both
        # capture batches carry `+0x48 = 0` and `+0x4C = 120` on the first frame
        # they appear, without exception. This port started them in WAITING, so
        # every one of them armed itself on frame 120 whether the game's did or
        # not -- on `10138634` the game's three sit at (204,192), (156,192) and
        # (180,192) for the whole demo while this port teleported all three onto
        # the ninja's own path at frame 120 and killed it at 129.
        #
        # The trigger out of DORMANT is still unread (`+0x50`, "recording
        # source"). The game does fire it -- 23 of the 151 eventually move, and
        # the state-1 timer is seen counting 119, 118, 117 -- so this is not the
        # whole entity. But DORMANT is unambiguously where they start, and an
        # evil ninja that never arms cannot kill a ninja the game let live.
        #
        # `NCLONE_AB_EVIL_NINJA_ARMED=1` restores the old unconditional arming.
        self.state = (
            self.WAITING
            if os.environ.get("NCLONE_AB_EVIL_NINJA_ARMED") == "1"
            else self.DORMANT
        )
        #: `+0x4C`.
        self.timer = self.WAIT_FRAMES
        #: `+0x58`. Tape cursor; wraps with the recorder.
        self.cursor = 0
        #: `+0x74`.
        self.post_timer = 0
        #: `+0x40`.
        self.radius = self.RADIUS_DORMANT
        #: `+0x64`. Render-only fade factor.
        self.fade = 1.0
        self._initial_cell = self.cell

    def think(self):
        if self.state == self.DORMANT:
            return

        tape = getattr(self.sim, "ninja_tape", None)
        if tape is None:
            # No recorder on this map: nothing to replay, so full reset (the
            # binary's "buffer gone" branch).
            self._reset_to_spawn()
            return

        if self.timer > 0:
            self.timer -= 1

        if self.state == self.WAITING:
            if self.timer > 0:
                return
            self.radius = self.RADIUS_ACTIVE
            self.state = self.PLAYBACK
            # falls through to PLAYBACK on this same frame, as the binary does

        if self.state == self.PLAYBACK:
            write_index = self.sim.ninja_tape_index
            if self.cursor != write_index:
                lag = write_index - self.cursor
                if lag < self.FADE_DIVISOR:
                    self.fade = max(0.0, min(1.0, lag / self.FADE_DIVISOR))
                self.xpos, self.ypos = tape[self.cursor]
                self.cursor = 0 if self.cursor + 1 > len(tape) - 1 else self.cursor + 1
                self.grid_move()
                return
            # Tape exhausted: the replayer has caught up with the live ninja.
            self.state = self.POST
            self.xpos, self.ypos = self.spawn_xpos, self.spawn_ypos
            self.post_timer = self.POST_FRAMES
            self.timer = self.WAIT_FRAMES
            self.fade = 1.0
            self.grid_move()
            return

        if self.state == self.POST:
            self.post_timer -= 1
            if self.post_timer > 0:
                return
            self.post_timer = 0
            # Re-arm. See the module docstring: state 3 has no read transition.
            self.timer = self.POST_FRAMES
            self.state = self.WAITING
            self.radius = self.RADIUS_DORMANT
            self.fade = 1.0

    def logical_collision(self):
        """Lethal on contact, at whichever radius the current state carries."""
        if self.state != self.PLAYBACK:
            return
        ninja = self.sim.ninja
        if ninja is None or not ninja.is_valid_target():
            return
        if overlap_circle_vs_circle(
            self.xpos, self.ypos, self.radius, ninja.xpos, ninja.ypos, NINJA_RADIUS
        ):
            ninja.kill(0, 0, 0, 0, 0, cause="evil_ninja")

    def _reset_to_spawn(self):
        self.radius = self.RADIUS_DORMANT
        self.state = self.DORMANT
        self.xpos, self.ypos = self.spawn_xpos, self.spawn_ypos
        self.cursor = 0
        self.timer = 0
        self.post_timer = 0

    def reset_state(self):
        """Reset for fast environment reset (same level, no reallocation)."""
        self.xpos, self.ypos = self.spawn_xpos, self.spawn_ypos
        self.cell = self._initial_cell
        self.state = self.WAITING
        self.timer = self.WAIT_FRAMES
        self.cursor = 0
        self.post_timer = 0
        self.radius = self.RADIUS_DORMANT
        self.fade = 1.0
        self.active = True
