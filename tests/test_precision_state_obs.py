"""Precision-state observation block + the jump-useless single source of truth.

These lock in an obs revision aimed at FRAME-PRECISE maneuvers (the corner
walljump that separates a ~485-frame route from the ~465 optimum). Each assertion
below corresponds to a hole that was verified present in the shipped observation:

* ``wall_normal`` (game_state idx 16) is gated on ``walled``, so during the
  wall_buffer COYOTE window -- the frames when a walljump is still legal after
  leaving the wall -- the policy saw "walljump available" with no direction.
* The state one-hot merges Jumping(3) with Falling(4), and ``jump_duration`` is
  masked to 0 outside state 3 (which is itself unobservable), so "still rising,
  can extend" is indistinguishable from "falling, jump over".
* ``jump_is_useless`` existed in THREE disagreeing copies (action mask,
  Ninja.is_jump_useless, game_state feature); the observation could contradict
  the mask that governs the agent's own actions.
* The launch-pad buffer was normalized with the 5-frame window of the other
  buffers though its own window is 4, wasting a quantization step.
"""

import numpy as np
import pytest

from nclone.ninja import has_active_jump_buffer, jump_is_useless


class _FakeNinja:
    """Ninja-like stand-in. Field names match BOTH Ninja and native _NinjaProxy."""

    def __init__(self, **kw):
        self.airborn = False
        self.state = 1
        self.jump_input = 0
        self.jump_buffer = -1
        self.floor_buffer = -1
        self.wall_buffer = -1
        self.launch_pad_buffer = -1
        self.__dict__.update(kw)


# ---------------------------------------------------------------------------
# Single source of truth for "a jump here does nothing"
# ---------------------------------------------------------------------------


def test_grounded_jump_is_never_useless():
    assert jump_is_useless(_FakeNinja(airborn=False, state=1)) is False


def test_already_jumping_is_never_useless():
    assert jump_is_useless(_FakeNinja(airborn=True, state=3)) is False


def test_airborne_not_holding_jump_is_useful_because_it_arms_a_buffer():
    n = _FakeNinja(airborn=True, state=4, jump_input=0)
    assert jump_is_useless(n) is False


def test_airborne_holding_jump_with_no_buffer_is_useless():
    """No rising edge and nothing armed -> pressing jump truly does nothing."""
    n = _FakeNinja(airborn=True, state=4, jump_input=1)
    assert jump_is_useless(n) is True


def test_airborne_holding_jump_is_rescued_by_any_live_buffer():
    for buf in ("jump_buffer", "floor_buffer", "wall_buffer"):
        n = _FakeNinja(airborn=True, state=4, jump_input=1, **{buf: 0})
        assert jump_is_useless(n) is False, buf
    # launch_pad has a SHORTER window (4, not 5)
    n = _FakeNinja(airborn=True, state=4, jump_input=1, launch_pad_buffer=3)
    assert jump_is_useless(n) is False


def test_launch_pad_buffer_window_is_four_not_five():
    """Its window is ``-1 < b < 4``; 4 must read as expired, unlike the others."""
    assert has_active_jump_buffer(_FakeNinja(launch_pad_buffer=3)) is True
    assert has_active_jump_buffer(_FakeNinja(launch_pad_buffer=4)) is False
    # The 5-frame buffers still accept 4.
    assert has_active_jump_buffer(_FakeNinja(jump_buffer=4)) is True


def test_mask_and_observation_cannot_disagree():
    """The action mask and the obs feature must be the SAME predicate.

    Previously the obs added a ``jump_input_old != 0`` term the mask lacked, so
    the policy could perceive a jump as usable on frames the mask had forbidden.
    """
    from nclone.ninja import Ninja

    # Both the Ninja method and the module function must resolve identically.
    n = _FakeNinja(airborn=True, state=4, jump_input=1)
    assert jump_is_useless(n) is True
    assert Ninja.jump_is_useless_now(n) == jump_is_useless(n)


# ---------------------------------------------------------------------------
# The precision-state block
# ---------------------------------------------------------------------------

WALL_NORMAL_BUFFERED = 0
IS_JUMPING = 1
JUMP_DURATION = 2
AIRBORN_OLD = 3
FLOOR_DWELL = 4
WALL_DWELL = 5
STATE_RECENCY = 6
STATE_NORM = 7
PREV_STATE = 8
LAST_JUMP_BUFFERED = 9


class _RichNinja(_FakeNinja):
    def __init__(self, **kw):
        super().__init__()
        self.walled = False
        self.wall_normal = 0.0
        self.jump_duration = 0
        self.airborn_old = False
        self.consecutive_floor_frames = 0
        self.consecutive_wall_frames = 0
        self.state_change_frame = 0
        self.previous_state = 0
        self.last_jump_was_buffered = False
        self.__dict__.update(kw)


def _block(**kw):
    from nclone.gym_environment.npp_environment import NppEnvironment

    return NppEnvironment._compute_precision_state(NppEnvironment, _RichNinja(**kw))


def test_block_has_the_declared_width_and_is_bounded():
    from nclone.gym_environment.npp_environment import NppEnvironment

    v = _block()
    assert len(v) == NppEnvironment.PRECISION_STATE_DIM == 10
    assert np.all(v >= -1.0) and np.all(v <= 1.0)


def test_wall_normal_survives_the_coyote_window():
    """THE fix: direction must remain visible after leaving the wall.

    The shipped game_state idx 16 goes to 0 the instant ``walled`` clears, so the
    policy knew a walljump was still legal (wall_buffer) but not which way it
    would launch.
    """
    on_wall = _block(walled=True, wall_normal=1.0, wall_buffer=0)
    assert on_wall[WALL_NORMAL_BUFFERED] == pytest.approx(1.0)

    # Left the wall, buffer still live -> direction MUST persist.
    coyote = _block(walled=False, wall_normal=1.0, wall_buffer=2)
    assert coyote[WALL_NORMAL_BUFFERED] == pytest.approx(1.0)

    # Buffer consumed/expired -> no walljump available, so no direction.
    expired = _block(walled=False, wall_normal=1.0, wall_buffer=-1)
    assert expired[WALL_NORMAL_BUFFERED] == pytest.approx(0.0)


def test_jumping_is_distinguishable_from_falling():
    """The shipped state one-hot merges these two."""
    assert _block(state=3)[IS_JUMPING] == pytest.approx(1.0)
    assert _block(state=4)[IS_JUMPING] == pytest.approx(-1.0)
    # ...and full-resolution state separates them too.
    assert _block(state=3)[STATE_NORM] != _block(state=4)[STATE_NORM]


def test_jump_duration_is_not_masked_outside_state_three():
    """Shipped idx 10 forces 0 unless state==3, making 0 ambiguous."""
    rising = _block(state=3, jump_duration=10)
    falling = _block(state=4, jump_duration=10)
    # Same duration reads the same regardless of state — no masking.
    assert rising[JUMP_DURATION] == pytest.approx(falling[JUMP_DURATION])
    # And it actually varies with duration.
    assert _block(state=3, jump_duration=0)[JUMP_DURATION] < rising[JUMP_DURATION]


def test_frame_stacking_only_quantities_are_present():
    """airborn_old / dwell counters / previous_state were absent entirely."""
    v = _block(airborn_old=True, consecutive_floor_frames=15,
               consecutive_wall_frames=30, state_change_frame=15,
               previous_state=5, last_jump_was_buffered=True)
    assert v[AIRBORN_OLD] == pytest.approx(1.0)
    assert v[FLOOR_DWELL] == pytest.approx(0.0)      # 15/30 -> midpoint
    assert v[WALL_DWELL] == pytest.approx(1.0)       # 30/30 -> capped
    assert v[STATE_RECENCY] == pytest.approx(0.0)
    assert v[PREV_STATE] == pytest.approx((5 / 9.0) * 2 - 1)
    assert v[LAST_JUMP_BUFFERED] == pytest.approx(1.0)


def test_dwell_counters_saturate_rather_than_overflow():
    v = _block(consecutive_floor_frames=10_000)
    assert v[FLOOR_DWELL] == pytest.approx(1.0)
