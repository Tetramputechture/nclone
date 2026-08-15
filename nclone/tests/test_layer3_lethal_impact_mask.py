"""v26 W6b: the Layer-3 macro mask must veto fatal FALLS, not only mines.

Before v26 every hard safety guarantee the agent had was mine-only: the Layer-3
forward simulator asked "does this (action, duration) hit a deadly mine" and
``--takeoff-arc-gap-mask`` asked "does this jump land on a mine". Mine deaths are
~60% of v25's losses and impact deaths ~40%, so the interrupt/mask layer was
shortening macros near mines while leaving fatal-gap macros completely uncapped.

These tests pin: (1) the mine-only default is byte-compatible, (2) the
per-action rollout is equivalent to the old per-bin loop, (3) a committed fatal
fall is vetoed when impact masking is on, (4) the lookahead catches a death that
happens AFTER the macro's own frames (the actual commitment case), (5) the mask
never traps the agent.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

from nclone.gym_environment.layer3_macro_action_mask import Layer3MacroActionMasker


BINS = (1, 2, 4, 8, 16, 32, 64)
_NUM_ACTIONS = 6


class _FakeNativeSim:
    """Native-simulator stand-in with a scripted death frame.

    ``death_frame`` is 1-indexed over ticks since the last injection; ``None``
    means the ninja survives the whole horizon.
    """

    def __init__(self, death_frame=None, mine_frame=None):
        self.death_frame = death_frame
        self.mine_frame = mine_frame
        self.t = 0
        self.ticks = 0
        self.neutral_ticks = 0

    # -- injection surface used by the masker -------------------------------
    def get_ninja_jump_duration(self):
        return 0

    def fast_reset(self):
        self.t = 0

    def inject_ninja_for_value_iteration(self, x, y, vx, vy, jd):
        self.t = 0

    def tick_physics_only(self, hor, jump):
        self.t += 1
        self.ticks += 1
        if hor == 0 and jump == 0:
            self.neutral_ticks += 1

    def has_died(self):
        return self.death_frame is not None and self.t >= self.death_frame

    def get_ninja_pos(self):
        # Park the ninja on a mine exactly at mine_frame and after.
        if self.mine_frame is not None and self.t >= self.mine_frame:
            return (100.0, 100.0)
        return (0.0, 0.0)

    # -- snapshot surface used to branch each duration bin off the trunk -----
    def save_full_state(self):
        return [self.t]

    def restore_full_state(self, blob):
        self.t = int(blob[0])
        return True


class _FakeHeadless:
    def __init__(self, sim, mines=()):
        self._native_sim = sim
        self._mines = list(mines)
        self.current_map_data = [1, 2, 3]

    def get_mine_entities(self):
        class _M:
            def __init__(self, x, y):
                self.xpos, self.ypos, self.state = x, y, 0

        return ([_M(x, y) for x, y in self._mines], [])

    def ninja_position(self):
        return (0.0, 0.0)

    def ninja_velocity(self):
        return (0.0, 0.0)


def _masker(**kw):
    m = Layer3MacroActionMasker(duration_bins=BINS, **kw)
    return m


def _compute(masker, sim, mines=(), **kw):
    headless = _FakeHeadless(sim, mines)
    masker._native_sim = sim
    masker._native_map_data = list(headless.current_map_data)
    return masker.compute(
        headless=headless,
        base_action_mask=np.ones(_NUM_ACTIONS, dtype=np.bool_),
        actions_to_execute=lambda a: (1, 0),
        **kw,
    )


def test_mine_only_default_ignores_a_fatal_fall():
    """Regression guard: the pre-v26 contract must be unchanged when the new
    flag is off, even on a trajectory that certainly kills."""
    res = _compute(_masker(), _FakeNativeSim(death_frame=3))
    assert res.duration_mask.all(), "mine-only masking must not veto an impact"
    assert res.masked_by_impact == 0


def test_no_mines_and_no_impact_check_short_circuits():
    sim = _FakeNativeSim(death_frame=3)
    res = _compute(_masker(), sim)
    assert res.duration_mask.all()
    assert sim.ticks == 0, "the no-hazard early-out must not simulate at all"


def test_lethal_impact_vetoes_only_bins_long_enough_to_reach_it():
    """Both predicates are prefix-monotone in duration, so a death at frame 5
    condemns exactly the bins >= 5 (16, 32, 64) and spares 1, 2, 4."""
    res = _compute(
        _masker(include_lethal_impact=True), _FakeNativeSim(death_frame=5)
    )
    for a in range(_NUM_ACTIONS):
        got = res.duration_mask[a].astype(bool)
        expected = np.array([b < 5 for b in BINS])
        assert (got == expected).all(), (a, got, expected)
    assert res.masked_by_impact > 0
    assert res.masked_by_mine == 0


def test_matches_the_legacy_per_bin_loop_for_mines():
    """The per-action rollout replaced a per-(action, duration) loop; on the mine
    predicate it must produce an identical mask."""
    mine_frame = 6
    res = _compute(
        _masker(include_lethal_impact=True),
        _FakeNativeSim(mine_frame=mine_frame),
        mines=[(100.0, 100.0)],
    )
    expected = np.array([b < mine_frame for b in BINS])
    for a in range(_NUM_ACTIONS):
        assert (res.duration_mask[a].astype(bool) == expected).all()
    assert res.masked_by_mine > 0
    assert res.masked_by_impact == 0


def test_lookahead_catches_a_death_after_the_macro_ends():
    """The commitment case: no bin is long enough to reach the death, so without
    a lookahead every candidate reads safe — which is exactly how a jump that
    steps off a fatal ledge slipped through."""
    # max bin is 64, lookahead 40; this sim dies on its 80th simulated frame
    # regardless of input, so a bin is fatal exactly when duration + 40 >= 80.
    sim = _FakeNativeSim(death_frame=80)
    res = _compute(
        _masker(include_lethal_impact=True, lethal_impact_lookahead=40), sim
    )
    expected = np.array([b + 40 < 80 for b in BINS])
    for a in range(_NUM_ACTIONS):
        assert (res.duration_mask[a].astype(bool) == expected).all(), (
            a, res.duration_mask[a], expected
        )
    assert res.masked_by_impact > 0
    assert sim.neutral_ticks > 0, "the lookahead must run NEUTRAL input"

    # Same trajectory, no lookahead -> invisible.
    res_none = _compute(
        _masker(include_lethal_impact=True), _FakeNativeSim(death_frame=80)
    )
    assert res_none.duration_mask.all()


def test_lookahead_death_does_not_veto_bins_that_release_in_time():
    """Pipeline audit P0.2, false-masking direction.

    The old scheme simulated ONE rollout per action -- input held to the longest
    bin, then neutral -- and vetoed EVERY bin when that rollout died in the
    post-maximum window. A short bin releases its input much earlier and reaches
    the fatal frame much later (or not at all), so it must be judged on its own
    continuation.
    """
    sim = _FakeNativeSim(death_frame=70)
    res = _compute(
        _masker(include_lethal_impact=True, lethal_impact_lookahead=20), sim
    )
    surviving = res.duration_mask[0].astype(bool)
    # duration + 20 < 70 survives: bins 1..32 live, 64 dies.
    assert surviving.any(), "short bins must not be condemned by a long bin's fate"
    assert not res.all_masked_event
    assert (surviving == np.array([b + 20 < 70 for b in BINS])).all()


def test_fall_risk_gate_restores_the_early_out():
    sim = _FakeNativeSim(death_frame=3)
    res = _compute(
        _masker(include_lethal_impact=True), sim, fall_risk_nearby=False
    )
    assert res.duration_mask.all()
    assert sim.ticks == 0, "the caller's gate must skip simulation entirely"


def test_never_leaves_the_agent_with_no_action():
    """Every action fatal at frame 1 -> the noop fallback must keep one open."""
    res = _compute(
        _masker(include_lethal_impact=True), _FakeNativeSim(death_frame=1)
    )
    assert res.all_masked_event
    assert res.action_mask.any(), "the agent must never be fully trapped"
    assert bool(res.duration_mask[0, 0])


def test_base_mask_actions_are_never_simulated():
    sim = _FakeNativeSim(death_frame=5)
    masker = _masker(include_lethal_impact=True)
    headless = _FakeHeadless(sim)
    masker._native_sim = sim
    masker._native_map_data = list(headless.current_map_data)
    base = np.zeros(_NUM_ACTIONS, dtype=np.bool_)
    base[2] = True
    res = masker.compute(
        headless=headless,
        base_action_mask=base,
        actions_to_execute=lambda a: (1, 0),
    )
    assert not res.action_mask[1]
    # Exactly one action rolled out, and it stopped at the death frame rather
    # than running the horizon out.
    assert sim.ticks == 5


@pytest.mark.parametrize("lookahead", [0, 20])
def test_survivor_keeps_every_duration(lookahead):
    res = _compute(
        _masker(include_lethal_impact=True, lethal_impact_lookahead=lookahead),
        _FakeNativeSim(),
    )
    assert res.duration_mask.all()
    assert res.masked_by_impact == 0 and res.masked_by_mine == 0


class _ReleaseSensitiveSim:
    """A sim whose fate depends on WHEN the input is released.

    Models the real asymmetry the shared-rollout scheme was blind to: committing
    to a jump and holding it clears the gap, but letting go early drops the ninja
    into it. Death happens on the first NEUTRAL frame if fewer than ``min_hold``
    held frames preceded it.
    """

    def __init__(self, min_hold: int):
        self.min_hold = min_hold
        self.held = 0
        self.dead = False

    def get_ninja_jump_duration(self):
        return 0

    def fast_reset(self):
        self.held = 0
        self.dead = False

    def inject_ninja_for_value_iteration(self, x, y, vx, vy, jd):
        self.fast_reset()

    def tick_physics_only(self, hor, jump):
        if hor == 0 and jump == 0:
            if self.held < self.min_hold:
                self.dead = True
        else:
            self.held += 1

    def has_died(self):
        return self.dead

    def get_ninja_pos(self):
        return (0.0, 0.0)

    def save_full_state(self):
        return [self.held, int(self.dead)]

    def restore_full_state(self, blob):
        self.held, self.dead = int(blob[0]), bool(blob[1])
        return True


def test_short_bins_are_not_falsely_certified_by_a_long_holds_survival():
    """Pipeline audit P0.2, false-certification direction.

    The old scheme evaluated the post-macro window ONCE, after holding the input
    to the longest bin. A short bin's own release was therefore never simulated,
    so bins that let go too early and fall were certified safe by a rollout that
    never let go early at all.
    """
    min_hold = 10
    res = _compute(
        _masker(include_lethal_impact=True, lethal_impact_lookahead=20),
        _ReleaseSensitiveSim(min_hold=min_hold),
    )
    expected = np.array([b >= min_hold for b in BINS])
    for a in range(_NUM_ACTIONS):
        assert (res.duration_mask[a].astype(bool) == expected).all(), (
            a, res.duration_mask[a], expected
        )
    # Bins 1, 2, 4 and 8 release before the commitment is complete and die.
    assert res.masked_by_impact > 0
    assert not res.all_masked_event, "the long bins are genuinely safe"


def test_fallback_fatal_flags_a_knowingly_lethal_noop():
    """When everything is masked the NOOP fail-open fires. If that NOOP was
    itself evaluated as fatal, say so rather than presenting it as safe."""
    res = _compute(
        _masker(include_lethal_impact=True), _FakeNativeSim(death_frame=1)
    )
    assert res.all_masked_event
    assert res.fallback_fatal, (
        "the forced NOOP dies at frame 1 here; the result must not imply it is safe"
    )

    # A survivor never trips the flag.
    ok = _compute(_masker(include_lethal_impact=True), _FakeNativeSim())
    assert not ok.all_masked_event and not ok.fallback_fatal
