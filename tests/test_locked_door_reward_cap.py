"""Completion must dominate the locked-door milestone, on every level.

The failure this pins is not hypothetical. Measured over all 808 official
locked-door levels: 294 require at least one door, and while the median is 2 the
tail runs to 71. At a flat R_LOCKED_DOOR_SWITCH = 0.5 that is 35.5 for opening
doors against R_EXIT = 10.0 for finishing -- and locked doors never re-close, so
the agent can bank the whole payout and then decline the exit leg's death risk.
Nine levels pay more for doors than for winning; 24 pay more than half.

So the milestone is capped as a TOTAL and spread over the level's necessary
doors. These tests fix both halves: the common case must be untouched, and the
tail must never outpay completion.
"""

import pytest

from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.reward_calculation.reward_constants import (
    R_EXIT,
    R_LOCKED_DOOR_SWITCH,
    R_LOCKED_DOOR_TOTAL_CAP,
)


def _calc(**kw):
    return RewardCalculator(reward_config=RewardConfig(
        enable_locked_door_rewards=True, **kw))


def _obs(n_open, n_total):
    return {"necessary_doors_open": n_open, "necessary_doors_total": n_total}


def _open_all(calc, n_total):
    """Total milestone paid for opening every necessary door, one at a time."""
    return sum(
        calc._compute_locked_door_reward(_obs(i, n_total))[0]
        for i in range(n_total + 1)
    )


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_common_case_pays_the_full_per_door_bonus(k):
    """225 of the 294 door-needing official levels need <= 4 doors.

    The cap exists for the tail; it must not quietly re-tune the ordinary case.
    """
    assert _open_all(_calc(), k) == pytest.approx(k * R_LOCKED_DOOR_SWITCH)


@pytest.mark.parametrize("k", [5, 16, 21, 42, 71])
def test_tail_is_capped_and_never_outpays_completion(k):
    """``071`` here is `059 scratchware manifesto`, the corpus worst case."""
    total = _open_all(_calc(), k)
    assert total <= R_LOCKED_DOOR_TOTAL_CAP + 1e-9
    assert total < R_EXIT, (
        "opening every door pays at least as much as winning -- and doors are "
        "irreversible, so timing out after them dominates completing"
    )


def test_cap_is_spread_evenly_not_truncated():
    """Every door pays the same, so no door is the one that 'stops counting'.

    Truncating instead (full bonus until the cap, then zero) would make the last
    doors of a chain worth nothing -- which is precisely the doors furthest from
    the spawn and hardest to reach.
    """
    calc = _calc()
    k = 20
    # The first call only latches the baseline, so start from zero doors open.
    per = [calc._compute_locked_door_reward(_obs(i, k))[0] for i in range(k + 1)][1:]
    assert len(set(round(p, 12) for p in per)) == 1
    assert sum(per) == pytest.approx(R_LOCKED_DOOR_TOTAL_CAP)


def test_missing_total_degrades_to_the_uncapped_bonus():
    """An env that does not publish the total must still pay something.

    Silently paying zero would look identical to "the flag is off", which is the
    exact confusion that let a dead reward survive two whole runs.
    """
    calc = _calc()
    calc._compute_locked_door_reward({"necessary_doors_open": 0})
    r, _ = calc._compute_locked_door_reward({"necessary_doors_open": 1})
    assert r == pytest.approx(R_LOCKED_DOOR_SWITCH)


def test_cap_can_be_disabled_for_an_ab():
    calc = _calc(_locked_door_total_cap_override=0.0)
    assert _open_all(calc, 71) == pytest.approx(71 * R_LOCKED_DOOR_SWITCH)


def test_disabled_flag_still_pays_nothing():
    calc = RewardCalculator(reward_config=RewardConfig())
    assert _open_all(calc, 4) == 0.0
