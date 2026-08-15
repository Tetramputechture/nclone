"""PBRS severance must be observable, because it is otherwise silent.

`_potential` clamps a non-finite cost-to-completion to -1.0. That bound is
deliberate -- an infinite potential would make shaping explode -- but -1.0 is
ALSO `Phi(spawn)`, so a route the graph cannot cost yields a potential that is
flat at the worst value for as long as it stays severed. Zero dense gradient,
and numerically indistinguishable from standing on the spawn.

That is exactly what an unmodelled traversal entity produces: not a dropped
level, which a level count would reveal, but a dead shaping signal on precisely
the levels that need it most. These tests pin the counters that make it visible,
and in particular the distinction that decides who is at fault:

* severed from the FIRST evaluation -> the route was never costable. An
  unmodelled entity, or a genuinely unsolvable level.
* severed only later -> the agent walked off the costed region. Normal, and it
  recovers on its own.
"""

from __future__ import annotations

import math

import pytest

from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig


def _calc():
    return RewardCalculator(RewardConfig())


def test_clean_episode_reports_no_severance():
    calc = _calc()
    calc._pbrs_d0 = 100.0
    for cost in (100.0, 80.0, 40.0, 0.0):
        calc._potential(cost)
    m = calc.get_episode_reward_metrics()
    assert m["pbrs_severed_fraction"] == 0.0
    assert m["pbrs_severed_from_spawn"] == 0.0


def test_severed_from_spawn_is_flagged():
    """The unmodelled-entity signature: the very first potential is uncostable."""
    calc = _calc()
    calc._pbrs_d0 = 100.0
    calc._potential(float("inf"))
    calc._potential(float("inf"))
    m = calc.get_episode_reward_metrics()
    assert m["pbrs_severed_from_spawn"] == 1.0
    assert m["pbrs_severed_fraction"] == 1.0


def test_severance_appearing_later_is_not_blamed_on_the_level():
    """Walking off the costed region must NOT look like an unmodelled entity."""
    calc = _calc()
    calc._pbrs_d0 = 100.0
    calc._potential(90.0)          # costable at spawn
    calc._potential(float("inf"))  # wandered off
    calc._potential(70.0)          # came back
    m = calc.get_episode_reward_metrics()
    assert m["pbrs_severed_from_spawn"] == 0.0
    assert m["pbrs_severed_fraction"] == pytest.approx(1 / 3)


def test_missing_d0_counts_as_severance():
    """No D0 means no potential basis at all -- the same dead gradient, and it
    must not be silently reported as a healthy episode."""
    calc = _calc()
    calc._pbrs_d0 = None
    calc._potential(50.0)
    m = calc.get_episode_reward_metrics()
    assert m["pbrs_severed_from_spawn"] == 1.0
    assert m["pbrs_severed_fraction"] == 1.0


def test_severed_potential_is_still_the_spawn_value():
    """The property that makes this invisible without a counter, pinned so the
    reasoning above stays true if the clamp is ever changed."""
    calc = _calc()
    calc._pbrs_d0 = 100.0
    assert calc._potential(float("inf")) == -1.0
    assert calc._potential(100.0) == -1.0  # Phi(spawn), same number


def test_counters_reset_between_episodes():
    calc = _calc()
    calc._pbrs_d0 = 100.0
    calc._potential(float("inf"))
    assert calc.get_episode_reward_metrics()["pbrs_severed_fraction"] == 1.0
    calc.reset()
    calc._pbrs_d0 = 100.0
    calc._potential(50.0)
    m = calc.get_episode_reward_metrics()
    assert m["pbrs_severed_fraction"] == 0.0
    assert m["pbrs_severed_from_spawn"] == 0.0
