"""Validation of the game-faithful gold time bonus (``gold_time_bonus_frames``).

In real N++ each gold adds a second to the countdown. The sim's truncation limit
is computed ONCE per level (surface area + reachable mines + route cost) and was
static thereafter, so a gold detour cost frames AND brought the timeout forward —
measured on the v24 scratch run as timeouts nearly doubling (29 -> 49) as
gold/solve rose.

Guards:
  1. Off by default (0) — the limit is exactly the static one, and no gold scan
     is performed at all.
  2. Each collected gold extends the effective limit by exactly N frames.
  3. The bonus is added AFTER the curriculum multiplier, so a gold is worth the
     same 1 second in every curriculum phase.
  4. The OBSERVED limit (which drives ``time_remaining``) matches the limit that
     actually truncates — otherwise the policy's clock disagrees with the env's.
  5. An episode that would have timed out survives once enough gold is banked.
"""

from __future__ import annotations

import types

from nclone.gym_environment.base_environment import BaseNppEnvironment
from nclone.gym_environment.truncation_checker import TruncationChecker

BASE_LIMIT = 1000
PER_GOLD = 60


class _FakeSim:
    def __init__(self, frame: int):
        self.frame = frame


class _FakeHeadless:
    """Minimal stand-in exposing only what the truncation path touches."""

    def __init__(self, frame: int, gold_state):
        self.sim = _FakeSim(frame)
        self._gold_state = gold_state
        self.gold_scans = 0

    def get_gold_state(self):
        self.gold_scans += 1
        return list(self._gold_state)


def _env(*, frame: int, n_gold: int, n_collected: int, per_gold: int):
    """A bare object wearing the base env's truncation methods.

    Constructing a real environment would load a map and a simulator; the
    truncation logic under test only reads ``config``, ``nplay_headless`` and
    ``reward_calculator.config``, so bind the unbound methods instead.
    """
    gold_state = [
        (0.0, 0.0, i >= n_collected)  # active=False for the first n_collected
        for i in range(n_gold)
    ]
    env = types.SimpleNamespace(
        config=types.SimpleNamespace(gold_time_bonus_frames=per_gold),
        nplay_headless=_FakeHeadless(frame, gold_state),
        reward_calculator=types.SimpleNamespace(
            config=types.SimpleNamespace(training_phase="late")
        ),
        _checkpoint_replay_frame_count=0,
    )
    checker = TruncationChecker(env)
    checker.current_truncation_limit = BASE_LIMIT
    env.truncation_checker = checker
    env._gold_time_bonus_frames = types.MethodType(
        BaseNppEnvironment._gold_time_bonus_frames, env
    )
    env._get_curriculum_aware_truncation_limit = types.MethodType(
        BaseNppEnvironment._get_curriculum_aware_truncation_limit, env
    )
    env._check_curriculum_aware_truncation = types.MethodType(
        BaseNppEnvironment._check_curriculum_aware_truncation, env
    )
    return env


def test_disabled_by_default_and_does_no_gold_scan():
    env = _env(frame=0, n_gold=5, n_collected=3, per_gold=0)
    assert env._gold_time_bonus_frames() == 0
    assert env._get_curriculum_aware_truncation_limit() == BASE_LIMIT
    assert env.nplay_headless.gold_scans == 0, (
        "the disabled path must not pay for a gold scan"
    )


def test_bonus_is_per_collected_gold():
    for collected in (0, 1, 3, 5):
        env = _env(frame=0, n_gold=5, n_collected=collected, per_gold=PER_GOLD)
        assert env._gold_time_bonus_frames() == PER_GOLD * collected
        assert (
            env._get_curriculum_aware_truncation_limit()
            == BASE_LIMIT + PER_GOLD * collected
        )


def test_uncollected_gold_grants_nothing():
    env = _env(frame=0, n_gold=9, n_collected=0, per_gold=PER_GOLD)
    assert env._gold_time_bonus_frames() == 0


def test_observed_limit_matches_the_truncating_limit():
    """time_remaining is derived from the observed limit; they must agree."""
    collected = 4
    limit = BASE_LIMIT + PER_GOLD * collected
    # One frame before the (extended) limit: alive, and the observed limit agrees.
    env = _env(frame=limit - 1, n_gold=5, n_collected=collected, per_gold=PER_GOLD)
    assert env._get_curriculum_aware_truncation_limit() == limit
    assert env._check_curriculum_aware_truncation() is False
    # Exactly at the extended limit: truncates.
    env = _env(frame=limit, n_gold=5, n_collected=collected, per_gold=PER_GOLD)
    assert env._check_curriculum_aware_truncation() is True


def test_gold_rescues_an_episode_that_would_have_timed_out():
    frame = BASE_LIMIT + 10  # past the static budget
    without = _env(frame=frame, n_gold=5, n_collected=0, per_gold=PER_GOLD)
    assert without._check_curriculum_aware_truncation() is True
    with_gold = _env(frame=frame, n_gold=5, n_collected=1, per_gold=PER_GOLD)
    assert with_gold._check_curriculum_aware_truncation() is False, (
        "one gold (=60 frames) must buy past a 10-frame overrun"
    )


def test_bonus_is_not_scaled_by_the_curriculum_multiplier():
    """A gold is worth 1 second in EVERY phase, not multiplier x 1 second."""
    env = _env(frame=0, n_gold=5, n_collected=2, per_gold=PER_GOLD)
    expected = BASE_LIMIT + 2 * PER_GOLD  # multiplier is 1.0 in every phase today
    assert env._get_curriculum_aware_truncation_limit() == expected
    # Directly exercise the checker's contract with a non-unit multiplier.
    assert (
        env.truncation_checker.update_and_check_for_truncation(2.0, 2 * PER_GOLD)
        is False
    )
    env.nplay_headless.sim.frame = 2 * BASE_LIMIT + 2 * PER_GOLD
    assert (
        env.truncation_checker.update_and_check_for_truncation(2.0, 2 * PER_GOLD)
        is True
    )
