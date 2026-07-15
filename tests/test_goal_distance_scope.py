"""Bucket-scoping of the goal-distance reverse curriculum.

The goal-distance reverse curriculum (``set_goal_distance_fraction``: succeed at an
intermediate cost-to-go threshold, annealed outward) is a global env knob. When
``EnvironmentConfig.goal_distance_curriculum_buckets`` is non-empty it is applied
ONLY on levels whose reset ``bucket`` is in the set; every other level is forced to
1.0 (off, true-goal-only). This lets the p^6 -> p^1 reverse-spawn be scoped to
PLATFORM_STAIRS without shortening episodes on already-winnable buckets.

Exercises ``BaseNppEnvironment._apply_goal_distance_scope`` /
``set_goal_distance_fraction`` in isolation (no full env instantiation) via a
lightweight stand-in carrying only the attributes the methods touch.
See docs/PLATFORM_STAIRS_CURRICULUM_FOLLOWUP.md §4.2/§6.4.
"""

from types import SimpleNamespace

from nclone.gym_environment.base_environment import BaseNppEnvironment


def _fake_env(scope=()):
    env = SimpleNamespace()
    env.config = SimpleNamespace(goal_distance_curriculum_buckets=tuple(scope))
    env.reward_calculator = SimpleNamespace(
        config=SimpleNamespace(goal_distance_fraction=1.0)
    )
    # Bind the real methods onto the stand-in.
    env._apply_goal_distance_scope = (
        BaseNppEnvironment._apply_goal_distance_scope.__get__(env)
    )
    env.set_goal_distance_fraction = (
        BaseNppEnvironment.set_goal_distance_fraction.__get__(env)
    )
    return env


def _frac(env):
    return env.reward_calculator.config.goal_distance_fraction


def test_no_scope_applies_globally():
    """Empty scope = legacy behavior: the target fraction hits every level."""
    env = _fake_env(scope=())
    env.set_goal_distance_fraction(0.4)
    env._current_bucket = "FLAT_RUN"
    env._apply_goal_distance_scope()
    assert _frac(env) == 0.4
    env._current_bucket = "PLATFORM_STAIRS"
    env._apply_goal_distance_scope()
    assert _frac(env) == 0.4


def test_scoped_bucket_gets_the_fraction():
    env = _fake_env(scope=("PLATFORM_STAIRS",))
    env.set_goal_distance_fraction(0.3)
    env._current_bucket = "PLATFORM_STAIRS"
    env._apply_goal_distance_scope()
    assert _frac(env) == 0.3


def test_out_of_scope_bucket_forced_off():
    env = _fake_env(scope=("PLATFORM_STAIRS",))
    env.set_goal_distance_fraction(0.3)
    env._current_bucket = "FLAT_RUN"
    env._apply_goal_distance_scope()
    assert _frac(env) == 1.0  # off (true goal only) on non-scoped buckets


def test_reapplied_per_reset_as_bucket_changes():
    """The effective fraction is re-resolved each reset as the level changes."""
    env = _fake_env(scope=("PLATFORM_STAIRS",))
    env.set_goal_distance_fraction(0.25)
    # reset onto a stairs level -> active
    env._current_bucket = "PLATFORM_STAIRS"
    env._apply_goal_distance_scope()
    assert _frac(env) == 0.25
    # next reset onto a mine level -> forced off
    env._current_bucket = "MINE_NAV"
    env._apply_goal_distance_scope()
    assert _frac(env) == 1.0
    # back to stairs -> active again (target remembered)
    env._current_bucket = "PLATFORM_STAIRS"
    env._apply_goal_distance_scope()
    assert _frac(env) == 0.25


def test_unknown_bucket_none_forced_off_when_scoped():
    env = _fake_env(scope=("PLATFORM_STAIRS",))
    env.set_goal_distance_fraction(0.5)
    env._current_bucket = None  # no bucket in reset options
    env._apply_goal_distance_scope()
    assert _frac(env) == 1.0
