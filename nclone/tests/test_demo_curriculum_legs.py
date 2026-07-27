"""v25: demo reverse-curriculum leg routing and its own difficulty knob.

The demo curricula are built from verified winning planner routes on the OFFICIAL
maps and reach the env via ``set_start_state_curricula``. Two defects made them
inert for the whole v24 scratch run:

  * they read the PRE-switch difficulty knob, which the easy procgen start-state
    buckets (winning ~0.92 of their seeded episodes) ratcheted to 1.0 = "start at
    spawn" — silently switching the official curriculum off; and
  * only the switch (approach) leg was ever loaded, so the switch->exit RETURN
    leg was never seeded, despite the builder having written it — leaving the
    measured exit_given_switch weakness (0.65) unaddressed.

These guard the env half of the fix: a per-leg mapping, an independent demo
difficulty knob, and a distinct ``seeded_leg`` tag so the callback can route the
anneal signal.
"""

from __future__ import annotations

import random
import types

from nclone.gym_environment.npp_environment import NppEnvironment


class _FakeCurriculum:
    """Records the difficulty it was sampled at."""

    def __init__(self, tag):
        self.tag = tag
        self.sampled_at = []

    def sample(self, difficulty):
        self.sampled_at.append(float(difficulty))
        # An empty action list makes _maybe_apply_start_state bail right after
        # the leg/difficulty decision, which is all this test cares about.
        return {"start_actions": []}


def _env(ssc_map, *, seed=0):
    env = types.SimpleNamespace(
        rng=random.Random(seed),
        _ssc=None,
        _ssc_map=ssc_map,
        map_loader=types.SimpleNamespace(current_map_name="mapA"),
        nplay_headless=types.SimpleNamespace(current_map_data=None),
        config=types.SimpleNamespace(
            start_state_curriculum=False,
            post_switch_curriculum=False,
            start_state_natural_fraction=0.0,
        ),
    )
    for name in (
        "_maybe_apply_start_state",
        "set_start_state_difficulty",
        "set_start_state_difficulty_pre",
        "set_start_state_difficulty_post",
        "set_start_state_difficulty_demo",
        "_pre_switch_difficulty",
        "_post_switch_difficulty",
        "_demo_curriculum_difficulty",
    ):
        setattr(env, name, types.MethodType(getattr(NppEnvironment, name), env))
    return env


def test_demo_curriculum_uses_its_own_difficulty_knob():
    cur = _FakeCurriculum("switch")
    env = _env({"mapA": {"switch": cur}})
    env.set_start_state_difficulty_pre(1.0)   # procgen ratcheted itself to spawn
    env.set_start_state_difficulty_demo(0.2)  # demo leg is still early
    env._maybe_apply_start_state(None)
    assert cur.sampled_at == [0.2], (
        "the demo curriculum must follow the demo knob, not the procgen one "
        "that easy buckets drive to 1.0"
    )
    assert env._episode_seeded_leg == "demo_pre"


def test_demo_difficulty_falls_back_to_pre_for_older_callbacks():
    cur = _FakeCurriculum("switch")
    env = _env({"mapA": {"switch": cur}})
    env.set_start_state_difficulty_pre(0.4)  # no demo setter called at all
    env._maybe_apply_start_state(None)
    assert cur.sampled_at == [0.4]


def test_both_legs_are_reachable_and_tagged_distinctly():
    switch_cur = _FakeCurriculum("switch")
    exit_cur = _FakeCurriculum("exit")
    entry = {"switch": switch_cur, "exit": exit_cur}
    seen = set()
    # Different seeds exercise the per-reset alternation.
    for seed in range(20):
        env = _env({"mapA": entry}, seed=seed)
        env.set_start_state_difficulty_demo(0.3)
        env._maybe_apply_start_state(None)
        seen.add(env._episode_seeded_leg)
    assert seen == {"demo_pre", "demo_post"}, (
        f"both legs must be seeded across resets; saw {seen}"
    )
    assert switch_cur.sampled_at and exit_cur.sampled_at


def test_single_leg_entry_still_works():
    exit_cur = _FakeCurriculum("exit")
    env = _env({"mapA": {"exit": exit_cur}})
    env.set_start_state_difficulty_demo(0.5)
    env._maybe_apply_start_state(None)
    assert exit_cur.sampled_at == [0.5]
    assert env._episode_seeded_leg == "demo_post"


def test_legacy_flat_mapping_is_still_accepted():
    """Pre-v25 callers pass {key: curriculum} with no leg dict."""
    cur = _FakeCurriculum("flat")
    env = _env({"mapA": cur})
    env.set_start_state_difficulty_demo(0.25)
    env._maybe_apply_start_state(None)
    assert cur.sampled_at == [0.25]
    assert env._episode_seeded_leg == "demo_pre"


def test_stored_single_level_curriculum_still_uses_the_pre_knob():
    """The single-level (file mode) path must be unchanged."""
    cur = _FakeCurriculum("single")
    env = _env({})
    env._ssc = cur
    env.set_start_state_difficulty_pre(0.7)
    env.set_start_state_difficulty_demo(0.1)
    env._maybe_apply_start_state(None)
    assert cur.sampled_at == [0.7], (
        "a stored single-level curriculum is not a demo curriculum"
    )
    assert env._episode_seeded_leg == "pre"


def test_no_matching_level_seeds_nothing():
    cur = _FakeCurriculum("switch")
    env = _env({"other_map": {"switch": cur}})
    env._maybe_apply_start_state(None)
    assert cur.sampled_at == []
    assert env._episode_seeded_leg is None


def test_empty_leg_entry_is_ignored():
    env = _env({"mapA": {}})
    env._maybe_apply_start_state(None)
    assert env._episode_seeded_leg is None
