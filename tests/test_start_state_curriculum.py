"""Tests for the start-state curriculum (faithful reachable mid-route starts).

Validates the general mechanism: a reverse-curriculum reachable-state generator
(faithful via tick_physics_mines_only) producing replayable action prefixes, the
difficulty->progress mapping, save/load, and the env reset hook that replays a
prefix on the env's own sim so the episode begins mid-route.
"""

from __future__ import annotations

import numpy as np
import pytest

from nclone.map_generation.map import Map
from nclone.gym_environment.start_state_curriculum import StartStateCurriculum


def _corridor_md() -> list:
    m = Map()
    for x in range(2, 41):
        m.set_tile(x, 12, 1)
    m.set_ninja_spawn(4, 11)
    m.add_entity(3, 38, 11, 0, 0, switch_x=36, switch_y=11)  # exit + switch (grid 36, far right)
    return [int(v) for v in m.map_data()]


@pytest.fixture(scope="module")
def corridor_curriculum():
    return StartStateCurriculum.generate(_corridor_md(), n_states=200, max_iter=40000, seed=0)


def test_generate_produces_progress_spread(corridor_curriculum):
    cur = corridor_curriculum
    assert len(cur) > 20
    progs = [p for _, p in cur.states]
    assert progs == sorted(progs)            # sorted ascending
    assert cur.max_progress > 0
    assert min(progs) < cur.max_progress      # a real spread, not all-equal


def test_difficulty_maps_to_progress(corridor_curriculum):
    cur = corridor_curriculum
    rng = __import__("random").Random(1)
    easy = np.mean([cur.sample(0.1, rng)["progress"] for _ in range(40)])
    hard = np.mean([cur.sample(0.9, rng)["progress"] for _ in range(40)])
    # low difficulty -> nearer the goal (lower cost-to-go) than high difficulty
    assert easy < hard


def test_save_load_roundtrip(corridor_curriculum, tmp_path):
    p = tmp_path / "cur.json"
    corridor_curriculum.save(str(p))
    loaded = StartStateCurriculum.load(str(p))
    assert len(loaded) == len(corridor_curriculum)
    assert loaded.max_progress == corridor_curriculum.max_progress
    assert loaded.sample(0.5).get("start_actions") is not None


def test_empty_curriculum_safe():
    cur = StartStateCurriculum()
    assert cur.sample(0.5) == {}


@pytest.fixture(scope="module")
def corridor_post_switch():
    return StartStateCurriculum.generate_post_switch(
        _corridor_md(), n_states=120, max_iter=40000, seed=0
    )


def test_post_switch_states_activate_the_switch(corridor_post_switch):
    """Every post-switch start state, replayed on FULL physics, must consume the
    exit switch — that is the whole point (seed the return-to-exit leg)."""
    import nclone_native

    cur = corridor_post_switch
    assert len(cur) > 0
    assert cur.goal_kind == "exit"
    md = _corridor_md()
    sim = nclone_native.Simulator.load(list(md))
    act = {0: (0, 0), 1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (1, 1), 5: (-1, 1)}
    rng = __import__("random").Random(0)
    consumed = 0
    trials = 12
    for _ in range(trials):
        s = cur.sample(1.0, rng)  # difficulty 1.0 == full return leg from switch
        sim.fast_reset()
        for a in s["start_actions"]:
            h, j = act[int(a)]
            sim.tick(h, j)
            if sim.has_died() or sim.has_won():
                break
        # exit switch (entity_type 4) is ``active`` until touched.
        if any(e.entity_type == 4 and not e.active for e in sim.get_state().entities):
            consumed += 1
    assert consumed == trials, f"only {consumed}/{trials} states activated the switch"


def test_post_switch_lazy_cache_roundtrips(tmp_path, monkeypatch):
    """The lazy per-level cache builds once and reloads from disk."""
    monkeypatch.setenv("NCLONE_POST_SWITCH_CACHE_DIR", str(tmp_path))
    import importlib
    from nclone.gym_environment import post_switch_curriculum_cache as psc

    importlib.reload(psc)
    md = _corridor_md()
    a = psc.get_or_build(md, n_states=80, max_iter=30000)
    assert len(a) > 0 and a.goal_kind == "exit"
    # A json cache file is written; a second call returns an equal-length pool.
    assert any(p.suffix == ".json" for p in tmp_path.iterdir())
    psc._MEM_CACHE.clear()  # force the disk path
    b = psc.get_or_build(md, n_states=80, max_iter=30000)
    assert len(b) == len(a)


@pytest.mark.slow
def test_env_reset_hook_starts_midroute(corridor_curriculum):
    from nclone.gym_environment.npp_environment import NppEnvironment
    from nclone.gym_environment.config import EnvironmentConfig
    md = _corridor_md()
    # defer_initial_map_load: the first level is supplied via reset options
    # (payload-driven, like benchmark training) — avoids the default-category
    # generator path (and its registry flakiness).
    env = NppEnvironment(config=EnvironmentConfig(defer_initial_map_load=True))
    env.reset(options={"map_data": md})
    spawn_x = env.nplay_headless.ninja_position()[0]
    env.reset(options={"map_data": md, "start_state_curriculum": corridor_curriculum,
                       "start_state_difficulty": 0.1})
    near_x = env.nplay_headless.ninja_position()[0]
    # difficulty 0.1 should start the ninja well past spawn, toward the goal.
    assert near_x > spawn_x + 100
