"""Regression guard: the RFCL reverse curriculum never produces a start state
from which the objective is unreachable (i.e. it never moves the spawn outside
the reachable area).

The bug: ``StartStateCurriculum.generate`` explores mid-route states with a
physics RRT, then tagged EVERY node by ``progress = cost_to_go`` while clamping
an INFINITE cost-to-go (goal unreachable from that node -- a dead-end pit/ledge)
to the max finite value. So goal-unreachable states were kept and, being tagged
at ~spawn difficulty, got sampled as episode start states -- moving the ninja
outside the reachable region. At runtime that yields an unsolvable episode, an
``inf`` free-space PBRS / goal-distance signal, and a flood of
"position far outside graph bounds" warnings.

The fix discards any state with infinite cost-to-go at generation time (both the
switch-goal and post-switch exit-goal curricula), and an apply-time guard reverts
to spawn if a replayed prefix still drifts outside the reachable region.

This test asserts the core invariant directly on real dataset levels.
"""
import glob
import math
import pickle

import pytest

pytest.importorskip("nclone_native")

from nclone.gym_environment.start_state_curriculum import StartStateCurriculum, ACT
from nclone.value_function.free_space_distance import FreeSpaceDistanceSource

# Buckets whose procgen most readily produces dead-end pockets the RRT can reach.
_LEVEL_GLOBS = [
    "datasets/mechanic_benchmark_v4/train/MINE_NAVIGATION/T3/*.pkl",
    "datasets/mechanic_benchmark_v4/train/CHAMBER_MAZE/T2/*.pkl",
    "datasets/mechanic_benchmark_v4/train/PLATFORMING/T2/*.pkl",
]


def _first_level():
    for g in _LEVEL_GLOBS:
        hits = glob.glob(g)
        if hits:
            return pickle.load(open(hits[0], "rb"))["map_data"]
    pytest.skip("no benchmark dataset available locally")


def _endpoint(sim, actions):
    sim.fast_reset()
    for a in actions:
        h, j = ACT.get(int(a), (0, 0))
        sim.tick_physics_mines_only(h, j)
        if sim.has_died():
            break
    return sim.get_ninja_pos()


def test_switch_curriculum_states_all_reachable():
    """Every generated switch-goal start state can still reach the switch."""
    import logging
    logging.disable(logging.CRITICAL)
    import nclone_native

    md = _first_level()
    cur = StartStateCurriculum.generate(
        md, goal_kind="switch", n_states=300, max_iter=40000, seed=1
    )
    assert len(cur) > 0, "curriculum was over-filtered to empty"

    fs = FreeSpaceDistanceSource(list(md))
    assert fs.available, "free-space source unavailable for this level"
    sim = nclone_native.Simulator.load(list(md))

    inf = 0
    for actions, _progress in cur.states:
        x, y = _endpoint(sim, actions)
        if not math.isfinite(fs.cost_to_go(x, y, False)):
            inf += 1
    assert inf == 0, f"{inf}/{len(cur)} start states are goal-unreachable"


def test_curriculum_progress_tags_are_finite():
    """No stored progress tag is inf/NaN (the clamp-to-_d_max path is gone)."""
    md = _first_level()
    cur = StartStateCurriculum.generate(
        md, goal_kind="switch", n_states=200, max_iter=30000, seed=7
    )
    assert all(math.isfinite(p) for _a, p in cur.states)
    assert cur.max_progress > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
