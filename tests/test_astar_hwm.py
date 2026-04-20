"""NPP-18 T1: A* HWM monotonicity + mine-toggle recompute (toggle-mines chunk).

This test file consolidates the two T1 assertions called out by the Layer 1
local-validation ticket (NPP-18):

1. **HWM monotonicity** — the High-Water-Mark shaping reward emitted by
   :class:`nclone.gym_environment.reward_calculation.hwm_shaping.EpisodeRewardState`
   is non-negative on every step and the cumulative normalised reward per
   phase is bounded by 1. This guarantees that the shaping term is a valid
   potential-style signal that can only improve over the course of an
   episode (pre- and post-switch phases are independent).

2. **Mine-toggle recompute** — toggling a toggle-mine's state invalidates the
   shared A* distance-map cache (NPP-44) for the affected level so the next
   query recomputes the map. Also confirmed at the
   :class:`nclone.graph.reachability.path_distance_calculator.CachedPathDistanceCalculator`
   level: the level cache rebuilds whenever any toggle-mine state changes.

These two behaviours together satisfy the "A* HWM monotonicity + mine-toggle
recompute" deliverable in the NPP-18 acceptance criteria and are run as part
of the `tests/` pytest suite on CPU.
"""

from __future__ import annotations

import math

import numpy as np

from nclone.constants.entity_types import EntityType
from nclone.graph.level_data import LevelData
from nclone.graph.reachability.path_distance_calculator import (
    CachedPathDistanceCalculator,
)
from nclone.gym_environment.reward_calculation.hwm_shaping import EpisodeRewardState

from npp_rl.training.astar_cache import (
    AStarDistanceMapCache,
    PHASE_EXIT,
    PHASE_SWITCH,
)


# ---------------------------------------------------------------------------
# T1.a — HWM monotonicity on a two-phase trajectory.
# ---------------------------------------------------------------------------


def test_hwm_monotonic_non_negative_bounded_per_phase():
    """Per-step HWM progress is non-negative; cumulative <= 1 per phase.

    Mirrors the invariant audited in ``docs/LAYER1_MARKOV_AUDIT.md`` for the
    toggle-mines reward model: switch-phase shaping is a function of
    ``dist_switch`` only, exit-phase shaping is a function of ``dist_exit``
    only, and the transition step emits 0.
    """
    state = EpisodeRewardState()
    state.reset(
        initial_dist_switch=100.0,
        max_dist_switch=100.0,
        initial_dist_exit=80.0,
        max_dist_exit=80.0,
    )

    switch_trace = [100.0, 90.0, 95.0, 70.0, 70.0, 50.0, 30.0]
    switch_total = 0.0
    for d in switch_trace:
        p = state.step(dist_switch=d, dist_exit=999.0, switch_activated=False)
        assert p >= 0.0, f"HWM emitted a negative shaping reward {p} for d={d}"
        assert not math.isnan(p), "HWM emitted NaN"
        switch_total += p
    assert switch_total <= 1.0 + 1e-6, (
        f"Switch-phase cumulative normalized reward {switch_total} exceeds 1.0"
    )

    transition = state.step(dist_switch=28.0, dist_exit=60.0, switch_activated=True)
    assert transition == 0.0, "Transition step must emit 0 (phase swap)"

    exit_trace = [60.0, 55.0, 55.0, 20.0, 5.0]
    exit_total = 0.0
    for d in exit_trace:
        p = state.step(dist_switch=0.0, dist_exit=d, switch_activated=True)
        assert p >= 0.0
        assert not math.isnan(p)
        exit_total += p
    assert exit_total <= 1.0 + 1e-6, (
        f"Exit-phase cumulative normalized reward {exit_total} exceeds 1.0"
    )


def test_hwm_no_reward_for_regression():
    """Back-tracking strictly further than the HWM must not emit reward."""
    state = EpisodeRewardState()
    state.reset(
        initial_dist_switch=50.0,
        max_dist_switch=50.0,
        initial_dist_exit=50.0,
        max_dist_exit=50.0,
    )

    first = state.step(dist_switch=40.0, dist_exit=999.0, switch_activated=False)
    assert first > 0.0

    regression = state.step(dist_switch=45.0, dist_exit=999.0, switch_activated=False)
    assert regression == 0.0, "HWM shaping must not reward moving further from target"


# ---------------------------------------------------------------------------
# T1.b — Mine-toggle recompute (shared A* cache and per-level path cache).
# ---------------------------------------------------------------------------


def test_astar_cache_invalidation_on_mine_state_change():
    """NPP-44 cache: a mine toggle forces a recompute of the A* distance map.

    We model a stateful "distance map loader" whose returned array embeds the
    deadly-mine count. The caller invalidates the cache whenever a toggle
    mine flips, matching the callsite pattern used by ``LevelPool`` and the
    environment reset path.
    """
    cache = AStarDistanceMapCache()
    call_count = {"n": 0}
    mine_state = {"deadly": 0}

    def loader(level_id: int, phase: str) -> np.ndarray:
        call_count["n"] += 1
        return np.full((3, 3), mine_state["deadly"], dtype=np.float32)

    level_id = 7
    first = cache.get_or_compute(level_id, PHASE_SWITCH, loader)
    assert call_count["n"] == 1
    assert np.allclose(first, 0.0)

    second = cache.get_or_compute(level_id, PHASE_SWITCH, loader)
    assert call_count["n"] == 1, "Second query must be a cache hit"
    assert second is first, "Cache hit should return the same ndarray object"

    # Toggle a mine: flip deadly count and invalidate both phases.
    mine_state["deadly"] = 1
    removed = cache.invalidate(level_id)
    assert removed >= 1, "Invalidate should drop at least the switch-phase entry"

    third = cache.get_or_compute(level_id, PHASE_SWITCH, loader)
    assert call_count["n"] == 2, "Mine toggle must force a fresh compute"
    assert np.allclose(third, 1.0)

    # Exit-phase is also recomputed independently (it was never cached here,
    # but the same level_id still routes through the loader exactly once).
    cache.get_or_compute(level_id, PHASE_EXIT, loader)
    assert call_count["n"] == 3


def test_cached_path_distance_calculator_rebuilds_on_mine_flip():
    """CachedPathDistanceCalculator rebuilds whenever any toggle-mine changes.

    This exercises the in-tree path-distance cache consumed by the curriculum
    reachability pipeline; a flip in any toggle-mine state must force the
    next ``build_level_cache`` call to recompute rather than reuse the prior
    graph.
    """
    adjacency = {
        (0, 0): [((12, 0), 12.0)],
        (12, 0): [((24, 0), 12.0)],
        (24, 0): [],
    }
    mine = {
        "type": EntityType.TOGGLE_MINE,
        "x": 18,
        "y": 0,
        "state": 1,  # toggled
    }
    level = LevelData(
        start_position=(24, 24),
        tiles=np.zeros((23, 42), dtype=np.int32),
        entities=[mine],
        switch_states={},
        level_id="npp_18_t1_mine_flip",
    )

    calc = CachedPathDistanceCalculator(max_cache_size=32)
    rebuilt_first = calc.build_level_cache(level, adjacency, adjacency, None)
    assert rebuilt_first is True, "Initial build should populate the cache"

    mine["state"] = 0
    rebuilt_after_toggle = calc.build_level_cache(level, adjacency, adjacency, None)
    assert rebuilt_after_toggle is True, "Mine flip must force a rebuild"

    # No state change: the third call should short-circuit on cache hit.
    mine["state"] = 0
    rebuilt_unchanged = calc.build_level_cache(level, adjacency, adjacency, None)
    assert rebuilt_unchanged is False, "Unchanged state must not rebuild the cache"
