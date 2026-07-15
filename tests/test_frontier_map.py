"""v18 C1: spatial frontier map + go-to-frontier shaping contract tests.

Covers:
  * obs-space declaration / shape / bounds when ``expose_frontier_map`` is on,
    and its ABSENCE when off (Layer-1 contract preserved);
  * frontier semantics: frontier = reachable AND unvisited — never overlaps
    visited cells, always a subset of the reachable mask, and partitions the
    reachable set together with the visited∩reachable cells;
  * co-registration guard: the frontier channel is 0 exactly where the
    visit-count grid is > 0 (the two grids share cells by construction — the
    Y-flip / +1-border bug class would break this immediately);
  * monotone consumption: frontier cells never increase within an episode and
    reset on a fresh episode;
  * HWM approach-bonus math: re-arm on new-cell visits, pay coef*(hwm - d) on
    a new closest approach over visited ground, nothing on retreat or repeat;
  * weighted mode stays in [0.25, 1] on frontier cells and 0 elsewhere;
  * feature fully off => no obs key, no bonus, zero overhead.
"""

from __future__ import annotations

import random

import numpy as np

from nclone.gym_environment.config import EnvironmentConfig
from nclone.gym_environment.constants import (
    FRONTIER_MAP_WIDTH,
    FRONTIER_MAP_HEIGHT,
    FRONTIER_MAP_CHANNELS,
    VISITATION_MAP_WIDTH,
    VISITATION_MAP_HEIGHT,
    VISITATION_CELL_PX,
    VISITATION_ORIGIN_PX,
)
from nclone.gym_environment.generator_registry import GeneratorRegistry
from nclone.gym_environment.npp_environment import NppEnvironment


def _map(category: str = "simple", seed: int = 7):
    reg = GeneratorRegistry(random.Random(seed))
    return [int(v) for v in reg.get_map(category, seed).map_data()]


def _make_env(expose=True, coef=0.0, weighted=False):
    cfg = EnvironmentConfig.for_training(seed=0, sim_backend="python")
    cfg.reachability.expose_cost_to_go_map = True
    cfg.reachability.expose_frontier_map = expose
    cfg.reachability.frontier_map_weighted = weighted
    cfg.frontier_approach_bonus = coef
    return NppEnvironment(cfg)


def _cell_center_px(cx: int, cy: int):
    return (
        VISITATION_ORIGIN_PX + (cx + 0.5) * VISITATION_CELL_PX,
        VISITATION_ORIGIN_PX + (cy + 0.5) * VISITATION_CELL_PX,
    )


class TestFrontierMap:
    def test_grid_coregistered_constants(self):
        assert FRONTIER_MAP_WIDTH == VISITATION_MAP_WIDTH
        assert FRONTIER_MAP_HEIGHT == VISITATION_MAP_HEIGHT
        assert FRONTIER_MAP_CHANNELS == 1

    def test_obs_space_and_shape(self):
        env = _make_env(expose=True)
        try:
            assert "frontier_map" in env.observation_space.spaces
            sp = env.observation_space.spaces["frontier_map"]
            assert sp.shape == (
                FRONTIER_MAP_CHANNELS,
                FRONTIER_MAP_WIDTH,
                FRONTIER_MAP_HEIGHT,
            )
            obs, _ = env.reset(options={"map_data": _map(), "map_name": "f1"})
            fm = obs["frontier_map"]
            assert fm.shape == sp.shape
            assert fm.dtype == np.float32
            assert np.isfinite(fm).all()
            assert 0.0 <= fm.min() and fm.max() <= 1.0
            # A fresh reachability source must yield SOME frontier on reset.
            assert fm.sum() > 0
        finally:
            env.close()

    def test_absent_when_disabled(self):
        env = _make_env(expose=False, coef=0.0)
        try:
            assert "frontier_map" not in env.observation_space.spaces
            assert env._frontier_enabled is False
            assert env._visit_map_enabled is False  # no hidden overhead
            obs, _ = env.reset(options={"map_data": _map(), "map_name": "f2"})
            assert "frontier_map" not in obs
        finally:
            env.close()

    def test_frontier_semantics_and_coregistration(self):
        env = _make_env(expose=True)
        try:
            env.reset(options={"map_data": _map(), "map_name": "f3"})
            for _ in range(60):
                _, _, term, trunc, _ = env.step(int(env.action_space.sample()))
                if term or trunc:
                    break
            frontier = env._frontier_grid()
            assert frontier is not None
            reach = env._visit_reachable_mask()
            visited = env._visit_counts > 0
            # frontier ⊆ reachable, frontier ∩ visited = ∅,
            # frontier ∪ (reachable ∧ visited) = reachable.
            assert not (frontier & ~reach).any()
            assert not (frontier & visited).any()
            np.testing.assert_array_equal(
                frontier | (reach & visited), reach
            )
            # The obs channel mirrors the grid (binary mode).
            fm = env._compute_frontier_map()[0]
            np.testing.assert_array_equal(fm > 0, frontier)
            # Co-registration guard: everywhere the visit grid is > 0 the
            # frontier channel must be exactly 0.
            assert (fm[visited] == 0.0).all()
        finally:
            env.close()

    def test_frontier_monotone_and_resets(self):
        env = _make_env(expose=True)
        try:
            env.reset(options={"map_data": _map(), "map_name": "f4"})
            counts = []
            for _ in range(80):
                _, _, term, trunc, info = env.step(int(env.action_space.sample()))
                counts.append(info["frontier_cells"])
                if term or trunc:
                    break
            assert counts[0] > 0
            assert all(b <= a for a, b in zip(counts, counts[1:]))
            n_end = counts[-1]
            env.reset(options={"map_data": _map(), "map_name": "f4"})
            env.step(0)
            # Fresh episode restores the frontier (>= where the last one ended).
            fresh = env._frontier_grid().sum()
            assert fresh >= n_end
        finally:
            env.close()

    def test_hwm_bonus_math(self):
        env = _make_env(expose=True, coef=0.05)
        try:
            env.reset(options={"map_data": _map(), "map_name": "f5"})
            env._reset_visit_counts()
            # Two probe cells; register both as visited (new-cell visits re-arm
            # the HWM each time, paying nothing).
            p_near = _cell_center_px(10, 10)
            p_far = _cell_center_px(30, 10)
            for p in (p_near, p_far):
                env._update_visit_counts(*p)
                env._update_frontier_bonus(*p)
                assert env._frontier_bonus == 0.0  # new cell -> re-arm only
            d_far = env._frontier_dist
            assert d_far > 0
            # Move back to the already-visited NEAR cell: not a new cell, so if
            # it is closer to the frontier the HWM pays exactly coef*(d_far-d).
            env._update_visit_counts(*p_near)
            env._update_frontier_bonus(*p_near)
            d_near = env._frontier_dist
            if d_near < d_far:  # geometry-dependent, holds on this layout
                assert np.isclose(
                    env._frontier_bonus, 0.05 * (d_far - d_near), atol=1e-9
                )
            hwm_after = env._frontier_hwm
            # Repeat the same cell: no new minimum -> no pay, hwm unchanged.
            env._update_visit_counts(*p_near)
            env._update_frontier_bonus(*p_near)
            assert env._frontier_bonus == 0.0
            assert env._frontier_hwm == hwm_after
            # Retreat to the far cell: d rises -> no pay, hwm unchanged.
            env._update_visit_counts(*p_far)
            env._update_frontier_bonus(*p_far)
            assert env._frontier_bonus == 0.0
            assert env._frontier_hwm == hwm_after
        finally:
            env.close()

    def test_info_keys_and_zero_coef(self):
        env = _make_env(expose=True, coef=0.0)
        try:
            env.reset(options={"map_data": _map(), "map_name": "f6"})
            for _ in range(20):
                _, _, term, trunc, info = env.step(int(env.action_space.sample()))
                assert info["frontier_bonus"] == 0.0
                assert info["frontier_cells"] >= 0
                assert 0.0 <= info["frontier_frac"] <= 1.0
                if term or trunc:
                    break
        finally:
            env.close()

    def test_bonus_only_no_obs_key(self):
        env = _make_env(expose=False, coef=0.05)
        try:
            assert env._frontier_enabled is True
            assert env._visit_map_enabled is True  # grid maintained for shaping
            assert "frontier_map" not in env.observation_space.spaces
        finally:
            env.close()

    def test_weighted_mode_bounds(self):
        env = _make_env(expose=True, weighted=True)
        try:
            env.reset(options={"map_data": _map(), "map_name": "f7"})
            for _ in range(30):
                _, _, term, trunc, _ = env.step(int(env.action_space.sample()))
                if term or trunc:
                    break
            fm = env._compute_frontier_map()[0]
            frontier = env._frontier_grid()
            assert (fm[~frontier] == 0.0).all()
            if frontier.any():
                vals = fm[frontier]
                assert (vals >= 0.25 - 1e-6).all()
                assert (vals <= 1.0 + 1e-6).all()
        finally:
            env.close()
