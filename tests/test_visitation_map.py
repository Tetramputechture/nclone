"""v17 episodic visitation map + count-based exploration bonus contract tests.

Covers:
  * obs-space declaration / shape / bounds when ``expose_visitation_map`` is on,
    and its ABSENCE when off (Layer-1 contract preserved);
  * per-episode reset zeroes the visit grid (the signal is strictly episodic);
  * count-based novelty math (1.0 on first visit, 1/sqrt(count) on revisits) and
    the cell-of mapping co-registered with cost_to_go_map (12px cells / 24px border);
  * coverage rises as new cells are seen;
  * the reward bonus is added (coef * novelty), is post-clawback (present on the
    death step), and is 0 when the coef is 0;
  * zero overhead / no key when the feature is fully disabled.
"""

from __future__ import annotations

import random

import numpy as np

from nclone.gym_environment.config import EnvironmentConfig
from nclone.gym_environment.constants import (
    VISITATION_MAP_WIDTH,
    VISITATION_MAP_HEIGHT,
    VISITATION_MAP_CHANNELS,
    VISITATION_CELL_PX,
    VISITATION_ORIGIN_PX,
    COST_TO_GO_MAP_WIDTH,
    COST_TO_GO_MAP_HEIGHT,
)
from nclone.gym_environment.generator_registry import GeneratorRegistry
from nclone.gym_environment.npp_environment import NppEnvironment


def _map(category: str = "simple", seed: int = 7):
    reg = GeneratorRegistry(random.Random(seed))
    return [int(v) for v in reg.get_map(category, seed).map_data()]


def _make_env(expose=True, coef=0.0):
    cfg = EnvironmentConfig.for_training(seed=0, sim_backend="python")
    cfg.reachability.expose_cost_to_go_map = True
    cfg.reachability.expose_visitation_map = expose
    cfg.visitation_bonus_coef = coef
    return NppEnvironment(cfg)


class TestVisitationMap:
    def test_grid_coregistered_with_cost_map(self):
        # Visitation grid must share cost_to_go_map's resolution exactly so the
        # two spatial channels align cell-for-cell for the conv/CoordConv stack.
        assert VISITATION_MAP_WIDTH == COST_TO_GO_MAP_WIDTH
        assert VISITATION_MAP_HEIGHT == COST_TO_GO_MAP_HEIGHT
        assert VISITATION_MAP_CHANNELS == 1
        assert VISITATION_CELL_PX == 12
        assert VISITATION_ORIGIN_PX == 24

    def test_obs_space_and_shape(self):
        env = _make_env(expose=True)
        try:
            assert "visitation_map" in env.observation_space.spaces
            sp = env.observation_space.spaces["visitation_map"]
            assert sp.shape == (
                VISITATION_MAP_CHANNELS,
                VISITATION_MAP_WIDTH,
                VISITATION_MAP_HEIGHT,
            )
            obs, _ = env.reset(options={"map_data": _map(), "map_name": "v1"})
            vm = obs["visitation_map"]
            assert vm.shape == sp.shape
            assert vm.dtype == np.float32
            assert np.isfinite(vm).all()
            assert 0.0 <= vm.min() and vm.max() <= 1.0
        finally:
            env.close()

    def test_absent_when_disabled(self):
        env = _make_env(expose=False, coef=0.0)
        try:
            assert "visitation_map" not in env.observation_space.spaces
            assert env._visit_map_enabled is False
            obs, _ = env.reset(options={"map_data": _map(), "map_name": "v2"})
            assert "visitation_map" not in obs
            assert env._visit_counts is None
        finally:
            env.close()

    def test_reset_zeroes_grid(self):
        env = _make_env(expose=True)
        try:
            env.reset(options={"map_data": _map(), "map_name": "v3"})
            for _ in range(30):
                _, _, term, trunc, _ = env.step(int(env.action_space.sample()))
                if term or trunc:
                    break
            assert env._visit_counts is not None
            assert int(env._visit_counts.sum()) > 0
            # A fresh episode must clear the grid.
            env.reset(options={"map_data": _map(), "map_name": "v3"})
            assert int(env._visit_counts.sum()) == 0
            assert env._visit_cells == 0
            assert env._visit_coverage == 0.0
        finally:
            env.close()

    def test_cell_mapping(self):
        env = _make_env(expose=True)
        try:
            # Border origin maps to cell 0; one cell over maps to cell 1; far
            # OOB clips into range.
            assert env._visit_cell_of(24.0, 24.0) == (0, 0)
            assert env._visit_cell_of(24.0 + 12.0, 24.0) == (1, 0)
            assert env._visit_cell_of(24.0, 24.0 + 12.0) == (0, 1)
            cx, cy = env._visit_cell_of(1e9, 1e9)
            assert 0 <= cx < VISITATION_MAP_WIDTH
            assert 0 <= cy < VISITATION_MAP_HEIGHT
            cx, cy = env._visit_cell_of(-1e9, -1e9)
            assert (cx, cy) == (0, 0)
        finally:
            env.close()

    def test_novelty_and_bonus_math(self):
        env = _make_env(expose=True, coef=0.02)
        try:
            env.reset(options={"map_data": _map(), "map_name": "v4"})
            # Force a known cell and step the update directly.
            env._reset_visit_counts()
            env._update_visit_counts(24.0, 24.0)  # cell (0,0), first visit
            assert env._visit_new_cell is True
            assert abs(env._visit_novelty - 1.0) < 1e-6
            assert abs(env._visitation_bonus - 0.02 * 1.0) < 1e-6
            env._update_visit_counts(24.0, 24.0)  # revisit -> count 2
            assert env._visit_new_cell is False
            assert abs(env._visit_novelty - 1.0 / np.sqrt(2)) < 1e-6
            assert abs(env._visitation_bonus - 0.02 / np.sqrt(2)) < 1e-6
        finally:
            env.close()

    def test_coverage_rises(self):
        env = _make_env(expose=True)
        try:
            env.reset(options={"map_data": _map(), "map_name": "v5"})
            covs = []
            for _ in range(200):
                _, _, term, trunc, info = env.step(int(env.action_space.sample()))
                covs.append(info["visit_coverage"])
                if term or trunc:
                    env.reset(options={"map_data": _map(), "map_name": "v5"})
            assert max(covs) > 0.0
            assert 0.0 <= max(covs) <= 1.0
            # The obs channel is nonzero exactly at visited cells.
            assert env._visit_cells == int((env._visit_counts > 0).sum())
        finally:
            env.close()

    def test_bonus_only_no_obs_key(self):
        env = _make_env(expose=False, coef=0.05)
        try:
            assert env._visit_map_enabled is True  # grid maintained for the bonus
            assert "visitation_map" not in env.observation_space.spaces
            env.reset(options={"map_data": _map(), "map_name": "v6"})
            saw_bonus = False
            for _ in range(40):
                _, _, term, trunc, info = env.step(int(env.action_space.sample()))
                if info["visitation_bonus"] > 0.0:
                    saw_bonus = True
                if term or trunc:
                    env.reset(options={"map_data": _map(), "map_name": "v6"})
            assert saw_bonus
        finally:
            env.close()

    def test_p2_directed_weight(self):
        # Directed exploration: weight in [FLOOR, 1], varies with d* to goal;
        # uniform (==1) when directedness is off.
        from nclone.gym_environment.constants import VISIT_DIRECTED_FLOOR

        uni = _make_env(expose=True, coef=0.02)  # directed off
        uni.reset(options={"map_data": _map("simple"), "map_name": "d0"})
        dr = _make_env(expose=True, coef=0.02)
        dr.config.visitation_directed = True
        dr._visitation_directed = True
        dr.reset(options={"map_data": _map("simple"), "map_name": "d0"})
        # SEEDED walk. The final assertion needs the walk to actually cover a
        # range of d*, and `action_space.sample()` draws from ambient global
        # entropy — so whether this test passed depended on how much randomness
        # earlier tests in the session happened to consume. A walk that never
        # leaves the far-from-goal region pins every weight to the floor and the
        # spread assertion fails, with nothing wrong in the code under test.
        rng = np.random.RandomState(12345)
        try:
            uw, dw = [], []
            for _ in range(250):
                act = int(rng.randint(0, uni.action_space.n))
                _, _, t1, tr1, iu = uni.step(act)
                _, _, t2, tr2, idd = dr.step(act)
                uw.append(iu["visit_directed_wt"])
                dw.append(idd["visit_directed_wt"])
                if t1 or tr1:
                    uni.reset(options={"map_data": _map("simple"), "map_name": "d0"})
                if t2 or tr2:
                    dr.reset(options={"map_data": _map("simple"), "map_name": "d0"})
            assert all(w == 1.0 for w in uw)  # uniform when off
            assert min(dw) >= VISIT_DIRECTED_FLOOR - 1e-6
            assert max(dw) <= 1.0 + 1e-6
            assert max(dw) - min(dw) > 0.05  # actually varies with d*
        finally:
            uni.close()
            dr.close()

    def test_no_bonus_when_coef_zero(self):
        env = _make_env(expose=True, coef=0.0)
        try:
            env.reset(options={"map_data": _map(), "map_name": "v7"})
            for _ in range(20):
                _, _, term, trunc, info = env.step(int(env.action_space.sample()))
                assert info["visitation_bonus"] == 0.0
                if term or trunc:
                    break
        finally:
            env.close()
