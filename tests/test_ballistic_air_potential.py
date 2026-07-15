"""Audit II P0-2: ballistic airborne potential + F4 routing fix.

Covers:
- mine-HARD jump field (jump5): deadly kill-radius cells are excluded from
  support/wall and block arcs + the fall projection;
- F4: with use_jump_potential, the jump-aware source takes precedence over
  MineThreadingDistanceSource on mined levels (it was silently unused there);
- ballistic_air_potential: airborne PBRS distances are evaluated at the
  no-input ghost-arc landing; grounded frames are unchanged; never crashes
  when the projector is unavailable.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from nclone.gym_environment.config import EnvironmentConfig
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.gym_environment.generator_registry import GeneratorRegistry


def _map_with_mines(seed: int = 11) -> list[int]:
    reg = GeneratorRegistry(random.Random(seed))
    return [int(v) for v in reg.get_map("simplest_with_mines", seed).map_data()]


class TestMineHardJumpField:
    def test_support_excluded_inside_kill_radius(self):
        from nclone.value_function.jump_distance import JumpDistanceSource

        md = _map_with_mines()
        src = JumpDistanceSource(md, name="minehard_test")
        assert src.available
        mine_block = getattr(src, "_mine_block", None)
        support = getattr(src, "_support", None)
        if mine_block is None or support is None:
            pytest.skip("cache-restored source (build attrs absent); rerun clean")
        if len(getattr(src, "_deadly", [])) == 0:
            pytest.skip("generated level had no deadly mines")
        assert mine_block.any(), "deadly mines present but no blocked cells"
        assert not (support & mine_block).any(), (
            "support cells inside a deadly-mine kill radius"
        )

    def test_mine_free_level_unaffected(self):
        from nclone.value_function.jump_distance import JumpDistanceSource

        reg = GeneratorRegistry(random.Random(3))
        md = [int(v) for v in reg.get_map("simple", 3).map_data()]
        src = JumpDistanceSource(md, name="minefree_test")
        assert src.available
        mb = getattr(src, "_mine_block", None)
        if mb is not None:
            assert not mb.any()


class TestF4RoutingAndBallistic:
    def _env(self, ballistic: bool) -> NppEnvironment:
        cfg = EnvironmentConfig.for_training(seed=0, sim_backend="python")
        cfg.reachability.use_jump_potential = True
        cfg.reachability.ballistic_air_potential = ballistic
        return NppEnvironment(cfg)

    def test_jump_source_takes_precedence_on_mined_level(self):
        from nclone.value_function.jump_distance import JumpDistanceSource

        md = _map_with_mines()
        env = self._env(ballistic=False)
        try:
            env.reset(options={"map_data": md, "map_name": "f4_routing"})
            env.step(0)
            # F4: the committed potential source on a MINED level must be the
            # jump-aware source (previously mine-threading pre-empted it).
            assert env._potential_substituted
            fs = getattr(env, "_freespace_source", None)
            assert isinstance(fs, JumpDistanceSource), type(fs)
            assert getattr(env, "_use_freespace_this_episode", False)
        finally:
            env.close()

    def test_ballistic_eval_finite_and_stable(self):
        md = _map_with_mines()
        env = self._env(ballistic=True)
        try:
            env.reset(options={"map_data": md, "map_name": "ballistic_smoke"})
            saw_air = False
            for i in range(60):
                _obs, _r, term, trunc, info = env.step(3 if i < 20 else 0)
                assert np.isfinite(env._cached_switch_distance)
                assert np.isfinite(env._cached_exit_distance)
                if info.get("airborne"):
                    saw_air = True
                if term or trunc:
                    break
            assert saw_air, "test never went airborne; jump action mapping changed?"
        finally:
            env.close()

    def test_grounded_matches_non_ballistic(self):
        """On grounded frames the ballistic flag must not change the cached
        distances (grounded states degenerate to today's potential exactly)."""
        md = _map_with_mines()
        env_a = self._env(ballistic=False)
        env_b = self._env(ballistic=True)
        try:
            env_a.reset(options={"map_data": md, "map_name": "grounded_parity"})
            env_b.reset(options={"map_data": md, "map_name": "grounded_parity"})
            # Step with NOOP so both stay grounded at spawn.
            for _ in range(5):
                _oa, *_ = env_a.step(0)
                _ob, *_ = env_b.step(0)
            assert env_a._cached_switch_distance == pytest.approx(
                env_b._cached_switch_distance
            )
            assert env_a._cached_exit_distance == pytest.approx(
                env_b._cached_exit_distance
            )
        finally:
            env_a.close()
            env_b.close()
