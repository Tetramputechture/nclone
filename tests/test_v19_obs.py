"""v19 observation/telemetry tests (R4a recency, R4b ego-residual, R3 telemetry).

Covers:
  * ego_residual cost_to_go_map: values in [-1, 1], reachable mask matches the
    legacy modes, unreachable filler == 1.0, and the channel is POSITION-DEPENDENT
    (changes as the ninja moves — the whole point vs the per-level-constant modes);
  * recency (time-decay) visitation channel: tau>0 emits exp(-age/tau) with a
    freshly-visited cell near 1.0; tau==0 reproduces the legacy 1/sqrt(count);
  * R3 telemetry: waypoint_active / waypoint_index / waypoint_count /
    waypoint_reach_bonus and pbrs_spawn_cost_finite are present in info.
"""

from __future__ import annotations

import random

import numpy as np

from nclone.gym_environment.config import EnvironmentConfig
from nclone.gym_environment.generator_registry import GeneratorRegistry
from nclone.gym_environment.npp_environment import NppEnvironment


def _map(category: str = "simple", seed: int = 7):
    reg = GeneratorRegistry(random.Random(seed))
    return [int(v) for v in reg.get_map(category, seed).map_data()]


def _reset(env):
    obs, _ = env.reset(options={"map_data": _map(), "map_name": "v19"})
    return obs


class TestEgoResidualCostMap:
    def _env(self, norm):
        cfg = EnvironmentConfig.for_training(seed=0, sim_backend="python")
        cfg.reachability.expose_cost_to_go_map = True
        cfg.reachability.cost_map_norm = norm
        return NppEnvironment(cfg)

    def test_ego_residual_in_bounds_and_mask_matches(self):
        env = self._env("ego_residual")
        env_fx = self._env("fixed_k")
        try:
            obs = _reset(env)
            obs_fx = _reset(env_fx)
            cm = np.asarray(obs["cost_to_go_map"], dtype=np.float32)
            cmf = np.asarray(obs_fx["cost_to_go_map"], dtype=np.float32)
            # Values bounded to [-1, 1] on the residual channels.
            assert cm[0].min() >= -1.0 - 1e-6 and cm[0].max() <= 1.0 + 1e-6
            assert cm[1].min() >= -1.0 - 1e-6 and cm[1].max() <= 1.0 + 1e-6
            assert np.isfinite(cm).all()
            # Reachable mask identical to the legacy mode (same source slice).
            np.testing.assert_array_equal(cm[2], cmf[2])
            # Unreachable filler saturates at +1.0.
            unreach = cm[2] < 0.5
            if unreach.any():
                assert float(cm[0][unreach].min()) == 1.0
        finally:
            env.close()
            env_fx.close()

    def test_ego_residual_is_position_dependent(self):
        # The defining property: unlike per-level-constant modes, the residual map
        # changes as the ninja moves (re-referenced to the ninja's current cost).
        env = self._env("ego_residual")
        try:
            obs0 = _reset(env)
            cm0 = np.asarray(obs0["cost_to_go_map"], dtype=np.float32).copy()
            changed = False
            for _ in range(40):
                obs, _, term, trunc, _ = env.step(1)  # move right
                cm = np.asarray(obs["cost_to_go_map"], dtype=np.float32)
                if not np.allclose(cm[0], cm0[0], atol=1e-5):
                    changed = True
                    break
                if term or trunc:
                    break
            assert changed, "ego_residual cost map did not change as the ninja moved"
        finally:
            env.close()


class TestRecencyVisitationChannel:
    def _env(self, tau):
        cfg = EnvironmentConfig.for_training(seed=0, sim_backend="python")
        cfg.reachability.expose_visitation_map = True
        cfg.reachability.visitation_recency_tau = tau
        return NppEnvironment(cfg)

    def test_recency_mode_fresh_cell_near_one(self):
        env = self._env(600.0)
        try:
            _reset(env)
            last = None
            for _ in range(30):
                obs, _, term, trunc, _ = env.step(1)
                last = obs
                if term or trunc:
                    break
            vm = np.asarray(last["visitation_map"], dtype=np.float32)
            vis = vm[0][vm[0] > 0]
            assert vis.size > 0
            # exp(-age/tau) in (0, 1]; the just-visited cell is ~1.0.
            assert vis.max() <= 1.0 + 1e-6
            assert vis.max() > 0.9
            assert vis.min() > 0.0
        finally:
            env.close()

    def test_legacy_mode_is_count_decay(self):
        env = self._env(0.0)
        try:
            _reset(env)
            last = None
            for _ in range(20):
                obs, _, term, trunc, _ = env.step(1)
                last = obs
                if term or trunc:
                    break
            vm = np.asarray(last["visitation_map"], dtype=np.float32)
            vis = vm[0][vm[0] > 0]
            assert vis.size > 0
            # 1/sqrt(count): every value is of the form 1/sqrt(n) for integer n>=1.
            recip_sq = 1.0 / (vis ** 2)
            assert np.allclose(recip_sq, np.round(recip_sq), atol=1e-3)
        finally:
            env.close()


class TestWaypointAndSpawnTelemetry:
    def _env(self):
        cfg = EnvironmentConfig.for_training(seed=0, sim_backend="python")
        rc = cfg.reachability
        rc.enable_waypoint_pbrs = True
        rc.enable_waypoint_tracker = True
        rc.expose_waypoint_features = True
        rc.use_jump_potential = True
        rc.expose_cost_to_go_map = True
        return NppEnvironment(cfg)

    def test_info_has_waypoint_and_spawn_keys(self):
        env = self._env()
        try:
            _reset(env)
            _, _, _, _, info = env.step(1)
            for k in (
                "waypoint_active",
                "waypoint_index",
                "waypoint_count",
                "waypoint_reach_bonus",
                "pbrs_spawn_cost_finite",
            ):
                assert k in info, f"missing telemetry key: {k}"
            assert isinstance(info["waypoint_active"], bool)
            assert isinstance(info["waypoint_count"], int)
            assert isinstance(info["pbrs_spawn_cost_finite"], bool)
        finally:
            env.close()
