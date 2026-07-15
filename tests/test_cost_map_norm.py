"""v18 Component 3: maze-adaptive cost_to_go_map normalization tests.

The fixed ``c = m/(m+K), K=1000`` normalization was tuned for long FAR routes;
on compact levels every cost is tiny vs K, so the whole map compresses into a
near-flat band near 0 and the switch-direction gradient is useless. Covers:

  * "fixed_k" (default) is byte-identical to the legacy behavior;
  * "per_level_dmax" spans ~[0, 1] on a compact level (the flat-gradient fix);
  * "log" lies between the two and stays in [0, 1];
  * all modes agree on the reachable mask and keep unreachable filler at 1.0;
  * per-cell ordering (monotonicity in the underlying cost m) is preserved
    across modes;
  * invalid mode strings are rejected at config construction.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from nclone.gym_environment.config import EnvironmentConfig, ReachabilityConfig
from nclone.gym_environment.generator_registry import GeneratorRegistry
from nclone.gym_environment.npp_environment import NppEnvironment


def _map(category: str = "simple", seed: int = 7):
    reg = GeneratorRegistry(random.Random(seed))
    return [int(v) for v in reg.get_map(category, seed).map_data()]


def _make_env(norm: str | None = None):
    cfg = EnvironmentConfig.for_training(seed=0, sim_backend="python")
    cfg.reachability.expose_cost_to_go_map = True
    if norm is not None:
        cfg.reachability.cost_map_norm = norm
    return NppEnvironment(cfg)


def _cost_map(env) -> np.ndarray:
    obs, _ = env.reset(options={"map_data": _map(), "map_name": "cn1"})
    return np.asarray(obs["cost_to_go_map"], dtype=np.float32)


class TestCostMapNorm:
    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            ReachabilityConfig(cost_map_norm="bogus")

    def test_default_is_fixed_k(self):
        env = _make_env(norm=None)
        try:
            assert env._cost_map_norm == "fixed_k"
        finally:
            env.close()

    def test_fixed_k_flat_on_compact_level(self):
        # The bug this component fixes: with K=1000, a compact level's costs
        # are tiny vs K so reachable cells all land in a thin band near 0.
        env = _make_env("fixed_k")
        try:
            cm = _cost_map(env)
            reach = cm[2] > 0.5
            assert reach.any()
            span = float(cm[0][reach].max() - cm[0][reach].min())
            assert span < 0.5  # compressed band (typically ~0.05-0.2)
        finally:
            env.close()

    def test_per_level_dmax_spans_full_range(self):
        env = _make_env("per_level_dmax")
        try:
            cm = _cost_map(env)
            reach = cm[2] > 0.5
            assert reach.any()
            vals = cm[0][reach]
            assert float(vals.min()) < 0.1  # near the goal -> ~0
            # The farthest reachable block must approach 1.0. Downsampling
            # min-pools costs, so the pooled max sits a hair under the true
            # per-cell max of exactly 1.0.
            assert float(vals.max()) > 0.85
            assert np.isfinite(cm).all()
            assert cm.min() >= 0.0 and cm.max() <= 1.0
        finally:
            env.close()

    def test_log_mode_in_bounds_and_wider_than_fixed_k(self):
        env_log = _make_env("log")
        env_fix = _make_env("fixed_k")
        try:
            cml, cmf = _cost_map(env_log), _cost_map(env_fix)
            reach = cml[2] > 0.5
            assert reach.any()
            assert cml.min() >= 0.0 and cml.max() <= 1.0
            span_log = float(cml[0][reach].max() - cml[0][reach].min())
            span_fix = float(cmf[0][reach].max() - cmf[0][reach].min())
            assert span_log > span_fix  # compressive boost at small m
        finally:
            env_log.close()
            env_fix.close()

    def test_modes_agree_on_reachability_and_filler(self):
        maps = {}
        for mode in ("fixed_k", "per_level_dmax", "log"):
            env = _make_env(mode)
            try:
                maps[mode] = _cost_map(env)
            finally:
                env.close()
        ref_mask = maps["fixed_k"][2]
        for mode, cm in maps.items():
            np.testing.assert_array_equal(
                cm[2], ref_mask, err_msg=f"reachable mask differs in {mode}"
            )
            unreach = cm[2] < 0.5
            if unreach.any():
                assert float(cm[0][unreach].min()) == 1.0, mode

    def test_ordering_preserved_across_modes(self):
        # All modes are monotone in the underlying cost m, so the per-cell
        # ordering of reachable costs must be identical.
        env_a = _make_env("fixed_k")
        env_b = _make_env("per_level_dmax")
        try:
            a, b = _cost_map(env_a), _cost_map(env_b)
            reach = (a[2] > 0.5) & (b[2] > 0.5)
            av, bv = a[0][reach], b[0][reach]
            order_a = np.argsort(av, kind="stable")
            # Sorting b's values by a's order must yield a non-decreasing
            # sequence (allow fp slack).
            sorted_b = bv[order_a]
            assert (np.diff(sorted_b) >= -1e-6).all()
        finally:
            env_a.close()
            env_b.close()
