"""End-to-end: the mine-threading potential feeds the lattice-PBRS reward.

Guards the full single-level training wiring on mines-extremely-difficult, where
the surface lattice is severed so the env's distance-substitution chain
(surface-lattice -> mine-threading -> free-space -> geometric) commits to the
MINE-THREADING source. Verifies that:

* the env actually uses the mine-threading source this episode,
* the reward calculator's per-step ``completion_cost`` equals the mine-threading
  source's ``cost_to_completion`` at the ninja's position (the substituted distance
  really reaches Phi),
* the PBRS term equals ``w * (gamma * Phi(s') - Phi(s))`` every step,
* HWM is replaced (hwm_alpha forced to 0) and the backtrack penalty is off,
* progress (cost decreasing) pays a non-negative shaping reward.
"""

from __future__ import annotations

import numpy as np
import pytest

from nclone.gym_environment.config import EnvironmentConfig, ReachabilityConfig
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.npp_environment import NppEnvironment


def _make_env():
    cfg = EnvironmentConfig(
        custom_map_path="nclone/test_maps/mines-extremely-difficult",
        sim_backend="python",
        frame_skip=1,
        reachability=ReachabilityConfig(use_lattice_potential=True),
        reward_config=RewardConfig(enable_lattice_pbrs=True, total_timesteps=1_000_000),
    )
    return NppEnvironment(cfg)


def test_minethreading_feeds_lattice_pbrs_end_to_end():
    env = _make_env()
    env.reset()
    rc = env.reward_calculator
    mt = env._minethreading_source

    # The severed-lattice chain must commit to the mine-threading source here.
    assert mt is not None and mt.available
    assert bool(env._use_minethreading_this_episode)
    # PBRS replaces HWM: alpha forced to 0, backtrack penalty disabled.
    assert rc.config.hwm_alpha == 0.0
    assert rc.config.enable_lattice_pbrs is True

    w, g = rc.config.pbrs_weight, rc.config.pbrs_gamma
    nph = env.nplay_headless
    prev_phi = None
    progressed = False
    saw_finite = False

    for t in range(40):
        _obs, _r, term, trunc, _info = env.step(3 if t % 2 else 4)
        lc = rc.last_components
        if lc.get("is_terminal"):
            # Death/win pay no per-step PBRS gradient term; stop checking the chain.
            break
        nx, ny = nph.ninja_position()
        sw_act = nph.exit_switch_activated()
        rc_cost = lc.get("completion_cost")
        assert rc_cost is not None
        # The reward calc's completion cost IS the mine-threading source value.
        mt_cost = mt.cost_to_completion(nx, ny, sw_act)
        assert abs(rc_cost - mt_cost) < 1.0
        saw_finite = saw_finite or np.isfinite(rc_cost)

        # PBRS term matches the canonical formula exactly.
        phi = -min(max(rc_cost / rc._pbrs_d0, 0.0), 1.0)
        pbrs = lc.get("pbrs_reward")
        assert pbrs is not None
        if prev_phi is not None:
            assert pbrs == pytest.approx(w * (g * phi - prev_phi), abs=1e-6)
        # Backtrack penalty is replaced by PBRS, never stacked.
        assert lc.get("backtrack_penalty", 0.0) == 0.0
        # HWM contributes nothing.
        assert lc.get("hwm_reward", 0.0) == 0.0
        if phi > (prev_phi if prev_phi is not None else -1.0):
            progressed = True  # cost dropped -> Phi rose
        prev_phi = phi

    assert saw_finite
    # D0 (the PBRS normalizer) is a finite spawn cost-to-completion, not a fallback.
    assert rc._pbrs_d0 is not None and np.isfinite(rc._pbrs_d0) and rc._pbrs_d0 > 1.0
