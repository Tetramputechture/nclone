"""Lattice/free-space PBRS reward term (RewardConfig.enable_lattice_pbrs).

The term is a true potential-based shaping signal that REPLACES the HWM ratchet:

    Phi(s) = -clamp(completion_cost(s) / D0, 0, 1)        in [-1, 0]
    F(s, s') = pbrs_weight * (pbrs_gamma * Phi(s') - Phi(s))

These tests drive ``RewardCalculator`` directly, patching ``_get_distances`` to
inject the cost-to-completion sequence the env's lattice/free-space substitution
would otherwise supply. They assert the canonical PBRS properties: telescoping to
``w*(Phi_T - Phi_0)``, symmetry (regress is penalized, not zeroed), continuity
across switch activation, HWM/backtrack replacement, inf clamping, and that death
pays no shaping bonus.
"""

from unittest.mock import patch

import pytest

from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.reward_calculation.reward_constants import (
    LATTICE_PBRS_WEIGHT,
    LATTICE_PBRS_GAMMA,
    HWM_ALPHA,
)


def _obs(**kwargs):
    base = {
        "player_x": 0.0,
        "player_y": 0.0,
        "player_dead": False,
        "player_won": False,
        "switch_activated": False,
        "switch_x": 1.0,
        "switch_y": 1.0,
        "exit_door_x": 2.0,
        "exit_door_y": 2.0,
        "_adjacency_graph": {},
        "level_data": None,
        "_graph_data": None,
        "gold_collected": 0,
        "total_gold": 0,
    }
    base.update(kwargs)
    return base


def _calc(weight=None, gamma=None):
    cfg = RewardConfig(
        enable_lattice_pbrs=True,
        _time_penalty_per_step_override=0.0,  # isolate the PBRS term
        _pbrs_weight_override=weight,
        _pbrs_gamma_override=gamma,
    )
    calc = RewardCalculator(reward_config=cfg)
    calc.reset()  # env lifecycle: reset before stepping (initializes episode state)
    return calc


def _step(calc, dist_switch, dist_exit, **obs_kw):
    """One env step with injected (cost-to-completion, cost-to-exit) distances."""
    with patch.object(
        RewardCalculator, "_get_distances", return_value=(dist_switch, dist_exit)
    ):
        prev = _obs()
        cur = _obs(**obs_kw)
        calc.calculate_reward(cur, prev, frames_executed=1)
    return calc.last_components


# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = RewardConfig(enable_lattice_pbrs=True)
    assert cfg.pbrs_weight == LATTICE_PBRS_WEIGHT
    assert cfg.pbrs_gamma == LATTICE_PBRS_GAMMA
    # PBRS replaces HWM: alpha is forced to 0 so the two never stack.
    assert cfg.hwm_alpha == 0.0
    # Off by default.
    assert RewardConfig().enable_lattice_pbrs is False
    assert RewardConfig().hwm_alpha == HWM_ALPHA


def test_potential_normalized_to_unit_interval():
    calc = _calc()
    # First step sets D0 from the first injected pre-switch distance (= spawn).
    _step(calc, dist_switch=100.0, dist_exit=50.0)
    assert calc._pbrs_d0 == pytest.approx(100.0)
    assert calc._potential(100.0) == pytest.approx(-1.0)  # at spawn
    assert calc._potential(0.0) == pytest.approx(0.0)     # at completion
    assert calc._potential(50.0) == pytest.approx(-0.5)
    # Beyond D0 (transient) clamps to the worst potential, never below -1.
    assert calc._potential(500.0) == pytest.approx(-1.0)


def test_progress_positive_regress_negative():
    """Symmetric: closing distance pays positive, opening it pays negative."""
    calc = _calc(gamma=1.0)  # gamma=1 => F = w*(Phi' - Phi), pure progress delta
    _step(calc, 100.0, 50.0)              # seed D0=100, prev_phi=-1.0
    fwd = _step(calc, 80.0, 50.0)         # moved closer
    assert fwd["pbrs_reward"] > 0
    back = _step(calc, 95.0, 50.0)        # moved away
    assert back["pbrs_reward"] < 0


def test_telescopes_to_endpoint_difference():
    """Sum of F over a trajectory == w*(gamma-weighted Phi telescoping).

    With gamma=1 the discounted telescoping collapses to w*(Phi_last - Phi_first),
    independent of the path taken between them — the policy-invariance hallmark.
    """
    w = 2.0
    calc = _calc(weight=w, gamma=1.0)
    dists = [100.0, 70.0, 85.0, 40.0, 60.0, 20.0]  # noisy but net progress
    total = 0.0
    phi_first = None
    last_phi = None
    for d in dists:
        comp = _step(calc, d, 50.0)
        total += comp["pbrs_reward"]
        if phi_first is None:
            # After the first step prev_potential == Phi(spawn); seed term is ~0.
            phi_first = -1.0
        last_phi = comp["potential"]
    assert total == pytest.approx(w * (last_phi - phi_first), abs=1e-6)


def test_win_pays_completion_bonus_and_completes_telescope():
    """Phi(terminal)=0 => terminal F = -w*Phi(s) >= 0; total episode shaping ~ w."""
    w = 3.0
    calc = _calc(weight=w, gamma=1.0)
    total = 0.0
    for d in [100.0, 60.0, 30.0]:
        total += _step(calc, d, 40.0)["pbrs_reward"]
    win = _step(calc, 10.0, 5.0, player_won=True)
    total += win["pbrs_reward"]
    assert win["pbrs_reward"] > 0
    # Telescopes from Phi(spawn)=-1 to Phi(terminal)=0 => ~ w * 1.
    assert total == pytest.approx(w * 1.0, abs=1e-6)


def test_death_pays_no_pbrs_bonus():
    """A positive terminal bonus on death would reward dying — must be 0."""
    calc = _calc()
    _step(calc, 100.0, 50.0)
    dead = _step(calc, 90.0, 50.0, player_dead=True, death_cause="mine")
    assert dead.get("pbrs_reward", 0.0) == 0.0
    assert dead["terminal_type"] == "death"


def test_switch_activation_is_continuous_no_spike():
    """Under the composed cost-to-completion the potential is continuous across
    switch activation, so PBRS does NOT emit a large negative spike on the step
    the switch fires. We feed a monotone completion sequence that crosses the
    phase boundary (pre-switch distances are completion-cost; post-switch are the
    matching exit-cost) and require the activation step's shaping to stay positive.
    """
    calc = _calc(gamma=1.0)
    # Pre-switch completion cost decreasing toward the switch.
    _step(calc, 200.0, 0.0)                 # seed D0=200
    _step(calc, 150.0, 0.0)
    pre = _step(calc, 120.0, 0.0)
    assert pre["pbrs_reward"] > 0
    # Activation step: completion continues to drop (118 -> exit cost 118), and
    # switch flips. current_goal_dist switches basis to dist_exit but the VALUE is
    # continuous, so shaping stays positive (no punish-the-switch spike).
    act = _step(calc, 999.0, 118.0, switch_activated=True)
    assert act["pbrs_reward"] > 0
    # Post-switch progress on the exit field keeps paying positively.
    post = _step(calc, 999.0, 90.0, switch_activated=True)
    assert post["pbrs_reward"] > 0


def test_inf_distance_clamps_potential_bounded():
    """A transient inf cost-to-completion clamps Phi to -1; the shaping stays
    finite and bounded (no inf/NaN leaking into the reward)."""
    calc = _calc(gamma=1.0)
    _step(calc, 100.0, 50.0)               # seed D0
    comp = _step(calc, float("inf"), 50.0)
    assert comp["pbrs_reward"] == pytest.approx(LATTICE_PBRS_WEIGHT * (-1.0 - -1.0))
    # Recovering from inf back to a finite near-goal distance pays positive.
    rec = _step(calc, 20.0, 50.0)
    assert rec["pbrs_reward"] > 0


def test_no_backtrack_penalty_when_pbrs_on():
    """The HWM-pipeline backtrack penalty is replaced by PBRS, not stacked."""
    calc = _calc(gamma=1.0)
    _step(calc, 200.0, 50.0)               # seed D0=200 so a regress stays in range
    _step(calc, 100.0, 50.0)               # progress
    comp = _step(calc, 130.0, 50.0)        # regressed (still within [0, D0])
    assert comp["backtrack_penalty"] == 0.0  # penalty disabled under PBRS
    assert comp["pbrs_reward"] < 0           # but PBRS still penalizes the regress
    # hwm shaping contributes nothing.
    assert comp["hwm_reward"] == 0.0
