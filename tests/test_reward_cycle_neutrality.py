"""Reward-farm resistance: cycle-neutrality + completion-dominance guarantees.

Locks in the reward-hacking audit in ``docs/PLATFORM_STAIRS_CURRICULUM_FOLLOWUP.md``
(§9): the ENV reward is farm-resistant by construction, so an oscillating /
wall-jumping closed loop (e.g. the observed post-switch wall-jumps near the exit)
can never bank net-positive extrinsic reward. Every dense term is exercised at
once on a genuine closed-loop trajectory (return to the same state):

- PBRS terms (lattice/free-space, per-leg waypoint, hazard, impact) use γ_shaping=1,
  so F = w*(Φ(s') - Φ(s)) TELESCOPES to 0 around any cycle.
- HWM terms (frontier watermark, grounded-landing HWM) are strict high-water-marks:
  they pay ≤ once per closest-approach and are NEVER re-paid, so REPEATING an
  identical loop pays 0.
- The per-frame time penalty makes any closed loop strictly net-NEGATIVE.
- Completion (R_EXIT=10) dominates every bounded shaping the agent can accrue by
  merely approaching the goal, so finishing always beats approach-and-time-out.

Guards the two-gamma wiring called out in §9: if a future change wired the 0.99
*value* gamma into the *shaping* term, closed loops over Φ<0 would earn a positive
survival drift ∝ w·|Φ|·(1-γ) — ``test_two_gamma_guard`` demonstrates the drift and
asserts the shipped default keeps γ_shaping=1.0.

These tests drive ``RewardCalculator`` directly, patching ``_get_distances`` to
inject the cost-to-completion sequence the env's reachability substitution supplies
(same technique as ``test_lattice_pbrs.py``).
"""

from unittest.mock import patch

import pytest

from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.reward_calculation.reward_constants import (
    LATTICE_PBRS_GAMMA,
    R_EXIT,
    TIME_PENALTY_PER_STEP,
)
from nclone.gym_environment.constants import (
    GAME_STATE_NEAREST_MINE_IDX,
    NINJA_STATE_DIM,
)

# game_state long enough to carry time_remaining (NINJA_STATE_DIM-1) and the
# nearest-mine normalized distance (GAME_STATE_NEAREST_MINE_IDX).
_GS_LEN = GAME_STATE_NEAREST_MINE_IDX + 4


def _game_state(mine_dist=1.0, time_remaining=1.0):
    gs = [0.0] * _GS_LEN
    gs[NINJA_STATE_DIM - 1] = float(time_remaining)
    gs[GAME_STATE_NEAREST_MINE_IDX] = float(mine_dist)
    return gs


def _obs(**kwargs):
    base = {
        "player_x": 0.0,
        "player_y": 0.0,
        "player_dead": False,
        "player_won": False,
        "player_airborn": False,  # grounded so the landing HWM can fire
        "player_walled": False,
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
        "game_state": _game_state(),
        "_impact_risk": 0.0,
        "waypoint_leg_distance": 0.0,
        "waypoint_index": 0,
    }
    base.update(kwargs)
    return base


def _calc_all():
    """A calculator with EVERY dense shaping term enabled at once."""
    cfg = RewardConfig(
        enable_lattice_pbrs=True,   # forces hwm_alpha=0 (PBRS replaces HWM)
        enable_waypoint_pbrs=True,
        enable_hazard_pbrs=True,
        enable_impact_pbrs=True,
        enable_frontier_bonus=True,
        enable_landing_pbrs=True,
        _pbrs_gamma_override=None,   # use the shipped default (1.0)
    )
    calc = RewardCalculator(reward_config=cfg)
    calc.reset()
    return calc


def _step(calc, dist_switch, dist_exit, **obs_kw):
    """One env step with injected (cost-to-switch, cost-to-exit) distances."""
    with patch.object(
        RewardCalculator, "_get_distances", return_value=(dist_switch, dist_exit)
    ):
        prev = _obs()
        cur = _obs(**obs_kw)
        r = calc.calculate_reward(cur, prev, frames_executed=1)
    return r, calc.last_components


# A closed loop over the pre-switch objective: return to the start distance.
# All auxiliary channels (mine dist, impact risk, waypoint leg) also return to
# their starting value so every PBRS potential is identical at loop start/end.
#   state:  (dist, mine_dist, impact_risk, leg_d)
_LOOP = [
    (120.0, 0.8, 0.2, 60.0),   # A -> B  (progress)
    (80.0, 0.6, 0.4, 40.0),    # B -> C  (more progress, closer to mine, riskier)
    (200.0, 1.0, 0.0, 100.0),  # C -> A  (all the way back to the start state)
]
_LOOP_START = (200.0, 1.0, 0.0, 100.0)


def _run_loop(calc):
    """Run one A->B->C->A loop; return (total_reward, summed_components)."""
    total = 0.0
    agg = {"pbrs_reward": 0.0, "frontier_bonus": 0.0, "landing_bonus": 0.0,
           "time_penalty": 0.0, "hwm_reward": 0.0, "backtrack_penalty": 0.0}
    for dist, mine_d, risk, leg_d in _LOOP:
        r, comp = _step(
            calc, dist, 50.0,
            game_state=_game_state(mine_dist=mine_d),
            _impact_risk=risk,
            waypoint_leg_distance=leg_d,
            waypoint_index=0,  # held constant: no forward reach-bonus
        )
        total += r
        for k in agg:
            agg[k] += comp.get(k, 0.0) or 0.0
    return total, agg


def _seed(calc):
    """Establish the loop start-state as the previous state (D0 + all Φ bases)."""
    dist, mine_d, risk, leg_d = _LOOP_START
    _step(
        calc, dist, 50.0,
        game_state=_game_state(mine_dist=mine_d),
        _impact_risk=risk,
        waypoint_leg_distance=leg_d,
        waypoint_index=0,
    )


# ---------------------------------------------------------------------------
# PBRS telescoping: every γ=1 shaping term nets exactly 0 around a closed loop.
# ---------------------------------------------------------------------------


def test_pbrs_terms_telescope_to_zero_on_closed_loop():
    calc = _calc_all()
    _seed(calc)
    _total, agg = _run_loop(calc)
    # lattice + waypoint + hazard + impact PBRS all accumulate into pbrs_reward;
    # returning to the identical state telescopes the whole sum to 0.
    assert agg["pbrs_reward"] == pytest.approx(0.0, abs=1e-9)
    # HWM ratchet is fully replaced by PBRS -> contributes nothing.
    assert agg["hwm_reward"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Non-farmability: REPEATING the identical loop is strictly net-negative, and
# the HWM bonuses (frontier, landing) are never re-paid.
# ---------------------------------------------------------------------------


def test_repeated_cycle_is_strictly_negative():
    calc = _calc_all()
    _seed(calc)
    _run_loop(calc)                 # loop 1: HWMs converge to the closest point
    total2, agg2 = _run_loop(calc)  # loop 2: identical trajectory
    # Every dense channel is 0 on the repeat; only the time penalty remains.
    assert agg2["pbrs_reward"] == pytest.approx(0.0, abs=1e-9)
    assert agg2["frontier_bonus"] == pytest.approx(0.0, abs=1e-12)
    assert agg2["landing_bonus"] == pytest.approx(0.0, abs=1e-12)
    assert total2 < 0.0
    # The only surviving term is 3 frames of time penalty.
    assert total2 == pytest.approx(3 * TIME_PENALTY_PER_STEP, abs=1e-9)


def test_hwm_bonuses_not_repaid_on_repeat():
    calc = _calc_all()
    _seed(calc)
    _t1, agg1 = _run_loop(calc)
    _t2, agg2 = _run_loop(calc)
    # First descent to a new closest-approach legitimately pays the HWM bonuses...
    assert agg1["frontier_bonus"] > 0.0
    assert agg1["landing_bonus"] > 0.0
    # ...but an identical repeat pays nothing (strict high-water-mark).
    assert agg2["frontier_bonus"] == pytest.approx(0.0, abs=1e-12)
    assert agg2["landing_bonus"] == pytest.approx(0.0, abs=1e-12)


def test_wall_jump_oscillation_near_goal_pays_no_net_positive():
    """The observed failure mode: post-progress wall-jump oscillation IN PLACE.

    A ninja that has already reached its closest approach then wall-jumps back
    and forth, returning to the same spot, must earn <= 0 extrinsic. The
    oscillation is a genuine CLOSED loop (start == end), so PBRS telescopes to 0;
    it never beats the closest approach (80), so the landing/frontier HWMs — incl.
    the wall-jump branch, gated on a strictly-new closest contact — pay nothing.
    Only the time penalty survives. (A trajectory that ends nearer than it started
    is real PROGRESS and legitimately pays; the farm concern is oscillation that
    RETURNS to the same state, which is what this asserts.)
    """
    calc = _calc_all()
    _seed(calc)
    _run_loop(calc)  # reach the closest approach (80), end back at the start (200)
    # Closed wall-jump oscillation: dip toward the wall and back to the start
    # state (dist 200, mine 1.0, impact 0, leg 100) — never beating 80.
    osc = [(150.0, 75.0), (170.0, 85.0), (200.0, 100.0)]  # (dist, leg_d); last == start
    total = 0.0
    for dist, leg_d in osc:
        r, _comp = _step(
            calc, dist, 50.0,
            player_walled=True, player_airborn=True,  # wall-jumping, not grounded
            game_state=_game_state(mine_dist=1.0),
            _impact_risk=0.0,
            waypoint_leg_distance=leg_d,
            waypoint_index=0,
        )
        total += r
    assert total < 0.0  # closed oscillation: net cost, never a farm
    assert total == pytest.approx(len(osc) * TIME_PENALTY_PER_STEP, abs=1e-9)


# ---------------------------------------------------------------------------
# Completion dominance: finishing beats approach-and-time-out.
# ---------------------------------------------------------------------------


def test_completion_dominates_approach_and_timeout():
    approach = _calc_all()
    approach.reset()
    _seed(approach)
    total_timeout = 0.0
    for dist in (150.0, 100.0, 60.0, 30.0):
        r, _ = _step(approach, dist, 50.0)
        total_timeout += r
    # Deep approach then time out.
    r_trunc, comp_trunc = _step(approach, 25.0, 50.0)
    total_timeout += r_trunc
    with patch.object(RewardCalculator, "_get_distances", return_value=(25.0, 50.0)):
        r_t = approach.calculate_reward(
            _obs(), _obs(), frames_executed=1, truncated=True
        )
    total_timeout += r_t

    winner = _calc_all()
    winner.reset()
    _seed(winner)
    total_win = 0.0
    for dist in (150.0, 100.0, 60.0, 30.0):
        r, _ = _step(winner, dist, 50.0)
        total_win += r
    r_win, comp_win = _step(winner, 5.0, 5.0, player_won=True)
    total_win += r_win

    assert comp_win["terminal_type"] == "win"
    # Completing pays the full R_EXIT terminal; the margin over the best
    # approach-then-timeout outcome is large and unconditional.
    assert total_win > total_timeout + R_EXIT / 2.0


def test_win_terminal_exceeds_any_dense_shaping():
    """R_EXIT alone dwarfs the whole-episode dense shaping bound."""
    calc = _calc_all()
    _seed(calc)
    _t, _agg = _run_loop(calc)
    _win_r, comp = _step(calc, 1.0, 1.0, player_won=True)
    # The completion terminal (R_EXIT + bounded PBRS bonus) is >> R_SWITCH etc.
    assert comp["terminal_reward"] >= R_EXIT


# ---------------------------------------------------------------------------
# Two-gamma guard: shipped shaping gamma is 1.0; a 0.99 misconfig would leak drift.
# ---------------------------------------------------------------------------


def test_two_gamma_guard():
    # The shipped default keeps the SHAPING gamma at 1.0 (pure telescoping),
    # distinct from the 0.99 value/BSRS horizon gamma.
    assert LATTICE_PBRS_GAMMA == 1.0
    assert RewardConfig(enable_lattice_pbrs=True).pbrs_gamma == 1.0

    # γ=1: closed loop telescopes to 0 (no survival drift).
    calc1 = _calc_all()
    _seed(calc1)
    _t, agg = _run_loop(calc1)
    assert agg["pbrs_reward"] == pytest.approx(0.0, abs=1e-9)

    # CANARY: if a regression wired the 0.99 value-gamma into the shaping term,
    # a closed loop over Φ<0 states would earn POSITIVE net PBRS drift (the
    # classic γ<1 PBRS hack) — this asserts the test would catch it.
    cfg = RewardConfig(
        enable_lattice_pbrs=True,
        enable_waypoint_pbrs=False,
        enable_hazard_pbrs=False,
        enable_impact_pbrs=False,
        _pbrs_gamma_override=0.99,
    )
    calc99 = RewardCalculator(reward_config=cfg)
    calc99.reset()
    # seed + closed loop, lattice PBRS only (isolate the drift).
    with patch.object(RewardCalculator, "_get_distances", return_value=(200.0, 50.0)):
        calc99.calculate_reward(_obs(), _obs(), frames_executed=1)
    pbrs_sum = 0.0
    for dist in (120.0, 80.0, 200.0):
        with patch.object(
            RewardCalculator, "_get_distances", return_value=(dist, 50.0)
        ):
            calc99.calculate_reward(_obs(), _obs(), frames_executed=1)
        pbrs_sum += calc99.last_components["pbrs_reward"]
    # Φ<0 everywhere on the loop -> γ<1 leaks strictly-positive drift.
    assert pbrs_sum > 1e-4
