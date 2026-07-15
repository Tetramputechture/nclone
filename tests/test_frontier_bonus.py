"""Frontier watermark bonus: pay a small one-sided bonus only on a strictly-new
closest-approach (new minimum free-space cost-to-go) to the current live
objective.

See RewardConfig.enable_frontier_bonus. The bonus rewards monotonic progress
around a long "loop" level (committing to the climb) rather than basin-wandering:
a step that strictly decreases the cost-to-go to the objective pays
``frontier_bonus_weight * (best_prev - now)``; a non-improving step pays 0; the
watermark is rebased (no spurious bonus) when the live objective flips
switch->exit. Off by default and additive: behavior is unchanged when off.

The cost-to-go reused is ``current_goal_dist`` (the SAME phase-aware quantity the
lattice/free-space PBRS potential is built from): pre-switch it is the switch
distance, post-switch the exit distance.
"""

from unittest.mock import patch

from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig


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


def _calc(enable_frontier, weight=0.02):
    cfg = RewardConfig(
        _time_penalty_per_step_override=0.0,
        _hwm_alpha_override=0.0,
        enable_frontier_bonus=enable_frontier,
        frontier_bonus_weight=weight,
    )
    calc = RewardCalculator(reward_config=cfg)
    calc.reset()
    return calc


def _step(calc, dist_switch, dist_exit, switch_activated=False, **obs_kwargs):
    """Drive one env step with mocked A* distances to (switch, exit); return the
    isolated frontier-bonus contribution for that step.

    We read the ``frontier_bonus`` component (not the total reward) so the
    assertions are not confounded by the per-step BACKTRACK_PENALTY, which fires
    on any step where the cost-to-go increases (this pipeline runs with HWM/PBRS
    off). The total reward is checked separately in the flag-off baseline test.
    """
    cur = _obs(switch_activated=switch_activated, **obs_kwargs)
    prev = _obs(switch_activated=switch_activated)
    with patch.object(
        RewardCalculator, "_get_distances", return_value=(dist_switch, dist_exit)
    ):
        calc.calculate_reward(cur, prev, frames_executed=1)
    return calc.last_components.get("frontier_bonus", 0.0)


# ---------------------------------------------------------------------------
# (a) Flag OFF -> rewards unchanged (no bonus, ever).
# ---------------------------------------------------------------------------


def test_flag_off_pays_no_frontier_bonus_even_when_improving():
    calc = _calc(enable_frontier=False)
    # Strict improvements that WOULD pay a bonus when on contribute nothing here.
    assert abs(_step(calc, 100.0, 200.0)) < 1e-9
    assert abs(_step(calc, 80.0, 200.0)) < 1e-9
    assert abs(_step(calc, 50.0, 200.0)) < 1e-9


def test_flag_off_total_reward_unchanged_vs_on_for_improving_stream():
    """On a strictly-improving (monotone forward) trajectory the only difference
    between OFF and ON is the frontier bonus: every other term (here all zeroed,
    and no backtrack since distance only decreases) is identical. Proves the
    feature is purely additive."""
    traj = [100.0, 80.0, 60.0, 45.0, 30.0]  # monotone decreasing -> no backtrack
    cfg_off = RewardConfig(
        _time_penalty_per_step_override=0.0,
        _hwm_alpha_override=0.0,
        enable_frontier_bonus=False,
    )
    cfg_on = RewardConfig(
        _time_penalty_per_step_override=0.0,
        _hwm_alpha_override=0.0,
        enable_frontier_bonus=True,
        frontier_bonus_weight=0.02,
    )
    off = RewardCalculator(reward_config=cfg_off)
    on = RewardCalculator(reward_config=cfg_on)
    off.reset()
    on.reset()

    def total(calc, d):
        cur, prev = _obs(), _obs()
        with patch.object(RewardCalculator, "_get_distances", return_value=(d, 999.0)):
            return calc.calculate_reward(cur, prev, frames_executed=1)

    off_stream = [total(off, d) for d in traj]
    on_stream = [total(on, d) for d in traj]
    # OFF is exactly 0 on every step (all terms zeroed, monotone -> no backtrack).
    assert all(abs(r) < 1e-9 for r in off_stream)
    # ON differs ONLY by the frontier bonus (>= 0), and pays on each improvement.
    assert on_stream[0] == off_stream[0]  # first step seeds, no bonus
    assert all(on >= off - 1e-12 for on, off in zip(on_stream, off_stream))
    assert sum(on_stream) > sum(off_stream)  # net additive gain


# ---------------------------------------------------------------------------
# (b) Flag ON -> strict-improvement bonus proportional to the improvement;
#     non-improving step pays 0.
# ---------------------------------------------------------------------------


def test_first_valid_step_rebases_no_bonus():
    calc = _calc(enable_frontier=True)
    # The first finite measurement seeds the watermark; it must not pay a bonus.
    assert abs(_step(calc, 100.0, 200.0)) < 1e-9


def test_strict_improvement_pays_proportional_bonus():
    w = 0.02
    calc = _calc(enable_frontier=True, weight=w)
    _step(calc, 100.0, 200.0)  # seed watermark at 100 (D0 = 100)
    # improve to 70 -> bonus = w * (100 - 70) / D0 (D0-normalized; see RewardConfig)
    r = _step(calc, 70.0, 200.0)
    assert abs(r - w * (100.0 - 70.0) / 100.0) < 1e-9
    # improve to 55 -> bonus = w * (70 - 55) / D0 (improvement is vs the new best)
    r2 = _step(calc, 55.0, 200.0)
    assert abs(r2 - w * (70.0 - 55.0) / 100.0) < 1e-9


def test_non_improving_step_pays_zero():
    calc = _calc(enable_frontier=True)
    _step(calc, 100.0, 200.0)  # seed
    _step(calc, 70.0, 200.0)  # improve -> bonus
    # equal to the best -> no strict improvement -> 0
    assert abs(_step(calc, 70.0, 200.0)) < 1e-9
    # worse than the best (basin wander away) -> 0, and watermark is NOT raised
    assert abs(_step(calc, 90.0, 200.0)) < 1e-9
    # returning to 70 is still not below the 70 watermark -> 0
    assert abs(_step(calc, 70.0, 200.0)) < 1e-9
    # only going strictly below 70 pays again
    assert _step(calc, 60.0, 200.0) > 0.0


def test_inf_cost_is_safe_no_bonus_no_crash():
    calc = _calc(enable_frontier=True)
    # Unreachable objective: no bonus, no crash, watermark untouched.
    assert abs(_step(calc, float("inf"), 200.0)) < 1e-9
    # Once finite, this seeds (no bonus) ...
    assert abs(_step(calc, 80.0, 200.0)) < 1e-9
    # ... and a later improvement pays normally.
    assert _step(calc, 50.0, 200.0) > 0.0


# ---------------------------------------------------------------------------
# (c) Objective switch (switch->exit) does NOT pay a spurious bonus.
# ---------------------------------------------------------------------------


def test_switch_to_exit_transition_rebases_no_spurious_bonus():
    w = 0.02
    calc = _calc(enable_frontier=True, weight=w)
    # Approach the switch: switch-distance is the live objective.
    _step(calc, 100.0, 999.0, switch_activated=False)  # seed at 100 (D0 = 100)
    r = _step(calc, 40.0, 999.0, switch_activated=False)
    assert abs(r - w * (100.0 - 40.0) / 100.0) < 1e-9  # bonus on switch approach

    # Switch activates: live objective flips to the EXIT. The exit distance (300)
    # is much larger than the switch watermark (40); without a rebase the basis
    # change would look like a catastrophic regress (no bonus) or, if exit were
    # smaller, a spurious huge bonus. Either way the transition step must pay 0.
    r_trans = _step(calc, 40.0, 300.0, switch_activated=True)
    assert abs(r_trans) < 1e-9

    # And if the exit distance happens to be SMALLER than the prior switch
    # watermark, it still must not pay a spurious bonus on the transition.
    calc2 = _calc(enable_frontier=True, weight=w)
    _step(calc2, 100.0, 5.0, switch_activated=False)  # seed switch watermark at 100
    _step(calc2, 40.0, 5.0, switch_activated=False)
    r_trans2 = _step(calc2, 40.0, 5.0, switch_activated=True)  # exit dist 5 << 40
    assert abs(r_trans2) < 1e-9  # rebased to exit basis, no spurious bonus

    # After the rebase, post-switch exit approach pays normally.
    r_exit = _step(calc2, 40.0, 2.0, switch_activated=True)
    assert abs(r_exit - w * (5.0 - 2.0) / 100.0) < 1e-9
