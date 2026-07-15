"""Per-leg waypoint PBRS reward term (RewardConfig.enable_waypoint_pbrs).

A potential-based term toward the CURRENT local waypoint, rebased on each
waypoint advance so every leg is short and the per-step gradient is
route-length-invariant (the long-horizon distance-dropoff fix):

    Phi(s) = -clamp(waypoint_leg_distance / waypoint_leg_scale, 0, 1)   in [-1, 0]
    F(s, s') = pbrs_weight * (pbrs_gamma * Phi(s') - Phi(s))            within a leg
    (on a waypoint advance the basis is rebased -> the hand-off pays 0)

These tests drive ``RewardCalculator`` directly, injecting the
``waypoint_leg_distance`` + ``waypoint_index`` the env's WaypointTracker would
supply via obs. The distance signal is patched to inf (unused by this term) to
isolate the waypoint shaping.
"""

from unittest.mock import patch

from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.reward_calculation.reward_constants import (
    LATTICE_PBRS_WEIGHT,
    LATTICE_PBRS_GAMMA,
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


def _calc(leg_scale=96.0):
    cfg = RewardConfig(
        enable_waypoint_pbrs=True,
        waypoint_leg_scale=leg_scale,
        _time_penalty_per_step_override=0.0,  # isolate the PBRS term
    )
    calc = RewardCalculator(reward_config=cfg)
    calc.reset()
    return calc


def _step(calc, leg_distance, waypoint_index, **obs_kw):
    with patch.object(
        RewardCalculator, "_get_distances", return_value=(float("inf"), float("inf"))
    ):
        prev = _obs()
        cur = _obs(
            waypoint_leg_distance=leg_distance,
            waypoint_index=waypoint_index,
            **obs_kw,
        )
        calc.calculate_reward(cur, prev, frames_executed=1)
    return calc.last_components["pbrs_reward"]


def test_first_step_rebase_pays_zero():
    calc = _calc()
    r = _step(calc, leg_distance=96.0, waypoint_index=0)
    assert abs(r) < 1e-9  # first observation rebases the basis -> no spurious reward


def test_within_leg_progress_telescopes_to_weight():
    calc = _calc(leg_scale=96.0)
    _step(calc, 96.0, 0)                 # rebase, phi=-1.0
    r2 = _step(calc, 48.0, 0)            # phi=-0.5: 3*( -0.5 - -1.0 ) = 1.5
    r3 = _step(calc, 0.0, 0)             # phi= 0.0: 3*(  0.0 - -0.5 ) = 1.5
    assert r2 == LATTICE_PBRS_WEIGHT * 0.5
    assert r3 == LATTICE_PBRS_WEIGHT * 0.5
    # full leg (Phi -1 -> 0) banks exactly w*(0 - (-1)) = pbrs_weight
    assert r2 + r3 == LATTICE_PBRS_WEIGHT


def test_regress_within_leg_is_penalized_symmetrically():
    calc = _calc(leg_scale=96.0)
    _step(calc, 48.0, 0)                 # rebase at phi=-0.5
    r = _step(calc, 72.0, 0)             # phi=-0.75: 3*(-0.75 - -0.5) = -0.75
    assert r == LATTICE_PBRS_WEIGHT * (-0.25)


def test_waypoint_advance_pays_zero_handoff():
    calc = _calc(leg_scale=96.0)
    _step(calc, 96.0, 0)                 # leg 0 rebase
    _step(calc, 10.0, 0)                 # approach leg 0
    # advance to leg 1, new waypoint 96px away: rebase -> NO negative spike
    r = _step(calc, 96.0, 1)
    assert abs(r) < 1e-9


def test_clamp_caps_far_leg_potential():
    calc = _calc(leg_scale=96.0)
    _step(calc, 500.0, 0)                # phi clamped to -1.0 (rebase)
    r = _step(calc, 96.0, 0)             # phi=-1.0 still: delta 0
    assert abs(r) < 1e-9


def test_absent_waypoint_fields_are_safe_noop():
    calc = _calc()
    with patch.object(
        RewardCalculator, "_get_distances", return_value=(float("inf"), float("inf"))
    ):
        # no waypoint_* keys in obs -> term must not fire or crash
        calc.calculate_reward(_obs(), _obs(), frames_executed=1)
    assert calc.last_components["pbrs_reward"] == 0.0


def test_mutual_exclusion_forces_hwm_alpha_zero():
    cfg = RewardConfig(enable_waypoint_pbrs=True)
    assert cfg.hwm_alpha == 0.0
    assert cfg.pbrs_weight == LATTICE_PBRS_WEIGHT
    assert cfg.pbrs_gamma == LATTICE_PBRS_GAMMA
