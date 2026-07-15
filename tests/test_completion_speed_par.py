"""V15 par-referenced completion-speed bonus.

The bonus is terminal-only, win-gated, bounded < R_EXIT, and (par form) an
efficiency ratio that CLAMPS at 1.0. These tests lock:
  - par form: full bonus at/under par, decaying as the win takes more frames;
  - clamp: beating par never exceeds completion_speed_bonus;
  - dominance: speed_bonus < R_EXIT always;
  - never paid on death;
  - fallback to the legacy time_remaining form when par is unavailable;
  - legacy form unchanged when the flag is off.
"""
from unittest.mock import patch

import pytest

from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.reward_calculation.reward_constants import (
    R_EXIT,
    GLOBAL_REWARD_SCALE,
)
from nclone.gym_environment.constants import NINJA_STATE_DIM

# game_state long enough to carry time_remaining at NINJA_STATE_DIM-1.
_GS_LEN = NINJA_STATE_DIM + 4


def _game_state(time_remaining=1.0):
    gs = [0.0] * _GS_LEN
    gs[NINJA_STATE_DIM - 1] = float(time_remaining)
    return gs


def _obs(won=False, dead=False, sim_frame=0, time_remaining=1.0, **kw):
    base = {
        "player_x": 0.0, "player_y": 0.0,
        "player_dead": dead, "player_won": won,
        "player_airborn": False, "player_walled": False,
        "switch_activated": True,  # so the win is a clean exit reach
        "switch_x": 1.0, "switch_y": 1.0,
        "exit_door_x": 2.0, "exit_door_y": 2.0,
        "_adjacency_graph": {}, "level_data": None, "_graph_data": None,
        "gold_collected": 0, "total_gold": 0,
        "game_state": _game_state(time_remaining),
        "_impact_risk": 0.0,
        "waypoint_leg_distance": 0.0, "waypoint_index": 0,
        "sim_frame": sim_frame,
        "death_cause": "impact" if dead else None,
    }
    base.update(kw)
    return base


def _calc(**cfg_kw):
    cfg = RewardConfig(completion_speed_bonus=3.0, **cfg_kw)
    c = RewardCalculator(reward_config=cfg)
    c.reset()
    return c


def _seed_and_win(calc, dist_switch, dist_exit, sim_frame, time_remaining=0.5,
                  won=True, dead=False):
    """Seed one non-terminal step (sets optimal_path_length = ds+de), then a
    terminal step. Returns (scaled_reward, speed_bonus_component)."""
    with patch.object(
        RewardCalculator, "_get_distances", return_value=(dist_switch, dist_exit)
    ):
        calc.calculate_reward(_obs(), _obs(), frames_executed=1)  # init step
        cur = _obs(won=won, dead=dead, sim_frame=sim_frame,
                   time_remaining=time_remaining)
        calc.calculate_reward(cur, _obs(), frames_executed=1)
    return calc.episode_speed_bonus_reward


# optimal_path_length = 200 + 50 = 250; par_speed_px 2.5 -> par_frames = 100.
PAR_FRAMES = 100.0


def test_par_form_full_bonus_at_or_under_par():
    calc = _calc(completion_speed_par_norm=True, completion_speed_par_speed_px=2.5)
    sb = _seed_and_win(calc, 200.0, 50.0, sim_frame=60)  # 60 < par 100
    assert sb == pytest.approx(3.0)  # ratio clamps at 1.0 -> full bonus


def test_par_form_decays_when_slow():
    calc = _calc(completion_speed_par_norm=True, completion_speed_par_speed_px=2.5)
    sb = _seed_and_win(calc, 200.0, 50.0, sim_frame=400)  # 4x par
    assert sb == pytest.approx(3.0 * (PAR_FRAMES / 400.0))  # 0.75


def test_par_form_monotonic_in_speed():
    fast = _seed_and_win(
        _calc(completion_speed_par_norm=True), 200.0, 50.0, sim_frame=120)
    slow = _seed_and_win(
        _calc(completion_speed_par_norm=True), 200.0, 50.0, sim_frame=600)
    assert fast > slow > 0.0


def test_par_bonus_never_exceeds_r_exit():
    # Even an instant win (huge par, tiny frames) is capped by completion_speed_bonus.
    sb = _seed_and_win(
        _calc(completion_speed_par_norm=True), 5000.0, 5000.0, sim_frame=1)
    assert sb <= 3.0 < R_EXIT


def test_no_speed_bonus_on_death():
    calc = _calc(completion_speed_par_norm=True)
    sb = _seed_and_win(calc, 200.0, 50.0, sim_frame=50, won=False, dead=True)
    assert sb == 0.0


def test_par_fallback_to_time_remaining_when_no_par():
    # optimal_path_length unavailable -> fall back to the legacy time_remaining form.
    calc = _calc(completion_speed_par_norm=True)
    with patch.object(RewardCalculator, "_get_distances", return_value=(200.0, 50.0)):
        calc.calculate_reward(_obs(), _obs(), frames_executed=1)  # init
        calc.optimal_path_length = None  # simulate unavailable par
        cur = _obs(won=True, sim_frame=50, time_remaining=0.4)
        calc.calculate_reward(cur, _obs(), frames_executed=1)
    assert calc.episode_speed_bonus_reward == pytest.approx(3.0 * 0.4)


def test_legacy_form_when_flag_off():
    calc = _calc(completion_speed_par_norm=False)
    sb = _seed_and_win(calc, 200.0, 50.0, sim_frame=50, time_remaining=0.7)
    assert sb == pytest.approx(3.0 * 0.7)  # legacy: bonus = W * time_remaining


def test_win_reward_dominates_speed_bonus():
    # The scaled win reward includes R_EXIT (10) + speed bonus (<=3); completion
    # is the strict dominant term regardless of speed.
    calc = _calc(completion_speed_par_norm=True)
    with patch.object(RewardCalculator, "_get_distances", return_value=(200.0, 50.0)):
        calc.calculate_reward(_obs(), _obs(), frames_executed=1)
        cur = _obs(won=True, sim_frame=400, time_remaining=0.5)
        scaled = calc.calculate_reward(cur, _obs(), frames_executed=1)
    # terminal win reward = R_EXIT + switch(already active so no re-add) + speed
    assert scaled >= R_EXIT * GLOBAL_REWARD_SCALE
    assert calc.episode_speed_bonus_reward < R_EXIT
