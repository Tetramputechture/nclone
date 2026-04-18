"""L1.5: R_SWITCH = 0.5 fires once on switch_just_activated (non-terminal)."""

from unittest.mock import patch

from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig


def _minimal_obs(**kwargs):
    base = {
        "player_x": 0.0,
        "player_y": 0.0,
        "player_dead": False,
        "player_won": False,
        "switch_x": 1.0,
        "switch_y": 1.0,
        "exit_door_x": 2.0,
        "exit_door_y": 2.0,
        "_adjacency_graph": {},
        "level_data": None,
        "_graph_data": None,
    }
    base.update(kwargs)
    return base


def test_switch_bonus_exactly_half_with_no_hwm_no_time():
    cfg = RewardConfig(
        _time_penalty_per_step_override=0.0,
        _hwm_alpha_override=0.0,
    )
    calc = RewardCalculator(reward_config=cfg)

    prev = _minimal_obs(switch_activated=False)
    cur = _minimal_obs(switch_activated=True)

    with patch.object(RewardCalculator, "_get_distances", return_value=(10.0, 20.0)):
        r_switch = calc.calculate_reward(cur, prev, frames_executed=1)
    assert abs(r_switch - 0.5) < 1e-6

    with patch.object(RewardCalculator, "_get_distances", return_value=(10.0, 20.0)):
        r_after = calc.calculate_reward(cur, cur, frames_executed=1)
    assert abs(r_after - 0.0) < 1e-6
