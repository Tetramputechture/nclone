"""Gold collection rewards: per-pickup delta, calibrated to 120 frames/gold.

See docs/GOLD_COLLECTION_HANDOFF.md. Each gold is worth GOLD_FRAME_VALUE (120)
frames of saved time; R_GOLD is calibrated to the per-frame time penalty so the
exchange is exact. Gold is a soft objective gated by
RewardConfig.enable_gold_rewards; it never gates completion. No all-gold bonus by
default (uniform valuation).
"""

from unittest.mock import patch

from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.reward_calculation.reward_constants import (
    R_GOLD,
    R_ALL_GOLD,
    R_EXIT,
    GOLD_FRAME_VALUE,
    TIME_PENALTY_PER_STEP,
)


def test_gold_value_calibrated_to_time_penalty():
    """R_GOLD == 120 frames of time penalty: collecting a gold exactly cancels
    the time cost of a GOLD_FRAME_VALUE-frame detour (break-even)."""
    assert abs(R_GOLD - GOLD_FRAME_VALUE * abs(TIME_PENALTY_PER_STEP)) < 1e-9
    # Sanity: at the default -0.002/frame this is 0.24.
    assert abs(R_GOLD - 0.24) < 1e-9


def test_all_gold_bonus_off_by_default():
    """Uniform valuation: no one-shot spike on the final gold."""
    assert R_ALL_GOLD == 0.0


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
        "total_gold": 4,
    }
    base.update(kwargs)
    return base


def _calc(enable_gold):
    cfg = RewardConfig(
        _time_penalty_per_step_override=0.0,
        _hwm_alpha_override=0.0,
        enable_gold_rewards=enable_gold,
    )
    return RewardCalculator(reward_config=cfg)


def _no_dist(calc, cur, prev):
    with patch.object(RewardCalculator, "_get_distances", return_value=(10.0, 20.0)):
        return calc.calculate_reward(cur, prev, frames_executed=1)


def test_gold_disabled_pays_nothing():
    calc = _calc(enable_gold=False)
    r = _no_dist(calc, _obs(gold_collected=2), _obs(gold_collected=0))
    assert abs(r) < 1e-6


def test_per_gold_delta_rewarded_order_agnostic():
    calc = _calc(enable_gold=True)
    # collect 2 gold this step -> 2 * R_GOLD
    r = _no_dist(calc, _obs(gold_collected=2), _obs(gold_collected=0))
    assert abs(r - 2 * R_GOLD) < 1e-6
    # collect 1 more -> 1 * R_GOLD (delta only, never re-pays prior gold)
    r2 = _no_dist(calc, _obs(gold_collected=3), _obs(gold_collected=2))
    assert abs(r2 - R_GOLD) < 1e-6
    # no new gold -> 0
    r3 = _no_dist(calc, _obs(gold_collected=3), _obs(gold_collected=3))
    assert abs(r3) < 1e-6


def test_no_all_gold_bonus_by_default():
    """Default uniform valuation: the final gold pays exactly R_GOLD, no spike."""
    calc = _calc(enable_gold=True)
    _no_dist(calc, _obs(gold_collected=3), _obs(gold_collected=0))
    # collecting the last (4th) gold -> just R_GOLD (no bonus)
    r = _no_dist(calc, _obs(gold_collected=4), _obs(gold_collected=3))
    assert abs(r - R_GOLD) < 1e-6


def test_optional_all_gold_bonus_override_fires_once():
    """The collect-everything variant (override) still pays a one-shot bonus."""
    cfg = RewardConfig(
        _time_penalty_per_step_override=0.0,
        _hwm_alpha_override=0.0,
        enable_gold_rewards=True,
        _all_gold_bonus_override=2.0,
    )
    calc = RewardCalculator(reward_config=cfg)
    _no_dist(calc, _obs(gold_collected=3), _obs(gold_collected=0))
    r = _no_dist(calc, _obs(gold_collected=4), _obs(gold_collected=3))
    assert abs(r - (R_GOLD + 2.0)) < 1e-6
    # one-shot: staying at all-collected pays nothing more
    r2 = _no_dist(calc, _obs(gold_collected=4), _obs(gold_collected=4))
    assert abs(r2) < 1e-6


def test_gold_counts_on_winning_step():
    """Gold collected on the same step the ninja wins is still rewarded."""
    calc = _calc(enable_gold=True)
    cur = _obs(gold_collected=4, player_won=True)
    prev = _obs(gold_collected=3)
    r = _no_dist(calc, cur, prev)
    # R_EXIT + last gold (R_GOLD); no all-gold bonus by default
    assert abs(r - (R_EXIT + R_GOLD)) < 1e-6


def test_gold_free_level_no_bonus():
    """A level with zero gold never triggers the all-gold bonus."""
    calc = _calc(enable_gold=True)
    r = _no_dist(calc, _obs(gold_collected=0, total_gold=0),
                 _obs(gold_collected=0, total_gold=0))
    assert abs(r) < 1e-6
