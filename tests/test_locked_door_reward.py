"""Tests for the necessary-locked-door-switch reward + HWM sub-goal phase.

Covers the reward-side wiring that teaches the agent to route through gating
switches: a one-shot bonus per newly-opened NECESSARY locked door, and the HWM
watermark reset when the pre-exit-switch sub-goal advances to the next door.
"""

from __future__ import annotations

from nclone.gym_environment.reward_calculation.hwm_shaping import EpisodeRewardState
from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.reward_calculation.reward_constants import (
    GLOBAL_REWARD_SCALE,
    R_LOCKED_DOOR_SWITCH,
)


# --- HWM sub-goal phase --------------------------------------------------
def test_hwm_resets_switch_watermark_on_subgoal_advance():
    s = EpisodeRewardState()
    s.reset(initial_dist_switch=100.0, max_dist_switch=100.0,
            initial_dist_exit=200.0, max_dist_exit=200.0)
    # Progress toward the first sub-goal (door switch): distance drops 100 -> 60.
    r = s.step(dist_switch=60.0, dist_exit=200.0, switch_activated=False, subgoal_phase=0)
    assert r > 0
    assert s.watermark_switch == 60.0
    # A necessary door opens: sub-goal jumps to the next (farther) switch. The
    # distance metric jumps up; the watermark must reset (no spurious penalty,
    # no progress this frame).
    r = s.step(dist_switch=150.0, dist_exit=200.0, switch_activated=False, subgoal_phase=1)
    assert r == 0.0
    assert s.watermark_switch == 150.0
    # Now progress on the new leg is credited again.
    r = s.step(dist_switch=120.0, dist_exit=200.0, switch_activated=False, subgoal_phase=1)
    assert r > 0


def test_hwm_unchanged_without_doors():
    """subgoal_phase stays 0 -> identical to the original two-phase machine."""
    s = EpisodeRewardState()
    s.reset(100.0, 100.0, 200.0, 200.0)
    assert s.step(80.0, 200.0, False) > 0          # progress to switch
    assert s.step(90.0, 200.0, False) == 0.0       # backtrack -> zero
    assert s.step(50.0, 200.0, True) == 0.0        # phase transition
    assert s.step(50.0, 120.0, True) > 0           # progress to exit


# --- Locked-door reward --------------------------------------------------
def _obs(necessary_doors_open=0, **kw):
    base = {
        "player_x": 100.0, "player_y": 100.0,
        "switch_x": 200.0, "switch_y": 100.0,
        "exit_door_x": 300.0, "exit_door_y": 100.0,
        "switch_activated": False,
        "player_dead": False, "player_won": False,
        "_cached_switch_distance": 50.0,
        "_cached_exit_distance": 150.0,
        "necessary_doors_open": necessary_doors_open,
    }
    base.update(kw)
    return base


def _calc(enable: bool) -> RewardCalculator:
    return RewardCalculator(reward_config=RewardConfig(enable_locked_door_rewards=enable))


def test_necessary_door_open_pays_bonus_once():
    calc = _calc(enable=True)
    prev = _obs(0)
    # Step 1: no door opened yet.
    calc.calculate_reward(_obs(0), prev)
    assert calc.last_components.get("locked_door_reward", 0.0) == 0.0
    # Step 2: a necessary door opens -> one-shot bonus.
    calc.calculate_reward(_obs(1), _obs(0))
    assert calc.last_components["locked_door_reward"] == R_LOCKED_DOOR_SWITCH
    # Step 3: still 1 open -> no repeat.
    calc.calculate_reward(_obs(1), _obs(1))
    assert calc.last_components["locked_door_reward"] == 0.0


def test_two_doors_open_same_step_pays_double():
    calc = _calc(enable=True)
    calc.calculate_reward(_obs(0), _obs(0))
    calc.calculate_reward(_obs(2), _obs(0))
    assert calc.last_components["locked_door_reward"] == 2 * R_LOCKED_DOOR_SWITCH


def test_no_reward_when_disabled():
    calc = _calc(enable=False)
    calc.calculate_reward(_obs(0), _obs(0))
    calc.calculate_reward(_obs(1), _obs(0))
    assert calc.last_components.get("locked_door_reward", 0.0) == 0.0


def test_door_bonus_counts_on_winning_step():
    calc = _calc(enable=True)
    calc.calculate_reward(_obs(0), _obs(0))
    # Win on the same step a necessary door opens.
    r = calc.calculate_reward(_obs(1, player_won=True), _obs(0))
    comp = calc.last_components
    assert comp["terminal_type"] == "win"
    assert comp["locked_door_reward"] == R_LOCKED_DOOR_SWITCH
    # The bonus is included in the terminal reward.
    assert comp["terminal_reward"] == (
        calc.config.level_completion_reward + R_LOCKED_DOOR_SWITCH
    )
