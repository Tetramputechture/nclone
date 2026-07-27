"""Validation of the gold clawback on death (``gold_death_clawback``).

Collecting gold en route pays ``R_GOLD`` per piece. Without the clawback a
"grab-gold-then-mine-die" trajectory nets ``+gold - death_penalty``, making
risky gold detours look cheaper than they are. When ``gold_death_clawback > 0``
the episode-cumulative banked gold reward is subtracted from the terminal
reward ON DEATH ONLY, so dying erases the gold's value. This guards:

  1. clawback=0.0 (DEFAULT) leaves the legacy behavior unchanged: gold banked
     even on death.
  2. clawback=1.0 on a death removes 100% of the episode gold from the return
     (gold collected on the dying step -> terminal step == death-penalty-only;
     gold collected earlier -> the episode RETURN == death-penalty-only).
  3. A WIN always keeps the full gold regardless of the flag.
  4. Truncation always keeps the full gold regardless of the flag.
"""
from __future__ import annotations

from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_constants import R_GOLD

# Fixed, mine-free geometry so no HWM / PBRS / hazard term perturbs the terminal
# reward. Gold rewards ON; all optional shaping OFF to isolate the gold term.
N_GOLD = 3
GOLD_BANKED = N_GOLD * R_GOLD  # 3 * 0.24 = 0.72


def _cfg(clawback: float) -> RewardConfig:
    return RewardConfig(
        enable_gold_rewards=True,
        gold_death_clawback=clawback,
        enable_lattice_pbrs=False,
        enable_frontier_bonus=False,
        enable_landing_pbrs=False,
        enable_hazard_pbrs=False,
    )


def _obs(*, gold, dead=False, won=False, death_cause=None):
    o = {
        "player_x": 50.0,
        "player_y": 50.0,
        "switch_x": 200.0,
        "switch_y": 200.0,
        "exit_door_x": 300.0,
        "exit_door_y": 300.0,
        "switch_activated": False,
        "player_dead": dead,
        "player_won": won,
        "player_airborn": False,
        "player_walled": False,
        "gold_collected": gold,
        "total_gold": 10,  # never all-gold, so all_gold_bonus never fires
        "_cached_switch_distance": 400.0,
        "_cached_exit_distance": 400.0,
    }
    if death_cause is not None:
        o["death_cause"] = death_cause
    return o


def _death_only(cfg: RewardConfig) -> float:
    return cfg.death_penalty + cfg.mine_death_penalty  # -0.5 + -3.0 = -3.5


def test_gold_collected_on_death_step_clawed_back_to_death_only():
    """Gold picked up on the SAME step the ninja dies: terminal == death-only."""
    cfg = _cfg(clawback=1.0)
    rc = RewardCalculator(reward_config=cfg)
    rc.reset()
    prev = _obs(gold=0)
    # Single terminal step that both grabs the gold AND is a mine death.
    obs = _obs(gold=N_GOLD, dead=True, death_cause="mine")
    r = rc.calculate_reward(obs, prev, frames_executed=1, truncated=False)
    assert abs(r - _death_only(cfg)) < 1e-9, (
        f"clawback should erase the gold grabbed on the dying step: "
        f"got {r}, expected death-only {_death_only(cfg)}"
    )
    assert abs(rc.last_components["gold_clawback"] - (-GOLD_BANKED)) < 1e-9


def test_gold_collected_earlier_then_die_episode_return_is_death_only():
    """Gold banked on an earlier step then a mine death: EPISODE RETURN == death-only."""
    cfg = _cfg(clawback=1.0)
    rc = RewardCalculator(reward_config=cfg)
    rc.reset()
    prev = _obs(gold=0)
    ep_return = 0.0
    # Step 1: collect the gold (non-terminal).
    step1 = _obs(gold=N_GOLD)
    ep_return += rc.calculate_reward(step1, prev, frames_executed=1, truncated=False)
    # Step 2: mine death, no new gold this step.
    step2 = _obs(gold=N_GOLD, dead=True, death_cause="mine")
    ep_return += rc.calculate_reward(step2, step1, frames_executed=1, truncated=False)
    # death-only plus the non-gold shaping: the step-1 per-frame time penalty.
    expected = _death_only(cfg) + cfg.time_penalty_per_step * 1
    assert abs(ep_return - expected) < 1e-9, (
        f"episode return should net out the gold on death: got {ep_return}, "
        f"expected death-only + time penalty {expected}"
    )


def test_clawback_disabled_is_legacy_behavior():
    """clawback=0.0: gold stays banked even on death (legacy)."""
    cfg = _cfg(clawback=0.0)
    rc = RewardCalculator(reward_config=cfg)
    rc.reset()
    prev = _obs(gold=0)
    obs = _obs(gold=N_GOLD, dead=True, death_cause="mine")
    r = rc.calculate_reward(obs, prev, frames_executed=1, truncated=False)
    # Legacy: death penalty + full gold reward, no clawback.
    assert abs(r - (_death_only(cfg) + GOLD_BANKED)) < 1e-9, (
        f"legacy death reward must still bank the gold: got {r}, "
        f"expected {_death_only(cfg) + GOLD_BANKED}"
    )
    assert rc.last_components["gold_clawback"] == 0.0


def test_win_keeps_gold_even_with_clawback():
    """A WIN keeps the full gold regardless of gold_death_clawback."""
    cfg = _cfg(clawback=1.0)
    rc = RewardCalculator(reward_config=cfg)
    rc.reset()
    prev = _obs(gold=0)
    obs = _obs(gold=N_GOLD, won=True)
    r = rc.calculate_reward(obs, prev, frames_executed=1, truncated=False)
    expected = cfg.level_completion_reward + GOLD_BANKED  # speed bonus off by default
    assert abs(r - expected) < 1e-9, (
        f"win must keep gold: got {r}, expected {expected}"
    )


def test_truncation_keeps_gold_even_with_clawback():
    """A truncation (timeout) keeps the full gold regardless of the flag."""
    cfg = _cfg(clawback=1.0)
    rc = RewardCalculator(reward_config=cfg)
    rc.reset()
    prev = _obs(gold=0)
    obs = _obs(gold=N_GOLD)  # alive, not won
    r = rc.calculate_reward(obs, prev, frames_executed=1, truncated=True)
    # Non-terminal step: gold + time_penalty + truncation_penalty (no clawback).
    tp = cfg.time_penalty_per_step * 1
    expected = GOLD_BANKED + tp + cfg.truncation_penalty
    assert abs(r - expected) < 1e-6, (
        f"truncation must keep gold: got {r}, expected {expected}"
    )
