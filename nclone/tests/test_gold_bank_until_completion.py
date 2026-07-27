"""Validation of completion-primary gold banking (``gold_bank_until_completion``).

Measured on the v24 scratch run (18M -> 21M steps): deterministic completion fell
18.8% -> 16.1% while gold/solve rose 7.0 -> 17.4 and timeouts nearly doubled. The
gold *death* clawback did not prevent that, because the losing trade was
"detour for gold, then TIME OUT" — truncation kept the full gold.

Banking closes it: the pickup reward is accumulated but pays nothing until the
episode COMPLETES. Guards:

  1. Gold picked up mid-episode pays 0 at the time of pickup.
  2. A death pays 0 gold (and no spurious clawback penalty on top).
  3. A truncation pays 0 gold — the case the clawback missed.
  4. A WIN pays the entire episode's banked gold in one lump, including gold
     grabbed on the winning step itself.
  5. Banking off = byte-identical legacy behavior.
  6. The episode reconciliation stays exact (components_sum == actual return)
     under banking AND under the death clawback.
"""

from __future__ import annotations

from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_constants import (
    R_GOLD,
    GLOBAL_REWARD_SCALE,
)

N_GOLD = 3
GOLD_BANKED = N_GOLD * R_GOLD


def _cfg(bank: bool, clawback: float = 0.0) -> RewardConfig:
    return RewardConfig(
        enable_gold_rewards=True,
        gold_bank_until_completion=bank,
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
    return cfg.death_penalty + cfg.mine_death_penalty


def test_pickup_pays_nothing_while_banking():
    cfg = _cfg(bank=True)
    rc = RewardCalculator(reward_config=cfg)
    rc.reset()
    prev = _obs(gold=0)
    r = rc.calculate_reward(_obs(gold=N_GOLD), prev, frames_executed=1, truncated=False)
    # Only the per-frame time penalty; the gold is banked, not paid.
    assert abs(r - cfg.time_penalty_per_step) < 1e-9, (
        f"banked gold must pay 0 at pickup: got {r}"
    )
    assert rc.last_components["gold_reward"] == 0.0
    assert abs(rc.last_components["gold_pickup_reward"] - GOLD_BANKED) < 1e-9
    # Still tracked for telemetry.
    assert abs(rc.episode_gold_reward_total - GOLD_BANKED) < 1e-9
    assert rc.episode_gold_paid_total == 0.0


def test_death_pays_zero_gold_and_no_double_penalty():
    """Banking + clawback=1.0 must NOT net a gold PENALTY on death."""
    cfg = _cfg(bank=True, clawback=1.0)
    rc = RewardCalculator(reward_config=cfg)
    rc.reset()
    prev = _obs(gold=0)
    r = rc.calculate_reward(
        _obs(gold=N_GOLD, dead=True, death_cause="mine"),
        prev,
        frames_executed=1,
        truncated=False,
    )
    assert abs(r - _death_only(cfg)) < 1e-9, (
        f"death under banking is death-only: got {r}, expected {_death_only(cfg)}"
    )
    assert rc.last_components["gold_clawback"] == 0.0, (
        "clawback must be inert under banking (nothing was paid to claw back)"
    )


def test_truncation_pays_zero_gold():
    """The case the death clawback missed: gold detour -> timeout pays nothing."""
    cfg = _cfg(bank=True, clawback=1.0)
    rc = RewardCalculator(reward_config=cfg)
    rc.reset()
    prev = _obs(gold=0)
    r = rc.calculate_reward(_obs(gold=N_GOLD), prev, frames_executed=1, truncated=True)
    expected = cfg.time_penalty_per_step + cfg.truncation_penalty
    assert abs(r - expected) < 1e-6, (
        f"timeout under banking must pay 0 gold: got {r}, expected {expected}"
    )


def test_win_pays_the_whole_banked_episode():
    """Gold banked over several steps is paid in one lump on the win."""
    cfg = _cfg(bank=True)
    rc = RewardCalculator(reward_config=cfg)
    rc.reset()
    prev = _obs(gold=0)
    # Step 1: pick up 2 gold (pays nothing).
    step1 = _obs(gold=2)
    rc.calculate_reward(step1, prev, frames_executed=1, truncated=False)
    # Step 2: pick up the 3rd gold AND win on the same step.
    step2 = _obs(gold=N_GOLD, won=True)
    r = rc.calculate_reward(step2, step1, frames_executed=1, truncated=False)
    expected = cfg.level_completion_reward + GOLD_BANKED
    assert abs(r - expected) < 1e-9, (
        f"win must pay the full banked gold incl. the winning step's pickup: "
        f"got {r}, expected {expected}"
    )
    assert abs(rc.episode_gold_paid_total - GOLD_BANKED) < 1e-9


def test_banking_off_is_legacy():
    cfg = _cfg(bank=False)
    rc = RewardCalculator(reward_config=cfg)
    rc.reset()
    prev = _obs(gold=0)
    r = rc.calculate_reward(_obs(gold=N_GOLD), prev, frames_executed=1, truncated=False)
    assert abs(r - (GOLD_BANKED + cfg.time_penalty_per_step)) < 1e-9
    assert abs(rc.last_components["gold_reward"] - GOLD_BANKED) < 1e-9


def _reconciles(cfg: RewardConfig, *, dead: bool, won: bool) -> None:
    rc = RewardCalculator(reward_config=cfg)
    rc.reset()
    prev = _obs(gold=0)
    actual = 0.0
    step1 = _obs(gold=N_GOLD)
    actual += rc.calculate_reward(step1, prev, frames_executed=1, truncated=False)
    step2 = _obs(
        gold=N_GOLD, dead=dead, won=won, death_cause="mine" if dead else None
    )
    actual += rc.calculate_reward(step2, step1, frames_executed=1, truncated=False)
    summary = rc.get_episode_reward_metrics()
    assert abs(summary["reconciliation_residual"]) < 1e-6, (
        f"component decomposition must be exact (bank={cfg.gold_bank_until_completion}, "
        f"clawback={cfg.gold_death_clawback}, dead={dead}, won={won}): "
        f"residual {summary['reconciliation_residual']}"
    )
    assert abs(summary["episode_return_internal"] - actual) < 1e-9


def test_reconciliation_exact_under_banking_and_clawback():
    """The residual is the canary that no reward is unaccounted for."""
    for bank in (False, True):
        for clawback in (0.0, 1.0):
            for dead, won in ((True, False), (False, True), (False, False)):
                _reconciles(_cfg(bank=bank, clawback=clawback), dead=dead, won=won)


def test_scaled_units_unchanged():
    """Banking changes WHEN gold pays, not the scale it pays in."""
    assert GLOBAL_REWARD_SCALE > 0.0
    cfg = _cfg(bank=True)
    rc = RewardCalculator(reward_config=cfg)
    rc.reset()
    prev = _obs(gold=0)
    r = rc.calculate_reward(_obs(gold=N_GOLD, won=True), prev, frames_executed=1, truncated=False)
    assert abs(r - (cfg.level_completion_reward + GOLD_BANKED)) < 1e-9
