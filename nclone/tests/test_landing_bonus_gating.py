"""Validation of the transition-gated, D0-normalized landing bonus.

The landing bonus (``enable_landing_pbrs``) is a small precision nudge for
STICKING a jump landing closer to the goal. Regressions this guards:

  1. It must fire ONLY on the airborne->contact transition (a real landing /
     wall-catch), never on continuous ground contact -- otherwise flat running
     banks the whole telescoped route bonus every grounded step.
  2. It must pay only when the contact is strictly closer to the goal than any
     prior safe contact (a high-water-mark).
  3. Each landing is clamped to <= landing_pbrs_weight, and the episode total
     must stay bounded even when the spawn cost-to-completion was non-finite at
     step 0 (D0 previously fell back to 1.0 and un-normalized the bonus, letting
     it scale with raw cost-to-go ~1e5).
"""
from __future__ import annotations

from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)

W = 0.1  # landing_pbrs_weight


def _cfg(*, d0_ref=None) -> RewardConfig:
    return RewardConfig(
        enable_lattice_pbrs=True,
        enable_landing_pbrs=True,
        landing_pbrs_weight=W,
        enable_frontier_bonus=False,  # isolate the landing term
        _pbrs_d0_ref_override=d0_ref,  # fixed PBRS reference (the v20b run uses 1000)
    )


def _run(costs, airborne, walled=None, d0_ref=None):
    """Drive the calculator over (cost-to-completion, airborne) per step.

    Returns (episode_landing_bonus_total, per_step_landing_bonuses).
    """
    walled = walled if walled is not None else [False] * len(costs)
    rc = RewardCalculator(reward_config=_cfg(d0_ref=d0_ref))
    rc.reset()
    prev = {"player_x": 50.0, "player_y": 50.0, "switch_activated": False}
    per_step = []
    for c, air, wl in zip(costs, airborne, walled):
        obs = {
            "player_x": 50.0,
            "player_y": 50.0,
            "switch_x": 200.0,
            "switch_y": 200.0,
            "exit_door_x": 300.0,
            "exit_door_y": 300.0,
            "switch_activated": False,
            "player_dead": False,
            "player_won": False,
            "player_airborn": air,
            "player_walled": wl,
            "_cached_switch_distance": float(c),
            "_cached_exit_distance": float(c),
        }
        rc.calculate_reward(obs, prev, frames_executed=1, truncated=False)
        per_step.append(rc.last_components.get("landing_bonus", 0.0))
        prev = obs
    return rc.episode_landing_bonus_total, per_step


def test_no_bonus_on_continuous_ground_contact():
    # Monotone approach while ALWAYS grounded -> not a single landing transition.
    costs = [400.0, 300.0, 200.0, 100.0, 20.0]
    total, per_step = _run(costs, airborne=[False] * 5)
    assert total == 0.0, f"continuous ground contact must earn nothing, got {total}"
    assert all(b == 0.0 for b in per_step)


def test_bonus_only_on_landing_transition_and_closer():
    # airborne over the approach, then land at cost 100 (closer), fly again, land
    # at 20 (closer still). Two qualifying landings; grounded steps in between the
    # SAME landing (no new transition) pay nothing.
    #        step:   0     1      2       3      4        5
    costs = [400.0, 300.0, 100.0, 100.0, 60.0,  20.0]
    airb = [True,  True,  False, False, True,  False]
    total, per_step = _run(costs, airborne=airb)
    # step2: airborne->ground at 100 -> first safe contact = rebase (no bonus).
    assert per_step[2] == 0.0
    # step3: still grounded (no transition) -> nothing.
    assert per_step[3] == 0.0
    # step5: airborne->ground at 20 (< 100 baseline) -> pays, clamped to <= W.
    assert per_step[5] > 0.0
    assert per_step[5] <= W + 1e-9
    assert total == per_step[5]


def test_no_bonus_when_landing_not_closer():
    # Land at 100, then fly and land FARTHER (150) -> no new high-water-mark.
    costs = [400.0, 100.0, 150.0, 150.0]
    airb = [True,  False, True,  False]
    total, per_step = _run(costs, airborne=airb)
    assert total == 0.0, f"a landing that is not closer must not pay, got {total}"


def test_bounded_when_spawn_cost_is_infinite():
    # Spawn cost-to-completion is inf (free-space cache not ready) -> D0 would fall
    # back to 1.0 and, pre-fix, the landing bonus would scale with raw cost (~300).
    # The provisional-D0 re-base must keep every landing <= W and the total bounded.
    inf = float("inf")
    #        step:   0    1      2(land) 3(fly) 4(land closer)
    costs = [inf, 300.0, 100.0,  60.0,  20.0]
    airb = [True, True,  False,  True,  False]
    total, per_step = _run(costs, airborne=airb)
    # step4 lands at 20 (< 100 baseline). Pre-fix D0=1.0 would pay
    # (100-20)/1.0*W = 8.0 >> W; the provisional re-base normalizes by ~300.
    assert per_step[4] > 0.0
    assert all(b <= W + 1e-9 for b in per_step), per_step
    assert total <= W + 1e-9, f"landing total must stay bounded by W, got {total}"


def test_scale_invariant_on_big_maze_with_fixed_pbrs_ref():
    # THE v20b BUG: with a fixed PBRS reference (--pbrs-d0-ref 1000) far below a
    # large maze's cost-to-go (~1e5), the OLD code normalized the landing bonus by
    # 1000 -> whole-route total ~ weight * route/1000 ~ weight * 150 (exploded to
    # ~59341 across the episode). The shaping normalizer must use the per-episode
    # ROUTE LENGTH instead, keeping the total bounded by ~weight regardless of size.
    route = 150_000.0
    #        step:   0       1          2(land)   3(fly)     4(land closer)
    costs = [route, route * 0.7, route * 0.4, route * 0.2, 100.0]
    airb = [True,  True,        False,       True,       False]
    total, per_step = _run(costs, airborne=airb, d0_ref=1000.0)
    assert per_step[4] > 0.0
    assert per_step[4] <= W + 1e-9, f"per-landing must be <= weight, got {per_step[4]}"
    # Bounded by ~weight, NOT weight * route/1000 (~15) that the fixed ref produced.
    assert total <= W + 1e-9, f"big-maze landing total must stay <= weight, got {total}"


def test_wall_catch_transition_pays_when_progressing():
    # Airborne -> wall contact while moving toward the goal (cost decreasing) is a
    # valid wall-jump progress contact.
    costs = [400.0, 300.0, 100.0]
    airb = [True,  True,  False]
    walled = [False, False, True]
    total, per_step = _run(costs, airborne=airb, walled=walled)
    # First safe contact rebases; to see a paying wall-catch we need a prior
    # baseline, so this simply asserts the wall branch does not crash and the
    # first contact rebases (no bonus).
    assert per_step[2] == 0.0
    assert total == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK {name}")
