"""Regression: `_prev_goal_dist` must hold the PREVIOUS step's cost-to-completion
for every consumer inside `calculate_reward`, and `components_sum` must account
for every term that reaches the returned reward.

Pipeline audit P0.3 and P0.4. Two writes advanced `_prev_goal_dist` mid-step:

  1. the lattice-PBRS branch wrote it right after computing the potential, i.e.
     BEFORE momentum shaping, the hazard progress gate, the landing bonus and the
     backtrack counter ever read it;
  2. `_momentum_shaping` advanced it again on its way out.

Every one of those consumers tests `current_goal_dist` against `_prev_goal_dist`,
so with either write in place the comparison is `x` against `x` and the feature
is silently inert. `calculate_reward` already advances the field exactly once, at
the end of the step; these tests pin that it is the ONLY writer.

`_last_cost_to_go_raw` had the mirror-image bug: it was assigned only inside the
lattice-PBRS branch, so on the free-space/jump potential path -- the one the runs
actually take -- it never left `inf` and momentum read `inf` forever.
"""
from __future__ import annotations

from nclone.gym_environment.constants import GAME_STATE_NEAREST_MINE_IDX
from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig

MOM_W = 0.3
HAZ_W = 0.5


def _obs(cost, *, d_mine=None, airborne=False, walled=False):
    o = {
        "player_x": 50.0,
        "player_y": 50.0,
        "switch_x": 200.0,
        "switch_y": 200.0,
        "exit_door_x": 300.0,
        "exit_door_y": 300.0,
        "switch_activated": False,
        "player_dead": False,
        "player_won": False,
        "player_airborn": airborne,
        "player_walled": walled,
        "_cached_switch_distance": float(cost),
        "_cached_exit_distance": float(cost),
    }
    if d_mine is not None:
        gs = [0.0] * (GAME_STATE_NEAREST_MINE_IDX + 1)
        gs[GAME_STATE_NEAREST_MINE_IDX] = float(d_mine)
        o["game_state"] = gs
    return o


def _drive(rc, obs_seq, frames=1):
    """Feed a sequence of observations through the calculator."""
    prev = obs_seq[0]
    for obs in obs_seq:
        rc.calculate_reward(obs, prev, frames_executed=frames, truncated=False)
        prev = obs


# --- P0.4: momentum shaping is live -----------------------------------------

def _momentum_total(*, lattice: bool) -> float:
    rc = RewardCalculator(
        reward_config=RewardConfig(
            enable_lattice_pbrs=lattice,
            enable_momentum_pbrs=True,
            momentum_pbrs_weight=MOM_W,
            enable_frontier_bonus=False,
            enable_landing_pbrs=False,
        )
    )
    rc.reset()
    # Strictly decreasing cost-to-go: the ninja is closing on the goal every step.
    _drive(rc, [_obs(c) for c in (500.0, 400.0, 300.0, 200.0, 100.0)])
    return rc.episode_momentum_pbrs_reward


def test_momentum_nonzero_on_closing_trajectory_with_lattice_pbrs():
    """The audit's stated refutation criterion for P0.4."""
    assert abs(_momentum_total(lattice=True)) > 0.0


def test_momentum_nonzero_on_closing_trajectory_without_lattice_pbrs():
    """The configuration the runs actually use takes the free-space potential
    path, so `enable_lattice_pbrs` is falsy. Momentum was hoisted out of that
    branch to make it live -- but `_last_cost_to_go_raw` was left behind in it,
    so momentum still read `inf` and paid exactly nothing."""
    assert abs(_momentum_total(lattice=False)) > 0.0


def test_momentum_sign_follows_closing_speed():
    """Closing on the goal must score above retreating from it."""
    closing = _momentum_total(lattice=True)
    rc = RewardCalculator(
        reward_config=RewardConfig(
            enable_lattice_pbrs=True,
            enable_momentum_pbrs=True,
            momentum_pbrs_weight=MOM_W,
            enable_frontier_bonus=False,
            enable_landing_pbrs=False,
        )
    )
    rc.reset()
    _drive(rc, [_obs(c) for c in (100.0, 200.0, 300.0, 400.0, 500.0)])
    assert closing > rc.episode_momentum_pbrs_reward


# --- P0.4: the hazard progress gate is reachable ------------------------------

def _hazard_total(*, gated: bool) -> float:
    rc = RewardCalculator(
        reward_config=RewardConfig(
            enable_lattice_pbrs=True,
            enable_hazard_pbrs=True,
            hazard_pbrs_weight=HAZ_W,
            hazard_pbrs_one_sided=True,
            hazard_pbrs_progress_gated=gated,
            enable_frontier_bonus=False,
            enable_landing_pbrs=False,
        )
    )
    rc.reset()
    # Closing on the goal (500 -> 100) while the nearest mine gets closer too:
    # a NECESSARY mine-adjacent approach, which the gate exists to exempt.
    _drive(
        rc,
        [
            _obs(c, d_mine=d)
            for c, d in zip(
                (500.0, 400.0, 300.0, 200.0, 100.0),
                (1.0, 0.8, 0.6, 0.4, 0.2),
            )
        ],
    )
    return rc.episode_hazard_pbrs_reward


def test_hazard_progress_gate_exempts_a_productive_approach():
    """Ungated, closing on a mine while closing on the goal is taxed. Gated, it
    is free. The premature `_prev_goal_dist` write made the two identical, so the
    progress gate -- validated in v39 and shipped ON -- never fired once."""
    ungated = _hazard_total(gated=False)
    gated = _hazard_total(gated=True)
    assert ungated < 0.0, "a one-sided hazard approach must cost something ungated"
    assert gated == 0.0, f"the gate must exempt a productive approach, got {gated}"


# --- P0.4: backtrack telemetry ------------------------------------------------

def test_backtrack_steps_counted_under_lattice_pbrs():
    """`regressed` compares against `_prev_goal_dist`, so the premature write
    pinned `backtrack_steps` at 0 for every lattice-PBRS episode."""
    rc = RewardCalculator(
        reward_config=RewardConfig(
            enable_lattice_pbrs=True,
            enable_frontier_bonus=False,
            enable_landing_pbrs=False,
        )
    )
    rc.reset()
    _drive(rc, [_obs(c) for c in (100.0, 200.0, 300.0, 400.0)])
    assert rc.episode_backtrack_steps == 3


# --- P0.3: the reconciliation identity ---------------------------------------

def test_reconciliation_residual_zero_with_momentum_active():
    """`reconciliation_residual` is the completeness proof for the reward
    decomposition audit. It omitted the momentum, pace and new-record terms, so
    it read non-zero -- by exactly the omitted amount -- whenever they were on."""
    rc = RewardCalculator(
        reward_config=RewardConfig(
            enable_lattice_pbrs=True,
            enable_momentum_pbrs=True,
            momentum_pbrs_weight=MOM_W,
            enable_hazard_pbrs=True,
            hazard_pbrs_weight=HAZ_W,
            enable_frontier_bonus=False,
            enable_landing_pbrs=False,
        )
    )
    rc.reset()
    _drive(
        rc,
        [
            _obs(c, d_mine=d)
            for c, d in zip(
                (500.0, 400.0, 450.0, 300.0, 100.0),
                (1.0, 0.7, 0.9, 0.5, 0.3),
            )
        ],
    )
    m = rc.get_episode_reward_metrics()
    assert abs(m["momentum_pbrs_total"]) > 0.0, "test must exercise the omitted term"
    assert abs(m["reconciliation_residual"]) < 1e-9, (
        f"residual {m['reconciliation_residual']} != 0; components_sum "
        f"{m['components_sum']} vs return {m['episode_return_internal']}"
    )
