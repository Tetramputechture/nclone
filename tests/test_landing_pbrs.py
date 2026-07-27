"""Grounded-landing high-water-mark bonus: pay a one-sided, D0-normalized bonus
only on a strictly-new closest-ever *grounded* contact with the current live
objective -- i.e. safely landing (or standing) on a surface closer to the goal
than any surface touched this episode.

See RewardConfig.enable_landing_pbrs. This is the frontier bonus gated on solid
ground: the per-step free-space/jump PBRS already rewards airborne approach
(weighting a mid-air drift toward a gap the same as a completed jump), so this
term specifically up-weights STICKING the landing on precision buckets
(single-tile platforms, mines under every gap, zero recovery). Because landing on
a mine is death (not a surface contact), it rewards only SAFE landings. Monotone
HWM (never re-paid) so it cannot be farmed by oscillating; rebased on the
switch->exit flip so the basis change is not paid; a no-op when off.

The bonus for a strict grounded improvement is
``landing_pbrs_weight * (best_prev - now) / D0`` where D0 is the spawn
cost-to-completion (``_pbrs_d0``), matching the frontier bonus normalization.
In these tests the first step seeds D0 = initial dist_switch = 100.

TRANSITION-GATED: the bonus fires only on the airborne->contact EDGE (an actual
landing or wall-catch this step), never while the ninja is already standing on
the ground. Continuous ground running is not a landing and earns nothing. Every
test below therefore drives an explicit airborne step before each contact it
expects to pay -- driving only grounded steps exercises nothing and (before this
was corrected) made the whole suite assert against semantics the calculator had
deliberately dropped.
"""

from unittest.mock import patch

from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig


def _obs(airborn=False, walled=False, **kwargs):
    base = {
        "player_x": 0.0,
        "player_y": 0.0,
        "player_dead": False,
        "player_won": False,
        "player_airborn": airborn,
        "player_walled": walled,
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


def _calc(enable_landing, weight=0.5):
    cfg = RewardConfig(
        _time_penalty_per_step_override=0.0,
        _hwm_alpha_override=0.0,
        enable_landing_pbrs=enable_landing,
        landing_pbrs_weight=weight,
    )
    calc = RewardCalculator(reward_config=cfg)
    calc.reset()
    return calc


def _step(calc, dist_switch, dist_exit, airborn=False, walled=False,
          switch_activated=False):
    """Drive one env step with mocked A* distances and grounded/wall flags;
    return the isolated landing-bonus component for that step."""
    cur = _obs(airborn=airborn, walled=walled, switch_activated=switch_activated)
    prev = _obs(airborn=airborn, walled=walled, switch_activated=switch_activated)
    with patch.object(
        RewardCalculator, "_get_distances", return_value=(dist_switch, dist_exit)
    ):
        calc.calculate_reward(cur, prev, frames_executed=1)
    return calc.last_components.get("landing_bonus", 0.0)


def _land(calc, dist_switch, dist_exit, switch_activated=False, air_dist=None):
    """Fly, then land: the airborne->grounded EDGE the bonus is gated on.

    The airborne leg deliberately does not move the watermark, so the returned
    value is the bonus for the landing alone.
    """
    _step(calc, air_dist if air_dist is not None else dist_switch, dist_exit,
          airborn=True, switch_activated=switch_activated)
    return _step(calc, dist_switch, dist_exit, airborn=False,
                 switch_activated=switch_activated)


# D0 == first step's dist_switch (100) in every test below.
_D0 = 100.0


# ---------------------------------------------------------------------------
# (a) Flag OFF -> no landing bonus, ever (even grounded + improving).
# ---------------------------------------------------------------------------


def test_flag_off_pays_no_landing_bonus():
    calc = _calc(enable_landing=False)
    assert abs(_step(calc, 100.0, 200.0)) < 1e-9
    assert abs(_land(calc, 80.0, 200.0)) < 1e-9  # landing improvement, still 0
    assert abs(_land(calc, 50.0, 200.0)) < 1e-9


# ---------------------------------------------------------------------------
# (b) Flag ON, GROUNDED -> strict-improvement bonus; non-improving pays 0.
# ---------------------------------------------------------------------------


def test_first_grounded_step_rebases_no_bonus():
    calc = _calc(enable_landing=True)
    assert abs(_land(calc, 100.0, 200.0)) < 1e-9  # seeds watermark, no bonus


def test_continuous_ground_running_never_pays():
    """The transition gate: without a landing edge there is no landing."""
    calc = _calc(enable_landing=True)
    _land(calc, 100.0, 200.0)  # seed via a real landing
    assert abs(_step(calc, 70.0, 200.0)) < 1e-9, (
        "running along the ground toward the goal is not a landing"
    )
    assert abs(_step(calc, 40.0, 200.0)) < 1e-9


def test_grounded_strict_improvement_pays_normalized_bonus():
    w = 0.5
    calc = _calc(enable_landing=True, weight=w)
    _land(calc, 100.0, 200.0)  # seed at 100 (D0=100)
    r = _land(calc, 70.0, 200.0)  # land 100->70
    assert abs(r - w * (100.0 - 70.0) / _D0) < 1e-9
    r2 = _land(calc, 55.0, 200.0)  # land 70->55 (vs new best)
    assert abs(r2 - w * (70.0 - 55.0) / _D0) < 1e-9


def test_non_improving_grounded_step_pays_zero_and_not_farmable():
    calc = _calc(enable_landing=True)
    _land(calc, 100.0, 200.0)  # seed
    assert _land(calc, 70.0, 200.0) > 0.0  # improve -> bonus
    # Re-landing cannot farm: equal, then worse, then back -- all pay 0.
    assert abs(_land(calc, 70.0, 200.0)) < 1e-9  # equal to best
    assert abs(_land(calc, 90.0, 200.0)) < 1e-9  # worse (fell back), watermark held
    assert abs(_land(calc, 70.0, 200.0)) < 1e-9  # returned, still not below 70
    assert _land(calc, 60.0, 200.0) > 0.0  # only a strictly-new closest pays again


# ---------------------------------------------------------------------------
# (c) AIRBORNE -> never pays, and does NOT advance the watermark. Mid-air
#     proximity to the goal earns nothing; only the safe landing does.
# ---------------------------------------------------------------------------


def test_airborne_pays_nothing_and_does_not_move_watermark():
    w = 0.5
    calc = _calc(enable_landing=True, weight=w)
    _land(calc, 100.0, 200.0)  # grounded seed at 100
    # Fly closer through the air (would be a big frontier gain) -> no landing bonus.
    assert abs(_step(calc, 40.0, 200.0, airborn=True)) < 1e-9
    assert abs(_step(calc, 30.0, 200.0, airborn=True)) < 1e-9
    # Land at 60 (worse than the 40 we flew over, but the watermark never moved off
    # 100 while airborne) -> pays the full grounded improvement 100->60.
    r_land = _step(calc, 60.0, 200.0, airborn=False)
    assert abs(r_land - w * (100.0 - 60.0) / _D0) < 1e-9


# ---------------------------------------------------------------------------
# (d) Robustness: inf cost is safe; switch->exit flip does not pay a spurious
#     bonus (rebased on the objective change, like the frontier bonus).
# ---------------------------------------------------------------------------


def test_inf_cost_is_safe_no_bonus_no_crash():
    calc = _calc(enable_landing=True)
    assert abs(_land(calc, float("inf"), 200.0)) < 1e-9  # unreachable -> no bonus
    assert abs(_land(calc, 80.0, 200.0)) < 1e-9  # first finite landing -> seeds
    assert _land(calc, 50.0, 200.0) > 0.0  # later landing improvement pays


def test_switch_to_exit_transition_rebases_no_spurious_bonus():
    w = 0.5
    calc = _calc(enable_landing=True, weight=w)
    _land(calc, 100.0, 5.0, switch_activated=False)  # seed switch watermark at 100
    assert _land(calc, 40.0, 5.0, switch_activated=False) > 0.0  # approach pays
    # Switch activates: live objective flips to the exit (dist 5 << 40). Must rebase
    # to the exit basis on the next landing, paying 0 (no spurious bonus).
    r_trans = _land(calc, 40.0, 5.0, switch_activated=True)
    assert abs(r_trans) < 1e-9
    # After the rebase, post-switch exit approach pays normally (5 -> 2).
    r_exit = _land(calc, 40.0, 2.0, switch_activated=True)
    assert abs(r_exit - w * (5.0 - 2.0) / _D0) < 1e-9


# ---------------------------------------------------------------------------
# (e) WALL-JUMP branch: a wall contact while MOVING TOWARD the goal (cost-to-go
#     strictly decreasing) counts as a safe progress contact -- even airborne
#     (wall-slide) -- so wall-jump ascent is rewarded. A wall contact NOT
#     progressing pays nothing.
# ---------------------------------------------------------------------------


def test_wall_contact_while_progressing_pays():
    w = 0.5
    calc = _calc(enable_landing=True, weight=w)
    _land(calc, 100.0, 200.0)  # grounded spawn seed at 100 (D0=100), prev=100
    # Jump off: the wall-catch below is also a CONTACT EDGE, so it needs a
    # preceding airborne frame. Holding cost-to-go at 100 keeps the improvement
    # under test the wall-slide's own 100->70.
    _step(calc, 100.0, 200.0, airborn=True, walled=False)
    # Wall-slide (airborne + walled) at 70, cost-to-go decreased 100->70 = moving
    # toward the goal -> wall progress contact -> pays the HWM improvement.
    r = _step(calc, 70.0, 200.0, airborn=True, walled=True)
    assert abs(r - w * (100.0 - 70.0) / _D0) < 1e-9
    # Continue up the wall to 50 (still progressing) -> pays vs the new best.
    r2 = _step(calc, 50.0, 200.0, airborn=True, walled=True)
    assert abs(r2 - w * (70.0 - 50.0) / _D0) < 1e-9


def test_wall_contact_not_progressing_pays_zero():
    calc = _calc(enable_landing=True)
    _land(calc, 100.0, 200.0)  # grounded seed at 100
    # Approach to 50 through the air (not walled) -> no contact, best stays 100,
    # prev becomes 50.
    assert abs(_step(calc, 50.0, 200.0, airborn=True, walled=False)) < 1e-9
    # Now touch a wall at the SAME cost-to-go (50): not strictly decreasing this
    # step -> velocity is NOT toward the goal path -> no bonus, even though 50 < 100.
    assert abs(_step(calc, 50.0, 200.0, airborn=True, walled=True)) < 1e-9
    # Push up the wall (48 < 50 = progressing) -> now it pays.
    assert _step(calc, 48.0, 200.0, airborn=True, walled=True) > 0.0


def test_wall_branch_off_when_not_walled_even_if_progressing():
    calc = _calc(enable_landing=True)
    _step(calc, 100.0, 200.0)  # grounded seed
    # Airborne, NOT walled, moving toward goal (80<100): not a safe contact
    # (not grounded, not walled) -> no bonus. Pure mid-air approach earns nothing.
    assert abs(_step(calc, 80.0, 200.0, airborn=True, walled=False)) < 1e-9
