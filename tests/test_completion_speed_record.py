"""Record-relative completion-speed bonus (single-level speedrun).

The par/budget forms of the completion-speed bonus both CLAMP at 1.0, so once a
policy is at or under par the time gradient vanishes. That is deliberate for
generalist training (no pull toward risky sub-par shortcuts) but fatal when the
objective is to grind a single level toward its theoretical minimum frame count:
there is nothing left to optimize against.

``completion_speed_record_norm`` references the bonus to the fastest win recorded
so far on the level instead, so beating your own best always pays and the target
ratchets down. These tests lock in the properties that make it safe to ship:

- the record is a genuine ratchet (monotone non-increasing),
- faster wins pay strictly more, so the gradient never saturates,
- the bonus stays bounded below R_EXIT, preserving "a slow win beats dying",
- it remains terminal-only and win-gated (never paid on death/timeout).
"""

from unittest.mock import patch

import pytest

from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.reward_calculation.reward_constants import R_EXIT
from nclone.gym_environment.constants import (
    GAME_STATE_NEAREST_MINE_IDX,
    NINJA_STATE_DIM,
)

_GS_LEN = GAME_STATE_NEAREST_MINE_IDX + 4
_BONUS = 3.0


def _game_state(time_remaining=1.0):
    gs = [0.0] * _GS_LEN
    gs[NINJA_STATE_DIM - 1] = float(time_remaining)
    gs[GAME_STATE_NEAREST_MINE_IDX] = 1.0
    return gs


def _obs(**kwargs):
    base = {
        "player_x": 0.0,
        "player_y": 0.0,
        "player_dead": False,
        "player_won": False,
        "player_airborn": False,
        "player_walled": False,
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
        "game_state": _game_state(),
        "_impact_risk": 0.0,
        "waypoint_leg_distance": 0.0,
        "waypoint_index": 0,
    }
    base.update(kwargs)
    return base


def _calc(record_norm=True, cap=1.5):
    cfg = RewardConfig(
        completion_speed_bonus=_BONUS,
        completion_speed_record_norm=record_norm,
        completion_speed_record_cap=cap,
    )
    calc = RewardCalculator(reward_config=cfg)
    calc.reset()
    return calc


def _win(calc, frames):
    """Finish the level in ``frames`` sim frames; return the speed bonus paid."""
    with patch.object(RewardCalculator, "_get_distances", return_value=(1.0, 1.0)):
        calc.calculate_reward(
            _obs(player_won=True, sim_frame=float(frames)),
            _obs(),
            frames_executed=1,
        )
    return calc.episode_speed_bonus_reward


def test_first_win_seeds_the_record_at_base_rate():
    calc = _calc()
    assert _win(calc, 500) == pytest.approx(_BONUS)
    assert calc._best_completion_frames == pytest.approx(500.0)


def test_beating_the_record_pays_more_than_matching_it():
    calc = _calc()
    _win(calc, 500)              # seed the record
    tie = _win(calc, 500)        # equal time -> 1.0x
    calc2 = _calc()
    _win(calc2, 500)
    faster = _win(calc2, 400)    # 20% faster -> 500/400 = 1.25x

    assert tie == pytest.approx(_BONUS)
    assert faster > tie
    assert faster == pytest.approx(_BONUS * 1.25)


def test_record_is_a_ratchet_and_gradient_never_saturates():
    """Each improvement still pays >1.0x, however fast the policy already is."""
    calc = _calc()
    _win(calc, 800)
    bonuses = []
    for frames in (700, 600, 500, 400, 300):
        bonuses.append(_win(calc, frames))
        assert calc._best_completion_frames == pytest.approx(float(frames))

    # Every step of the ladder is strictly better than merely tying, i.e. there
    # is still a positive incentive to go faster at 300 frames just as at 700.
    # This is the property the par form loses the moment it clamps.
    assert all(b > _BONUS for b in bonuses)


def test_slower_win_pays_less_but_stays_positive():
    calc = _calc()
    _win(calc, 400)
    slow = _win(calc, 800)  # half speed -> 0.5x
    assert slow == pytest.approx(_BONUS * 0.5)
    assert slow > 0.0
    # A slower run must not move the record backwards.
    assert calc._best_completion_frames == pytest.approx(400.0)


def test_bonus_stays_below_r_exit_so_a_slow_win_still_beats_dying():
    """The cap is what keeps the reward hierarchy intact."""
    calc = _calc(cap=1.5)
    _win(calc, 1000)
    absurdly_fast = _win(calc, 1)  # ratio 1000x, must clamp to the cap
    assert absurdly_fast == pytest.approx(_BONUS * 1.5)
    assert absurdly_fast < R_EXIT


def test_not_paid_on_death_or_timeout():
    calc = _calc()
    with patch.object(RewardCalculator, "_get_distances", return_value=(1.0, 1.0)):
        calc.calculate_reward(
            _obs(player_dead=True, sim_frame=10.0), _obs(), frames_executed=1
        )
    assert calc.episode_speed_bonus_reward == 0.0
    assert calc._best_completion_frames is None

    calc2 = _calc()
    with patch.object(RewardCalculator, "_get_distances", return_value=(1.0, 1.0)):
        calc2.calculate_reward(
            _obs(sim_frame=10.0), _obs(), frames_executed=1, truncated=True
        )
    assert calc2.episode_speed_bonus_reward == 0.0
    assert calc2._best_completion_frames is None


def test_record_survives_episode_reset():
    """The record is a cross-episode personal best, not per-episode state."""
    calc = _calc()
    _win(calc, 600)
    calc.reset()
    assert calc._best_completion_frames == pytest.approx(600.0)
    assert _win(calc, 300) == pytest.approx(_BONUS * 1.5)  # capped 2.0x -> 1.5


def test_record_norm_off_preserves_legacy_clamp():
    """With the flag off, beating par cannot exceed the base bonus."""
    calc = _calc(record_norm=False)
    first = _win(calc, 500)
    second = _win(calc, 10)
    assert first == pytest.approx(_BONUS)
    assert second == pytest.approx(_BONUS)  # clamped at 1.0x, no extra credit


def test_checkpoint_started_wins_cannot_set_the_record():
    """A Go-Explore/demo restore starts MID-trajectory, so its completion time is
    a demo prefix + an agent suffix. Letting it set the record would report a
    time the agent cannot reproduce from spawn -- a false result against the
    run's own target."""
    calc = _calc()
    _win(calc, 600)                      # genuine spawn run seeds the record
    assert calc._best_completion_frames == pytest.approx(600.0)

    # A restored episode "finishes" at 470 -- must NOT move the high-water mark.
    with patch.object(RewardCalculator, "_get_distances", return_value=(1.0, 1.0)):
        calc.calculate_reward(
            _obs(player_won=True, sim_frame=470.0, from_checkpoint=True),
            _obs(), frames_executed=1,
        )
    assert calc._best_completion_frames == pytest.approx(600.0)
    # ...but it DOES still earn the bonus, which is what teaches the fast finish.
    assert calc.episode_speed_bonus_reward > 0.0


def test_genuine_spawn_win_still_sets_the_record():
    calc = _calc()
    _win(calc, 600)
    _win(calc, 500)  # from_checkpoint absent/False
    assert calc._best_completion_frames == pytest.approx(500.0)


def _calc_rb(bonus=2.0):
    cfg = RewardConfig(
        completion_speed_bonus=_BONUS,
        completion_speed_record_norm=True,
        completion_speed_record_cap=1.5,
        completion_speed_record_bonus=bonus,
    )
    c = RewardCalculator(reward_config=cfg)
    c.reset()
    return c


def _win_rb(calc, frames, from_checkpoint=False):
    with patch.object(RewardCalculator, "_get_distances", return_value=(1.0, 1.0)):
        calc.calculate_reward(
            _obs(player_won=True, sim_frame=float(frames),
                 from_checkpoint=from_checkpoint),
            _obs(), frames_executed=1,
        )
    return calc.episode_new_record_reward


def test_new_record_bonus_pays_only_on_a_strict_improvement():
    """The ratio form is nearly flat near the record (479/465 = 1.03), so a
    14-frame gain was worth ~1.9% of episode return. This makes it visible."""
    c = _calc_rb(2.0)
    assert _win_rb(c, 500) == 0.0        # first win only seeds the record
    assert _win_rb(c, 490) == 2.0        # strict improvement -> paid
    assert _win_rb(c, 495) == 0.0        # slower -> nothing
    assert _win_rb(c, 490) == 0.0        # equalling the record is not beating it


def test_new_record_bonus_is_unfarmable_because_the_bar_ratchets():
    """Each claim raises the bar, so repeating the SAME time never pays twice."""
    c = _calc_rb(2.0)
    _win_rb(c, 500)
    total = sum(_win_rb(c, 480) for _ in range(10))
    assert total == 2.0                  # exactly one payment, not ten
    assert c._best_completion_frames == pytest.approx(480.0)


def test_checkpoint_started_run_cannot_claim_the_record_bonus():
    """A restored episode is a demo prefix + agent suffix; it may not claim a
    record it did not earn from spawn."""
    c = _calc_rb(2.0)
    _win_rb(c, 500)
    assert _win_rb(c, 400, from_checkpoint=True) == 0.0
    assert c._best_completion_frames == pytest.approx(500.0)


def test_record_bonus_off_by_default():
    c = _calc()                          # no completion_speed_record_bonus
    _win(c, 500)
    _win(c, 480)
    assert getattr(c, "episode_new_record_reward", 0.0) == 0.0
