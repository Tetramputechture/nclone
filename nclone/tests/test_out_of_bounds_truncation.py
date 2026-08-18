"""The escaped-ninja guard must fire, end the episode, and stay observable.

A gamma-3 run logged 31,837 `ninja_pos ... outside valid level bounds` warnings
with nothing terminating on the condition, so those envs burned a full episode
each while being handed ZERO reachability features. The guard now truncates and
counts; these tests pin the bounds logic and the reporting contract, because the
counter previously existed and was read by nothing.
"""

import pytest

from nclone.gym_environment.base_environment import BaseNppEnvironment


class _Headless:
    def __init__(self, position):
        self._position = position

    def ninja_position(self):
        if self._position == "raise":
            raise RuntimeError("headless read failed")
        return self._position


def _env(position):
    env = object.__new__(BaseNppEnvironment)
    env.nplay_headless = _Headless(position)
    return env


@pytest.mark.parametrize(
    "position",
    [(0.0, 0.0), (1056.0, 600.0), (528.0, 300.0)],
)
def test_in_bounds_positions_never_trip_the_guard(position):
    # The border ring is solid tile type 1 on all four sides, so legal play can
    # never reach this. A false positive would end healthy episodes.
    assert _env(position)._ninja_out_of_bounds() is False


@pytest.mark.parametrize(
    "position",
    [(452.0, 28383.0), (3193.0, 37877.0), (-1.0, 300.0), (528.0, -1.0)],
)
def test_escaped_positions_trip_the_guard(position):
    # The first two are the positions actually observed falling at terminal
    # velocity for thousands of frames in gamma-3.
    assert _env(position)._ninja_out_of_bounds() is True


def test_a_failed_position_read_does_not_end_a_healthy_episode():
    assert _env("raise")._ninja_out_of_bounds() is False


def test_episode_info_reports_the_cause_and_clears_it():
    """`ended` is per-episode; `total` is the env's lifetime count.

    Without the clear, every subsequent episode on that env would report itself
    as out-of-bounds and the rate would read a constant 1.0.
    """
    env = _env((452.0, 28383.0))
    env._oob_truncations = 3
    env._episode_ended_out_of_bounds = True

    first = {}
    env._report_out_of_bounds(first)
    assert first == {
        "episode_ended_out_of_bounds": True,
        "oob_truncations_total": 3,
    }

    second = {}
    env._report_out_of_bounds(second)
    assert second == {
        "episode_ended_out_of_bounds": False,
        "oob_truncations_total": 3,
    }


def test_build_episode_info_actually_calls_the_reporter():
    """Guards the wiring, not just the helper.

    The previous counter was incremented in `step` and read by nothing in
    either repo, so a helper that is correct but uncalled reproduces exactly
    the defect this file exists to prevent.
    """
    import inspect

    source = inspect.getsource(BaseNppEnvironment._build_episode_info)
    assert "_report_out_of_bounds(info)" in source
