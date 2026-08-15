"""D1 timeout-geometry diagnostics: floor-mine span + kill-disk test.

``BaseNppEnvironment._floor_mine_span`` / ``._timeout_geometry_info`` answer "did
this timeout end on unmined catch floor?" -- the single claim most worth checking
before acting on the PLATFORMING/PLATFORM_STAIRS root-cause story in
``npp-rl/docs/RESEARCH_RESPONSE_generalization.md``.

The anchors below are the independently measured values from that document's
probes §7.1/§7.4, on PLATFORMING/T2 level ``131837852402414503``:

    57 floor mines, all at y = 480.0 exactly, spanning x = 264.0 .. 936.0
    a ninja at (685.1, 467.8) is 12.2 px from a mine -> DIED
    a ninja at (950.0, 470.0) is 17.2 px from a mine -> survived 5000 frames

14 px is the kill threshold (NINJA_RADIUS 10 + toggle-mine radius 4), so those two
points straddle it. Any change to the geometry helpers must keep reproducing the
lethal/safe verdict at both.
"""

from __future__ import annotations

from nclone.gym_environment.base_environment import BaseNppEnvironment


class _Sim:
    """Minimal stand-in exposing only what the geometry helpers read."""

    def __init__(self, mines):
        self._mines = mines

    def get_all_mine_data_for_visualization(self):
        return list(self._mines)


class _Loader:
    def __init__(self, name):
        self.current_map_name = name


class _Env:
    """A bare object with the two helpers bound to it.

    Deliberately not a real environment: the helpers are pure functions of the
    mine list and the ninja position, and constructing a full env here would test
    the simulator rather than the diagnostic.
    """

    _MINE_KILL_RADIUS_PX = BaseNppEnvironment._MINE_KILL_RADIUS_PX
    _floor_mine_span = BaseNppEnvironment._floor_mine_span
    _timeout_geometry_info = BaseNppEnvironment._timeout_geometry_info

    def __init__(self, mines, name="lvl"):
        self.nplay_headless = _Sim(mines)
        self.map_loader = _Loader(name)


def _platforming_t2_line():
    """The measured 131837852402414503 floor-mine line: 57 mines at y=480."""
    n = 57
    x0, x1 = 264.0, 936.0
    step = (x1 - x0) / (n - 1)
    return [{"x": x0 + i * step, "y": 480.0, "state": 0, "radius": 4.0}
            for i in range(n)]


def test_span_matches_the_measured_platforming_t2_line():
    env = _Env(_platforming_t2_line())
    span = env._floor_mine_span()
    assert span["y"] == 480.0
    assert span["x_min"] == 264.0
    assert span["x_max"] == 936.0
    assert span["n"] == 57.0


def test_lethal_point_inside_span_is_flagged_in_kill_disk():
    env = _Env(_platforming_t2_line())
    out = env._timeout_geometry_info(685.1, 467.8)
    assert out["timeout_inside_mine_span"] == 1.0
    assert out["timeout_outside_mine_span"] == 0.0
    assert out["timeout_nearest_mine_px"] <= 14.0
    assert out["timeout_in_kill_disk"] == 1.0


def test_survivable_point_outside_span_is_not_in_kill_disk():
    """The failure mode under investigation: standing safely past the line's end."""
    env = _Env(_platforming_t2_line())
    out = env._timeout_geometry_info(950.0, 470.0)
    assert out["timeout_outside_mine_span"] == 1.0
    assert out["timeout_px_outside_span"] == 14.0
    assert out["timeout_nearest_mine_px"] > 14.0
    assert out["timeout_in_kill_disk"] == 0.0
    # At floor height, so this is a catch-floor stall rather than platform dither.
    assert out["timeout_height_above_mine_row"] == 10.0


def test_height_above_mine_row_separates_platform_dithering():
    env = _Env(_platforming_t2_line())
    out = env._timeout_geometry_info(600.0, 280.0)
    assert out["timeout_height_above_mine_row"] == 200.0
    assert out["timeout_in_kill_disk"] == 0.0


def test_only_the_lowest_row_counts_as_floor_mines():
    """A mine floating mid-chamber must not widen the catch-floor span."""
    mines = _platforming_t2_line() + [{"x": 60.0, "y": 240.0, "state": 0,
                                       "radius": 4.0}]
    env = _Env(mines)
    span = env._floor_mine_span()
    assert span["x_min"] == 264.0  # not 60.0
    assert span["n"] == 57.0
    assert span["n_total"] == 58.0


def test_no_mines_yields_no_span_and_no_geometry_keys():
    """A mine-free level must report nothing, not a misleading zero span."""
    env = _Env([])
    assert env._floor_mine_span() is None
    out = env._timeout_geometry_info(100.0, 100.0)
    assert out == {"timeout_x": 100.0, "timeout_y": 100.0}


def test_span_cache_is_invalidated_when_the_level_changes():
    """The cache is keyed on the map name; a new level must not inherit the old
    level's mine line (which would silently mis-attribute every timeout)."""
    env = _Env(_platforming_t2_line(), name="a")
    assert env._floor_mine_span()["x_min"] == 264.0
    env.nplay_headless = _Sim([{"x": 24.0, "y": 480.0, "state": 0, "radius": 4.0}])
    env.map_loader = _Loader("b")
    assert env._floor_mine_span()["x_min"] == 24.0
