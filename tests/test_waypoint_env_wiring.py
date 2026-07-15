"""Env-side glue for the waypoint tracker (ReachabilityMixin).

Exercises the lightweight integration without spinning up the full NppEnvironment:
a stub carrying the mixin methods + a real ReachabilityConfig, fed injected
(precomputed) waypoints so no lattice build is needed. Verifies that
``_maybe_update_waypoints`` populates the reward/obs fields and advances the cursor.
"""

from types import SimpleNamespace

import numpy as np

from nclone.gym_environment.config import ReachabilityConfig
from nclone.gym_environment.mixins.reachability_mixin import ReachabilityMixin
from nclone.value_function.waypoint_planner import Waypoint


class _Stub(ReachabilityMixin):
    def __init__(self, reach_cfg):
        self.config = SimpleNamespace(reachability=reach_cfg)
        self.nplay_headless = None
        self._lattice_map_name = None
        # subset of _init_reachability_system relevant to waypoints
        self._waypoint_planner = None
        self._waypoint_tracker = None
        self._injected_waypoints = None
        self._cached_waypoint_nav = np.zeros(4, dtype=np.float32)
        self._waypoint_leg_distance = None
        self._waypoint_index = None


def _route():
    return [
        Waypoint(0.0, 0.0, 14.0, 0, 150.0, False),
        Waypoint(50.0, 0.0, 16.0, 0, 80.0, True),    # switch
        Waypoint(100.0, 0.0, 22.0, 1, 0.0, True),    # exit
    ]


def test_disabled_is_noop():
    s = _Stub(ReachabilityConfig())  # enable_waypoint_tracker defaults False
    s.set_level_waypoints(_route())
    s._maybe_update_waypoints((0.0, 0.0), False)
    assert s._waypoint_leg_distance is None
    assert s._waypoint_index is None


def test_injected_waypoints_populate_fields():
    s = _Stub(ReachabilityConfig(enable_waypoint_tracker=True))
    s.set_level_waypoints(_route())
    s._maybe_update_waypoints((0.0, 0.0), False)  # reach wp0 -> target wp1 (switch)
    assert s._waypoint_index == 1
    assert s._waypoint_leg_distance == 50.0  # dist (0,0)->(50,0)
    assert s._cached_waypoint_nav.shape == (9,)  # P2-11 contract
    assert s._cached_waypoint_nav[0] == 1.0  # unit dir +x toward (50,0)


def test_switch_activation_advances_to_exit_leg():
    s = _Stub(ReachabilityConfig(enable_waypoint_tracker=True))
    s.set_level_waypoints(_route())
    s._maybe_update_waypoints((50.0, 0.0), True)  # switch active -> phase-1 (exit)
    assert s._waypoint_index == 2
    assert s._waypoint_leg_distance == 50.0  # (50,0)->(100,0)


def test_set_waypoints_none_disables_tracker():
    s = _Stub(ReachabilityConfig(enable_waypoint_tracker=True, waypoint_build_on_reset=False))
    s.set_level_waypoints(None)
    s._maybe_update_waypoints((0.0, 0.0), False)  # no waypoints, no build -> no-op
    assert s._waypoint_tracker is None
    assert s._waypoint_leg_distance is None
