"""Unit tests for the lattice-backed waypoint planner / tracker.

The expensive part (building the lattice) is bypassed: we exercise the pure
assembly + cursor logic deterministically by injecting synthetic arcs, mirroring
``test_lattice_distance.py``. A single opt-in end-to-end build test is gated
behind an env var so the default suite stays fast.
"""

import math
import os
import types
import unittest

import numpy as np

from nclone.value_function.waypoint_planner import (
    LatticeWaypointPlanner,
    Waypoint,
    WaypointTracker,
    _densify,
)


def _arc(x, y, cost):
    """Minimal stand-in for ArcTarget (only the fields _assemble reads)."""
    return types.SimpleNamespace(landing_x=x, landing_y=y, cumulative_cost=cost)


def _assemble(phase0, phase1, info, **kw):
    """Call the planner's assembler without running a lattice build."""
    p = LatticeWaypointPlanner.__new__(LatticeWaypointPlanner)
    p.switch = kw.pop("switch", (60.0, 100.0))
    p.exit = kw.pop("exit", (120.0, 100.0))
    return p._assemble(
        phase0,
        phase1,
        info,
        leg_tol=kw.get("leg_tol", 14.0),
        switch_tol=kw.get("switch_tol", 16.0),
        exit_tol=kw.get("exit_tol", 22.0),
        max_leg_px=kw.get("max_leg_px", None),
    )


class TestAssembly(unittest.TestCase):
    def test_route_order_and_objectives(self):
        # phase0: spawn->...->switch (last arc dropped, switch entity appended)
        phase0 = [_arc(0, 100, 100.0), _arc(30, 100, 70.0), _arc(60, 100, 55.0)]
        phase1 = [_arc(60, 100, 40.0), _arc(120, 100, 0.0)]
        info = {"spawn_d_exit_from_switch": 40.0}
        wps = _assemble(phase0, phase1, info)

        # last entry is the exit objective; a switch objective sits in the middle
        self.assertTrue(wps[-1].is_objective)
        self.assertEqual((wps[-1].x, wps[-1].y), (120.0, 100.0))
        self.assertEqual(wps[-1].phase, 1)
        self.assertAlmostEqual(wps[-1].cost_to_go, 0.0)

        objectives = [w for w in wps if w.is_objective]
        self.assertEqual(len(objectives), 2)  # switch + exit
        switch_wp = objectives[0]
        self.assertEqual((switch_wp.x, switch_wp.y), (60.0, 100.0))
        self.assertEqual(switch_wp.phase, 0)
        self.assertAlmostEqual(switch_wp.cost_to_go, 40.0)

    def test_cost_to_go_monotonic_nonincreasing(self):
        phase0 = [_arc(0, 100, 100.0), _arc(30, 100, 70.0), _arc(60, 100, 55.0)]
        phase1 = [_arc(60, 100, 40.0), _arc(90, 100, 20.0), _arc(120, 100, 0.0)]
        info = {"spawn_d_exit_from_switch": 40.0}
        wps = _assemble(phase0, phase1, info)
        costs = [w.cost_to_go for w in wps]
        for a, b in zip(costs[:-1], costs[1:]):
            self.assertGreaterEqual(a + 1e-6, b, f"cost_to_go increased: {costs}")

    def test_switch_cost_falls_back_to_phase1_when_info_infinite(self):
        phase0 = [_arc(0, 100, 100.0), _arc(60, 100, 55.0)]
        phase1 = [_arc(60, 100, 42.0), _arc(120, 100, 0.0)]
        info = {"spawn_d_exit_from_switch": float("inf")}
        wps = _assemble(phase0, phase1, info)
        switch_wp = [w for w in wps if w.is_objective][0]
        self.assertAlmostEqual(switch_wp.cost_to_go, 42.0)


class TestDensify(unittest.TestCase):
    def test_long_leg_is_subdivided(self):
        wps = [
            Waypoint(0.0, 0.0, 14.0, 0, 200.0, False),
            Waypoint(300.0, 0.0, 14.0, 0, 0.0, True),
        ]
        out = _densify(wps, max_leg_px=100.0)
        # 300px / 100px -> 2 interior points, endpoints preserved => 4 total
        self.assertEqual(len(out), 4)
        self.assertEqual((out[0].x, out[-1].x), (0.0, 300.0))
        # interior cost-to-go interpolated and decreasing
        costs = [w.cost_to_go for w in out]
        for a, b in zip(costs[:-1], costs[1:]):
            self.assertGreaterEqual(a, b)
        # objective endpoint preserved, interior points are not objectives
        self.assertTrue(out[-1].is_objective)
        self.assertFalse(any(w.is_objective for w in out[1:-1]))

    def test_short_leg_unchanged(self):
        wps = [
            Waypoint(0.0, 0.0, 14.0, 0, 50.0, False),
            Waypoint(40.0, 0.0, 14.0, 1, 0.0, True),
        ]
        out = _densify(wps, max_leg_px=100.0)
        self.assertEqual(len(out), 2)


def _route():
    """A simple 3-leg route: spawn(0) -> switch(100) -> exit(200), y=0."""
    return [
        Waypoint(0.0, 0.0, 14.0, 0, 150.0, False),    # spawn-ish landing
        Waypoint(50.0, 0.0, 14.0, 0, 110.0, False),
        Waypoint(100.0, 0.0, 16.0, 0, 80.0, True),     # exit switch (objective)
        Waypoint(150.0, 0.0, 14.0, 1, 40.0, False),
        Waypoint(200.0, 0.0, 22.0, 1, 0.0, True),      # exit (objective)
    ]


class TestTracker(unittest.TestCase):
    def test_advances_through_intermediate_waypoints(self):
        t = WaypointTracker(_route())
        self.assertEqual(t.index, 0)
        # within tol of wp0 -> advance to wp1
        t.update(2.0, 0.0, switch_activated=False)
        self.assertEqual(t.index, 1)
        # not near wp1 yet
        t.update(30.0, 0.0, switch_activated=False)
        self.assertEqual(t.index, 1)
        # reach wp1
        t.update(50.0, 0.0, switch_activated=False)
        self.assertEqual(t.index, 2)

    def test_does_not_skip_switch_objective_on_proximity(self):
        t = WaypointTracker(_route())
        # walk sequentially up to the switch: reach wp0, wp1, then sit ON the
        # switch (wp2) WITHOUT activating it.
        t.update(0.0, 0.0, switch_activated=False)    # reach wp0 -> idx 1
        t.update(50.0, 0.0, switch_activated=False)   # reach wp1 -> idx 2 (switch)
        self.assertEqual(t.index, 2)
        for _ in range(3):
            t.update(100.0, 0.0, switch_activated=False)  # on switch, not active
        self.assertEqual(t.index, 2)  # parked on the switch objective, waiting
        self.assertTrue(t.current().is_objective)

    def test_switch_activation_fast_forwards_to_phase1(self):
        t = WaypointTracker(_route())
        t.update(0.0, 0.0, switch_activated=False)  # somewhere early
        t.update(0.0, 0.0, switch_activated=True)   # switch now active
        self.assertEqual(t.current().phase, 1)
        self.assertEqual(t.index, 3)

    def test_never_advances_past_final_exit(self):
        t = WaypointTracker(_route())
        # switch active -> fast-forward past phase-0 to wp3 (first phase-1)
        t.update(0.0, 0.0, switch_activated=True)
        self.assertEqual(t.index, 3)
        t.update(150.0, 0.0, switch_activated=True)  # reach wp3 -> idx 4 (exit)
        self.assertEqual(t.index, 4)
        for _ in range(5):
            t.update(200.0, 0.0, switch_activated=True)
        self.assertEqual(t.index, len(_route()) - 1)
        self.assertFalse(t.done)  # parked ON the last waypoint, not past it
        self.assertEqual(t.current().x, 200.0)

    def test_nav_feature_points_at_current_waypoint(self):
        t = WaypointTracker(_route())
        feat = t.nav_feature(0.0, 0.0, switch_activated=False)
        self.assertEqual(feat.shape, (9,))  # P2-11: 4 legacy + 5 executable dims
        # after consuming wp0 the target is wp1 at (50,0): unit dir +x
        self.assertAlmostEqual(feat[0], 1.0, places=5)
        self.assertAlmostEqual(feat[1], 0.0, places=5)
        # cost_to_go_norm = 110/150 in [0,1]
        self.assertAlmostEqual(feat[2], 110.0 / 150.0, places=5)
        self.assertEqual(feat[3], 0.0)  # phase 0

    def test_nav_feature_phase_flips_with_switch(self):
        t = WaypointTracker(_route())
        feat = t.nav_feature(0.0, 0.0, switch_activated=True)
        self.assertEqual(feat[3], 1.0)
        self.assertEqual(t.current().phase, 1)

    def test_leg_distance(self):
        t = WaypointTracker(_route())
        t.update(0.0, 0.0, switch_activated=False)  # target now wp1 (50,0)
        self.assertAlmostEqual(t.leg_distance(20.0, 0.0), 30.0, places=5)

    def test_empty_route_is_safe(self):
        t = WaypointTracker([])
        self.assertIsNone(t.update(0.0, 0.0, False))
        np.testing.assert_array_equal(
            t.nav_feature(0.0, 0.0, False), np.zeros(9, dtype=np.float32)
        )


@unittest.skipUnless(
    os.environ.get("NCLONE_WAYPOINT_E2E") == "1",
    "set NCLONE_WAYPOINT_E2E=1 to run the (slow) lattice build test",
)
class TestEndToEndBuild(unittest.TestCase):
    def test_build_on_simple_level(self):
        # Opt-in: builds a real lattice on a known-simple official level.
        from nclone.value_function.tools.physics_route_spike import _load_map_data

        map_path = os.environ.get("NCLONE_WAYPOINT_MAP", "maps/flat-run-open")
        map_data = _load_map_data(map_path)
        planner = LatticeWaypointPlanner(map_data, use_cache=True)
        self.assertTrue(planner.available)
        self.assertGreaterEqual(len(planner.waypoints), 2)
        self.assertTrue(planner.waypoints[-1].is_objective)
        self.assertAlmostEqual(planner.waypoints[-1].cost_to_go, 0.0, places=3)
        costs = [w.cost_to_go for w in planner.waypoints]
        for a, b in zip(costs[:-1], costs[1:]):
            self.assertGreaterEqual(a + 1e-6, b)


if __name__ == "__main__":
    unittest.main()
