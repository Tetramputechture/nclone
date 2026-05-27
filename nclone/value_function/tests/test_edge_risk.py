"""Unit tests for the completion>loops>speed planner reweighting.

Covers:
  1. compute_edge_risk — mine proximity, jump/trick execution difficulty.
  2. apply_risk_weighting — soft bounded cost, w_risk==0 no-op, cap.
  3. dedup_position_loops — physical-loop collapse, clean-route preservation.
"""

import unittest

import numpy as np

from nclone.value_function.surface_graph import (
    build_graph,
    node_id,
    dijkstra_backward_with_tree,
)
from nclone.value_function.surface_extraction import (
    SurfacePoint, SURFACE_FLOOR, SURFACE_WALL,
)
from nclone.value_function.edge_enumeration import populate_graph_edges, EdgeResult
from nclone.value_function.edge_risk import (
    compute_edge_risk,
    apply_risk_weighting,
    _MINE_KILL_RADIUS,
    _MINE_RISK_EPSILON,
    _TRICK_RISK,
    _WALL_CHAIN_RISK,
    _LOW_LAUNCH_SPEED_RISK_MAX,
)
from nclone.value_function.route_extraction import (
    ArcTarget,
    dedup_position_loops,
)


def _floor_point(x, y, seg_id):
    return SurfacePoint(x=x, y=y, normal_x=0.0, normal_y=-1.0,
                        surface_type=SURFACE_FLOOR, segment_id=seg_id,
                        is_edge=False)


def _wall_point(x, y, seg_id):
    return SurfacePoint(x=x, y=y, normal_x=-1.0, normal_y=0.0,
                        surface_type=SURFACE_WALL, segment_id=seg_id,
                        is_edge=False)


def _single_edge_graph(src_pt, dst_pt, *, src_bin, dst_bin, jump_hold):
    """One directed jump edge between two surface points at the given bins."""
    g = build_graph([src_pt, dst_pt])
    e = EdgeResult(
        src_node=node_id(0, src_bin, g.max_vel_bins),
        dst_node=node_id(1, dst_bin, g.max_vel_bins),
        weight=10.0, jump_hold_frames=jump_hold,
        h_input_during=1, h_input_after=1,
    )
    populate_graph_edges(g, [e])
    return g


def _chain_graph(xs, *, vel_bin=7, weight=10.0, jump_hold=0):
    """Floor points at (x, 100) chained left->right with one edge each."""
    pts = [_floor_point(x, 100.0, i) for i, x in enumerate(xs)]
    g = build_graph(pts)
    edges = []
    for i in range(len(xs) - 1):
        edges.append(EdgeResult(
            src_node=node_id(i, vel_bin, g.max_vel_bins),
            dst_node=node_id(i + 1, vel_bin, g.max_vel_bins),
            weight=weight,
            jump_hold_frames=jump_hold,
            h_input_during=1,
            h_input_after=1,
        ))
    populate_graph_edges(g, edges)
    return g


def _arc(lx, ly, dx, dy, jh=0, trick=None):
    return ArcTarget(
        launch_segment_id=0, launch_x=lx, launch_y=ly, launch_vx=0.0,
        surface_normal_x=0.0, surface_normal_y=-1.0,
        jump_hold_frames=jh, h_input_during=1, h_input_after=1,
        landing_x=dx, landing_y=dy, landing_vx=0.0,
        arc_cost_frames=10.0, cumulative_cost=0.0, trick_tape=trick,
    )


class TestComputeEdgeRisk(unittest.TestCase):
    def test_runs_no_mines_zero_risk(self):
        g = _chain_graph([0, 60, 120, 180])
        risk = compute_edge_risk(g, deadly_mines=None)
        self.assertEqual(len(risk), 3)
        np.testing.assert_array_equal(risk, np.zeros(3, dtype=np.float32))

    def test_jump_edges_have_risk(self):
        g = _chain_graph([0, 60, 120], jump_hold=40)  # max-duration jumps
        risk = compute_edge_risk(g, deadly_mines=None)
        self.assertTrue(np.all(risk > 0.0))

    def test_trick_edge_high_risk(self):
        g = _chain_graph([0, 60, 120])
        g.trick_tapes = {0: [(1, 1), (1, 0)]}  # edge index 0 is a trick
        risk = compute_edge_risk(g, deadly_mines=None)
        self.assertGreaterEqual(float(risk[0]), _TRICK_RISK - 1e-6)
        self.assertEqual(float(risk[1]), 0.0)

    def test_mine_proximity_risk(self):
        # Points at x=0,60,120,180; put a deadly mine right at the x=60 point.
        g = _chain_graph([0, 60, 120, 180])
        mine = (60.0, 100.0)
        risk = compute_edge_risk(g, deadly_mines=[mine])
        # Edges touching the x=60 point (edge0: 0->60, edge1: 60->120) are risky;
        # edge2 (120->180) is >eps away from the mine -> ~0.
        self.assertGreater(float(risk[0]), 0.5)
        self.assertGreater(float(risk[1]), 0.5)
        self.assertEqual(float(risk[2]), 0.0)

    def test_mine_far_beyond_epsilon_zero(self):
        g = _chain_graph([0, 60, 120])  # 2 edges
        far = (1000.0, 1000.0)
        risk = compute_edge_risk(g, deadly_mines=[far])
        np.testing.assert_array_equal(risk, np.zeros(2, dtype=np.float32))


class TestMomentumRisk(unittest.TestCase):
    """Wall-chain re-contact + low-launch-speed risk (momentum preservation)."""

    def test_upward_wall_recontact_is_chain_risk(self):
        # Wall bin 0 center = WALL_VEL_MIN (-2.0) => upward re-contact chain-link.
        g = _single_edge_graph(_floor_point(0.0, 100.0, 0),
                               _wall_point(60.0, 100.0, 1),
                               src_bin=14, dst_bin=0, jump_hold=5)
        risk = compute_edge_risk(g, deadly_mines=None)
        self.assertGreaterEqual(float(risk[0]), _WALL_CHAIN_RISK - 1e-6)

    def test_downward_wall_landing_no_chain_risk(self):
        # Wall bin 4 center = 0.4 (downward slide) => NOT a chain-link; only the
        # small jump-hold risk applies, well below the chain penalty.
        g = _single_edge_graph(_floor_point(0.0, 100.0, 0),
                               _wall_point(60.0, 100.0, 1),
                               src_bin=14, dst_bin=4, jump_hold=5)
        risk = compute_edge_risk(g, deadly_mines=None)
        self.assertLess(float(risk[0]), _WALL_CHAIN_RISK)

    def test_low_launch_speed_penalises_slow_floor_jump(self):
        # Floor bin 6 center = 0.0 (standstill); bin 14 = MAX_HOR_SPEED (sprint).
        def floor_jump_risk(src_bin):
            g = _single_edge_graph(_floor_point(0.0, 100.0, 0),
                                   _floor_point(120.0, 100.0, 1),
                                   src_bin=src_bin, dst_bin=7, jump_hold=5)
            return float(compute_edge_risk(g, deadly_mines=None)[0])
        slow = floor_jump_risk(6)
        fast = floor_jump_risk(14)
        self.assertGreater(slow, fast)
        # The standstill jump carries ~the full low-launch penalty.
        self.assertGreaterEqual(slow - fast, _LOW_LAUNCH_SPEED_RISK_MAX - 1e-3)


class TestJumpTax(unittest.TestCase):
    def test_jump_tax_adds_per_jump_frames(self):
        g = _chain_graph([0, 60, 120], jump_hold=5)  # 2 jump edges
        before = g.edges_weight.copy()
        stats = apply_risk_weighting(g, w_risk=0.0, w_loop=0.0, jump_tax=3.0)
        self.assertTrue(stats["reweighted"])
        self.assertEqual(stats["n_jump_taxed"], 2)
        np.testing.assert_allclose(g.edges_weight, before + 3.0, atol=1e-4)

    def test_jump_tax_skips_runs(self):
        g = _chain_graph([0, 60, 120], jump_hold=0)  # runs, not jumps
        before = g.edges_weight.copy()
        apply_risk_weighting(g, w_risk=0.0, w_loop=0.0, jump_tax=3.0)
        np.testing.assert_array_equal(g.edges_weight, before)

    def test_all_shaping_off_is_noop(self):
        g = _chain_graph([0, 60, 120], jump_hold=5)
        before = g.edges_weight.copy()
        stats = apply_risk_weighting(g, w_risk=0.0, w_loop=0.0, jump_tax=0.0)
        self.assertFalse(stats["reweighted"])
        np.testing.assert_array_equal(g.edges_weight, before)


class TestApplyRiskWeighting(unittest.TestCase):
    def test_w_risk_zero_is_noop(self):
        g = _chain_graph([0, 60, 120, 180], jump_hold=40)
        before = g.edges_weight.copy()
        stats = apply_risk_weighting(g, deadly_mines=[(60.0, 100.0)],
                                     w_risk=0.0, w_loop=0.0)
        self.assertFalse(stats["reweighted"])
        np.testing.assert_array_equal(g.edges_weight, before)

    def test_w_risk_inflates_and_never_shrinks(self):
        g = _chain_graph([0, 60, 120, 180])
        before = g.edges_weight.copy()
        apply_risk_weighting(g, deadly_mines=[(60.0, 100.0)],
                             w_risk=0.5, w_loop=0.0)
        self.assertTrue(np.all(g.edges_weight >= before - 1e-3))
        # The near-mine edges grew; the far edge is unchanged.
        self.assertGreater(g.edges_weight[0], before[0] + 1e-3)
        self.assertAlmostEqual(g.edges_weight[2], before[2], places=4)

    def test_cap_bounds_factor(self):
        g = _chain_graph([0, 60, 120], jump_hold=40)
        before = g.edges_weight.copy()
        apply_risk_weighting(g, deadly_mines=[(0.0, 100.0), (60.0, 100.0)],
                             w_risk=100.0, w_loop=0.0, risk_cap=3.0)
        factor = g.edges_weight / before
        self.assertTrue(np.all(factor <= 3.0 + 1e-4))


class TestDedupPositionLoops(unittest.TestCase):
    def test_clean_route_unchanged(self):
        route = [_arc(0, 0, 60, 0), _arc(60, 0, 120, 0), _arc(120, 0, 180, 0)]
        out = dedup_position_loops(route)
        self.assertEqual(len(out), 3)
        self.assertEqual([a.launch_x for a in out], [0, 60, 120])

    def test_short_route_unchanged(self):
        route = [_arc(0, 0, 60, 0), _arc(60, 0, 0, 0)]
        out = dedup_position_loops(route)
        self.assertEqual(len(out), 2)

    def test_physical_loop_collapsed(self):
        # Launch sequence 0 -> 60 -> 120 -> 60(back!) -> 180.
        # The 60->120->60 cycle (arcs 1,2) splices out: arc3 (launch 60 ->
        # land 180) re-attaches to arc0's landing (60), leaving launches
        # [0, 60, 180] (landings [60, 180, 240]).
        route = [
            _arc(0, 0, 60, 0),     # 0
            _arc(60, 0, 120, 0),   # 1  (loop start)
            _arc(120, 0, 60, 0),   # 2  (returns toward x=60)
            _arc(60, 0, 180, 0),   # 3  (revisits x=60 launch)
            _arc(180, 0, 240, 0),  # 4
        ]
        out = dedup_position_loops(route)
        launches = [a.launch_x for a in out]
        self.assertEqual(launches, [0, 60, 180])


class TestDijkstraNonIntegerWeights(unittest.TestCase):
    """Regression: backward Dijkstra must stay correct for NON-INTEGER weights.

    The risk reweighting (completion>loops>speed) produces non-integer edge
    costs.  A latent float32/float64 mismatch in ``dijkstra_backward_with_tree``
    truncated working distances to float32 while the heap held float64 keys, so a
    weight that float32 rounds *down* (e.g. 0.7) made the popped key test
    ``dist_u > d_star[u]`` true and the node was wrongly skipped as stale — its
    finite cost then never propagated, leaving reachable nodes at ``inf``.  This
    was invisible with integer frame weights (exact in float32).
    """

    def _chain(self, n_pts, weight):
        pts = [_floor_point(i * 60.0, 100.0, i) for i in range(n_pts)]
        g = build_graph(pts)
        edges = []
        for i in range(n_pts - 1):  # i -> i+1, so the goal is the last node
            edges.append(EdgeResult(
                src_node=node_id(i, 7, g.max_vel_bins),
                dst_node=node_id(i + 1, 7, g.max_vel_bins),
                weight=weight, h_input_during=1, h_input_after=1,
            ))
        populate_graph_edges(g, edges)
        return g

    def test_weight_that_rounds_down_in_float32_stays_reachable(self):
        # 0.7 rounds DOWN in float32 -> triggered the spurious-skip on the first
        # relaxation with the old float32 working array.
        n = 12
        g = self._chain(n, weight=0.7)
        goal = node_id(n - 1, 7, g.max_vel_bins)
        d_star, _pred, _best = dijkstra_backward_with_tree(g, [goal])
        for i in range(n):
            nid = node_id(i, 7, g.max_vel_bins)
            self.assertTrue(np.isfinite(d_star[nid]),
                            f"node {i} should be reachable (non-integer weights)")
            self.assertAlmostEqual(float(d_star[nid]), 0.7 * (n - 1 - i), places=3)

    def test_integer_weights_unchanged(self):
        n = 6
        g = self._chain(n, weight=10.0)
        goal = node_id(n - 1, 7, g.max_vel_bins)
        d_star, _pred, _best = dijkstra_backward_with_tree(g, [goal])
        for i in range(n):
            nid = node_id(i, 7, g.max_vel_bins)
            self.assertAlmostEqual(float(d_star[nid]), 10.0 * (n - 1 - i), places=3)


if __name__ == "__main__":
    unittest.main()
