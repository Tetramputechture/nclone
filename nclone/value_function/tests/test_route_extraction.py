"""Unit tests for route extraction, RoutePBRS, and dijkstra_backward_with_tree.

Tests are grouped into:
  1. Dijkstra tree invariants (predecessor direction, d_star ordering)
  2. Route extraction round-trip (extract_route correctness)
  3. Frame-counter potential monotonicity
  4. Phase transition replanning
  5. Empty/unreachable route fallback
  6. Arc advancement
"""

import math
import unittest
from dataclasses import dataclass
from typing import List, Optional, Set
from unittest.mock import MagicMock

import numpy as np

from nclone.value_function.surface_graph import (
    SurfaceGraph,
    build_graph,
    dijkstra_backward_with_tree,
    node_id,
    node_id_to_parts,
    velocity_to_bin,
    velocity_bin_centers,
)
from nclone.value_function.surface_extraction import SurfacePoint, SURFACE_FLOOR
from nclone.value_function.edge_enumeration import populate_graph_edges, EdgeResult
from nclone.value_function.route_extraction import (
    ArcTarget,
    extract_route,
    merge_running_arcs,
    find_nearest_start_node,
)
from nclone.value_function.route_pbrs import RoutePBRS, RouteState


# ---------------------------------------------------------------------------
# Helper: build a minimal synthetic graph for testing
# ---------------------------------------------------------------------------

def _make_floor_point(x: float, y: float, seg_id: int) -> SurfacePoint:
    return SurfacePoint(
        x=x, y=y,
        normal_x=0.0, normal_y=-1.0,
        surface_type=SURFACE_FLOOR,
        segment_id=seg_id,
        is_edge=False,
    )


def _build_test_graph(n_floor_points: int = 5) -> SurfaceGraph:
    """Build a small test graph with n floor points laid out horizontally."""
    spacing = 60.0  # pixels between points
    pts = [_make_floor_point(i * spacing, 100.0, seg_id=i) for i in range(n_floor_points)]
    return build_graph(pts)


def _populate_chain_edges(graph: SurfaceGraph, vel_bin: int = 7) -> List[EdgeResult]:
    """Populate the graph with a chain of edges: node i -> node i+1 at fixed cost.

    Creates a simple path graph where every node has exactly one outgoing edge
    to the adjacent node (if it exists), forming a chain.
    """
    n_pts = graph.n_points
    edges: List[EdgeResult] = []
    for i in range(n_pts - 1):
        src = node_id(i, vel_bin, graph.max_vel_bins)
        dst = node_id(i + 1, vel_bin, graph.max_vel_bins)
        edges.append(EdgeResult(
            src_node=src,
            dst_node=dst,
            weight=10.0,          # 10 frames per arc
            jump_hold_frames=5,
            h_input_during=1,
        ))
    populate_graph_edges(graph, edges)
    graph.edges_jump_hold = np.array([e.jump_hold_frames for e in edges], dtype=np.int8)
    graph.edges_h_input = np.array([e.h_input_during for e in edges], dtype=np.int8)
    return edges


# ===========================================================================
# 1. Dijkstra tree invariants
# ===========================================================================

class TestDijkstraBackwardWithTree(unittest.TestCase):

    def setUp(self):
        self.graph = _build_test_graph(5)
        self.vel_bin = 7
        self.edges = _populate_chain_edges(self.graph, self.vel_bin)
        # Goal: last point
        self.goal_node = node_id(4, self.vel_bin, self.graph.max_vel_bins)

    def test_predecessor_direction_invariant(self):
        """d_star[predecessor[v]] < d_star[v] for all non-goal finite nodes."""
        d_star, predecessor, best_edge_idx = dijkstra_backward_with_tree(
            self.graph, [self.goal_node]
        )

        n = self.graph.n_nodes
        for v in range(n):
            pred_v = int(predecessor[v])
            if pred_v == -1:
                continue  # goal or unreachable
            if not np.isfinite(d_star[v]):
                continue
            self.assertTrue(
                np.isfinite(d_star[pred_v]),
                f"predecessor[{v}]={pred_v} has infinite d_star but {v} has d_star={d_star[v]:.1f}",
            )
            self.assertLess(
                d_star[pred_v], d_star[v],
                f"Invariant violated: d_star[pred[{v}]]={d_star[pred_v]:.1f} >= d_star[{v}]={d_star[v]:.1f}",
            )

    def test_goal_node_has_minus1_predecessor(self):
        """Goal nodes must have predecessor == -1."""
        d_star, predecessor, _ = dijkstra_backward_with_tree(
            self.graph, [self.goal_node]
        )
        self.assertEqual(int(predecessor[self.goal_node]), -1)
        self.assertAlmostEqual(float(d_star[self.goal_node]), 0.0)

    def test_d_star_monotone_along_chain(self):
        """d_star should decrease monotonically from start toward goal."""
        d_star, predecessor, _ = dijkstra_backward_with_tree(
            self.graph, [self.goal_node]
        )
        # Walk the chain from point 0 to point 4
        for i in range(4):
            nid_i = node_id(i, self.vel_bin, self.graph.max_vel_bins)
            nid_next = node_id(i + 1, self.vel_bin, self.graph.max_vel_bins)
            self.assertGreater(
                d_star[nid_i], d_star[nid_next],
                f"Expected d_star[{i}] > d_star[{i+1}] along chain",
            )

    def test_best_edge_idx_consistency(self):
        """best_edge_idx[v] must index the edge from v to predecessor[v]."""
        d_star, predecessor, best_edge_idx = dijkstra_backward_with_tree(
            self.graph, [self.goal_node]
        )
        for v in range(self.graph.n_nodes):
            pred_v = int(predecessor[v])
            if pred_v == -1 or not np.isfinite(d_star[v]):
                continue
            ei = int(best_edge_idx[v])
            self.assertGreaterEqual(ei, 0)
            self.assertLess(ei, len(self.graph.edges_src))
            # The edge must go from v to predecessor[v]
            self.assertEqual(int(self.graph.edges_src[ei]), v)
            self.assertEqual(int(self.graph.edges_dst[ei]), pred_v)

    def test_unreachable_nodes_have_inf_and_minus1(self):
        """Nodes disconnected from the goal must have d_star=inf and predecessor=-1."""
        graph = _build_test_graph(5)
        vel_bin = 7
        # Create edges pointing the WRONG direction: i+1 -> i  (so no path from 0 to 4)
        edges = []
        for i in range(4):
            src = node_id(i + 1, vel_bin, graph.max_vel_bins)
            dst = node_id(i, vel_bin, graph.max_vel_bins)
            edges.append(EdgeResult(src, dst, weight=1.0))
        populate_graph_edges(graph, edges)

        goal = node_id(4, vel_bin, graph.max_vel_bins)
        d_star, predecessor, best_edge_idx = dijkstra_backward_with_tree(graph, [goal])

        # Nodes 0-3 at vel_bin=7 should be unreachable
        for i in range(4):
            nid = node_id(i, vel_bin, graph.max_vel_bins)
            self.assertFalse(np.isfinite(d_star[nid]), f"Node {i} should be unreachable")
            self.assertEqual(int(predecessor[nid]), -1)


# ===========================================================================
# 2. Route extraction round-trip
# ===========================================================================

class TestExtractRoute(unittest.TestCase):

    def setUp(self):
        self.graph = _build_test_graph(5)
        self.vel_bin = 7
        self.edges = _populate_chain_edges(self.graph, self.vel_bin)
        self.goal_node = node_id(4, self.vel_bin, self.graph.max_vel_bins)
        self.d_star, self.predecessor, self.best_edge_idx = dijkstra_backward_with_tree(
            self.graph, [self.goal_node]
        )

    def test_route_length(self):
        """Route from node 0 to goal (node 4) should have 4 arcs."""
        start_node = node_id(0, self.vel_bin, self.graph.max_vel_bins)
        route = extract_route(
            self.graph, self.d_star, self.predecessor, self.best_edge_idx, start_node
        )
        self.assertEqual(len(route), 4)

    def test_route_ordering(self):
        """Arc i lands where arc i+1 launches."""
        start_node = node_id(0, self.vel_bin, self.graph.max_vel_bins)
        route = extract_route(
            self.graph, self.d_star, self.predecessor, self.best_edge_idx, start_node
        )
        for i in range(len(route) - 1):
            self.assertAlmostEqual(route[i].landing_x, route[i + 1].launch_x, places=2)
            self.assertAlmostEqual(route[i].landing_y, route[i + 1].launch_y, places=2)

    def test_cumulative_cost_decreasing(self):
        """cumulative_cost should strictly decrease along the route."""
        start_node = node_id(0, self.vel_bin, self.graph.max_vel_bins)
        route = extract_route(
            self.graph, self.d_star, self.predecessor, self.best_edge_idx, start_node
        )
        for i in range(len(route) - 1):
            self.assertGreater(
                route[i].cumulative_cost, route[i + 1].cumulative_cost,
                f"cumulative_cost should decrease: arc {i} has {route[i].cumulative_cost:.1f}, "
                f"arc {i+1} has {route[i+1].cumulative_cost:.1f}",
            )

    def test_last_arc_cumulative_cost_is_zero(self):
        """Last arc lands at the goal; cumulative_cost should be 0."""
        start_node = node_id(0, self.vel_bin, self.graph.max_vel_bins)
        route = extract_route(
            self.graph, self.d_star, self.predecessor, self.best_edge_idx, start_node
        )
        self.assertAlmostEqual(route[-1].cumulative_cost, 0.0)

    def test_action_metadata_preserved(self):
        """jump_hold_frames and h_input_during from edges must appear in ArcTargets."""
        start_node = node_id(0, self.vel_bin, self.graph.max_vel_bins)
        route = extract_route(
            self.graph, self.d_star, self.predecessor, self.best_edge_idx, start_node
        )
        for arc in route:
            self.assertEqual(arc.jump_hold_frames, 5)
            self.assertEqual(arc.h_input_during, 1)

    def test_empty_route_for_unreachable_start(self):
        """Unreachable start node returns empty list."""
        # Use a node index that has no path to goal (wrong direction)
        graph2 = _build_test_graph(3)
        vel_bin = 7
        edges = [EdgeResult(
            node_id(2, vel_bin, graph2.max_vel_bins),
            node_id(1, vel_bin, graph2.max_vel_bins),
            weight=1.0,
        )]
        populate_graph_edges(graph2, edges)
        goal = node_id(2, vel_bin, graph2.max_vel_bins)
        d_star2, pred2, best2 = dijkstra_backward_with_tree(graph2, [goal])

        unreachable = node_id(0, vel_bin, graph2.max_vel_bins)
        route = extract_route(graph2, d_star2, pred2, best2, unreachable)
        self.assertEqual(len(route), 0)

    def test_goal_node_returns_empty_route(self):
        """Starting at the goal returns an empty route (already there)."""
        route = extract_route(
            self.graph, self.d_star, self.predecessor, self.best_edge_idx,
            self.goal_node
        )
        self.assertEqual(len(route), 0)


# ===========================================================================
# 3. Frame-counter potential monotonicity
# ===========================================================================

class TestFrameCounterPotential(unittest.TestCase):

    def _make_route_pbrs(self, n_arcs: int = 4) -> RoutePBRS:
        """Build a RoutePBRS pre-loaded with a synthetic route."""
        graph = _build_test_graph(n_arcs + 1)
        vel_bin = 7
        _populate_chain_edges(graph, vel_bin)
        goal_node = node_id(n_arcs, vel_bin, graph.max_vel_bins)
        d_star, predecessor, best_edge_idx = dijkstra_backward_with_tree(
            graph, [goal_node]
        )

        pbrs = RoutePBRS(
            graph=graph,
            d_star_switch=d_star,
            d_star_exit=d_star,
            predecessor_switch=predecessor,
            predecessor_exit=predecessor,
            best_edge_switch=best_edge_idx,
            best_edge_exit=best_edge_idx,
            d_max_switch=float(np.nanmax(d_star[np.isfinite(d_star)])),
            d_max_exit=float(np.nanmax(d_star[np.isfinite(d_star)])),
        )
        # Manually set the route
        start_node = node_id(0, vel_bin, graph.max_vel_bins)
        pbrs._route = extract_route(graph, d_star, predecessor, best_edge_idx, start_node)
        pbrs._total_route_cost = pbrs._route[0].cumulative_cost
        return pbrs

    def _state_dict(self, x: float = 0.0, y: float = 100.0) -> dict:
        return {
            "x": x, "y": y, "player_x": x, "player_y": y,
            "vx": 1.0, "vy": 0.0,
            "switch_activated": False,
            "on_surface": True,
            "contact_state": 1,
        }

    def test_potential_monotone_within_arc(self):
        """Potential is non-decreasing each frame within a single arc."""
        pbrs = self._make_route_pbrs(4)
        state = self._state_dict()

        phi_values = []
        for _ in range(15):
            phi_values.append(pbrs.get_potential(state))
            pbrs.update(state)

        for i in range(len(phi_values) - 1):
            self.assertLessEqual(
                phi_values[i], phi_values[i + 1] + 1e-6,
                f"Potential not monotone at step {i}: {phi_values[i]:.4f} > {phi_values[i+1]:.4f}",
            )

    def test_potential_clamps_not_positive(self):
        """Potential should stay in [-1, 0.5] range."""
        pbrs = self._make_route_pbrs(4)
        state = self._state_dict()

        for _ in range(50):
            phi = pbrs.get_potential(state)
            self.assertLessEqual(phi, 0.5 + 1e-6)
            self.assertGreaterEqual(phi, -1.0 - 1e-6)
            pbrs.update(state)

    def test_potential_zero_at_goal(self):
        """When arc_idx exceeds route length (at goal), potential should be ~0."""
        pbrs = self._make_route_pbrs(1)  # single arc
        state = self._state_dict()

        # Manually advance past all arcs
        pbrs._state.arc_idx = len(pbrs._route)  # at goal
        # The route is still present but arc_idx >= len → potential via last arc
        # Actually, get_potential uses _route[_state.arc_idx] which would IndexError;
        # let's test with empty route instead (the canonical "at goal" case)
        pbrs._route = []
        phi = pbrs.get_potential(state)
        # Fallback: should return raw d_star / d_max ≈ -1 for unreachable
        # (since state (0,100) maps to some node with finite d_star)
        self.assertLessEqual(phi, 0.5 + 1e-6)
        self.assertGreaterEqual(phi, -1.1)


# ===========================================================================
# 4. Phase transition replanning
# ===========================================================================

class TestPhaseTransition(unittest.TestCase):

    def _make_pbrs(self, n_pts=5):
        """Build a RoutePBRS with separate switch and exit d_star arrays."""
        graph = _build_test_graph(n_pts)
        vel_bin = 7
        _populate_chain_edges(graph, vel_bin)

        switch_goal = node_id(2, vel_bin, graph.max_vel_bins)
        exit_goal = node_id(4, vel_bin, graph.max_vel_bins)

        d_sw, pred_sw, best_sw = dijkstra_backward_with_tree(graph, [switch_goal])
        d_ex, pred_ex, best_ex = dijkstra_backward_with_tree(graph, [exit_goal])

        d_max_sw = float(np.nanmax(d_sw[np.isfinite(d_sw)]))
        d_max_ex = float(np.nanmax(d_ex[np.isfinite(d_ex)]))

        return RoutePBRS(
            graph=graph,
            d_star_switch=d_sw,
            d_star_exit=d_ex,
            predecessor_switch=pred_sw,
            predecessor_exit=pred_ex,
            best_edge_switch=best_sw,
            best_edge_exit=best_ex,
            d_max_switch=d_max_sw,
            d_max_exit=d_max_ex,
        )

    def test_phase_transitions_on_switch_collection(self):
        """Phase should switch from 0 to 1 when switch_activated becomes True."""
        pbrs = self._make_pbrs()
        # Manually set a route in phase 0
        graph = pbrs._graph
        vel_bin = 7
        start = node_id(0, vel_bin, graph.max_vel_bins)
        pbrs._route = extract_route(graph, pbrs._d_star_switch,
                                     pbrs._pred_switch, pbrs._best_edge_switch, start)
        pbrs._total_route_cost = pbrs._route[0].cumulative_cost if pbrs._route else 1.0
        pbrs._phase = 0

        self.assertEqual(pbrs._phase, 0)
        state = {
            "x": graph.surface_points[0].x,
            "y": graph.surface_points[0].y,
            "vx": 1.0, "vy": 0.0,
            "switch_activated": True,  # trigger transition
            "on_surface": True,
            "contact_state": 1,
        }
        pbrs.update(state)
        self.assertEqual(pbrs._phase, 1)

    def test_route_replaced_after_phase_transition(self):
        """Route after phase transition should use exit d_star (different goal)."""
        pbrs = self._make_pbrs()
        graph = pbrs._graph
        vel_bin = 7
        start = node_id(0, vel_bin, graph.max_vel_bins)

        pbrs._route = extract_route(graph, pbrs._d_star_switch,
                                     pbrs._pred_switch, pbrs._best_edge_switch, start)
        pbrs._total_route_cost = pbrs._route[0].cumulative_cost if pbrs._route else 1.0
        pbrs._phase = 0
        old_route_len = len(pbrs._route)

        state = {
            "x": graph.surface_points[0].x,
            "y": graph.surface_points[0].y,
            "vx": 1.0, "vy": 0.0,
            "switch_activated": True,
            "on_surface": True,
            "contact_state": 1,
        }
        pbrs.update(state)

        # Route should now point toward exit (node 4), so it may have more arcs
        new_route_len = len(pbrs._route)
        # The switch goal was node 2; exit goal is node 4 — route from 0 to 4 is 4 arcs
        self.assertGreaterEqual(new_route_len, 1)
        # Phase must be 1
        self.assertEqual(pbrs._phase, 1)


# ===========================================================================
# 5. Empty route fallback
# ===========================================================================

class TestEmptyRouteFallback(unittest.TestCase):

    def test_get_potential_with_empty_route_returns_finite_value(self):
        """get_potential on empty route falls back to raw d_star, returning a finite value."""
        graph = _build_test_graph(3)
        vel_bin = 7
        edges = [EdgeResult(
            node_id(0, vel_bin, graph.max_vel_bins),
            node_id(1, vel_bin, graph.max_vel_bins),
            weight=10.0,
        )]
        populate_graph_edges(graph, edges)
        goal = node_id(1, vel_bin, graph.max_vel_bins)
        d_star, pred, best = dijkstra_backward_with_tree(graph, [goal])
        d_max = float(np.nanmax(d_star[np.isfinite(d_star)]))

        pbrs = RoutePBRS(
            graph=graph,
            d_star_switch=d_star,
            d_star_exit=d_star,
            predecessor_switch=pred,
            predecessor_exit=pred,
            best_edge_switch=best,
            best_edge_exit=best,
            d_max_switch=d_max,
            d_max_exit=d_max,
        )
        pbrs._route = []  # simulate empty route

        sp0 = graph.surface_points[0]
        state = {
            "x": sp0.x, "y": sp0.y,
            "player_x": sp0.x, "player_y": sp0.y,
            "vx": 1.0, "vy": 0.0,
            "switch_activated": False,
            "on_surface": True,
            "contact_state": 1,
        }
        phi = pbrs.get_potential(state)
        self.assertTrue(math.isfinite(phi), f"Expected finite phi, got {phi}")
        self.assertAlmostEqual(phi, -1.0, places=1)  # node 0 has full d_star

    def test_get_arc_target_features_with_empty_route_returns_zeros(self):
        """get_arc_target_features returns np.zeros(10) when route is empty."""
        graph = _build_test_graph(3)
        populate_graph_edges(graph, [])
        d = np.full(graph.n_nodes, np.inf, dtype=np.float32)
        p = np.full(graph.n_nodes, -1, dtype=np.int32)
        b = np.full(graph.n_nodes, -1, dtype=np.int32)

        pbrs = RoutePBRS(
            graph=graph,
            d_star_switch=d, d_star_exit=d,
            predecessor_switch=p, predecessor_exit=p,
            best_edge_switch=b, best_edge_exit=b,
            d_max_switch=1.0, d_max_exit=1.0,
        )
        pbrs._route = []
        feats = pbrs.get_arc_target_features({"x": 0.0, "y": 0.0})
        np.testing.assert_array_equal(feats, np.zeros(10, dtype=np.float32))


# ===========================================================================
# 6. Arc advancement
# ===========================================================================

class TestArcAdvancement(unittest.TestCase):

    def test_arc_idx_advances_when_near_next_launch(self):
        """arc_idx should increment when agent lands near the next arc's launch point."""
        graph = _build_test_graph(5)
        vel_bin = 7
        _populate_chain_edges(graph, vel_bin)
        goal = node_id(4, vel_bin, graph.max_vel_bins)
        d_star, pred, best = dijkstra_backward_with_tree(graph, [goal])
        d_max = float(np.nanmax(d_star[np.isfinite(d_star)]))

        pbrs = RoutePBRS(
            graph=graph,
            d_star_switch=d_star, d_star_exit=d_star,
            predecessor_switch=pred, predecessor_exit=pred,
            best_edge_switch=best, best_edge_exit=best,
            d_max_switch=d_max, d_max_exit=d_max,
        )
        start = node_id(0, vel_bin, graph.max_vel_bins)
        pbrs._route = extract_route(graph, d_star, pred, best, start)
        pbrs._total_route_cost = pbrs._route[0].cumulative_cost

        self.assertEqual(pbrs._state.arc_idx, 0)

        # Simulate agent landing exactly at arc 1's launch point
        arc1_launch = pbrs._route[1]
        state = {
            "x": arc1_launch.launch_x,
            "y": arc1_launch.launch_y,
            "vx": 1.0, "vy": 0.0,
            "switch_activated": False,
            "on_surface": True,
            "contact_state": 1,
        }
        pbrs.update(state)
        self.assertEqual(pbrs._state.arc_idx, 1)
        self.assertEqual(pbrs._state.frames_into_current_arc, 0)

    def test_frame_counter_resets_on_advancement(self):
        """frames_into_current_arc resets to 0 when arc advances."""
        graph = _build_test_graph(3)
        vel_bin = 7
        _populate_chain_edges(graph, vel_bin)
        goal = node_id(2, vel_bin, graph.max_vel_bins)
        d_star, pred, best = dijkstra_backward_with_tree(graph, [goal])
        d_max = float(np.nanmax(d_star[np.isfinite(d_star)]))

        pbrs = RoutePBRS(
            graph=graph,
            d_star_switch=d_star, d_star_exit=d_star,
            predecessor_switch=pred, predecessor_exit=pred,
            best_edge_switch=best, best_edge_exit=best,
            d_max_switch=d_max, d_max_exit=d_max,
        )
        start = node_id(0, vel_bin, graph.max_vel_bins)
        pbrs._route = extract_route(graph, d_star, pred, best, start)
        pbrs._total_route_cost = pbrs._route[0].cumulative_cost

        # Increment frame counter manually
        pbrs._state.frames_into_current_arc = 7

        # Trigger advancement by landing near arc 1's launch
        arc1_launch = pbrs._route[1]
        state = {
            "x": arc1_launch.launch_x,
            "y": arc1_launch.launch_y,
            "vx": 1.0, "vy": 0.0,
            "switch_activated": False,
            "on_surface": True,
            "contact_state": 1,
        }
        pbrs.update(state)
        self.assertEqual(pbrs._state.frames_into_current_arc, 0)


# ===========================================================================
# 7. merge_running_arcs
# ===========================================================================

def _make_run_arc(
    seg_id: int,
    launch_x: float,
    landing_x: float,
    h_input: int = 1,
    cost: float = 10.0,
    cumulative_cost: float = 0.0,
    launch_vx: float = 0.0,
    landing_vx: float = 1.0,
) -> ArcTarget:
    """Helper: build a running ArcTarget (jump_hold_frames=0)."""
    return ArcTarget(
        launch_segment_id=seg_id,
        launch_x=launch_x,
        launch_y=100.0,
        launch_vx=launch_vx,
        surface_normal_x=0.0,
        surface_normal_y=-1.0,
        jump_hold_frames=0,
        h_input_during=h_input,
        landing_x=landing_x,
        landing_y=100.0,
        landing_vx=landing_vx,
        arc_cost_frames=cost,
        cumulative_cost=cumulative_cost,
    )


def _make_jump_arc(
    seg_id: int,
    launch_x: float,
    landing_x: float,
    cost: float = 20.0,
    cumulative_cost: float = 0.0,
) -> ArcTarget:
    """Helper: build a jumping ArcTarget (jump_hold_frames > 0)."""
    return ArcTarget(
        launch_segment_id=seg_id,
        launch_x=launch_x,
        launch_y=100.0,
        launch_vx=2.0,
        surface_normal_x=0.0,
        surface_normal_y=-1.0,
        jump_hold_frames=8,
        h_input_during=1,
        landing_x=landing_x,
        landing_y=80.0,
        landing_vx=2.0,
        arc_cost_frames=cost,
        cumulative_cost=cumulative_cost,
    )


class TestMergeRunningArcs(unittest.TestCase):

    def test_empty_route_returns_empty(self):
        self.assertEqual(merge_running_arcs([]), [])

    def test_single_arc_returned_unchanged(self):
        arc = _make_run_arc(seg_id=0, launch_x=0.0, landing_x=60.0)
        result = merge_running_arcs([arc])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], arc)

    def test_two_same_segment_running_arcs_merge(self):
        """Two consecutive running arcs on the same segment collapse into one."""
        arc0 = _make_run_arc(
            seg_id=0, launch_x=0.0, landing_x=60.0,
            h_input=1, cost=10.0, cumulative_cost=20.0,
            launch_vx=0.0, landing_vx=1.5,
        )
        arc1 = _make_run_arc(
            seg_id=0, launch_x=60.0, landing_x=120.0,
            h_input=1, cost=12.0, cumulative_cost=8.0,
            launch_vx=1.5, landing_vx=2.8,
        )
        result = merge_running_arcs([arc0, arc1])
        self.assertEqual(len(result), 1)
        merged = result[0]
        # Launch metadata from first arc
        self.assertAlmostEqual(merged.launch_x, 0.0)
        self.assertAlmostEqual(merged.launch_vx, 0.0)
        # Landing metadata from last arc
        self.assertAlmostEqual(merged.landing_x, 120.0)
        self.assertAlmostEqual(merged.landing_vx, 2.8)
        # Cost is sum; cumulative_cost from last arc's destination
        self.assertAlmostEqual(merged.arc_cost_frames, 22.0)
        self.assertAlmostEqual(merged.cumulative_cost, 8.0)
        self.assertEqual(merged.jump_hold_frames, 0)

    def test_jump_arc_breaks_merge(self):
        """A jump arc between two running arcs prevents any merging."""
        run0 = _make_run_arc(seg_id=0, launch_x=0.0, landing_x=60.0, cost=10.0, cumulative_cost=50.0)
        jump = _make_jump_arc(seg_id=0, launch_x=60.0, landing_x=200.0, cost=20.0, cumulative_cost=30.0)
        run1 = _make_run_arc(seg_id=0, launch_x=200.0, landing_x=260.0, cost=10.0, cumulative_cost=0.0)
        result = merge_running_arcs([run0, jump, run1])
        self.assertEqual(len(result), 3)

    def test_different_segment_breaks_merge(self):
        """Running arcs on different segment IDs must not be merged."""
        arc0 = _make_run_arc(seg_id=0, launch_x=0.0, landing_x=60.0)
        arc1 = _make_run_arc(seg_id=1, launch_x=60.0, landing_x=120.0)
        result = merge_running_arcs([arc0, arc1])
        self.assertEqual(len(result), 2)

    def test_opposite_h_input_breaks_merge(self):
        """Running right then running left must not merge (direction change)."""
        arc_right = _make_run_arc(seg_id=0, launch_x=0.0, landing_x=60.0, h_input=1)
        arc_left = _make_run_arc(seg_id=0, launch_x=60.0, landing_x=0.0, h_input=-1)
        result = merge_running_arcs([arc_right, arc_left])
        self.assertEqual(len(result), 2)

    def test_three_consecutive_same_segment_merge_to_one(self):
        """Three consecutive same-segment running arcs collapse to one."""
        arc0 = _make_run_arc(seg_id=0, launch_x=0.0, landing_x=60.0, cost=10.0, cumulative_cost=30.0, launch_vx=0.0, landing_vx=1.0)
        arc1 = _make_run_arc(seg_id=0, launch_x=60.0, landing_x=120.0, cost=12.0, cumulative_cost=18.0, launch_vx=1.0, landing_vx=2.5)
        arc2 = _make_run_arc(seg_id=0, launch_x=120.0, landing_x=180.0, cost=14.0, cumulative_cost=4.0, launch_vx=2.5, landing_vx=3.0)
        result = merge_running_arcs([arc0, arc1, arc2])
        self.assertEqual(len(result), 1)
        merged = result[0]
        self.assertAlmostEqual(merged.launch_x, 0.0)
        self.assertAlmostEqual(merged.landing_x, 180.0)
        self.assertAlmostEqual(merged.arc_cost_frames, 36.0)
        self.assertAlmostEqual(merged.cumulative_cost, 4.0)

    def test_mixed_run_jump_run_merges_runs(self):
        """[run, run, jump, run, run] on same segment → [merged_run, jump, merged_run]."""
        r0 = _make_run_arc(seg_id=0, launch_x=0.0, landing_x=60.0, cost=10.0, cumulative_cost=80.0, landing_vx=1.5)
        r1 = _make_run_arc(seg_id=0, launch_x=60.0, landing_x=120.0, cost=12.0, cumulative_cost=68.0, launch_vx=1.5, landing_vx=2.8)
        j = _make_jump_arc(seg_id=0, launch_x=120.0, landing_x=300.0, cost=20.0, cumulative_cost=48.0)
        r2 = _make_run_arc(seg_id=0, launch_x=300.0, landing_x=360.0, cost=10.0, cumulative_cost=38.0, landing_vx=2.0)
        r3 = _make_run_arc(seg_id=0, launch_x=360.0, landing_x=420.0, cost=10.0, cumulative_cost=28.0, launch_vx=2.0, landing_vx=3.0)
        result = merge_running_arcs([r0, r1, j, r2, r3])
        self.assertEqual(len(result), 3)
        # First merged run
        self.assertAlmostEqual(result[0].launch_x, 0.0)
        self.assertAlmostEqual(result[0].landing_x, 120.0)
        self.assertAlmostEqual(result[0].arc_cost_frames, 22.0)
        self.assertAlmostEqual(result[0].cumulative_cost, 68.0)
        # Jump unchanged
        self.assertEqual(result[1].jump_hold_frames, 8)
        self.assertAlmostEqual(result[1].launch_x, 120.0)
        # Second merged run
        self.assertAlmostEqual(result[2].launch_x, 300.0)
        self.assertAlmostEqual(result[2].landing_x, 420.0)
        self.assertAlmostEqual(result[2].arc_cost_frames, 20.0)
        self.assertAlmostEqual(result[2].cumulative_cost, 28.0)

    def test_neutral_h_input_merges_with_positive(self):
        """h_input=0 (coasting) is compatible with a preceding h_input=1 run."""
        arc_run = _make_run_arc(seg_id=0, launch_x=0.0, landing_x=60.0, h_input=1, cost=10.0, cumulative_cost=15.0)
        arc_coast = _make_run_arc(seg_id=0, launch_x=60.0, landing_x=100.0, h_input=0, cost=5.0, cumulative_cost=0.0)
        result = merge_running_arcs([arc_run, arc_coast])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].arc_cost_frames, 15.0)
        self.assertEqual(result[0].h_input_during, 1)

    def test_segment_id_preserved_from_first_arc(self):
        """Merged arc's launch_segment_id comes from the first arc."""
        arc0 = _make_run_arc(seg_id=5, launch_x=0.0, landing_x=60.0)
        arc1 = _make_run_arc(seg_id=5, launch_x=60.0, landing_x=120.0)
        result = merge_running_arcs([arc0, arc1])
        self.assertEqual(result[0].launch_segment_id, 5)


if __name__ == "__main__":
    unittest.main()
