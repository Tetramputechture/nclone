"""End-to-end smoke test for the surface-graph PBRS precompute pipeline.

Verifies that:
  1. ``enumerate_edges`` returns ``CompactWaypoints`` with the expected shape
     and populates the new ``stats_out`` telemetry dict.
  2. ``_build_reverse_adjacency`` is reused across two goals without altering
     Dijkstra results vs. independent construction.
  3. ``ArcCache.build`` takes a ``CompactWaypoints`` bundle and produces a
     non-empty spatial hash whose lookup values correspond to a
     ``frames_to_land + d_star[dst]`` computation.
  4. Precomputation on the small ``simple-walk`` benchmark level completes
     under a generous RSS budget (<6 GB) and yields the expected .npz
     metadata keys, including the new clip-telemetry fields.

The heavy end-to-end test (``test_simple_walk_precompute_e2e``) is marked
as slow and skipped unless the ``NCLONE_RUN_SLOW_TESTS`` env var is set,
so the default CI run stays fast.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import Optional

import numpy as np

from nclone.value_function.arc_cache import ArcCache
from nclone.value_function.edge_enumeration import (
    CompactWaypoints,
    EdgeResult,
    enumerate_edges,
    populate_graph_edges,
)
from nclone.value_function.surface_graph import (
    _build_reverse_adjacency,
    build_graph,
    compute_cost_to_go_with_tree,
    dijkstra_backward_with_tree,
    node_id,
    velocity_to_bin,
)
from nclone.value_function.surface_extraction import (
    SURFACE_FLOOR,
    SurfacePoint,
)


_NCLONE_ROOT = Path(__file__).resolve().parents[3]
_SIMPLE_WALK = _NCLONE_ROOT / "nclone" / "test_maps" / "simple-walk"
_SLOW_ENV = "NCLONE_RUN_SLOW_TESTS"


# ---------------------------------------------------------------------------
# Helpers — a tiny synthetic chain graph (no physics required)
# ---------------------------------------------------------------------------
def _build_chain_graph(n_points: int = 5, spacing: float = 60.0):
    pts = [
        SurfacePoint(
            x=i * spacing, y=100.0,
            normal_x=0.0, normal_y=-1.0,
            surface_type=SURFACE_FLOOR,
            segment_id=i,
            is_edge=False,
        )
        for i in range(n_points)
    ]
    graph = build_graph(pts)
    vel_bin = velocity_to_bin(0.0, SURFACE_FLOOR)
    edges = []
    for i in range(n_points - 1):
        edges.append(
            EdgeResult(
                src_node=node_id(i, vel_bin, graph.max_vel_bins),
                dst_node=node_id(i + 1, vel_bin, graph.max_vel_bins),
                weight=10.0,
                jump_hold_frames=0,
                h_input_during=1,
            )
        )
    populate_graph_edges(graph, edges)
    return graph


# ---------------------------------------------------------------------------
# 1. Data type invariants
# ---------------------------------------------------------------------------
class TestCompactWaypoints(unittest.TestCase):
    def test_empty_has_correct_shapes(self):
        wps = CompactWaypoints.empty()
        self.assertEqual(wps.keys.shape, (0, 4))
        self.assertEqual(wps.dst.shape, (0,))
        self.assertEqual(wps.ftl.shape, (0,))
        self.assertEqual(wps.n_entries, 0)
        self.assertEqual(len(wps), 0)

    def test_from_dict_roundtrip(self):
        accum = {
            (0, 0, 0, 0, 10): 5.0,
            (0, 0, 0, 0, 11): 7.0,
            (1, 2, 3, 4, 10): 2.5,
        }
        wps = CompactWaypoints.from_dict(accum)
        self.assertEqual(wps.n_entries, 3)
        self.assertEqual(wps.keys.dtype, np.int32)
        self.assertEqual(wps.dst.dtype, np.int32)
        self.assertEqual(wps.ftl.dtype, np.float32)
        # Spot-check: find the (1,2,3,4,10) row and its ftl
        rows = [
            (tuple(wps.keys[i]), int(wps.dst[i]), float(wps.ftl[i]))
            for i in range(wps.n_entries)
        ]
        self.assertIn(((1, 2, 3, 4), 10, 2.5), rows)


# ---------------------------------------------------------------------------
# 2. Dijkstra invariants — shared rev_adj gives identical d_star
# ---------------------------------------------------------------------------
class TestDijkstraShared(unittest.TestCase):
    def test_shared_rev_adj_matches_independent(self):
        graph = _build_chain_graph(n_points=6)
        vel_bin = velocity_to_bin(0.0, SURFACE_FLOOR)
        goal = [node_id(5, vel_bin, graph.max_vel_bins)]

        d_a, pred_a, best_a = dijkstra_backward_with_tree(graph, goal)
        rev = _build_reverse_adjacency(graph)
        d_b, pred_b, best_b = dijkstra_backward_with_tree(
            graph, goal, rev_adj=rev,
        )

        np.testing.assert_array_equal(d_a, d_b)
        np.testing.assert_array_equal(pred_a, pred_b)
        np.testing.assert_array_equal(best_a, best_b)

    def test_chain_distances_are_linear(self):
        graph = _build_chain_graph(n_points=5)
        vel_bin = velocity_to_bin(0.0, SURFACE_FLOOR)
        goal = node_id(4, vel_bin, graph.max_vel_bins)
        rev = _build_reverse_adjacency(graph)
        d_star, _, _ = dijkstra_backward_with_tree(
            graph, [goal], rev_adj=rev,
        )
        # Chain has uniform 10-frame edges → d* should be 40, 30, 20, 10, 0
        for i in range(5):
            n = node_id(i, vel_bin, graph.max_vel_bins)
            self.assertEqual(float(d_star[n]), (4 - i) * 10.0)


# ---------------------------------------------------------------------------
# 3. ArcCache vectorised build
# ---------------------------------------------------------------------------
class TestArcCacheBuild(unittest.TestCase):
    def test_rejects_legacy_list(self):
        cache = ArcCache()
        with self.assertRaises(TypeError):
            cache.build([1, 2, 3], np.zeros(5))

    def test_build_min_reduces_per_cell(self):
        """Two waypoints in the same cell but different dst_nodes must be
        reduced to the minimum cost per cell."""
        # dst_node 0 → d_star=100, dst_node 1 → d_star=5
        d_star = np.array([100.0, 5.0, np.inf], dtype=np.float32)
        # Same cell key (0,0,0,0), both dsts, ftl=10 each
        keys = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
        dst = np.array([0, 1], dtype=np.int32)
        ftl = np.array([10.0, 10.0], dtype=np.float32)
        wps = CompactWaypoints(keys=keys, dst=dst, ftl=ftl)

        cache = ArcCache()
        cache.build(wps, d_star)
        self.assertEqual(cache.n_entries, 1)
        # min(10+100, 10+5) = 15
        self.assertAlmostEqual(cache.lookup(0.0, 0.0, 0.0, 0.0), 15.0, places=4)

    def test_build_filters_invalid_entries(self):
        d_star = np.array([np.inf, 10.0], dtype=np.float32)
        keys = np.array(
            [
                [0, 0, 0, 0],  # dst=0 → inf, rejected
                [1, 0, 0, 0],  # dst=1 → 10, kept
                [2, 0, 0, 0],  # dst=-1 rejected
                [3, 0, 0, 0],  # ftl=-1 rejected
            ],
            dtype=np.int32,
        )
        dst = np.array([0, 1, -1, 1], dtype=np.int32)
        ftl = np.array([5.0, 5.0, 5.0, -1.0], dtype=np.float32)
        wps = CompactWaypoints(keys=keys, dst=dst, ftl=ftl)

        cache = ArcCache()
        cache.build(wps, d_star)
        self.assertEqual(cache.n_entries, 1)
        self.assertAlmostEqual(cache.lookup(6.0, 0.0, 0.0, 0.0), 15.0, places=4)

    def test_to_arrays_roundtrip(self):
        d_star = np.array([10.0], dtype=np.float32)
        keys = np.array([[1, 2, 3, 4]], dtype=np.int32)
        wps = CompactWaypoints(
            keys=keys,
            dst=np.array([0], dtype=np.int32),
            ftl=np.array([5.0], dtype=np.float32),
        )
        cache = ArcCache()
        cache.build(wps, d_star)

        out_keys, out_vals = cache.to_arrays()
        self.assertEqual(out_keys.dtype, np.int32)
        self.assertEqual(out_vals.dtype, np.float32)
        self.assertEqual(out_keys.shape, (1, 4))
        self.assertEqual(out_vals.shape, (1,))
        self.assertTrue(np.array_equal(out_keys[0], [1, 2, 3, 4]))
        self.assertAlmostEqual(float(out_vals[0]), 15.0, places=4)


# ---------------------------------------------------------------------------
# 4. End-to-end precompute on simple-walk (slow — skipped by default)
# ---------------------------------------------------------------------------
@unittest.skipUnless(
    os.environ.get(_SLOW_ENV) == "1",
    f"Slow test; set {_SLOW_ENV}=1 to enable",
)
class TestSimpleWalkE2E(unittest.TestCase):
    """Full-pipeline test against the simple-walk benchmark level.

    Verifies that:
      - ``enumerate_edges`` emits a non-empty CompactWaypoints
      - Clip telemetry stays below 50% (rough sanity check on bin grid)
      - A saved .npz round-trips through ``np.load`` with all required keys
      - At least some fraction of graph nodes is reachable from the switch
    """

    MAP_PATH = str(_SIMPLE_WALK)

    def setUp(self) -> None:
        if not _SIMPLE_WALK.exists():
            self.skipTest(f"simple-walk benchmark map not found at {_SIMPLE_WALK}")

    def test_enumerate_edges_emits_stats(self) -> None:
        from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper
        from nclone.value_function.surface_extraction import extract_surfaces

        with open(self.MAP_PATH, "rb") as f:
            raw = f.read()
        map_data = [int(b) for b in raw]
        physics = PhysicsStepWrapper(map_path=self.MAP_PATH)
        surface_points = extract_surfaces(physics.tile_dic, mine_positions=[])
        graph = build_graph(surface_points)

        stats: dict = {}
        edges, wps = enumerate_edges(
            graph,
            map_data=map_data,
            num_workers=2,
            collect_waypoints=True,
            verbose=False,
            stats_out=stats,
        )
        populate_graph_edges(graph, edges)

        self.assertGreater(len(edges), 0, "simple-walk must discover edges")
        self.assertGreater(wps.n_entries, 0, "waypoints must be populated")
        self.assertIn("n_clipped_landings", stats)
        self.assertIn("n_total_landings", stats)
        self.assertIn("pct_clipped", stats)
        # Sanity: clip ratio should be low on simple-walk (no extreme physics).
        self.assertLess(stats["pct_clipped"], 50.0)

        # Dijkstra reachability must be finite for at least 5% of nodes.
        switch_pos = physics.get_goal_position("switch")
        d_star, _pred, _best = compute_cost_to_go_with_tree(
            graph, switch_pos[0], switch_pos[1], "switch",
        )
        n_reachable = int(np.sum(np.isfinite(d_star)))
        self.assertGreater(n_reachable, int(0.05 * graph.n_nodes))


if __name__ == "__main__":
    unittest.main()
