"""Validation tests for the slope-jump trick primitive + edges.

Asserts:
  1. the adaptive primitive FIRES off a real slope (leaves the ground, gains
     height) on the ``squeeze`` test map's hump slopes;
  2. ``enumerate_slope_jump_edges`` produces routable edges (valid src/dst,
     positive climb) off the slope FLOOR nodes — covering all slope types;
  3. PARITY: slope-jump edges are OFF by default, so the precomputed d* /
     enumerated edge set is byte-identical with and without the (default-empty)
     flag; enabling it is purely additive (only adds trick edges).
"""

import os
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

from nclone.value_function.tools.physics_route_spike import _load_map_data
from nclone.value_function.lattice_distance import LatticeDistanceSource

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SQUEEZE = _REPO_ROOT / "nclone/test_maps/squeeze"


@unittest.skipUnless(_SQUEEZE.exists(), f"map not found: {_SQUEEZE}")
class TestSlopeJump(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md = _load_map_data(str(_SQUEEZE))

    def test_primitive_fires_off_slope(self):
        """The adaptive slope-jump leaves the ground and gains height on a slope."""
        import nclone_native as N
        from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper
        from nclone.value_function.surface_extraction import (
            extract_surfaces, SURFACE_FLOOR,
        )
        from nclone.value_function.slope_jump_primitive import execute_slope_jump

        phys = PhysicsStepWrapper(map_data=self.md)
        sps = extract_surfaces(phys.tile_dic, mine_positions=[],
                               momentum_model="tangential")
        slopes = [sp for sp in sps
                  if sp.surface_type == SURFACE_FLOOR and abs(sp.normal_x) > 0.05]
        self.assertGreater(len(slopes), 0, "squeeze should have slope nodes")

        sim = N.Simulator.load(self.md)
        fired_any = False
        best_climb = 0.0
        for sp in slopes:
            sim.fast_reset()
            sim.inject_ninja_for_value_iteration_grounded(
                float(sp.x), float(sp.y), 0.0, 0.0,
                float(sp.normal_x), float(sp.normal_y))
            ex = execute_slope_jump(sim, approach_dir=1, jump_dirs=(1, -1),
                                    hold_frames=35, approach_frames=10)
            if ex.fired:
                fired_any = True
                best_climb = max(best_climb, ex.result.climb_px)
        self.assertTrue(fired_any, "no slope jump fired off any slope node")
        self.assertGreater(best_climb, 30.0,
                           "slope jump should gain meaningful height")

    def test_enumerate_edges_are_routable(self):
        """Enumerated slope-jump edges have valid src/dst and positive climb."""
        import nclone_native as N
        from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper
        from nclone.value_function.surface_extraction import extract_surfaces
        from nclone.value_function.surface_graph import build_graph
        from nclone.value_function.edge_enumeration import (
            enumerate_edges, populate_graph_edges,
        )
        from nclone.value_function.slope_jump_edges import enumerate_slope_jump_edges
        from nclone.value_function.tools.physics_route_spike import _goal_positions

        phys = PhysicsStepWrapper(map_data=self.md)
        sps = extract_surfaces(phys.tile_dic, mine_positions=[],
                               momentum_model="legacy")
        g = build_graph(sps)
        g.momentum_model = "legacy"
        edges, _ = enumerate_edges(g, self.md, num_workers=4,
                                   collect_waypoints=False, use_native=True)
        populate_graph_edges(g, edges)
        sim = N.Simulator.load(self.md)
        sw, ex, _ = _goal_positions(sim)
        tedges = enumerate_slope_jump_edges(g, sim, sw, ex)
        self.assertGreater(len(tedges), 0, "no slope-jump edges enumerated")
        for t in tedges:
            self.assertGreaterEqual(t.climb_px, 40.0)
            self.assertEqual(t.launch_mode, "slope_jump")
            self.assertGreater(t.cost_frames, 0)
            if t.dst_node is not None:
                self.assertNotEqual(t.dst_node, t.src_node)
                self.assertTrue(0 <= t.dst_node < g.n_nodes)

    def test_parity_default_off(self):
        """Default (flag empty) adds NO trick edges; enabling is purely additive."""
        off = LatticeDistanceSource(self.md, use_cache=False, name="squeeze",
                                    slope_jump_modes=None)
        on = LatticeDistanceSource(self.md, use_cache=False, name="squeeze",
                                   slope_jump_modes="on")
        self.assertTrue(off.available and on.available)
        g_off, g_on = off._graph, on._graph
        # Same node set; ON has >= edges (only trick edges added).
        self.assertEqual(g_off.n_nodes, g_on.n_nodes)
        n_trick_off = len(getattr(g_off, "trick_tapes", None) or {})
        n_trick_on = len(getattr(g_on, "trick_tapes", None) or {})
        self.assertEqual(n_trick_off, 0, "default must add no slope-jump edges")
        self.assertGreater(n_trick_on, 0, "enabled should add slope-jump edges")
        self.assertEqual(len(g_on.edges_src), len(g_off.edges_src) + n_trick_on)
        # The base (non-trick) edge set is byte-identical (additive only).
        base = len(g_off.edges_src)
        self.assertTrue(np.array_equal(g_off.edges_src, g_on.edges_src[:base]))
        self.assertTrue(np.array_equal(g_off.edges_dst, g_on.edges_dst[:base]))
        self.assertTrue(np.array_equal(g_off.edges_weight, g_on.edges_weight[:base]))


if __name__ == "__main__":
    unittest.main()
