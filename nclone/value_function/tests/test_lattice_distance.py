"""Unit tests for LatticeDistanceSource (reward distance signal).

Bypasses the heavy precompute by injecting a synthetic graph + d* arrays, so we
test the lookup semantics deterministically: velocity-agnostic min-over-bins,
phase selection (d_comb vs d_ex), transient-inf clamping, and spawn_finite.
"""

import unittest

import numpy as np

from nclone.value_function.surface_graph import (
    build_graph,
    node_id,
    NUM_VELOCITY_BINS_FLOOR,
)
from nclone.value_function.surface_extraction import SurfacePoint, SURFACE_FLOOR
from nclone.value_function.lattice_distance import LatticeDistanceSource


def _floor_point(x, y, seg):
    return SurfacePoint(x=x, y=y, normal_x=0.0, normal_y=-1.0,
                        surface_type=SURFACE_FLOOR, segment_id=seg, is_edge=False)


def _make_source():
    """Three floor points at x=0,60,120 (y=100); inject d* arrays directly."""
    pts = [_floor_point(0.0, 100.0, 0), _floor_point(60.0, 100.0, 1),
           _floor_point(120.0, 100.0, 2)]
    g = build_graph(pts)
    d_comb = np.full(g.n_nodes, np.inf, dtype=np.float32)
    d_ex = np.full(g.n_nodes, np.inf, dtype=np.float32)
    mvb = g.max_vel_bins
    # phase-0 (d_comb): cost-to-completion decreasing toward x=120.
    d_comb[node_id(0, 7, mvb)] = 50.0
    d_comb[node_id(1, 7, mvb)] = 30.0
    d_comb[node_id(2, 7, mvb)] = 10.0
    # a cheaper bin at point 0 to verify min-over-bins
    d_comb[node_id(0, 3, mvb)] = 45.0
    # phase-1 (d_ex): different scale.
    d_ex[node_id(0, 7, mvb)] = 20.0
    d_ex[node_id(2, 7, mvb)] = 5.0

    src = LatticeDistanceSource.__new__(LatticeDistanceSource)
    src.available = True
    src._graph = g
    src._d_comb = d_comb
    src._d_ex = d_ex
    src._d_max_comb = float(d_comb[np.isfinite(d_comb)].max())
    src._d_max_ex = float(d_ex[np.isfinite(d_ex)].max())
    src.switch = (60.0, 100.0)
    src.exit = (120.0, 100.0)
    return src, mvb


class TestLatticeDistanceSource(unittest.TestCase):
    def test_phase0_uses_d_comb_min_over_bins(self):
        src, _ = _make_source()
        # At x=0 the cheapest finite bin is 45.0 (not the 50.0 at bin 7).
        self.assertAlmostEqual(src.cost_to_go(0.0, 100.0, False), 45.0, places=3)
        self.assertAlmostEqual(src.cost_to_go(120.0, 100.0, False), 10.0, places=3)

    def test_phase1_uses_d_ex(self):
        src, _ = _make_source()
        self.assertAlmostEqual(src.cost_to_go(0.0, 100.0, True), 20.0, places=3)
        self.assertAlmostEqual(src.cost_to_go(120.0, 100.0, True), 5.0, places=3)

    def test_transient_inf_clamped_to_dmax(self):
        # x=60 has NO finite d_ex bin -> clamp to per-level max finite d_ex (20),
        # never inf (no scale jump mid-episode).
        src, _ = _make_source()
        c = src.cost_to_go(60.0, 100.0, True)
        self.assertTrue(np.isfinite(c))
        self.assertAlmostEqual(c, src._d_max_ex, places=3)

    def test_no_surface_point_returns_inf(self):
        src, _ = _make_source()
        self.assertFalse(np.isfinite(src.cost_to_go(10000.0, 10000.0, False)))

    def test_spawn_finite(self):
        src, _ = _make_source()
        self.assertTrue(src.spawn_finite(0.0, 100.0))
        self.assertFalse(src.spawn_finite(10000.0, 10000.0))

    def test_unavailable_source_is_inf(self):
        src = LatticeDistanceSource.__new__(LatticeDistanceSource)
        src.available = False
        self.assertFalse(np.isfinite(src.cost_to_go(0.0, 0.0, False)))
        self.assertFalse(src.spawn_finite(0.0, 0.0))

    def test_nav_feature_points_down_gradient(self):
        # Points at x=0,60,120 with d_comb 50,30,10 (decreasing toward +x).
        # step_px=60 lands neighbors on adjacent points; from x=60 the +x
        # neighbor (x=120, cost 10) beats c0=30 -> next-hop = +x.
        src, _ = _make_source()
        f = src.nav_feature(60.0, 100.0, False, step_px=60.0)
        self.assertEqual(len(f), 4)
        self.assertAlmostEqual(f[0], 1.0, places=3)   # dx_hop = +x
        self.assertAlmostEqual(f[1], 0.0, places=3)   # dy_hop = 0
        self.assertAlmostEqual(f[2], 30.0 / 50.0, places=3)  # cost_to_go_norm
        self.assertEqual(f[3], 0.0)                   # phase 0

    def test_nav_feature_zero_at_min(self):
        # At x=120 (cost 10, the minimum) no neighbour is lower -> no direction.
        src, _ = _make_source()
        f = src.nav_feature(120.0, 100.0, False, step_px=60.0)
        self.assertAlmostEqual(f[0], 0.0, places=3)
        self.assertAlmostEqual(f[1], 0.0, places=3)

    def test_nav_feature_unavailable_zeros(self):
        src = LatticeDistanceSource.__new__(LatticeDistanceSource)
        src.available = False
        np.testing.assert_array_equal(
            src.nav_feature(0.0, 0.0, False), np.zeros(4, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
