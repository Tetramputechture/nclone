"""Tests for activation-aware goal distance in the reachability A*.

The geometric distance to a switch/exit must reflect two facts:
  * entities are placed on the 6px grid (world coords ``≡0 mod 6``) while graph
    nodes sit at ``≡3 mod 6`` -> the snapped goal node is up to ~4.24px off the
    true entity center;
  * activation fires when the ninja's EDGE touches the entity's edge, i.e. when
    the ninja CENTER is within ``ninja_radius + entity_radius`` of the entity
    center (overlap_circle_vs_circle in physics.py).

So the reported distance is the center-to-center path distance minus the
combined activation reach, clamped at 0, and is unbiased w.r.t. the side the
ninja approaches from.
"""

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from nclone.constants.entity_types import EntityType
from nclone.constants.physics_constants import (
    NINJA_RADIUS,
    EXIT_SWITCH_RADIUS,
)
from nclone.graph.level_data import LevelData
from nclone.graph.reachability.mine_proximity_cache import MineProximityCostCache
from nclone.graph.reachability.path_distance_calculator import (
    CachedPathDistanceCalculator,
    _signed_goal_residual,
)


class TestSignedGoalResidual(unittest.TestCase):
    """Unit test of the signed node->center projection helper."""

    def test_positive_when_center_beyond_node(self):
        # Node short of the center along the approach (ninja far to the left).
        r = _signed_goal_residual(50.0, 27.0, (54.0, 27.0), (0.0, 27.0))
        self.assertAlmostEqual(r, 4.0, places=6)

    def test_negative_when_node_beyond_center(self):
        # Same geometry, ninja on the right: the node is now PAST the center, so
        # the residual is negative (path overshoots and must come back).
        r = _signed_goal_residual(50.0, 27.0, (54.0, 27.0), (100.0, 27.0))
        self.assertAlmostEqual(r, -4.0, places=6)

    def test_symmetric_magnitude(self):
        left = _signed_goal_residual(50.0, 27.0, (54.0, 27.0), (0.0, 27.0))
        right = _signed_goal_residual(50.0, 27.0, (54.0, 27.0), (108.0, 27.0))
        self.assertAlmostEqual(abs(left), abs(right), places=6)

    def test_degenerate_direction_returns_zero(self):
        # Ninja on top of the center -> no approach direction.
        r = _signed_goal_residual(50.0, 27.0, (54.0, 27.0), (54.0, 27.0))
        self.assertEqual(r, 0.0)


class TestActivationAwareReachability(unittest.TestCase):
    """Integration test through CachedPathDistanceCalculator.get_path_cost."""

    SWITCH_WORLD = (84.0, 24.0)  # on the 6px grid (84 = 14*6, 24 = 4*6)

    def setUp(self):
        # Horizontal corridor of nodes on the realistic 6px grid (tile coords
        # ≡3 mod 6); switch center at x=84 (≡0 mod 6) sits 3px from nodes 81/87,
        # symmetric about the center -> exercises the snap offset + no-bias path.
        # Wide corridor centered on the switch tile (60) so symmetric offsets up
        # to ~100px stay on-graph on BOTH sides (≡3 mod 6 tile coords).
        xs = list(range(-39, 165, 6))  # -39, -33, ..., 159 (tile space)
        node_y = 0
        self.adjacency = {}
        for i, x in enumerate(xs):
            nbrs = []
            if i > 0:
                nbrs.append(((xs[i - 1], node_y), 6.0))
            if i < len(xs) - 1:
                nbrs.append(((xs[i + 1], node_y), 6.0))
            self.adjacency[(x, node_y)] = nbrs
        self.base_adjacency = self.adjacency
        self.physics_cache = {
            (x, node_y): {"grounded": True, "walled": False} for x in xs
        }
        self.level_data = LevelData(
            start_position=(27, 24),
            tiles=np.zeros((23, 42), dtype=np.int32),
            entities=[
                {
                    "type": EntityType.EXIT_SWITCH,
                    "x": self.SWITCH_WORLD[0],
                    "y": self.SWITCH_WORLD[1],
                    "active": True,
                }
            ],
            switch_states={},
        )
        self.graph_data = {
            "adjacency": self.adjacency,
            "base_adjacency": self.base_adjacency,
            "node_physics": self.physics_cache,
            "spatial_hash": None,
            "subcell_lookup": None,
        }
        self.calc = CachedPathDistanceCalculator()
        self.calc.mine_proximity_cache = MineProximityCostCache()
        self.calc.build_level_cache(
            self.level_data, self.adjacency, self.base_adjacency, self.graph_data
        )

    def _dist(self, ninja_world):
        return self.calc.get_path_cost(
            ninja_world,
            self.SWITCH_WORLD,
            self.adjacency,
            self.base_adjacency,
            level_data=self.level_data,
            graph_data=self.graph_data,
            entity_radius=EXIT_SWITCH_RADIUS,
            ninja_radius=NINJA_RADIUS,
            goal_id="switch",
        )

    def test_zero_inside_activation_reach(self):
        reach = NINJA_RADIUS + EXIT_SWITCH_RADIUS  # 16
        sx, sy = self.SWITCH_WORLD
        # Ninja center within the combined reach -> already activating -> 0.
        self.assertEqual(self._dist((sx, sy)), 0.0)
        self.assertEqual(self._dist((sx + (reach - 2), sy)), 0.0)

    def test_positive_outside_reach(self):
        sx, sy = self.SWITCH_WORLD
        self.assertGreater(self._dist((sx + 48, sy)), 0.0)

    def test_no_left_right_snap_bias(self):
        # Equidistant ninja positions on either side of the switch return nearly
        # identical distances: the signed goal residual removes the ~3px snap bias
        # that an unsigned node-distance would introduce.
        sx, sy = self.SWITCH_WORLD
        for offset in (36.0, 60.0, 90.0):
            left = self._dist((sx - offset, sy))
            right = self._dist((sx + offset, sy))
            self.assertLess(
                abs(left - right),
                1.0,
                msg=f"asymmetry {abs(left - right):.3f}px at offset {offset}px",
            )

    def test_magnitude_is_center_to_center_minus_reach(self):
        # Far along the (straight, on-grid) corridor the path distance equals the
        # center-to-center distance, so the value is that minus the activation
        # reach (within sub-pixel snap/quantization tolerance).
        reach = NINJA_RADIUS + EXIT_SWITCH_RADIUS
        sx, sy = self.SWITCH_WORLD
        offset = 60.0
        expected = offset - reach
        self.assertAlmostEqual(self._dist((sx + offset, sy)), expected, delta=1.0)


if __name__ == "__main__":
    unittest.main()
