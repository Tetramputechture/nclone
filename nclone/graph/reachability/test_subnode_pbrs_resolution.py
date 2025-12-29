"""
Test suite for sub-node PBRS resolution system.

Verifies that the next-hop projection mechanism provides dense rewards
for agent movements as small as 0.05px per step, despite the 12px grid spacing.
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from nclone.graph.reachability.pathfinding_utils import (
    calculate_geometric_path_distance,
)
from nclone.graph.reachability.path_distance_calculator import (
    CachedPathDistanceCalculator,
)
from nclone.graph.reachability.mine_proximity_cache import (
    MineProximityCostCache,
)
from nclone.graph.level_data import LevelData


class TestSubNodeProjection(unittest.TestCase):
    """Test sub-node projection math for dense PBRS rewards."""

    def setUp(self):
        """Set up simple test graph with known optimal path."""
        # Create simple linear graph: (0,0) → (12,0) → (24,0) → (36,0)
        # This gives us a horizontal path where we can easily test projection
        self.adjacency = {
            (0, 0): [((12, 0), 12.0)],
            (12, 0): [((24, 0), 12.0)],
            (24, 0): [((36, 0), 12.0)],
            (36, 0): [],  # Goal node
        }
        self.base_adjacency = self.adjacency

        # Physics cache: all nodes grounded for simple testing
        self.physics_cache = {
            (0, 0): {"grounded": True, "walled": False},
            (12, 0): {"grounded": True, "walled": False},
            (24, 0): {"grounded": True, "walled": False},
            (36, 0): {"grounded": True, "walled": False},
        }

        # Empty level data (no mines)
        self.level_data = LevelData(
            start_position=(24, 24),  # World space (tile data + 24)
            tiles=np.zeros((23, 42), dtype=np.int32),
            entities=[],
            switch_states={},
        )

        # Create mine proximity cache (empty for this test)
        from nclone.graph.reachability.mine_proximity_cache import (
            MineProximityCostCache,
        )

        self.mine_cache = MineProximityCostCache()

    def test_projection_moving_toward_next_hop(self):
        """Test that moving toward next_hop reduces distance proportionally."""
        # Start position: (24, 24) world = (0, 0) tile data
        # Goal position: (60, 24) world = (36, 0) tile data
        # Agent at start_node, should project to 0

        start_world = (24, 24)  # Exactly at node (0, 0)
        goal_world = (60, 24)  # Exactly at node (36, 0)

        # Calculate baseline distance (agent exactly at start node)
        baseline_dist = calculate_geometric_path_distance(
            start_world,
            goal_world,
            self.adjacency,
            self.base_adjacency,
            physics_cache=self.physics_cache,
            entity_radius=0.0,
            ninja_radius=10.0,
            level_data=self.level_data,
            mine_proximity_cache=self.mine_cache,
        )

        # Expected: 36px (3 edges of 12px) - 10px (ninja_radius) = 26px
        self.assertAlmostEqual(baseline_dist, 26.0, places=1)

        # Now move agent 1px toward next_hop (RIGHT, toward node (12, 0))
        # Agent at (25, 24) world = 1px ahead of start_node
        moved_forward = (25, 24)
        forward_dist = calculate_geometric_path_distance(
            moved_forward,
            goal_world,
            self.adjacency,
            self.base_adjacency,
            physics_cache=self.physics_cache,
            entity_radius=0.0,
            ninja_radius=10.0,
            level_data=self.level_data,
            mine_proximity_cache=self.mine_cache,
        )

        # Expected: baseline - 1px = 25px
        # Agent moved 1px closer along optimal path
        self.assertAlmostEqual(forward_dist, 25.0, places=1)

        # Verify dense feedback: 1px movement = 1px distance reduction
        distance_change = baseline_dist - forward_dist
        self.assertAlmostEqual(distance_change, 1.0, places=1)

    def test_projection_moving_away_from_next_hop(self):
        """Test that moving away from next_hop increases distance proportionally."""
        start_world = (24, 24)  # Exactly at node (0, 0)
        goal_world = (60, 24)  # Exactly at node (36, 0)

        # Baseline distance
        baseline_dist = calculate_geometric_path_distance(
            start_world,
            goal_world,
            self.adjacency,
            self.base_adjacency,
            physics_cache=self.physics_cache,
            entity_radius=0.0,
            ninja_radius=10.0,
            level_data=self.level_data,
            mine_proximity_cache=self.mine_cache,
        )

        # Move agent 1px away from next_hop (LEFT, away from optimal path)
        # Agent at (23, 24) world = 1px behind start_node
        moved_backward = (23, 24)
        backward_dist = calculate_geometric_path_distance(
            moved_backward,
            goal_world,
            self.adjacency,
            self.base_adjacency,
            physics_cache=self.physics_cache,
            entity_radius=0.0,
            ninja_radius=10.0,
            level_data=self.level_data,
            mine_proximity_cache=self.mine_cache,
        )

        # Expected: baseline + 1px = 27px
        # Agent moved 1px further from goal
        self.assertAlmostEqual(backward_dist, 27.0, places=1)

        # Verify dense feedback: 1px backward = 1px distance increase
        distance_change = backward_dist - baseline_dist
        self.assertAlmostEqual(distance_change, 1.0, places=1)

    def test_micro_movement_produces_nonzero_change(self):
        """Test that micro-movements (0.05px) produce non-zero distance changes."""
        start_world = (24, 24)
        goal_world = (60, 24)

        # Baseline
        baseline_dist = calculate_geometric_path_distance(
            start_world,
            goal_world,
            self.adjacency,
            self.base_adjacency,
            physics_cache=self.physics_cache,
            entity_radius=0.0,
            ninja_radius=10.0,
            level_data=self.level_data,
            mine_proximity_cache=self.mine_cache,
        )

        # Move 0.05px forward (minimum agent movement)
        micro_forward = (24.05, 24)
        micro_dist = calculate_geometric_path_distance(
            micro_forward,
            goal_world,
            self.adjacency,
            self.base_adjacency,
            physics_cache=self.physics_cache,
            entity_radius=0.0,
            ninja_radius=10.0,
            level_data=self.level_data,
            mine_proximity_cache=self.mine_cache,
        )

        # Verify non-zero change
        distance_change = abs(baseline_dist - micro_dist)
        self.assertGreater(
            distance_change,
            0.0,
            "Micro-movement (0.05px) should produce non-zero distance change",
        )

        # Verify proportional change (within tolerance)
        # Expected: ~0.05px change
        self.assertAlmostEqual(distance_change, 0.05, places=2)

    def test_perpendicular_movement_minimal_change(self):
        """Test that perpendicular movement produces minimal distance change."""
        start_world = (24, 24)
        goal_world = (60, 24)

        # Baseline
        baseline_dist = calculate_geometric_path_distance(
            start_world,
            goal_world,
            self.adjacency,
            self.base_adjacency,
            physics_cache=self.physics_cache,
            entity_radius=0.0,
            ninja_radius=10.0,
            level_data=self.level_data,
            mine_proximity_cache=self.mine_cache,
        )

        # Move 1px perpendicular (UP, perpendicular to horizontal path)
        # Projection onto horizontal path direction = 0
        perpendicular = (24, 23)
        perp_dist = calculate_geometric_path_distance(
            perpendicular,
            goal_world,
            self.adjacency,
            self.base_adjacency,
            physics_cache=self.physics_cache,
            entity_radius=0.0,
            ninja_radius=10.0,
            level_data=self.level_data,
            mine_proximity_cache=self.mine_cache,
        )

        # Distance should be nearly identical (projection ≈ 0)
        distance_change = abs(baseline_dist - perp_dist)
        self.assertLess(
            distance_change,
            0.1,
            "Perpendicular movement should produce minimal distance change",
        )


class TestSubNodeWithCachedPathCalculator(unittest.TestCase):
    """Test sub-node resolution with full CachedPathDistanceCalculator."""

    def setUp(self):
        """Set up calculator with simple test graph."""
        # Create simple 4-node linear graph
        self.adjacency = {
            (0, 0): [((12, 0), 12.0)],
            (12, 0): [((24, 0), 12.0)],
            (24, 0): [((36, 0), 12.0)],
            (36, 0): [],
        }
        self.base_adjacency = self.adjacency

        # Physics cache
        self.physics_cache = {
            (0, 0): {"grounded": True, "walled": False},
            (12, 0): {"grounded": True, "walled": False},
            (24, 0): {"grounded": True, "walled": False},
            (36, 0): {"grounded": True, "walled": False},
        }

        # Level data
        self.level_data = LevelData(
            start_position=(24, 24),
            tiles=np.zeros((23, 42), dtype=np.int32),
            entities=[],
            switch_states={},
        )

        # Graph data
        self.graph_data = {
            "adjacency": self.adjacency,
            "base_adjacency": self.base_adjacency,
            "node_physics": self.physics_cache,
            "spatial_hash": None,
            "subcell_lookup": None,
        }

        # Create calculator
        self.calculator = CachedPathDistanceCalculator(use_astar=True)

        # Build level cache
        from nclone.graph.reachability.mine_proximity_cache import (
            MineProximityCostCache,
        )

        self.calculator.mine_proximity_cache = MineProximityCostCache()
        self.calculator.build_level_cache(
            self.level_data, self.adjacency, self.base_adjacency, self.graph_data
        )

    def test_fast_path_sub_node_resolution(self):
        """Test fast path (level cache hit) applies sub-node resolution."""
        # Start at node (0, 0) in world space (24, 24)
        # Goal at node (36, 0) in world space (60, 24)
        start_world = (24, 24)
        goal_world = (60, 24)

        # Get baseline distance (agent at node center)
        baseline = self.calculator.get_geometric_distance(
            start_world,
            goal_world,
            self.adjacency,
            self.base_adjacency,
            level_data=self.level_data,
            graph_data=self.graph_data,
            entity_radius=0.0,
            ninja_radius=10.0,
            goal_id="test_goal",
        )

        # Move 2px forward (toward next_hop)
        forward_pos = (26, 24)
        forward_dist = self.calculator.get_geometric_distance(
            forward_pos,
            goal_world,
            self.adjacency,
            self.base_adjacency,
            level_data=self.level_data,
            graph_data=self.graph_data,
            entity_radius=0.0,
            ninja_radius=10.0,
            goal_id="test_goal",
        )

        # Distance should decrease by ~2px
        change = baseline - forward_dist
        self.assertAlmostEqual(change, 2.0, places=1)

    def test_consistent_between_fast_and_slow_paths(self):
        """Test that fast path and slow path produce similar results."""
        # This test verifies both paths use sub-node resolution consistently
        # Fast path: uses level cache with next_hop projection
        # Slow path: uses BFS with path-based next_hop extraction

        # Position between nodes to trigger sub-node resolution
        start_world = (26, 24)  # 2px ahead of node (0, 0)
        goal_world = (60, 24)

        # Get distance using calculator (should hit level cache = fast path)
        fast_dist = self.calculator.get_geometric_distance(
            start_world,
            goal_world,
            self.adjacency,
            self.base_adjacency,
            level_data=self.level_data,
            graph_data=self.graph_data,
            entity_radius=0.0,
            ninja_radius=10.0,
            goal_id="test_goal",
        )

        # Get distance using direct function (uses BFS = slow path)
        slow_dist = calculate_geometric_path_distance(
            start_world,
            goal_world,
            self.adjacency,
            self.base_adjacency,
            physics_cache=self.physics_cache,
            entity_radius=0.0,
            ninja_radius=10.0,
            level_data=self.level_data,
            mine_proximity_cache=self.calculator.mine_proximity_cache,
        )

        # Both should produce very similar results (within 0.5px tolerance)
        self.assertAlmostEqual(fast_dist, slow_dist, delta=0.5)


class TestSubNodeWithComplexPath(unittest.TestCase):
    """Test sub-node resolution with path that requires detour."""

    def setUp(self):
        """Set up graph with obstacle requiring detour."""
        # Create L-shaped path: must go DOWN then RIGHT
        # (0,0) → (0,12) → (12,12) → (24,12)
        #
        # Goal is at (24, 12), directly RIGHT of start
        # But optimal path goes DOWN first (obstacle blocks direct path)
        self.adjacency = {
            (0, 0): [((0, 12), 12.0)],  # DOWN only (no direct RIGHT)
            (0, 12): [((12, 12), 12.0)],  # RIGHT from bottom
            (12, 12): [((24, 12), 12.0)],  # RIGHT to goal
            (24, 12): [],  # Goal
        }
        self.base_adjacency = self.adjacency

        # Physics cache
        self.physics_cache = {
            (0, 0): {"grounded": True, "walled": False},
            (0, 12): {"grounded": True, "walled": False},
            (12, 12): {"grounded": True, "walled": False},
            (24, 12): {"grounded": True, "walled": False},
        }

        # Level data
        self.level_data = LevelData(
            start_position=(24, 24),
            tiles=np.zeros((23, 42), dtype=np.int32),
            entities=[],
            switch_states={},
        )

    def test_path_aware_not_euclidean(self):
        """Test that projection uses path direction, not Euclidean direction."""
        # Agent at start node (0, 0) in tile space = (24, 24) in world space
        # Goal at (24, 12) in tile space = (48, 36) in world space
        # Optimal path goes DOWN first (next_hop = (0, 12))

        start_world = (24, 24)  # At start node
        goal_world = (48, 36)  # At goal node

        # Baseline distance (at start node)
        baseline = calculate_geometric_path_distance(
            start_world,
            goal_world,
            self.adjacency,
            self.base_adjacency,
            physics_cache=self.physics_cache,
            entity_radius=0.0,
            ninja_radius=10.0,
            level_data=self.level_data,
            mine_proximity_cache=MineProximityCostCache(),
        )

        # Move 1px RIGHT (toward goal in Euclidean sense, but AWAY from next_hop)
        # next_hop is (0, 12) which is DOWN, so moving RIGHT is perpendicular/negative
        moved_right = (25, 24)
        right_dist = calculate_geometric_path_distance(
            moved_right,
            goal_world,
            self.adjacency,
            self.base_adjacency,
            physics_cache=self.physics_cache,
            entity_radius=0.0,
            ninja_radius=10.0,
            level_data=self.level_data,
            mine_proximity_cache=MineProximityCostCache(),
        )

        # Moving RIGHT (perpendicular to path) should NOT reduce distance much
        # Path direction is DOWN, so RIGHT movement projects to ~0
        change = baseline - right_dist
        self.assertLess(
            abs(change),
            0.5,
            "Moving perpendicular to optimal path should not reduce distance",
        )

        # Move 1px DOWN (toward next_hop on optimal path)
        moved_down = (24, 25)
        down_dist = calculate_geometric_path_distance(
            moved_down,
            goal_world,
            self.adjacency,
            self.base_adjacency,
            physics_cache=self.physics_cache,
            entity_radius=0.0,
            ninja_radius=10.0,
            level_data=self.level_data,
            mine_proximity_cache=MineProximityCostCache(),
        )

        # Moving DOWN (along optimal path) should reduce distance by ~1px
        down_change = baseline - down_dist
        self.assertAlmostEqual(
            down_change, 1.0, places=1, msg="Moving along optimal path should reduce distance"
        )


class TestDenseRewardGeneration(unittest.TestCase):
    """Test that PBRS provides dense rewards for small movements."""

    def test_movement_spectrum(self):
        """Test reward density across full movement spectrum (0.05px to 3.33px)."""
        # Create simple linear graph
        adjacency = {
            (0, 0): [((12, 0), 12.0)],
            (12, 0): [((24, 0), 12.0)],
            (24, 0): [],
        }
        base_adjacency = adjacency
        physics_cache = {
            (0, 0): {"grounded": True, "walled": False},
            (12, 0): {"grounded": True, "walled": False},
            (24, 0): {"grounded": True, "walled": False},
        }
        level_data = LevelData(
            start_position=(24, 24),
            tiles=np.zeros((23, 42), dtype=np.int32),
            entities=[],
            switch_states={},
        )
        mine_cache = MineProximityCostCache()

        start_world = (24, 24)
        goal_world = (48, 24)

        # Test different movement magnitudes (typical range with frame_skip=4)
        movement_sizes = [0.05, 0.1, 0.5, 1.0, 2.0, 3.33]  # pixels per step
        
        for move_px in movement_sizes:
            # Calculate baseline
            baseline = calculate_geometric_path_distance(
                start_world,
                goal_world,
                adjacency,
                base_adjacency,
                physics_cache=physics_cache,
                entity_radius=0.0,
                ninja_radius=10.0,
                level_data=level_data,
                mine_proximity_cache=mine_cache,
            )

            # Move forward by move_px
            moved_pos = (start_world[0] + move_px, start_world[1])
            moved_dist = calculate_geometric_path_distance(
                moved_pos,
                goal_world,
                adjacency,
                base_adjacency,
                physics_cache=physics_cache,
                entity_radius=0.0,
                ninja_radius=10.0,
                level_data=level_data,
                mine_proximity_cache=mine_cache,
            )

            # Verify proportional distance change
            distance_change = baseline - moved_dist
            self.assertAlmostEqual(
                distance_change,
                move_px,
                places=2,
                msg=f"Moving {move_px}px should reduce distance by ~{move_px}px",
            )

            # Verify reward is non-zero (CRITICAL for learning)
            self.assertNotAlmostEqual(
                distance_change,
                0.0,
                places=3,
                msg=f"Moving {move_px}px MUST produce non-zero distance change for dense PBRS",
            )


def run_tests():
    """Run the test suite."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSubNodeProjection))
    suite.addTests(loader.loadTestsFromTestCase(TestSubNodeWithComplexPath))
    suite.addTests(loader.loadTestsFromTestCase(TestDenseRewardGeneration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

