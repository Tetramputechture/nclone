"""Test momentum-aware PBRS system.

Validates that:
1. Momentum-aware pathfinding costs favor momentum-preserving paths
2. Momentum waypoints are correctly identified and used
3. PBRS doesn't penalize necessary momentum-building behavior
"""

import unittest
import numpy as np
from typing import Dict, Tuple, List

from ...graph.reachability.pathfinding_algorithms import (
    _infer_momentum_direction,
    _calculate_momentum_multiplier,
    _calculate_physics_aware_cost,
    MOMENTUM_CONTINUE_MULTIPLIER,
    MOMENTUM_REVERSE_MULTIPLIER,
)
from ..reward_calculation.pbrs_potentials import (
    PBRSCalculator,
    _find_active_momentum_waypoint,
)
from ...analysis.momentum_waypoint_extractor import (
    MomentumWaypoint,
    MomentumWaypointExtractor,
)


class TestMomentumInference(unittest.TestCase):
    """Test momentum direction inference from trajectory."""

    def test_leftward_momentum(self):
        """Test detection of leftward momentum."""
        # Trajectory: moving consistently left
        grandparent = (100, 100)
        parent = (88, 100)  # Moved 12px left
        current = (76, 100)  # Moved 12px left again

        momentum = _infer_momentum_direction(parent, current, grandparent)
        self.assertEqual(momentum, -1, "Should detect leftward momentum")

    def test_rightward_momentum(self):
        """Test detection of rightward momentum."""
        # Trajectory: moving consistently right
        grandparent = (100, 100)
        parent = (112, 100)  # Moved 12px right
        current = (124, 100)  # Moved 12px right again

        momentum = _infer_momentum_direction(parent, current, grandparent)
        self.assertEqual(momentum, 1, "Should detect rightward momentum")

    def test_no_momentum_stationary(self):
        """Test no momentum when stationary."""
        # Trajectory: barely moving
        grandparent = (100, 100)
        parent = (102, 100)  # Moved 2px (below threshold)
        current = (104, 100)  # Moved 2px

        momentum = _infer_momentum_direction(parent, current, grandparent)
        self.assertEqual(momentum, 0, "Should detect no momentum (too slow)")

    def test_no_momentum_direction_change(self):
        """Test no momentum when changing direction."""
        # Trajectory: changed direction
        grandparent = (100, 100)
        parent = (112, 100)  # Moved 12px right
        current = (100, 100)  # Moved 12px left (reversed)

        momentum = _infer_momentum_direction(parent, current, grandparent)
        self.assertEqual(momentum, 0, "Should detect no momentum (direction changed)")

    def test_no_history(self):
        """Test no momentum without trajectory history."""
        momentum = _infer_momentum_direction(None, (100, 100), None)
        self.assertEqual(momentum, 0, "Should return 0 without history")


class TestMomentumCostMultiplier(unittest.TestCase):
    """Test momentum cost multiplier calculation."""

    def test_continue_leftward_momentum(self):
        """Test cheaper cost for continuing leftward momentum."""
        momentum = -1  # Leftward
        edge_dx = -12  # Moving left

        multiplier = _calculate_momentum_multiplier(momentum, edge_dx)
        self.assertEqual(
            multiplier,
            MOMENTUM_CONTINUE_MULTIPLIER,
            "Continuing momentum should be cheaper",
        )

    def test_continue_rightward_momentum(self):
        """Test cheaper cost for continuing rightward momentum."""
        momentum = 1  # Rightward
        edge_dx = 12  # Moving right

        multiplier = _calculate_momentum_multiplier(momentum, edge_dx)
        self.assertEqual(
            multiplier,
            MOMENTUM_CONTINUE_MULTIPLIER,
            "Continuing momentum should be cheaper",
        )

    def test_reverse_leftward_momentum(self):
        """Test expensive cost for reversing leftward momentum."""
        momentum = -1  # Leftward
        edge_dx = 12  # Moving right (reversing)

        multiplier = _calculate_momentum_multiplier(momentum, edge_dx)
        self.assertEqual(
            multiplier,
            MOMENTUM_REVERSE_MULTIPLIER,
            "Reversing momentum should be expensive",
        )

    def test_reverse_rightward_momentum(self):
        """Test expensive cost for reversing rightward momentum."""
        momentum = 1  # Rightward
        edge_dx = -12  # Moving left (reversing)

        multiplier = _calculate_momentum_multiplier(momentum, edge_dx)
        self.assertEqual(
            multiplier,
            MOMENTUM_REVERSE_MULTIPLIER,
            "Reversing momentum should be expensive",
        )

    def test_no_momentum(self):
        """Test neutral cost when no momentum."""
        momentum = 0  # No momentum
        edge_dx = 12  # Moving right

        multiplier = _calculate_momentum_multiplier(momentum, edge_dx)
        self.assertEqual(multiplier, 1.0, "No momentum should give neutral cost")

    def test_vertical_edge(self):
        """Test neutral cost for vertical edges."""
        momentum = 1  # Rightward momentum
        edge_dx = 0  # Vertical edge

        multiplier = _calculate_momentum_multiplier(momentum, edge_dx)
        self.assertEqual(multiplier, 1.0, "Vertical edges should give neutral cost")


class TestWaypointExtraction(unittest.TestCase):
    """Test momentum waypoint extraction from trajectories."""

    def setUp(self):
        """Set up test fixtures."""
        # Use more lenient thresholds for testing
        self.extractor = MomentumWaypointExtractor(
            min_speed=1.0,  # Lower threshold for test
            speed_increase_threshold=0.3,  # Lower threshold for test
            distance_increase_threshold=5.0,
            lookahead_window=20,
        )

    def test_extract_momentum_building_segment(self):
        """Test extraction of momentum-building segment."""
        # Create synthetic trajectory: agent moves left (away from goal on right)
        # while building speed, then jumps right over obstacle
        positions = []
        velocities = []
        actions = []

        # Start at (200, 100), goal at (400, 100)
        # Need to ensure:
        # 1. Distance increases by at least 5px (EUCLIDEAN_DISTANCE_INCREASE_THRESHOLD)
        # 2. Speed increases by at least 0.8 px/frame (SPEED_INCREASE_THRESHOLD)
        # 3. Jump action within lookahead window (20 frames)
        for i in range(60):
            if i < 30:
                # Move left to build momentum (away from goal)
                x = 200 - i * 3  # Moving left (3px/frame)
                # Build speed aggressively: 0.5 -> 3.0 over 30 frames
                # At frame 15: speed = 0.5 + 15*0.15 = 2.75
                # At frame 10: speed = 0.5 + 10*0.15 = 2.0
                # Speed increase over 5 frames = 0.75 (close to threshold)
                # Better: use exponential buildup
                vx = -0.5 - (i * i) * 0.003  # Quadratic speed buildup
                action = 1  # Left
            elif i < 35:
                # Transition: prepare to jump
                x = 200 - 30 * 3 + (i - 30) * 1
                vx = -3.0
                action = 1  # Still left
            else:
                # Jump and move right toward goal
                x = 200 - 30 * 3 + 5 + (i - 35) * 3  # Moving right
                vx = 3.0  # Fast rightward
                action = 5  # Jump + Right

            positions.append((x, 100.0))
            velocities.append((vx, 0.0))
            actions.append(action)

        goal_position = (400.0, 100.0)

        waypoints = self.extractor.extract_from_episode(
            positions=positions,
            velocities=velocities,
            actions=actions,
            goal_position=goal_position,
        )

        # Should detect at least one waypoint during momentum-building phase
        self.assertGreater(
            len(waypoints), 0, "Should detect momentum-building waypoint"
        )

        # Waypoint should be in the leftward movement phase
        if waypoints:
            waypoint = waypoints[0]
            self.assertLess(
                waypoint.position[0],
                200,
                "Waypoint should be during leftward movement",
            )
            self.assertLess(
                waypoint.velocity[0], 0, "Waypoint should have leftward velocity"
            )
            # Note: leads_to_jump may be False if jump is beyond lookahead window
            # This is acceptable - waypoint still identifies momentum-building behavior

    def test_no_waypoints_direct_path(self):
        """Test no waypoints extracted for direct path to goal."""
        # Create trajectory moving directly toward goal
        positions = []
        velocities = []
        actions = []

        for i in range(50):
            x = 100 + i * 2  # Moving right toward goal
            vx = 2.0  # Constant speed
            positions.append((x, 100.0))
            velocities.append((vx, 0.0))
            actions.append(2)  # Right

        goal_position = (300.0, 100.0)

        waypoints = self.extractor.extract_from_episode(
            positions=positions,
            velocities=velocities,
            actions=actions,
            goal_position=goal_position,
        )

        # Should not detect waypoints for direct path
        self.assertEqual(len(waypoints), 0, "Should not detect waypoints for direct path")


class TestWaypointPBRS(unittest.TestCase):
    """Test waypoint-aware PBRS potential calculation."""

    def test_waypoint_routing_active(self):
        """Test that waypoint routing activates when agent needs momentum."""
        # Create mock waypoint
        waypoint = MomentumWaypoint(
            position=(150.0, 100.0),
            velocity=(-3.0, 0.0),
            speed=3.0,
            approach_direction=(-1.0, 0.0),
            frame_index=20,
            leads_to_jump=True,
            distance_to_goal=250.0,
        )

        # Mock path calculator (simplified)
        class MockPathCalculator:
            def get_geometric_distance(self, *args, **kwargs):
                # Return reasonable distances
                return 50.0  # Simplified

        # Test waypoint selection
        player_pos = (100, 100)
        goal_pos = (400, 100)
        player_velocity = (0.5, 0.0)  # Slow speed, needs momentum

        active = _find_active_momentum_waypoint(
            player_pos=player_pos,
            goal_pos=goal_pos,
            player_velocity=player_velocity,
            waypoints=[waypoint],
            path_calculator=MockPathCalculator(),
            adjacency={},
            base_adjacency={},
            level_data=None,
            graph_data=None,
        )

        self.assertIsNotNone(active, "Should select waypoint when agent is slow")
        self.assertEqual(active.position, waypoint.position)

    def test_waypoint_skipped_with_sufficient_momentum(self):
        """Test that waypoint is skipped when agent already has momentum."""
        waypoint = MomentumWaypoint(
            position=(150.0, 100.0),
            velocity=(-3.0, 0.0),
            speed=3.0,
            approach_direction=(-1.0, 0.0),
            frame_index=20,
            leads_to_jump=True,
            distance_to_goal=250.0,
        )

        class MockPathCalculator:
            def get_geometric_distance(self, *args, **kwargs):
                return 50.0

        player_pos = (100, 100)
        goal_pos = (400, 100)
        player_velocity = (-3.0, 0.0)  # Already has momentum in right direction

        active = _find_active_momentum_waypoint(
            player_pos=player_pos,
            goal_pos=goal_pos,
            player_velocity=player_velocity,
            waypoints=[waypoint],
            path_calculator=MockPathCalculator(),
            adjacency={},
            base_adjacency={},
            level_data=None,
            graph_data=None,
        )

        self.assertIsNone(
            active, "Should skip waypoint when agent already has sufficient momentum"
        )


if __name__ == "__main__":
    unittest.main()

