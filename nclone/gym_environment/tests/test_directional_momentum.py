"""Unit tests for directional momentum bonus (Tier 1 path efficiency)."""

import unittest
from unittest.mock import patch
import numpy as np

from ..reward_calculation.main_reward_calculator import RewardCalculator
from ..reward_calculation.reward_constants import (
    DIRECTIONAL_MOMENTUM_BONUS_PER_STEP,
    BACKWARD_VELOCITY_PENALTY,
)


class TestDirectionalMomentum(unittest.TestCase):
    """Test directional momentum bonus functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.reward_calculator = RewardCalculator()
        
        # Mock adjacency graph and level data
        self.mock_adjacency = {
            (100, 100): [((150, 100), 50.0), ((100, 150), 50.0)],
            (150, 100): [((100, 100), 50.0), ((200, 100), 50.0)],
            (200, 100): [((150, 100), 50.0)],
        }
        self.mock_level_data = Mock()
        self.mock_graph_data = {}
        
        # Base observation with graph data
        self.base_obs = {
            "player_x": 100.0,
            "player_y": 100.0,
            "player_xspeed": 0.0,
            "player_yspeed": 0.0,
            "switch_x": 200.0,
            "switch_y": 100.0,
            "exit_door_x": 300.0,
            "exit_door_y": 100.0,
            "switch_activated": False,
            "_adjacency_graph": self.mock_adjacency,
            "level_data": self.mock_level_data,
            "_graph_data": self.mock_graph_data,
        }

    def test_forward_velocity_rewarded(self):
        """Test that velocity toward objective gets bonus."""
        # Mock path calculator to return decreasing distances (moving forward)
        def mock_get_distance(start, goal, *args, **kwargs):
            if start == (100.0, 100.0):
                return 100.0  # Current distance
            elif start == (120.0, 100.0):  # Sample position ahead
                return 80.0  # Closer (forward progress)
            return float('inf')
        
        self.reward_calculator.path_calculator.get_distance = mock_get_distance
        
        # Set velocity toward objective
        obs = self.base_obs.copy()
        obs["player_xspeed"] = 2.0
        obs["player_yspeed"] = 0.0
        
        # Force direction update
        self.reward_calculator.frame_count = 0
        
        reward = self.reward_calculator.calculate_directional_momentum_bonus(obs)
        
        # Should be positive (forward movement rewarded)
        self.assertGreater(reward, 0.0)
        self.assertAlmostEqual(
            reward,
            np.linalg.norm([2.0, 0.0]) * DIRECTIONAL_MOMENTUM_BONUS_PER_STEP,
            places=5
        )

    def test_backward_velocity_penalized(self):
        """Test that velocity away from objective gets penalty."""
        # Mock path calculator to return increasing distances (moving backward)
        def mock_get_distance(start, goal, *args, **kwargs):
            if start == (100.0, 100.0):
                return 100.0  # Current distance
            elif start == (80.0, 100.0):  # Sample position behind
                return 120.0  # Farther (backward movement)
            return float('inf')
        
        self.reward_calculator.path_calculator.get_distance = mock_get_distance
        
        # Set velocity away from objective
        obs = self.base_obs.copy()
        obs["player_xspeed"] = -2.0  # Moving left (away from switch at x=200)
        obs["player_yspeed"] = 0.0
        
        # Force direction update
        self.reward_calculator.frame_count = 0
        
        reward = self.reward_calculator.calculate_directional_momentum_bonus(obs)
        
        # Should be negative (backward movement penalized)
        self.assertLess(reward, 0.0)
        self.assertAlmostEqual(
            abs(reward),
            np.linalg.norm([-2.0, 0.0]) * BACKWARD_VELOCITY_PENALTY,
            places=5
        )

    def test_stationary_no_reward(self):
        """Test that no velocity results in no reward."""
        obs = self.base_obs.copy()
        obs["player_xspeed"] = 0.0
        obs["player_yspeed"] = 0.0
        
        reward = self.reward_calculator.calculate_directional_momentum_bonus(obs)
        
        # Should be zero (no movement)
        self.assertEqual(reward, 0.0)

    def test_direction_cache_updates(self):
        """Test that direction cache updates periodically."""
        call_count = [0]
        
        def mock_get_distance(start, goal, *args, **kwargs):
            call_count[0] += 1
            if start == (100.0, 100.0):
                return 100.0
            elif start == (120.0, 100.0):
                return 80.0  # Forward
            return float('inf')
        
        self.reward_calculator.path_calculator.get_distance = mock_get_distance
        
        obs = self.base_obs.copy()
        obs["player_xspeed"] = 2.0
        obs["player_yspeed"] = 0.0
        
        # First call should update cache
        self.reward_calculator.frame_count = 0
        reward1 = self.reward_calculator.calculate_directional_momentum_bonus(obs)
        first_call_count = call_count[0]
        
        # Next few calls should use cache (no pathfinding calls)
        for i in range(1, 5):
            self.reward_calculator.frame_count = i
            reward2 = self.reward_calculator.calculate_directional_momentum_bonus(obs)
            # Reward should be same (using cached direction)
            self.assertAlmostEqual(reward1, reward2, places=5)
        
        # Call count should not have increased (using cache)
        self.assertEqual(call_count[0], first_call_count)
        
        # After update interval, should recalculate
        self.reward_calculator.frame_count = 5  # Update interval
        reward3 = self.reward_calculator.calculate_directional_momentum_bonus(obs)
        # Should have made more pathfinding calls
        self.assertGreater(call_count[0], first_call_count)

    def test_disabled_returns_zero(self):
        """Test that disabled directional momentum returns zero."""
        with patch('nclone.gym_environment.reward_calculation.main_reward_calculator.DIRECTIONAL_MOMENTUM_ENABLED', False):
            obs = self.base_obs.copy()
            obs["player_xspeed"] = 2.0
            obs["player_yspeed"] = 0.0
            
            reward = self.reward_calculator.calculate_directional_momentum_bonus(obs)
            self.assertEqual(reward, 0.0)

    def test_missing_graph_data_returns_zero(self):
        """Test that missing graph data returns zero (fallback)."""
        obs = self.base_obs.copy()
        obs.pop("_adjacency_graph")  # Remove graph data
        obs["player_xspeed"] = 2.0
        obs["player_yspeed"] = 0.0
        
        reward = self.reward_calculator.calculate_directional_momentum_bonus(obs)
        self.assertEqual(reward, 0.0)


if __name__ == "__main__":
    unittest.main()

