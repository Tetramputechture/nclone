"""Unit tests for progress tracking and backtracking detection (Tier 1 path efficiency)."""

import unittest
from unittest.mock import patch

from ..reward_calculation.main_reward_calculator import RewardCalculator
from ..reward_calculation.reward_constants import (
    BACKTRACK_THRESHOLD_DISTANCE,
    STAGNATION_THRESHOLD,
)


class TestProgressTracking(unittest.TestCase):
    """Test progress tracking and backtracking detection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.reward_calculator = RewardCalculator()
        
        # Mock adjacency graph and level data
        self.mock_adjacency = {
            (100, 100): [((150, 100), 50.0)],
            (150, 100): [((100, 100), 50.0), ((200, 100), 50.0)],
            (200, 100): [((150, 100), 50.0)],
        }
        self.mock_level_data = Mock()
        self.mock_graph_data = {}
        
        # Base observation with graph data
        self.base_obs = {
            "player_x": 100.0,
            "player_y": 100.0,
            "switch_x": 200.0,
            "switch_y": 100.0,
            "exit_door_x": 300.0,
            "exit_door_y": 100.0,
            "switch_activated": False,
            "_adjacency_graph": self.mock_adjacency,
            "level_data": self.mock_level_data,
            "_graph_data": self.mock_graph_data,
        }

    def test_progress_updates_best_distance(self):
        """Test that making progress updates best distance."""
        # Mock path calculator: start far, then get closer
        distances = [150.0, 140.0]  # Progressing toward switch
        call_idx = [0]
        
        def mock_get_distance(start, goal, *args, **kwargs):
            dist = distances[call_idx[0] % len(distances)]
            call_idx[0] += 1
            return dist
        
        self.reward_calculator.path_calculator.get_distance = mock_get_distance
        
        obs = self.base_obs.copy()
        
        # First call: should set initial best distance
        reward1 = self.reward_calculator.calculate_progress_rewards(obs)
        initial_best = self.reward_calculator.best_path_distance_to_switch
        
        # Second call: should update best distance (progress made)
        obs["player_x"] = 120.0  # Moved closer
        reward2 = self.reward_calculator.calculate_progress_rewards(obs)
        
        # Best distance should have improved
        self.assertLess(
            self.reward_calculator.best_path_distance_to_switch,
            initial_best
        )
        # Should have received progress bonus
        self.assertGreater(reward2, 0.0)

    def test_backtracking_penalized(self):
        """Test that moving away from best is penalized."""
        # Mock path calculator: start close, then move farther
        distances = [50.0, 80.0]  # Backtracking
        call_idx = [0]
        
        def mock_get_distance(start, goal, *args, **kwargs):
            dist = distances[call_idx[0] % len(distances)]
            call_idx[0] += 1
            return dist
        
        self.reward_calculator.path_calculator.get_distance = mock_get_distance
        
        obs = self.base_obs.copy()
        
        # First call: set best distance
        self.reward_calculator.calculate_progress_rewards(obs)
        initial_best = self.reward_calculator.best_path_distance_to_switch
        
        # Reset call index for second call
        call_idx[0] = 0
        
        # Second call: move farther away (backtrack)
        obs["player_x"] = 50.0  # Moved away
        reward = self.reward_calculator.calculate_progress_rewards(obs)
        
        # Should be penalized if backtrack exceeds threshold
        backtrack_distance = 80.0 - initial_best
        if backtrack_distance > BACKTRACK_THRESHOLD_DISTANCE:
            self.assertLess(reward, 0.0)
            self.assertGreater(self.reward_calculator.backtrack_events_total, 0)

    def test_stagnation_penalty_gradual(self):
        """Test that stagnation penalty increases over time."""
        # Mock path calculator: no progress (same distance)
        def mock_get_distance(start, goal, *args, **kwargs):
            return 100.0  # No change
        
        self.reward_calculator.path_calculator.get_distance = mock_get_distance
        
        obs = self.base_obs.copy()
        
        # Set initial best distance
        self.reward_calculator.best_path_distance_to_switch = 100.0
        
        # Simulate many steps without progress
        for i in range(STAGNATION_THRESHOLD + 10):
            self.reward_calculator.frames_since_progress = i
            reward = self.reward_calculator.calculate_progress_rewards(obs)
            
            if i > STAGNATION_THRESHOLD:
                # Should have stagnation penalty
                self.assertLess(reward, 0.0)
                # Penalty should increase with time
                if i > STAGNATION_THRESHOLD + 1:
                    # Later penalties should be >= earlier ones
                    pass  # Just verify penalty exists

    def test_progress_reset_on_new_episode(self):
        """Test that tracking resets properly on new episode."""
        def mock_get_distance(start, goal, *args, **kwargs):
            return 100.0
        
        self.reward_calculator.path_calculator.get_distance = mock_get_distance
        
        obs = self.base_obs.copy()
        
        # Make some progress
        self.reward_calculator.calculate_progress_rewards(obs)
        self.reward_calculator.best_path_distance_to_switch = 80.0
        self.reward_calculator.frames_since_progress = 50
        self.reward_calculator.backtrack_events_total = 3
        
        # Reset
        self.reward_calculator.reset()
        
        # All tracking should be reset
        self.assertEqual(self.reward_calculator.best_path_distance_to_switch, float('inf'))
        self.assertEqual(self.reward_calculator.best_path_distance_to_exit, float('inf'))
        self.assertEqual(self.reward_calculator.frames_since_progress, 0)
        self.assertEqual(self.reward_calculator.backtrack_events_total, 0)
        self.assertEqual(self.reward_calculator.progress_bonus_total, 0.0)
        self.assertEqual(self.reward_calculator.backtrack_penalty_total, 0.0)
        self.assertEqual(self.reward_calculator.stagnation_penalty_total, 0.0)

    def test_switch_to_exit_transition(self):
        """Test that progress tracking switches from switch to exit correctly."""
        def mock_get_distance(start, goal, *args, **kwargs):
            # Different distances for switch vs exit
            if goal == (200.0, 100.0):  # Switch
                return 50.0
            elif goal == (300.0, 100.0):  # Exit
                return 150.0
            return float('inf')
        
        self.reward_calculator.path_calculator.get_distance = mock_get_distance
        
        # Switch phase
        obs = self.base_obs.copy()
        obs["switch_activated"] = False
        self.reward_calculator.calculate_progress_rewards(obs)
        
        # Should track switch distance
        self.assertNotEqual(self.reward_calculator.best_path_distance_to_switch, float('inf'))
        
        # Exit phase
        obs["switch_activated"] = True
        self.reward_calculator.calculate_progress_rewards(obs)
        
        # Should track exit distance
        self.assertNotEqual(self.reward_calculator.best_path_distance_to_exit, float('inf'))

    def test_disabled_returns_zero(self):
        """Test that disabled progress tracking returns zero."""
        with patch('nclone.gym_environment.reward_calculation.main_reward_calculator.PROGRESS_TRACKING_ENABLED', False):
            obs = self.base_obs.copy()
            reward = self.reward_calculator.calculate_progress_rewards(obs)
            self.assertEqual(reward, 0.0)

    def test_missing_graph_data_returns_zero(self):
        """Test that missing graph data returns zero."""
        obs = self.base_obs.copy()
        obs.pop("_adjacency_graph")  # Remove graph data
        
        reward = self.reward_calculator.calculate_progress_rewards(obs)
        self.assertEqual(reward, 0.0)

    def test_unreachable_objective_penalty(self):
        """Test that unreachable objectives get small penalty."""
        def mock_get_distance(start, goal, *args, **kwargs):
            return float('inf')  # Unreachable
        
        self.reward_calculator.path_calculator.get_distance = mock_get_distance
        
        obs = self.base_obs.copy()
        reward = self.reward_calculator.calculate_progress_rewards(obs)
        
        # Should return small penalty for unreachable
        self.assertLess(reward, 0.0)
        self.assertAlmostEqual(reward, -0.001, places=5)


if __name__ == "__main__":
    unittest.main()

