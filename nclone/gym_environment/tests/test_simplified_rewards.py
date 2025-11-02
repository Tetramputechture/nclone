"""Unit tests for the simplified completion-focused reward system."""

import unittest

from ..reward_calculation.main_reward_calculator import RewardCalculator
from ..reward_calculation.pbrs_potentials import PBRSCalculator, PBRSPotentials

# NavigationRewardCalculator removed - milestone rewards redundant with PBRS
from ..reward_calculation.exploration_reward_calculator import (
    ExplorationRewardCalculator,
)


class TestSimplifiedRewardSystem(unittest.TestCase):
    """Test the completion-focused reward system."""

    def setUp(self):
        """Set up test fixtures."""
        self.reward_calculator = RewardCalculator(enable_pbrs=True)

        # Sample game states for testing
        self.base_obs = {
            "player_x": 100.0,
            "player_y": 100.0,
            "switch_x": 200.0,
            "switch_y": 200.0,
            "exit_door_x": 300.0,
            "exit_door_y": 300.0,
            "switch_activated": False,
            "player_dead": False,
            "player_won": False,
            "doors_opened": 0,
            "game_state": [0.0] * 30,  # Mock game state array
        }

        self.prev_obs = self.base_obs.copy()

    def test_reward_constants(self):
        """Test that reward constants match the specification."""
        self.assertEqual(RewardCalculator.SWITCH_ACTIVATION_REWARD, 0.1)
        self.assertEqual(RewardCalculator.EXIT_COMPLETION_REWARD, 1.0)
        self.assertEqual(RewardCalculator.DEATH_PENALTY, -0.5)
        self.assertEqual(RewardCalculator.TIME_PENALTY, -0.01)

    def test_time_penalty_applied(self):
        """Test that time penalty is applied on each step."""
        obs = self.base_obs.copy()
        prev_obs = self.prev_obs.copy()

        reward = self.reward_calculator.calculate_reward(obs, prev_obs)

        # Should include time penalty, but may have other small rewards too
        # Check that the reward is negative (indicating time penalty is working)
        self.assertLess(reward, 0.0)  # Should be negative due to time penalty

    def test_death_penalty(self):
        """Test death penalty is applied correctly."""
        obs = self.base_obs.copy()
        obs["player_dead"] = True
        prev_obs = self.prev_obs.copy()

        reward = self.reward_calculator.calculate_reward(obs, prev_obs)

        self.assertEqual(reward, RewardCalculator.DEATH_PENALTY)

    def test_switch_activation_reward(self):
        """Test switch activation reward is applied correctly."""
        obs = self.base_obs.copy()
        obs["switch_activated"] = True
        prev_obs = self.prev_obs.copy()
        prev_obs["switch_activated"] = False

        reward = self.reward_calculator.calculate_reward(obs, prev_obs)

        # Should include switch activation reward plus time penalty
        expected_min = (
            RewardCalculator.SWITCH_ACTIVATION_REWARD + RewardCalculator.TIME_PENALTY
        )
        self.assertGreaterEqual(reward, expected_min)

    def test_exit_completion_reward(self):
        """Test exit completion reward is applied correctly."""
        obs = self.base_obs.copy()
        obs["player_won"] = True
        prev_obs = self.prev_obs.copy()

        reward = self.reward_calculator.calculate_reward(obs, prev_obs)

        # Should include exit completion reward plus time penalty
        expected_min = (
            RewardCalculator.EXIT_COMPLETION_REWARD + RewardCalculator.TIME_PENALTY
        )
        self.assertGreaterEqual(reward, expected_min)

    def test_no_gold_related_rewards(self):
        """Test that no gold-related rewards are present."""
        obs = self.base_obs.copy()
        # Add some gold-related fields that should be ignored
        obs["gold_collected"] = 5
        obs["gold_x"] = [50.0, 150.0]
        obs["gold_y"] = [50.0, 150.0]
        prev_obs = self.prev_obs.copy()
        prev_obs["gold_collected"] = 3

        reward = self.reward_calculator.calculate_reward(obs, prev_obs)

        # Reward should only be time penalty (and any navigation/exploration)
        # but definitely not include gold collection bonuses
        self.assertLessEqual(reward, 0.1)  # No large positive rewards from gold


class TestPBRSPotentials(unittest.TestCase):
    """Test the simplified PBRS potential functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.pbrs_calculator = PBRSCalculator()

        self.state_switch_inactive = {
            "player_x": 100.0,
            "player_y": 100.0,
            "switch_x": 200.0,
            "switch_y": 200.0,
            "exit_door_x": 300.0,
            "exit_door_y": 300.0,
            "switch_activated": False,
        }

        self.state_switch_active = {
            "player_x": 100.0,
            "player_y": 100.0,
            "switch_x": 200.0,
            "switch_y": 200.0,
            "exit_door_x": 300.0,
            "exit_door_y": 300.0,
            "switch_activated": True,
        }

    def test_pbrs_constants(self):
        """Test PBRS constants match specification."""
        self.assertEqual(PBRSCalculator.PBRS_SWITCH_DISTANCE, 0.05)
        self.assertEqual(PBRSCalculator.PBRS_EXIT_DISTANCE, 0.05)

    def test_switch_focus_when_inactive(self):
        """Test PBRS focuses on switch when inactive."""
        potential = self.pbrs_calculator.calculate_combined_potential(
            self.state_switch_inactive
        )

        # Should be positive (closer to switch = higher potential)
        self.assertGreater(potential, 0.0)

        # Should use switch distance scaling
        objective_pot = PBRSPotentials.objective_distance_potential(
            self.state_switch_inactive
        )
        expected = PBRSCalculator.PBRS_SWITCH_DISTANCE * objective_pot
        self.assertAlmostEqual(potential, expected, places=6)

    def test_exit_focus_when_active(self):
        """Test PBRS focuses on exit when switch is active."""
        potential = self.pbrs_calculator.calculate_combined_potential(
            self.state_switch_active
        )

        # Should be positive (closer to exit = higher potential)
        self.assertGreater(potential, 0.0)

        # Should use exit distance scaling
        objective_pot = PBRSPotentials.objective_distance_potential(
            self.state_switch_active
        )
        expected = PBRSCalculator.PBRS_EXIT_DISTANCE * objective_pot
        self.assertAlmostEqual(potential, expected, places=6)

    def test_disabled_potentials(self):
        """Test that hazard, impact, and exploration potentials are disabled."""
        components = self.pbrs_calculator.get_potential_components(
            self.state_switch_inactive
        )

        self.assertEqual(components["hazard"], 0.0)
        self.assertEqual(components["impact"], 0.0)
        self.assertEqual(components["exploration"], 0.0)

    def test_objective_distance_potential(self):
        """Test objective distance potential calculation."""
        # Test switch phase
        potential_switch = PBRSPotentials.objective_distance_potential(
            self.state_switch_inactive
        )
        self.assertGreater(potential_switch, 0.0)
        self.assertLessEqual(potential_switch, 1.0)

        # Test exit phase
        potential_exit = PBRSPotentials.objective_distance_potential(
            self.state_switch_active
        )
        self.assertGreater(potential_exit, 0.0)
        self.assertLessEqual(potential_exit, 1.0)


# NOTE: TestNavigationRewardCalculator removed - NavigationRewardCalculator was removed
# because milestone-based distance rewards are redundant with PBRS continuous rewards.
# PBRS provides better guidance with path-aware distances and policy invariance.


class TestExplorationRewardCalculator(unittest.TestCase):
    """Test the exploration reward calculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.exploration_calculator = ExplorationRewardCalculator()

    def test_exploration_reward_positive(self):
        """Test that exploring new areas gives positive reward."""
        # First position
        reward1 = self.exploration_calculator.calculate_exploration_reward(100.0, 100.0)
        self.assertGreater(reward1, 0.0)

        # Different position should also give reward
        reward2 = self.exploration_calculator.calculate_exploration_reward(200.0, 200.0)
        self.assertGreater(reward2, 0.0)

    def test_exploration_reward_scales(self):
        """Test that exploration rewards are appropriately scaled."""
        reward = self.exploration_calculator.calculate_exploration_reward(100.0, 100.0)

        # Should be small positive values
        self.assertGreater(reward, 0.0)
        self.assertLess(reward, 0.01)  # Should be smaller than main rewards

    def test_reset_functionality(self):
        """Test that reset clears exploration history."""
        # Visit some positions
        self.exploration_calculator.calculate_exploration_reward(100.0, 100.0)
        self.exploration_calculator.calculate_exploration_reward(200.0, 200.0)

        # Reset
        self.exploration_calculator.reset()

        # Should get reward again for same position
        reward = self.exploration_calculator.calculate_exploration_reward(100.0, 100.0)
        self.assertGreater(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
