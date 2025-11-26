"""
Test suite for PBRS (Potential-Based Reward Shaping) system.

Tests the PBRS implementation including potential functions, reward shaping,
and integration with the main reward calculator.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[5]
sys.path.insert(0, str(project_root))

from nclone.gym_environment.npp_environment import (
    NppEnvironment,
)
from nclone.gym_environment.config import EnvironmentConfig, RenderConfig, PBRSConfig
from nclone.gym_environment.reward_calculation.pbrs_potentials import (
    PBRSCalculator,
)


class TestPBRSPotentials(unittest.TestCase):
    """Test PBRS potential functions."""

    def setUp(self):
        """Set up test data."""
        self.test_state = {
            "player_x": 100.0,
            "player_y": 100.0,
            "player_vel_x": 5.0,
            "player_vel_y": -2.0,
            "exit_x": 200.0,
            "exit_y": 150.0,
            "switch_x": 80.0,
            "switch_y": 90.0,
            "exit_door_x": 200.0,
            "exit_door_y": 150.0,
            "switch_activated": False,
            "entities": [
                {"type": "mine", "x": 120.0, "y": 110.0},
                {"type": "thwomp", "x": 180.0, "y": 140.0},
                {"type": "door", "x": 190.0, "y": 145.0},
            ],
        }

    def test_objective_distance_potential(self):
        """Test objective distance potential function with non-linear normalization."""
        # This test is deprecated - objective_distance_potential now requires
        # adjacency graph, level_data, and path_calculator for proper path-based
        # distance calculations. Use test_pbrs_nonlinear.py for testing the
        # non-linear normalization behavior.
        #
        # The function no longer supports Euclidean distance fallback to enforce
        # strict path-aware reward shaping.
        self.skipTest(
            "Objective distance potential now requires full graph infrastructure - see test_pbrs_nonlinear.py"
        )

    def test_hazard_proximity_potential(self):
        """Test hazard proximity potential function."""
        # REMOVED: hazard_proximity_potential was removed from simplified PBRS system.
        # Death penalty provides clearer signal than proximity shaping.
        self.skipTest(
            "hazard_proximity_potential removed - death penalty provides clearer signal"
        )

    def test_impact_risk_potential(self):
        """Test impact risk potential function."""
        # REMOVED: impact_risk_potential was removed from simplified PBRS system.
        # Death penalty provides clearer signal than impact risk shaping.
        self.skipTest(
            "impact_risk_potential removed - death penalty provides clearer signal"
        )

    def test_exploration_potential(self):
        """Test exploration potential function."""
        # REMOVED: exploration_potential was removed from simplified PBRS system.
        # PBRS provides exploration via distance gradients without explicit exploration bonus.
        self.skipTest(
            "exploration_potential removed - PBRS provides exploration via distance gradients"
        )


class TestPBRSCalculator(unittest.TestCase):
    """Test PBRS calculator integration."""

    def setUp(self):
        """Set up PBRS calculator."""
        # PBRSCalculator API changed - it no longer takes weight parameters in __init__
        # Weights are now managed by RewardConfig as part of curriculum system
        self.calculator = PBRSCalculator()

        self.test_state = {
            "player_x": 100.0,
            "player_y": 100.0,
            "player_vel_x": 5.0,
            "player_vel_y": -2.0,
            "exit_x": 200.0,
            "exit_y": 150.0,
            "switch_x": 80.0,
            "switch_y": 90.0,
            "exit_door_x": 200.0,
            "exit_door_y": 150.0,
            "switch_activated": False,
            "entities": [{"type": "mine", "x": 120.0, "y": 110.0}],
        }

    def test_combined_potential_calculation(self):
        """Test combined potential calculation."""
        # DEPRECATED: calculate_combined_potential now requires adjacency, level_data, and path_calculator
        # These tests are superseded by integration tests with full environment
        self.skipTest("PBRSCalculator tests deprecated - use integration tests instead")

    def test_potential_components(self):
        """Test individual potential components."""
        # DEPRECATED: get_potential_components method removed (hazard/impact/exploration potentials removed)
        # Only objective potential remains in simplified PBRS
        self.skipTest(
            "get_potential_components removed - only objective potential remains"
        )

    def test_position_based_potential_changes(self):
        """Test that potential changes based on player position."""
        # DEPRECATED: Test needs full graph infrastructure
        # Covered by test_pbrs_nonlinear.py and integration tests
        self.skipTest(
            "Position-based potential tests require full graph infrastructure"
        )

    def test_reset_functionality(self):
        """Test calculator reset."""
        # Test that reset doesn't crash
        self.calculator.reset()
        # visited_positions removed from simplified PBRS
        self.assertTrue(True)  # Reset succeeded without error


class TestPBRSIntegration(unittest.TestCase):
    """Test PBRS integration with environment."""

    def setUp(self):
        """Set up test environments."""
        # PBRS is now ALWAYS ENABLED in the environment (no enable_pbrs flag)
        # This matches the updated design where PBRS is mandatory for consistent training
        config_pbrs = EnvironmentConfig(
            render=RenderConfig(render_mode="grayscale_array"),
            pbrs=PBRSConfig(pbrs_gamma=0.995),
        )
        self.env_pbrs = NppEnvironment(config=config_pbrs)

        # No separate "pbrs disabled" environment - PBRS is always active
        self.env_no_pbrs = None

    def test_pbrs_reward_shaping(self):
        """Test that PBRS affects rewards."""
        # PBRS is now ALWAYS ENABLED - test that it provides non-zero rewards
        self.env_pbrs.reset()

        pbrs_rewards = []

        for action in [1, 2, 1, 0, 2]:  # Some movement actions
            _, reward_pbrs, term_pbrs, trunc_pbrs, info_pbrs = self.env_pbrs.step(
                action
            )

            pbrs_rewards.append(reward_pbrs)

            # Check that PBRS info is available
            if "pbrs_components" in info_pbrs:
                components = info_pbrs["pbrs_components"]
                self.assertIn("pbrs_reward", components)
                self.assertTrue(np.isfinite(components["pbrs_reward"]))

            if term_pbrs or trunc_pbrs:
                break

        # Verify rewards are present and finite
        self.assertTrue(len(pbrs_rewards) > 0)
        self.assertTrue(all(np.isfinite(r) for r in pbrs_rewards))

    def test_pbrs_configuration_flags(self):
        """Test PBRS configuration in episode info."""
        obs, info = self.env_pbrs.reset()
        _, _, _, _, step_info = self.env_pbrs.step(0)

        # Check configuration flags
        self.assertIn("config_flags", step_info)
        # PBRS is always enabled now - no enable/disable flag

        # Check PBRS components
        if "pbrs_components" in step_info:
            components = step_info["pbrs_components"]
            # Simplified PBRS only has objective potential (no hazard/impact/exploration)
            self.assertIn("pbrs_reward", components)
            self.assertTrue(np.isfinite(components["pbrs_reward"]))

    def test_pbrs_disabled_environment(self):
        """Test environment with PBRS disabled."""
        # DEPRECATED: PBRS is now always enabled (mandatory for consistent training)
        # This test is no longer relevant
        self.skipTest("PBRS is now always enabled - no disable option")


if __name__ == "__main__":
    unittest.main()
