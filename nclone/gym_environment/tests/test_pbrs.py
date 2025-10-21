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
    PBRSPotentials,
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
        """Test objective distance potential function."""
        potential = PBRSPotentials.objective_distance_potential(self.test_state)

        # Should be a finite value
        self.assertTrue(np.isfinite(potential))

        # Should be between 0 and 1 (normalized)
        self.assertGreaterEqual(potential, 0.0)
        self.assertLessEqual(potential, 1.0)

        # Closer to switch (current objective) should give higher potential
        close_state = self.test_state.copy()
        close_state["player_x"] = 82.0  # Close to switch
        close_state["player_y"] = 92.0

        close_potential = PBRSPotentials.objective_distance_potential(close_state)
        self.assertGreater(close_potential, potential)

    def test_hazard_proximity_potential(self):
        """Test hazard proximity potential function."""
        potential = PBRSPotentials.hazard_proximity_potential(self.test_state)

        # Should be a finite value
        self.assertTrue(np.isfinite(potential))

        # Should be between 0 and 1
        self.assertGreaterEqual(potential, 0.0)
        self.assertLessEqual(potential, 1.0)

        # Note: Current implementation returns 1.0 (neutral) as placeholder
        # This test verifies the function exists and returns valid values
        # Full hazard detection will be implemented in future phases
        self.assertEqual(potential, 1.0)  # Current placeholder behavior

    def test_impact_risk_potential(self):
        """Test impact risk potential function."""
        potential = PBRSPotentials.impact_risk_potential(self.test_state)

        # Should be a finite value
        self.assertTrue(np.isfinite(potential))

        # Should be between 0 and 1
        self.assertGreaterEqual(potential, 0.0)
        self.assertLessEqual(potential, 1.0)

        # High velocity toward hazards should give lower potential
        risky_state = self.test_state.copy()
        risky_state["player_vel_x"] = 20.0  # High velocity toward mine
        risky_state["player_vel_y"] = 10.0

        risky_potential = PBRSPotentials.impact_risk_potential(risky_state)
        self.assertLessEqual(risky_potential, potential)

    def test_exploration_potential(self):
        """Test exploration potential function."""
        visited_positions = [(50.0, 50.0), (75.0, 75.0)]

        potential = PBRSPotentials.exploration_potential(
            self.test_state, visited_positions
        )

        # Should be a finite value
        self.assertTrue(np.isfinite(potential))

        # Should be non-negative
        self.assertGreaterEqual(potential, 0.0)

        # Being in a new area should give higher potential than visited area
        visited_state = self.test_state.copy()
        visited_state["player_x"] = 52.0  # Close to visited position
        visited_state["player_y"] = 52.0

        visited_potential = PBRSPotentials.exploration_potential(
            visited_state, visited_positions
        )
        self.assertGreater(potential, visited_potential)


class TestPBRSCalculator(unittest.TestCase):
    """Test PBRS calculator integration."""

    def setUp(self):
        """Set up PBRS calculator."""
        self.calculator = PBRSCalculator(
            objective_weight=1.0,
            hazard_weight=0.5,
            impact_weight=0.3,
            exploration_weight=0.2,
        )

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
        potential = self.calculator.calculate_combined_potential(self.test_state)

        # Should be a finite value
        self.assertTrue(np.isfinite(potential))

        # Should be reasonable range based on weights
        self.assertGreater(potential, 0.0)
        self.assertLess(potential, 10.0)  # Sum of weights * max potential

    def test_potential_components(self):
        """Test individual potential components."""
        components = self.calculator.get_potential_components(self.test_state)

        # Should have all expected components
        expected_components = ["objective", "hazard", "impact", "exploration"]
        for component in expected_components:
            self.assertIn(component, components)
            self.assertTrue(np.isfinite(components[component]))

    def test_position_based_potential_changes(self):
        """Test that potential changes based on player position."""
        # Initial potential
        pot1 = self.calculator.calculate_combined_potential(self.test_state)

        # Move to new position (closer to switch)
        new_state = self.test_state.copy()
        new_state["player_x"] = self.test_state["switch_x"] - 10.0  # Closer to switch
        new_state["player_y"] = self.test_state["switch_y"] - 10.0

        pot2 = self.calculator.calculate_combined_potential(new_state)

        # Potential should be different (and generally higher when closer to objective)
        self.assertNotEqual(pot1, pot2)
        self.assertTrue(np.isfinite(pot1))
        self.assertTrue(np.isfinite(pot2))

        # Components should also be finite
        components1 = self.calculator.get_potential_components(self.test_state)
        components2 = self.calculator.get_potential_components(new_state)

        for key in components1:
            self.assertTrue(
                np.isfinite(components1[key]), f"Component {key} not finite"
            )
            self.assertTrue(
                np.isfinite(components2[key]), f"Component {key} not finite"
            )

    def test_reset_functionality(self):
        """Test calculator reset."""
        # Calculate potentials before reset
        pot_before = self.calculator.calculate_combined_potential(self.test_state)
        self.assertTrue(np.isfinite(pot_before))

        # Reset calculator
        self.calculator.reset()

        # visited_positions list should exist and be empty after reset
        self.assertEqual(len(self.calculator.visited_positions), 0)

        # Potential calculation should still work after reset
        pot_after = self.calculator.calculate_combined_potential(self.test_state)
        self.assertTrue(np.isfinite(pot_after))
        self.assertEqual(pot_before, pot_after)  # Same state should give same potential


class TestPBRSIntegration(unittest.TestCase):
    """Test PBRS integration with environment."""

    def setUp(self):
        """Set up test environments."""
        config_pbrs = EnvironmentConfig(
            render=RenderConfig(render_mode="grayscale_array"),
            pbrs=PBRSConfig(enable_pbrs=True, pbrs_gamma=0.99),
        )
        self.env_pbrs = NppEnvironment(config=config_pbrs)

        config_no_pbrs = EnvironmentConfig(
            render=RenderConfig(render_mode="grayscale_array"),
            pbrs=PBRSConfig(enable_pbrs=False),
        )
        self.env_no_pbrs = NppEnvironment(config=config_no_pbrs)

    def test_pbrs_reward_shaping(self):
        """Test that PBRS affects rewards."""
        # Reset both environments
        self.env_pbrs.reset()
        self.env_no_pbrs.reset()

        # Take same actions in both environments
        pbrs_rewards = []
        normal_rewards = []

        for action in [1, 2, 1, 0, 2]:  # Some movement actions
            _, reward_pbrs, term_pbrs, trunc_pbrs, info_pbrs = self.env_pbrs.step(
                action
            )
            _, reward_normal, term_normal, trunc_normal, info_normal = (
                self.env_no_pbrs.step(action)
            )

            pbrs_rewards.append(reward_pbrs)
            normal_rewards.append(reward_normal)

            # Check that PBRS info is available
            if "pbrs_components" in info_pbrs:
                components = info_pbrs["pbrs_components"]
                self.assertIn("pbrs_reward", components)
                self.assertTrue(np.isfinite(components["pbrs_reward"]))

            if term_pbrs or trunc_pbrs or term_normal or trunc_normal:
                break

        # PBRS and normal rewards might be different due to shaping
        self.assertTrue(len(pbrs_rewards) > 0)
        self.assertTrue(len(normal_rewards) > 0)

        # All rewards should be finite
        self.assertTrue(all(np.isfinite(r) for r in pbrs_rewards))
        self.assertTrue(all(np.isfinite(r) for r in normal_rewards))

    def test_pbrs_configuration_flags(self):
        """Test PBRS configuration in episode info."""
        obs, info = self.env_pbrs.reset()
        _, _, _, _, step_info = self.env_pbrs.step(0)

        # Check configuration flags
        self.assertIn("config_flags", step_info)
        self.assertIn("pbrs_enabled", step_info)
        self.assertTrue(step_info["pbrs_enabled"])

        # Check PBRS components
        if "pbrs_components" in step_info:
            components = step_info["pbrs_components"]
            self.assertIn("pbrs_components", components)
            pbrs_potentials = components["pbrs_components"]

            # Should have all potential components
            expected_potentials = ["objective", "hazard", "impact", "exploration"]
            for potential in expected_potentials:
                self.assertIn(potential, pbrs_potentials)

    def test_pbrs_disabled_environment(self):
        """Test environment with PBRS disabled."""
        obs, info = self.env_no_pbrs.reset()
        _, _, _, _, step_info = self.env_no_pbrs.step(0)

        # Check that PBRS is disabled
        self.assertIn("pbrs_enabled", step_info)
        self.assertFalse(step_info["pbrs_enabled"])

        # PBRS components might still be present but with zero values
        if "pbrs_components" in step_info:
            components = step_info["pbrs_components"]
            if "pbrs_reward" in components:
                self.assertEqual(components["pbrs_reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
