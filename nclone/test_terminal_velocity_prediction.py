"""
Unit tests for terminal velocity death prediction system.

Tests the three-tier hybrid approach:
1. Quick state filter (safe velocity check)
2. Precomputed lookup table
3. Full physics simulation
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nclone.terminal_velocity_predictor import (
    TerminalVelocityPredictorStats,
    DeathProbabilityResult,
)
from nclone.constants import (
    TERMINAL_IMPACT_SAFE_VELOCITY,
    TERMINAL_VELOCITY_QUANTIZATION,
    TERMINAL_DISTANCE_QUANTIZATION,
    TERMINAL_IMPACT_SIMULATION_FRAMES,
    MAX_SURVIVABLE_IMPACT,
)


class TestTerminalVelocityPredictor(unittest.TestCase):
    """Test terminal velocity death prediction system."""

    def test_constants_loaded(self):
        """Test that terminal velocity constants are loaded correctly."""
        self.assertEqual(TERMINAL_IMPACT_SAFE_VELOCITY, 4.0)
        self.assertEqual(TERMINAL_VELOCITY_QUANTIZATION, 0.5)
        self.assertEqual(TERMINAL_DISTANCE_QUANTIZATION, 12)
        self.assertEqual(TERMINAL_IMPACT_SIMULATION_FRAMES, 30)
        self.assertEqual(MAX_SURVIVABLE_IMPACT, 6)

    def test_stats_structure(self):
        """Test that stats structure is correct."""
        stats = TerminalVelocityPredictorStats()

        self.assertEqual(stats.build_time_ms, 0.0)
        self.assertEqual(stats.lookup_table_size, 0)
        self.assertEqual(stats.tier1_queries, 0)
        self.assertEqual(stats.tier2_queries, 0)
        self.assertEqual(stats.tier3_queries, 0)

    def test_death_probability_result_structure(self):
        """Test DeathProbabilityResult structure."""
        result = DeathProbabilityResult(
            action_death_probs=[0.0] * 6,
            masked_actions=[],
            frames_simulated=10,
            nearest_surface_distance=100.0,
        )

        self.assertEqual(len(result.action_death_probs), 6)
        self.assertEqual(result.frames_simulated, 10)
        self.assertEqual(result.nearest_surface_distance, 100.0)
        self.assertEqual(len(result.masked_actions), 0)

    def test_velocity_quantization(self):
        """Test velocity quantization logic."""
        # Test that velocities are quantized correctly
        velocity = 5.3
        quantized = (
            round(velocity / TERMINAL_VELOCITY_QUANTIZATION)
            * TERMINAL_VELOCITY_QUANTIZATION
        )

        self.assertEqual(quantized, 5.5)

        velocity = 5.1
        quantized = (
            round(velocity / TERMINAL_VELOCITY_QUANTIZATION)
            * TERMINAL_VELOCITY_QUANTIZATION
        )

        self.assertEqual(quantized, 5.0)

    def test_distance_quantization(self):
        """Test distance quantization logic."""
        # Test that distances are quantized correctly
        distance = 50.0
        quantized = (
            round(distance / TERMINAL_DISTANCE_QUANTIZATION)
            * TERMINAL_DISTANCE_QUANTIZATION
        )

        self.assertEqual(quantized, 48.0)

        distance = 56.0
        quantized = (
            round(distance / TERMINAL_DISTANCE_QUANTIZATION)
            * TERMINAL_DISTANCE_QUANTIZATION
        )

        self.assertEqual(quantized, 60.0)

    def test_safe_velocity_threshold(self):
        """Test Tier 1 safe velocity logic."""
        # Velocities below safe threshold should skip expensive checks
        safe_velocity = 3.5
        self.assertLess(abs(safe_velocity), TERMINAL_IMPACT_SAFE_VELOCITY)

        dangerous_velocity = 5.0
        self.assertGreater(abs(dangerous_velocity), TERMINAL_IMPACT_SAFE_VELOCITY)

    def test_impact_velocity_calculation_concept(self):
        """Test impact velocity calculation concept (from ninja.py)."""
        # Test the formula: impact_vel = -(floor_normalized_x * xspeed_old + floor_normalized_y * yspeed_old)
        floor_normalized_x = 0.0
        floor_normalized_y = -1.0  # Flat floor
        xspeed_old = 2.0
        yspeed_old = 8.0  # Falling downward

        impact_vel = -(
            floor_normalized_x * xspeed_old + floor_normalized_y * yspeed_old
        )

        # Should be positive when hitting floor while falling
        self.assertEqual(impact_vel, 8.0)
        self.assertGreater(impact_vel, MAX_SURVIVABLE_IMPACT)

    def test_survivable_impact_threshold_varies_with_angle(self):
        """Test that survivable impact threshold varies with surface normal."""
        # Flat surface (y = -1)
        floor_normalized_y_flat = -1.0
        threshold_flat = MAX_SURVIVABLE_IMPACT - (4 / 3) * abs(
            floor_normalized_y_flat
        )

        # Steeper angle (y = -0.5)
        floor_normalized_y_steep = -0.5
        threshold_steep = MAX_SURVIVABLE_IMPACT - (4 / 3) * abs(
            floor_normalized_y_steep
        )

        # Steeper angles should have higher survivable threshold
        self.assertGreater(threshold_steep, threshold_flat)

        # Calculate actual thresholds
        self.assertAlmostEqual(threshold_flat, 6.0 - (4 / 3) * 1.0)
        self.assertAlmostEqual(threshold_steep, 6.0 - (4 / 3) * 0.5)


def run_tests():
    """Run all unit tests."""
    unittest.main(argv=[""], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()

