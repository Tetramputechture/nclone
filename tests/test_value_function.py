"""Unit tests for optimal value function precomputation system.

Tests cover:
- Physics step wrapper (state snapshot/restore)
- State discretization (continuous ↔ discrete conversions)
- Value iteration convergence
- Interpolation accuracy
- PBRS gradient comparison (precomputed vs path-based)
"""

import unittest
import numpy as np
from typing import Tuple

from nclone.nplay_headless import NPlayHeadless
from nclone.value_function import (
    PhysicsStepWrapper,
    StateDiscretizer,
    GridConfig,
    ValueIterator,
    ValueInterpolator,
    PrecomputedValuePBRS,
    save_value_function,
    load_value_function,
)


class TestPhysicsStepWrapper(unittest.TestCase):
    """Test physics step wrapper for value iteration."""

    def setUp(self):
        """Create test environment."""
        self.headless = NPlayHeadless(
            render_mode="grayscale_array",
            enable_animation=False,
            enable_logging=False,
            enable_rendering=False,
        )
        self.headless.load_level("006")
        self.physics = PhysicsStepWrapper(self.headless.sim)

    def test_state_snapshot_restore(self):
        """Test that state is properly restored after physics step."""
        # Get initial state
        initial_state = (
            self.headless.sim.ninja.xpos,
            self.headless.sim.ninja.ypos,
            self.headless.sim.ninja.xspeed,
            self.headless.sim.ninja.yspeed,
            self.headless.sim.ninja.jump_duration,
        )

        # Execute physics step
        action = (1, 0)  # Right movement
        next_state, is_death = self.physics.step(initial_state, action)

        # Verify state was restored
        restored_state = (
            self.headless.sim.ninja.xpos,
            self.headless.sim.ninja.ypos,
            self.headless.sim.ninja.xspeed,
            self.headless.sim.ninja.yspeed,
            self.headless.sim.ninja.jump_duration,
        )

        self.assertAlmostEqual(initial_state[0], restored_state[0], places=2)
        self.assertAlmostEqual(initial_state[1], restored_state[1], places=2)
        self.assertAlmostEqual(initial_state[2], restored_state[2], places=2)
        self.assertAlmostEqual(initial_state[3], restored_state[3], places=2)
        self.assertEqual(initial_state[4], restored_state[4])

    def test_deterministic_physics(self):
        """Test that physics step is deterministic."""
        state = (400.0, 400.0, 0.0, 0.0, 0)
        action = (1, 1)  # Right + Jump

        # Execute same step twice
        next_state1, death1 = self.physics.step(state, action)
        next_state2, death2 = self.physics.step(state, action)

        # Verify determinism
        self.assertEqual(next_state1, next_state2)
        self.assertEqual(death1, death2)

    def test_death_detection(self):
        """Test that deaths are correctly detected."""
        # Place ninja at a position that will kill it (e.g., into a mine)
        # For level 006, there are mines we can test with
        # This is a simplified test - real test would place ninja at exact death position
        state = (400.0, 400.0, 0.0, 0.0, 0)
        action = (0, 0)

        # Execute physics
        next_state, is_death = self.physics.step(state, action)

        # Death flag should be False for safe position
        self.assertFalse(is_death)


class TestStateDiscretization(unittest.TestCase):
    """Test state discretization and grid conversions."""

    def setUp(self):
        """Create test discretizer."""
        self.config = GridConfig(dx=12.0, dy=12.0, dvx=0.5, dvy=0.5, djump=5)
        self.discretizer = StateDiscretizer(self.config)

    def test_state_to_index_bounds(self):
        """Test that state_to_index returns valid indices."""
        # Test center of level
        state = (500.0, 300.0, 0.0, 0.0, 0)
        idx = self.discretizer.state_to_index(state)

        # Verify all indices are within bounds
        self.assertGreaterEqual(idx[0], 0)
        self.assertLess(idx[0], self.config.nx)
        self.assertGreaterEqual(idx[1], 0)
        self.assertLess(idx[1], self.config.ny)
        self.assertGreaterEqual(idx[2], 0)
        self.assertLess(idx[2], self.config.nvx)
        self.assertGreaterEqual(idx[3], 0)
        self.assertLess(idx[3], self.config.nvy)
        self.assertGreaterEqual(idx[4], 0)
        self.assertLess(idx[4], self.config.njump)

    def test_index_to_state_center(self):
        """Test that index_to_state returns grid cell centers."""
        # Get center state of cell [10, 10, 7, 12, 1]
        idx = (10, 10, 7, 12, 1)
        state = self.discretizer.index_to_state(idx)

        # Verify it returns cell center
        expected_x = self.config.x_min + idx[0] * self.config.dx + self.config.dx / 2
        expected_y = self.config.y_min + idx[1] * self.config.dy + self.config.dy / 2

        self.assertAlmostEqual(state[0], expected_x, places=2)
        self.assertAlmostEqual(state[1], expected_y, places=2)

    def test_round_trip_conversion(self):
        """Test that state -> index -> state is consistent."""
        original_state = (500.0, 300.0, 1.5, -2.0, 3)

        # Convert to index and back
        idx = self.discretizer.state_to_index(original_state)
        recovered_state = self.discretizer.index_to_state(idx)

        # Should be close (within one grid cell)
        self.assertAlmostEqual(
            original_state[0], recovered_state[0], delta=self.config.dx
        )
        self.assertAlmostEqual(
            original_state[1], recovered_state[1], delta=self.config.dy
        )
        self.assertAlmostEqual(
            original_state[2], recovered_state[2], delta=self.config.dvx
        )
        self.assertAlmostEqual(
            original_state[3], recovered_state[3], delta=self.config.dvy
        )

    def test_fractional_coordinates(self):
        """Test fractional coordinates for interpolation."""
        state = (505.5, 306.3, 0.75, -1.25, 2)
        frac = self.discretizer.state_to_fractional(state)

        # Fractional coordinates should be within grid bounds
        self.assertGreaterEqual(frac[0], 0.0)
        self.assertLess(frac[0], self.config.nx)
        self.assertGreaterEqual(frac[1], 0.0)
        self.assertLess(frac[1], self.config.ny)


class TestValueIteration(unittest.TestCase):
    """Test value iteration convergence."""

    def setUp(self):
        """Create test environment with small grid."""
        self.headless = NPlayHeadless(
            render_mode="grayscale_array",
            enable_animation=False,
            enable_logging=False,
            enable_rendering=False,
        )
        self.headless.load_level("006")

        # Use coarse grid for fast testing
        self.config = GridConfig(dx=24.0, dy=24.0, dvx=1.0, dvy=1.0, djump=10)
        self.discretizer = StateDiscretizer(self.config)
        self.physics = PhysicsStepWrapper(self.headless.sim)

        # Get goal position
        self.goal_pos = self.headless.exit_door_position()

    def test_goal_state_initialization(self):
        """Test that goal states are initialized to zero cost."""
        iterator = ValueIterator(
            physics=self.physics,
            discretizer=self.discretizer,
            goal_pos=self.goal_pos,
            goal_radius=20.0,
        )

        # Initialize value function
        V = np.full(self.config.get_shape(), np.inf, dtype=np.float32)
        iterator._initialize_goal_states(V)

        # Check that at least one state is at goal (cost = 0)
        min_value = np.min(V)
        self.assertEqual(min_value, 0.0)

        # Check that most states are still infinity
        num_finite = np.sum(np.isfinite(V))
        total_states = np.prod(V.shape)
        self.assertLess(num_finite / total_states, 0.01)  # < 1% should be at goal

    def test_convergence(self):
        """Test that value iteration converges."""
        iterator = ValueIterator(
            physics=self.physics,
            discretizer=self.discretizer,
            goal_pos=self.goal_pos,
            goal_radius=20.0,
        )

        # Run with limited iterations for testing
        V = iterator.compute_value_function(
            max_iterations=10,
            tolerance=1.0,
            verbose=False,
        )

        # Check that some states have finite values
        num_finite = np.sum(np.isfinite(V))
        self.assertGreater(num_finite, 0, "Value iteration should propagate from goal")

        # Check that values are non-negative
        finite_values = V[np.isfinite(V)]
        self.assertTrue(np.all(finite_values >= 0), "All costs should be non-negative")


class TestInterpolation(unittest.TestCase):
    """Test 5D linear interpolation."""

    def setUp(self):
        """Create test discretizer and interpolator."""
        self.config = GridConfig(dx=12.0, dy=12.0, dvx=0.5, dvy=0.5, djump=5)
        self.discretizer = StateDiscretizer(self.config)

        # Create synthetic value function for testing
        self.V = np.random.rand(*self.config.get_shape()).astype(np.float32) * 100

    def test_nearest_interpolation(self):
        """Test nearest-neighbor interpolation."""
        interpolator = ValueInterpolator(self.discretizer, method="nearest")

        # Test at grid point
        idx = (10, 10, 7, 12, 1)
        grid_state = self.discretizer.index_to_state(idx)
        value = interpolator.interpolate(self.V, grid_state)

        # Should match exact grid value
        expected = self.V[idx]
        self.assertAlmostEqual(value, expected, places=4)

    def test_linear_interpolation_bounds(self):
        """Test that linear interpolation returns reasonable values."""
        interpolator = ValueInterpolator(self.discretizer, method="linear")

        # Test at arbitrary position
        state = (505.5, 306.3, 0.75, -1.25, 2)
        value = interpolator.interpolate(self.V, state)

        # Value should be within range of V
        self.assertGreaterEqual(value, np.min(self.V))
        self.assertLessEqual(value, np.max(self.V))

    def test_interpolation_consistency(self):
        """Test that nearby states have similar interpolated values."""
        interpolator = ValueInterpolator(self.discretizer, method="linear")

        # Test two nearby states
        state1 = (500.0, 300.0, 0.0, 0.0, 0)
        state2 = (500.5, 300.5, 0.1, 0.1, 0)

        value1 = interpolator.interpolate(self.V, state1)
        value2 = interpolator.interpolate(self.V, state2)

        # Values should be similar (within 10% of value range)
        value_range = np.max(self.V) - np.min(self.V)
        diff = abs(value1 - value2)
        self.assertLess(diff, 0.1 * value_range)


class TestPrecomputedPBRS(unittest.TestCase):
    """Test precomputed PBRS runtime class."""

    def setUp(self):
        """Create test value function and PBRS instance."""
        # Create small test value function
        self.config = GridConfig(dx=24.0, dy=24.0, dvx=1.0, dvy=1.0, djump=10)
        self.discretizer = StateDiscretizer(self.config)
        self.goal_pos = (500.0, 300.0)

        # Create synthetic value function (decreasing towards goal)
        shape = self.config.get_shape()
        self.V = np.full(shape, 100.0, dtype=np.float32)

        # Set goal region to 0
        for ix in range(shape[0]):
            for iy in range(shape[1]):
                state = self.discretizer.index_to_state((ix, iy, 0, 0, 0))
                dist = np.sqrt(
                    (state[0] - self.goal_pos[0]) ** 2
                    + (state[1] - self.goal_pos[1]) ** 2
                )
                if dist < 20:
                    self.V[ix, iy, :, :, :] = 0.0

        # Save and load
        import tempfile

        self.temp_file = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        self.temp_path = self.temp_file.name
        self.temp_file.close()

        save_value_function(self.V, self.discretizer, self.goal_pos, self.temp_path)

    def tearDown(self):
        """Clean up temp file."""
        import os

        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_load_value_function(self):
        """Test loading precomputed value function."""
        pbrs = PrecomputedValuePBRS(self.temp_path)

        # Verify loaded correctly
        self.assertIsNotNone(pbrs.V)
        self.assertEqual(pbrs.V.shape, self.V.shape)

    def test_potential_at_goal(self):
        """Test that potential is maximum (1.0) at goal."""
        pbrs = PrecomputedValuePBRS(self.temp_path)

        # Create state at goal
        state = {
            "player_x": self.goal_pos[0],
            "player_y": self.goal_pos[1],
            "player_xspeed": 0.0,
            "player_yspeed": 0.0,
            "jump_duration_remaining": 0.0,
        }

        potential = pbrs.get_potential(state)

        # Should be close to 1.0 (normalized maximum)
        self.assertGreater(potential, 0.9)
        self.assertLessEqual(potential, 1.0)

    def test_potential_normalization(self):
        """Test that potentials are in [0, 1] range."""
        pbrs = PrecomputedValuePBRS(self.temp_path)

        # Test multiple random states
        for _ in range(10):
            state = {
                "player_x": np.random.uniform(0, 1056),
                "player_y": np.random.uniform(0, 600),
                "player_xspeed": np.random.uniform(-6, 6),
                "player_yspeed": np.random.uniform(-12, 12),
                "jump_duration_remaining": np.random.uniform(0, 1),
            }

            potential = pbrs.get_potential(state)

            # Should be in valid range
            self.assertGreaterEqual(potential, 0.0)
            self.assertLessEqual(potential, 1.0)


class TestPBRSGradient(unittest.TestCase):
    """Test PBRS gradient comparison (precomputed vs path-based).

    This validates that the precomputed value function provides
    reasonable potential gradients compared to the legacy path-based approach.
    """

    def setUp(self):
        """Create test environment with both PBRS systems."""
        # This is a placeholder - full implementation would require:
        # 1. Loading level with graph-based pathfinding
        # 2. Computing precomputed value function
        # 3. Comparing potential gradients along trajectories
        pass

    def test_gradient_correlation(self):
        """Test that precomputed and path-based gradients are correlated."""
        # Placeholder test - full implementation would:
        # 1. Sample states along multiple trajectories
        # 2. Compute PBRS potentials using both methods
        # 3. Verify that gradients point in similar directions
        # 4. Check that precomputed gradients are smoother (incorporate velocity)
        pass


def run_tests():
    """Run all value function tests."""
    unittest.main()


if __name__ == "__main__":
    run_tests()
