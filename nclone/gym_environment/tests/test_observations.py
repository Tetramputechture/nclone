"""
Test suite for multi-modal observation system.

Tests the enhanced observation system with minimal and rich feature profiles,
ensuring proper feature extraction, normalization, and stability.
"""

import unittest
import numpy as np
import sys
from pathlib import Path
from nclone.gym_environment.npp_environment import (
    NppEnvironment,
)
from nclone.gym_environment.config import EnvironmentConfig, RenderConfig
from nclone.gym_environment.constants import (
    GAME_STATE_CHANNELS,
)

# Add project root to path
project_root = Path(__file__).parents[5]
sys.path.insert(0, str(project_root))


class TestObservationProfiles(unittest.TestCase):
    """Test observation profile functionality."""

    def setUp(self):
        """Set up test environments."""
        config = EnvironmentConfig(render=RenderConfig(render_mode="grayscale_array"))
        self.env_rich = NppEnvironment(config=config)

    def test_observation_space_shapes(self):
        """Test that observation spaces have correct shapes."""
        # Rich profile
        rich_obs_space = self.env_rich.observation_space
        self.assertEqual(rich_obs_space["game_state"].shape, (GAME_STATE_CHANNELS,))

    def test_observation_consistency(self):
        """Test that observations are consistent across resets."""
        # Test rich profile
        obs1 = self.env_rich.reset()[0]
        obs2 = self.env_rich.reset()[0]

        self.assertEqual(obs1["game_state"].shape, obs2["game_state"].shape)
        self.assertEqual(obs1["player_frame"].shape, obs2["player_frame"].shape)
        self.assertEqual(obs1["global_view"].shape, obs2["global_view"].shape)

    def test_game_state_features(self):
        """Test game state feature extraction."""
        # Reset environments
        rich_obs = self.env_rich.reset()[0]

        # Check feature vector size (should have at least ninja_state features)
        # May include additional entity_states beyond the base 30 ninja features
        self.assertGreaterEqual(len(rich_obs["game_state"]), GAME_STATE_CHANNELS)

        # Check that all features are finite
        self.assertTrue(np.all(np.isfinite(rich_obs["game_state"])))

        # Check feature ranges for ninja_state (first 30 features should be normalized)
        ninja_state_features = rich_obs["game_state"][:GAME_STATE_CHANNELS]

        # Ninja state features should be normalized to reasonable ranges
        self.assertTrue(np.all(ninja_state_features >= -10.0))
        self.assertTrue(np.all(ninja_state_features <= 10.0))

    def test_feature_stability_across_steps(self):
        """Test that features remain stable across environment steps."""
        obs_reset = self.env_rich.reset()[0]
        initial_state_size = len(obs_reset["game_state"])
        initial_player_frame_shape = obs_reset["player_frame"].shape
        initial_global_view_shape = obs_reset["global_view"].shape

        # Take several steps and check feature consistency
        for _ in range(10):
            obs, _, terminated, truncated, _ = self.env_rich.step(0)  # No action

            # Check that game_state size remains consistent across steps
            self.assertEqual(len(obs["game_state"]), initial_state_size)

            # Check that at least ninja_state features are present
            self.assertGreaterEqual(len(obs["game_state"]), GAME_STATE_CHANNELS)
            self.assertTrue(np.all(np.isfinite(obs["game_state"])))

            # Check image shapes remain consistent
            self.assertEqual(obs["player_frame"].shape, initial_player_frame_shape)
            self.assertEqual(obs["global_view"].shape, initial_global_view_shape)

            # Check that frame dimensions match expected base dimensions
            self.assertEqual(obs["player_frame"].shape[:2], (84, 84))
            self.assertEqual(obs["global_view"].shape[:2], (176, 100))

            if terminated or truncated:
                break


class TestFrameStability(unittest.TestCase):
    """Test frame stability and processing."""

    def setUp(self):
        """Set up test environment."""
        config = EnvironmentConfig(render=RenderConfig(render_mode="grayscale_array"))
        self.env = NppEnvironment(config=config)

    def test_frame_dtype_consistency(self):
        """Test that frames have consistent dtype."""
        obs = self.env.reset()[0]

        # Check initial frame dtypes
        self.assertEqual(obs["player_frame"].dtype, np.uint8)
        self.assertEqual(obs["global_view"].dtype, np.uint8)

        # Check after steps
        for _ in range(5):
            obs, _, terminated, truncated, _ = self.env.step(1)

            self.assertEqual(obs["player_frame"].dtype, np.uint8)
            self.assertEqual(obs["global_view"].dtype, np.uint8)

            if terminated or truncated:
                break

    def test_frame_value_ranges(self):
        """Test that frame values are in valid ranges."""
        obs = self.env.reset()[0]

        # Check value ranges
        self.assertTrue(np.all(obs["player_frame"] >= 0))
        self.assertTrue(np.all(obs["player_frame"] <= 255))
        self.assertTrue(np.all(obs["global_view"] >= 0))
        self.assertTrue(np.all(obs["global_view"] <= 255))

        # Check after steps
        for _ in range(5):
            obs, _, terminated, truncated, _ = self.env.step(2)

            self.assertTrue(np.all(obs["player_frame"] >= 0))
            self.assertTrue(np.all(obs["player_frame"] <= 255))
            self.assertTrue(np.all(obs["global_view"] >= 0))
            self.assertTrue(np.all(obs["global_view"] <= 255))

            if terminated or truncated:
                break


class TestEntityFeatures(unittest.TestCase):
    """Test enhanced entity feature extraction."""

    def setUp(self):
        """Set up test environment."""
        config = EnvironmentConfig(render=RenderConfig(render_mode="grayscale_array"))
        self.env = NppEnvironment(config=config)

    def test_ninja_physics_features(self):
        """Test ninja physics state features."""
        obs = self.env.reset()[0]
        game_state = obs["game_state"]

        # Rich profile should include ninja physics features
        # Check that we have position, velocity, and physics state
        ninja_features = game_state[:24]  # First 24 features are ninja physics

        # Position features should be reasonable
        pos_x, pos_y = ninja_features[0], ninja_features[1]
        self.assertTrue(np.isfinite(pos_x))
        self.assertTrue(np.isfinite(pos_y))

        # Velocity features
        vel_x, vel_y = ninja_features[2], ninja_features[3]
        self.assertTrue(np.isfinite(vel_x))
        self.assertTrue(np.isfinite(vel_y))

        # Physics state features (normalized)
        physics_features = ninja_features[4:24]
        self.assertTrue(np.all(np.isfinite(physics_features)))

    def test_entity_distance_features(self):
        """Test entity distance and velocity features."""
        obs = self.env.reset()[0]
        game_state = obs["game_state"]

        # Entity features start after ninja features
        entity_features = game_state[24:]

        # Should have distance and velocity features for entities
        self.assertTrue(len(entity_features) > 0)
        self.assertTrue(np.all(np.isfinite(entity_features)))


if __name__ == "__main__":
    unittest.main()
