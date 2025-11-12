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
    NINJA_STATE_DIM,
    PATH_AWARE_OBJECTIVES_DIM,
    MINE_FEATURES_DIM,
    PROGRESS_FEATURES_DIM,
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
        obs = self.env_rich.reset()[0]

        # Verify game_state has correct shape
        self.assertEqual(obs["game_state"].shape, (GAME_STATE_CHANNELS,))
        self.assertEqual(len(obs["game_state"]), GAME_STATE_CHANNELS)

        # Verify all features are in valid ranges
        game_state = obs["game_state"]

        # Calculate feature offsets
        path_aware_start = NINJA_STATE_DIM
        mine_start = path_aware_start + PATH_AWARE_OBJECTIVES_DIM
        progress_start = mine_start + MINE_FEATURES_DIM

        # Path-aware objectives
        # - exit_switch_collected: [0, 1]
        self.assertGreaterEqual(game_state[path_aware_start], 0.0)
        self.assertLessEqual(game_state[path_aware_start], 1.0)
        # - rel_x, rel_y: [-1, 1]
        self.assertGreaterEqual(game_state[path_aware_start + 1], -1.0)
        self.assertLessEqual(game_state[path_aware_start + 1], 1.0)
        self.assertGreaterEqual(game_state[path_aware_start + 2], -1.0)
        self.assertLessEqual(game_state[path_aware_start + 2], 1.0)
        # - path_distance: [0, 1]
        self.assertGreaterEqual(game_state[path_aware_start + 3], 0.0)
        self.assertLessEqual(game_state[path_aware_start + 3], 1.0)

        # Mine features
        # - rel_x, rel_y: [-1, 1]
        self.assertGreaterEqual(game_state[mine_start], -1.0)
        self.assertLessEqual(game_state[mine_start], 1.0)
        self.assertGreaterEqual(game_state[mine_start + 1], -1.0)
        self.assertLessEqual(game_state[mine_start + 1], 1.0)
        # - mine_state: [-1, 1] (can be -1, 0, 0.5, or 1)
        self.assertGreaterEqual(game_state[mine_start + 2], -1.0)
        self.assertLessEqual(game_state[mine_start + 2], 1.0)
        # - path_distance: [0, 1]
        self.assertGreaterEqual(game_state[mine_start + 3], 0.0)
        self.assertLessEqual(game_state[mine_start + 3], 1.0)
        # - deadly_mines_nearby: [0, 1]
        self.assertGreaterEqual(game_state[mine_start + 4], 0.0)
        self.assertLessEqual(game_state[mine_start + 4], 1.0)

        # Progress features
        # - current_objective_type: [0, 1]
        self.assertGreaterEqual(game_state[progress_start], 0.0)
        self.assertLessEqual(game_state[progress_start], 1.0)
        # - objectives_completed_ratio: [0, 1]
        self.assertGreaterEqual(game_state[progress_start + 1], 0.0)
        self.assertLessEqual(game_state[progress_start + 1], 1.0)
        # - total_path_distance_remaining: [0, 1]
        self.assertGreaterEqual(game_state[progress_start + 2], 0.0)
        self.assertLessEqual(game_state[progress_start + 2], 1.0)

    def test_path_aware_features_normalization(self):
        """Test that path-aware features are properly normalized."""
        obs = self.env_rich.reset()[0]
        game_state = obs["game_state"]

        # Calculate feature offsets
        path_aware_start = NINJA_STATE_DIM
        mine_start = path_aware_start + PATH_AWARE_OBJECTIVES_DIM
        progress_start = mine_start + MINE_FEATURES_DIM

        # Verify all features are finite
        self.assertTrue(np.all(np.isfinite(game_state)))

        # Verify path-aware objective features are normalized
        path_aware_end = path_aware_start + PATH_AWARE_OBJECTIVES_DIM
        path_aware_features = game_state[path_aware_start:path_aware_end]
        # Binary flags should be 0 or 1
        self.assertIn(path_aware_features[0], [0.0, 1.0])  # exit_switch_collected
        # Relative positions should be in [-1, 1]
        self.assertTrue(np.all(path_aware_features[1:3] >= -1.0))
        self.assertTrue(np.all(path_aware_features[1:3] <= 1.0))
        # Path distances should be in [0, 1]
        self.assertTrue(
            np.all(path_aware_features[3::3] >= 0.0)
        )  # Every 3rd feature starting at index 3
        self.assertTrue(np.all(path_aware_features[3::3] <= 1.0))

        # Verify mine features are normalized
        mine_end = mine_start + MINE_FEATURES_DIM
        mine_features = game_state[mine_start:mine_end]
        # Relative positions [-1, 1]
        self.assertTrue(np.all(mine_features[0:2] >= -1.0))
        self.assertTrue(np.all(mine_features[0:2] <= 1.0))
        # Mine state should be -1, 0, 0.5, or 1
        self.assertIn(mine_features[2], [-1.0, 0.0, 0.5, 1.0])
        # Path distance and count [0, 1]
        self.assertTrue(np.all(mine_features[3:5] >= 0.0))
        self.assertTrue(np.all(mine_features[3:5] <= 1.0))

        # Verify progress features are normalized
        progress_end = progress_start + PROGRESS_FEATURES_DIM
        progress_features = game_state[progress_start:progress_end]
        # All should be in [0, 1]
        self.assertTrue(np.all(progress_features >= 0.0))
        self.assertTrue(np.all(progress_features <= 1.0))

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
        # Check that we have velocity, movement state, inputs, buffers, contact info, and physics
        ninja_features = game_state[:GAME_STATE_CHANNELS]  # All ninja state features

        # All features should be finite and normalized
        self.assertTrue(np.all(np.isfinite(ninja_features)))

        # Features should be in reasonable normalized range [-1, 1] (with some tolerance)
        self.assertTrue(np.all(ninja_features >= -10.0))
        self.assertTrue(np.all(ninja_features <= 10.0))

    def test_entity_distance_features(self):
        """Test entity distance and velocity features."""
        obs = self.env.reset()[0]
        game_state = obs["game_state"]

        # Entity features start after ninja features
        entity_features = game_state[GAME_STATE_CHANNELS:]

        # Should have distance and velocity features for entities
        self.assertTrue(len(entity_features) > 0)
        self.assertTrue(np.all(np.isfinite(entity_features)))


if __name__ == "__main__":
    unittest.main()
