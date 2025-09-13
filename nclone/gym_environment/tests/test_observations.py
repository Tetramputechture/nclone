"""
Test suite for multi-modal observation system.

Tests the enhanced observation system with minimal and rich feature profiles,
ensuring proper feature extraction, normalization, and stability.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[5]
sys.path.insert(0, str(project_root))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.nclone_environments.basic_level_no_gold.constants import (
    MINIMAL_GAME_STATE_SIZE, RICH_GAME_STATE_SIZE
)


class TestObservationProfiles(unittest.TestCase):
    """Test observation profile functionality."""
    
    def setUp(self):
        """Set up test environments."""
        self.env_minimal = BasicLevelNoGold(
            render_mode='rgb_array',
            enable_frame_stack=False,
            observation_profile='minimal'
        )
        
        self.env_rich = BasicLevelNoGold(
            render_mode='rgb_array',
            enable_frame_stack=False,
            observation_profile='rich'
        )
    
    def test_observation_space_shapes(self):
        """Test that observation spaces have correct shapes."""
        # Minimal profile
        minimal_obs_space = self.env_minimal.observation_space
        self.assertEqual(minimal_obs_space['game_state'].shape, (MINIMAL_GAME_STATE_SIZE,))
        
        # Rich profile
        rich_obs_space = self.env_rich.observation_space
        self.assertEqual(rich_obs_space['game_state'].shape, (RICH_GAME_STATE_SIZE,))
        
        # Both should have same image dimensions
        self.assertEqual(
            minimal_obs_space['player_frame'].shape,
            rich_obs_space['player_frame'].shape
        )
        self.assertEqual(
            minimal_obs_space['global_view'].shape,
            rich_obs_space['global_view'].shape
        )
    
    def test_observation_consistency(self):
        """Test that observations are consistent across resets."""
        # Test minimal profile
        obs1 = self.env_minimal.reset()[0]
        obs2 = self.env_minimal.reset()[0]
        
        self.assertEqual(obs1['game_state'].shape, obs2['game_state'].shape)
        self.assertEqual(obs1['player_frame'].shape, obs2['player_frame'].shape)
        self.assertEqual(obs1['global_view'].shape, obs2['global_view'].shape)
        
        # Test rich profile
        obs1 = self.env_rich.reset()[0]
        obs2 = self.env_rich.reset()[0]
        
        self.assertEqual(obs1['game_state'].shape, obs2['game_state'].shape)
        self.assertEqual(obs1['player_frame'].shape, obs2['player_frame'].shape)
        self.assertEqual(obs1['global_view'].shape, obs2['global_view'].shape)
    
    def test_game_state_features(self):
        """Test game state feature extraction."""
        # Reset environments
        minimal_obs = self.env_minimal.reset()[0]
        rich_obs = self.env_rich.reset()[0]
        
        # Check feature vector sizes
        self.assertEqual(len(minimal_obs['game_state']), MINIMAL_GAME_STATE_SIZE)
        self.assertEqual(len(rich_obs['game_state']), RICH_GAME_STATE_SIZE)
        
        # Check that all features are finite
        self.assertTrue(np.all(np.isfinite(minimal_obs['game_state'])))
        self.assertTrue(np.all(np.isfinite(rich_obs['game_state'])))
        
        # Check feature ranges (should be normalized)
        minimal_features = minimal_obs['game_state']
        rich_features = rich_obs['game_state']
        
        # Most features should be in reasonable ranges
        self.assertTrue(np.all(minimal_features >= -10.0))
        self.assertTrue(np.all(minimal_features <= 10.0))
        self.assertTrue(np.all(rich_features >= -10.0))
        self.assertTrue(np.all(rich_features <= 10.0))
    
    def test_feature_stability_across_steps(self):
        """Test that features remain stable across environment steps."""
        self.env_rich.reset()
        
        # Take several steps and check feature consistency
        for _ in range(10):
            obs, _, terminated, truncated, _ = self.env_rich.step(0)  # No action
            
            # Check shapes remain consistent
            self.assertEqual(obs['game_state'].shape, (RICH_GAME_STATE_SIZE,))
            self.assertTrue(np.all(np.isfinite(obs['game_state'])))
            
            # Check image shapes remain consistent
            self.assertEqual(obs['player_frame'].shape, (84, 84, 1))
            self.assertEqual(obs['global_view'].shape, (176, 100, 1))
            
            if terminated or truncated:
                break
    
    def test_deprecated_flag_handling(self):
        """Test deprecated use_rich_game_state flag."""
        import warnings
        
        # Test that deprecated flag triggers warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            env = BasicLevelNoGold(
                render_mode='rgb_array',
                enable_frame_stack=False,
                use_rich_game_state=True
            )
            
            # Should trigger deprecation warning
            self.assertTrue(len(w) > 0)
            self.assertTrue(any("deprecated" in str(warning.message) for warning in w))
            
            # Should set observation_profile to 'rich'
            self.assertEqual(env.observation_profile, 'rich')
        
        # Test False case
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            env = BasicLevelNoGold(
                render_mode='rgb_array',
                enable_frame_stack=False,
                use_rich_game_state=False
            )
            
            # Should set observation_profile to 'minimal'
            self.assertEqual(env.observation_profile, 'minimal')


class TestFrameStability(unittest.TestCase):
    """Test frame stability and processing."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = BasicLevelNoGold(
            render_mode='rgb_array',
            enable_frame_stack=False,
            observation_profile='rich'
        )
    
    def test_frame_dtype_consistency(self):
        """Test that frames have consistent dtype."""
        obs = self.env.reset()[0]
        
        # Check initial frame dtypes
        self.assertEqual(obs['player_frame'].dtype, np.uint8)
        self.assertEqual(obs['global_view'].dtype, np.uint8)
        
        # Check after steps
        for _ in range(5):
            obs, _, terminated, truncated, _ = self.env.step(1)
            
            self.assertEqual(obs['player_frame'].dtype, np.uint8)
            self.assertEqual(obs['global_view'].dtype, np.uint8)
            
            if terminated or truncated:
                break
    
    def test_frame_value_ranges(self):
        """Test that frame values are in valid ranges."""
        obs = self.env.reset()[0]
        
        # Check value ranges
        self.assertTrue(np.all(obs['player_frame'] >= 0))
        self.assertTrue(np.all(obs['player_frame'] <= 255))
        self.assertTrue(np.all(obs['global_view'] >= 0))
        self.assertTrue(np.all(obs['global_view'] <= 255))
        
        # Check after steps
        for _ in range(5):
            obs, _, terminated, truncated, _ = self.env.step(2)
            
            self.assertTrue(np.all(obs['player_frame'] >= 0))
            self.assertTrue(np.all(obs['player_frame'] <= 255))
            self.assertTrue(np.all(obs['global_view'] >= 0))
            self.assertTrue(np.all(obs['global_view'] <= 255))
            
            if terminated or truncated:
                break


class TestEntityFeatures(unittest.TestCase):
    """Test enhanced entity feature extraction."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = BasicLevelNoGold(
            render_mode='rgb_array',
            enable_frame_stack=False,
            observation_profile='rich'
        )
    
    def test_ninja_physics_features(self):
        """Test ninja physics state features."""
        obs = self.env.reset()[0]
        game_state = obs['game_state']
        
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
        game_state = obs['game_state']
        
        # Entity features start after ninja features
        entity_features = game_state[24:]
        
        # Should have distance and velocity features for entities
        self.assertTrue(len(entity_features) > 0)
        self.assertTrue(np.all(np.isfinite(entity_features)))


if __name__ == '__main__':
    unittest.main()