"""Test that observation cache is isolated from external mutations.

This test verifies the fix for the masked action bug root cause:
observation dictionary mutation by wrappers.
"""

import numpy as np
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.gym_environment.config import EnvironmentConfig


class TestObservationCacheIsolation:
    """Test observation cache isolation from external mutations."""

    def test_cached_observation_immune_to_external_mutation(self):
        """Test that cached observation dict is not affected by external modifications.
        
        This test verifies the fix for the root cause of the masked action bug:
        - Environment caches observation dictionary
        - External code (wrappers) modifies the returned observation
        - Next step should validate against the ORIGINAL cached observation, not mutated one
        """
        config = EnvironmentConfig.for_training()
        env = NppEnvironment(config=config)
        
        # Reset environment
        obs, info = env.reset()
        
        # Get initial action mask
        initial_mask = obs["action_mask"].copy()
        
        # Take a step to populate the cache
        valid_actions = np.where(initial_mask)[0]
        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # At this point, env._prev_obs_cache should contain the observation from step 1
        # Let's verify it exists
        assert hasattr(env, "_prev_obs_cache"), "Environment should have _prev_obs_cache"
        assert env._prev_obs_cache is not None, "_prev_obs_cache should not be None after step"
        
        # Store the cached mask for comparison
        cached_mask = env._prev_obs_cache["action_mask"].copy()
        
        # Simulate what a wrapper does: modify the observation dict in-place
        # This is what was causing the bug - modifying the dict that the environment cached
        original_obs_id = id(obs)
        new_mask = np.ones(6, dtype=np.int8)  # All actions valid
        obs["action_mask"] = new_mask  # Modify in-place
        
        # Verify the returned observation was modified
        assert np.array_equal(obs["action_mask"], new_mask), "Returned observation should be modified"
        
        # CRITICAL TEST: The cached observation should NOT be affected by external mutation
        # This is the fix - we deep copy before caching
        assert not np.array_equal(env._prev_obs_cache["action_mask"], new_mask), (
            "Cached observation was mutated by external code! "
            "This is the bug that causes masked action errors. "
            "The environment must deep copy observations before caching."
        )
        
        # Verify cached mask is still the original
        assert np.array_equal(env._prev_obs_cache["action_mask"], cached_mask), (
            "Cached mask should remain unchanged despite external mutations"
        )
        
        env.close()
        print("\n✓ Cached observation is properly isolated from external mutations")
        print("  This confirms the fix for the masked action bug root cause")

    def test_multiple_steps_preserve_cache_independence(self):
        """Test that cache independence is maintained across multiple steps."""
        config = EnvironmentConfig.for_training()
        env = NppEnvironment(config=config)
        
        obs, info = env.reset()
        
        for step in range(10):
            # Take a step
            valid_actions = np.where(obs["action_mask"])[0]
            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, info = env.reset()
                continue
            
            # Store cached mask
            if hasattr(env, "_prev_obs_cache") and env._prev_obs_cache is not None:
                cached_mask = env._prev_obs_cache["action_mask"].copy()
                
                # Mutate returned observation (simulating wrapper behavior)
                obs["action_mask"] = np.ones(6, dtype=np.int8)
                
                # Verify cache wasn't affected
                assert np.array_equal(env._prev_obs_cache["action_mask"], cached_mask), (
                    f"Cache was mutated at step {step}!"
                )
        
        env.close()
        print("\n✓ Cache independence maintained across multiple steps")


if __name__ == "__main__":
    test = TestObservationCacheIsolation()
    
    print("\n" + "=" * 80)
    print("Testing Observation Cache Isolation (Masked Action Bug Root Cause Fix)")
    print("=" * 80)
    
    print("\n[1/2] Testing cached observation immunity to external mutations...")
    test.test_cached_observation_immune_to_external_mutation()
    
    print("\n[2/2] Testing cache independence across multiple steps...")
    test.test_multiple_steps_preserve_cache_independence()
    
    print("\n" + "=" * 80)
    print("✓ All Observation Cache Isolation Tests Passed!")
    print("=" * 80)

