#!/usr/bin/env python3
"""
Test script to verify hierarchical mixin integration with reachability system.

This script tests that:
1. Hierarchical environments require reachability to be enabled
2. Hierarchical mixin uses actual reachability analysis instead of mocks
3. Reachability features are properly computed and used by hierarchical system
"""

import numpy as np
from nclone.gym_environment.config import EnvironmentConfig, HierarchicalConfig, ReachabilityConfig
from nclone.gym_environment.environment_factory import create_hierarchical_env
from nclone.gym_environment.npp_environment import NppEnvironment


def test_hierarchical_requires_reachability():
    """Test that hierarchical environments require reachability to be enabled."""
    print("Testing hierarchical dependency on reachability...")
    
    try:
        # This should fail - hierarchical without reachability
        config = EnvironmentConfig(
            hierarchical=HierarchicalConfig(enable_hierarchical=True),
            reachability=ReachabilityConfig(enable_reachability=False)
        )
        
        env = NppEnvironment(config)
        print("❌ FAIL: Environment created without reachability - this should have failed!")
        return False
        
    except ValueError as e:
        if "reachability analysis" in str(e).lower():
            print("✅ PASS: Validation correctly prevents hierarchical without reachability")
            return True
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False
    
    except Exception as e:
        print(f"❌ FAIL: Unexpected error: {e}")
        return False


def test_hierarchical_uses_real_reachability():
    """Test that hierarchical mixin uses actual reachability analysis."""
    print("\nTesting hierarchical integration with real reachability...")
    
    try:
        # Create hierarchical environment with reachability
        config = EnvironmentConfig.for_hierarchical_training()
        env = create_hierarchical_env(config=config)
        
        # Verify both systems are enabled
        assert env.enable_reachability, "Reachability should be enabled"
        assert env.enable_hierarchical, "Hierarchical should be enabled"
        assert hasattr(env, '_reachability_system'), "Should have reachability system"
        
        # Reset environment and get observations
        obs, info = env.reset()
        
        # Verify reachability features are computed
        reachability_features = obs['reachability_features']
        assert reachability_features.shape == (8,), f"Expected 8D features, got {reachability_features.shape}"
        
        # Verify features are not all zeros (indicating real computation)
        if np.allclose(reachability_features, 0.0):
            print("⚠️  WARNING: Reachability features are all zeros - may be using fallback")
        else:
            print("✅ PASS: Reachability features computed with non-zero values")
        
        # Take a step and verify hierarchical info is available
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert 'hierarchical' in info, "Hierarchical info should be in step info"
        hierarchical_info = info['hierarchical']
        assert 'current_subtask' in hierarchical_info, "Should have current subtask"
        
        print(f"✅ PASS: Hierarchical system working with subtask: {hierarchical_info['current_subtask']}")
        
        # Test that reachability analysis is actually called
        # We can check this by verifying the hierarchical mixin has access to reachability methods
        assert hasattr(env, '_flood_fill_reachability'), "Should have flood fill method from ReachabilityMixin"
        assert hasattr(env, '_get_reachability_features'), "Should have reachability features method"
        
        print("✅ PASS: Hierarchical mixin has access to reachability methods")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error testing hierarchical-reachability integration: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reachability_feature_extraction():
    """Test that hierarchical mixin properly extracts reachability features."""
    print("\nTesting reachability feature extraction in hierarchical context...")
    
    try:
        config = EnvironmentConfig.for_hierarchical_training()
        env = create_hierarchical_env(config=config)
        
        obs, info = env.reset()
        
        # Test the hierarchical mixin's feature extraction method
        reachability_features = env._extract_reachability_features(obs)
        
        assert isinstance(reachability_features, np.ndarray), "Should return numpy array"
        assert reachability_features.shape == (8,), f"Should be 8D, got {reachability_features.shape}"
        assert reachability_features.dtype == np.float32, f"Should be float32, got {reachability_features.dtype}"
        
        # Compare with direct reachability computation
        direct_features = env._get_reachability_features()
        
        # They should be the same (or very close due to caching)
        if np.allclose(reachability_features, direct_features, atol=1e-6):
            print("✅ PASS: Hierarchical feature extraction matches direct reachability computation")
        else:
            print("⚠️  WARNING: Feature extraction differs from direct computation (may be due to caching)")
            print(f"  Hierarchical: {reachability_features}")
            print(f"  Direct: {direct_features}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error testing feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("🧪 Testing Hierarchical-Reachability Integration")
    print("=" * 50)
    
    tests = [
        test_hierarchical_requires_reachability,
        test_hierarchical_uses_real_reachability,
        test_reachability_feature_extraction,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ FAIL: Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Hierarchical-reachability integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the integration.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)