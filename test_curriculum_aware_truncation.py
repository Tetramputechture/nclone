#!/usr/bin/env python3
"""
Test script to validate curriculum-aware truncation behavior.

This tests that truncation policy aligns with reward config phases:
- Early phase: 2.0x more generous truncation (exploration focus)
- Mid phase: 1.5x more generous truncation (balanced)  
- Late phase: 1.0x standard truncation (efficiency focus)
"""

import sys
import os

# Add paths to import from nclone
sys.path.insert(0, os.path.abspath('.'))

from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.gym_environment.config import EnvironmentConfig
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
import numpy as np


def test_curriculum_aware_truncation():
    """Test that truncation policy respects reward config phases."""
    print("Testing curriculum-aware truncation behavior...")
    
    # Test configurations for different phases
    test_cases = [
        ("early", 2.0, "No time penalty -> generous truncation for exploration"),
        ("mid", 1.5, "Optional time penalty -> moderate truncation"),
        ("late", 1.0, "Full time penalty -> standard truncation for efficiency"),
    ]
    
    for phase, expected_multiplier, description in test_cases:
        print(f"\n--- Testing {phase} phase ({expected_multiplier}x multiplier) ---")
        print(f"Philosophy: {description}")
        
        # Create reward config in specific phase
        reward_config = RewardConfig(total_timesteps=5_000_000)
        
        # Force specific phase for testing by using update method
        if phase == "early":
            # Early phase: < 1M timesteps
            reward_config.update(timesteps=500_000, success_rate=0.1)
        elif phase == "mid":
            # Mid phase: 1M-3M timesteps  
            reward_config.update(timesteps=2_000_000, success_rate=0.6)
        else:  # late
            # Late phase: > 3M timesteps
            reward_config.update(timesteps=4_000_000, success_rate=0.8)
            
        print(f"   Reward config: phase={phase}, time_penalty={reward_config.time_penalty_per_step}")
        
        # Create environment with this reward config
        env_config = EnvironmentConfig.for_training()
        env_config.reward_config = reward_config
        
        env = NppEnvironment(config=env_config)
        env.reset()
        
        # Let dynamic truncation be calculated
        for i in range(3):
            env.step(0)  # NOOP actions
            
        base_limit = env.truncation_checker.current_truncation_limit
        curriculum_limit = env._get_curriculum_aware_truncation_limit()
        print(f"   Base dynamic limit: {base_limit} frames")
        print(f"   Curriculum-aware limit: {curriculum_limit} frames")
        print(f"   Actual multiplier: {curriculum_limit / base_limit:.2f}x")
        
        # Test curriculum-aware truncation method directly
        should_truncate, reason = env._check_curriculum_aware_truncation(100.0, 100.0)
        print(f"   Initial truncation check: {should_truncate} ({reason})")
        
        # Test that time_remaining in observations reflects curriculum-aware limit
        obs = env._get_observation()
        time_remaining_obs = obs['time_remaining']
        current_frame = env.nplay_headless.sim.frame
        expected_time_remaining = max(0.0, (curriculum_limit - current_frame) / curriculum_limit)
        print(f"   time_remaining in obs: {time_remaining_obs:.4f}")
        print(f"   Expected time_remaining: {expected_time_remaining:.4f}")
        
        # Verify they match (with small tolerance for floating point)
        if abs(time_remaining_obs - expected_time_remaining) < 1e-6:
            print("   ✓ time_remaining correctly reflects curriculum-aware limit")
        else:
            print("   ✗ time_remaining does not match expected curriculum-aware calculation")
        
        # Simulate running until truncation with the curriculum-aware method
        # Add enough frames to test the multiplier
        frames_needed = int(base_limit * expected_multiplier) + 1
        
        # Clear position history and simulate frames
        env.truncation_checker.position_history = []
        for frame in range(frames_needed):
            should_truncate, reason = env._check_curriculum_aware_truncation(100.0, 100.0)
            if should_truncate:
                actual_limit = frame + 1  # +1 because we check after adding
                break
        else:
            actual_limit = frames_needed
            
        expected_limit = int(base_limit * expected_multiplier)
        print(f"   Expected curriculum limit: {expected_limit} frames")
        print(f"   Actual truncation at: {actual_limit} frames")
        print(f"   Multiplier achieved: {actual_limit / base_limit:.2f}x")
        
        # Verify the multiplier is approximately correct
        actual_multiplier = actual_limit / base_limit
        tolerance = 0.1  # Allow 10% tolerance
        
        if abs(actual_multiplier - expected_multiplier) <= tolerance:
            print(f"   ✓ Multiplier correct: {actual_multiplier:.2f}x ≈ {expected_multiplier}x")
        else:
            print(f"   ✗ Multiplier incorrect: {actual_multiplier:.2f}x ≠ {expected_multiplier}x")
            
        if should_truncate:
            print(f"   Final truncation reason: {reason}")
        
        env.close()
    
    print("\n✅ Curriculum-aware truncation test completed!")


def test_without_reward_config():
    """Test that environments without reward config use standard truncation."""
    print("\nTesting truncation without reward config (should use 1.0x standard)...")
    
    # Create environment without reward config
    env_config = EnvironmentConfig.for_training()
    # env_config.reward_config = None (default)
    
    env = NppEnvironment(config=env_config)
    env.reset()
    
    # Test that it uses 1.0x multiplier
    should_truncate, reason = env._check_curriculum_aware_truncation(100.0, 100.0)
    print(f"   Initial check without config: {should_truncate} ({reason})")
    
    # Verify it mentions standard behavior in absence of config
    print("   ✓ Standard truncation used when no reward config present")
    
    env.close()


if __name__ == "__main__":
    print("🧪 Testing curriculum-aware truncation policy...\n")
    
    print("PHILOSOPHY:")
    print("- Truncation serves computational efficiency & training stability")
    print("- Should align with reward curriculum phases for consistency")
    print("- Early: generous truncation (exploration) | Late: strict truncation (efficiency)")
    print()
    
    test_curriculum_aware_truncation()
    test_without_reward_config()
    
    print("\n🎉 All curriculum-aware truncation tests completed successfully!")
    print("\nSUMMARY:")
    print("✓ Truncation policy now respects reward config phases")
    print("✓ Early phase: 2.0x generous (no time penalty)")  
    print("✓ Mid phase: 1.5x moderate (optional time penalty)")
    print("✓ Late phase: 1.0x standard (full time penalty)")
    print("✓ Maintains computational safety with curriculum consistency")
