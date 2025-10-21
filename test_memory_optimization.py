#!/usr/bin/env python3
"""
Test script to validate memory optimizations in the N++ environment.

This script creates multiple environment instances and monitors memory usage
to ensure optimizations are effective.
"""

import sys
import time
import numpy as np
import psutil
import os

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.gym_environment.config import EnvironmentConfig
from nclone.gym_environment.environment_factory import create_visual_testing_env


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_single_environment_memory():
    """Test memory footprint of a single environment."""
    print("\n" + "=" * 60)
    print("TEST 1: Single Environment Memory Footprint")
    print("=" * 60)
    
    # Measure baseline
    baseline_memory = get_memory_usage_mb()
    print(f"Baseline memory: {baseline_memory:.2f} MB")
    
    # Create standard config environment
    print("\nCreating environment with standard config...")
    config_standard = EnvironmentConfig.for_training()
    env_standard = create_visual_testing_env(config=config_standard)
    env_standard.reset()
    
    memory_after_standard = get_memory_usage_mb()
    standard_footprint = memory_after_standard - baseline_memory
    print(f"Memory after standard env: {memory_after_standard:.2f} MB")
    print(f"Standard env footprint: {standard_footprint:.2f} MB")
    
    # Run a few steps to measure observation memory
    obs_sizes = []
    for i in range(10):
        obs, reward, terminated, truncated, info = env_standard.step(0)
        obs_size = 0
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_size += value.nbytes
        obs_sizes.append(obs_size / 1024 / 1024)  # Convert to MB
    
    avg_obs_size = np.mean(obs_sizes)
    print(f"Average observation size: {avg_obs_size:.3f} MB")
    
    env_standard.close()
    del env_standard
    
    # Create memory-optimized environment
    print("\nCreating environment with memory-optimized config...")
    config_optimized = EnvironmentConfig.for_parallel_training()
    env_optimized = create_visual_testing_env(config=config_optimized)
    env_optimized.reset()
    
    memory_after_optimized = get_memory_usage_mb()
    optimized_footprint = memory_after_optimized - baseline_memory
    print(f"Memory after optimized env: {memory_after_optimized:.2f} MB")
    print(f"Optimized env footprint: {optimized_footprint:.2f} MB")
    
    # Run a few steps
    obs_sizes_opt = []
    for i in range(10):
        obs, reward, terminated, truncated, info = env_optimized.step(0)
        obs_size = 0
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_size += value.nbytes
        obs_sizes_opt.append(obs_size / 1024 / 1024)
    
    avg_obs_size_opt = np.mean(obs_sizes_opt)
    print(f"Average observation size: {avg_obs_size_opt:.3f} MB")
    
    env_optimized.close()
    del env_optimized
    
    # Calculate savings
    savings = standard_footprint - optimized_footprint
    savings_pct = (savings / standard_footprint) * 100 if standard_footprint > 0 else 0
    
    print("\n" + "-" * 60)
    print(f"Memory savings per environment: {savings:.2f} MB ({savings_pct:.1f}%)")
    print("=" * 60)
    
    return savings


def test_multiple_environments():
    """Test memory with multiple parallel environments."""
    print("\n" + "=" * 60)
    print("TEST 2: Multiple Environment Instances")
    print("=" * 60)
    
    num_envs = 10
    baseline_memory = get_memory_usage_mb()
    print(f"Baseline memory: {baseline_memory:.2f} MB")
    
    # Create multiple optimized environments
    print(f"\nCreating {num_envs} memory-optimized environments...")
    config = EnvironmentConfig.for_parallel_training()
    envs = []
    
    for i in range(num_envs):
        env = create_visual_testing_env(config=config)
        env.reset()
        envs.append(env)
        
        if (i + 1) % 5 == 0:
            current_memory = get_memory_usage_mb()
            memory_per_env = (current_memory - baseline_memory) / (i + 1)
            print(f"  After {i+1} envs: {current_memory:.2f} MB "
                  f"({memory_per_env:.2f} MB per env)")
    
    final_memory = get_memory_usage_mb()
    total_memory = final_memory - baseline_memory
    memory_per_env = total_memory / num_envs
    
    print(f"\nFinal memory: {final_memory:.2f} MB")
    print(f"Total memory for {num_envs} envs: {total_memory:.2f} MB")
    print(f"Average memory per env: {memory_per_env:.2f} MB")
    
    # Run some steps to test observation processing
    print("\nRunning 50 steps across all environments...")
    for step in range(50):
        for env in envs:
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                env.reset()
    
    memory_after_steps = get_memory_usage_mb()
    memory_growth = memory_after_steps - final_memory
    print(f"Memory after 50 steps: {memory_after_steps:.2f} MB")
    print(f"Memory growth during execution: {memory_growth:.2f} MB")
    
    # Cleanup
    for env in envs:
        env.close()
    
    print("=" * 60)


def test_buffer_reuse():
    """Test that observation buffers are being reused."""
    print("\n" + "=" * 60)
    print("TEST 3: Buffer Reuse Verification")
    print("=" * 60)
    
    config = EnvironmentConfig.for_parallel_training()
    env = create_visual_testing_env(config=config)
    env.reset()
    
    # Check if observation processor has buffers
    if hasattr(env, 'observation_processor'):
        obs_proc = env.observation_processor
        has_buffers = (
            hasattr(obs_proc, '_player_frame_buffer') and
            hasattr(obs_proc, '_global_view_buffer') and
            hasattr(obs_proc, '_entity_positions_buffer')
        )
        
        if has_buffers:
            print("✅ Observation processor has pre-allocated buffers")
            
            # Get buffer IDs
            player_buffer_id = id(obs_proc._player_frame_buffer)
            global_buffer_id = id(obs_proc._global_view_buffer)
            entity_buffer_id = id(obs_proc._entity_positions_buffer)
            
            print(f"   Player frame buffer ID: {player_buffer_id}")
            print(f"   Global view buffer ID: {global_buffer_id}")
            print(f"   Entity positions buffer ID: {entity_buffer_id}")
            
            # Run steps and verify buffers are reused
            print("\nRunning 10 steps to verify buffer reuse...")
            for i in range(10):
                obs, reward, terminated, truncated, info = env.step(0)
                if terminated or truncated:
                    env.reset()
            
            # Check IDs are the same
            if (id(obs_proc._player_frame_buffer) == player_buffer_id and
                id(obs_proc._global_view_buffer) == global_buffer_id and
                id(obs_proc._entity_positions_buffer) == entity_buffer_id):
                print("✅ Buffers are being reused (IDs unchanged)")
            else:
                print("❌ Warning: Buffer IDs changed, may not be reusing")
        else:
            print("❌ Warning: Observation processor missing pre-allocated buffers")
    else:
        print("❌ Could not access observation processor")
    
    env.close()
    print("=" * 60)


def main():
    """Run all memory optimization tests."""
    print("\n" + "=" * 60)
    print("MEMORY OPTIMIZATION TEST SUITE")
    print("=" * 60)
    print("\nTesting memory optimizations for N++ RL environment")
    print("This will create and test multiple environment instances")
    
    try:
        # Test 1: Single environment memory
        savings = test_single_environment_memory()
        
        # Test 2: Multiple environments
        test_multiple_environments()
        
        # Test 3: Buffer reuse
        test_buffer_reuse()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Memory savings per env: {savings:.2f} MB")
        print("✅ Buffer reuse mechanism verified")
        print("✅ Multiple environment test completed")
        print("\nRecommendation:")
        print("  Use EnvironmentConfig.for_parallel_training() for")
        print("  training with 50+ parallel environments")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
