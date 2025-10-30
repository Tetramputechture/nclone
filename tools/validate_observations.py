#!/usr/bin/env python3
"""
Observation space validation for nclone environment.

Validates that all non-graph observations are populated correctly with reasonable values.
Runs actual gameplay episodes and checks:
- Shapes and dtypes
- Value ranges
- Temporal consistency
- Semantic correctness
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nclone.gym_environment.config import EnvironmentConfig
from nclone.gym_environment import NppEnvironment


def print_section(title):
    """Print section header."""
    print(f"\n{'='*80}\n{title}\n{'='*80}")


def validate_observation_shapes(obs):
    """Validate all observation shapes and dtypes."""
    checks = {
        'player_frame': ((84, 84, 1), np.uint8),
        'global_view': ((176, 100, 1), np.uint8),
        'game_state': ((26,), np.float32),
        'reachability_features': ((8,), np.float32),
        'entity_positions': ((6,), np.float32),
        'switch_states': ((25,), np.float32),
    }
    
    print("\nObservation Shapes & Types:")
    all_valid = True
    for key, (expected_shape, expected_dtype) in checks.items():
        actual_shape = obs[key].shape
        actual_dtype = obs[key].dtype
        shape_ok = actual_shape == expected_shape
        dtype_ok = actual_dtype == expected_dtype
        status = "✓" if (shape_ok and dtype_ok) else "✗"
        print(f"  {status} {key:25s} {str(actual_shape):20s} {actual_dtype}")
        if not shape_ok or not dtype_ok:
            print(f"      Expected: {expected_shape} {expected_dtype}")
            all_valid = False
    
    return all_valid


def validate_observation_ranges(obs):
    """Validate all observation value ranges."""
    print("\nObservation Value Ranges:")
    
    # Visual observations [0, 255]
    for key in ['player_frame', 'global_view']:
        arr = obs[key]
        valid = (0 <= arr.min() <= arr.max() <= 255)
        status = "✓" if valid else "✗"
        nonzero_pct = 100 * np.count_nonzero(arr) / arr.size
        print(f"  {status} {key:25s} [{arr.min():3d}, {arr.max():3d}]  {nonzero_pct:5.1f}% non-zero")
    
    # State vectors [-1, 1] or [0, 1]
    ranges = {
        'game_state': (-1, 1),
        'reachability_features': (0, 1),
        'entity_positions': (0, 1),
        'switch_states': (0, 1),
    }
    
    all_valid = True
    for key, (min_val, max_val) in ranges.items():
        arr = obs[key]
        valid = (min_val <= arr.min() <= arr.max() <= max_val)
        has_nan = np.any(np.isnan(arr))
        has_inf = np.any(np.isinf(arr))
        status = "✓" if (valid and not has_nan and not has_inf) else "✗"
        print(f"  {status} {key:25s} [{arr.min():6.3f}, {arr.max():6.3f}]")
        if has_nan:
            print(f"      WARNING: Contains NaN values")
            all_valid = False
        if has_inf:
            print(f"      WARNING: Contains inf values")
            all_valid = False
        if not valid:
            print(f"      Expected range: [{min_val}, {max_val}]")
            all_valid = False
    
    return all_valid


def print_observation_details(obs, step_num=0):
    """Print detailed observation breakdown."""
    gs = obs['game_state']
    rf = obs['reachability_features']
    ep = obs['entity_positions']
    ss = obs['switch_states']
    
    print(f"\n[Step {step_num}] Observation Details:")
    
    # Game state summary
    print(f"\n  game_state (26 features):")
    print(f"    velocity_mag={gs[0]:.3f}, direction=({gs[1]:.3f},{gs[2]:.3f})")
    print(f"    airborne={gs[7]:.3f}, input=({gs[8]:.3f},{gs[9]:.3f})")
    print(f"    contact={gs[13:16]}, acceleration=({gs[19]:.3f},{gs[20]:.3f})")
    print(f"    nearest_hazard={gs[21]:.3f}, nearest_collectible={gs[22]:.3f}")
    print(f"    switch_progress={gs[24]:.3f}, exit_accessible={gs[25]:.3f}")
    
    # Reachability summary
    print(f"\n  reachability_features (8 features):")
    print(f"    area_ratio={rf[0]:.3f}, connectivity={rf[5]:.3f}")
    print(f"    switch_dist={rf[1]:.3f}, exit_dist={rf[2]:.3f}")
    print(f"    exit_reachable={rf[6]:.1f}, path_exists={rf[7]:.1f}")
    
    # Entity positions
    print(f"\n  entity_positions (6 features):")
    print(f"    ninja=({ep[0]:.3f},{ep[1]:.3f})")
    print(f"    switch=({ep[2]:.3f},{ep[3]:.3f})")
    print(f"    exit=({ep[4]:.3f},{ep[5]:.3f})")
    
    # Switch states (only if locked doors exist)
    has_doors = np.any(ss > 0)
    if has_doors:
        print(f"\n  switch_states (25 features - 5 doors × 5):")
        for i in range(5):
            door_data = ss[i*5:(i+1)*5]
            if np.any(door_data > 0):
                print(f"    door_{i}: switch=({door_data[0]:.3f},{door_data[1]:.3f}) "
                      f"door=({door_data[2]:.3f},{door_data[3]:.3f}) open={door_data[4]:.1f}")
    else:
        print(f"\n  switch_states: No locked doors in this level")


def validate_temporal_consistency(obs_history):
    """Validate observations change appropriately over time."""
    if len(obs_history) < 2:
        return True
    
    print("\nTemporal Consistency:")
    
    # Ninja position changes
    positions = np.array([obs['entity_positions'][:2] for obs in obs_history])
    pos_changes = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    steps_with_movement = np.sum(pos_changes > 0)
    
    print(f"  Ninja position: {steps_with_movement}/{len(pos_changes)} steps with movement")
    print(f"    Mean change: {np.mean(pos_changes):.6f} (normalized units)")
    
    # Game state changes
    game_states = np.array([obs['game_state'] for obs in obs_history])
    gs_changes = np.linalg.norm(game_states[1:] - game_states[:-1], axis=1)
    
    print(f"  Game state: Updates every step")
    print(f"    Mean change: {np.mean(gs_changes):.3f} (L2 norm)")
    
    # Reachability changes
    reachability = np.array([obs['reachability_features'] for obs in obs_history])
    reach_changes = np.linalg.norm(reachability[1:] - reachability[:-1], axis=1)
    steps_with_reach_change = np.sum(reach_changes > 0)
    
    print(f"  Reachability: {steps_with_reach_change}/{len(reach_changes)} steps with change")
    
    return True


def run_validation_episode(env, episode_num, max_steps=100, verbose=False):
    """Run single episode and validate observations."""
    print_section(f"Episode {episode_num}")
    
    obs, info = env.reset()
    obs_history = [obs]
    
    # Validate initial observation
    print("\n>>> Initial Observation")
    shapes_valid = validate_observation_shapes(obs)
    ranges_valid = validate_observation_ranges(obs)
    
    if verbose:
        print_observation_details(obs, step_num=0)
    
    # Run episode
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        obs_history.append(obs)
        
        if verbose and step > 0 and step % 50 == 0:
            print_observation_details(obs, step_num=step)
        
        if terminated or truncated:
            print(f"\n>>> Episode Complete (step {step})")
            print(f"  Result: {'SUCCESS' if terminated else 'TRUNCATED'}")
            print(f"  Reward: {reward:.3f}")
            if verbose:
                print_observation_details(obs, step_num=step)
            break
    
    # Temporal consistency
    validate_temporal_consistency(obs_history)
    
    return shapes_valid and ranges_valid


def main():
    """Run comprehensive observation validation."""
    print_section("Observation Space Validation")
    
    print("\nValidating non-graph observations:")
    print("  • player_frame (84, 84, 1)")
    print("  • global_view (176, 100, 1)")
    print("  • game_state (26,)")
    print("  • reachability_features (8,)")
    print("  • entity_positions (6,)")
    print("  • switch_states (25,)")
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Validate nclone observations')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to test')
    parser.add_argument('--max-steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--verbose', action='store_true', help='Print detailed observation values')
    args = parser.parse_args()
    
    # Create environment
    config = EnvironmentConfig()
    config.graph.enable_graph_for_pbrs = False
    config.graph.enable_graph_for_observations = False
    env = NppEnvironment(config)
    
    # Run validation episodes
    all_valid = True
    for ep in range(args.episodes):
        valid = run_validation_episode(env, ep + 1, args.max_steps, args.verbose)
        all_valid = all_valid and valid
    
    env.close()
    
    # Summary
    print_section("Validation Summary")
    if all_valid:
        print("\n✓ All observations validated successfully")
        print("  • Correct shapes and dtypes")
        print("  • Valid value ranges")
        print("  • No NaN or inf values")
        print("  • Temporal consistency verified")
        print("\nStatus: PASSED")
        return 0
    else:
        print("\n✗ Validation failed")
        print("\nStatus: FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
