#!/usr/bin/env python3
"""
Diagnostic script to understand PBRS reward magnitude.
Checks weight application and expected values.
"""

import sys
sys.path.insert(0, '/home/tetra/projects/nclone')

from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.reward_calculation.reward_constants import (
    PBRS_SWITCH_DISTANCE_SCALE,
    PBRS_EXIT_DISTANCE_SCALE,
    PBRS_GAMMA,
)

def diagnose_pbrs():
    """Diagnose PBRS calculation and expected magnitudes."""
    
    print("=" * 70)
    print("PBRS DIAGNOSTIC")
    print("=" * 70)
    
    # Test config at 0-5% success rate (current situation)
    config = RewardConfig()
    config.update(timesteps=1_200_000, success_rate=0.03)  # 3% success
    
    print(f"\n1. REWARD CONFIG STATE:")
    print(f"   Success rate: {config.recent_success_rate:.1%}")
    print(f"   Training phase: {config.training_phase}")
    print(f"   PBRS objective weight: {config.pbrs_objective_weight}")
    print(f"   PBRS normalization scale: {config.pbrs_normalization_scale}")
    
    print(f"\n2. PBRS CONSTANTS:")
    print(f"   PBRS_GAMMA: {PBRS_GAMMA}")
    print(f"   PBRS_SWITCH_DISTANCE_SCALE: {PBRS_SWITCH_DISTANCE_SCALE}")
    print(f"   PBRS_EXIT_DISTANCE_SCALE: {PBRS_EXIT_DISTANCE_SCALE}")
    
    # Simulate PBRS calculation
    print(f"\n3. EXPECTED PBRS MAGNITUDES:")
    print(f"   (Based on level with ~200px path, 800px cap)")
    
    combined_path_distance = 200.0  # From user's graph
    max_normalization = 800.0
    effective_normalization = min(combined_path_distance, max_normalization)
    
    print(f"\n   Path setup:")
    print(f"   - Combined path distance: {combined_path_distance}px")
    print(f"   - Normalization cap: {max_normalization}px")
    print(f"   - Effective normalization: {effective_normalization}px")
    
    # Calculate potentials at different positions
    print(f"\n   Potential function Φ(s):")
    
    distances = [200, 150, 100, 50, 0]  # Different distances to goal
    potentials = []
    
    for dist in distances:
        # This is the objective_distance_potential calculation
        normalized_dist = dist / effective_normalization
        objective_pot = max(0.0, min(1.0, 1.0 - normalized_dist))
        
        # Apply weight and scale (switch phase)
        potential = PBRS_SWITCH_DISTANCE_SCALE * objective_pot * config.pbrs_objective_weight
        potentials.append(potential)
        
        print(f"   - At distance {dist:3d}px: Φ = {potential:7.2f}")
    
    # Calculate PBRS for typical movements
    print(f"\n   PBRS reward F(s,s') = γ * Φ(s') - Φ(s):")
    
    movements = [
        (200, 197, "3px toward goal (typical with 4-frame skip)"),
        (200, 199, "1px toward goal (very slow)"),
        (200, 195, "5px toward goal (good movement)"),
        (100, 103, "3px away from goal (backtracking)"),
    ]
    
    for start_dist, end_dist, description in movements:
        # Calculate start potential
        start_norm = start_dist / effective_normalization
        start_obj = max(0.0, min(1.0, 1.0 - start_norm))
        start_pot = PBRS_SWITCH_DISTANCE_SCALE * start_obj * config.pbrs_objective_weight
        
        # Calculate end potential
        end_norm = end_dist / effective_normalization
        end_obj = max(0.0, min(1.0, 1.0 - end_norm))
        end_pot = PBRS_SWITCH_DISTANCE_SCALE * end_obj * config.pbrs_objective_weight
        
        # PBRS reward
        pbrs = PBRS_GAMMA * end_pot - start_pot
        
        print(f"   - {description}:")
        print(f"     Φ(s)={start_pot:.4f}, Φ(s')={end_pot:.4f}, F(s,s')={pbrs:+.4f}")
    
    # Compare with observed values
    print(f"\n4. COMPARISON WITH OBSERVED VALUES:")
    print(f"   TensorBoard shows: pbrs_mean ≈ 0.01 to 0.03 per step")
    print(f"   Expected (3px movement): ~0.75 per step")
    print(f"   Ratio: {0.02 / 0.75:.4f} (observed/expected)")
    print(f"\n   DIAGNOSIS:")
    print(f"   - If agent moving 3px/step → Weight not being applied properly")
    print(f"   - If agent moving 0.08px/step → Movement is the problem")
    print(f"   - Check actual movement distance in TensorBoard or logs")
    
    # Calculate what movement would give observed PBRS
    observed_pbrs = 0.02  # Middle of 0.01-0.03 range
    movement_needed = observed_pbrs * effective_normalization / config.pbrs_objective_weight
    print(f"\n   To get PBRS={observed_pbrs:.3f}, agent must move:")
    print(f"   - Distance: {movement_needed:.3f}px per action")
    print(f"   - With 4-frame skip, that's {movement_needed/4:.4f}px per frame")
    print(f"   - This is EXTREMELY SLOW (typical is 0.5-3px/frame)")
    
    print(f"\n5. RECOMMENDED ACTIONS:")
    print(f"   1. Check actual movement distance per action in logs")
    print(f"   2. Verify config.pbrs_objective_weight is being used (not default)")
    print(f"   3. Check if weight is being divided somewhere incorrectly")
    print(f"   4. Verify frame skip is working (should be 4 frames/action)")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    diagnose_pbrs()

