#!/usr/bin/env python3
"""Verify PBRS path calculation for level 006 'both flavours of ramp jumping'.

This script checks:
1. Level loads correctly in environment
2. PBRS combined path distance is calculated
3. PBRS potential is non-zero and varies with position
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.gym_environment.config import EnvironmentConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    level_path = "./test-single-level/006 both flavours of ramp jumping (and the control thereof)"
    
    logger.info(f"Initializing environment with level: {level_path}")
    
    # Create configuration
    config = EnvironmentConfig()
    config.custom_map_path = level_path
    config.enable_logging = False
    config.enable_visual_observations = False  # Skip rendering for speed
    
    # Create environment with the specific level
    env = NppEnvironment(config=config)
    
    # Reset to load the level
    logger.info("Resetting environment to load level...")
    obs, info = env.reset()
    
    # Extract PBRS metrics from the first step
    spawn_x = obs['player_x']
    spawn_y = obs['player_y']
    switch_x = obs['switch_x']
    switch_y = obs['switch_y']
    exit_x = obs['exit_door_x']
    exit_y = obs['exit_door_y']
    
    logger.info(f"\n=== Level Entity Positions ===")
    logger.info(f"Spawn: ({spawn_x:.0f}, {spawn_y:.0f})")
    logger.info(f"Switch: ({switch_x:.0f}, {switch_y:.0f})")
    logger.info(f"Exit: ({exit_x:.0f}, {exit_y:.0f})")
    
    # Take a step to trigger PBRS calculation
    obs, reward, terminated, truncated, info = env.step(0)  # NOOP
    
    # Check PBRS components
    if 'pbrs_components' in info:
        pbrs = info['pbrs_components']
        logger.info(f"\n=== PBRS Components (First Step) ===")
        logger.info(f"Current potential: {pbrs.get('current_potential', 'N/A'):.4f}")
        logger.info(f"Combined path distance: {pbrs.get('combined_path_distance', 'N/A'):.1f}px")
        logger.info(f"Distance to goal: {pbrs.get('distance_to_goal', 'N/A'):.1f}px")
        
        combined_path = pbrs.get('combined_path_distance', 0)
        if combined_path == 0:
            logger.error("❌ Combined path distance is 0 (path not calculated)")
            return 1
        elif combined_path == float('inf'):
            logger.error("❌ Combined path distance is inf (path unreachable)")
            return 1
        else:
            logger.info(f"✓ Combined path distance: {combined_path:.1f}px (path exists)")
    else:
        logger.error("❌ No PBRS components in info dict!")
        return 1
    
    # Test movement and PBRS gradient
    logger.info(f"\n=== Testing PBRS Gradient Direction ===")
    
    # Calculate direction to switch vs exit
    dx_switch = switch_x - spawn_x
    dy_switch = switch_y - spawn_y
    euclid_dist_switch = (dx_switch*dx_switch + dy_switch*dy_switch)**0.5
    
    dx_exit = exit_x - spawn_x
    dy_exit = exit_y - spawn_y
    euclid_dist_exit = (dx_exit*dx_exit + dy_exit*dy_exit)**0.5
    
    combined_path = pbrs.get('combined_path_distance', 0)
    initial_dist_to_goal = pbrs.get('distance_to_goal', 0)
    path_to_switch = combined_path - initial_dist_to_goal
    
    logger.info(f"Switch is LEFT/UP from spawn: dx={dx_switch:.0f}, dy={dy_switch:.0f} (euclidean: {euclid_dist_switch:.1f}px)")
    logger.info(f"Exit is RIGHT from spawn: dx={dx_exit:.0f}, dy={dy_exit:.0f} (euclidean: {euclid_dist_exit:.1f}px)")
    logger.info(f"Path distance spawn→switch: {path_to_switch:.1f}px")
    logger.info(f"Ratio (path/euclidean): {path_to_switch / max(1, euclid_dist_switch):.2f}x")
    
    # Test 1: Move RIGHT (toward exit direction, away from switch)
    logger.info(f"\n--- Test 1: Moving RIGHT (toward exit, away from switch) ---")
    env.reset()  # Reset to spawn
    obs, info = env.reset()
    initial_potential = info.get('pbrs_components', {}).get('current_potential', 0)
    
    steps_taken = 0
    potential_increased = 0
    potential_decreased = 0
    potential_unchanged = 0
    potential_changes_list = []
    
    for i in range(50):  # Take 50 steps moving RIGHT
        action = 2  # RIGHT
        
        obs, reward, terminated, truncated, info = env.step(action)
        steps_taken += 1
        
        if 'pbrs_components' in info:
            potential_change = info['pbrs_components'].get('potential_change', 0)
            potential_changes_list.append(potential_change)
            
            if potential_change > 0.001:
                potential_increased += 1
            elif potential_change < -0.001:
                potential_decreased += 1
            else:
                potential_unchanged += 1
                
            # Log first few steps
            if i < 5:
                curr_potential = info['pbrs_components'].get('current_potential', 0)
                logger.info(f"  Step {i+1}: potential_change={potential_change:+.4f}, current_potential={curr_potential:.4f}")
        
        if terminated or truncated:
            logger.info(f"Episode ended after {steps_taken} steps")
            break
    
    if steps_taken > 0:
        inc_pct = (potential_increased / steps_taken) * 100
        dec_pct = (potential_decreased / steps_taken) * 100
        unc_pct = (potential_unchanged / steps_taken) * 100
        
        logger.info(f"\nPotential changes over {steps_taken} steps:")
        logger.info(f"  Increased: {potential_increased} ({inc_pct:.1f}%)")
        logger.info(f"  Decreased: {potential_decreased} ({dec_pct:.1f}%)")
        logger.info(f"  Unchanged: {potential_unchanged} ({unc_pct:.1f}%)")
        
        # Analysis
        if inc_pct > 60:
            logger.info("✓ STRONG positive PBRS gradient (>60% increase)")
        elif inc_pct > 40:
            logger.info("✓ Good PBRS gradient (>40% increase)")
        elif inc_pct > dec_pct:
            logger.info("⚠ Weak positive gradient. Path may require non-obvious movements.")
        elif dec_pct > inc_pct:
            logger.warning("❌ NEGATIVE gradient dominant! Moving straight toward goal decreases potential.")
            logger.warning("   This suggests optimal path requires going AWAY from goal first (detour/momentum).")
        else:
            logger.warning("⚠ No clear gradient. Agent may be stuck or PBRS not providing signal.")
        
        # Check final distance
        if 'pbrs_components' in info:
            final_dist = info['pbrs_components'].get('distance_to_goal', 999)
            
            if final_dist < initial_dist_to_goal * 0.8:
                logger.info(f"✓ Agent moved closer to goal: {initial_dist_to_goal:.1f}px → {final_dist:.1f}px")
            else:
                logger.warning(f"⚠ Agent did not make significant progress: {initial_dist_to_goal:.1f}px → {final_dist:.1f}px")
        
        # Compute average potential change when moving RIGHT
        if potential_changes_list:
            avg_change = sum(potential_changes_list) / len(potential_changes_list)
            logger.info(f"\nAverage potential change when moving RIGHT: {avg_change:+.6f}")
    
    # Test 2: Move LEFT (toward switch, away from exit)
    logger.info(f"\n--- Test 2: Moving LEFT (toward switch, away from exit) ---")
    obs, info = env.reset()
    
    steps_taken2 = 0
    potential_increased2 = 0
    potential_decreased2 = 0
    potential_unchanged2 = 0
    potential_changes_list2 = []
    
    for i in range(50):  # Take 50 steps moving LEFT
        action = 1  # LEFT
        
        obs, reward, terminated, truncated, info = env.step(action)
        steps_taken2 += 1
        
        if 'pbrs_components' in info:
            potential_change = info['pbrs_components'].get('potential_change', 0)
            potential_changes_list2.append(potential_change)
            
            if potential_change > 0.001:
                potential_increased2 += 1
            elif potential_change < -0.001:
                potential_decreased2 += 1
            else:
                potential_unchanged2 += 1
                
            # Log first few steps
            if i < 5:
                curr_potential = info['pbrs_components'].get('current_potential', 0)
                logger.info(f"  Step {i+1}: potential_change={potential_change:+.4f}, current_potential={curr_potential:.4f}")
        
        if terminated or truncated:
            logger.info(f"Episode ended after {steps_taken2} steps")
            break
    
    if steps_taken2 > 0:
        inc_pct2 = (potential_increased2 / steps_taken2) * 100
        dec_pct2 = (potential_decreased2 / steps_taken2) * 100
        
        logger.info(f"\nPotential changes over {steps_taken2} steps:")
        logger.info(f"  Increased: {potential_increased2} ({inc_pct2:.1f}%)")
        logger.info(f"  Decreased: {potential_decreased2} ({dec_pct2:.1f}%)")
        
        if potential_changes_list2:
            avg_change2 = sum(potential_changes_list2) / len(potential_changes_list2)
            logger.info(f"Average potential change when moving LEFT: {avg_change2:+.6f}")
    
    # Compare results
    logger.info(f"\n=== PBRS GRADIENT ANALYSIS ===")
    if potential_changes_list and potential_changes_list2:
        avg_right = sum(potential_changes_list) / len(potential_changes_list)
        avg_left = sum(potential_changes_list2) / len(potential_changes_list2)
        
        logger.info(f"Moving RIGHT: avg potential change = {avg_right:+.6f}")
        logger.info(f"Moving LEFT:  avg potential change = {avg_left:+.6f}")
        
        if avg_right > 0.01 and avg_right > avg_left:
            logger.info("✓ CORRECT: Moving RIGHT gives MORE positive PBRS gradient")
            logger.info("  This suggests optimal path goes RIGHT first (toward exit direction)")
        elif avg_left > 0.01 and avg_left > avg_right:
            logger.warning("⚠ Moving LEFT gives more positive gradient")
            logger.warning("  This suggests optimal path goes LEFT first (toward switch)")
        elif abs(avg_right) < 0.01 and abs(avg_left) < 0.01:
            logger.warning("⚠ Both directions give near-zero gradient (agent stuck at spawn?)")
        else:
            logger.warning(f"⚠ Unclear gradient: RIGHT={avg_right:+.4f}, LEFT={avg_left:+.4f}")
    
    env.close()
    
    logger.info("\n" + "="*60)
    logger.info("✓ PBRS verification complete. Check log for any warnings.")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

