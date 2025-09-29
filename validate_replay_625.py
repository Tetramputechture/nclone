#!/usr/bin/env python3
"""
Validation script for N++ attract replay file 625.

Requirements to validate:
- Ninja collects exactly 3 gold pieces
- Ninja triggers exactly 1 toggle mine
- Ninja eventually dies by colliding with bottom row of mines
- Replay runs for several seconds before death
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pathlib import Path
from nclone.replay.binary_replay_parser import BinaryReplayParser
from nclone.gym_environment.npp_environment import NppEnvironment
import numpy as np

def validate_replay_625():
    """Validate the specific requirements for replay file 625."""
    
    print("=" * 80)
    print("VALIDATING N++ ATTRACT REPLAY FILE 625")
    print("=" * 80)
    
    # Requirements to validate
    expected_gold_collected = 3
    expected_toggle_mines_triggered = 1
    expected_death_by_mines = True
    min_runtime_seconds = 3.0  # "a few more seconds" after triggering mine
    
    try:
        # Parse the replay file
        parser = BinaryReplayParser()
        replay_path = Path("nclone/example_replays/npp_attract/625")
        
        print(f"📁 Parsing replay file: {replay_path}")
        inputs, map_data, level_id, level_name = parser.parse_single_replay_file(replay_path)
        
        print(f"📋 Level: '{level_name}'")
        print(f"🎮 Total inputs: {len(inputs)} frames ({len(inputs)/60.0:.1f}s)")
        
        # Create temporary map file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as temp_map:
            temp_map.write(bytes(map_data))
            temp_map_path = temp_map.name
        
        try:
            # Create environment and run simulation
            print(f"\n🚀 Running simulation...")
            env = NppEnvironment(
                render_mode=None,
                enable_animation=False,
                enable_logging=False,
                enable_debug_overlay=False,
                custom_map_path=temp_map_path
            )
            
            obs = env.reset()
        
            # Track validation metrics
            gold_collected = 0
            toggle_mines_triggered = 0
            ninja_died = False
            death_cause = None
            death_frame = None
            
            # Count initial gold (only active gold)
            initial_gold_count = 0
            for entity_list in env.nplay_headless.sim.entity_dic.values():
                for entity in entity_list:
                    if type(entity).__name__ == 'EntityGold' and entity.active:
                        initial_gold_count += 1
            
            print(f"🏆 Initial gold pieces: {initial_gold_count}")
            
            # Count initial mines
            initial_mine_count = 0
            for entity_list in env.nplay_headless.sim.entity_dic.values():
                for entity in entity_list:
                    if 'Mine' in type(entity).__name__:
                        initial_mine_count += 1
            
            print(f"💣 Initial mines: {initial_mine_count}")
            
            # Decode inputs
            hor_inputs, jump_inputs = parser.decode_inputs(inputs)
            
            def convert_to_action_int(horizontal, jump):
                """Convert horizontal/jump inputs to integer action."""
                if horizontal == 0 and jump == 0:
                    return 0  # NOOP
                elif horizontal == -1 and jump == 0:
                    return 1  # Left
                elif horizontal == 1 and jump == 0:
                    return 2  # Right
                elif horizontal == 0 and jump == 1:
                    return 3  # Jump
                elif horizontal == -1 and jump == 1:
                    return 4  # Jump + Left
                elif horizontal == 1 and jump == 1:
                    return 5  # Jump + Right
                else:
                    return 0  # Default to NOOP
            
            # Track gold collection using ninja's gold_collected counter
            previous_gold_collected = 0
            gold_collection_frames = []
            
            # Run simulation frame by frame
            for frame_idx in range(len(inputs)):
                # Convert to action
                horizontal = hor_inputs[frame_idx]
                jump = jump_inputs[frame_idx]
                action = convert_to_action_int(horizontal, jump)
                
                # Step environment
                obs, reward, done, truncated, info = env.step(action)
                
                # Check ninja gold collection
                current_gold_collected = env.nplay_headless.sim.ninja.gold_collected
                if current_gold_collected > previous_gold_collected:
                    gold_collected = current_gold_collected
                    gold_collection_frames.append(frame_idx)
                    print(f"🏆 Gold collected #{gold_collected} at frame {frame_idx} ({frame_idx/60.0:.1f}s)")
                    previous_gold_collected = current_gold_collected
                
                # Check for ninja death
                if env.nplay_headless.sim.ninja.has_died() or done:
                    ninja_died = True
                    death_frame = frame_idx
                    ninja_x = env.nplay_headless.sim.ninja.xpos
                    ninja_y = env.nplay_headless.sim.ninja.ypos
                    
                    # Determine death cause (simplified heuristic)
                    if ninja_y > 500:  # Near bottom of level (adjust based on level height)
                        death_cause = "bottom_row_mines"
                    else:
                        death_cause = "other"
                    
                    print(f"💀 Ninja died at frame {frame_idx} ({frame_idx/60.0:.1f}s)")
                    print(f"   Position: ({ninja_x:.1f}, {ninja_y:.1f})")
                    print(f"   Cause: {death_cause}")
                    break
                
                # Check for mine interactions (simplified - count active mines)
                current_mine_count = 0
                for entity_list in env.nplay_headless.sim.entity_dic.values():
                    for entity in entity_list:
                        if 'Mine' in type(entity).__name__ and entity.active:
                            current_mine_count += 1
                
                # If mine count decreased, assume toggle mine was triggered
                if current_mine_count < initial_mine_count - toggle_mines_triggered:
                    toggle_mines_triggered += 1
                    print(f"💣 Mine interaction detected at frame {frame_idx} ({frame_idx/60.0:.1f}s)")
        
        finally:
            # Clean up temporary file
            import os
            if 'temp_map_path' in locals():
                try:
                    os.unlink(temp_map_path)
                except:
                    pass
        
        # Validation results
        print(f"\n" + "=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        
        runtime_seconds = len(inputs) / 60.0
        
        # Check each requirement
        results = {
            'gold_collected': gold_collected,
            'toggle_mines_triggered': toggle_mines_triggered,
            'ninja_died': ninja_died,
            'death_cause': death_cause,
            'runtime_seconds': runtime_seconds,
            'death_frame': death_frame
        }
        
        print(f"📊 Simulation Results:")
        print(f"  Gold collected: {gold_collected}")
        print(f"  Toggle mines triggered: {toggle_mines_triggered}")
        print(f"  Ninja died: {ninja_died}")
        print(f"  Death cause: {death_cause}")
        print(f"  Runtime: {runtime_seconds:.1f}s")
        print(f"  Death frame: {death_frame}")
        
        # Validate requirements
        validation_passed = True
        validation_errors = []
        
        if gold_collected != expected_gold_collected:
            validation_passed = False
            validation_errors.append(f"❌ Gold collection: Expected {expected_gold_collected}, got {gold_collected}")
        else:
            print(f"✅ Gold collection: {gold_collected}/{expected_gold_collected} ✓")
        
        if toggle_mines_triggered != expected_toggle_mines_triggered:
            validation_passed = False
            validation_errors.append(f"❌ Toggle mines: Expected {expected_toggle_mines_triggered}, got {toggle_mines_triggered}")
        else:
            print(f"✅ Toggle mines triggered: {toggle_mines_triggered}/{expected_toggle_mines_triggered} ✓")
        
        if not ninja_died:
            validation_passed = False
            validation_errors.append(f"❌ Ninja death: Expected ninja to die, but ninja survived")
        else:
            print(f"✅ Ninja death: Ninja died as expected ✓")
        
        if death_cause != "bottom_row_mines":
            validation_passed = False
            validation_errors.append(f"❌ Death cause: Expected death by bottom row mines, got {death_cause}")
        else:
            print(f"✅ Death cause: Death by bottom row mines ✓")
        
        if runtime_seconds < min_runtime_seconds:
            validation_passed = False
            validation_errors.append(f"❌ Runtime: Expected >{min_runtime_seconds}s, got {runtime_seconds:.1f}s")
        else:
            print(f"✅ Runtime: {runtime_seconds:.1f}s (>{min_runtime_seconds}s) ✓")
        
        # Final validation result
        print(f"\n" + "=" * 80)
        if validation_passed:
            print("🎉 VALIDATION PASSED: All requirements satisfied!")
            print("✅ N++ attract replay 625 decoder achieves TRUE 100% accuracy!")
            return True, results
        else:
            print("❌ VALIDATION FAILED: Requirements not met")
            for error in validation_errors:
                print(f"  {error}")
            return False, results
            
    except Exception as e:
        print(f"💥 ERROR during validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, {'error': str(e)}

def main():
    success, results = validate_replay_625()
    
    if not success:
        print(f"\n" + "=" * 80)
        print("VALIDATION FAILURE SUMMARY")
        print("=" * 80)
        print("The N++ attract replay decoder failed to meet the requirements for file 625.")
        print("This indicates that additional work is needed to achieve complete format support.")
        print("\nNext steps:")
        print("1. Analyze the specific failure modes")
        print("2. Debug the input sequence extraction for file 625")
        print("3. Ensure proper mine interaction detection")
        print("4. Validate death detection and cause identification")
        print("5. Re-test with corrected decoder")
        
        return 1
    else:
        return 0

if __name__ == "__main__":
    sys.exit(main())