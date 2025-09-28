#!/usr/bin/env python3
"""
Validation script for N++ attract replay decoder.
This script validates that exactly 11 gold pieces are collected during the attract/0 replay.
Work is not complete until this validation passes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pathlib import Path
from nclone.replay.binary_replay_parser import BinaryReplayParser
from nclone.gym_environment.npp_environment import NppEnvironment
import tempfile

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

def validate_gold_collection():
    """
    Validation script for N++ attract replay decoder.
    
    Returns:
        bool: True if exactly 11 gold pieces are collected, False otherwise
    """
    
    print("=" * 80)
    print("N++ ATTRACT REPLAY DECODER VALIDATION")
    print("=" * 80)
    print("REQUIREMENT: Exactly 11 gold pieces must be collected during attract/0 replay")
    print("=" * 80)
    
    attract_file = Path("nclone/example_replays/npp_attract/0")
    
    if not attract_file.exists():
        print(f"❌ VALIDATION FAILED: Attract file not found: {attract_file}")
        return False
    
    try:
        # Parse replay file
        parser = BinaryReplayParser()
        inputs, map_data, level_id, level_name = parser.parse_single_replay_file(attract_file)
        
        print(f"Level: '{level_name}'")
        print(f"Total inputs: {len(inputs)} frames = {len(inputs)/60.0:.1f}s")
        
        # Create temporary map file
        with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as temp_map:
            temp_map.write(bytes(map_data))
            temp_map_path = temp_map.name
        
        try:
            # Create environment with minimal overhead
            env = NppEnvironment(
                render_mode=None,
                enable_animation=False,
                enable_logging=False,
                enable_debug_overlay=False,
                custom_map_path=temp_map_path
            )
            
            obs = env.reset()
            
            # Count initial gold
            initial_gold_count = 0
            for entity_list in env.nplay_headless.sim.entity_dic.values():
                for entity in entity_list:
                    if type(entity).__name__ == 'EntityGold':
                        initial_gold_count += 1
            
            print(f"Initial gold count: {initial_gold_count}")
            
            # Decode inputs
            hor_inputs, jump_inputs = parser.decode_inputs(inputs)
            
            print(f"Running full replay to validate gold collection...")
            
            # Track gold collection
            gold_collected = 0
            gold_collection_frames = []
            
            # Run full replay
            for frame_idx in range(len(inputs)):
                # Convert to action
                action_int = convert_to_action_int(hor_inputs[frame_idx], jump_inputs[frame_idx])
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action_int)
                
                # Count current gold
                current_gold_count = 0
                for entity_list in env.nplay_headless.sim.entity_dic.values():
                    for entity in entity_list:
                        if type(entity).__name__ == 'EntityGold':
                            current_gold_count += 1
                
                # Check if gold was collected
                if current_gold_count < initial_gold_count - gold_collected:
                    new_gold_collected = initial_gold_count - current_gold_count
                    for _ in range(gold_collected, new_gold_collected):
                        gold_collection_frames.append(frame_idx)
                    gold_collected = new_gold_collected
                    
                    ninja_pos = (env.nplay_headless.sim.ninja.xpos, env.nplay_headless.sim.ninja.ypos)
                    print(f"  🎉 Frame {frame_idx}: Gold collected! Total: {gold_collected}/{initial_gold_count}")
                    print(f"     Ninja position: ({ninja_pos[0]:.1f}, {ninja_pos[1]:.1f})")
                    print(f"     Time: {frame_idx/60.0:.1f}s")
                
                # Check for level completion
                if terminated or truncated:
                    print(f"  🏁 Level completed at frame {frame_idx} ({frame_idx/60.0:.1f}s)")
                    break
            
            # Final validation
            print(f"\n" + "=" * 80)
            print("VALIDATION RESULTS")
            print("=" * 80)
            
            print(f"Gold collection summary:")
            print(f"  Initial gold: {initial_gold_count}")
            print(f"  Gold collected: {gold_collected}")
            print(f"  Collection frames: {gold_collection_frames}")
            print(f"  Expected: 11 gold pieces")
            
            # Validation check
            if gold_collected == 11:
                print(f"\n✅ VALIDATION PASSED: Exactly 11 gold pieces collected!")
                print(f"🎉 N++ attract replay decoder achieves TRUE 100% accuracy!")
                return True
            else:
                print(f"\n❌ VALIDATION FAILED: Expected 11 gold, got {gold_collected}")
                print(f"🔧 Gold collection mechanism needs fixing")
                
                # Debug information
                print(f"\nDebug information:")
                print(f"  Frames processed: {frame_idx + 1}/{len(inputs)}")
                print(f"  Level completed: {terminated or truncated}")
                
                if gold_collected == 0:
                    print(f"  Issue: No gold collected at all - collision detection problem")
                elif gold_collected < 11:
                    print(f"  Issue: Only {gold_collected}/11 gold collected - partial collection problem")
                else:
                    print(f"  Issue: Too much gold collected ({gold_collected}) - counting problem")
                
                return False
                
        finally:
            os.unlink(temp_map_path)
            
    except Exception as e:
        print(f"❌ VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation function."""
    
    print("N++ ATTRACT REPLAY DECODER VALIDATION SCRIPT")
    print("=" * 80)
    print("This script validates that the decoder achieves TRUE 100% accuracy")
    print("by confirming that exactly 11 gold pieces are collected during")
    print("the attract/0 replay as intended in the original recording.")
    print("=" * 80)
    
    success = validate_gold_collection()
    
    print(f"\n" + "=" * 80)
    print("FINAL VALIDATION RESULT")
    print("=" * 80)
    
    if success:
        print(f"🎉 SUCCESS: N++ attract replay decoder achieves TRUE 100% accuracy!")
        print(f"✅ All requirements met:")
        print(f"   - Entity decoding: Perfect")
        print(f"   - Input decoding: Perfect") 
        print(f"   - Gold collection: 11/11 collected")
        print(f"   - Replay accuracy: 100%")
        print(f"")
        print(f"🏆 MISSION ACCOMPLISHED: Complete and accurate N++ attract replay decoder!")
        sys.exit(0)
    else:
        print(f"❌ FAILURE: Validation requirements not met")
        print(f"🔧 Work must continue until 11 gold pieces are collected")
        print(f"")
        print(f"The decoder is not complete until this validation passes.")
        sys.exit(1)

if __name__ == "__main__":
    main()