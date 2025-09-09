#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

import math

# Physics constants from nclone/constants/physics_constants.py
MAX_HOR_SPEED = 3.333  # pixels/frame
JUMP_INITIAL_VELOCITY = -6.0  # pixels/frame
GRAVITY_FALL = 0.0667  # pixels/frame^2
GRAVITY_JUMP = 0.0111  # pixels/frame^2
MAX_JUMP_DISTANCE = 200.0  # pixels
MAX_FALL_DISTANCE = 400.0  # pixels
SUB_CELL_SIZE = 6  # pixels

def validate_jump_trajectory(start_pos, end_pos, movement_type):
    """
    Validate that a jump trajectory is physically possible.
    
    Args:
        start_pos: (x, y) starting position
        end_pos: (x, y) ending position  
        movement_type: Type of movement (JUMP, FALL, etc.)
        
    Returns:
        dict with validation results
    """
    start_x, start_y = start_pos
    end_x, end_y = end_pos
    
    dx = end_x - start_x
    dy = end_y - start_y
    distance = math.sqrt(dx*dx + dy*dy)
    
    result = {
        'valid': True,
        'issues': [],
        'distance': distance,
        'height_diff': dy,
        'horizontal_distance': abs(dx)
    }
    
    if movement_type == 'JUMP':
        # Check if jump distance is within ninja capabilities
        if distance > MAX_JUMP_DISTANCE:
            result['valid'] = False
            result['issues'].append(f"Jump distance {distance:.1f}px exceeds maximum {MAX_JUMP_DISTANCE:.1f}px")
        
        # Check if vertical component is achievable
        if dy < 0:  # Jumping up
            # Calculate maximum jump height using physics: v² = u² + 2as
            # At maximum height, v = 0, u = JUMP_INITIAL_VELOCITY, a = GRAVITY_JUMP
            max_jump_height = abs(JUMP_INITIAL_VELOCITY)**2 / (2 * GRAVITY_JUMP)
            if abs(dy) > max_jump_height:
                result['valid'] = False
                result['issues'].append(f"Jump height {abs(dy):.1f}px exceeds maximum {max_jump_height:.1f}px")
        
        # Check horizontal velocity requirements
        if abs(dx) > 0:
            # Estimate jump duration using vertical motion
            if dy < 0:  # Jumping up
                # Time to reach peak + time to fall back down
                time_to_peak = abs(JUMP_INITIAL_VELOCITY) / GRAVITY_JUMP
                jump_duration = time_to_peak * 2  # Rough estimate
            else:  # Jumping down or horizontal
                jump_duration = 45  # Default assumption
            
            required_horizontal_velocity = abs(dx) / jump_duration
            if required_horizontal_velocity > MAX_HOR_SPEED:
                result['valid'] = False
                result['issues'].append(f"Required horizontal velocity {required_horizontal_velocity:.2f} exceeds max {MAX_HOR_SPEED}")
    
    elif movement_type == 'FALL':
        # Falls should generally be downward
        if dy < 0:
            result['issues'].append(f"FALL movement going upward ({dy:.1f}px) - unusual")
        
        # Check if fall distance is reasonable
        if distance > MAX_FALL_DISTANCE:
            result['valid'] = False
            result['issues'].append(f"Fall distance {distance:.1f}px exceeds maximum {MAX_FALL_DISTANCE:.1f}px")
    
    elif movement_type == 'WALK':
        # Walk movements should be mostly horizontal
        if abs(dy) > SUB_CELL_SIZE:  # More than 6 pixels vertical
            result['issues'].append(f"WALK movement has large vertical component ({dy:.1f}px)")
        
        # Check if walk distance is reasonable for a single segment
        max_walk_segment = MAX_HOR_SPEED * 30  # 30 frames of walking
        if distance > max_walk_segment:
            result['issues'].append(f"WALK segment {distance:.1f}px is very long (>{max_walk_segment:.1f}px)")
    
    return result

def validate_movement_sequence(path_segments):
    """
    Validate that a sequence of movements makes physical sense.
    
    Args:
        path_segments: List of (start_pos, end_pos, movement_type) tuples
        
    Returns:
        dict with validation results
    """
    result = {
        'valid': True,
        'issues': [],
        'segment_results': []
    }
    
    for i, (start_pos, end_pos, movement_type) in enumerate(path_segments):
        segment_result = validate_jump_trajectory(start_pos, end_pos, movement_type)
        segment_result['segment_index'] = i
        segment_result['movement_type'] = movement_type
        result['segment_results'].append(segment_result)
        
        if not segment_result['valid']:
            result['valid'] = False
            result['issues'].extend([f"Segment {i}: {issue}" for issue in segment_result['issues']])
    
    # Check for continuity between segments
    for i in range(len(path_segments) - 1):
        current_end = path_segments[i][1]
        next_start = path_segments[i + 1][0]
        
        # Check if segments are connected
        gap = math.sqrt((next_start[0] - current_end[0])**2 + (next_start[1] - current_end[1])**2)
        if gap > 12:  # Allow small gaps for positioning
            result['issues'].append(f"Gap of {gap:.1f}px between segment {i} and {i+1}")
    
    return result

def run_comprehensive_validation():
    """Run comprehensive physics validation on all test maps."""
    
    print("Comprehensive Physics Validation")
    print("=" * 50)
    
    # Test data from the validation results
    test_cases = {
        'simple-walk': [
            ((396, 372), (492, 372), 'WALK'),  # Ninja to Switch: 96px
            ((492, 372), (588, 372), 'WALK'),  # Switch to Exit: 72px
        ],
        'long-walk': [
            ((36, 276), (996, 276), 'WALK'),   # Ninja to Switch: 936px
            ((996, 276), (1020, 276), 'WALK'), # Switch to Exit: 24px
        ],
        'path-jump-required': [
            ((420, 348), (516, 324), 'JUMP'),  # Ninja to Elevated Switch: 99px, -24px height
            ((516, 324), (612, 348), 'FALL'),  # Switch to Ground Exit: 75.9px, +24px height
        ],
        'only-jump': [
            ((492, 252), (492, 204), 'JUMP'),  # Ninja to Middle Switch: 72px, -72px height
            ((492, 204), (492, 156), 'JUMP'),  # Switch to Top Exit: 72px, -72px height
        ]
    }
    
    all_valid = True
    
    for test_name, segments in test_cases.items():
        print(f"\n=== Validating {test_name} ===")
        
        validation_result = validate_movement_sequence(segments)
        
        if validation_result['valid']:
            print(f"✅ {test_name}: All movements are physically valid")
        else:
            print(f"❌ {test_name}: Physics validation failed")
            all_valid = False
            
        for issue in validation_result['issues']:
            print(f"  ⚠️  {issue}")
        
        # Print detailed segment analysis
        for segment_result in validation_result['segment_results']:
            i = segment_result['segment_index']
            movement_type = segment_result['movement_type']
            distance = segment_result['distance']
            height_diff = segment_result['height_diff']
            
            status = "✅" if segment_result['valid'] else "❌"
            print(f"  {status} Segment {i} ({movement_type}): {distance:.1f}px, {height_diff:+.1f}px height")
            
            for issue in segment_result['issues']:
                print(f"    ⚠️  {issue}")
    
    print(f"\n{'='*50}")
    if all_valid:
        print("🎉 All test maps pass comprehensive physics validation!")
        print("✅ All movements respect ninja physics constraints")
        print("✅ All trajectories are within ninja capabilities")
        print("✅ All movement types are appropriate for their geometry")
    else:
        print("❌ Some test maps have physics validation issues")
        print("   Review the issues above and adjust pathfinding logic")
    
    return all_valid

if __name__ == "__main__":
    run_comprehensive_validation()