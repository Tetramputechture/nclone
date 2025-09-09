#!/usr/bin/env python3
"""
Test wall jump detection specifically.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.graph.movement_classifier import MovementClassifier, MovementType
from nclone.constants.physics_constants import TILE_PIXEL_SIZE

def test_wall_jump_detection():
    """Test wall jump detection in narrow vertical corridor."""
    print("=== Testing Wall Jump Detection ===")
    
    classifier = MovementClassifier()
    
    # Test vertical movement in very narrow corridor (1 tile wide)
    # This should trigger wall jump detection
    start_pos = (60, 180)  # Bottom of narrow corridor
    end_pos = (60, 108)    # Middle of corridor (72 pixels up)
    
    movement_type, physics_params = classifier.classify_movement(
        start_pos, end_pos, None, None
    )
    
    print(f"Narrow Vertical Corridor Movement:")
    print(f"  Start: {start_pos}")
    print(f"  End: {end_pos}")
    print(f"  Movement Type: {MovementType(movement_type).name}")
    print(f"  Distance: {physics_params['distance']:.1f} pixels")
    print(f"  Height Diff: {physics_params['height_diff']:.1f} pixels")
    print(f"  Horizontal Diff: {abs(end_pos[0] - start_pos[0]):.1f} pixels")
    print(f"  Expected: WALL_JUMP or JUMP for vertical movement")
    
    # Test with even more vertical movement
    start_pos2 = (60, 180)
    end_pos2 = (60, 36)  # Top of corridor (144 pixels up)
    
    movement_type2, physics_params2 = classifier.classify_movement(
        start_pos2, end_pos2, None, None
    )
    
    print(f"\nLarge Vertical Movement:")
    print(f"  Start: {start_pos2}")
    print(f"  End: {end_pos2}")
    print(f"  Movement Type: {MovementType(movement_type2).name}")
    print(f"  Distance: {physics_params2['distance']:.1f} pixels")
    print(f"  Height Diff: {physics_params2['height_diff']:.1f} pixels")
    print(f"  Expected: WALL_JUMP preferred for large vertical movement")
    
    # Test wall jump validation directly
    print(f"\n=== Direct Wall Jump Validation ===")
    if hasattr(classifier.trajectory_validator, 'validate_wall_jump_trajectory'):
        wall_jump_result = classifier.trajectory_validator.validate_wall_jump_trajectory(
            start_pos, end_pos, (-1, 0)  # Left wall
        )
        print(f"Wall Jump from Left Wall:")
        print(f"  Valid: {wall_jump_result.is_valid}")
        print(f"  Required Velocity: {wall_jump_result.required_velocity}")
        print(f"  Flight Time: {wall_jump_result.flight_time:.2f}")
        print(f"  Energy Cost: {wall_jump_result.energy_cost:.2f}")
        print(f"  Risk Factor: {wall_jump_result.risk_factor:.2f}")
        if wall_jump_result.failure_reason:
            print(f"  Failure Reason: {wall_jump_result.failure_reason}")
    else:
        print("Wall jump validation not available")

if __name__ == "__main__":
    test_wall_jump_detection()