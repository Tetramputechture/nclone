#!/usr/bin/env python3
"""
Simple test to understand current pathfinding behavior without complex dependencies.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.graph.movement_classifier import MovementClassifier, MovementType, NinjaState
from nclone.graph.level_data import LevelData
from nclone.constants.physics_constants import TILE_PIXEL_SIZE

def create_simple_walk_level():
    """Create a simple horizontal walking level for testing."""
    # Create a 9x5 tile array (simple horizontal platform)
    tiles = np.zeros((5, 9), dtype=int)
    
    # Add ground platform on row 3 (0-indexed)
    tiles[3, :] = 1  # Full solid tiles for ground
    
    # Create simple entities list
    entities = [
        {"type": 0, "x": 24, "y": 60},    # Ninja at left side
        {"type": 3, "x": 192, "y": 60}   # Exit at right side
    ]
    
    return LevelData(tiles, entities)

def test_movement_classification():
    """Test movement classification on simple scenarios."""
    print("=== Testing Movement Classification ===")
    
    classifier = MovementClassifier()
    level_data = create_simple_walk_level()
    
    # Test horizontal walking
    src_pos = (24, 60)   # Left side
    tgt_pos = (192, 60)  # Right side
    
    movement_type, physics_params = classifier.classify_movement(
        src_pos, tgt_pos, None, level_data
    )
    
    print(f"Horizontal movement classification:")
    print(f"  Source: {src_pos}")
    print(f"  Target: {tgt_pos}")
    print(f"  Movement Type: {movement_type} ({MovementType(movement_type).name})")
    print(f"  Distance: {physics_params['distance']:.1f}")
    print(f"  Height Diff: {physics_params['height_diff']:.1f}")
    print(f"  Energy Cost: {physics_params['energy_cost']:.2f}")
    print(f"  Time Estimate: {physics_params['time_estimate']:.2f}")
    
    # Test upward movement (should require jumping)
    src_pos = (96, 84)   # Ground level
    tgt_pos = (96, 36)   # One tile up
    
    movement_type, physics_params = classifier.classify_movement(
        src_pos, tgt_pos, None, level_data
    )
    
    print(f"\nUpward movement classification:")
    print(f"  Source: {src_pos}")
    print(f"  Target: {tgt_pos}")
    print(f"  Movement Type: {movement_type} ({MovementType(movement_type).name})")
    print(f"  Distance: {physics_params['distance']:.1f}")
    print(f"  Height Diff: {physics_params['height_diff']:.1f}")
    print(f"  Energy Cost: {physics_params['energy_cost']:.2f}")
    print(f"  Time Estimate: {physics_params['time_estimate']:.2f}")
    
    # Test downward movement (should be falling)
    src_pos = (96, 36)   # One tile up
    tgt_pos = (96, 84)   # Ground level
    
    movement_type, physics_params = classifier.classify_movement(
        src_pos, tgt_pos, None, level_data
    )
    
    print(f"\nDownward movement classification:")
    print(f"  Source: {src_pos}")
    print(f"  Target: {tgt_pos}")
    print(f"  Movement Type: {movement_type} ({MovementType(movement_type).name})")
    print(f"  Distance: {physics_params['distance']:.1f}")
    print(f"  Height Diff: {physics_params['height_diff']:.1f}")
    print(f"  Energy Cost: {physics_params['energy_cost']:.2f}")
    print(f"  Time Estimate: {physics_params['time_estimate']:.2f}")

def test_ninja_states():
    """Test ninja state transitions."""
    print("\n=== Testing Ninja States ===")
    
    # Test different ninja states
    states = [
        (0, "Immobile"),
        (1, "Running"),
        (2, "Ground Sliding"),
        (3, "Jumping"),
        (4, "Falling"),
        (5, "Wall Sliding")
    ]
    
    for state_id, state_name in states:
        ninja_state = NinjaState(
            movement_state=state_id,
            velocity=(1.0, -0.5) if state_id in [3, 4] else (2.0, 0.0),
            position=(100, 100),
            ground_contact=state_id in [0, 1, 2],
            wall_contact=state_id == 5
        )
        
        print(f"State {state_id} ({state_name}):")
        print(f"  Velocity: {ninja_state.velocity}")
        print(f"  Ground Contact: {ninja_state.ground_contact}")
        print(f"  Wall Contact: {ninja_state.wall_contact}")

if __name__ == "__main__":
    test_movement_classification()
    test_ninja_states()