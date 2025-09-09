#!/usr/bin/env python3
"""
Test pathfinding system against the four validation test maps.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.graph.movement_classifier import MovementClassifier, MovementType
from nclone.graph.level_data import LevelData
from nclone.constants.physics_constants import TILE_PIXEL_SIZE

def create_simple_walk_map():
    """Create the simple-walk test map: 9 tiles wide, single horizontal platform."""
    # 9x5 tile array
    tiles = np.zeros((5, 9), dtype=int)
    
    # Ground platform on row 3
    tiles[3, :] = 1  # Full solid tiles
    
    # Entities: ninja at left, exit switch in middle, exit door at right
    entities = [
        {"type": 0, "x": 24, "y": 60},     # Ninja at leftmost tile
        {"type": 4, "x": 120, "y": 60},    # Exit switch at middle tile (5th)
        {"type": 3, "x": 192, "y": 60}     # Exit door at rightmost tile
    ]
    
    return LevelData(tiles, entities)

def create_long_walk_map():
    """Create the long-walk test map: 42 tiles wide, single horizontal platform."""
    # 42x5 tile array
    tiles = np.zeros((5, 42), dtype=int)
    
    # Ground platform on row 3
    tiles[3, :] = 1  # Full solid tiles
    
    # Entities: ninja at left, exit switch near right, exit door at right
    entities = [
        {"type": 0, "x": 24, "y": 60},      # Ninja at leftmost tile
        {"type": 4, "x": 960, "y": 60},     # Exit switch at 41st tile
        {"type": 3, "x": 984, "y": 60}      # Exit door at rightmost tile
    ]
    
    return LevelData(tiles, entities)

def create_path_jump_required_map():
    """Create the path-jump-required test map: elevated switch position."""
    # 9x5 tile array
    tiles = np.zeros((5, 9), dtype=int)
    
    # Ground platform on row 3
    tiles[3, :] = 1  # Full solid tiles
    
    # Elevated platform for switch (single tile outcropping)
    tiles[2, 4] = 1  # Elevated tile at middle position
    
    # Entities: ninja at left, exit switch on elevated tile, exit door at right
    entities = [
        {"type": 0, "x": 24, "y": 60},     # Ninja at leftmost tile (ground level)
        {"type": 4, "x": 120, "y": 36},    # Exit switch on elevated tile
        {"type": 3, "x": 192, "y": 60}     # Exit door at rightmost tile (ground level)
    ]
    
    return LevelData(tiles, entities)

def create_only_jump_map():
    """Create the only-jump test map: vertical corridor for wall jumping."""
    # 5x9 tile array (rotated for vertical corridor)
    tiles = np.zeros((9, 5), dtype=int)
    
    # Vertical walls on left and right
    tiles[:, 0] = 1  # Left wall
    tiles[:, 4] = 1  # Right wall
    
    # Floor at bottom
    tiles[8, :] = 1  # Bottom floor
    
    # Entities: ninja at bottom, exit switch in middle, exit door at top
    entities = [
        {"type": 0, "x": 60, "y": 180},    # Ninja at bottom of corridor
        {"type": 4, "x": 60, "y": 108},    # Exit switch at middle height
        {"type": 3, "x": 60, "y": 36}      # Exit door at top of corridor
    ]
    
    return LevelData(tiles, entities)

def test_simple_walk():
    """Test pathfinding on simple-walk map."""
    print("=== Testing simple-walk map ===")
    
    level_data = create_simple_walk_map()
    classifier = MovementClassifier()
    
    # Test ninja to switch movement
    ninja_pos = (24, 60)
    switch_pos = (120, 60)
    
    movement_type, physics_params = classifier.classify_movement(
        ninja_pos, switch_pos, None, level_data
    )
    
    print(f"Ninja to Switch:")
    print(f"  Movement Type: {MovementType(movement_type).name}")
    print(f"  Distance: {physics_params['distance']:.1f} pixels")
    print(f"  Expected: WALK, ~96 pixels")
    
    # Test switch to exit movement
    switch_pos = (120, 60)
    exit_pos = (192, 60)
    
    movement_type, physics_params = classifier.classify_movement(
        switch_pos, exit_pos, None, level_data
    )
    
    print(f"Switch to Exit:")
    print(f"  Movement Type: {MovementType(movement_type).name}")
    print(f"  Distance: {physics_params['distance']:.1f} pixels")
    print(f"  Expected: WALK, ~72 pixels")
    
    # Total expected path: Single WALK segment covering ~192 pixels
    total_distance = physics_params['distance'] + 96  # Approximate
    print(f"Total path distance: ~{total_distance:.1f} pixels (Expected: ~192)")

def test_long_walk():
    """Test pathfinding on long-walk map."""
    print("\n=== Testing long-walk map ===")
    
    level_data = create_long_walk_map()
    classifier = MovementClassifier()
    
    # Test ninja to switch movement (full map traversal)
    ninja_pos = (24, 60)
    switch_pos = (960, 60)
    
    movement_type, physics_params = classifier.classify_movement(
        ninja_pos, switch_pos, None, level_data
    )
    
    print(f"Ninja to Switch:")
    print(f"  Movement Type: {MovementType(movement_type).name}")
    print(f"  Distance: {physics_params['distance']:.1f} pixels")
    print(f"  Expected: WALK, ~936 pixels")
    
    # Test switch to exit movement
    switch_pos = (960, 60)
    exit_pos = (984, 60)
    
    movement_type, physics_params = classifier.classify_movement(
        switch_pos, exit_pos, None, level_data
    )
    
    print(f"Switch to Exit:")
    print(f"  Movement Type: {MovementType(movement_type).name}")
    print(f"  Distance: {physics_params['distance']:.1f} pixels")
    print(f"  Expected: WALK, ~24 pixels")

def test_path_jump_required():
    """Test pathfinding on path-jump-required map."""
    print("\n=== Testing path-jump-required map ===")
    
    level_data = create_path_jump_required_map()
    classifier = MovementClassifier()
    
    # Test ninja to elevated switch (should require jumping)
    ninja_pos = (24, 60)
    switch_pos = (120, 36)
    
    movement_type, physics_params = classifier.classify_movement(
        ninja_pos, switch_pos, None, level_data
    )
    
    print(f"Ninja to Elevated Switch:")
    print(f"  Movement Type: {MovementType(movement_type).name}")
    print(f"  Distance: {physics_params['distance']:.1f} pixels")
    print(f"  Height Diff: {physics_params['height_diff']:.1f} pixels")
    print(f"  Expected: JUMP (upward movement to elevated platform)")
    
    # Test switch back to ground level (should be falling)
    switch_pos = (120, 36)
    exit_pos = (192, 60)
    
    movement_type, physics_params = classifier.classify_movement(
        switch_pos, exit_pos, None, level_data
    )
    
    print(f"Switch to Ground Exit:")
    print(f"  Movement Type: {MovementType(movement_type).name}")
    print(f"  Distance: {physics_params['distance']:.1f} pixels")
    print(f"  Height Diff: {physics_params['height_diff']:.1f} pixels")
    print(f"  Expected: FALL or JUMP (downward movement from elevated platform)")

def test_only_jump():
    """Test pathfinding on only-jump map."""
    print("\n=== Testing only-jump map ===")
    
    level_data = create_only_jump_map()
    classifier = MovementClassifier()
    
    # Test ninja to middle switch (vertical movement up)
    ninja_pos = (60, 180)
    switch_pos = (60, 108)
    
    movement_type, physics_params = classifier.classify_movement(
        ninja_pos, switch_pos, None, level_data
    )
    
    print(f"Ninja to Middle Switch:")
    print(f"  Movement Type: {MovementType(movement_type).name}")
    print(f"  Distance: {physics_params['distance']:.1f} pixels")
    print(f"  Height Diff: {physics_params['height_diff']:.1f} pixels")
    print(f"  Expected: JUMP (vertical ascent in corridor)")
    
    # Test switch to top exit (more vertical movement up)
    switch_pos = (60, 108)
    exit_pos = (60, 36)
    
    movement_type, physics_params = classifier.classify_movement(
        switch_pos, exit_pos, None, level_data
    )
    
    print(f"Switch to Top Exit:")
    print(f"  Movement Type: {MovementType(movement_type).name}")
    print(f"  Distance: {physics_params['distance']:.1f} pixels")
    print(f"  Height Diff: {physics_params['height_diff']:.1f} pixels")
    print(f"  Expected: JUMP (continued vertical ascent)")

def main():
    """Run all validation tests."""
    print("Physics-Aware Pathfinding Validation Tests")
    print("=" * 50)
    
    test_simple_walk()
    test_long_walk()
    test_path_jump_required()
    test_only_jump()
    
    print("\n" + "=" * 50)
    print("Validation Summary:")
    print("- simple-walk: Should show single WALK segments")
    print("- long-walk: Should show single WALK segment for full map")
    print("- path-jump-required: Should show JUMP up, FALL/JUMP down")
    print("- only-jump: Should show JUMP segments for vertical movement")

if __name__ == "__main__":
    main()