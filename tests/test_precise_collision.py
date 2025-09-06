#!/usr/bin/env python3
"""
Test script to validate the PreciseTileCollision class against known collision scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from nclone.graph.precise_collision import PreciseTileCollision
from nclone.constants.physics_constants import NINJA_RADIUS, TILE_PIXEL_SIZE


def test_basic_collision():
    """Test basic collision detection scenarios."""
    collision_detector = PreciseTileCollision()
    
    # Test case 1: Empty level - should be traversable
    empty_level = {
        'level_id': 'test_empty',
        'tiles': {}
    }
    
    result = collision_detector.is_path_traversable(
        src_x=50.0, src_y=50.0,
        tgt_x=100.0, tgt_y=50.0,
        level_data=empty_level
    )
    print(f"Empty level traversal: {result} (expected: True)")
    assert result == True, "Empty level should be traversable"
    
    # Test case 2: Single solid tile blocking path
    blocked_level = {
        'level_id': 'test_blocked',
        'tiles': {
            (3, 2): 1  # Solid tile at position (3, 2) - coordinates 72,48 to 96,72
        }
    }
    
    # Path that goes through the solid tile
    result = collision_detector.is_path_traversable(
        src_x=60.0, src_y=60.0,  # Before the tile
        tgt_x=90.0, tgt_y=60.0,  # Through the tile
        level_data=blocked_level
    )
    print(f"Blocked path traversal: {result} (expected: False)")
    assert result == False, "Path through solid tile should be blocked"
    
    # Test case 3: Path that goes around the tile
    result = collision_detector.is_path_traversable(
        src_x=60.0, src_y=30.0,  # Above the tile
        tgt_x=90.0, tgt_y=30.0,  # Still above the tile
        level_data=blocked_level
    )
    print(f"Clear path traversal: {result} (expected: True)")
    assert result == True, "Path around solid tile should be clear"
    
    print("✓ All basic collision tests passed!")


def test_ninja_radius_collision():
    """Test that ninja radius is properly considered in collision detection."""
    collision_detector = PreciseTileCollision()
    
    # Single solid tile
    level_data = {
        'level_id': 'test_radius',
        'tiles': {
            (4, 4): 1  # Solid tile at 96,96 to 120,120
        }
    }
    
    # Test path that would be clear for a point but blocked for ninja radius
    tile_edge_x = 4 * TILE_PIXEL_SIZE  # 96
    tile_edge_y = 4 * TILE_PIXEL_SIZE  # 96
    
    # Path just outside ninja radius - should be clear
    result = collision_detector.is_path_traversable(
        src_x=tile_edge_x - NINJA_RADIUS - 1,  # Just outside collision range
        src_y=tile_edge_y - 10,
        tgt_x=tile_edge_x - NINJA_RADIUS - 1,
        tgt_y=tile_edge_y + TILE_PIXEL_SIZE + 10,
        level_data=level_data
    )
    print(f"Path outside ninja radius: {result} (expected: True)")
    assert result == True, "Path outside ninja radius should be clear"
    
    # Path within ninja radius - should be blocked
    result = collision_detector.is_path_traversable(
        src_x=tile_edge_x - NINJA_RADIUS + 2,  # Within collision range (but not overlapping at start)
        src_y=tile_edge_y - 10,
        tgt_x=tile_edge_x - NINJA_RADIUS + 2,
        tgt_y=tile_edge_y + TILE_PIXEL_SIZE + 10,
        level_data=level_data
    )
    print(f"Path within ninja radius: {result} (expected: False)")
    assert result == False, "Path within ninja radius should be blocked"
    
    print("✓ Ninja radius collision tests passed!")


def test_different_tile_formats():
    """Test that different tile data formats work correctly."""
    collision_detector = PreciseTileCollision()
    
    # Dictionary format
    dict_level = {
        'level_id': 'test_dict',
        'tiles': {(2, 2): 1}
    }
    
    # List format
    list_level = {
        'level_id': 'test_list',
        'tiles': [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],  # Solid tile at (2, 2)
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
    }
    
    # Test same collision scenario with both formats
    test_path = {
        'src_x': 40.0, 'src_y': 60.0,
        'tgt_x': 80.0, 'tgt_y': 60.0
    }
    
    dict_result = collision_detector.is_path_traversable(
        level_data=dict_level, **test_path
    )
    
    list_result = collision_detector.is_path_traversable(
        level_data=list_level, **test_path
    )
    
    print(f"Dict format result: {dict_result}")
    print(f"List format result: {list_result}")
    assert dict_result == list_result, "Different tile formats should give same results"
    
    print("✓ Different tile format tests passed!")


def main():
    """Run all collision detection tests."""
    print("Testing PreciseTileCollision class...")
    print(f"Using NINJA_RADIUS: {NINJA_RADIUS}")
    print(f"Using TILE_PIXEL_SIZE: {TILE_PIXEL_SIZE}")
    print()
    
    try:
        test_basic_collision()
        print()
        test_ninja_radius_collision()
        print()
        test_different_tile_formats()
        print()
        print("🎉 All tests passed! PreciseTileCollision is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())