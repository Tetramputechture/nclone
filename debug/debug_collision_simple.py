#!/usr/bin/env python3
"""
Simple collision test to understand the issue.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.precise_collision import PreciseTileCollision
from nclone.constants.physics_constants import NINJA_RADIUS


def debug_collision_simple():
    """Simple collision test."""
    print("=" * 80)
    print("SIMPLE COLLISION TEST")
    print("=" * 80)
    
    # Create environment
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42
    )
    
    # Reset to load the map
    env.reset()
    
    # Get level data
    level_data = env.level_data
    
    print(f"Level size: {level_data.width}x{level_data.height} tiles")
    print(f"Ninja radius: {NINJA_RADIUS} pixels")
    
    # Create collision detector
    collision = PreciseTileCollision()
    
    # Find an empty tile
    empty_tile = None
    for tile_y in range(level_data.height):
        for tile_x in range(level_data.width):
            if level_data.get_tile(tile_y, tile_x) == 0:  # Empty tile
                empty_tile = (tile_x, tile_y)
                break
        if empty_tile:
            break
    
    if not empty_tile:
        print("No empty tiles found!")
        return
    
    tile_x, tile_y = empty_tile
    print(f"Testing empty tile at ({tile_x}, {tile_y})")
    
    # Check surrounding tiles
    print(f"Surrounding tiles:")
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            check_x = tile_x + dx
            check_y = tile_y + dy
            if 0 <= check_x < level_data.width and 0 <= check_y < level_data.height:
                tile_value = level_data.get_tile(check_y, check_x)
                print(f"  ({check_x}, {check_y}): {tile_value} ({'empty' if tile_value == 0 else 'solid'})")
    
    # Test movement within the empty tile
    print(f"\nTesting movement within empty tile:")
    
    # Tile pixel boundaries
    tile_left = tile_x * 24
    tile_right = (tile_x + 1) * 24
    tile_top = tile_y * 24
    tile_bottom = (tile_y + 1) * 24
    
    print(f"Tile pixel bounds: ({tile_left}, {tile_top}) to ({tile_right}, {tile_bottom})")
    
    # Test center to center movement (should be safe)
    center_x = tile_left + 12
    center_y = tile_top + 12
    
    test_cases = [
        # Within tile movements
        (center_x, center_y, center_x + 3, center_y),      # 3 pixels right
        (center_x, center_y, center_x + 6, center_y),      # 6 pixels right (1 sub-cell)
        (center_x, center_y, center_x, center_y + 3),      # 3 pixels down
        (center_x, center_y, center_x, center_y + 6),      # 6 pixels down (1 sub-cell)
        (center_x, center_y, center_x + 3, center_y + 3),  # 3 pixels diagonal
        
        # Edge cases - near tile boundaries
        (center_x, center_y, tile_right - 11, center_y),   # Near right edge (1 pixel clearance)
        (center_x, center_y, tile_right - 10, center_y),   # At right edge (exactly ninja radius)
        (center_x, center_y, tile_right - 9, center_y),    # Beyond right edge
    ]
    
    for i, (src_x, src_y, tgt_x, tgt_y) in enumerate(test_cases):
        print(f"\nTest {i+1}: ({src_x}, {src_y}) -> ({tgt_x}, {tgt_y})")
        
        # Calculate distance
        distance = ((tgt_x - src_x)**2 + (tgt_y - src_y)**2)**0.5
        print(f"  Distance: {distance:.1f} pixels")
        
        # Test with different radii
        for radius in [0, 5, 10, 15]:
            result = collision.is_path_traversable(src_x, src_y, tgt_x, tgt_y, level_data.tiles, radius)
            print(f"  Radius {radius:2d}: {'✅ PASS' if result else '❌ FAIL'}")


if __name__ == '__main__':
    debug_collision_simple()