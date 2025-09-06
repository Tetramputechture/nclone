#!/usr/bin/env python3
"""
Test the exact coordinates that failed in the traversability test.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.precise_collision import PreciseTileCollision
from nclone.constants.physics_constants import NINJA_RADIUS


def debug_exact_coordinates():
    """Test exact coordinates that failed."""
    print("=" * 80)
    print("TESTING EXACT FAILING COORDINATES")
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
    
    # Create collision detector
    collision = PreciseTileCollision()
    
    # Test the exact failing coordinates
    test_cases = [
        # From the traversability test that failed
        (489, 177, 495, 177),  # (29, 81) -> (29, 82)
        (489, 177, 489, 183),  # (29, 81) -> (30, 81)
        (489, 177, 495, 183),  # (29, 81) -> (30, 82)
        
        # Some successful cases from the simple test
        (492, 180, 495, 180),  # Center to 3 pixels right
        (492, 180, 498, 180),  # Center to 6 pixels right
    ]
    
    for i, (src_x, src_y, tgt_x, tgt_y) in enumerate(test_cases):
        print(f"\nTest {i+1}: ({src_x}, {src_y}) -> ({tgt_x}, {tgt_y})")
        
        # Calculate distance
        distance = ((tgt_x - src_x)**2 + (tgt_y - src_y)**2)**0.5
        print(f"  Distance: {distance:.1f} pixels")
        
        # Check what tiles these coordinates are in
        tile_x_src = src_x // 24
        tile_y_src = src_y // 24
        tile_x_tgt = tgt_x // 24
        tile_y_tgt = tgt_y // 24
        
        tile_val_src = level_data.get_tile(tile_y_src, tile_x_src) if 0 <= tile_x_src < level_data.width and 0 <= tile_y_src < level_data.height else -1
        tile_val_tgt = level_data.get_tile(tile_y_tgt, tile_x_tgt) if 0 <= tile_x_tgt < level_data.width and 0 <= tile_y_tgt < level_data.height else -1
        
        print(f"  Source tile ({tile_x_src}, {tile_y_src}): {tile_val_src}")
        print(f"  Target tile ({tile_x_tgt}, {tile_y_tgt}): {tile_val_tgt}")
        
        # Test collision
        result = collision.is_path_traversable(src_x, src_y, tgt_x, tgt_y, level_data.tiles, NINJA_RADIUS)
        print(f"  Collision result: {'✅ PASS' if result else '❌ FAIL'}")
        
        # If it failed, test with smaller radius to see if it's a radius issue
        if not result:
            for radius in [0, 5, 8]:
                small_result = collision.is_path_traversable(src_x, src_y, tgt_x, tgt_y, level_data.tiles, radius)
                print(f"    With radius {radius}: {'✅ PASS' if small_result else '❌ FAIL'}")


if __name__ == '__main__':
    debug_exact_coordinates()