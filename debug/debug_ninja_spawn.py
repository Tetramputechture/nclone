#!/usr/bin/env python3
"""
Debug ninja spawn position in different maps to understand if this is a doortest-specific issue.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold


def debug_ninja_spawn():
    """Debug ninja spawn position."""
    print("=" * 80)
    print("DEBUGGING NINJA SPAWN POSITION")
    print("=" * 80)
    
    # Test with different seeds to see if ninja spawn varies
    for seed in [42, 0, 1, 2, 3]:
        print(f"\n--- Testing with seed {seed} ---")
        
        # Create environment
        env = BasicLevelNoGold(
            render_mode="rgb_array",
            enable_frame_stack=False,
            enable_debug_overlay=False,
            eval_mode=False,
            seed=seed
        )
        
        # Reset to load the map
        env.reset()
        
        # Get level data and ninja position
        level_data = env.level_data
        ninja_pos = env.nplay_headless.ninja_position()
        
        print(f"Ninja position: {ninja_pos}")
        
        # Calculate tile coordinates
        ninja_tile_x = int(ninja_pos[0] // 24)
        ninja_tile_y = int(ninja_pos[1] // 24)
        
        print(f"Ninja tile: ({ninja_tile_x}, {ninja_tile_y})")
        
        # Check tile value
        if (0 <= ninja_tile_y < level_data.height and 0 <= ninja_tile_x < level_data.width):
            tile_value = level_data.get_tile(ninja_tile_y, ninja_tile_x)
            print(f"Tile value: {tile_value} ({'empty' if tile_value == 0 else 'solid' if tile_value == 1 else 'other'})")
        
        # Check if there are any empty tiles nearby
        empty_tiles_nearby = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                check_x = ninja_tile_x + dx
                check_y = ninja_tile_y + dy
                
                if (0 <= check_y < level_data.height and 0 <= check_x < level_data.width):
                    tile_value = level_data.get_tile(check_y, check_x)
                    if tile_value == 0:
                        distance = (dx*dx + dy*dy)**0.5
                        empty_tiles_nearby.append((check_x, check_y, distance))
        
        empty_tiles_nearby.sort(key=lambda x: x[2])  # Sort by distance
        
        if empty_tiles_nearby:
            print(f"Nearest empty tiles: {empty_tiles_nearby[:3]}")
        else:
            print("No empty tiles nearby!")
        
        env.close()
    
    # Also test if we can manually check the doortest map structure
    print("\n" + "=" * 60)
    print("ANALYZING DOORTEST MAP STRUCTURE")
    print("=" * 60)
    
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42
    )
    
    env.reset()
    level_data = env.level_data
    
    print(f"Map size: {level_data.width}x{level_data.height}")
    
    # Count tile types
    tile_counts = {}
    for y in range(level_data.height):
        for x in range(level_data.width):
            tile_value = level_data.get_tile(y, x)
            tile_counts[tile_value] = tile_counts.get(tile_value, 0) + 1
    
    print("Tile type distribution:")
    for tile_value, count in sorted(tile_counts.items()):
        percentage = count / (level_data.width * level_data.height) * 100
        tile_type = 'empty' if tile_value == 0 else 'solid' if tile_value == 1 else f'type_{tile_value}'
        print(f"  {tile_type} ({tile_value}): {count} tiles ({percentage:.1f}%)")
    
    # Find all empty tiles
    empty_tiles = []
    for y in range(level_data.height):
        for x in range(level_data.width):
            if level_data.get_tile(y, x) == 0:
                empty_tiles.append((x, y))
    
    print(f"\nTotal empty tiles: {len(empty_tiles)}")
    if empty_tiles:
        print("First 10 empty tiles:")
        for i, (x, y) in enumerate(empty_tiles[:10]):
            pixel_x = x * 24 + 12  # Center of tile
            pixel_y = y * 24 + 12
            print(f"  {i+1}. Tile ({x}, {y}) -> pixel ({pixel_x}, {pixel_y})")
    
    env.close()


if __name__ == '__main__':
    debug_ninja_spawn()