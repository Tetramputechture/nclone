#!/usr/bin/env python3
"""
Debug tiles around ninja position to understand connectivity issues.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold


def debug_nearby_tiles():
    """Debug tiles around ninja position."""
    print("=" * 80)
    print("DEBUGGING TILES AROUND NINJA")
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
    
    # Get level data and ninja position
    level_data = env.level_data
    ninja_pos = env.nplay_headless.ninja_position()
    
    print(f"Ninja position: {ninja_pos}")
    
    # Calculate ninja's tile position
    ninja_tile_x = int(ninja_pos[0] // 24)
    ninja_tile_y = int(ninja_pos[1] // 24)
    
    print(f"Ninja tile: ({ninja_tile_x}, {ninja_tile_y})")
    
    if 0 <= ninja_tile_x < level_data.width and 0 <= ninja_tile_y < level_data.height:
        ninja_tile_value = level_data.get_tile(ninja_tile_y, ninja_tile_x)
        print(f"Ninja tile value: {ninja_tile_value} ({'empty' if ninja_tile_value == 0 else 'solid'})")
    
    # Check surrounding tiles in a larger area
    print(f"\nTiles around ninja (7x7 area):")
    print("Legend: 0=empty, 1=solid, ?=out of bounds")
    
    for dy in range(-3, 4):
        row_str = ""
        for dx in range(-3, 4):
            check_x = ninja_tile_x + dx
            check_y = ninja_tile_y + dy
            
            if 0 <= check_x < level_data.width and 0 <= check_y < level_data.height:
                tile_value = level_data.get_tile(check_y, check_x)
                if dx == 0 and dy == 0:
                    row_str += f"[{tile_value}]"  # Mark ninja position
                else:
                    row_str += f" {tile_value} "
            else:
                row_str += " ? "
        
        print(f"  {row_str}")
    
    # Find nearest empty tiles
    print(f"\nFinding nearest empty tiles:")
    
    empty_tiles = []
    max_distance = 10  # Check within 10 tiles
    
    for dy in range(-max_distance, max_distance + 1):
        for dx in range(-max_distance, max_distance + 1):
            check_x = ninja_tile_x + dx
            check_y = ninja_tile_y + dy
            
            if 0 <= check_x < level_data.width and 0 <= check_y < level_data.height:
                tile_value = level_data.get_tile(check_y, check_x)
                if tile_value == 0:  # Empty tile
                    distance = (dx**2 + dy**2)**0.5
                    empty_tiles.append((check_x, check_y, distance))
    
    # Sort by distance
    empty_tiles.sort(key=lambda x: x[2])
    
    print(f"Found {len(empty_tiles)} empty tiles within {max_distance} tile radius")
    print(f"Nearest empty tiles:")
    
    for i, (tile_x, tile_y, distance) in enumerate(empty_tiles[:10]):
        pixel_x = tile_x * 24 + 12
        pixel_y = tile_y * 24 + 12
        pixel_distance = ((ninja_pos[0] - pixel_x)**2 + (ninja_pos[1] - pixel_y)**2)**0.5
        
        print(f"  {i+1}. Tile ({tile_x}, {tile_y}) at distance {distance:.1f} tiles")
        print(f"     Pixel center: ({pixel_x}, {pixel_y}) at distance {pixel_distance:.1f} pixels")
    
    # Check if there are any empty tiles adjacent to ninja's tile
    adjacent_empty = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            
            check_x = ninja_tile_x + dx
            check_y = ninja_tile_y + dy
            
            if 0 <= check_x < level_data.width and 0 <= check_y < level_data.height:
                tile_value = level_data.get_tile(check_y, check_x)
                if tile_value == 0:
                    adjacent_empty.append((check_x, check_y))
    
    if adjacent_empty:
        print(f"\n✅ Found {len(adjacent_empty)} adjacent empty tiles:")
        for tile_x, tile_y in adjacent_empty:
            print(f"  Tile ({tile_x}, {tile_y})")
    else:
        print(f"\n❌ No adjacent empty tiles found")
        print(f"The ninja is surrounded by solid tiles, which explains the isolation.")


if __name__ == '__main__':
    debug_nearby_tiles()