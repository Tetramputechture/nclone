#!/usr/bin/env python3
"""
Analyze the coordinate system to understand the 1-tile border padding.
"""

import sys
import os

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold

def analyze_coordinate_system():
    """Analyze the coordinate system and tile mapping."""
    print("=" * 70)
    print("🗺️  COORDINATE SYSTEM ANALYSIS")
    print("=" * 70)
    
    # Load environment
    env = BasicLevelNoGold(render_mode="rgb_array")
    env.reset()
    
    ninja_pos = env.nplay_headless.ninja_position()
    print(f"✅ Ninja position: {ninja_pos}")
    
    # Analyze level data structure
    print(f"\n📐 LEVEL DATA STRUCTURE:")
    print(f"   Level data width: {env.level_data.width}")
    print(f"   Level data height: {env.level_data.height}")
    print(f"   Pixel dimensions: {env.level_data.width*24}x{env.level_data.height*24}")
    
    # Check if there's a border by examining edge tiles
    print(f"\n🔍 BORDER ANALYSIS:")
    
    # Check top row
    top_row_tiles = []
    for x in range(min(10, env.level_data.width)):
        tile_value = env.level_data.get_tile(0, x)
        top_row_tiles.append(tile_value)
    print(f"   Top row (first 10): {top_row_tiles}")
    
    # Check bottom row
    bottom_row_tiles = []
    for x in range(min(10, env.level_data.width)):
        tile_value = env.level_data.get_tile(env.level_data.height-1, x)
        bottom_row_tiles.append(tile_value)
    print(f"   Bottom row (first 10): {bottom_row_tiles}")
    
    # Check left column
    left_col_tiles = []
    for y in range(min(10, env.level_data.height)):
        tile_value = env.level_data.get_tile(y, 0)
        left_col_tiles.append(tile_value)
    print(f"   Left column (first 10): {left_col_tiles}")
    
    # Check right column
    right_col_tiles = []
    for y in range(min(10, env.level_data.height)):
        tile_value = env.level_data.get_tile(y, env.level_data.width-1)
        right_col_tiles.append(tile_value)
    print(f"   Right column (first 10): {right_col_tiles}")
    
    # Analyze ninja position with and without border offset
    print(f"\n🥷 NINJA POSITION ANALYSIS:")
    
    # Direct tile calculation
    direct_tile_x = int(ninja_pos[0] // 24)
    direct_tile_y = int(ninja_pos[1] // 24)
    print(f"   Direct tile calculation: ({direct_tile_x}, {direct_tile_y})")
    
    if (0 <= direct_tile_x < env.level_data.width and 
        0 <= direct_tile_y < env.level_data.height):
        direct_tile_value = env.level_data.get_tile(direct_tile_y, direct_tile_x)
        print(f"   Direct tile value: {direct_tile_value}")
    else:
        print(f"   Direct tile: OUT OF BOUNDS")
    
    # With 1-tile border offset
    border_tile_x = int((ninja_pos[0] - 24) // 24)
    border_tile_y = int((ninja_pos[1] - 24) // 24)
    print(f"   With border offset: ({border_tile_x}, {border_tile_y})")
    
    if (0 <= border_tile_x < env.level_data.width and 
        0 <= border_tile_y < env.level_data.height):
        border_tile_value = env.level_data.get_tile(border_tile_y, border_tile_x)
        print(f"   Border offset tile value: {border_tile_value}")
    else:
        print(f"   Border offset tile: OUT OF BOUNDS")
    
    # Check surrounding tiles around ninja position
    print(f"\n🔍 SURROUNDING TILES ANALYSIS:")
    
    for offset_name, (offset_x, offset_y) in [
        ("No offset", (0, 0)),
        ("1-tile border", (-24, -24)),
        ("Half-tile offset", (-12, -12))
    ]:
        adj_x = ninja_pos[0] + offset_x
        adj_y = ninja_pos[1] + offset_y
        tile_x = int(adj_x // 24)
        tile_y = int(adj_y // 24)
        
        print(f"   {offset_name}: pixel ({adj_x:.1f}, {adj_y:.1f}) -> tile ({tile_x}, {tile_y})")
        
        if (0 <= tile_x < env.level_data.width and 
            0 <= tile_y < env.level_data.height):
            tile_value = env.level_data.get_tile(tile_y, tile_x)
            print(f"      Tile value: {tile_value}")
            
            # Check 3x3 area around this position
            print(f"      3x3 area around tile:")
            for dy in [-1, 0, 1]:
                row = []
                for dx in [-1, 0, 1]:
                    check_x = tile_x + dx
                    check_y = tile_y + dy
                    if (0 <= check_x < env.level_data.width and 
                        0 <= check_y < env.level_data.height):
                        val = env.level_data.get_tile(check_y, check_x)
                        row.append(f"{val:2d}")
                    else:
                        row.append("XX")
                print(f"         {' '.join(row)}")
        else:
            print(f"      OUT OF BOUNDS")
        print()

if __name__ == "__main__":
    analyze_coordinate_system()