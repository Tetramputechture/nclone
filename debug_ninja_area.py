#!/usr/bin/env python3
"""
Debug traversability around ninja's immediate area
"""

import os
import sys
import numpy as np

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.graph_construction import GraphConstructor
from nclone.graph.feature_extraction import FeatureExtractor
from nclone.graph.edge_building import EdgeBuilder

def debug_ninja_area():
    print("=" * 60)
    print("NINJA AREA TRAVERSABILITY DEBUG")
    print("=" * 60)
    
    # Create environment
    env = BasicLevelNoGold(render_mode="rgb_array", enable_frame_stack=False, enable_debug_overlay=False, eval_mode=False, seed=42)
    env.reset()
    ninja_position = env.nplay_headless.ninja_position()
    level_data = env.level_data
    
    print(f"Ninja position: {ninja_position}")
    
    # Create graph constructor
    feature_extractor = FeatureExtractor()
    edge_builder = EdgeBuilder(feature_extractor)
    constructor = GraphConstructor(feature_extractor, edge_builder)
    
    # Test a 5x5 grid around the ninja (every 6 pixels = 1 sub-cell)
    ninja_x, ninja_y = ninja_position
    
    print(f"\nTesting 9x9 grid around ninja (step=6 pixels):")
    print("Legend: ✅ = traversable, ❌ = non-traversable, 🥷 = ninja")
    
    for dy in range(-24, 25, 6):  # -4 to +4 sub-cells
        row = ""
        for dx in range(-24, 25, 6):  # -4 to +4 sub-cells
            test_x = ninja_x + dx
            test_y = ninja_y + dy
            
            if dx == 0 and dy == 0:
                row += "🥷"
            else:
                is_traversable = constructor._is_position_traversable(test_x, test_y, level_data.tiles)
                row += "✅" if is_traversable else "❌"
        print(f"y={ninja_y + dy:3.0f}: {row}")
    
    print(f"\nDetailed analysis of ninja's exact position:")
    ninja_traversable = constructor._is_position_traversable(ninja_x, ninja_y, level_data.tiles)
    print(f"Ninja at ({ninja_x}, {ninja_y}): {'✅ traversable' if ninja_traversable else '❌ non-traversable'}")
    
    # Check tile information
    tile_x = int(ninja_x // 24)  # TILE_PIXEL_SIZE = 24
    tile_y = int(ninja_y // 24)
    
    if 0 <= tile_y < level_data.height and 0 <= tile_x < level_data.width:
        tile_value = level_data.tiles[tile_y, tile_x]
        print(f"Ninja is in tile ({tile_x}, {tile_y}) with value {tile_value}")
        
        # Check local position within tile
        local_x = ninja_x % 24
        local_y = ninja_y % 24
        print(f"Local position within tile: ({local_x:.1f}, {local_y:.1f})")
    else:
        print(f"❌ Ninja is outside map bounds!")
    
    # Test some specific nearby positions
    test_positions = [
        (ninja_x - 6, ninja_y),      # Left
        (ninja_x + 6, ninja_y),      # Right  
        (ninja_x, ninja_y - 6),      # Up
        (ninja_x, ninja_y + 6),      # Down
        (ninja_x - 6, ninja_y - 6),  # Up-left
        (ninja_x + 6, ninja_y - 6),  # Up-right
        (ninja_x - 6, ninja_y + 6),  # Down-left
        (ninja_x + 6, ninja_y + 6),  # Down-right
    ]
    
    directions = ["Left", "Right", "Up", "Down", "Up-left", "Up-right", "Down-left", "Down-right"]
    
    print(f"\nTesting 8 adjacent positions:")
    for i, (test_x, test_y) in enumerate(test_positions):
        is_traversable = constructor._is_position_traversable(test_x, test_y, level_data.tiles)
        status = "✅ traversable" if is_traversable else "❌ non-traversable"
        
        # Check tile info for this position
        tile_x = int(test_x // 24)
        tile_y = int(test_y // 24)
        if 0 <= tile_y < level_data.height and 0 <= tile_x < level_data.width:
            tile_value = level_data.tiles[tile_y, tile_x]
            print(f"  {directions[i]:10s} ({test_x:6.1f}, {test_y:6.1f}): {status} - tile ({tile_x}, {tile_y}) value {tile_value}")
        else:
            print(f"  {directions[i]:10s} ({test_x:6.1f}, {test_y:6.1f}): {status} - outside bounds")

if __name__ == "__main__":
    debug_ninja_area()
    print("\n🎉 NINJA AREA DEBUG COMPLETE!")