#!/usr/bin/env python3
"""
Debug level tile structure
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold

def debug_level_tiles():
    print("=" * 60)
    print("LEVEL TILE STRUCTURE DEBUG")
    print("=" * 60)
    
    # Create environment
    env = BasicLevelNoGold(render_mode="rgb_array", enable_frame_stack=False, enable_debug_overlay=False, eval_mode=False, seed=42)
    env.reset()
    ninja_position = env.nplay_headless.ninja_position()
    level_data = env.level_data
    
    print(f"Ninja position: {ninja_position}")
    print(f"Level size: {level_data.width}x{level_data.height} tiles")
    
    # Analyze tile distribution
    tiles = level_data.tiles
    unique_values, counts = np.unique(tiles, return_counts=True)
    
    print(f"\nTile value distribution:")
    for value, count in zip(unique_values, counts):
        percentage = (count / tiles.size) * 100
        print(f"  Tile {value:2d}: {count:4d} tiles ({percentage:5.1f}%)")
    
    # Find ninja tile
    ninja_tile_x = int(ninja_position[0] // 24)
    ninja_tile_y = int(ninja_position[1] // 24)
    ninja_tile_value = tiles[ninja_tile_y, ninja_tile_x]
    
    print(f"\nNinja is in tile ({ninja_tile_x}, {ninja_tile_y}) with value {ninja_tile_value}")
    
    # Show a region around the ninja
    print(f"\nTile map around ninja (5x5 region):")
    for dy in range(-2, 3):
        row = ""
        for dx in range(-2, 3):
            tile_x = ninja_tile_x + dx
            tile_y = ninja_tile_y + dy
            
            if 0 <= tile_y < level_data.height and 0 <= tile_x < level_data.width:
                tile_value = tiles[tile_y, tile_x]
                if dx == 0 and dy == 0:
                    row += f"[{tile_value:2d}]"
                else:
                    row += f" {tile_value:2d} "
            else:
                row += " -- "
        print(f"  {row}")
    
    # Create visualization of the entire level
    plt.figure(figsize=(15, 8))
    
    # Create a color map for different tile types
    # Use a discrete colormap to clearly distinguish tile types
    plt.imshow(tiles, cmap='tab20', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Tile Value')
    
    # Mark ninja position
    plt.scatter([ninja_tile_x], [ninja_tile_y], c='red', s=100, marker='*', label='Ninja', edgecolors='white', linewidth=2)
    
    # Mark entities
    for entity in level_data.entities:
        entity_x = entity.get("x", 0) // 24
        entity_y = entity.get("y", 0) // 24
        entity_type = entity.get("type", -1)
        
        if entity_type == 6:  # LOCKED_DOOR switches
            plt.scatter([entity_x], [entity_y], c='purple', s=50, marker='s', alpha=0.8)
    
    plt.xlabel('Tile X')
    plt.ylabel('Tile Y')
    plt.title('Level Tile Map')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/nclone/level_tiles_debug.png', dpi=150, bbox_inches='tight')
    print("📊 Saved level tile visualization to level_tiles_debug.png")
    
    # Check for potential paths
    print(f"\nLooking for potential paths from ninja to leftmost switch...")
    
    # Find leftmost switch
    leftmost_switch = None
    leftmost_x = float('inf')
    
    for entity in level_data.entities:
        if entity.get("type") == 6:  # LOCKED_DOOR
            switch_x = entity.get("x", 0)
            if switch_x < leftmost_x:
                leftmost_x = switch_x
                leftmost_switch = entity
    
    if leftmost_switch:
        switch_x = leftmost_switch.get("x", 0)
        switch_y = leftmost_switch.get("y", 0)
        switch_tile_x = int(switch_x // 24)
        switch_tile_y = int(switch_y // 24)
        switch_tile_value = tiles[switch_tile_y, switch_tile_x]
        
        print(f"Leftmost switch at ({switch_x}, {switch_y}) -> tile ({switch_tile_x}, {switch_tile_y}) value {switch_tile_value}")
        
        # Check tiles along a straight line path
        print(f"\nTile values along direct path:")
        num_steps = 20
        for i in range(num_steps + 1):
            t = i / num_steps
            path_x = ninja_position[0] + t * (switch_x - ninja_position[0])
            path_y = ninja_position[1] + t * (switch_y - ninja_position[1])
            
            path_tile_x = int(path_x // 24)
            path_tile_y = int(path_y // 24)
            
            if 0 <= path_tile_y < level_data.height and 0 <= path_tile_x < level_data.width:
                tile_value = tiles[path_tile_y, path_tile_x]
                print(f"  Step {i:2d}: ({path_x:6.1f}, {path_y:6.1f}) -> tile ({path_tile_x:2d}, {path_tile_y:2d}) value {tile_value}")
            else:
                print(f"  Step {i:2d}: ({path_x:6.1f}, {path_y:6.1f}) -> outside bounds")

if __name__ == "__main__":
    debug_level_tiles()
    print("\n🎉 LEVEL TILE DEBUG COMPLETE!")