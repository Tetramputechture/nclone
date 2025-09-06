#!/usr/bin/env python3
"""
Debug the ninja's tile position and why it has no edges.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold


def debug_ninja_tile():
    """Debug the ninja's tile position."""
    print("=" * 80)
    print("DEBUGGING NINJA TILE POSITION")
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
    
    # Calculate tile coordinates
    ninja_tile_x = int(ninja_pos[0] // 24)
    ninja_tile_y = int(ninja_pos[1] // 24)
    
    print(f"Ninja tile coordinates: ({ninja_tile_x}, {ninja_tile_y})")
    
    # Check tile value
    if (0 <= ninja_tile_y < level_data.height and 0 <= ninja_tile_x < level_data.width):
        tile_value = level_data.get_tile(ninja_tile_y, ninja_tile_x)
        print(f"Ninja tile value: {tile_value}")
        
        if tile_value == 0:
            print("✅ Ninja is in an empty tile (should be traversable)")
        elif tile_value == 1:
            print("❌ Ninja is in a solid tile (not traversable)")
        else:
            print(f"⚠️ Ninja is in a tile with value {tile_value}")
    else:
        print("❌ Ninja tile coordinates are out of bounds")
    
    # Check surrounding tiles
    print("\nSurrounding tiles (3x3 grid around ninja):")
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            check_x = ninja_tile_x + dx
            check_y = ninja_tile_y + dy
            
            if (0 <= check_y < level_data.height and 0 <= check_x < level_data.width):
                tile_value = level_data.get_tile(check_y, check_x)
                marker = "N" if (dx == 0 and dy == 0) else str(tile_value)
                print(f"  ({check_x:2d}, {check_y:2d}): {marker}", end="")
            else:
                print(f"  ({check_x:2d}, {check_y:2d}): X", end="")
        print()
    
    # Check if ninja is in entities list
    print(f"\nTotal entities: {len(level_data.entities)}")
    ninja_entity = None
    for i, entity in enumerate(level_data.entities):
        if entity.get('type') == 0:  # NINJA type
            ninja_entity = entity
            print(f"Found ninja entity at index {i}: {entity}")
            break
    
    if ninja_entity is None:
        print("❌ No ninja entity found in entities list")
        print("This might be why the ninja node has no edges!")
    else:
        print("✅ Ninja entity found")
        
        # Check if ninja entity position matches ninja position
        entity_x = ninja_entity.get('x', 0)
        entity_y = ninja_entity.get('y', 0)
        print(f"Ninja entity position: ({entity_x}, {entity_y})")
        
        distance = ((entity_x - ninja_pos[0])**2 + (entity_y - ninja_pos[1])**2)**0.5
        print(f"Distance between ninja position and entity position: {distance:.1f}")
        
        if distance > 1:
            print("⚠️ Ninja entity position doesn't match ninja position")


if __name__ == '__main__':
    debug_ninja_tile()