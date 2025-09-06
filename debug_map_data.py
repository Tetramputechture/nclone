#!/usr/bin/env python3
"""
Debug the map data to see what's at indices 1231 and 1232 (ninja spawn position).
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold


def debug_map_data():
    """Debug the map data ninja spawn position."""
    print("=" * 80)
    print("DEBUGGING MAP DATA NINJA SPAWN POSITION")
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
    
    # Get map data from simulator
    map_data = env.nplay_headless.sim.map_data
    
    print(f"Map data length: {len(map_data)}")
    
    # Check ninja spawn position indices
    ninja_x_raw = map_data[1231]
    ninja_y_raw = map_data[1232]
    
    print(f"Raw ninja position from map_data[1231:1233]: ({ninja_x_raw}, {ninja_y_raw})")
    print(f"Scaled ninja position (raw * 6): ({ninja_x_raw * 6}, {ninja_y_raw * 6})")
    
    # Compare with actual ninja position
    actual_ninja_pos = env.nplay_headless.ninja_position()
    print(f"Actual ninja position: {actual_ninja_pos}")
    
    # Check if they match
    expected_x = ninja_x_raw * 6
    expected_y = ninja_y_raw * 6
    
    if abs(expected_x - actual_ninja_pos[0]) < 0.1 and abs(expected_y - actual_ninja_pos[1]) < 0.1:
        print("✅ Map data ninja position matches actual ninja position")
    else:
        print("❌ Map data ninja position doesn't match actual ninja position")
    
    # Convert to tile coordinates
    ninja_tile_x = int(actual_ninja_pos[0] // 24)
    ninja_tile_y = int(actual_ninja_pos[1] // 24)
    
    print(f"Ninja tile coordinates: ({ninja_tile_x}, {ninja_tile_y})")
    
    # Check what tile the ninja is supposed to be in
    level_data = env.level_data
    if (0 <= ninja_tile_y < level_data.height and 0 <= ninja_tile_x < level_data.width):
        tile_value = level_data.get_tile(ninja_tile_y, ninja_tile_x)
        print(f"Ninja tile value: {tile_value} ({'empty' if tile_value == 0 else 'solid' if tile_value == 1 else 'other'})")
    
    # Look for empty tiles near the ninja spawn
    print("\nLooking for empty tiles near ninja spawn:")
    empty_tiles_nearby = []
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            check_x = ninja_tile_x + dx
            check_y = ninja_tile_y + dy
            
            if (0 <= check_y < level_data.height and 0 <= check_x < level_data.width):
                tile_value = level_data.get_tile(check_y, check_x)
                if tile_value == 0:
                    distance = (dx*dx + dy*dy)**0.5
                    pixel_x = check_x * 24 + 12  # Center of tile
                    pixel_y = check_y * 24 + 12
                    empty_tiles_nearby.append((check_x, check_y, pixel_x, pixel_y, distance))
    
    empty_tiles_nearby.sort(key=lambda x: x[4])  # Sort by distance
    
    if empty_tiles_nearby:
        print("Nearest empty tiles:")
        for i, (tx, ty, px, py, dist) in enumerate(empty_tiles_nearby[:5]):
            print(f"  {i+1}. Tile ({tx}, {ty}) -> pixel ({px}, {py}) distance: {dist:.1f}")
        
        # Suggest a better ninja spawn position
        best_tile = empty_tiles_nearby[0]
        print(f"\nSuggested ninja spawn position: ({best_tile[2]}, {best_tile[3]})")
        print(f"This would require map_data[1231] = {best_tile[2] / 6:.1f}")
        print(f"This would require map_data[1232] = {best_tile[3] / 6:.1f}")
    else:
        print("No empty tiles found nearby!")
    
    # Check if there are any entities at the ninja position
    print(f"\nChecking for entities at ninja position...")
    ninja_x, ninja_y = actual_ninja_pos
    
    entities_at_ninja = []
    for i, entity in enumerate(level_data.entities):
        entity_x = entity.get('x', 0)
        entity_y = entity.get('y', 0)
        distance = ((entity_x - ninja_x)**2 + (entity_y - ninja_y)**2)**0.5
        
        if distance < 24:  # Within one tile
            entities_at_ninja.append((i, entity, distance))
    
    if entities_at_ninja:
        print(f"Found {len(entities_at_ninja)} entities near ninja position:")
        for i, (idx, entity, dist) in enumerate(entities_at_ninja):
            print(f"  {i+1}. Entity {idx}: type={entity.get('type')}, pos=({entity.get('x')}, {entity.get('y')}), distance={dist:.1f}")
    else:
        print("No entities found near ninja position")


if __name__ == '__main__':
    debug_map_data()