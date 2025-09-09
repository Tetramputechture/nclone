#!/usr/bin/env python3
"""Debug script to examine entity data in test maps."""

import os

def debug_map_entities(map_path):
    """Debug entity data in a map file."""
    print(f"Debugging entities in {map_path}")
    
    with open(map_path, "rb") as map_file:
        map_data = [int(b) for b in map_file.read()]
    
    print(f"Total map data size: {len(map_data)} bytes")
    
    # Ninja spawn position (bytes 1150-1151)
    ninja_x = map_data[1150]
    ninja_y = map_data[1151]
    print(f"Ninja spawn: x={ninja_x}, y={ninja_y}")
    
    # Entity data starts at byte 1230 (same as MapLoader)
    entities_found = [(1, ninja_x, ninja_y)]  # Add ninja
    
    index = 1230
    while index < len(map_data) - 4:
        entity_type_id = map_data[index]
        if entity_type_id != 0:  # Skip empty entity slots
            x = map_data[index + 1]
            y = map_data[index + 2]
            entities_found.append((entity_type_id, x, y))
            print(f"Entity: type={entity_type_id}, x={x}, y={y}")
            print(f"  Raw bytes: {map_data[index:index+5]}")
        index += 5
    
    print(f"Found {len(entities_found)} entities")
    print()

def main():
    test_maps = [
        "nclone/test_maps/simple-walk",
        "nclone/test_maps/long-walk", 
        "nclone/test_maps/path-jump-required",
        "nclone/test_maps/only-jump",
    ]
    
    for map_path in test_maps:
        if os.path.exists(map_path):
            debug_map_entities(map_path)
        else:
            print(f"Map not found: {map_path}")

if __name__ == "__main__":
    main()