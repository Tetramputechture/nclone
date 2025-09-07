#!/usr/bin/env python3
"""
Create a test map with solid tiles to test graph functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from nclone.map_generation.map import Map
from nclone.map_generation.constants import *

def create_test_map_with_solid_tiles():
    """Create a test map with some solid tiles for testing."""
    print("=== CREATING TEST MAP WITH SOLID TILES ===")
    
    # Create a new map
    map_obj = Map()
    
    # Add some solid tiles in the middle area
    # Create a simple pattern with solid tiles (tile type 1 = solid)
    for x in range(10, 20):
        for y in range(8, 12):
            map_obj.set_tile(x, y, 1)  # 1 = solid tile
    
    # Add some platforms
    for x in range(5, 15):
        map_obj.set_tile(x, 15, 1)  # 1 = solid tile
    
    for x in range(25, 35):
        map_obj.set_tile(x, 10, 1)  # 1 = solid tile
    
    # Add some switches and doors for functional edge testing
    # Entity type 3 = switch, entity type 4 = door (based on common N++ entity types)
    map_obj.add_entity(3, 8, 16, switch_x=20, switch_y=8)  # Switch at (8,16) connected to door at (20,8)
    map_obj.add_entity(4, 20, 8, switch_x=8, switch_y=16)  # Door at (20,8) connected to switch at (8,16)
    
    map_obj.add_entity(3, 30, 11, switch_x=15, switch_y=15)  # Switch at (30,11) connected to door at (15,15)
    map_obj.add_entity(4, 15, 15, switch_x=30, switch_y=11)  # Door at (15,15) connected to switch at (30,11)
    
    # Add ninja spawn (entity type 0 is usually ninja)
    map_obj.add_entity(0, 5, 20)
    
    # Save the map
    map_data = map_obj.map_data()
    
    # Debug: check for values outside byte range
    out_of_range = [x for x in map_data if x < 0 or x > 255]
    if out_of_range:
        print(f"Warning: Found {len(out_of_range)} values outside byte range: {out_of_range[:10]}")
        # Clamp values to byte range
        map_data = [max(0, min(255, x)) for x in map_data]
    
    # Convert to bytes and write to file
    map_bytes = bytes(map_data)
    with open('debug_test_map', 'wb') as f:
        f.write(map_bytes)
    
    print(f"✅ Created test map with {len(map_data)} bytes")
    print("Map features:")
    print("- Solid tile block from (10,8) to (19,11)")
    print("- Platform from (5,15) to (14,15)")
    print("- Platform from (25,10) to (34,10)")
    print("- Switch at (8,16) connected to door at (20,8)")
    print("- Switch at (30,11) connected to door at (15,15)")
    print("- Ninja spawn at (5,20)")
    
    return map_data

if __name__ == "__main__":
    create_test_map_with_solid_tiles()