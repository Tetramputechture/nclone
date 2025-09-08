#!/usr/bin/env python3
"""
Debug script to examine the structure of level_data from the environment.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold

def debug_level_data():
    """Debug the structure of level_data."""
    print("🔍 Debugging level_data structure...")
    
    # Initialize environment
    env = BasicLevelNoGold(render_mode="rgb_array", custom_map_path="nclone/test_maps/doortest")
    env.reset()
    
    print(f"Type of env.level_data: {type(env.level_data)}")
    
    if isinstance(env.level_data, list):
        print(f"Level data is a list with {len(env.level_data)} elements")
        if len(env.level_data) > 0:
            print(f"First element type: {type(env.level_data[0])}")
            print(f"First element: {env.level_data[0]}")
            
            if len(env.level_data[0]) > 0:
                print(f"First row length: {len(env.level_data[0])}")
                print(f"First few elements of first row: {env.level_data[0][:10]}")
        
        print(f"Level dimensions: {len(env.level_data)} rows x {len(env.level_data[0]) if env.level_data else 0} cols")
    else:
        print(f"Level data attributes: {dir(env.level_data)}")
        if hasattr(env.level_data, 'height'):
            print(f"Height: {env.level_data.height}")
        if hasattr(env.level_data, 'width'):
            print(f"Width: {env.level_data.width}")
        if hasattr(env.level_data, 'tiles'):
            print(f"Tiles type: {type(env.level_data.tiles)}")
    
    # Check entities
    print(f"\nType of env.entities: {type(env.entities)}")
    if hasattr(env, 'entities'):
        print(f"Number of entities: {len(env.entities) if env.entities else 0}")
        if env.entities and len(env.entities) > 0:
            print(f"First entity: {env.entities[0]}")

if __name__ == "__main__":
    debug_level_data()