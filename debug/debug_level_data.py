#!/usr/bin/env python3
"""
Debug script to investigate level data extraction issues.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from nclone.nsim import Simulator
from nclone.sim_config import SimConfig
from validate_with_bblock_test import load_bblock_test_map, create_level_data_from_simulator

def debug_level_data():
    """Debug level data extraction."""
    print("=== DEBUGGING LEVEL DATA EXTRACTION ===")
    
    # Load the bblock_test map
    map_data = load_bblock_test_map()
    if map_data is None:
        return
    
    print(f"Raw map data: {len(map_data)} bytes")
    print(f"First 100 bytes: {map_data[:100]}")
    
    # Create a simulator to extract level data
    config = SimConfig()
    config.level_name = "bblock_test"
    
    try:
        sim = Simulator(config)
        print(f"✅ Simulator created successfully")
        
        # Check simulator attributes
        print(f"Simulator attributes: {dir(sim)}")
        
        # Check simulator state
        if hasattr(sim, 'level'):
            print(f"Simulator level: {sim.level}")
            if hasattr(sim.level, 'tiles'):
                print(f"Level tiles shape: {sim.level.tiles.shape}")
                print(f"Level tiles dtype: {sim.level.tiles.dtype}")
                print(f"Unique tile values: {np.unique(sim.level.tiles)}")
        else:
            print("❌ Simulator has no 'level' attribute")
        
        # Check tile_dic
        print(f"Simulator tile_dic keys (first 20): {list(sim.tile_dic.keys())[:20]}")
        print(f"Simulator tile_dic values (first 20): {list(sim.tile_dic.values())[:20]}")
        print(f"Total tiles in tile_dic: {len(sim.tile_dic)}")
        
        # Check entity_dic
        print(f"Simulator entity_dic keys: {list(sim.entity_dic.keys())}")
        for entity_type, entity_list in sim.entity_dic.items():
            print(f"  Entity type {entity_type}: {len(entity_list)} entities")
        
        # Try to create level data anyway
        level_data_obj = create_level_data_from_simulator(sim)
        print(f"✅ Level data: {level_data_obj.tiles.shape} tiles, {len(level_data_obj.entities)} entities")
        
        # Check tile_dic after loading
        print(f"After loading - tile_dic keys (first 20): {list(sim.tile_dic.keys())[:20]}")
        print(f"After loading - tile_dic values (first 20): {list(sim.tile_dic.values())[:20]}")
        print(f"After loading - total tiles in tile_dic: {len(sim.tile_dic)}")
        
        # Check entity_dic after loading
        print(f"After loading - entity_dic keys: {list(sim.entity_dic.keys())}")
        for entity_type, entity_list in sim.entity_dic.items():
            if len(entity_list) > 0:
                print(f"  Entity type {entity_type}: {len(entity_list)} entities")
        
        # Look for non-zero tiles
        non_zero_tiles = [(k, v) for k, v in sim.tile_dic.items() if v != 0]
        print(f"Non-zero tiles: {non_zero_tiles[:20]}")  # Show first 20 non-zero tiles
        print(f"Total non-zero tiles: {len(non_zero_tiles)}")
        
        # Check coordinate ranges
        all_coords = list(sim.tile_dic.keys())
        x_coords = [x for x, y in all_coords]
        y_coords = [y for x, y in all_coords]
        print(f"X coordinate range: {min(x_coords)} to {max(x_coords)}")
        print(f"Y coordinate range: {min(y_coords)} to {max(y_coords)}")
        
        # Check what tiles are in the inner area (1-42, 1-23)
        inner_tiles = {k: v for k, v in sim.tile_dic.items() if 1 <= k[0] <= 42 and 1 <= k[1] <= 23}
        inner_non_zero = [(k, v) for k, v in inner_tiles.items() if v != 0]
        print(f"Inner area non-zero tiles: {inner_non_zero[:20]}")
        print(f"Total inner area non-zero tiles: {len(inner_non_zero)}")
        
        # Check level data object
        print(f"LevelData tiles shape: {level_data_obj.tiles.shape}")
        print(f"LevelData tiles dtype: {level_data_obj.tiles.dtype}")
        print(f"Unique tile values in LevelData: {np.unique(level_data_obj.tiles)}")
        
        # Show a sample of tiles from level data
        print("Sample tiles from LevelData (first 10x10):")
        sample = level_data_obj.tiles[:10, :10]
        for i in range(sample.shape[0]):
            row_str = " ".join(f"{sample[i, j]:2d}" for j in range(sample.shape[1]))
            print(f"  Row {i:2d}: {row_str}")
        
    except Exception as e:
        print(f"❌ Failed to create simulator: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    debug_level_data()