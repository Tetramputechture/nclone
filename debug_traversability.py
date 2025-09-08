#!/usr/bin/env python3
"""
Debug traversability system to understand why WALK edges aren't being created.
"""

import sys
import os
import numpy as np

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.edge_building import EdgeBuilder
from nclone.graph.feature_extraction import FeatureExtractor
from nclone.constants.physics_constants import TILE_PIXEL_SIZE

def debug_traversability():
    """Debug why traversability checks are failing."""
    print("=" * 80)
    print("🔍 DEBUGGING TRAVERSABILITY SYSTEM")
    print("=" * 80)
    
    # Load doortest map
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42
    )
    
    env.reset()
    level_data = env.level_data
    ninja_position = env.nplay_headless.ninja_position()
    
    print(f"🗺️  Map: {level_data.width}x{level_data.height} tiles")
    print(f"🥷 Ninja: {ninja_position}")
    
    # Create edge builder to test traversability
    feature_extractor = FeatureExtractor()
    edge_builder = EdgeBuilder(feature_extractor)
    
    # Test some specific positions around the ninja
    ninja_tile_x = int(ninja_position[0] // TILE_PIXEL_SIZE)
    ninja_tile_y = int(ninja_position[1] // TILE_PIXEL_SIZE)
    
    print(f"🎯 Ninja tile: ({ninja_tile_x}, {ninja_tile_y})")
    
    # Check tiles around ninja
    print(f"\n🔍 Tile analysis around ninja:")
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            tile_x = ninja_tile_x + dx
            tile_y = ninja_tile_y + dy
            
            if 0 <= tile_x < level_data.width and 0 <= tile_y < level_data.height:
                tile_value = level_data.get_tile(tile_y, tile_x)
                print(f"   Tile ({tile_x}, {tile_y}): value={tile_value}")
    
    # Test sub-cell traversability around ninja
    print(f"\n🔍 Sub-cell traversability tests:")
    SUB_CELL_SIZE = 6  # From constants
    
    # Convert ninja position to sub-cell coordinates
    ninja_sub_col = int(ninja_position[0] // SUB_CELL_SIZE)
    ninja_sub_row = int(ninja_position[1] // SUB_CELL_SIZE)
    
    print(f"🎯 Ninja sub-cell: ({ninja_sub_row}, {ninja_sub_col})")
    
    # Test 8-connected neighbors
    directions = [
        (0, 1, "right"),
        (1, 0, "down"),
        (0, -1, "left"),
        (-1, 0, "up"),
        (1, 1, "down-right"),
        (1, -1, "down-left"),
        (-1, 1, "up-right"),
        (-1, -1, "up-left"),
    ]
    
    traversable_count = 0
    total_count = 0
    
    for dr, dc, direction in directions:
        src_row = ninja_sub_row
        src_col = ninja_sub_col
        tgt_row = ninja_sub_row + dr
        tgt_col = ninja_sub_col + dc
        
        total_count += 1
        
        # Test basic tile check first
        src_x = src_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        src_y = src_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        tgt_x = tgt_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        tgt_y = tgt_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        
        src_tile_x = int(src_x // TILE_PIXEL_SIZE)
        src_tile_y = int(src_y // TILE_PIXEL_SIZE)
        tgt_tile_x = int(tgt_x // TILE_PIXEL_SIZE)
        tgt_tile_y = int(tgt_y // TILE_PIXEL_SIZE)
        
        src_clear = (0 <= src_tile_x < level_data.width and 
                    0 <= src_tile_y < level_data.height and
                    level_data.get_tile(src_tile_y, src_tile_x) == 0)
        tgt_clear = (0 <= tgt_tile_x < level_data.width and 
                    0 <= tgt_tile_y < level_data.height and
                    level_data.get_tile(tgt_tile_y, tgt_tile_x) == 0)
        
        basic_traversable = src_clear and tgt_clear
        
        # Test full traversability check
        try:
            full_traversable = edge_builder.is_traversable_with_hazards(
                src_row, src_col, tgt_row, tgt_col, level_data, ninja_position
            )
        except Exception as e:
            full_traversable = False
            print(f"   ❌ {direction}: Exception in traversability check: {e}")
            continue
        
        if basic_traversable:
            traversable_count += 1
        
        status = "✅" if basic_traversable else "❌"
        full_status = "✅" if full_traversable else "❌"
        
        print(f"   {status} {direction}: basic={basic_traversable}, full={full_traversable}")
        print(f"      src=({src_x}, {src_y}) tile=({src_tile_x}, {src_tile_y}) clear={src_clear}")
        print(f"      tgt=({tgt_x}, {tgt_y}) tile=({tgt_tile_x}, {tgt_tile_y}) clear={tgt_clear}")
        
        if basic_traversable != full_traversable:
            print(f"      ⚠️  MISMATCH: Basic says {basic_traversable}, Full says {full_traversable}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Basic traversable directions: {traversable_count}/{total_count}")
    print(f"   This explains why ninja can only reach {traversable_count + 1} nodes (including itself)")
    
    if traversable_count == 0:
        print(f"   ❌ NO traversable directions - ninja is completely isolated!")
        print(f"   🔧 This is why no WALK edges are created")
    else:
        print(f"   ✅ Some traversable directions found")

if __name__ == "__main__":
    debug_traversability()