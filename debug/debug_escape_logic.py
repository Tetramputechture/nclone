#!/usr/bin/env python3
"""
Debug the escape logic to see which connections are being made.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.edge_building import EdgeBuilder
from nclone.graph.common import SUB_CELL_SIZE, SUB_GRID_WIDTH, SUB_GRID_HEIGHT
import math


def debug_escape_logic():
    """Debug the escape logic."""
    print("=" * 80)
    print("DEBUGGING ESCAPE LOGIC")
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
    
    # Create edge builder
    from nclone.graph.feature_extraction import FeatureExtractor
    feature_extractor = FeatureExtractor()
    edge_builder = EdgeBuilder(feature_extractor)
    
    # Test the escape logic for specific node pairs
    ninja_x, ninja_y = ninja_pos
    ninja_escape_range = SUB_CELL_SIZE * 4  # 24 pixels
    
    print(f"Ninja escape range: {ninja_escape_range} pixels")
    
    # Test nodes around ninja
    ninja_sub_col = int(ninja_x // SUB_CELL_SIZE)
    ninja_sub_row = int(ninja_y // SUB_CELL_SIZE)
    
    print(f"Ninja sub-grid position: ({ninja_sub_row}, {ninja_sub_col})")
    
    # Check nodes in ninja's solid tile
    solid_nodes = [
        (73, 21),  # (129, 441)
        (73, 22),  # (135, 441)
        (74, 21),  # (129, 447)
        (74, 22),  # (135, 447)
    ]
    
    # Check nodes in adjacent empty tiles
    empty_nodes = [
        (71, 21),  # (129, 429) - tile (5, 17)
        (71, 22),  # (135, 429) - tile (5, 17)
        (71, 19),  # (117, 429) - tile (4, 17)
        (71, 20),  # (123, 429) - tile (4, 17)
    ]
    
    print(f"\nTesting escape logic for solid->empty connections:")
    
    for solid_row, solid_col in solid_nodes:
        src_x = solid_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        src_y = solid_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        
        for empty_row, empty_col in empty_nodes:
            tgt_x = empty_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            tgt_y = empty_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            
            # Calculate distances
            ninja_distance_src = math.sqrt((src_x - ninja_x)**2 + (src_y - ninja_y)**2)
            ninja_distance_tgt = math.sqrt((tgt_x - ninja_x)**2 + (tgt_y - ninja_y)**2)
            
            # Check if near ninja
            near_ninja = (ninja_distance_src <= ninja_escape_range or 
                         ninja_distance_tgt <= ninja_escape_range)
            
            # Check tile types
            src_tile_x = int(src_x // 24)
            src_tile_y = int(src_y // 24)
            tgt_tile_x = int(tgt_x // 24)
            tgt_tile_y = int(tgt_y // 24)
            
            src_solid = (0 <= src_tile_x < level_data.width and 
                        0 <= src_tile_y < level_data.height and
                        level_data.get_tile(src_tile_y, src_tile_x) == 1)
            tgt_empty = (0 <= tgt_tile_x < level_data.width and 
                        0 <= tgt_tile_y < level_data.height and
                        level_data.get_tile(tgt_tile_y, tgt_tile_x) == 0)
            
            # Check tile adjacency
            tile_distance = max(abs(src_tile_x - tgt_tile_x), abs(src_tile_y - tgt_tile_y))
            
            # Test escape logic conditions
            escape_conditions = (near_ninja and src_solid and tgt_empty and tile_distance <= 1)
            
            print(f"  ({solid_row}, {solid_col}) -> ({empty_row}, {empty_col}):")
            print(f"    Positions: ({src_x}, {src_y}) -> ({tgt_x}, {tgt_y})")
            print(f"    Tiles: ({src_tile_x}, {src_tile_y}) -> ({tgt_tile_x}, {tgt_tile_y})")
            print(f"    Ninja distances: {ninja_distance_src:.1f}, {ninja_distance_tgt:.1f}")
            print(f"    Near ninja: {near_ninja}")
            print(f"    Src solid: {src_solid}, Tgt empty: {tgt_empty}")
            print(f"    Tile distance: {tile_distance}")
            print(f"    Escape logic triggered: {escape_conditions}")
            
            if escape_conditions:
                # Test the actual traversability check
                traversable = edge_builder.is_traversable_with_hazards(
                    solid_row, solid_col, empty_row, empty_col, level_data, ninja_pos
                )
                print(f"    Traversable: {traversable}")
            
            print()


if __name__ == '__main__':
    debug_escape_logic()