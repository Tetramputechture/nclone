#!/usr/bin/env python3
"""
Debug traversability checks to see why so few edges are created.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.edge_building import EdgeBuilder
from nclone.graph.feature_extraction import FeatureExtractor
from nclone.graph.hazard_system import HazardClassificationSystem
from nclone.graph.precise_collision import PreciseTileCollision
from nclone.graph.common import SUB_CELL_SIZE


def debug_traversability():
    """Debug traversability checks."""
    print("=" * 80)
    print("DEBUGGING TRAVERSABILITY CHECKS")
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
    print(f"Level size: {level_data.width}x{level_data.height} tiles")
    
    # Create edge builder
    feature_extractor = FeatureExtractor()
    edge_builder = EdgeBuilder(feature_extractor)
    
    # Find some empty tiles to test
    empty_tiles = []
    for tile_y in range(level_data.height):
        for tile_x in range(level_data.width):
            if level_data.get_tile(tile_y, tile_x) == 0:  # Empty tile
                empty_tiles.append((tile_x, tile_y))
    
    print(f"Found {len(empty_tiles)} empty tiles")
    
    if len(empty_tiles) < 2:
        print("Not enough empty tiles to test traversability")
        return
    
    # Test traversability between nearby empty tiles
    print(f"\nTesting traversability between empty tiles:")
    
    # Convert tile coordinates to sub-cell coordinates
    test_pairs = []
    for i in range(min(5, len(empty_tiles))):
        for j in range(i+1, min(i+6, len(empty_tiles))):
            tile1 = empty_tiles[i]
            tile2 = empty_tiles[j]
            
            # Convert to sub-cell coordinates
            sub_row1 = tile1[1] * 4 + 2  # Center of tile in sub-cells
            sub_col1 = tile1[0] * 4 + 2
            sub_row2 = tile2[1] * 4 + 2
            sub_col2 = tile2[0] * 4 + 2
            
            # Calculate distance
            distance = ((sub_col2 - sub_col1)**2 + (sub_row2 - sub_row1)**2)**0.5
            
            if distance < 20:  # Only test nearby tiles
                test_pairs.append(((sub_row1, sub_col1), (sub_row2, sub_col2), distance))
    
    test_pairs.sort(key=lambda x: x[2])  # Sort by distance
    
    print(f"Testing {len(test_pairs)} tile pairs:")
    
    traversable_count = 0
    
    for i, ((src_row, src_col), (tgt_row, tgt_col), distance) in enumerate(test_pairs[:10]):
        print(f"\nTest {i+1}: ({src_row}, {src_col}) -> ({tgt_row}, {tgt_col}) (distance: {distance:.1f})")
        
        # Convert to pixel coordinates
        src_x = src_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        src_y = src_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        tgt_x = tgt_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        tgt_y = tgt_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        
        print(f"  Pixel coordinates: ({src_x}, {src_y}) -> ({tgt_x}, {tgt_y})")
        
        # Test precise collision
        precise_result = edge_builder.is_precise_traversable(src_x, src_y, tgt_x, tgt_y, level_data.tiles)
        print(f"  Precise collision: {'✅ PASS' if precise_result else '❌ FAIL'}")
        
        if precise_result:
            # Test full traversability with hazards
            full_result = edge_builder.is_traversable_with_hazards(
                src_row, src_col, tgt_row, tgt_col, level_data, ninja_pos
            )
            print(f"  Full traversability: {'✅ PASS' if full_result else '❌ FAIL'}")
            
            if full_result:
                traversable_count += 1
        else:
            print(f"  Full traversability: ❌ SKIP (failed precise collision)")
    
    print(f"\nTraversability results:")
    print(f"  Traversable pairs: {traversable_count} out of {len(test_pairs[:10])}")
    print(f"  Success rate: {traversable_count / len(test_pairs[:10]) * 100:.1f}%")
    
    # Test a simple case: adjacent sub-cells in the same empty tile
    print(f"\nTesting adjacent sub-cells within the same empty tile:")
    
    if empty_tiles:
        tile_x, tile_y = empty_tiles[0]
        print(f"Using empty tile at ({tile_x}, {tile_y})")
        
        # Test all 4 sub-cells within this tile
        base_sub_row = tile_y * 4
        base_sub_col = tile_x * 4
        
        adjacent_tests = [
            ((base_sub_row + 1, base_sub_col + 1), (base_sub_row + 1, base_sub_col + 2)),  # horizontal
            ((base_sub_row + 1, base_sub_col + 1), (base_sub_row + 2, base_sub_col + 1)),  # vertical
            ((base_sub_row + 1, base_sub_col + 1), (base_sub_row + 2, base_sub_col + 2)),  # diagonal
        ]
        
        for i, ((src_row, src_col), (tgt_row, tgt_col)) in enumerate(adjacent_tests):
            print(f"\n  Adjacent test {i+1}: ({src_row}, {src_col}) -> ({tgt_row}, {tgt_col})")
            
            src_x = src_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            src_y = src_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            tgt_x = tgt_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            tgt_y = tgt_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            
            precise_result = edge_builder.is_precise_traversable(src_x, src_y, tgt_x, tgt_y, level_data.tiles)
            print(f"    Precise collision: {'✅ PASS' if precise_result else '❌ FAIL'}")
            
            if precise_result:
                full_result = edge_builder.is_traversable_with_hazards(
                    src_row, src_col, tgt_row, tgt_col, level_data, ninja_pos
                )
                print(f"    Full traversability: {'✅ PASS' if full_result else '❌ FAIL'}")


if __name__ == '__main__':
    debug_traversability()