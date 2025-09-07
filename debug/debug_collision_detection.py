#!/usr/bin/env python3
"""
Debug script to test collision detection for walkable edges in solid tiles.
"""

import os
import sys
import numpy as np

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.graph.precise_collision import PreciseTileCollision
from nclone.graph.common import SUB_CELL_SIZE
from nclone.constants import TILE_PIXEL_SIZE


def test_collision_detection():
    """Test collision detection in solid tiles."""
    print("=== TESTING COLLISION DETECTION ===")
    
    # Create a simple level with solid tiles
    width, height = 5, 5
    tiles = np.zeros((height, width), dtype=int)
    
    # Make all tiles solid
    tiles[:, :] = 1
    
    print("Tile layout (all solid):")
    for row in tiles:
        print("".join("█" if tile == 1 else "." for tile in row))
    
    # Create collision detector
    collision_detector = PreciseTileCollision()
    
    # Test various paths through solid tiles
    test_cases = [
        # Path entirely within a solid tile
        {
            'name': 'Within solid tile (0,0)',
            'src_x': 0.25 * TILE_PIXEL_SIZE,
            'src_y': 0.25 * TILE_PIXEL_SIZE,
            'tgt_x': 0.75 * TILE_PIXEL_SIZE,
            'tgt_y': 0.25 * TILE_PIXEL_SIZE,
        },
        # Path from solid tile to solid tile
        {
            'name': 'Solid to solid (0,0) -> (1,0)',
            'src_x': 0.75 * TILE_PIXEL_SIZE,
            'src_y': 0.5 * TILE_PIXEL_SIZE,
            'tgt_x': 1.25 * TILE_PIXEL_SIZE,
            'tgt_y': 0.5 * TILE_PIXEL_SIZE,
        },
        # Path through center of solid tile
        {
            'name': 'Through center of solid tile (1,1)',
            'src_x': 1.25 * TILE_PIXEL_SIZE,
            'src_y': 1.25 * TILE_PIXEL_SIZE,
            'tgt_x': 1.75 * TILE_PIXEL_SIZE,
            'tgt_y': 1.75 * TILE_PIXEL_SIZE,
        }
    ]
    
    print(f"\nTesting collision detection:")
    print(f"TILE_PIXEL_SIZE = {TILE_PIXEL_SIZE}")
    print(f"SUB_CELL_SIZE = {SUB_CELL_SIZE}")
    
    for test_case in test_cases:
        is_traversable = collision_detector.is_path_traversable(
            test_case['src_x'],
            test_case['src_y'],
            test_case['tgt_x'],
            test_case['tgt_y'],
            tiles
        )
        
        # Convert to tile coordinates for display
        src_tile_x = test_case['src_x'] / TILE_PIXEL_SIZE
        src_tile_y = test_case['src_y'] / TILE_PIXEL_SIZE
        tgt_tile_x = test_case['tgt_x'] / TILE_PIXEL_SIZE
        tgt_tile_y = test_case['tgt_y'] / TILE_PIXEL_SIZE
        
        print(f"  {test_case['name']}:")
        print(f"    From: ({src_tile_x:.2f}, {src_tile_y:.2f}) tile coords")
        print(f"    To:   ({tgt_tile_x:.2f}, {tgt_tile_y:.2f}) tile coords")
        print(f"    Traversable: {is_traversable} {'❌ WRONG!' if is_traversable else '✅ Correct'}")
    
    # Test sub-cell coordinates like the edge builder uses
    print(f"\n=== TESTING SUB-CELL COORDINATES ===")
    
    # Test the same coordinates that the edge builder would use
    sub_cell_tests = [
        {
            'name': 'Sub-cell (0,0) to (0,1)',
            'src_row': 0, 'src_col': 0,
            'tgt_row': 0, 'tgt_col': 1,
        },
        {
            'name': 'Sub-cell (1,1) to (1,2)',
            'src_row': 1, 'src_col': 1,
            'tgt_row': 1, 'tgt_col': 2,
        },
        {
            'name': 'Sub-cell (2,2) to (3,2)',
            'src_row': 2, 'src_col': 2,
            'tgt_row': 3, 'tgt_col': 2,
        }
    ]
    
    for test_case in sub_cell_tests:
        # Convert sub-cell coordinates to pixel coordinates (same as edge builder)
        src_x = test_case['src_col'] * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        src_y = test_case['src_row'] * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        tgt_x = test_case['tgt_col'] * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        tgt_y = test_case['tgt_row'] * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        
        is_traversable = collision_detector.is_path_traversable(
            src_x, src_y, tgt_x, tgt_y, tiles
        )
        
        # Convert to tile coordinates for display
        src_tile_x = src_x / TILE_PIXEL_SIZE
        src_tile_y = src_y / TILE_PIXEL_SIZE
        tgt_tile_x = tgt_x / TILE_PIXEL_SIZE
        tgt_tile_y = tgt_y / TILE_PIXEL_SIZE
        
        print(f"  {test_case['name']}:")
        print(f"    Sub-cell: ({test_case['src_row']}, {test_case['src_col']}) -> ({test_case['tgt_row']}, {test_case['tgt_col']})")
        print(f"    Pixels: ({src_x}, {src_y}) -> ({tgt_x}, {tgt_y})")
        print(f"    Tile coords: ({src_tile_x:.2f}, {src_tile_y:.2f}) -> ({tgt_tile_x:.2f}, {tgt_tile_y:.2f})")
        print(f"    Traversable: {is_traversable} {'❌ WRONG!' if is_traversable else '✅ Correct'}")


if __name__ == "__main__":
    test_collision_detection()