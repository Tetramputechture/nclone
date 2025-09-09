#!/usr/bin/env python3
"""
Debug the coordinate system conversion issue.
"""

import sys
import numpy as np

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.constants.physics_constants import TILE_PIXEL_SIZE

def debug_coordinate_conversion():
    """Debug the coordinate system conversion."""
    print("COORDINATE SYSTEM DEBUG")
    print("=" * 50)
    
    # Create the same test level
    level_data = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Top wall
        [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],  # Corridor with wall in middle
        [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],  # Corridor with wall in middle
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Open corridor at bottom
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Open corridor at bottom
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Bottom wall
    ])
    
    print(f"Level data shape: {level_data.shape}")
    print(f"TILE_PIXEL_SIZE: {TILE_PIXEL_SIZE}")
    
    # Show level layout with coordinates
    print("\nLevel layout with array indices:")
    for i, row in enumerate(level_data):
        row_str = f"Row {i}: "
        for j, val in enumerate(row):
            row_str += f"{val}"
        print(row_str)
    
    # Test ninja position
    ninja_x = 1 * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2  # 36.0
    ninja_y = 1 * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2  # 36.0
    
    print(f"\nNinja position: ({ninja_x}, {ninja_y})")
    
    # Current conversion logic (from edge_building.py)
    tile_x = int(ninja_x // TILE_PIXEL_SIZE)
    tile_y = int(ninja_y // TILE_PIXEL_SIZE)
    print(f"Tile coordinates: ({tile_x}, {tile_y})")
    
    # Adjustment for padding
    data_tile_x = tile_x - 1
    data_tile_y = tile_y - 1
    print(f"Data coordinates: ({data_tile_x}, {data_tile_y})")
    
    if 0 <= data_tile_y < level_data.shape[0] and 0 <= data_tile_x < level_data.shape[1]:
        tile_value = level_data[data_tile_y, data_tile_x]
        print(f"Tile value at data[{data_tile_y}][{data_tile_x}]: {tile_value}")
        print(f"Is clear: {tile_value == 0}")
    else:
        print("Out of bounds!")
    
    # What should it be?
    print(f"\nExpected: Ninja should be in empty space")
    print(f"Level data at [1][1]: {level_data[1, 1]} (should be 0)")
    print(f"Level data at [0][0]: {level_data[0, 0]} (this is 1 - wall)")
    
    # Test different positions
    print(f"\nTesting different positions:")
    test_positions = [
        (12, 12),   # Should be tile (0,0) -> data[-1][-1] (out of bounds)
        (36, 36),   # Should be tile (1,1) -> data[0][0] (currently wrong)
        (60, 36),   # Should be tile (2,1) -> data[0][1] 
    ]
    
    for test_x, test_y in test_positions:
        tile_x = int(test_x // TILE_PIXEL_SIZE)
        tile_y = int(test_y // TILE_PIXEL_SIZE)
        data_tile_x = tile_x - 1
        data_tile_y = tile_y - 1
        
        print(f"Position ({test_x}, {test_y}) -> tile ({tile_x}, {tile_y}) -> data ({data_tile_x}, {data_tile_y})")
        if 0 <= data_tile_y < level_data.shape[0] and 0 <= data_tile_x < level_data.shape[1]:
            tile_value = level_data[data_tile_y, data_tile_x]
            print(f"  Tile value: {tile_value}")
        else:
            print(f"  OUT OF BOUNDS")

if __name__ == "__main__":
    debug_coordinate_conversion()