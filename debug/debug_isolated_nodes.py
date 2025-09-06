#!/usr/bin/env python3
"""
Debug script to understand why nodes are being isolated by the collision detection fix.
"""

import os
import sys
import numpy as np

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.level_data import LevelData
from nclone.graph.common import SUB_CELL_SIZE, SUB_GRID_WIDTH, SUB_GRID_HEIGHT
from nclone.graph.precise_collision import PreciseTileCollision
from nclone.constants import TILE_PIXEL_SIZE
from nclone.constants.physics_constants import NINJA_RADIUS


def debug_isolated_nodes():
    """Debug why nodes are being isolated."""
    print("=== DEBUGGING ISOLATED NODES ===")
    
    # Create a simple level
    width, height = 5, 5
    tiles = np.zeros((height, width), dtype=int)
    
    # Make borders solid
    tiles[0, :] = 1  # Top border
    tiles[-1, :] = 1  # Bottom border
    tiles[:, 0] = 1  # Left border
    tiles[:, -1] = 1  # Right border
    
    print("Tile layout:")
    for y, row in enumerate(tiles):
        print(f"Row {y}: " + "".join("█" if tile == 1 else "." for tile in row))
    
    print(f"\nConstants:")
    print(f"TILE_PIXEL_SIZE = {TILE_PIXEL_SIZE}")
    print(f"SUB_CELL_SIZE = {SUB_CELL_SIZE}")
    print(f"NINJA_RADIUS = {NINJA_RADIUS}")
    print(f"SUB_GRID_WIDTH = {SUB_GRID_WIDTH}")
    print(f"SUB_GRID_HEIGHT = {SUB_GRID_HEIGHT}")
    
    # Test the collision detection directly
    collision_detector = PreciseTileCollision()
    
    print(f"\n=== TESTING POSITION TRAVERSABILITY ===")
    
    # Test some specific positions
    test_positions = [
        # Center of empty tiles
        (1.5 * TILE_PIXEL_SIZE, 1.5 * TILE_PIXEL_SIZE),  # Center of tile (1,1)
        (2.5 * TILE_PIXEL_SIZE, 2.5 * TILE_PIXEL_SIZE),  # Center of tile (2,2)
        
        # Near borders of empty tiles
        (1.1 * TILE_PIXEL_SIZE, 1.1 * TILE_PIXEL_SIZE),  # Near corner of tile (1,1)
        (1.9 * TILE_PIXEL_SIZE, 1.9 * TILE_PIXEL_SIZE),  # Near opposite corner
        
        # Very close to solid tiles
        (1.0 * TILE_PIXEL_SIZE + NINJA_RADIUS + 1, 1.5 * TILE_PIXEL_SIZE),  # Just inside from left border
        (2.0 * TILE_PIXEL_SIZE - NINJA_RADIUS - 1, 1.5 * TILE_PIXEL_SIZE),  # Just inside from right border
    ]
    
    for i, (x, y) in enumerate(test_positions):
        is_traversable = collision_detector._is_position_traversable(x, y, tiles, NINJA_RADIUS)
        tile_x = x / TILE_PIXEL_SIZE
        tile_y = y / TILE_PIXEL_SIZE
        print(f"Position {i}: ({x:.1f}, {y:.1f}) = tile({tile_x:.2f}, {tile_y:.2f}) -> {'✅' if is_traversable else '❌'}")
    
    # Test sub-cell positions systematically
    print(f"\n=== TESTING SUB-CELL POSITIONS ===")
    
    # Focus on a small area to understand the pattern
    test_area_tiles = [(1, 1), (1, 2), (2, 1), (2, 2)]  # 2x2 area of empty tiles
    
    for tile_x, tile_y in test_area_tiles:
        print(f"\nTile ({tile_x}, {tile_y}) - should be empty:")
        
        # Test all sub-cells within this tile
        for sub_y in range(4):  # 4 sub-cells per tile
            for sub_x in range(4):
                # Calculate sub-cell center position
                pixel_x = tile_x * TILE_PIXEL_SIZE + sub_x * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
                pixel_y = tile_y * TILE_PIXEL_SIZE + sub_y * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
                
                is_traversable = collision_detector._is_position_traversable(pixel_x, pixel_y, tiles, NINJA_RADIUS)
                
                print(f"  Sub-cell ({sub_y},{sub_x}): pos({pixel_x},{pixel_y}) -> {'✅' if is_traversable else '❌'}")
    
    # Test edge cases near solid tiles
    print(f"\n=== TESTING EDGE CASES NEAR SOLID TILES ===")
    
    # Test positions near the border between empty and solid tiles
    border_positions = [
        # Near left border (solid tile at x=0)
        (TILE_PIXEL_SIZE + NINJA_RADIUS - 1, 1.5 * TILE_PIXEL_SIZE),  # Just too close
        (TILE_PIXEL_SIZE + NINJA_RADIUS, 1.5 * TILE_PIXEL_SIZE),     # Exactly at limit
        (TILE_PIXEL_SIZE + NINJA_RADIUS + 1, 1.5 * TILE_PIXEL_SIZE), # Just safe
        
        # Near top border (solid tile at y=0)
        (1.5 * TILE_PIXEL_SIZE, TILE_PIXEL_SIZE + NINJA_RADIUS - 1),  # Just too close
        (1.5 * TILE_PIXEL_SIZE, TILE_PIXEL_SIZE + NINJA_RADIUS),     # Exactly at limit
        (1.5 * TILE_PIXEL_SIZE, TILE_PIXEL_SIZE + NINJA_RADIUS + 1), # Just safe
    ]
    
    for i, (x, y) in enumerate(border_positions):
        is_traversable = collision_detector._is_position_traversable(x, y, tiles, NINJA_RADIUS)
        distance_to_border = min(x - TILE_PIXEL_SIZE, y - TILE_PIXEL_SIZE) if x > TILE_PIXEL_SIZE and y > TILE_PIXEL_SIZE else 0
        print(f"Border test {i}: pos({x:.1f},{y:.1f}) dist_to_solid={distance_to_border:.1f} -> {'✅' if is_traversable else '❌'}")


if __name__ == "__main__":
    debug_isolated_nodes()