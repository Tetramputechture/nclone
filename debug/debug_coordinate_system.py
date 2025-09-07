#!/usr/bin/env python3
"""
Debug script to understand the coordinate system used in the graph.
"""

import os
import sys
import numpy as np

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.level_data import LevelData
from nclone.graph.common import SUB_CELL_SIZE, SUB_GRID_WIDTH, SUB_GRID_HEIGHT
from nclone.constants import TILE_PIXEL_SIZE
from nclone.constants.physics_constants import FULL_MAP_WIDTH_PX, FULL_MAP_HEIGHT_PX


def debug_coordinate_system():
    """Debug the coordinate system used in the graph."""
    print("=== DEBUGGING COORDINATE SYSTEM ===")
    
    # Create a simple 5x5 level
    width, height = 5, 5
    tiles = np.zeros((height, width), dtype=int)
    
    # Make some tiles solid for reference
    tiles[1, 1] = 1  # Solid tile at (1, 1)
    tiles[2, 2] = 1  # Solid tile at (2, 2)
    
    print("Tile layout:")
    for y, row in enumerate(tiles):
        print(f"Row {y}: " + "".join("█" if tile == 1 else "." for tile in row))
    
    # No entities for this test
    entities = []
    
    level_data = LevelData(
        tiles=tiles,
        entities=entities,
        level_id='debug_coordinates'
    )
    
    # Build graph
    builder = HierarchicalGraphBuilder()
    ninja_position = (2.5 * TILE_PIXEL_SIZE, 2.5 * TILE_PIXEL_SIZE)
    ninja_velocity = (0.0, 0.0)
    ninja_state = 0
    
    hierarchical_graph_data = builder.build_graph(
        level_data, ninja_position, ninja_velocity, ninja_state
    )
    
    graph_data = hierarchical_graph_data.sub_cell_graph
    print(f"\nGraph built: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    print(f"\nConstants:")
    print(f"TILE_PIXEL_SIZE = {TILE_PIXEL_SIZE}")
    print(f"SUB_CELL_SIZE = {SUB_CELL_SIZE}")
    print(f"SUB_GRID_WIDTH = {SUB_GRID_WIDTH}")
    print(f"SUB_GRID_HEIGHT = {SUB_GRID_HEIGHT}")
    print(f"FULL_MAP_WIDTH_PX = {FULL_MAP_WIDTH_PX}")
    print(f"FULL_MAP_HEIGHT_PX = {FULL_MAP_HEIGHT_PX}")
    
    # Analyze some specific sub-cell nodes
    print(f"\n=== ANALYZING SUB-CELL NODES ===")
    
    def analyze_sub_cell_node(node_idx):
        """Analyze a specific sub-cell node."""
        if node_idx >= SUB_GRID_WIDTH * SUB_GRID_HEIGHT:
            print(f"Node {node_idx}: Entity node (skipping)")
            return
        
        # Calculate sub-cell coordinates from node index
        sub_row = node_idx // SUB_GRID_WIDTH
        sub_col = node_idx % SUB_GRID_WIDTH
        
        # Convert to pixel coordinates (center of sub-cell)
        pixel_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        pixel_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        
        # Convert to tile coordinates
        tile_x = pixel_x / TILE_PIXEL_SIZE
        tile_y = pixel_y / TILE_PIXEL_SIZE
        
        # Check if in solid tile (accounting for potential border)
        # The level data might have a border, so we need to check the actual tile
        actual_tile_x = int(tile_x)
        actual_tile_y = int(tile_y)
        
        # Check bounds and get tile value
        is_solid = False
        if 0 <= actual_tile_y < tiles.shape[0] and 0 <= actual_tile_x < tiles.shape[1]:
            is_solid = tiles[actual_tile_y, actual_tile_x] == 1
        
        print(f"Node {node_idx}:")
        print(f"  Sub-cell: ({sub_row}, {sub_col})")
        print(f"  Pixel: ({pixel_x}, {pixel_y})")
        print(f"  Tile: ({tile_x:.2f}, {tile_y:.2f})")
        print(f"  Actual tile: ({actual_tile_x}, {actual_tile_y})")
        print(f"  Is solid: {is_solid}")
    
    # Analyze first few nodes
    for i in range(min(10, SUB_GRID_WIDTH * SUB_GRID_HEIGHT)):
        analyze_sub_cell_node(i)
        if i < 9:
            print()
    
    # Look for nodes that should be in solid tiles
    print(f"\n=== LOOKING FOR NODES IN SOLID TILES ===")
    
    solid_tile_nodes = []
    for node_idx in range(min(100, SUB_GRID_WIDTH * SUB_GRID_HEIGHT)):
        sub_row = node_idx // SUB_GRID_WIDTH
        sub_col = node_idx % SUB_GRID_WIDTH
        
        pixel_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        pixel_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        
        tile_x = pixel_x / TILE_PIXEL_SIZE
        tile_y = pixel_y / TILE_PIXEL_SIZE
        
        actual_tile_x = int(tile_x)
        actual_tile_y = int(tile_y)
        
        if 0 <= actual_tile_y < tiles.shape[0] and 0 <= actual_tile_x < tiles.shape[1]:
            if tiles[actual_tile_y, actual_tile_x] == 1:
                solid_tile_nodes.append(node_idx)
    
    print(f"Found {len(solid_tile_nodes)} nodes in solid tiles: {solid_tile_nodes[:10]}...")
    
    if solid_tile_nodes:
        print(f"\nAnalyzing first solid tile node:")
        analyze_sub_cell_node(solid_tile_nodes[0])


if __name__ == "__main__":
    debug_coordinate_system()