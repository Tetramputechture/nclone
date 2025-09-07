#!/usr/bin/env python3
"""
Debug script to investigate walkable edges in solid tiles.
"""

import os
import sys
import numpy as np
import pygame

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.visualization import GraphVisualizer, VisualizationConfig
from nclone.graph.common import GraphData, NodeType, EdgeType, SUB_GRID_WIDTH, SUB_GRID_HEIGHT, SUB_CELL_SIZE
from nclone.graph.level_data import LevelData
from nclone.constants.entity_types import EntityType
from nclone.constants import TILE_PIXEL_SIZE
from nclone.constants.physics_constants import FULL_MAP_WIDTH_PX, FULL_MAP_HEIGHT_PX


def create_level_with_solid_tiles():
    """Create a level with solid tiles to test walkable edges."""
    # Create a 10x10 level with various solid patterns
    width, height = 10, 10
    tiles = np.zeros((height, width), dtype=int)
    
    # Add border walls
    tiles[0, :] = 1
    tiles[height-1, :] = 1
    tiles[:, 0] = 1
    tiles[:, width-1] = 1
    
    # Add some solid blocks in the middle
    tiles[3:6, 3:6] = 1  # 3x3 solid block
    tiles[2, 7] = 1      # Single solid tile
    tiles[7, 2:5] = 1    # Horizontal solid line
    
    # No entities for this test
    entities = []
    
    level_data = LevelData(
        tiles=tiles,
        entities=entities,
        level_id='debug_walkable_edges'
    )
    
    return level_data, tiles


def analyze_walkable_edges_in_solid_tiles():
    """Analyze walkable edges that appear in solid tiles."""
    print("=== DEBUGGING WALKABLE EDGES IN SOLID TILES ===")
    
    # Create test level
    level_data, tiles = create_level_with_solid_tiles()
    print(f"Created level with solid tile patterns")
    print("Tile layout:")
    for row in tiles:
        print("".join("█" if tile == 1 else "." for tile in row))
    
    # Build graph
    builder = HierarchicalGraphBuilder()
    ninja_position = (5.5 * TILE_PIXEL_SIZE, 5.5 * TILE_PIXEL_SIZE)
    ninja_velocity = (0.0, 0.0)
    ninja_state = 0
    
    hierarchical_graph_data = builder.build_graph(
        level_data, ninja_position, ninja_velocity, ninja_state
    )
    
    graph_data = hierarchical_graph_data.sub_cell_graph
    print(f"\nGraph built: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Analyze walkable edges
    walkable_edges = []
    for i in range(graph_data.num_edges):
        if graph_data.edge_mask[i] == 1 and graph_data.edge_types[i] == EdgeType.WALK:
            src_node = graph_data.edge_index[0, i]
            dst_node = graph_data.edge_index[1, i]
            walkable_edges.append((i, src_node, dst_node))
    
    print(f"Found {len(walkable_edges)} walkable edges")
    
    # Check which walkable edges are in solid tiles
    problematic_edges = []
    
    def get_sub_cell_position(node_idx):
        """Get sub-cell position from node index."""
        if node_idx >= SUB_GRID_WIDTH * SUB_GRID_HEIGHT:
            return None  # Entity node
        
        sub_row = node_idx // SUB_GRID_WIDTH
        sub_col = node_idx % SUB_GRID_WIDTH
        
        # Convert to tile coordinates (accounting for border)
        tile_x = (sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5) / TILE_PIXEL_SIZE - 1
        tile_y = (sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5) / TILE_PIXEL_SIZE - 1
        
        return (tile_x, tile_y, sub_row, sub_col)
    
    def is_in_solid_tile(tile_x, tile_y):
        """Check if position is in a solid tile."""
        tile_row = int(tile_y)
        tile_col = int(tile_x)
        
        if 0 <= tile_row < tiles.shape[0] and 0 <= tile_col < tiles.shape[1]:
            return tiles[tile_row, tile_col] == 1
        return False
    
    # Check first 100 walkable edges for analysis
    sample_edges = walkable_edges[:100]
    
    for edge_idx, src_node, dst_node in sample_edges:
        src_pos = get_sub_cell_position(src_node)
        dst_pos = get_sub_cell_position(dst_node)
        
        if src_pos is None or dst_pos is None:
            continue  # Skip entity nodes
        
        src_tile_x, src_tile_y, src_sub_row, src_sub_col = src_pos
        dst_tile_x, dst_tile_y, dst_sub_row, dst_sub_col = dst_pos
        
        src_in_solid = is_in_solid_tile(src_tile_x, src_tile_y)
        dst_in_solid = is_in_solid_tile(dst_tile_x, dst_tile_y)
        
        if src_in_solid or dst_in_solid:
            problematic_edges.append({
                'edge_idx': edge_idx,
                'src_node': src_node,
                'dst_node': dst_node,
                'src_pos': (src_tile_x, src_tile_y),
                'dst_pos': (dst_tile_x, dst_tile_y),
                'src_in_solid': src_in_solid,
                'dst_in_solid': dst_in_solid,
                'src_sub_cell': (src_sub_row, src_sub_col),
                'dst_sub_cell': (dst_sub_row, dst_sub_col)
            })
    
    print(f"\nFound {len(problematic_edges)} problematic walkable edges in solid tiles (from sample of {len(sample_edges)}):")
    
    for i, edge in enumerate(problematic_edges[:10]):  # Show first 10
        print(f"  Edge {edge['edge_idx']}: Node {edge['src_node']} -> {edge['dst_node']}")
        print(f"    Source: tile({edge['src_pos'][0]:.1f}, {edge['src_pos'][1]:.1f}) sub_cell({edge['src_sub_cell'][0]}, {edge['src_sub_cell'][1]}) solid={edge['src_in_solid']}")
        print(f"    Dest:   tile({edge['dst_pos'][0]:.1f}, {edge['dst_pos'][1]:.1f}) sub_cell({edge['dst_sub_cell'][0]}, {edge['dst_sub_cell'][1]}) solid={edge['dst_in_solid']}")
    
    if len(problematic_edges) > 10:
        print(f"    ... and {len(problematic_edges) - 10} more")
    
    # Create visualization to see the problem
    print(f"\n=== CREATING VISUALIZATION ===")
    
    config = VisualizationConfig(
        show_walk_edges=True,
        show_functional_edges=False,
        show_jump_edges=False,
        show_fall_edges=False,
        show_wall_slide_edges=False,
        show_one_way_edges=False,
        show_nodes=True,
        show_edges=True
    )
    
    visualizer = GraphVisualizer(config)
    
    pygame.init()
    try:
        surface = visualizer.create_standalone_visualization(
            graph_data,
            width=600,
            height=600,
            goal_position=None,
            start_position=None
        )
        
        pygame.image.save(surface, "debug_walkable_edges_in_solid_tiles.png")
        print("✅ Walkable edges visualization saved as 'debug_walkable_edges_in_solid_tiles.png'")
        
    except Exception as e:
        print(f"❌ Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    analyze_walkable_edges_in_solid_tiles()