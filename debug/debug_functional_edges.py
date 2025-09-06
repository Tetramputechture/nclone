#!/usr/bin/env python3
"""
Debug script specifically for functional edges visualization issue.
"""

import os
import sys
import numpy as np
import pygame
from typing import Dict, List, Any, Tuple

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.visualization import GraphVisualizer, VisualizationConfig
from nclone.graph.common import GraphData, NodeType, EdgeType
from nclone.graph.level_data import LevelData
from nclone.constants.entity_types import EntityType
from nclone.constants import TILE_PIXEL_SIZE


def create_simple_level_with_switch_door():
    """Create a minimal level with just a switch and door."""
    # Create a 5x5 level
    width, height = 5, 5
    tiles = np.zeros((height, width), dtype=int)
    
    # Add border walls
    tiles[0, :] = 1
    tiles[height-1, :] = 1
    tiles[:, 0] = 1
    tiles[:, width-1] = 1
    
    # Create entities: switch and door pair
    entities = [
        {
            'type': EntityType.EXIT_SWITCH,
            'entity_id': 1,
            'x': 1.5 * TILE_PIXEL_SIZE,  # Position in pixels
            'y': 2.5 * TILE_PIXEL_SIZE,
        },
        {
            'type': EntityType.EXIT_DOOR,
            'entity_id': 1,
            'switch_entity_id': 1,  # Links to the switch
            'x': 3.5 * TILE_PIXEL_SIZE,
            'y': 2.5 * TILE_PIXEL_SIZE,
        }
    ]
    
    level_data = LevelData(
        tiles=tiles,
        entities=entities,
        level_id='debug_functional_edges'
    )
    
    return level_data, entities


def debug_functional_edges():
    """Debug functional edges creation and visualization."""
    print("=== DEBUGGING FUNCTIONAL EDGES ===")
    
    # Create test level
    level_data, entities = create_simple_level_with_switch_door()
    print(f"Created level with {len(entities)} entities")
    
    # Build graph
    builder = HierarchicalGraphBuilder()
    ninja_position = (2.5 * TILE_PIXEL_SIZE, 2.5 * TILE_PIXEL_SIZE)
    ninja_velocity = (0.0, 0.0)
    ninja_state = 0
    
    hierarchical_graph_data = builder.build_graph(
        level_data, ninja_position, ninja_velocity, ninja_state
    )
    
    graph_data = hierarchical_graph_data.sub_cell_graph
    print(f"Graph built: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Analyze functional edges
    functional_edges = []
    for i in range(graph_data.num_edges):
        if graph_data.edge_mask[i] == 1 and graph_data.edge_types[i] == EdgeType.FUNCTIONAL:
            src_node = graph_data.edge_index[0, i]
            dst_node = graph_data.edge_index[1, i]
            functional_edges.append((i, src_node, dst_node))
    
    print(f"Found {len(functional_edges)} functional edges:")
    for edge_idx, src, dst in functional_edges:
        print(f"  Edge {edge_idx}: Node {src} -> Node {dst}")
        
        # Get node positions using the same method as visualization
        from nclone.constants.physics_constants import FULL_MAP_WIDTH_PX, FULL_MAP_HEIGHT_PX
        
        def get_entity_position(node_idx):
            node_features = graph_data.node_features[node_idx]
            tile_type_dim = 38
            entity_type_dim = 30
            state_offset = tile_type_dim + 4 + entity_type_dim
            
            if len(node_features) > state_offset + 2:
                norm_x = float(node_features[state_offset + 1])
                norm_y = float(node_features[state_offset + 2])
                x = norm_x * float(FULL_MAP_WIDTH_PX)
                y = norm_y * float(FULL_MAP_HEIGHT_PX)
                return (x, y)
            return (0.0, 0.0)
        
        src_x, src_y = get_entity_position(src)
        dst_x, dst_y = get_entity_position(dst)
        
        print(f"    Source position: ({src_x:.1f}, {src_y:.1f})")
        print(f"    Destination position: ({dst_x:.1f}, {dst_y:.1f})")
    
    # Test visualization configuration
    print(f"\n=== TESTING VISUALIZATION ===")
    
    # Create visualizer with functional edges enabled
    config = VisualizationConfig(
        show_functional_edges=True,
        show_walk_edges=False,  # Disable other edges to focus on functional
        show_jump_edges=False,
        show_fall_edges=False,
        show_wall_slide_edges=False,
        show_one_way_edges=False,
        show_nodes=True,
        show_edges=True
    )
    
    visualizer = GraphVisualizer(config)
    print(f"Visualizer config - show_functional_edges: {config.show_functional_edges}")
    
    # Test the _should_show_edge_type method
    should_show = visualizer._should_show_edge_type(EdgeType.FUNCTIONAL)
    print(f"Should show functional edges: {should_show}")
    
    # Check edge colors
    functional_color = visualizer.edge_colors.get(EdgeType.FUNCTIONAL, None)
    print(f"Functional edge color: {functional_color}")
    
    # Create visualization
    pygame.init()
    try:
        surface = visualizer.create_standalone_visualization(
            graph_data,
            width=400,
            height=400,
            goal_position=None,
            start_position=None
        )
        
        # Save the visualization
        pygame.image.save(surface, "debug_functional_edges_only.png")
        print("✅ Functional edges visualization saved as 'debug_functional_edges_only.png'")
        
        # Now create one with all edges to compare
        config_all = VisualizationConfig(
            show_functional_edges=True,
            show_walk_edges=True,
            show_jump_edges=True,
            show_fall_edges=True,
            show_wall_slide_edges=True,
            show_one_way_edges=True,
            show_nodes=True,
            show_edges=True
        )
        
        visualizer_all = GraphVisualizer(config_all)
        surface_all = visualizer_all.create_standalone_visualization(
            graph_data,
            width=400,
            height=400,
            goal_position=None,
            start_position=None
        )
        
        pygame.image.save(surface_all, "debug_all_edges.png")
        print("✅ All edges visualization saved as 'debug_all_edges.png'")
        
    except Exception as e:
        print(f"❌ Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_functional_edges()