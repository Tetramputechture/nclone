#!/usr/bin/env python3
"""
Debug script to investigate node positioning issues.
"""

import os
import sys
import numpy as np

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.common import GraphData, NodeType, EdgeType
from nclone.graph.level_data import LevelData
from nclone.constants.entity_types import EntityType
from nclone.constants import TILE_PIXEL_SIZE
from nclone.constants.physics_constants import FULL_MAP_WIDTH_PX, FULL_MAP_HEIGHT_PX


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
        level_id='debug_node_positions'
    )
    
    return level_data, entities


def debug_node_positions():
    """Debug node positioning in detail."""
    print("=== DEBUGGING NODE POSITIONS ===")
    
    # Create test level
    level_data, entities = create_simple_level_with_switch_door()
    print(f"Created level with {len(entities)} entities:")
    for i, entity in enumerate(entities):
        print(f"  Entity {i}: type={entity['type']}, x={entity['x']}, y={entity['y']}")
    
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
    
    # Find entity nodes
    print(f"\n=== ANALYZING ENTITY NODES ===")
    
    # Calculate sub-grid nodes count
    from nclone.graph.common import SUB_GRID_WIDTH, SUB_GRID_HEIGHT
    sub_grid_nodes_count = SUB_GRID_WIDTH * SUB_GRID_HEIGHT
    print(f"Sub-grid nodes: {sub_grid_nodes_count}")
    
    entity_nodes = []
    for node_idx in range(graph_data.num_nodes):
        if graph_data.node_mask[node_idx] == 1:
            node_type = NodeType(graph_data.node_types[node_idx])
            if node_type == NodeType.ENTITY:
                entity_nodes.append(node_idx)
    
    print(f"Found {len(entity_nodes)} entity nodes: {entity_nodes}")
    
    # Analyze each entity node
    for node_idx in entity_nodes:
        print(f"\n--- Entity Node {node_idx} ---")
        node_features = graph_data.node_features[node_idx]
        print(f"Feature vector length: {len(node_features)}")
        
        # Extract position using the same logic as visualization
        tile_type_dim = 38
        entity_type_dim = 30
        state_offset = tile_type_dim + 4 + entity_type_dim
        
        print(f"State offset: {state_offset}")
        print(f"Feature vector has {len(node_features)} elements")
        
        if len(node_features) > state_offset + 2:
            norm_x = float(node_features[state_offset + 1])
            norm_y = float(node_features[state_offset + 2])
            print(f"Normalized position: ({norm_x}, {norm_y})")
            
            # Denormalize
            x = norm_x * float(FULL_MAP_WIDTH_PX)
            y = norm_y * float(FULL_MAP_HEIGHT_PX)
            print(f"Denormalized position: ({x}, {y})")
            
            # Show some context around the position features
            print(f"Features around position:")
            start_idx = max(0, state_offset - 2)
            end_idx = min(len(node_features), state_offset + 6)
            for i in range(start_idx, end_idx):
                marker = " <-- X" if i == state_offset + 1 else " <-- Y" if i == state_offset + 2 else ""
                print(f"  [{i}]: {node_features[i]:.6f}{marker}")
        else:
            print(f"Feature vector too short! Expected > {state_offset + 2}")
    
    # Check functional edges
    print(f"\n=== ANALYZING FUNCTIONAL EDGES ===")
    functional_edges = []
    for i in range(graph_data.num_edges):
        if graph_data.edge_mask[i] == 1 and graph_data.edge_types[i] == EdgeType.FUNCTIONAL:
            src_node = graph_data.edge_index[0, i]
            dst_node = graph_data.edge_index[1, i]
            functional_edges.append((i, src_node, dst_node))
    
    print(f"Found {len(functional_edges)} functional edges:")
    for edge_idx, src, dst in functional_edges:
        print(f"  Edge {edge_idx}: Node {src} -> Node {dst}")


if __name__ == "__main__":
    debug_node_positions()