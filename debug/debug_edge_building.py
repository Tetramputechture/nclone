#!/usr/bin/env python3
"""
Debug edge building to understand ninja connection issues.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.edge_building import EdgeBuilder
from nclone.graph.feature_extraction import FeatureExtractor
from nclone.graph.common import SUB_CELL_SIZE, SUB_GRID_WIDTH, SUB_GRID_HEIGHT, NodeType
from nclone.constants.entity_types import EntityType
import numpy as np
import math


def debug_edge_building():
    """Debug edge building process."""
    print("=" * 80)
    print("DEBUGGING EDGE BUILDING")
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
    
    # Create sub-grid node map manually (like in graph construction)
    sub_grid_node_map = {}
    node_count = 0
    
    for sub_row in range(SUB_GRID_HEIGHT):
        for sub_col in range(SUB_GRID_WIDTH):
            node_idx = node_count
            sub_grid_node_map[(sub_row, sub_col)] = node_idx
            node_count += 1
    
    print(f"Created sub_grid_node_map with {len(sub_grid_node_map)} entries")
    
    # Create ninja entity node
    ninja_node_idx = node_count
    ninja_entity = {
        'type': EntityType.NINJA,
        'x': ninja_pos[0],
        'y': ninja_pos[1],
        'is_ninja': True
    }
    entity_nodes = [(ninja_node_idx, ninja_entity)]
    
    print(f"Ninja entity node: {ninja_node_idx}")
    
    # Now simulate the edge building process for the ninja
    ninja_x, ninja_y = ninja_pos
    ninja_sub_col = int(ninja_x // SUB_CELL_SIZE)
    ninja_sub_row = int(ninja_y // SUB_CELL_SIZE)
    
    print(f"Ninja sub-grid position: ({ninja_sub_row}, {ninja_sub_col})")
    
    # Check what's in the sub_grid_node_map around the ninja
    print(f"\nChecking sub_grid_node_map around ninja:")
    
    connections_made = []
    
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            nearby_row = ninja_sub_row + dr
            nearby_col = ninja_sub_col + dc
            
            print(f"  Checking sub-grid ({nearby_row}, {nearby_col}):")
            
            # Check if in map
            if (nearby_row, nearby_col) in sub_grid_node_map:
                grid_node_idx = sub_grid_node_map[(nearby_row, nearby_col)]
                print(f"    ✅ Found in map: node {grid_node_idx}")
                
                # Calculate distance
                grid_x = nearby_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
                grid_y = nearby_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
                distance = math.sqrt((ninja_x - grid_x) ** 2 + (ninja_y - grid_y) ** 2)
                
                print(f"    Grid position: ({grid_x}, {grid_y})")
                print(f"    Distance: {distance:.1f} pixels")
                
                # Check distance threshold
                if distance <= SUB_CELL_SIZE * 1.5:
                    print(f"    ✅ Within connection range ({SUB_CELL_SIZE * 1.5})")
                    connections_made.append((grid_node_idx, grid_x, grid_y, distance))
                else:
                    print(f"    ❌ Outside connection range ({SUB_CELL_SIZE * 1.5})")
            else:
                print(f"    ❌ Not found in sub_grid_node_map")
    
    print(f"\nExpected connections: {len(connections_made)}")
    for grid_node_idx, grid_x, grid_y, distance in connections_made:
        print(f"  Node {grid_node_idx}: ({grid_x}, {grid_y}) at distance {distance:.1f}")
    
    # Now let's see what the actual edge builder does
    print(f"\n" + "="*50)
    print("RUNNING ACTUAL EDGE BUILDER")
    print("="*50)
    
    # Build the actual graph to compare
    graph_builder = HierarchicalGraphBuilder()
    hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
    graph_data = hierarchical_data.sub_cell_graph
    
    # Find ninja node in actual graph
    ninja_node_actual = None
    for node_idx in range(graph_data.num_nodes):
        if graph_data.node_mask[node_idx] == 0:
            continue
        
        if hasattr(graph_data, 'node_types') and graph_data.node_types[node_idx] == NodeType.ENTITY:
            # Check if this is the ninja node by position
            node_features = graph_data.node_features[node_idx]
            node_x = node_features[0]
            node_y = node_features[1]
            
            if abs(node_x - ninja_x) < 0.1 and abs(node_y - ninja_y) < 0.1:
                ninja_node_actual = node_idx
                break
    
    if ninja_node_actual is not None:
        print(f"Found actual ninja node: {ninja_node_actual}")
        
        # Find its connections
        actual_connections = []
        for edge_idx in range(graph_data.num_edges):
            if graph_data.edge_mask[edge_idx] == 0:
                continue
            
            src = int(graph_data.edge_index[0, edge_idx])
            dst = int(graph_data.edge_index[1, edge_idx])
            
            if src == ninja_node_actual:
                # Get destination position
                dst_features = graph_data.node_features[dst]
                dst_x = dst_features[0]
                dst_y = dst_features[1]
                distance = math.sqrt((ninja_x - dst_x) ** 2 + (ninja_y - dst_y) ** 2)
                actual_connections.append((dst, dst_x, dst_y, distance))
        
        print(f"Actual connections: {len(actual_connections)}")
        for dst, dst_x, dst_y, distance in actual_connections:
            print(f"  Node {dst}: ({dst_x}, {dst_y}) at distance {distance:.1f}")
        
        # Compare expected vs actual
        print(f"\nComparison:")
        print(f"Expected {len(connections_made)} connections, got {len(actual_connections)}")
        
        if len(connections_made) != len(actual_connections):
            print("❌ Connection count mismatch!")
        else:
            print("✅ Connection count matches")


if __name__ == '__main__':
    debug_edge_building()