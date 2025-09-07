#!/usr/bin/env python3
"""
Debug ninja connections to understand why it's isolated.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine
from nclone.graph.common import NodeType, SUB_CELL_SIZE
import numpy as np


def debug_ninja_connections():
    """Debug ninja connections."""
    print("=" * 80)
    print("DEBUGGING NINJA CONNECTIONS")
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
    
    # Build graph
    graph_builder = HierarchicalGraphBuilder()
    hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
    
    # Use the sub-cell graph for pathfinding
    graph_data = hierarchical_data.sub_cell_graph
    
    print(f"Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Find ninja node
    pathfinding_engine = PathfindingEngine()
    ninja_node = pathfinding_engine._find_node_at_position(graph_data, ninja_pos)
    
    print(f"Ninja node: {ninja_node}")
    
    if ninja_node is None:
        print("❌ Ninja node not found!")
        return
    
    # Find all edges connected to the ninja node
    ninja_edges = []
    for edge_idx in range(graph_data.num_edges):
        if graph_data.edge_mask[edge_idx] == 0:
            continue
        
        src = int(graph_data.edge_index[0, edge_idx])
        dst = int(graph_data.edge_index[1, edge_idx])
        
        if src == ninja_node:
            ninja_edges.append((edge_idx, src, dst, 'outgoing'))
        elif dst == ninja_node:
            ninja_edges.append((edge_idx, src, dst, 'incoming'))
    
    print(f"\nNinja has {len(ninja_edges)} connected edges:")
    
    for edge_idx, src, dst, direction in ninja_edges:
        other_node = dst if direction == 'outgoing' else src
        other_pos = pathfinding_engine._get_node_position(graph_data, other_node)
        other_type = 'ENTITY' if hasattr(graph_data, 'node_types') and graph_data.node_types[other_node] == NodeType.ENTITY else 'GRID'
        
        distance = ((ninja_pos[0] - other_pos[0])**2 + (ninja_pos[1] - other_pos[1])**2)**0.5
        
        print(f"  Edge {edge_idx}: {direction} to node {other_node}")
        print(f"    Position: {other_pos} ({other_type})")
        print(f"    Distance: {distance:.1f} pixels")
    
    # Calculate expected connections
    print(f"\nExpected ninja connections:")
    ninja_x, ninja_y = ninja_pos
    ninja_sub_col = int(ninja_x // SUB_CELL_SIZE)
    ninja_sub_row = int(ninja_y // SUB_CELL_SIZE)
    
    print(f"Ninja sub-grid position: ({ninja_sub_row}, {ninja_sub_col})")
    
    expected_connections = []
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            nearby_row = ninja_sub_row + dr
            nearby_col = ninja_sub_col + dc
            
            # Calculate pixel position
            grid_x = nearby_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            grid_y = nearby_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            
            # Calculate distance
            distance = ((ninja_x - grid_x) ** 2 + (ninja_y - grid_y) ** 2) ** 0.5
            
            if distance <= SUB_CELL_SIZE * 1.5:
                expected_connections.append((nearby_row, nearby_col, grid_x, grid_y, distance))
                print(f"  Sub-grid ({nearby_row}, {nearby_col}) -> pixel ({grid_x}, {grid_y}), distance: {distance:.1f}")
    
    print(f"\nExpected {len(expected_connections)} connections, actual {len(ninja_edges)} connections")
    
    # Check if expected positions match actual positions
    actual_positions = set()
    for edge_idx, src, dst, direction in ninja_edges:
        other_node = dst if direction == 'outgoing' else src
        other_pos = pathfinding_engine._get_node_position(graph_data, other_node)
        if hasattr(graph_data, 'node_types') and graph_data.node_types[other_node] != NodeType.ENTITY:
            actual_positions.add((round(other_pos[0]), round(other_pos[1])))
    
    expected_positions = set()
    for _, _, grid_x, grid_y, _ in expected_connections:
        expected_positions.add((round(grid_x), round(grid_y)))
    
    print(f"\nPosition comparison:")
    print(f"Expected positions: {sorted(expected_positions)}")
    print(f"Actual positions: {sorted(actual_positions)}")
    
    missing = expected_positions - actual_positions
    extra = actual_positions - expected_positions
    
    if missing:
        print(f"Missing connections: {missing}")
    if extra:
        print(f"Extra connections: {extra}")
    
    # Check if the missing nodes exist in the graph
    if missing:
        print(f"\nChecking if missing nodes exist in graph:")
        for pos_x, pos_y in missing:
            # Find node at this position
            found_node = None
            for node_idx in range(graph_data.num_nodes):
                if graph_data.node_mask[node_idx] == 0:
                    continue
                
                node_pos = pathfinding_engine._get_node_position(graph_data, node_idx)
                if abs(node_pos[0] - pos_x) < 0.1 and abs(node_pos[1] - pos_y) < 0.1:
                    found_node = node_idx
                    break
            
            if found_node is not None:
                print(f"  Position ({pos_x}, {pos_y}): ✅ Node {found_node} exists")
            else:
                print(f"  Position ({pos_x}, {pos_y}): ❌ No node found")


if __name__ == '__main__':
    debug_ninja_connections()