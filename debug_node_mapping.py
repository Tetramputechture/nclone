#!/usr/bin/env python3
"""
Debug node mapping to understand indexing issues.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine
from nclone.graph.common import SUB_CELL_SIZE, SUB_GRID_WIDTH, SUB_GRID_HEIGHT, TILE_PIXEL_SIZE
import numpy as np


def debug_node_mapping():
    """Debug node mapping."""
    print("=" * 80)
    print("DEBUGGING NODE MAPPING")
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
    print(f"Sub-grid dimensions: {SUB_GRID_WIDTH} x {SUB_GRID_HEIGHT}")
    
    # Build graph
    graph_builder = HierarchicalGraphBuilder()
    hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
    graph_data = hierarchical_data.sub_cell_graph
    pathfinding_engine = PathfindingEngine()
    
    print(f"Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Check the first few sub-grid nodes
    print(f"\nChecking first 10 sub-grid nodes:")
    
    for node_idx in range(min(10, graph_data.num_nodes)):
        if graph_data.node_mask[node_idx] == 0:
            continue
        
        # Calculate expected sub-grid position from node index
        expected_sub_row = node_idx // SUB_GRID_WIDTH
        expected_sub_col = node_idx % SUB_GRID_WIDTH
        
        # Calculate expected pixel position
        expected_x_no_offset = expected_sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        expected_y_no_offset = expected_sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        expected_x_with_offset = expected_sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5 + TILE_PIXEL_SIZE
        expected_y_with_offset = expected_sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5 + TILE_PIXEL_SIZE
        
        # Get actual position from pathfinding engine
        actual_pos = pathfinding_engine._get_node_position(graph_data, node_idx)
        
        # Get position from features
        features = graph_data.node_features[node_idx]
        feature_x = features[0]
        feature_y = features[1]
        
        print(f"  Node {node_idx}:")
        print(f"    Expected sub-grid: ({expected_sub_row}, {expected_sub_col})")
        print(f"    Expected pos (no offset): ({expected_x_no_offset}, {expected_y_no_offset})")
        print(f"    Expected pos (with offset): ({expected_x_with_offset}, {expected_y_with_offset})")
        print(f"    Actual pos (pathfinding): {actual_pos}")
        print(f"    Feature pos: ({feature_x}, {feature_y})")
        
        # Check if positions match
        if abs(actual_pos[0] - expected_x_with_offset) < 0.1 and abs(actual_pos[1] - expected_y_with_offset) < 0.1:
            print(f"    ✅ Position matches (with offset)")
        elif abs(actual_pos[0] - feature_x) < 0.1 and abs(actual_pos[1] - feature_y) < 0.1:
            print(f"    ✅ Position matches features")
        else:
            print(f"    ❌ Position mismatch!")
    
    # Check specific nodes around ninja
    print(f"\nChecking nodes around ninja:")
    ninja_x, ninja_y = ninja_pos
    ninja_sub_col = int(ninja_x // SUB_CELL_SIZE)
    ninja_sub_row = int(ninja_y // SUB_CELL_SIZE)
    
    print(f"Ninja sub-grid position: ({ninja_sub_row}, {ninja_sub_col})")
    
    # Check the nodes that should be connected to ninja
    expected_connections = [
        (73, 21),  # Should be node 73 * 176 + 21 = 12869
        (73, 22),  # Should be node 73 * 176 + 22 = 12870
        (74, 21),  # Should be node 74 * 176 + 21 = 13045
        (74, 22),  # Should be node 74 * 176 + 22 = 13046
    ]
    
    for sub_row, sub_col in expected_connections:
        expected_node_idx = sub_row * SUB_GRID_WIDTH + sub_col
        
        print(f"\nSub-grid ({sub_row}, {sub_col}):")
        print(f"  Expected node index: {expected_node_idx}")
        
        if expected_node_idx < graph_data.num_nodes and graph_data.node_mask[expected_node_idx] > 0:
            actual_pos = pathfinding_engine._get_node_position(graph_data, expected_node_idx)
            features = graph_data.node_features[expected_node_idx]
            feature_x = features[0]
            feature_y = features[1]
            
            print(f"  ✅ Node exists")
            print(f"  Actual pos (pathfinding): {actual_pos}")
            print(f"  Feature pos: ({feature_x}, {feature_y})")
            
            # Calculate expected position
            expected_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            expected_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            print(f"  Expected feature pos: ({expected_x}, {expected_y})")
            
            if abs(feature_x - expected_x) < 0.1 and abs(feature_y - expected_y) < 0.1:
                print(f"  ✅ Feature position matches expected")
            else:
                print(f"  ❌ Feature position mismatch!")
        else:
            print(f"  ❌ Node does not exist or is masked")


if __name__ == '__main__':
    debug_node_mapping()