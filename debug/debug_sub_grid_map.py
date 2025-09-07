#!/usr/bin/env python3
"""
Debug the sub_grid_node_map to understand ninja connection issues.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine
from nclone.graph.common import SUB_CELL_SIZE
import numpy as np


def debug_sub_grid_map():
    """Debug sub_grid_node_map around ninja position."""
    print("=" * 80)
    print("DEBUGGING SUB_GRID_NODE_MAP")
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
    
    # We need to access the sub_grid_node_map from the graph builder
    # Let's modify the graph builder to expose this information
    
    # For now, let's manually check what should be in the map
    ninja_x, ninja_y = ninja_pos
    ninja_sub_col = int(ninja_x // SUB_CELL_SIZE)
    ninja_sub_row = int(ninja_y // SUB_CELL_SIZE)
    
    print(f"Ninja sub-grid position: ({ninja_sub_row}, {ninja_sub_col})")
    
    # Build graph and get pathfinding engine
    graph_builder = HierarchicalGraphBuilder()
    hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
    graph_data = hierarchical_data.sub_cell_graph
    pathfinding_engine = PathfindingEngine()
    
    # Check nodes around ninja position
    print(f"\nChecking nodes around ninja position:")
    
    for dr in range(-3, 4):  # Wider range to see more context
        for dc in range(-3, 4):
            check_row = ninja_sub_row + dr
            check_col = ninja_sub_col + dc
            
            # Calculate pixel position
            pixel_x = check_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            pixel_y = check_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            
            # Find node at this position
            found_node = None
            for node_idx in range(graph_data.num_nodes):
                if graph_data.node_mask[node_idx] == 0:
                    continue
                
                node_pos = pathfinding_engine._get_node_position(graph_data, node_idx)
                if abs(node_pos[0] - pixel_x) < 0.1 and abs(node_pos[1] - pixel_y) < 0.1:
                    found_node = node_idx
                    break
            
            # Calculate distance from ninja
            distance = ((ninja_x - pixel_x)**2 + (ninja_y - pixel_y)**2)**0.5
            
            # Check if this should be connected to ninja
            should_connect = distance <= SUB_CELL_SIZE * 1.5 and abs(dr) <= 1 and abs(dc) <= 1
            
            status = "✅ EXISTS" if found_node is not None else "❌ MISSING"
            connect_status = "🔗 SHOULD CONNECT" if should_connect else ""
            
            print(f"  Sub-grid ({check_row:2d}, {check_col:2d}) -> pixel ({pixel_x:3.0f}, {pixel_y:3.0f}) | dist: {distance:4.1f} | {status} {connect_status}")
            
            if found_node is not None:
                # Check if this node is actually connected to ninja
                ninja_node = pathfinding_engine._find_node_at_position(graph_data, ninja_pos)
                is_connected = False
                
                if ninja_node is not None:
                    for edge_idx in range(graph_data.num_edges):
                        if graph_data.edge_mask[edge_idx] == 0:
                            continue
                        
                        src = int(graph_data.edge_index[0, edge_idx])
                        dst = int(graph_data.edge_index[1, edge_idx])
                        
                        if (src == ninja_node and dst == found_node) or (src == found_node and dst == ninja_node):
                            is_connected = True
                            break
                
                if should_connect:
                    actual_status = "✅ CONNECTED" if is_connected else "❌ NOT CONNECTED"
                    print(f"    Node {found_node}: {actual_status}")


if __name__ == '__main__':
    debug_sub_grid_map()