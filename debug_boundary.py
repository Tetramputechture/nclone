#!/usr/bin/env python3
"""
Debug connectivity at the boundary between ninja and switch components
"""

import os
import sys
import numpy as np

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder

def debug_boundary():
    print("=" * 60)
    print("BOUNDARY CONNECTIVITY DEBUG")
    print("=" * 60)
    
    # Create environment
    env = BasicLevelNoGold(render_mode="rgb_array", enable_frame_stack=False, enable_debug_overlay=False, eval_mode=False, seed=42)
    env.reset()
    ninja_position = env.nplay_headless.ninja_position()
    level_data = env.level_data
    
    print(f"Ninja position: {ninja_position}")
    
    # Build graph
    builder = HierarchicalGraphBuilder()
    hierarchical_graph = builder.build_graph(level_data, ninja_position)
    graph = hierarchical_graph.sub_cell_graph
    
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Find nodes in the boundary region (x=180-220, y=420-450)
    boundary_nodes = []
    for node_idx in range(graph.num_nodes):
        if graph.node_mask[node_idx] == 1:  # Valid node
            node_x = graph.node_features[node_idx, 0]
            node_y = graph.node_features[node_idx, 1]
            
            # Check if in boundary region
            if 180 <= node_x <= 220 and 420 <= node_y <= 450:
                boundary_nodes.append((node_idx, node_x, node_y))
    
    print(f"\nFound {len(boundary_nodes)} nodes in boundary region (x=180-220, y=420-450):")
    for node_idx, x, y in boundary_nodes:
        print(f"  Node {node_idx}: ({x:.1f}, {y:.1f})")
    
    # Check connectivity of boundary nodes
    print(f"\nChecking connectivity of boundary nodes:")
    for node_idx, x, y in boundary_nodes:
        # Count edges from this node
        edge_count = 0
        connected_nodes = []
        
        for edge_idx in range(graph.num_edges):
            if graph.edge_mask[edge_idx] == 1:
                src = graph.edge_index[0, edge_idx]
                dst = graph.edge_index[1, edge_idx]
                
                if src == node_idx:
                    edge_count += 1
                    dst_x = graph.node_features[dst, 0]
                    dst_y = graph.node_features[dst, 1]
                    connected_nodes.append((dst, dst_x, dst_y))
        
        print(f"  Node {node_idx} at ({x:.1f}, {y:.1f}): {edge_count} edges")
        
        # Show a few connected nodes
        for i, (dst, dst_x, dst_y) in enumerate(connected_nodes[:3]):
            print(f"    -> Node {dst} at ({dst_x:.1f}, {dst_y:.1f})")
        if len(connected_nodes) > 3:
            print(f"    ... and {len(connected_nodes) - 3} more")
    
    # Find nodes just outside the boundary in both directions
    print(f"\nNodes just left of boundary (x=170-180):")
    left_nodes = []
    for node_idx in range(graph.num_nodes):
        if graph.node_mask[node_idx] == 1:
            node_x = graph.node_features[node_idx, 0]
            node_y = graph.node_features[node_idx, 1]
            
            if 170 <= node_x <= 180 and 420 <= node_y <= 450:
                left_nodes.append((node_idx, node_x, node_y))
    
    for node_idx, x, y in left_nodes[:5]:  # Show first 5
        print(f"  Node {node_idx}: ({x:.1f}, {y:.1f})")
    
    print(f"\nNodes just right of boundary (x=220-230):")
    right_nodes = []
    for node_idx in range(graph.num_nodes):
        if graph.node_mask[node_idx] == 1:
            node_x = graph.node_features[node_idx, 0]
            node_y = graph.node_features[node_idx, 1]
            
            if 220 <= node_x <= 230 and 420 <= node_y <= 450:
                right_nodes.append((node_idx, node_x, node_y))
    
    for node_idx, x, y in right_nodes[:5]:  # Show first 5
        print(f"  Node {node_idx}: ({x:.1f}, {y:.1f})")
    
    # Check if any boundary nodes connect to left or right nodes
    print(f"\nChecking cross-boundary connections:")
    cross_connections = 0
    
    for boundary_node_idx, bx, by in boundary_nodes:
        for edge_idx in range(graph.num_edges):
            if graph.edge_mask[edge_idx] == 1:
                src = graph.edge_index[0, edge_idx]
                dst = graph.edge_index[1, edge_idx]
                
                if src == boundary_node_idx:
                    dst_x = graph.node_features[dst, 0]
                    dst_y = graph.node_features[dst, 1]
                    
                    # Check if connects to left side
                    if dst_x < 180:
                        print(f"  Boundary node {boundary_node_idx} ({bx:.1f}, {by:.1f}) -> Left node {dst} ({dst_x:.1f}, {dst_y:.1f})")
                        cross_connections += 1
                    
                    # Check if connects to right side  
                    elif dst_x > 220:
                        print(f"  Boundary node {boundary_node_idx} ({bx:.1f}, {by:.1f}) -> Right node {dst} ({dst_x:.1f}, {dst_y:.1f})")
                        cross_connections += 1
    
    print(f"\nTotal cross-boundary connections: {cross_connections}")
    
    if cross_connections == 0:
        print("❌ NO CROSS-BOUNDARY CONNECTIONS FOUND - This explains the disconnection!")
    else:
        print("✅ Cross-boundary connections exist")

if __name__ == "__main__":
    debug_boundary()
    print("\n🎉 BOUNDARY DEBUG COMPLETE!")