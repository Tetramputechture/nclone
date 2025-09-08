#!/usr/bin/env python3
"""
Debug what types of edges are being created in the graph
"""

import os
import sys
import numpy as np

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.common import EdgeType

def debug_edge_types():
    print("=" * 60)
    print("EDGE TYPES DEBUG")
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
    
    # Count edge types
    edge_type_counts = {}
    
    for edge_idx in range(graph.num_edges):
        if graph.edge_mask[edge_idx] == 1:  # Valid edge
            edge_type = graph.edge_types[edge_idx]
            
            # Convert to EdgeType name if possible
            try:
                edge_type_name = EdgeType(edge_type).name
            except:
                edge_type_name = f"UNKNOWN_{edge_type}"
            
            edge_type_counts[edge_type_name] = edge_type_counts.get(edge_type_name, 0) + 1
    
    print(f"\nEdge type distribution:")
    total_edges = sum(edge_type_counts.values())
    for edge_type, count in sorted(edge_type_counts.items()):
        percentage = (count / total_edges) * 100
        print(f"  {edge_type:20s}: {count:4d} edges ({percentage:5.1f}%)")
    
    # Check if any jump/fall edges exist
    jump_fall_types = ['JUMP', 'FALL']
    jump_fall_count = sum(edge_type_counts.get(edge_type, 0) for edge_type in jump_fall_types)
    
    if jump_fall_count == 0:
        print(f"\n❌ NO JUMP/FALL EDGES FOUND!")
        print(f"This explains why there are isolated nodes - only WALK edges are being created.")
    else:
        print(f"\n✅ Found {jump_fall_count} jump/fall edges")
    
    # Sample some edges to see their properties
    print(f"\nSample edge analysis:")
    sample_count = 0
    for edge_idx in range(min(20, graph.num_edges)):  # Check first 20 edges
        if graph.edge_mask[edge_idx] == 1 and sample_count < 5:
            src = graph.edge_index[0, edge_idx]
            dst = graph.edge_index[1, edge_idx]
            edge_type = graph.edge_types[edge_idx]
            
            src_x = graph.node_features[src, 0]
            src_y = graph.node_features[src, 1]
            dst_x = graph.node_features[dst, 0]
            dst_y = graph.node_features[dst, 1]
            
            distance = ((src_x - dst_x)**2 + (src_y - dst_y)**2)**0.5
            
            try:
                edge_type_name = EdgeType(edge_type).name
            except:
                edge_type_name = f"UNKNOWN_{edge_type}"
            
            print(f"  Edge {edge_idx}: {src} -> {dst} ({edge_type_name})")
            print(f"    From ({src_x:.1f}, {src_y:.1f}) to ({dst_x:.1f}, {dst_y:.1f}), distance {distance:.1f}")
            
            sample_count += 1
    
    # Check for isolated nodes (nodes with 0 edges)
    isolated_nodes = []
    for node_idx in range(graph.num_nodes):
        if graph.node_mask[node_idx] == 1:  # Valid node
            edge_count = 0
            for edge_idx in range(graph.num_edges):
                if graph.edge_mask[edge_idx] == 1:
                    src = graph.edge_index[0, edge_idx]
                    dst = graph.edge_index[1, edge_idx]
                    if src == node_idx or dst == node_idx:
                        edge_count += 1
            
            if edge_count == 0:
                node_x = graph.node_features[node_idx, 0]
                node_y = graph.node_features[node_idx, 1]
                isolated_nodes.append((node_idx, node_x, node_y))
    
    print(f"\nIsolated nodes (0 edges): {len(isolated_nodes)}")
    if len(isolated_nodes) > 0:
        print(f"Sample isolated nodes:")
        for i, (node_idx, x, y) in enumerate(isolated_nodes[:10]):
            print(f"  Node {node_idx} at ({x:.1f}, {y:.1f})")
        
        if len(isolated_nodes) > 10:
            print(f"  ... and {len(isolated_nodes) - 10} more")

if __name__ == "__main__":
    debug_edge_types()
    print("\n🎉 EDGE TYPES DEBUG COMPLETE!")