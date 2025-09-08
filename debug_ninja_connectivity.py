#!/usr/bin/env python3
"""
Debug script to analyze ninja connectivity issues.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.common import SUB_CELL_SIZE

def main():
    """Debug ninja connectivity."""
    print("=" * 60)
    print("🔍 DEBUGGING NINJA CONNECTIVITY")
    print("=" * 60)
    
    # Create environment
    env = BasicLevelNoGold(render_mode="rgb_array")
    builder = HierarchicalGraphBuilder()
    
    # Get ninja position - use the known position from doortest map
    ninja_pos = (132, 444)  # Known ninja position from validation output
    
    # Build graph
    graph = builder.build_graph(env.level_data, ninja_pos)
    print(f"🥷 Ninja position: {ninja_pos}")
    
    # Convert to sub-cell coordinates
    ninja_sub_row = int(ninja_pos[1] // SUB_CELL_SIZE)
    ninja_sub_col = int(ninja_pos[0] // SUB_CELL_SIZE)
    print(f"🥷 Ninja sub-cell: ({ninja_sub_col}, {ninja_sub_row})")
    
    # Find nearby nodes in sub-cell graph
    print(f"\n🔍 Looking for nodes near ninja...")
    nearby_nodes = []
    
    sub_graph = graph.sub_cell_graph
    for node_idx in range(sub_graph.num_nodes):
        if sub_graph.node_mask[node_idx] == 0:  # Skip masked nodes
            continue
        node_features = sub_graph.node_features[node_idx]
        # Assuming first two features are x, y coordinates
        node_pos = (node_features[0], node_features[1])
        distance = ((node_pos[0] - ninja_pos[0])**2 + (node_pos[1] - ninja_pos[1])**2)**0.5
        if distance < 50:  # Within 50 pixels
            nearby_nodes.append((node_idx, node_pos, distance))
    
    # Sort by distance
    nearby_nodes.sort(key=lambda x: x[2])
    
    print(f"Found {len(nearby_nodes)} nodes within 50px of ninja:")
    for i, (node_idx, pos, dist) in enumerate(nearby_nodes[:10]):
        sub_row = int(pos[1] // SUB_CELL_SIZE)
        sub_col = int(pos[0] // SUB_CELL_SIZE)
        print(f"  {i+1}. Node {node_idx}: ({pos[0]:.1f}, {pos[1]:.1f}) sub=({sub_col}, {sub_row}) dist={dist:.1f}px")
    
    # Check ninja's edges
    print(f"\n🔗 Checking ninja's edges...")
    ninja_edges = []
    
    for edge_idx in range(sub_graph.num_edges):
        if sub_graph.edge_mask[edge_idx] == 0:  # Skip masked edges
            continue
        src_idx, tgt_idx = sub_graph.edge_index[0, edge_idx], sub_graph.edge_index[1, edge_idx]
        
        # Check if ninja is involved
        if src_idx < sub_graph.num_nodes and tgt_idx < sub_graph.num_nodes:
            src_features = sub_graph.node_features[src_idx]
            tgt_features = sub_graph.node_features[tgt_idx]
            src_pos = (src_features[0], src_features[1])
            tgt_pos = (tgt_features[0], tgt_features[1])
            
            src_dist = ((src_pos[0] - ninja_pos[0])**2 + (src_pos[1] - ninja_pos[1])**2)**0.5
            tgt_dist = ((tgt_pos[0] - ninja_pos[0])**2 + (tgt_pos[1] - ninja_pos[1])**2)**0.5
            
            if src_dist < 5:  # Source is ninja
                ninja_edges.append((edge_idx, src_idx, tgt_idx, tgt_pos, tgt_dist))
            elif tgt_dist < 5:  # Target is ninja
                ninja_edges.append((edge_idx, tgt_idx, src_idx, src_pos, src_dist))
    
    print(f"Found {len(ninja_edges)} edges connected to ninja:")
    for i, (edge_idx, ninja_idx, other_idx, other_pos, dist) in enumerate(ninja_edges[:10]):
        print(f"  {i+1}. Edge {edge_idx}: Ninja {ninja_idx} <-> Node {other_idx} at ({other_pos[0]:.1f}, {other_pos[1]:.1f}) dist={dist:.1f}px")
    
    # Test specific target positions from validation
    test_targets = [
        (129, 429),
        (135, 429), 
        (123, 429),
        (141, 429)
    ]
    
    print(f"\n🎯 Testing pathfinding to validation targets...")
    for target_pos in test_targets:
        # Find closest node to target
        closest_node = None
        min_dist = float('inf')
        
        for node_idx in range(sub_graph.num_nodes):
            if sub_graph.node_mask[node_idx] == 0:  # Skip masked nodes
                continue
            node_features = sub_graph.node_features[node_idx]
            node_pos = (node_features[0], node_features[1])
            dist = ((node_pos[0] - target_pos[0])**2 + (node_pos[1] - target_pos[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                closest_node = (node_idx, node_pos, dist)
        
        if closest_node:
            node_idx, node_pos, dist = closest_node
            print(f"  Target {target_pos} -> closest node {node_idx} at ({node_pos[0]:.1f}, {node_pos[1]:.1f}) dist={dist:.1f}px")
        else:
            print(f"  Target {target_pos} -> no nodes found!")

if __name__ == "__main__":
    main()