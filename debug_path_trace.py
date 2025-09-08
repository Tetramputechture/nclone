#!/usr/bin/env python3
"""
Trace the connectivity path from ninja to switch step by step
"""

import os
import sys
import numpy as np
from collections import deque

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder

def trace_connectivity():
    print("=" * 60)
    print("PATH CONNECTIVITY TRACE")
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
    
    # Find ninja and switch nodes
    ninja_node = None
    min_ninja_dist = float('inf')
    for node_idx in range(graph.num_nodes):
        if graph.node_mask[node_idx] == 1:
            node_x = graph.node_features[node_idx, 0]
            node_y = graph.node_features[node_idx, 1]
            dist = ((node_x - ninja_position[0])**2 + (node_y - ninja_position[1])**2)**0.5
            if dist < min_ninja_dist:
                min_ninja_dist = dist
                ninja_node = node_idx
    
    # Find leftmost switch
    leftmost_switch = None
    leftmost_x = float('inf')
    for entity in level_data.entities:
        if entity.get("type") == 6:  # LOCKED_DOOR
            switch_x = entity.get("x", 0)
            if switch_x < leftmost_x:
                leftmost_x = switch_x
                leftmost_switch = entity
    
    switch_node = None
    if leftmost_switch:
        switch_x = leftmost_switch.get("x", 0)
        switch_y = leftmost_switch.get("y", 0)
        min_switch_dist = float('inf')
        for node_idx in range(graph.num_nodes):
            if graph.node_mask[node_idx] == 1:
                node_x = graph.node_features[node_idx, 0]
                node_y = graph.node_features[node_idx, 1]
                dist = ((node_x - switch_x)**2 + (node_y - switch_y)**2)**0.5
                if dist < min_switch_dist:
                    min_switch_dist = dist
                    switch_node = node_idx
    
    print(f"Ninja node: {ninja_node} at ({graph.node_features[ninja_node, 0]:.1f}, {graph.node_features[ninja_node, 1]:.1f})")
    print(f"Switch node: {switch_node} at ({graph.node_features[switch_node, 0]:.1f}, {graph.node_features[switch_node, 1]:.1f})")
    
    # Perform BFS to find reachable nodes from ninja
    print(f"\nPerforming BFS from ninja node...")
    visited = set()
    queue = deque([(ninja_node, 0)])  # (node, distance)
    visited.add(ninja_node)
    reachable_nodes = {ninja_node: 0}
    
    # Track nodes by X coordinate ranges
    x_ranges = {
        "ninja_area": (0, 180),
        "boundary": (180, 220), 
        "mid_area": (220, 350),
        "switch_area": (350, 700)
    }
    
    range_counts = {name: 0 for name in x_ranges}
    range_nodes = {name: [] for name in x_ranges}
    
    max_distance = 10  # Limit BFS depth for performance
    
    while queue and len(visited) < 1000:  # Limit for performance
        current_node, distance = queue.popleft()
        
        if distance >= max_distance:
            continue
            
        current_x = graph.node_features[current_node, 0]
        current_y = graph.node_features[current_node, 1]
        
        # Categorize node by X coordinate
        for range_name, (min_x, max_x) in x_ranges.items():
            if min_x <= current_x < max_x:
                range_counts[range_name] += 1
                if len(range_nodes[range_name]) < 5:  # Keep first 5 examples
                    range_nodes[range_name].append((current_node, current_x, current_y, distance))
                break
        
        # Find neighbors
        for edge_idx in range(graph.num_edges):
            if graph.edge_mask[edge_idx] == 1:
                src = graph.edge_index[0, edge_idx]
                dst = graph.edge_index[1, edge_idx]
                
                next_node = None
                if src == current_node and dst not in visited:
                    next_node = dst
                elif dst == current_node and src not in visited:
                    next_node = src
                
                if next_node is not None:
                    visited.add(next_node)
                    queue.append((next_node, distance + 1))
                    reachable_nodes[next_node] = distance + 1
    
    print(f"BFS reached {len(visited)} nodes")
    
    # Show reachability by area
    print(f"\nReachability by area:")
    for range_name, count in range_counts.items():
        print(f"  {range_name:12s}: {count:3d} nodes")
        for node_idx, x, y, dist in range_nodes[range_name]:
            print(f"    Node {node_idx:3d} at ({x:6.1f}, {y:6.1f}) - distance {dist}")
    
    # Check if switch is reachable
    if switch_node in reachable_nodes:
        print(f"\n✅ SWITCH IS REACHABLE! Distance: {reachable_nodes[switch_node]}")
    else:
        print(f"\n❌ SWITCH IS NOT REACHABLE")
        
        # Find the furthest reachable node in the direction of the switch
        switch_x = graph.node_features[switch_node, 0]
        furthest_x = 0
        furthest_node = None
        
        for node_idx in reachable_nodes:
            node_x = graph.node_features[node_idx, 0]
            if node_x > furthest_x:
                furthest_x = node_x
                furthest_node = node_idx
        
        if furthest_node:
            furthest_y = graph.node_features[furthest_node, 1]
            print(f"Furthest reachable node: {furthest_node} at ({furthest_x:.1f}, {furthest_y:.1f})")
            print(f"Gap to switch: {switch_x - furthest_x:.1f} pixels")
    
    # Analyze connectivity gaps
    print(f"\n🔍 CONNECTIVITY GAP ANALYSIS:")
    
    # Look for nodes just beyond the reachable area
    gap_nodes = []
    for node_idx in range(graph.num_nodes):
        if graph.node_mask[node_idx] == 1 and node_idx not in reachable_nodes:
            node_x = graph.node_features[node_idx, 0]
            node_y = graph.node_features[node_idx, 1]
            
            # Check if it's close to the reachable boundary
            if furthest_x - 50 <= node_x <= furthest_x + 100:  # Within 50px behind to 100px ahead
                gap_nodes.append((node_idx, node_x, node_y))
    
    gap_nodes.sort(key=lambda x: x[1])  # Sort by X coordinate
    
    print(f"Found {len(gap_nodes)} nodes near the connectivity gap:")
    for i, (node_idx, x, y) in enumerate(gap_nodes[:10]):  # Show first 10
        # Check if this node has any connections
        edge_count = 0
        for edge_idx in range(graph.num_edges):
            if graph.edge_mask[edge_idx] == 1:
                src = graph.edge_index[0, edge_idx]
                dst = graph.edge_index[1, edge_idx]
                if src == node_idx or dst == node_idx:
                    edge_count += 1
        
        print(f"  Node {node_idx:3d} at ({x:6.1f}, {y:6.1f}) - {edge_count} edges")

if __name__ == "__main__":
    trace_connectivity()
    print("\n🎉 PATH TRACE COMPLETE!")