#!/usr/bin/env python3
"""
Test connectivity between ninja and leftmost locked door switch.
"""

import os
import sys
import numpy as np
from collections import deque

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.constants.entity_types import EntityType

def test_ninja_switch_connectivity():
    """Test if ninja and leftmost switch are in the same connected component."""
    print("=" * 80)
    print("🔍 TESTING NINJA → SWITCH CONNECTIVITY")
    print("=" * 80)
    
    # Create environment
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42
    )
    env.reset()
    ninja_position = env.nplay_headless.ninja_position()
    level_data = env.level_data
    
    print(f"Ninja position: {ninja_position}")
    
    # Build graph
    builder = HierarchicalGraphBuilder()
    hierarchical_graph = builder.build_graph(level_data, ninja_position)
    graph = hierarchical_graph.sub_cell_graph
    
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Find ninja node
    ninja_node = None
    min_ninja_dist = float('inf')
    
    for node_idx in range(graph.num_nodes):
        if graph.node_mask[node_idx] == 1:  # Valid node
            node_x = graph.node_features[node_idx, 0]
            node_y = graph.node_features[node_idx, 1]
            dist = ((node_x - ninja_position[0])**2 + (node_y - ninja_position[1])**2)**0.5
            if dist < min_ninja_dist:
                min_ninja_dist = dist
                ninja_node = node_idx
    
    print(f"Ninja node: {ninja_node} at distance {min_ninja_dist:.1f}px")
    
    # Find leftmost locked door switch
    locked_door_switches = []
    for entity in level_data.entities:
        if entity.get("type") == EntityType.LOCKED_DOOR:  # Type 6
            entity_x = entity.get("x", 0)
            entity_y = entity.get("y", 0)
            locked_door_switches.append((entity_x, entity_y))
    
    if not locked_door_switches:
        print("❌ No locked door switches found!")
        return False
    
    leftmost_switch = min(locked_door_switches, key=lambda pos: pos[0])
    target_x, target_y = leftmost_switch
    print(f"Target switch: ({target_x}, {target_y})")
    
    # Find target node
    target_node = None
    min_target_dist = float('inf')
    
    for node_idx in range(graph.num_nodes):
        if graph.node_mask[node_idx] == 1:  # Valid node
            node_x = graph.node_features[node_idx, 0]
            node_y = graph.node_features[node_idx, 1]
            dist = ((node_x - target_x)**2 + (node_y - target_y)**2)**0.5
            if dist < min_target_dist:
                min_target_dist = dist
                target_node = node_idx
    
    print(f"Target node: {target_node} at distance {min_target_dist:.1f}px")
    
    if ninja_node is None or target_node is None:
        print("❌ Could not find both nodes!")
        return False
    
    # Build adjacency list
    print("\n🔗 Building adjacency list...")
    adjacency = {}
    for i in range(graph.num_nodes):
        if graph.node_mask[i] > 0:
            adjacency[i] = []
    
    for i in range(graph.num_edges):
        if graph.edge_mask[i] > 0:
            src = graph.edge_index[0, i]
            dst = graph.edge_index[1, i]
            if src in adjacency and dst in adjacency:
                adjacency[src].append(dst)
                adjacency[dst].append(src)  # Undirected graph
    
    print(f"Adjacency list built: {len(adjacency)} nodes")
    
    # BFS to check connectivity
    print(f"\n🚀 Testing connectivity from ninja node {ninja_node} to target node {target_node}...")
    
    visited = set()
    queue = deque([ninja_node])
    visited.add(ninja_node)
    path_parent = {ninja_node: None}
    
    found_target = False
    nodes_explored = 0
    
    while queue and not found_target:
        current = queue.popleft()
        nodes_explored += 1
        
        if current == target_node:
            found_target = True
            break
        
        # Explore neighbors
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                path_parent[neighbor] = current
    
    print(f"Nodes explored: {nodes_explored}")
    print(f"Total reachable nodes: {len(visited)}")
    
    if found_target:
        print("✅ SUCCESS! Ninja and target switch are connected!")
        
        # Reconstruct path
        path = []
        current = target_node
        while current is not None:
            path.append(current)
            current = path_parent.get(current)
        path.reverse()
        
        print(f"Path length: {len(path)} nodes")
        print(f"Path: {path[:5]}...{path[-5:] if len(path) > 10 else path}")
        
        # Show path coordinates
        print("\nPath coordinates:")
        for i, node_idx in enumerate(path[:10]):  # Show first 10 nodes
            node_x = graph.node_features[node_idx, 0]
            node_y = graph.node_features[node_idx, 1]
            print(f"  {i+1}. Node {node_idx}: ({node_x:.1f}, {node_y:.1f})")
        if len(path) > 10:
            print(f"  ... and {len(path) - 10} more nodes")
        
        return True
    else:
        print("❌ FAILED! Ninja and target switch are NOT connected!")
        print(f"Ninja can reach {len(visited)} nodes, but target is not among them")
        
        # Check if target node is valid
        if target_node in adjacency:
            target_neighbors = len(adjacency[target_node])
            print(f"Target node {target_node} has {target_neighbors} neighbors")
            if target_neighbors == 0:
                print("❌ Target node is isolated (no edges)!")
        else:
            print("❌ Target node is not in adjacency list!")
        
        return False

def main():
    """Main function."""
    success = test_ninja_switch_connectivity()
    
    if success:
        print("\n" + "=" * 80)
        print("🎉 CONNECTIVITY TEST PASSED!")
        print("✅ Ninja and leftmost locked door switch are connected")
        print("✅ Pathfinding should work")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ CONNECTIVITY TEST FAILED!")
        print("❌ Ninja and leftmost locked door switch are NOT connected")
        print("❌ Pathfinding will fail")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())