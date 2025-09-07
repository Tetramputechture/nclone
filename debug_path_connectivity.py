#!/usr/bin/env python3
"""
Debug script to check path connectivity between ninja and entities.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm

def debug_path_connectivity():
    """Debug path connectivity between ninja and entities."""
    print("=" * 80)
    print("DEBUGGING PATH CONNECTIVITY")
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
    ninja_node = None
    min_dist = float('inf')
    for node_idx in range(graph_data.num_nodes):
        node_pos = (graph_data.node_features[node_idx][0], graph_data.node_features[node_idx][1])
        dist = ((node_pos[0] - ninja_pos[0])**2 + (node_pos[1] - ninja_pos[1])**2)**0.5
        if dist < min_dist:
            min_dist = dist
            ninja_node = node_idx
    
    print(f"Ninja node: {ninja_node} at distance {min_dist:.1f}")
    
    # Find entity nodes
    entity_nodes = []
    tile_type_dim = 38
    entity_type_offset = 2 + tile_type_dim + 4
    
    for node_idx in range(graph_data.num_nodes):
        node_features = graph_data.node_features[node_idx]
        position = (node_features[0], node_features[1])
        
        # Check entity type one-hot encoding
        for entity_type in range(30):
            if entity_type_offset + entity_type < len(node_features):
                if node_features[entity_type_offset + entity_type] > 0.5:
                    entity_nodes.append((node_idx, entity_type, position))
                    break
    
    print(f"\nFound {len(entity_nodes)} entity nodes:")
    for node_idx, entity_type, position in entity_nodes:
        print(f"  Node {node_idx}: type={entity_type}, pos={position}")
    
    # Test pathfinding to each entity
    pathfinding_engine = PathfindingEngine()
    
    print(f"\nTesting pathfinding from ninja node {ninja_node}:")
    for node_idx, entity_type, position in entity_nodes:
        if entity_type in [4, 6]:  # EXIT_SWITCH or LOCKED_DOOR
            print(f"\n  Testing path to entity type {entity_type} at node {node_idx}:")
            result = pathfinding_engine.find_shortest_path(
                graph_data, ninja_node, node_idx, PathfindingAlgorithm.A_STAR
            )
            
            if result.success:
                print(f"    ✅ Path found! Length: {len(result.path)} nodes, Cost: {result.cost:.2f}")
                print(f"    Path: {result.path[:5]}{'...' if len(result.path) > 5 else ''}")
            else:
                print(f"    ❌ No path found")
    
    # Check what nodes are reachable from ninja
    print(f"\nChecking reachability from ninja node {ninja_node}:")
    reachable_count = 0
    for target_node in range(graph_data.num_nodes):
        if target_node == ninja_node:
            continue
            
        result = pathfinding_engine.find_shortest_path(
            graph_data, ninja_node, target_node, PathfindingAlgorithm.A_STAR
        )
        
        if result.success:
            reachable_count += 1
    
    print(f"  Reachable nodes: {reachable_count}/{graph_data.num_nodes - 1}")
    
    # Check edges from ninja node
    print(f"\nEdges from ninja node {ninja_node}:")
    edge_count = 0
    for edge_idx in range(graph_data.num_edges):
        if graph_data.edge_index[0][edge_idx] == ninja_node:
            target_node = graph_data.edge_index[1][edge_idx]
            target_pos = (graph_data.node_features[target_node][0], graph_data.node_features[target_node][1])
            print(f"  Edge to node {target_node} at {target_pos}")
            edge_count += 1
    
    print(f"  Total outgoing edges: {edge_count}")

if __name__ == "__main__":
    debug_path_connectivity()