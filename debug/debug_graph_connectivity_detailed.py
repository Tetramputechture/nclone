#!/usr/bin/env python3
"""
Debug graph connectivity in detail to understand why pathfinding fails.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine
from nclone.graph.common import NodeType, EdgeType
import numpy as np


def debug_graph_connectivity_detailed():
    """Debug graph connectivity in detail."""
    print("=" * 80)
    print("DEBUGGING GRAPH CONNECTIVITY IN DETAIL")
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
    ninja_node_pos = pathfinding_engine._get_node_position(graph_data, ninja_node)
    
    print(f"Ninja node: {ninja_node} at {ninja_node_pos}")
    
    # Check ninja node type
    if hasattr(graph_data, 'node_types'):
        ninja_node_type = graph_data.node_types[ninja_node]
        print(f"Ninja node type: {ninja_node_type} ({'ENTITY' if ninja_node_type == NodeType.ENTITY else 'GRID_CELL'})")
    
    # Find ninja node's neighbors
    ninja_neighbors = []
    
    for edge_idx in range(graph_data.num_edges):
        if graph_data.edge_mask[edge_idx] == 0:
            continue
            
        src = int(graph_data.edge_index[0, edge_idx])
        dst = int(graph_data.edge_index[1, edge_idx])
        
        if src == ninja_node:
            neighbor_pos = pathfinding_engine._get_node_position(graph_data, dst)
            distance = ((neighbor_pos[0] - ninja_node_pos[0])**2 + (neighbor_pos[1] - ninja_node_pos[1])**2)**0.5
            ninja_neighbors.append((dst, neighbor_pos, distance, 'outgoing'))
        elif dst == ninja_node:
            neighbor_pos = pathfinding_engine._get_node_position(graph_data, src)
            distance = ((neighbor_pos[0] - ninja_node_pos[0])**2 + (neighbor_pos[1] - ninja_node_pos[1])**2)**0.5
            ninja_neighbors.append((src, neighbor_pos, distance, 'incoming'))
    
    print(f"\nNinja node has {len(ninja_neighbors)} connected neighbors:")
    for i, (neighbor_node, neighbor_pos, distance, direction) in enumerate(ninja_neighbors[:10]):
        neighbor_type = 'UNKNOWN'
        if hasattr(graph_data, 'node_types'):
            node_type = graph_data.node_types[neighbor_node]
            neighbor_type = 'ENTITY' if node_type == NodeType.ENTITY else 'GRID_CELL'
        
        print(f"  {i+1}. Node {neighbor_node} at {neighbor_pos} ({direction}, {neighbor_type}, distance: {distance:.1f})")
    
    # Check if ninja neighbors have further connections
    print(f"\nChecking connectivity of ninja's neighbors:")
    
    connected_components = set()
    
    for neighbor_node, neighbor_pos, distance, direction in ninja_neighbors[:5]:  # Check first 5 neighbors
        neighbor_connections = 0
        
        for edge_idx in range(graph_data.num_edges):
            if graph_data.edge_mask[edge_idx] == 0:
                continue
                
            src = int(graph_data.edge_index[0, edge_idx])
            dst = int(graph_data.edge_index[1, edge_idx])
            
            if src == neighbor_node or dst == neighbor_node:
                neighbor_connections += 1
        
        print(f"  Neighbor {neighbor_node} has {neighbor_connections} total connections")
        
        # Try to find a path from this neighbor to a distant node
        # Find a target node far from ninja
        target_node = None
        max_distance = 0
        
        for node_idx in range(min(1000, graph_data.num_nodes)):  # Check first 1000 nodes
            if graph_data.node_mask[node_idx] == 0:
                continue
            
            node_pos = pathfinding_engine._get_node_position(graph_data, node_idx)
            distance = ((node_pos[0] - ninja_pos[0])**2 + (node_pos[1] - ninja_pos[1])**2)**0.5
            
            if distance > max_distance and distance > 100:  # At least 100 pixels away
                max_distance = distance
                target_node = node_idx
        
        if target_node is not None:
            target_pos = pathfinding_engine._get_node_position(graph_data, target_node)
            print(f"    Testing path from neighbor {neighbor_node} to distant node {target_node} at {target_pos} (distance: {max_distance:.1f})")
            
            try:
                path_result = pathfinding_engine.find_shortest_path(graph_data, neighbor_node, target_node)
                if path_result and path_result.path:
                    print(f"    ✅ Path found: {len(path_result.path)} nodes")
                    connected_components.add(neighbor_node)
                else:
                    print(f"    ❌ No path found")
            except Exception as e:
                print(f"    ❌ Pathfinding error: {e}")
    
    print(f"\nConnected neighbors: {len(connected_components)} out of {min(5, len(ninja_neighbors))}")
    
    # Try direct pathfinding from ninja to various nodes
    print(f"\nTesting direct pathfinding from ninja node:")
    
    # Test to entity nodes
    entity_nodes = []
    for node_idx in range(graph_data.num_nodes):
        if graph_data.node_mask[node_idx] == 0:
            continue
        if hasattr(graph_data, 'node_types') and graph_data.node_types[node_idx] == NodeType.ENTITY:
            if node_idx != ninja_node:  # Don't include ninja itself
                node_pos = pathfinding_engine._get_node_position(graph_data, node_idx)
                distance = ((node_pos[0] - ninja_pos[0])**2 + (node_pos[1] - ninja_pos[1])**2)**0.5
                entity_nodes.append((node_idx, node_pos, distance))
    
    entity_nodes.sort(key=lambda x: x[2])  # Sort by distance
    
    print(f"Testing pathfinding to {min(5, len(entity_nodes))} nearest entity nodes:")
    
    for i, (entity_node, entity_pos, distance) in enumerate(entity_nodes[:5]):
        print(f"  Test {i+1}: Ninja {ninja_node} -> Entity {entity_node} at {entity_pos} (distance: {distance:.1f})")
        
        try:
            path_result = pathfinding_engine.find_shortest_path(graph_data, ninja_node, entity_node)
            if path_result and path_result.path:
                print(f"    ✅ Path found: {len(path_result.path)} nodes")
            else:
                print(f"    ❌ No path found")
        except Exception as e:
            print(f"    ❌ Pathfinding error: {e}")
    
    # Check overall graph connectivity
    print(f"\nOverall graph connectivity analysis:")
    
    # Count nodes with no edges
    isolated_nodes = 0
    nodes_with_edges = 0
    
    for node_idx in range(graph_data.num_nodes):
        if graph_data.node_mask[node_idx] == 0:
            continue
            
        has_edges = False
        for edge_idx in range(graph_data.num_edges):
            if graph_data.edge_mask[edge_idx] == 0:
                continue
                
            src = int(graph_data.edge_index[0, edge_idx])
            dst = int(graph_data.edge_index[1, edge_idx])
            
            if src == node_idx or dst == node_idx:
                has_edges = True
                break
        
        if has_edges:
            nodes_with_edges += 1
        else:
            isolated_nodes += 1
    
    total_active_nodes = nodes_with_edges + isolated_nodes
    print(f"Total active nodes: {total_active_nodes}")
    print(f"Nodes with edges: {nodes_with_edges}")
    print(f"Isolated nodes: {isolated_nodes}")
    print(f"Connectivity: {nodes_with_edges / total_active_nodes * 100:.1f}%")


if __name__ == '__main__':
    debug_graph_connectivity_detailed()