#!/usr/bin/env python3
"""
Debug graph connectivity around ninja position to understand pathfinding issues.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine


def debug_graph_connectivity():
    """Debug graph connectivity around ninja position."""
    print("=" * 80)
    print("DEBUGGING GRAPH CONNECTIVITY")
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
    print(f"Level size: {level_data.width}x{level_data.height} tiles")
    
    # Build graph
    graph_builder = HierarchicalGraphBuilder()
    hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
    
    # Use the sub-cell graph for pathfinding
    graph_data = hierarchical_data.sub_cell_graph
    
    print(f"Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Create pathfinding engine
    pathfinding_engine = PathfindingEngine()
    
    # Find ninja node
    ninja_node = pathfinding_engine._find_node_at_position(graph_data, ninja_pos)
    print(f"Ninja node: {ninja_node}")
    
    if ninja_node is not None:
        ninja_node_pos = pathfinding_engine._get_node_position(graph_data, ninja_node)
        print(f"Ninja node position: {ninja_node_pos}")
        
        # Build adjacency list to check connectivity
        adjacency = {}
        for i in range(graph_data.num_edges):
            if graph_data.edge_mask[i] == 0:
                continue
                
            src = int(graph_data.edge_index[0, i])
            dst = int(graph_data.edge_index[1, i])
            
            if src not in adjacency:
                adjacency[src] = []
            adjacency[src].append(dst)
        
        # Check ninja node connectivity
        if ninja_node in adjacency:
            neighbors = adjacency[ninja_node]
            print(f"Ninja node has {len(neighbors)} outgoing edges")
            
            print("\nFirst 10 neighbors:")
            for i, neighbor in enumerate(neighbors[:10]):
                neighbor_pos = pathfinding_engine._get_node_position(graph_data, neighbor)
                distance = ((neighbor_pos[0] - ninja_node_pos[0])**2 + (neighbor_pos[1] - ninja_node_pos[1])**2)**0.5
                print(f"  Neighbor {i+1}: Node {neighbor} at {neighbor_pos} (distance: {distance:.1f})")
        else:
            print("❌ Ninja node has no outgoing edges!")
        
        # Check incoming edges to ninja node
        incoming_count = 0
        for src, dsts in adjacency.items():
            if ninja_node in dsts:
                incoming_count += 1
        
        print(f"Ninja node has {incoming_count} incoming edges")
        
        # Test pathfinding to nearby positions
        print("\n" + "=" * 60)
        print("TESTING PATHFINDING TO NEARBY POSITIONS")
        print("=" * 60)
        
        test_positions = [
            (ninja_pos[0] + 12, ninja_pos[1]),      # Half tile right
            (ninja_pos[0] - 12, ninja_pos[1]),      # Half tile left
            (ninja_pos[0], ninja_pos[1] + 12),      # Half tile down
            (ninja_pos[0], ninja_pos[1] - 12),      # Half tile up
        ]
        
        for i, test_pos in enumerate(test_positions):
            print(f"\nTest {i+1}: Pathfinding to {test_pos}")
            
            # Find target node
            target_node = pathfinding_engine._find_node_at_position(graph_data, test_pos)
            
            if target_node is not None:
                target_pos = pathfinding_engine._get_node_position(graph_data, target_node)
                distance = ((test_pos[0] - target_pos[0])**2 + (test_pos[1] - target_pos[1])**2)**0.5
                print(f"  Target node: {target_node} at {target_pos} (distance from requested: {distance:.1f})")
                
                # Check if target node has edges
                if target_node in adjacency:
                    print(f"  Target node has {len(adjacency[target_node])} outgoing edges")
                else:
                    print("  Target node has no outgoing edges")
                
                # Try pathfinding
                try:
                    path_result = pathfinding_engine.find_shortest_path(graph_data, ninja_node, target_node)
                    
                    if path_result.success:
                        print(f"  ✅ Path found: {len(path_result.path)} nodes, cost {path_result.total_cost:.2f}")
                        
                        # Show first few nodes in path
                        if len(path_result.path) > 1:
                            print("  Path nodes:")
                            for j, node in enumerate(path_result.path[:5]):
                                node_pos = pathfinding_engine._get_node_position(graph_data, node)
                                print(f"    {j+1}. Node {node} at {node_pos}")
                            if len(path_result.path) > 5:
                                print(f"    ... and {len(path_result.path) - 5} more nodes")
                    else:
                        print(f"  ❌ No path found")
                except Exception as e:
                    print(f"  ❌ Pathfinding error: {e}")
            else:
                print(f"  ❌ Could not find target node at {test_pos}")
    
    # Check overall graph connectivity
    print("\n" + "=" * 60)
    print("OVERALL GRAPH CONNECTIVITY ANALYSIS")
    print("=" * 60)
    
    # Count nodes with no edges
    nodes_with_no_edges = 0
    nodes_with_edges = 0
    
    for node_idx in range(graph_data.num_nodes):
        if graph_data.node_mask[node_idx] == 0:
            continue
            
        has_edges = False
        
        # Check outgoing edges
        for i in range(graph_data.num_edges):
            if graph_data.edge_mask[i] == 0:
                continue
            if int(graph_data.edge_index[0, i]) == node_idx:
                has_edges = True
                break
        
        # Check incoming edges if no outgoing edges found
        if not has_edges:
            for i in range(graph_data.num_edges):
                if graph_data.edge_mask[i] == 0:
                    continue
                if int(graph_data.edge_index[1, i]) == node_idx:
                    has_edges = True
                    break
        
        if has_edges:
            nodes_with_edges += 1
        else:
            nodes_with_no_edges += 1
    
    total_active_nodes = nodes_with_edges + nodes_with_no_edges
    print(f"Total active nodes: {total_active_nodes}")
    print(f"Nodes with edges: {nodes_with_edges}")
    print(f"Nodes with no edges: {nodes_with_no_edges}")
    
    if nodes_with_no_edges > 0:
        print(f"⚠️ {nodes_with_no_edges} nodes are isolated (no edges)")
    else:
        print("✅ All nodes have at least one edge")


if __name__ == '__main__':
    debug_graph_connectivity()