#!/usr/bin/env python3
"""
Test pathfinding to nearby targets within ninja's connected component.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine


def test_local_pathfinding():
    """Test pathfinding to nearby targets within ninja's component."""
    print("=" * 80)
    print("TESTING LOCAL PATHFINDING WITHIN NINJA'S COMPONENT")
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
    graph_data = hierarchical_data.sub_cell_graph
    pathfinding_engine = PathfindingEngine()
    
    # Find ninja node
    ninja_node = pathfinding_engine._find_node_at_position(graph_data, ninja_pos)
    print(f"Ninja node: {ninja_node}")
    
    # Find ninja's connected component
    visited = set()
    stack = [ninja_node]
    ninja_component = []
    
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        
        visited.add(current)
        ninja_component.append(current)
        
        # Find neighbors
        for edge_idx in range(graph_data.num_edges):
            if graph_data.edge_mask[edge_idx] == 0:
                continue
            
            src = int(graph_data.edge_index[0, edge_idx])
            dst = int(graph_data.edge_index[1, edge_idx])
            
            if src == current and dst not in visited:
                stack.append(dst)
            elif dst == current and src not in visited:
                stack.append(src)
    
    print(f"Ninja's component has {len(ninja_component)} nodes")
    
    # Test pathfinding to nodes within the component
    print(f"\nTesting pathfinding to nodes within ninja's component:")
    
    # Select some target nodes from the component (excluding ninja itself)
    target_nodes = [node for node in ninja_component if node != ninja_node][:10]
    
    successful_paths = 0
    
    for i, target_node in enumerate(target_nodes, 1):
        target_pos = pathfinding_engine._get_node_position(graph_data, target_node)
        
        # Calculate distance from ninja
        distance = ((ninja_pos[0] - target_pos[0])**2 + (ninja_pos[1] - target_pos[1])**2)**0.5
        
        # Attempt pathfinding
        path_result = pathfinding_engine.find_shortest_path(graph_data, ninja_node, target_node)
        
        if path_result and path_result.success and len(path_result.path) > 0:
            print(f"  Test {i}: Node {target_node} at {target_pos} (dist: {distance:.1f}) -> ✅ Path found ({len(path_result.path)} nodes, cost: {path_result.total_cost:.1f})")
            successful_paths += 1
        else:
            print(f"  Test {i}: Node {target_node} at {target_pos} (dist: {distance:.1f}) -> ❌ No path found")
    
    success_rate = (successful_paths / len(target_nodes)) * 100 if target_nodes else 0
    print(f"\nLocal pathfinding success rate: {successful_paths}/{len(target_nodes)} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("✅ LOCAL PATHFINDING IS WORKING EXCELLENTLY!")
    elif success_rate >= 50:
        print("✅ LOCAL PATHFINDING IS WORKING WELL!")
    else:
        print("❌ LOCAL PATHFINDING NEEDS IMPROVEMENT")
    
    # Test pathfinding to specific nearby empty tile positions
    print(f"\nTesting pathfinding to specific nearby empty tile positions:")
    
    nearby_targets = [
        (129, 429),  # Empty tile node
        (135, 429),  # Empty tile node
        (123, 429),  # Empty tile node
        (141, 429),  # Empty tile node
        (117, 429),  # Empty tile node
    ]
    
    nearby_successful = 0
    
    for i, target_pos in enumerate(nearby_targets, 1):
        target_node = pathfinding_engine._find_node_at_position(graph_data, target_pos)
        
        if target_node is None:
            print(f"  Test {i}: {target_pos} -> ❌ Target node not found")
            continue
        
        # Check if target is in ninja's component
        if target_node not in ninja_component:
            print(f"  Test {i}: {target_pos} -> ❌ Target not in ninja's component")
            continue
        
        # Calculate distance from ninja
        distance = ((ninja_pos[0] - target_pos[0])**2 + (ninja_pos[1] - target_pos[1])**2)**0.5
        
        # Attempt pathfinding
        path_result = pathfinding_engine.find_shortest_path(graph_data, ninja_node, target_node)
        
        if path_result and path_result.success and len(path_result.path) > 0:
            print(f"  Test {i}: {target_pos} (dist: {distance:.1f}) -> ✅ Path found ({len(path_result.path)} nodes, cost: {path_result.total_cost:.1f})")
            nearby_successful += 1
        else:
            print(f"  Test {i}: {target_pos} (dist: {distance:.1f}) -> ❌ No path found")
    
    nearby_success_rate = (nearby_successful / len(nearby_targets)) * 100
    print(f"\nNearby empty tile pathfinding success rate: {nearby_successful}/{len(nearby_targets)} ({nearby_success_rate:.1f}%)")
    
    return {
        'component_size': len(ninja_component),
        'local_success_rate': success_rate,
        'nearby_success_rate': nearby_success_rate
    }


if __name__ == '__main__':
    results = test_local_pathfinding()
    print(f"\nLocal pathfinding test completed. Results: {results}")