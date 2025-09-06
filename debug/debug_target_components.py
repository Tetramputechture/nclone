#!/usr/bin/env python3
"""
Debug which components contain the pathfinding test targets.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine


def find_component_containing_node(graph_data, target_node):
    """Find the connected component containing a specific node."""
    visited = set()
    stack = [target_node]
    component = []
    
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        
        visited.add(current)
        component.append(current)
        
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
    
    return component


def debug_target_components():
    """Debug which components contain pathfinding targets."""
    print("=" * 80)
    print("DEBUGGING TARGET COMPONENTS")
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
    
    print(f"Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Find ninja node and its component
    ninja_node = pathfinding_engine._find_node_at_position(graph_data, ninja_pos)
    print(f"Ninja node: {ninja_node}")
    
    ninja_component = find_component_containing_node(graph_data, ninja_node)
    print(f"Ninja component size: {len(ninja_component)} nodes")
    
    # Test pathfinding targets
    test_targets = [
        (156, 252),
        (228, 276),
        (204, 300),
        (228, 300),
        (252, 300),
        (156, 324),
        (180, 324),
        (204, 348),
    ]
    
    print(f"\nAnalyzing pathfinding test targets:")
    
    for i, target_pos in enumerate(test_targets, 1):
        target_node = pathfinding_engine._find_node_at_position(graph_data, target_pos)
        
        print(f"\nTest {i}: Target {target_pos}")
        print(f"  Target node: {target_node}")
        
        if target_node is None:
            print(f"  ❌ Target node not found")
            continue
        
        # Check if target is in ninja's component
        if target_node in ninja_component:
            print(f"  ✅ Target is in ninja's component")
        else:
            print(f"  ❌ Target is NOT in ninja's component")
            
            # Find target's component
            target_component = find_component_containing_node(graph_data, target_node)
            print(f"  Target component size: {len(target_component)} nodes")
            
            # Show some nodes in target's component
            print(f"  First 5 nodes in target component:")
            for node_idx in target_component[:5]:
                node_pos = pathfinding_engine._get_node_position(graph_data, node_idx)
                print(f"    Node {node_idx}: {node_pos}")
    
    # Find the largest connected components
    print(f"\nFinding largest connected components:")
    
    all_visited = set()
    components = []
    
    for node_idx in range(graph_data.num_nodes):
        if graph_data.node_mask[node_idx] == 0 or node_idx in all_visited:
            continue
        
        component = find_component_containing_node(graph_data, node_idx)
        components.append(component)
        all_visited.update(component)
    
    # Sort by size
    components.sort(key=len, reverse=True)
    
    print(f"Found {len(components)} connected components")
    print(f"Top 10 largest components:")
    
    for i, component in enumerate(components[:10]):
        contains_ninja = ninja_node in component
        ninja_status = "🥷 NINJA" if contains_ninja else ""
        
        # Check if any test targets are in this component
        target_count = 0
        for target_pos in test_targets:
            target_node = pathfinding_engine._find_node_at_position(graph_data, target_pos)
            if target_node and target_node in component:
                target_count += 1
        
        target_status = f"🎯 {target_count} TARGETS" if target_count > 0 else ""
        
        print(f"  Component {i+1}: {len(component)} nodes {ninja_status} {target_status}")


if __name__ == '__main__':
    debug_target_components()