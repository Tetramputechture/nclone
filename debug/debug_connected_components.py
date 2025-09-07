#!/usr/bin/env python3
"""
Analyze connected components in the graph.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine
from nclone.graph.common import NodeType
import numpy as np


def find_connected_components(graph_data):
    """Find connected components using DFS."""
    visited = set()
    components = []
    
    for node_idx in range(graph_data.num_nodes):
        if graph_data.node_mask[node_idx] == 0 or node_idx in visited:
            continue
        
        # Start a new component
        component = []
        stack = [node_idx]
        
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
        
        if component:
            components.append(component)
    
    return components


def debug_connected_components():
    """Analyze connected components."""
    print("=" * 80)
    print("ANALYZING CONNECTED COMPONENTS")
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
    
    print(f"Ninja node: {ninja_node}")
    
    # Find connected components
    print(f"\nFinding connected components...")
    components = find_connected_components(graph_data)
    
    print(f"Found {len(components)} connected components")
    
    # Sort components by size
    components.sort(key=len, reverse=True)
    
    # Analyze largest components
    for i, component in enumerate(components[:10]):
        print(f"\nComponent {i+1}: {len(component)} nodes")
        
        # Check if ninja is in this component
        ninja_in_component = ninja_node in component
        print(f"  Contains ninja: {'✅ YES' if ninja_in_component else '❌ NO'}")
        
        # Count node types in component
        entity_nodes = 0
        grid_nodes = 0
        
        for node_idx in component:
            if hasattr(graph_data, 'node_types'):
                if graph_data.node_types[node_idx] == NodeType.ENTITY:
                    entity_nodes += 1
                else:
                    grid_nodes += 1
        
        print(f"  Entity nodes: {entity_nodes}")
        print(f"  Grid nodes: {grid_nodes}")
        
        # Show some node positions
        if len(component) <= 10:
            print(f"  Node positions:")
            for node_idx in component:
                node_pos = pathfinding_engine._get_node_position(graph_data, node_idx)
                node_type = 'ENTITY' if hasattr(graph_data, 'node_types') and graph_data.node_types[node_idx] == NodeType.ENTITY else 'GRID'
                print(f"    Node {node_idx}: {node_pos} ({node_type})")
    
    # Check if there are any large connected components
    large_components = [c for c in components if len(c) >= 10]
    print(f"\nLarge components (≥10 nodes): {len(large_components)}")
    
    if large_components:
        largest = large_components[0]
        print(f"Largest component has {len(largest)} nodes")
        
        # Test pathfinding within the largest component
        if len(largest) >= 2:
            print(f"\nTesting pathfinding within largest component:")
            
            # Try a few random pairs
            import random
            random.seed(42)
            
            for i in range(min(5, len(largest) // 2)):
                src_node = random.choice(largest)
                tgt_node = random.choice(largest)
                
                if src_node != tgt_node:
                    src_pos = pathfinding_engine._get_node_position(graph_data, src_node)
                    tgt_pos = pathfinding_engine._get_node_position(graph_data, tgt_node)
                    
                    print(f"  Test {i+1}: Node {src_node} -> Node {tgt_node}")
                    print(f"    Positions: {src_pos} -> {tgt_pos}")
                    
                    try:
                        path_result = pathfinding_engine.find_shortest_path(graph_data, src_node, tgt_node)
                        if path_result and path_result.path:
                            print(f"    ✅ Path found: {len(path_result.path)} nodes")
                        else:
                            print(f"    ❌ No path found")
                    except Exception as e:
                        print(f"    ❌ Pathfinding error: {e}")
    
    else:
        print("No large connected components found - graph is highly fragmented")


if __name__ == '__main__':
    debug_connected_components()