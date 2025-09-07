#!/usr/bin/env python3
"""
Debug ninja's connected component to understand isolation.
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


def debug_ninja_component():
    """Debug ninja's connected component."""
    print("=" * 80)
    print("DEBUGGING NINJA'S CONNECTED COMPONENT")
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
    
    # Find ninja node
    ninja_node = pathfinding_engine._find_node_at_position(graph_data, ninja_pos)
    print(f"Ninja node: {ninja_node}")
    
    if ninja_node is None:
        print("❌ Ninja node not found!")
        return
    
    # Find ninja's connected component
    print(f"\nFinding ninja's connected component...")
    ninja_component = find_component_containing_node(graph_data, ninja_node)
    
    print(f"Ninja's component has {len(ninja_component)} nodes")
    
    # Analyze the component
    entity_nodes = []
    grid_nodes = []
    
    for node_idx in ninja_component:
        if hasattr(graph_data, 'node_types') and graph_data.node_types[node_idx] == NodeType.ENTITY:
            entity_nodes.append(node_idx)
        else:
            grid_nodes.append(node_idx)
    
    print(f"  Entity nodes: {len(entity_nodes)}")
    print(f"  Grid nodes: {len(grid_nodes)}")
    
    # Show all nodes in the component
    print(f"\nAll nodes in ninja's component:")
    for node_idx in ninja_component:
        node_pos = pathfinding_engine._get_node_position(graph_data, node_idx)
        node_type = 'ENTITY' if hasattr(graph_data, 'node_types') and graph_data.node_types[node_idx] == NodeType.ENTITY else 'GRID'
        
        # Count edges for this node
        edge_count = 0
        for edge_idx in range(graph_data.num_edges):
            if graph_data.edge_mask[edge_idx] == 0:
                continue
            
            src = int(graph_data.edge_index[0, edge_idx])
            dst = int(graph_data.edge_index[1, edge_idx])
            
            if src == node_idx or dst == node_idx:
                edge_count += 1
        
        print(f"  Node {node_idx}: {node_type} at {node_pos} ({edge_count} edges)")
    
    # Check if any grid nodes in the component have edges to nodes outside the component
    print(f"\nChecking for edges to nodes outside the component:")
    
    external_connections = []
    
    for node_idx in grid_nodes:
        for edge_idx in range(graph_data.num_edges):
            if graph_data.edge_mask[edge_idx] == 0:
                continue
            
            src = int(graph_data.edge_index[0, edge_idx])
            dst = int(graph_data.edge_index[1, edge_idx])
            
            other_node = None
            if src == node_idx and dst not in ninja_component:
                other_node = dst
            elif dst == node_idx and src not in ninja_component:
                other_node = src
            
            if other_node is not None:
                other_pos = pathfinding_engine._get_node_position(graph_data, other_node)
                node_pos = pathfinding_engine._get_node_position(graph_data, node_idx)
                distance = ((node_pos[0] - other_pos[0])**2 + (node_pos[1] - other_pos[1])**2)**0.5
                
                external_connections.append((node_idx, other_node, distance))
    
    if external_connections:
        print(f"Found {len(external_connections)} external connections:")
        for node_idx, other_node, distance in external_connections[:10]:  # Show first 10
            node_pos = pathfinding_engine._get_node_position(graph_data, node_idx)
            other_pos = pathfinding_engine._get_node_position(graph_data, other_node)
            print(f"  Node {node_idx} at {node_pos} -> Node {other_node} at {other_pos} (distance: {distance:.1f})")
    else:
        print("❌ No external connections found - component is completely isolated!")
    
    # Check what tiles the grid nodes are in
    print(f"\nTile analysis for grid nodes in component:")
    for node_idx in grid_nodes:
        node_pos = pathfinding_engine._get_node_position(graph_data, node_idx)
        tile_x = int(node_pos[0] // 24)
        tile_y = int(node_pos[1] // 24)
        
        if 0 <= tile_x < level_data.width and 0 <= tile_y < level_data.height:
            tile_value = level_data.get_tile(tile_y, tile_x)
            tile_status = "empty" if tile_value == 0 else "solid"
            print(f"  Node {node_idx} at {node_pos} -> tile ({tile_x}, {tile_y}) = {tile_value} ({tile_status})")
        else:
            print(f"  Node {node_idx} at {node_pos} -> tile out of bounds")


if __name__ == '__main__':
    debug_ninja_component()