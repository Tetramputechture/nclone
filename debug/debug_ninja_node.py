#!/usr/bin/env python3
"""
Debug the ninja entity node to see if it was created and where it is.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine


def debug_ninja_node():
    """Debug the ninja entity node."""
    print("=" * 80)
    print("DEBUGGING NINJA ENTITY NODE")
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
    
    # Look for ninja entity nodes
    ninja_nodes = []
    
    for node_idx in range(graph_data.num_nodes):
        if graph_data.node_mask[node_idx] == 0:
            continue
            
        # Check if this is an entity node
        from nclone.graph.common import NodeType
        if hasattr(graph_data, 'node_types') and graph_data.node_types[node_idx] == NodeType.ENTITY:
            # Get node position
            pathfinding_engine = PathfindingEngine()
            node_pos = pathfinding_engine._get_node_position(graph_data, node_idx)
            
            # Check if it's close to ninja position
            distance = ((node_pos[0] - ninja_pos[0])**2 + (node_pos[1] - ninja_pos[1])**2)**0.5
            
            if distance < 50:  # Within 50 pixels
                ninja_nodes.append((node_idx, node_pos, distance))
    
    ninja_nodes.sort(key=lambda x: x[2])  # Sort by distance
    
    print(f"\nFound {len(ninja_nodes)} entity nodes near ninja position:")
    for i, (node_idx, node_pos, distance) in enumerate(ninja_nodes):
        print(f"  {i+1}. Node {node_idx} at {node_pos} (distance: {distance:.1f})")
        
        # Check if this node has edges
        has_outgoing = False
        has_incoming = False
        
        for edge_idx in range(graph_data.num_edges):
            if graph_data.edge_mask[edge_idx] == 0:
                continue
                
            src = int(graph_data.edge_index[0, edge_idx])
            dst = int(graph_data.edge_index[1, edge_idx])
            
            if src == node_idx:
                has_outgoing = True
            if dst == node_idx:
                has_incoming = True
        
        print(f"      Outgoing edges: {'Yes' if has_outgoing else 'No'}")
        print(f"      Incoming edges: {'Yes' if has_incoming else 'No'}")
    
    # Also check what the pathfinding engine finds as the ninja node
    pathfinding_engine = PathfindingEngine()
    found_ninja_node = pathfinding_engine._find_node_at_position(graph_data, ninja_pos)
    found_ninja_pos = pathfinding_engine._get_node_position(graph_data, found_ninja_node)
    
    print(f"\nPathfinding engine finds ninja node: {found_ninja_node} at {found_ninja_pos}")
    
    # Check if the found node is an entity node or sub-grid node
    if hasattr(graph_data, 'node_types'):
        node_type = graph_data.node_types[found_ninja_node]
        print(f"Found ninja node type: {node_type} ({'ENTITY' if node_type == NodeType.ENTITY else 'GRID_CELL' if node_type == NodeType.GRID_CELL else 'OTHER'})")
    
    # Check if there are any ninja entity nodes at all
    entity_node_count = 0
    for node_idx in range(graph_data.num_nodes):
        if graph_data.node_mask[node_idx] == 0:
            continue
        if hasattr(graph_data, 'node_types') and graph_data.node_types[node_idx] == NodeType.ENTITY:
            entity_node_count += 1
    
    print(f"\nTotal entity nodes in graph: {entity_node_count}")


if __name__ == '__main__':
    debug_ninja_node()