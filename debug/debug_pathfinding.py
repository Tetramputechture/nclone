#!/usr/bin/env python3
"""
Debug script to analyze pathfinding issues in the graph system.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.level_data import LevelData
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.graph.common import EdgeType, NodeType, SUB_CELL_SIZE, SUB_GRID_WIDTH
from nclone.constants import TILE_PIXEL_SIZE


def debug_pathfinding():
    """Debug pathfinding issues in detail."""
    print("=== DEBUGGING PATHFINDING ===")
    
    # Create a simple level with clear paths
    width, height = 10, 10
    tiles = np.zeros((height, width), dtype=int)
    
    # Create a simple corridor with some obstacles
    # Make borders solid
    tiles[0, :] = 1  # Top border
    tiles[-1, :] = 1  # Bottom border
    tiles[:, 0] = 1  # Left border
    tiles[:, -1] = 1  # Right border
    
    # Add some internal obstacles
    tiles[3:7, 3] = 1  # Vertical wall
    tiles[5, 3:7] = 1  # Horizontal wall
    
    print("Tile layout:")
    for y, row in enumerate(tiles):
        print(f"Row {y}: " + "".join("█" if tile == 1 else "." for tile in row))
    
    # No entities for this test
    entities = []
    
    level_data = LevelData(
        tiles=tiles,
        entities=entities,
        level_id='debug_pathfinding'
    )
    
    # Build graph
    builder = HierarchicalGraphBuilder()
    ninja_position = (5 * TILE_PIXEL_SIZE, 5 * TILE_PIXEL_SIZE)  # Center of level
    ninja_velocity = (0.0, 0.0)
    ninja_state = 0
    
    hierarchical_graph_data = builder.build_graph(
        level_data, ninja_position, ninja_velocity, ninja_state
    )
    
    graph_data = hierarchical_graph_data.sub_cell_graph
    print(f"\nGraph built: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Analyze graph connectivity
    print(f"\n=== GRAPH CONNECTIVITY ANALYSIS ===")
    
    # Build adjacency list for connectivity analysis
    adjacency = {}
    for i in range(graph_data.num_nodes):
        if graph_data.node_mask[i] > 0:
            adjacency[i] = []
    
    for i in range(graph_data.num_edges):
        if graph_data.edge_mask[i] > 0:
            src = graph_data.edge_index[0, i]
            dst = graph_data.edge_index[1, i]
            if src in adjacency and dst in adjacency:
                adjacency[src].append(dst)
                adjacency[dst].append(src)  # Undirected graph
    
    # Find connected components using iterative DFS
    visited = set()
    components = []
    
    def iterative_dfs(start_node):
        """Iterative DFS to avoid recursion limit."""
        component = []
        stack = [start_node]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    stack.append(neighbor)
        
        return component
    
    # Sample a subset of nodes to avoid memory issues
    sample_nodes = list(adjacency.keys())[:1000]  # Sample first 1000 nodes
    
    for node in sample_nodes:
        if node not in visited:
            component = iterative_dfs(node)
            if component:
                components.append(component)
    
    print(f"Found {len(components)} connected components:")
    for i, component in enumerate(components):
        print(f"  Component {i}: {len(component)} nodes")
        if len(component) < 20:  # Show small components
            print(f"    Nodes: {sorted(component)}")
    
    # Test pathfinding within the largest component
    if components:
        largest_component = max(components, key=len)
        print(f"\nTesting pathfinding within largest component ({len(largest_component)} nodes)")
        
        pathfinder = PathfindingEngine()
        
        # Test multiple pairs within the same component
        test_pairs = []
        if len(largest_component) >= 2:
            # Test adjacent nodes
            test_pairs.append((largest_component[0], largest_component[1]))
            
            # Test nodes further apart
            if len(largest_component) >= 10:
                test_pairs.append((largest_component[0], largest_component[9]))
            
            # Test nodes at opposite ends
            if len(largest_component) >= 20:
                test_pairs.append((largest_component[0], largest_component[-1]))
        
        for start_node, goal_node in test_pairs:
            print(f"\n--- Testing path: {start_node} -> {goal_node} ---")
            
            # Get node positions for context
            start_pos = get_node_position(start_node)
            goal_pos = get_node_position(goal_node)
            print(f"Start position: sub-cell {start_pos}")
            print(f"Goal position: sub-cell {goal_pos}")
            
            # Test both algorithms
            for algorithm in [PathfindingAlgorithm.A_STAR, PathfindingAlgorithm.DIJKSTRA]:
                result = pathfinder.find_shortest_path(
                    graph_data, start_node, goal_node, algorithm
                )
                
                print(f"{algorithm.name}:")
                print(f"  Success: {result.success}")
                print(f"  Path length: {len(result.path) if result.success else 0}")
                print(f"  Total cost: {result.total_cost:.3f}")
                print(f"  Nodes explored: {result.nodes_explored}")
                
                if result.success and len(result.path) <= 10:
                    print(f"  Path: {result.path}")
                
                if not result.success:
                    print("  ❌ PATHFINDING FAILED")
                else:
                    print("  ✅ Pathfinding succeeded")
    
    # Test pathfinding between different components (should fail)
    if len(components) >= 2:
        print(f"\n=== TESTING PATHFINDING BETWEEN COMPONENTS (should fail) ===")
        start_node = components[0][0]
        goal_node = components[1][0]
        
        pathfinder = PathfindingEngine()
        result = pathfinder.find_shortest_path(
            graph_data, start_node, goal_node, PathfindingAlgorithm.A_STAR
        )
        
        print(f"Path between components {start_node} -> {goal_node}:")
        print(f"  Success: {result.success}")
        print(f"  Total cost: {result.total_cost}")
        print(f"  Nodes explored: {result.nodes_explored}")
        
        if not result.success:
            print("  ✅ Correctly failed (nodes not connected)")
        else:
            print("  ❌ Unexpectedly succeeded")


def get_node_position(node_idx):
    """Get sub-cell position from node index."""
    if node_idx >= SUB_GRID_WIDTH * 92:  # Assuming SUB_GRID_HEIGHT = 92
        return f"Entity node {node_idx}"
    
    sub_row = node_idx // SUB_GRID_WIDTH
    sub_col = node_idx % SUB_GRID_WIDTH
    return f"({sub_row}, {sub_col})"


if __name__ == "__main__":
    debug_pathfinding()