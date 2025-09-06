#!/usr/bin/env python3
"""
Debug script to analyze graph visualization issues.

This script creates a simple test level and analyzes the three main issues:
1. Missing functional edges between switches and doors
2. Walkable edges in solid tiles
3. Pathfinding not working on traversable paths
"""

import os
import sys
import numpy as np
import pygame
from typing import Dict, List, Any, Tuple

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.graph_construction import GraphConstructor
from nclone.graph.feature_extraction import FeatureExtractor
from nclone.graph.edge_building import EdgeBuilder
from nclone.graph.visualization import GraphVisualizer, VisualizationConfig
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.graph.common import GraphData, NodeType, EdgeType
from nclone.graph.level_data import LevelData
from nclone.constants.entity_types import EntityType
from nclone.constants import TILE_PIXEL_SIZE


def create_test_level_with_switches():
    """Create a simple test level with switches and doors for debugging."""
    # Create a 10x10 level
    width, height = 10, 10
    tiles = np.zeros((height, width), dtype=int)
    
    # Add some walls to create a simple level structure
    # Top and bottom walls
    tiles[0, :] = 1
    tiles[height-1, :] = 1
    # Left and right walls
    tiles[:, 0] = 1
    tiles[:, width-1] = 1
    
    # Add some internal walls to create interesting geometry
    tiles[3, 2:8] = 1  # Horizontal wall
    tiles[6, 2:8] = 1  # Another horizontal wall
    tiles[2:7, 5] = 1  # Vertical wall with gap
    
    # Create entities: switch and door pair
    entities = [
        {
            'type': EntityType.EXIT_SWITCH,
            'entity_id': 1,
            'x': 2.5 * TILE_PIXEL_SIZE,  # Position in pixels
            'y': 2.5 * TILE_PIXEL_SIZE,
        },
        {
            'type': EntityType.EXIT_DOOR,
            'entity_id': 1,
            'switch_entity_id': 1,  # Links to the switch
            'x': 7.5 * TILE_PIXEL_SIZE,
            'y': 2.5 * TILE_PIXEL_SIZE,
        }
    ]
    
    level_data = LevelData(
        tiles=tiles,
        entities=entities,
        level_id='debug_test_level'
    )
    
    return level_data, entities


def analyze_graph_data(graph_data: GraphData, level_data: LevelData, entities: List[Dict]):
    """Analyze the graph data for the three main issues."""
    print("=== GRAPH ANALYSIS ===")
    print(f"Total nodes: {graph_data.num_nodes}")
    print(f"Total edges: {graph_data.num_edges}")
    
    # Count edge types
    edge_type_counts = {}
    for i in range(graph_data.num_edges):
        if graph_data.edge_mask[i] == 1:
            edge_type = EdgeType(graph_data.edge_types[i])
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
    
    print("\nEdge type counts:")
    for edge_type, count in edge_type_counts.items():
        print(f"  {edge_type.name}: {count}")
    
    # Issue 1: Check for functional edges
    functional_edges = edge_type_counts.get(EdgeType.FUNCTIONAL, 0)
    print(f"\n=== ISSUE 1: FUNCTIONAL EDGES ===")
    print(f"Functional edges found: {functional_edges}")
    if functional_edges == 0:
        print("❌ NO FUNCTIONAL EDGES FOUND - This is the first issue!")
        print("Expected: Functional edges between switches and doors")
    else:
        print("✅ Functional edges are present")
    
    # Issue 2: Check for walkable edges in solid tiles
    print(f"\n=== ISSUE 2: WALKABLE EDGES IN SOLID TILES ===")
    tiles = level_data.tiles
    walk_edges_in_solid = 0
    
    for i in range(graph_data.num_edges):
        if graph_data.edge_mask[i] == 1 and graph_data.edge_types[i] == EdgeType.WALK:
            src_node = graph_data.edge_index[0, i]
            dst_node = graph_data.edge_index[1, i]
            
            # Get node positions (this is simplified - actual implementation may differ)
            src_features = graph_data.node_features[src_node]
            dst_features = graph_data.node_features[dst_node]
            
            # Check if nodes are in solid tiles (this is a simplified check)
            # We need to examine the actual node positions and tile data
            
    print("Checking walkable edges in solid tiles...")
    print("(This requires more detailed analysis of node positions)")
    
    # Issue 3: Check pathfinding
    print(f"\n=== ISSUE 3: PATHFINDING ===")
    pathfinder = PathfindingEngine()
    
    # Find nodes that are actually connected by building adjacency list
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
    
    # Find a node with connections
    connected_nodes = [node for node, neighbors in adjacency.items() if len(neighbors) > 0]
    
    if len(connected_nodes) >= 2:
        start_node = connected_nodes[0]
        # Find a node that's reachable from start_node
        goal_node = adjacency[start_node][0] if adjacency[start_node] else connected_nodes[1]
        
        result = pathfinder.find_shortest_path(
            graph_data, start_node, goal_node, PathfindingAlgorithm.A_STAR
        )
        
        print(f"Pathfinding test (node {start_node} -> {goal_node}):")
        print(f"  Success: {result.success}")
        print(f"  Path length: {len(result.path) if result.success else 0}")
        print(f"  Total cost: {result.total_cost}")
        print(f"  Nodes explored: {result.nodes_explored}")
        
        if not result.success:
            print("❌ PATHFINDING FAILED - This might be the third issue!")
        else:
            print("✅ Pathfinding succeeded")
            
        # Test a longer path
        if len(connected_nodes) >= 10:
            longer_goal = connected_nodes[9]
            result2 = pathfinder.find_shortest_path(
                graph_data, start_node, longer_goal, PathfindingAlgorithm.A_STAR
            )
            print(f"\nLonger pathfinding test (node {start_node} -> {longer_goal}):")
            print(f"  Success: {result2.success}")
            print(f"  Path length: {len(result2.path) if result2.success else 0}")
            print(f"  Total cost: {result2.total_cost}")
            print(f"  Nodes explored: {result2.nodes_explored}")
    else:
        print("❌ NO CONNECTED NODES FOUND - Graph connectivity issue!")


def create_visualization(graph_data: GraphData, level_data: LevelData, entities: List[Dict]):
    """Create a visualization to see the issues."""
    print(f"\n=== CREATING VISUALIZATION ===")
    
    # Initialize pygame
    pygame.init()
    
    # Create visualizer with all edge types enabled
    config = VisualizationConfig(
        show_functional_edges=True,
        show_walk_edges=True,
        show_jump_edges=True,
        show_fall_edges=True,
        show_wall_slide_edges=True,
        show_one_way_edges=True,
        show_nodes=True,
        show_edges=True
    )
    
    visualizer = GraphVisualizer(config)
    
    # Create standalone visualization
    try:
        surface = visualizer.create_standalone_visualization(
            graph_data,
            width=800,
            height=600,
            goal_position=None,
            start_position=None
        )
        
        # Save the visualization
        pygame.image.save(surface, "debug_graph_visualization.png")
        print("✅ Visualization saved as 'debug_graph_visualization.png'")
        
    except Exception as e:
        print(f"❌ Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main debug function."""
    print("=== DEBUGGING GRAPH VISUALIZATION ISSUES ===")
    
    # Create test level
    print("Creating test level with switches and doors...")
    level_data, entities = create_test_level_with_switches()
    
    # Build graph
    print("Building graph...")
    try:
        # Use the hierarchical builder which properly initializes all components
        builder = HierarchicalGraphBuilder()
        
        # Set ninja position in the middle of the level
        ninja_position = (4.5 * TILE_PIXEL_SIZE, 4.5 * TILE_PIXEL_SIZE)
        ninja_velocity = (0.0, 0.0)
        ninja_state = 0
        
        hierarchical_graph_data = builder.build_graph(
            level_data, ninja_position, ninja_velocity, ninja_state
        )
        
        # Get the sub-cell graph for analysis
        graph_data = hierarchical_graph_data.sub_cell_graph
        print(f"✅ Graph built successfully")
        
        # Analyze the graph
        analyze_graph_data(graph_data, level_data, entities)
        
        # Create visualization
        create_visualization(graph_data, level_data, entities)
        
    except Exception as e:
        print(f"❌ Failed to build graph: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== DEBUG COMPLETE ===")


if __name__ == "__main__":
    main()