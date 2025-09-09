#!/usr/bin/env python3
"""
Analyze the current pathfinding issue to understand why it's producing straight-line paths.
"""

import os
import sys
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Set, Any

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.graph.common import GraphData, NodeType, EdgeType
from nclone.constants.physics_constants import TILE_PIXEL_SIZE

def load_doortest_level():
    """Load the doortest level data."""
    doortest_path = '/workspace/nclone/nclone/test_maps/doortest'
    
    # Read level data
    with open(doortest_path, 'r') as f:
        lines = f.readlines()
    
    # Parse level dimensions and data
    level_data = []
    entities = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            continue
        if '|' in line:
            # This is a level row
            row = []
            for char in line:
                if char == '|':
                    continue
                elif char == ' ':
                    row.append(0)  # Empty space
                elif char == '#':
                    row.append(1)  # Wall
                elif char == 'N':
                    row.append(0)  # Ninja start position (empty space)
                    entities.append({
                        'type': 'ninja',
                        'x': len(row) - 1,
                        'y': len(level_data)
                    })
                elif char == 'S':
                    row.append(0)  # Switch (empty space)
                    entities.append({
                        'type': 'switch',
                        'x': len(row) - 1,
                        'y': len(level_data)
                    })
                else:
                    row.append(0)  # Default to empty
            if row:
                level_data.append(row)
    
    return np.array(level_data), entities

def analyze_graph_construction(level_data, entities):
    """Analyze how the graph is being constructed."""
    print("=== ANALYZING GRAPH CONSTRUCTION ===")
    print(f"Level dimensions: {level_data.shape}")
    print(f"Entities found: {len(entities)}")
    
    for entity in entities:
        print(f"  {entity['type']} at ({entity['x']}, {entity['y']})")
    
    # Build graph
    builder = HierarchicalGraphBuilder()
    
    # Convert level data to the format expected by the builder
    level_dict = {
        'tiles': level_data.tolist(),
        'width': level_data.shape[1],
        'height': level_data.shape[0]
    }
    
    try:
        graph_data = builder.build_graph(level_dict, entities)
        print(f"\nGraph built successfully:")
        print(f"  Nodes: {graph_data.num_nodes}")
        print(f"  Edges: {graph_data.num_edges}")
        
        # Analyze node distribution
        active_nodes = np.sum(graph_data.node_mask > 0)
        print(f"  Active nodes: {active_nodes}")
        
        # Analyze edge distribution
        active_edges = np.sum(graph_data.edge_mask > 0)
        print(f"  Active edges: {active_edges}")
        
        # Check edge types
        edge_type_counts = {}
        for i in range(graph_data.num_edges):
            if graph_data.edge_mask[i] > 0:
                edge_type = EdgeType(graph_data.edge_types[i])
                edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        print(f"  Edge types:")
        for edge_type, count in edge_type_counts.items():
            print(f"    {edge_type.name}: {count}")
        
        return graph_data, entities
        
    except Exception as e:
        print(f"Error building graph: {e}")
        import traceback
        traceback.print_exc()
        return None, entities

def analyze_pathfinding(graph_data, entities):
    """Analyze the pathfinding behavior."""
    if graph_data is None:
        print("Cannot analyze pathfinding - no graph data")
        return
    
    print("\n=== ANALYZING PATHFINDING ===")
    
    # Find ninja and switch positions
    ninja_pos = None
    switch_pos = None
    
    for entity in entities:
        if entity['type'] == 'ninja':
            ninja_pos = (entity['x'], entity['y'])
        elif entity['type'] == 'switch':
            switch_pos = (entity['x'], entity['y'])
    
    if ninja_pos is None or switch_pos is None:
        print("Could not find ninja or switch positions")
        return
    
    print(f"Ninja position: {ninja_pos}")
    print(f"Switch position: {switch_pos}")
    
    # Find corresponding nodes in the graph
    ninja_node = find_node_at_position(graph_data, ninja_pos)
    switch_node = find_node_at_position(graph_data, switch_pos)
    
    if ninja_node is None or switch_node is None:
        print(f"Could not find nodes - ninja_node: {ninja_node}, switch_node: {switch_node}")
        return
    
    print(f"Ninja node: {ninja_node}")
    print(f"Switch node: {switch_node}")
    
    # Create pathfinding engine
    engine = PathfindingEngine()
    
    # Test pathfinding with Dijkstra
    print("\nTesting Dijkstra pathfinding...")
    result = engine.find_shortest_path(
        graph_data, ninja_node, switch_node, PathfindingAlgorithm.DIJKSTRA
    )
    
    print(f"Path found: {result.success}")
    print(f"Path length: {len(result.path)}")
    print(f"Total cost: {result.total_cost}")
    print(f"Nodes explored: {result.nodes_explored}")
    
    if result.success:
        print(f"Path nodes: {result.path}")
        print(f"Path coordinates: {result.path_coordinates}")
        
        # Analyze the path - check if it's going through walls
        analyze_path_validity(result.path_coordinates, ninja_pos, switch_pos)

def find_node_at_position(graph_data, pos):
    """Find the node closest to the given position."""
    x, y = pos
    target_pos = (x * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2, 
                  y * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2)
    
    best_node = None
    best_distance = float('inf')
    
    for i in range(graph_data.num_nodes):
        if graph_data.node_mask[i] > 0:
            node_x = graph_data.node_positions[i, 0]
            node_y = graph_data.node_positions[i, 1]
            
            distance = math.sqrt((node_x - target_pos[0])**2 + (node_y - target_pos[1])**2)
            if distance < best_distance:
                best_distance = distance
                best_node = i
    
    return best_node

def analyze_path_validity(path_coordinates, ninja_pos, switch_pos):
    """Analyze if the path is valid (doesn't go through walls)."""
    print("\n=== ANALYZING PATH VALIDITY ===")
    
    if len(path_coordinates) < 2:
        print("Path too short to analyze")
        return
    
    print(f"Path has {len(path_coordinates)} waypoints")
    
    # Check if path is approximately straight line
    start_coord = path_coordinates[0]
    end_coord = path_coordinates[-1]
    
    print(f"Start coordinate: {start_coord}")
    print(f"End coordinate: {end_coord}")
    
    # Calculate direct distance
    direct_distance = math.sqrt(
        (end_coord[0] - start_coord[0])**2 + (end_coord[1] - start_coord[1])**2
    )
    
    # Calculate path distance
    path_distance = 0
    for i in range(len(path_coordinates) - 1):
        curr = path_coordinates[i]
        next_coord = path_coordinates[i + 1]
        segment_distance = math.sqrt(
            (next_coord[0] - curr[0])**2 + (next_coord[1] - curr[1])**2
        )
        path_distance += segment_distance
    
    print(f"Direct distance: {direct_distance:.2f}")
    print(f"Path distance: {path_distance:.2f}")
    print(f"Path efficiency: {direct_distance/path_distance:.2f}")
    
    # If efficiency is close to 1.0, it's likely a straight line
    if direct_distance / path_distance > 0.95:
        print("WARNING: Path appears to be nearly straight line!")
        print("This suggests the pathfinding is not respecting level geometry.")
    else:
        print("Path appears to follow level geometry correctly.")

def main():
    """Main analysis function."""
    print("Loading doortest level...")
    level_data, entities = load_doortest_level()
    
    print(f"Level loaded: {level_data.shape}")
    print("Level preview:")
    for i, row in enumerate(level_data[:10]):  # Show first 10 rows
        row_str = ""
        for val in row:
            row_str += "#" if val == 1 else " "
        print(f"{i:2d}: |{row_str}|")
    
    # Analyze graph construction
    graph_data, entities = analyze_graph_construction(level_data, entities)
    
    # Analyze pathfinding
    analyze_pathfinding(graph_data, entities)

if __name__ == "__main__":
    main()