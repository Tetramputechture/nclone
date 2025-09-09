#!/usr/bin/env python3
"""
Minimal analysis of the graph construction to identify the straight-line path issue.
This script examines the graph structure without requiring the full environment.
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
from nclone.graph.level_data import LevelData
from nclone.constants.physics_constants import TILE_PIXEL_SIZE
from nclone.constants.entity_types import EntityType

def create_simple_test_level():
    """Create a simple test level to analyze graph construction."""
    # Create a level with proper padding (border of solid tiles)
    # The coordinate system expects a 1-tile border around the playable area
    level_data = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Top border
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Top wall
        [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],  # Corridor with wall in middle
        [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],  # Corridor with wall in middle
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # Open corridor at bottom
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # Open corridor at bottom
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Bottom wall
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Bottom border
    ])
    
    # Add entities: ninja at (2,2) and switch at (8,2) in the padded coordinate system
    entities = [
        {'type': EntityType.NINJA, 'x': 2 * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2, 'y': 2 * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2},
        {'type': EntityType.EXIT_SWITCH, 'x': 8 * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2, 'y': 2 * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2}
    ]
    
    return level_data, entities

def analyze_graph_edges(graph_data):
    """Analyze the edges in the graph to identify long-distance connections."""
    print("\n=== ANALYZING GRAPH EDGES ===")
    
    edge_lengths = []
    long_edges = []
    
    for edge_idx in range(graph_data.num_edges):
        if graph_data.edge_mask[edge_idx] == 0:
            continue
            
        src_node = graph_data.edge_index[0, edge_idx]
        dst_node = graph_data.edge_index[1, edge_idx]
        edge_type = EdgeType(graph_data.edge_types[edge_idx])
        
        # Get node positions using the pathfinding engine method
        engine = PathfindingEngine()
        src_pos = engine._get_node_position(graph_data, src_node)
        dst_pos = engine._get_node_position(graph_data, dst_node)
        
        # Calculate edge length
        edge_length = math.sqrt((dst_pos[0] - src_pos[0])**2 + (dst_pos[1] - src_pos[1])**2)
        edge_lengths.append(edge_length)
        
        # Flag long edges (more than 1.5 tiles)
        if edge_length > TILE_PIXEL_SIZE * 1.5:
            long_edges.append({
                'src_node': src_node,
                'dst_node': dst_node,
                'src_pos': src_pos,
                'dst_pos': dst_pos,
                'length': edge_length,
                'type': edge_type
            })
    
    print(f"Total active edges: {len(edge_lengths)}")
    if edge_lengths:
        print(f"Average edge length: {np.mean(edge_lengths):.2f}")
        print(f"Max edge length: {np.max(edge_lengths):.2f}")
        print(f"Min edge length: {np.min(edge_lengths):.2f}")
        print(f"Edges longer than 1.5 tiles: {len(long_edges)}")
        
        # Show some long edges
        if long_edges:
            print("\nLong edges (potential problem):")
            for i, edge in enumerate(long_edges[:5]):  # Show first 5
                print(f"  Edge {i+1}: {edge['src_pos']} -> {edge['dst_pos']}")
                print(f"    Length: {edge['length']:.2f} pixels ({edge['length']/TILE_PIXEL_SIZE:.2f} tiles)")
                print(f"    Type: {edge['type'].name}")
    
    return long_edges

def test_pathfinding_on_simple_level():
    """Test pathfinding on a simple level to identify the issue."""
    print("=== TESTING PATHFINDING ON SIMPLE LEVEL ===")
    
    level_data, entities = create_simple_test_level()
    
    print(f"Test level dimensions: {level_data.shape}")
    print("Level layout:")
    for i, row in enumerate(level_data):
        row_str = ""
        for j, val in enumerate(row):
            # Check if ninja or switch is at this tile position
            tile_center_x = j * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2
            tile_center_y = i * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2
            if any(abs(e['x'] - tile_center_x) < TILE_PIXEL_SIZE/2 and abs(e['y'] - tile_center_y) < TILE_PIXEL_SIZE/2 and e['type'] == EntityType.NINJA for e in entities):
                row_str += "N"
            elif any(abs(e['x'] - tile_center_x) < TILE_PIXEL_SIZE/2 and abs(e['y'] - tile_center_y) < TILE_PIXEL_SIZE/2 and e['type'] == EntityType.EXIT_SWITCH for e in entities):
                row_str += "S"
            else:
                row_str += "#" if val == 1 else " "
        print(f"{i}: |{row_str}|")
    
    # Debug the specific ninja position
    print(f"\nDEBUG: Level data at specific positions:")
    print(f"data[1][1] = {level_data[1, 1]} (should be wall)")
    print(f"data[2][2] = {level_data[2, 2]} (should be empty - ninja position)")
    print(f"data[2][8] = {level_data[2, 8]} (should be empty - switch position)")
    
    # Check switch position coordinates
    switch_x = 8 * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2  # 204.0
    switch_y = 2 * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2  # 60.0
    switch_tile_x = int(switch_x // TILE_PIXEL_SIZE)  # 8
    switch_tile_y = int(switch_y // TILE_PIXEL_SIZE)  # 2
    print(f"Switch at ({switch_x}, {switch_y}) -> tile ({switch_tile_x}, {switch_tile_y}) -> data[{switch_tile_y}][{switch_tile_x}] = {level_data[switch_tile_y, switch_tile_x]}")
    
    # Convert to LevelData format expected by graph builder
    level_data_obj = LevelData(
        tiles=level_data,
        entities=entities
    )
    
    # Build graph
    try:
        builder = HierarchicalGraphBuilder()
        ninja_pos = (2 * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2, 2 * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2)
        hierarchical_data = builder.build_graph(level_data_obj, ninja_pos)
        graph_data = hierarchical_data.sub_cell_graph
        
        print(f"\nGraph built:")
        print(f"  Nodes: {graph_data.num_nodes}")
        print(f"  Edges: {graph_data.num_edges}")
        print(f"  Active nodes: {np.sum(graph_data.node_mask > 0)}")
        print(f"  Active edges: {np.sum(graph_data.edge_mask > 0)}")
        
        # Analyze edges
        long_edges = analyze_graph_edges(graph_data)
        
        # Test pathfinding
        ninja_node = find_closest_node(graph_data, (2 * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2, 
                                                   2 * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2))
        switch_node = find_closest_node(graph_data, (8 * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2, 
                                                    2 * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE/2))
        
        if ninja_node is not None and switch_node is not None:
            print(f"\nNinja node: {ninja_node}")
            print(f"Switch node: {switch_node}")
            
            # Check if nodes are connected using GraphData arrays
            ninja_edges = []
            switch_edges = []
            
            for i in range(graph_data.num_edges):
                if graph_data.edge_mask[i] == 1:  # Valid edge
                    from_node = graph_data.edge_index[0, i]
                    to_node = graph_data.edge_index[1, i]
                    
                    if from_node == ninja_node:
                        ninja_edges.append((from_node, to_node, i))
                    if to_node == switch_node:
                        switch_edges.append((from_node, to_node, i))
            
            print(f"\nNode connectivity:")
            print(f"  Ninja node {ninja_node} has {len(ninja_edges)} outgoing edges")
            print(f"  Switch node {switch_node} has {len(switch_edges)} incoming edges")
            
            if len(ninja_edges) > 0:
                print(f"  First few ninja edges:")
                for i, (from_node, to_node, edge_idx) in enumerate(ninja_edges[:5]):
                    edge_type = graph_data.edge_types[edge_idx]
                    print(f"    Edge {i}: {from_node} -> {to_node} (type: {edge_type})")
            
            # Check nodes near the switch position
            switch_pos = (204.0, 60.0)
            nearby_nodes = []
            for node_id in range(graph_data.num_nodes):
                if graph_data.node_mask[node_id] == 1:  # Valid node
                    node_x = graph_data.node_features[node_id, 0]
                    node_y = graph_data.node_features[node_id, 1]
                    distance = ((node_x - switch_pos[0])**2 + (node_y - switch_pos[1])**2)**0.5
                    if distance < 50:  # Within 50 pixels
                        nearby_nodes.append((node_id, node_x, node_y, distance))
            
            nearby_nodes.sort(key=lambda x: x[3])  # Sort by distance
            print(f"\nNodes near switch position {switch_pos}:")
            for node_id, x, y, dist in nearby_nodes[:10]:
                print(f"  Node {node_id}: ({x:.1f}, {y:.1f}) - {dist:.1f}px away")
            
            # Check if nearby nodes have edges
            if len(nearby_nodes) > 1:
                nearby_node_id = nearby_nodes[1][0]  # Second closest (first is switch itself)
                nearby_edges = []
                for i in range(graph_data.num_edges):
                    if graph_data.edge_mask[i] == 1:  # Valid edge
                        from_node = graph_data.edge_index[0, i]
                        to_node = graph_data.edge_index[1, i]
                        
                        if from_node == nearby_node_id or to_node == nearby_node_id:
                            nearby_edges.append((from_node, to_node, i))
                
                print(f"\nNode {nearby_node_id} (near switch) has {len(nearby_edges)} edges:")
                for i, (from_node, to_node, edge_idx) in enumerate(nearby_edges[:5]):
                    edge_type = graph_data.edge_types[edge_idx]
                    print(f"    Edge {i}: {from_node} -> {to_node} (type: {edge_type})")
            
            engine = PathfindingEngine()
            result = engine.find_shortest_path(
                graph_data, ninja_node, switch_node, PathfindingAlgorithm.DIJKSTRA
            )
            
            print(f"\nPathfinding result:")
            print(f"  Success: {result.success}")
            print(f"  Path length: {len(result.path)}")
            print(f"  Total cost: {result.total_cost}")
            
            if result.success and len(result.path_coordinates) > 1:
                # Check if path is straight line
                start_coord = result.path_coordinates[0]
                end_coord = result.path_coordinates[-1]
                
                direct_distance = math.sqrt(
                    (end_coord[0] - start_coord[0])**2 + (end_coord[1] - start_coord[1])**2
                )
                
                path_distance = 0
                for i in range(len(result.path_coordinates) - 1):
                    curr = result.path_coordinates[i]
                    next_coord = result.path_coordinates[i + 1]
                    segment_distance = math.sqrt(
                        (next_coord[0] - curr[0])**2 + (next_coord[1] - curr[1])**2
                    )
                    path_distance += segment_distance
                
                efficiency = direct_distance / path_distance if path_distance > 0 else 0
                print(f"  Direct distance: {direct_distance:.2f}")
                print(f"  Path distance: {path_distance:.2f}")
                print(f"  Efficiency: {efficiency:.3f}")
                
                if efficiency > 0.95:
                    print("  ⚠️  WARNING: Path appears to be nearly straight line!")
                    print("  This suggests the graph has incorrect long-distance connections.")
                else:
                    print("  ✅ Path appears to follow level geometry correctly.")
                    
                # Show path coordinates
                print(f"  Path coordinates:")
                for i, coord in enumerate(result.path_coordinates):
                    tile_x = coord[0] / TILE_PIXEL_SIZE
                    tile_y = coord[1] / TILE_PIXEL_SIZE
                    print(f"    {i}: ({coord[0]:.1f}, {coord[1]:.1f}) = tile ({tile_x:.1f}, {tile_y:.1f})")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def find_closest_node(graph_data, target_pos):
    """Find the node closest to the target position."""
    best_node = None
    best_distance = float('inf')
    engine = PathfindingEngine()
    
    for i in range(graph_data.num_nodes):
        if graph_data.node_mask[i] > 0:
            node_pos = engine._get_node_position(graph_data, i)
            distance = math.sqrt((node_pos[0] - target_pos[0])**2 + (node_pos[1] - target_pos[1])**2)
            if distance < best_distance:
                best_distance = distance
                best_node = i
    
    return best_node

def main():
    """Main analysis function."""
    print("MINIMAL GRAPH ANALYSIS")
    print("=" * 50)
    
    test_pathfinding_on_simple_level()

if __name__ == "__main__":
    main()