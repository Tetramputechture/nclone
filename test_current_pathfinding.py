#!/usr/bin/env python3
"""
Test script to understand current pathfinding behavior and identify issues.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nplay_headless import NPlayHeadless
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.graph.level_data import LevelData
from nclone.constants.entity_types import EntityType

def test_simple_walk():
    """Test pathfinding on simple-walk map."""
    print("=== Testing simple-walk map ===")
    
    try:
        # Load the simple-walk test map using NPlayHeadless
        nplay = NPlayHeadless()
        map_path = os.path.join(os.path.dirname(__file__), 'nclone', 'test_maps', 'simple-walk')
        nplay.load_map(map_path)
        
        # Extract level data
        tiles = nplay.sim.get_tile_array()
        entities = nplay.sim.get_entities()
        
        print(f"Map loaded: {tiles.shape} tiles, {len(entities)} entities")
        
        # Print tile data for debugging
        print(f"Tile array shape: {tiles.shape}")
        print(f"Non-zero tiles: {(tiles != 0).sum()}")
        
        # Print entity information
        print("Entities found:")
        for i, entity in enumerate(entities):
            print(f"  {i}: Type={entity.entity_type}, Pos=({entity.x}, {entity.y})")
        
        # Create level data
        level_data = LevelData(tiles, entities)
        
        # Build hierarchical graph
        graph_builder = HierarchicalGraphBuilder()
        hierarchical_data = graph_builder.build_hierarchical_graph(level_data)
        
        print(f"Graph built with {hierarchical_data.sub_cell_graph.num_nodes} sub-cell nodes")
        
        # Find ninja spawn and exit positions
        ninja_pos = None
        exit_pos = None
        
        for entity in entities:
            if entity.entity_type == EntityType.NINJA:
                ninja_pos = (entity.x, entity.y)
            elif entity.entity_type == EntityType.EXIT:
                exit_pos = (entity.x, entity.y)
        
        if not ninja_pos or not exit_pos:
            print("Could not find ninja spawn or exit positions")
            return
        
        print(f"Ninja spawn: {ninja_pos}")
        print(f"Exit position: {exit_pos}")
        
        # Run pathfinding using hierarchical approach
        pathfinder = PathfindingEngine(level_data=level_data.to_dict(), entities=entities)
        result = pathfinder.find_hierarchical_path(
            hierarchical_data, ninja_pos, exit_pos, PathfindingAlgorithm.DIJKSTRA
        )
        
        print(f"Pathfinding result: Success={result.success}, Cost={result.total_cost:.2f}")
        print(f"Path length: {len(result.path)} nodes")
        print(f"Nodes explored: {result.nodes_explored}")
        
        if result.success and len(result.path) > 1:
            print("\nPath coordinates (first 10):")
            for i, coord in enumerate(result.path_coordinates[:10]):
                print(f"  {i}: {coord}")
            
            print(f"\nEdge types: {result.edge_types}")
            
            # Analyze movement types
            movement_types = {}
            for edge_type in result.edge_types:
                movement_types[edge_type] = movement_types.get(edge_type, 0) + 1
            
            print(f"\nMovement type distribution:")
            for movement_type, count in movement_types.items():
                print(f"  {movement_type}: {count}")
    
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_walk()