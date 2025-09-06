#!/usr/bin/env python3
"""
Test the graph system with a map that has solid tiles.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from nclone.graph.level_data import LevelData
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine
from nclone.nsim import Simulator
from nclone.sim_config import SimConfig
from nclone.constants import TILE_PIXEL_SIZE
def create_level_data_from_custom_simulator(sim):
    """Extract level data from the loaded simulator (custom version)."""
    # Extract tile data (23 rows x 42 columns grid)
    # Simulator coordinates: x=0-43, y=0-24 (with 1-tile border)
    # Level data coordinates: row=0-22, col=0-41 (without border)
    tiles = np.zeros((23, 42), dtype=int)
    
    for (x, y), tile_id in sim.tile_dic.items():
        # Map simulator coordinates to level data coordinates
        # Simulator uses 0-based coordinates with 1-tile border
        # Level data uses 0-based coordinates without border
        if 1 <= x <= 42 and 1 <= y <= 23:
            tiles[y-1, x-1] = tile_id  # Note: y becomes row, x becomes column
    
    # Extract entities
    entities = []
    for entity_type, entity_list in sim.entity_dic.items():
        for entity in entity_list:
            # Try different attribute names for position
            x_pos = getattr(entity, 'xcoord', getattr(entity, 'x', getattr(entity, 'pos_x', 0)))
            y_pos = getattr(entity, 'ycoord', getattr(entity, 'y', getattr(entity, 'pos_y', 0)))
            
            entities.append({
                'type': entity_type,
                'entity_id': len(entities) + 1,
                'x': x_pos,
                'y': y_pos,
                'switch_id': getattr(entity, 'switch_id', None) if hasattr(entity, 'switch_id') else None
            })
    
    print(f"✅ Extracted level data: {tiles.shape} tiles, {len(entities)} entities")
    return LevelData(tiles=tiles, entities=entities)

def test_solid_tiles_map():
    """Test the graph system with our custom map that has solid tiles."""
    print("=== TESTING GRAPH SYSTEM WITH SOLID TILES MAP ===")
    
    # Load our custom map
    try:
        with open('debug_test_map', 'rb') as f:
            map_bytes = f.read()
        # Convert bytes to list of integers (like bblock_test loader does)
        map_data = [int(b) for b in map_bytes]
        print(f"✅ Loaded custom test map: {len(map_data)} bytes")
    except FileNotFoundError:
        print("❌ Custom test map not found. Run create_test_map.py first.")
        return
    
    # Create simulator and load the map
    sim_config = SimConfig()
    sim = Simulator(sim_config)
    sim.load(map_data)
    
    # Extract level data
    level_data = create_level_data_from_custom_simulator(sim)
    print(f"✅ Extracted level data: {level_data.tiles.shape} tiles, {len(level_data.entities)} entities")
    
    # Check for solid tiles
    unique_tiles = np.unique(level_data.tiles)
    print(f"Unique tile types: {unique_tiles}")
    print(f"Level data shape: {level_data.tiles.shape}")
    
    solid_tiles = np.where(level_data.tiles == 1)
    print(f"Found {len(solid_tiles[0])} solid tiles")
    
    if len(solid_tiles[0]) > 0:
        print("Sample solid tile positions:")
        for i in range(min(10, len(solid_tiles[0]))):
            x, y = solid_tiles[1][i], solid_tiles[0][i]
            print(f"  Solid tile at ({x}, {y})")
        
        # Check coordinate ranges
        max_x = np.max(solid_tiles[1])
        max_y = np.max(solid_tiles[0])
        print(f"Max solid tile coordinates: x={max_x}, y={max_y}")
    
    # Debug: check simulator tile_dic
    print(f"Simulator tile_dic size: {len(sim.tile_dic)}")
    sim_coords = list(sim.tile_dic.keys())
    if sim_coords:
        max_sim_x = max(x for x, y in sim_coords)
        max_sim_y = max(y for x, y in sim_coords)
        print(f"Max simulator coordinates: x={max_sim_x}, y={max_sim_y}")
        
        # Show some solid tiles from simulator
        solid_sim_tiles = [(x, y) for (x, y), tile_id in sim.tile_dic.items() if tile_id == 1]
        print(f"Solid tiles in simulator: {solid_sim_tiles[:10]}")
    
    # Build hierarchical graph
    print("\n=== BUILDING HIERARCHICAL GRAPH ===")
    builder = HierarchicalGraphBuilder()
    # Get ninja position from simulator
    if hasattr(sim, 'ninja') and sim.ninja:
        # Try different attribute names for ninja position
        ninja_x = getattr(sim.ninja, 'xcoord', getattr(sim.ninja, 'pos_x', 120))
        ninja_y = getattr(sim.ninja, 'ycoord', getattr(sim.ninja, 'pos_y', 480))
        ninja_pos = (ninja_x, ninja_y)
    else:
        ninja_pos = (120, 480)  # Default position
    print(f"Ninja position: {ninja_pos}")
    graph_data = builder.build_graph(level_data, ninja_pos)
    
    print(f"✅ Built graph with {graph_data.sub_cell_graph.num_nodes} nodes and {graph_data.sub_cell_graph.num_edges} edges")
    
    # Check for walkable edges in solid tiles
    print("\n=== CHECKING FOR WALKABLE EDGES IN SOLID TILES ===")
    walkable_edges_in_solid = 0
    
    # Get edge data from the GraphData structure
    edge_index = graph_data.sub_cell_graph.edge_index  # Shape: (2, num_edges)
    edge_types = graph_data.sub_cell_graph.edge_types  # Shape: (num_edges,)
    node_features = graph_data.sub_cell_graph.node_features  # Shape: (num_nodes, feature_dim)
    
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge types shape: {edge_types.shape}")
    print(f"Node features shape: {node_features.shape}")
    
    # Import EdgeType to check values
    from nclone.graph.common import EdgeType
    print(f"EdgeType.WALK value: {EdgeType.WALK.value}")
    
    # Iterate through edges
    for i in range(graph_data.sub_cell_graph.num_edges):
        node1_id = edge_index[0, i]
        node2_id = edge_index[1, i]
        edge_type = edge_types[i]
        
        # Check if this is a WALK edge
        if edge_type == EdgeType.WALK.value:
            # Get node positions from features (first 2 elements are x, y coordinates)
            node1_pos = (float(node_features[node1_id, 0]), float(node_features[node1_id, 1]))
            node2_pos = (float(node_features[node2_id, 0]), float(node_features[node2_id, 1]))
            
            # Check if either node is in a solid tile
            node1_tile_x = int(node1_pos[0] // TILE_PIXEL_SIZE)
            node1_tile_y = int(node1_pos[1] // TILE_PIXEL_SIZE)
            node2_tile_x = int(node2_pos[0] // TILE_PIXEL_SIZE)
            node2_tile_y = int(node2_pos[1] // TILE_PIXEL_SIZE)
            
            # Check if tiles are solid (within bounds)
            node1_in_solid = (0 <= node1_tile_y < level_data.height and 0 <= node1_tile_x < level_data.width and
                             level_data.get_tile(node1_tile_y, node1_tile_x) == 1)
            node2_in_solid = (0 <= node2_tile_y < level_data.height and 0 <= node2_tile_x < level_data.width and
                             level_data.get_tile(node2_tile_y, node2_tile_x) == 1)
            
            if node1_in_solid or node2_in_solid:
                walkable_edges_in_solid += 1
                if walkable_edges_in_solid <= 5:  # Show first 5 examples
                    print(f"  WALKABLE edge in solid tile: {node1_pos} -> {node2_pos}")
                    print(f"    Node1 tile: ({node1_tile_x}, {node1_tile_y}) = {level_data.get_tile(node1_tile_y, node1_tile_x) if 0 <= node1_tile_y < level_data.height and 0 <= node1_tile_x < level_data.width else 'OOB'}")
                    print(f"    Node2 tile: ({node2_tile_x}, {node2_tile_y}) = {level_data.get_tile(node2_tile_y, node2_tile_x) if 0 <= node2_tile_y < level_data.height and 0 <= node2_tile_x < level_data.width else 'OOB'}")
    
    print(f"Found {walkable_edges_in_solid} walkable edges in solid tiles")
    
    # Check for functional edges
    print("\n=== CHECKING FOR FUNCTIONAL EDGES ===")
    functional_edges = 0
    
    print(f"EdgeType.FUNCTIONAL value: {EdgeType.FUNCTIONAL.value}")
    
    # Count functional edges
    for i in range(graph_data.sub_cell_graph.num_edges):
        edge_type = edge_types[i]
        
        if edge_type == EdgeType.FUNCTIONAL.value:
            functional_edges += 1
            if functional_edges <= 5:  # Show first 5 examples
                node1_id = edge_index[0, i]
                node2_id = edge_index[1, i]
                node1_pos = (float(node_features[node1_id, 0]), float(node_features[node1_id, 1]))
                node2_pos = (float(node_features[node2_id, 0]), float(node_features[node2_id, 1]))
                print(f"  FUNCTIONAL edge: {node1_pos} -> {node2_pos}")
    
    print(f"Found {functional_edges} functional edges")
    
    # Check if the custom map has switches and doors
    print("\n=== CHECKING ENTITIES IN CUSTOM MAP ===")
    print(f"Total entities: {len(level_data.entities)}")
    for entity in level_data.entities:
        print(f"  Entity: {entity}")
    
    # Test pathfinding
    print("\n=== TESTING PATHFINDING ===")
    pathfinding_engine = PathfindingEngine()
    
    # Try to find a path in an open area (should work)
    start_pos = (120, 480)  # Open area
    end_pos = (200, 480)    # Open area
    
    # Find nodes at these positions
    start_node = pathfinding_engine._find_node_at_position(graph_data.sub_cell_graph, start_pos)
    end_node = pathfinding_engine._find_node_at_position(graph_data.sub_cell_graph, end_pos)
    
    if start_node is not None and end_node is not None:
        print(f"Found start node {start_node} at {start_pos} and end node {end_node} at {end_pos}")
        try:
            path_result = pathfinding_engine.find_shortest_path(graph_data.sub_cell_graph, start_node, end_node)
            if path_result.success:
                print(f"✅ Found path from {start_pos} to {end_pos}: {len(path_result.path)} nodes, cost {path_result.total_cost:.2f}")
                path_found = True
            else:
                print(f"❌ No path found from {start_pos} to {end_pos}")
                path_found = False
        except Exception as e:
            print(f"❌ Pathfinding error: {e}")
            path_found = False
    else:
        print(f"❌ Could not find nodes at positions {start_pos} or {end_pos}")
        path_found = False
    
    print("\n=== SUMMARY ===")
    if walkable_edges_in_solid == 0:
        print("✅ ISSUE #2 RESOLVED: No walkable edges in solid tiles")
    else:
        print(f"❌ ISSUE #2: {walkable_edges_in_solid} walkable edges in solid tiles")
    
    if functional_edges > 0:
        print(f"✅ ISSUE #1 RESOLVED: {functional_edges} functional edges found")
    else:
        print("❌ ISSUE #1: No functional edges found (custom map may not have proper switches/doors)")
    
    if path_found:
        print("✅ ISSUE #3 RESOLVED: Pathfinding working")
    else:
        print("❌ ISSUE #3: Pathfinding not working")

if __name__ == "__main__":
    test_solid_tiles_map()