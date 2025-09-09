#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

from nclone.level_loader import LevelLoader
from nclone.graph.graph_construction import GraphConstruction
from nclone.graph.edge_building import EdgeBuilding
from nclone.pathfinding.dijkstra import DijkstraPathfinder

def debug_simple_walk():
    """Debug the simple-walk map to see why ninja-to-switch path is not found."""
    
    # Load the simple-walk test map
    level_loader = LevelLoader()
    level_data = level_loader.load_level("nclone/test_maps/simple-walk")
    
    print(f"Map size: {level_data.width}x{level_data.height} tiles")
    print(f"Entities: {len(level_data.entities)}")
    
    for i, entity in enumerate(level_data.entities):
        entity_type = entity.get("type", 0)
        entity_x = entity.get("x", 0.0)
        entity_y = entity.get("y", 0.0)
        print(f"  Entity {i}: type={entity_type}, pos=({entity_x}, {entity_y})")
    
    # Build graph
    print("\nBuilding graph...")
    graph_constructor = GraphConstruction()
    nodes, sub_grid_node_map, entity_nodes = graph_constructor.build_graph(level_data)
    print(f"Created {len(nodes)} nodes")
    print(f"Entity nodes: {len(entity_nodes) if entity_nodes else 0}")
    
    # Build edges
    print("\nBuilding edges...")
    edge_builder = EdgeBuilding()
    ninja_position = (396.0, 372.0)  # From the debug output
    ninja_velocity = (0.0, 0.0)
    
    edge_index, edge_features, edge_mask, edge_types = edge_builder.build_edges(
        nodes, sub_grid_node_map, entity_nodes, level_data, ninja_position, ninja_velocity
    )
    
    print(f"Created {edge_index.shape[1]} edges")
    
    # Try pathfinding from ninja to switch
    print("\nTesting pathfinding...")
    pathfinder = DijkstraPathfinder()
    
    # Find ninja and switch nodes
    ninja_node = None
    switch_node = None
    
    if entity_nodes:
        for node_idx, entity_data in entity_nodes:
            entity_type = entity_data.get("type", 0)
            if entity_type == 0:  # Ninja
                ninja_node = node_idx
                print(f"Found ninja node: {node_idx}")
            elif entity_type == 4:  # Switch
                switch_node = node_idx
                print(f"Found switch node: {node_idx}")
    
    if ninja_node is not None and switch_node is not None:
        print(f"\nTrying to find path from ninja node {ninja_node} to switch node {switch_node}")
        
        try:
            path = pathfinder.find_path(
                edge_index, edge_features, edge_mask, edge_types,
                ninja_node, switch_node, nodes
            )
            
            if path:
                print(f"✅ Found path with {len(path)} waypoints")
                for i, waypoint in enumerate(path):
                    print(f"  Waypoint {i}: {waypoint}")
            else:
                print("❌ No path found")
                
        except Exception as e:
            print(f"❌ Pathfinding failed: {e}")
    else:
        print(f"❌ Could not find ninja node ({ninja_node}) or switch node ({switch_node})")

if __name__ == "__main__":
    debug_simple_walk()