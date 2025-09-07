#!/usr/bin/env python3
"""
Debug mouse pathfinding issues on bblock_test map.

This script investigates why pathfinding fails on clearly traversable paths
like the example: (150.44, 446) to (129, 446).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pygame
from nclone.nsim import Simulator
from nclone.sim_config import SimConfig
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.constants.entity_types import EntityType
import cv2

def load_bblock_test_map():
    """Load the bblock_test map data using the same approach as validation script."""
    # Import the function from our working validation script
    sys.path.append('.')
    from validate_with_bblock_test import load_bblock_test_map as load_map
    from validate_with_bblock_test import create_level_data_from_simulator
    
    map_data = load_map()
    if map_data is None:
        return None, None
    
    # Create a simulator to extract level data
    from nclone.nsim import Simulator
    from nclone.sim_config import SimConfig
    
    config = SimConfig()
    config.level_name = "bblock_test"
    
    try:
        sim = Simulator(config)
        level_data_obj = create_level_data_from_simulator(sim)
        return level_data_obj
    except Exception as e:
        print(f"❌ Failed to create simulator: {e}")
        return None

def debug_mouse_pathfinding():
    """Debug specific mouse pathfinding failure."""
    print("=== DEBUGGING MOUSE PATHFINDING ON BBLOCK_TEST ===")
    
    # Load the bblock_test map
    level_data_obj = load_bblock_test_map()
    if level_data_obj is None:
        return
    
    print(f"Level dimensions: {level_data_obj.tiles.shape}")
    print(f"Entities: {len(level_data_obj.entities)}")
    
    # Find ninja position
    ninja_pos = None
    for entity in level_data_obj.entities:
        if hasattr(entity, 'type') and entity['type'] == EntityType.NINJA:
            ninja_pos = (entity['x'], entity['y'])
            break
    
    # Use default ninja position if not found
    if ninja_pos is None:
        ninja_pos = (36, 564)  # Known position from validation script
        print(f"Using default ninja position: {ninja_pos}")
    else:
        print(f"Ninja position: {ninja_pos}")
    
    # Build graph
    print("Building hierarchical graph...")
    graph_builder = HierarchicalGraphBuilder()
    graph_data = graph_builder.build_graph(level_data_obj, ninja_pos)
    print(f"Graph built:")
    print(f"  Sub-cell: {np.sum(graph_data.sub_cell_graph.node_mask)} nodes, {np.sum(graph_data.sub_cell_graph.edge_mask)} edges")
    print(f"  Tile: {np.sum(graph_data.tile_graph.node_mask)} nodes, {np.sum(graph_data.tile_graph.edge_mask)} edges")
    print(f"  Region: {np.sum(graph_data.region_graph.node_mask)} nodes, {np.sum(graph_data.region_graph.edge_mask)} edges")
    
    # Test the specific failing pathfinding case
    start_pos = (150.441656766729, 446.0)
    end_pos = (129.0, 446.0)
    
    print(f"\n=== DEBUGGING SPECIFIC CASE ===")
    print(f"Start: {start_pos}")
    print(f"End: {end_pos}")
    
    # Find nearest nodes to these positions
    def find_nearest_node(pos, graph_data):
        """Find the nearest valid node to a position in the sub-cell graph."""
        # Use sub-cell graph for highest precision
        sub_graph = graph_data.sub_cell_graph
        
        valid_indices = np.where(sub_graph.node_mask)[0]
        if len(valid_indices) == 0:
            return None
            
        min_dist = float('inf')
        nearest_idx = None
        
        for idx in valid_indices:
            # Node positions are stored in the first two features
            node_pos = sub_graph.node_features[idx, :2]
            dist = np.sqrt((node_pos[0] - pos[0])**2 + (node_pos[1] - pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        
        return nearest_idx, min_dist
    
    start_node, start_dist = find_nearest_node(start_pos, graph_data)
    end_node, end_dist = find_nearest_node(end_pos, graph_data)
    
    print(f"Nearest start node: {start_node} (distance: {start_dist:.2f})")
    print(f"Nearest end node: {end_node} (distance: {end_dist:.2f})")
    
    if start_node is not None:
        start_node_pos = graph_data.sub_cell_graph.node_features[start_node, :2]
        print(f"Start node position: {start_node_pos}")
    
    if end_node is not None:
        end_node_pos = graph_data.sub_cell_graph.node_features[end_node, :2]
        print(f"End node position: {end_node_pos}")
    
    # Check if nodes are connected
    if start_node is not None and end_node is not None:
        # Check direct connectivity using edge_index
        edge_exists = False
        sub_graph = graph_data.sub_cell_graph
        valid_edges = np.where(sub_graph.edge_mask)[0]
        
        for edge_idx in valid_edges:
            source, target = sub_graph.edge_index[:, edge_idx]
            if (source == start_node and target == end_node) or (source == end_node and target == start_node):
                edge_exists = True
                break
        
        if edge_exists:
            print(f"✅ Direct edge exists between nodes {start_node} and {end_node}")
        else:
            print(f"❌ No direct edge between nodes {start_node} and {end_node}")
            
        # Debug: Check if both nodes are the same
        if start_node == end_node:
            print(f"⚠️  WARNING: Both nodes are the same ({start_node})")
            print(f"   This suggests the nearest node search isn't working properly")
            
            # Let's check a few more nodes
            print("   Checking first 10 valid nodes:")
            valid_indices = np.where(sub_graph.node_mask)[0][:10]
            for i, idx in enumerate(valid_indices):
                node_pos = sub_graph.node_features[idx, :2]
                start_dist = np.sqrt((node_pos[0] - start_pos[0])**2 + (node_pos[1] - start_pos[1])**2)
                end_dist = np.sqrt((node_pos[0] - end_pos[0])**2 + (node_pos[1] - end_pos[1])**2)
                print(f"     Node {idx}: pos={node_pos}, start_dist={start_dist:.2f}, end_dist={end_dist:.2f}")
        
        # Try pathfinding
        pathfinding_engine = PathfindingEngine()
        try:
            path_result = pathfinding_engine.find_shortest_path(
                graph_data.sub_cell_graph, start_node, end_node, PathfindingAlgorithm.A_STAR
            )
            
            if path_result and path_result.success:
                print(f"✅ Pathfinding successful: {len(path_result.path)} nodes")
                print(f"Path: {path_result.path[:10]}{'...' if len(path_result.path) > 10 else ''}")
                print(f"Total cost: {path_result.total_cost:.2f}")
            else:
                print(f"❌ Pathfinding failed")
                
                # Debug connectivity
                print("\n=== CONNECTIVITY ANALYSIS ===")
                
                # Check if nodes are in the same connected component
                def find_connected_component(node_idx, sub_graph, max_nodes=1000):
                    """Find all nodes connected to the given node."""
                    visited = set()
                    queue = [node_idx]
                    
                    # Build adjacency list from edge_index
                    adjacency = {}
                    valid_edges = np.where(sub_graph.edge_mask)[0]
                    for edge_idx in valid_edges:
                        source, target = sub_graph.edge_index[:, edge_idx]
                        if source not in adjacency:
                            adjacency[source] = []
                        if target not in adjacency:
                            adjacency[target] = []
                        adjacency[source].append(target)
                        adjacency[target].append(source)
                    
                    while queue and len(visited) < max_nodes:
                        current = queue.pop(0)
                        if current in visited:
                            continue
                        visited.add(current)
                        
                        # Find neighbors
                        if current in adjacency:
                            for neighbor in adjacency[current]:
                                if neighbor not in visited:
                                    queue.append(neighbor)
                    
                    return visited
                
                start_component = find_connected_component(start_node, graph_data.sub_cell_graph)
                print(f"Start node component size: {len(start_component)}")
                
                if end_node in start_component:
                    print("✅ Nodes are in the same connected component")
                else:
                    print("❌ Nodes are in different connected components")
                    end_component = find_connected_component(end_node, graph_data)
                    print(f"End node component size: {len(end_component)}")
                
        except Exception as e:
            print(f"❌ Pathfinding error: {e}")
    
    # Create debug visualization
    print("\n=== CREATING DEBUG VISUALIZATION ===")
    create_debug_visualization(graph_data, start_pos, end_pos, start_node, end_node, level_data_obj.tiles)

def create_debug_visualization(graph_data, start_pos, end_pos, start_node, end_node, level_data):
    """Create a debug visualization showing the pathfinding issue."""
    try:
        # Create a visualization image
        height, width = level_data.shape
        debug_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw level tiles
        for y in range(height):
            for x in range(width):
                if level_data[y, x] > 0:  # Solid tile
                    debug_img[y, x] = [100, 100, 100]  # Gray
                else:  # Empty space
                    debug_img[y, x] = [20, 20, 20]  # Dark
        
        # Draw graph nodes
        valid_indices = np.where(graph_data.node_mask)[0]
        for idx in valid_indices:
            pos = graph_data.node_positions[idx]
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < width and 0 <= y < height:
                debug_img[y, x] = [255, 255, 255]  # White dots for nodes
        
        # Highlight start and end positions
        start_x, start_y = int(start_pos[0]), int(start_pos[1])
        end_x, end_y = int(end_pos[0]), int(end_pos[1])
        
        # Draw circles for start/end positions
        cv2.circle(debug_img, (start_x, start_y), 5, (0, 255, 0), 2)  # Green for start
        cv2.circle(debug_img, (end_x, end_y), 5, (0, 0, 255), 2)  # Red for end
        
        # Highlight nearest nodes
        if start_node is not None:
            start_node_pos = graph_data.node_positions[start_node]
            cv2.circle(debug_img, (int(start_node_pos[0]), int(start_node_pos[1])), 3, (0, 255, 255), -1)  # Cyan
        
        if end_node is not None:
            end_node_pos = graph_data.node_positions[end_node]
            cv2.circle(debug_img, (int(end_node_pos[0]), int(end_node_pos[1])), 3, (255, 0, 255), -1)  # Magenta
        
        # Save debug image
        debug_path = "debug/pathfinding_debug.png"
        cv2.imwrite(debug_path, debug_img)
        print(f"✅ Debug visualization saved to {debug_path}")
        
    except Exception as e:
        print(f"❌ Failed to create debug visualization: {e}")

if __name__ == "__main__":
    debug_mouse_pathfinding()