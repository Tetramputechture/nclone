#!/usr/bin/env python3
"""
Debug nodes in empty tiles adjacent to ninja to understand escape routing.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine
from nclone.graph.common import SUB_CELL_SIZE
import numpy as np


def debug_escape_nodes():
    """Debug nodes in empty tiles adjacent to ninja."""
    print("=" * 80)
    print("DEBUGGING ESCAPE NODES")
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
    
    # Check adjacent empty tiles
    ninja_tile_x = int(ninja_pos[0] // 24)
    ninja_tile_y = int(ninja_pos[1] // 24)
    
    print(f"Ninja tile: ({ninja_tile_x}, {ninja_tile_y})")
    
    adjacent_empty_tiles = [
        (4, 17),  # Left-up
        (5, 17),  # Up
    ]
    
    print(f"\nChecking nodes in adjacent empty tiles:")
    
    for tile_x, tile_y in adjacent_empty_tiles:
        print(f"\nTile ({tile_x}, {tile_y}):")
        
        # Check tile value
        if 0 <= tile_x < level_data.width and 0 <= tile_y < level_data.height:
            tile_value = level_data.get_tile(tile_y, tile_x)
            print(f"  Tile value: {tile_value} ({'empty' if tile_value == 0 else 'solid'})")
            
            if tile_value != 0:
                print(f"  ❌ Tile is not empty!")
                continue
        else:
            print(f"  ❌ Tile out of bounds!")
            continue
        
        # Find nodes in this tile
        tile_pixel_x_start = tile_x * 24
        tile_pixel_y_start = tile_y * 24
        tile_pixel_x_end = tile_pixel_x_start + 24
        tile_pixel_y_end = tile_pixel_y_start + 24
        
        nodes_in_tile = []
        
        for node_idx in range(graph_data.num_nodes):
            if graph_data.node_mask[node_idx] == 0:
                continue
            
            node_pos = pathfinding_engine._get_node_position(graph_data, node_idx)
            node_x, node_y = node_pos
            
            if (tile_pixel_x_start <= node_x < tile_pixel_x_end and 
                tile_pixel_y_start <= node_y < tile_pixel_y_end):
                nodes_in_tile.append((node_idx, node_pos))
        
        print(f"  Found {len(nodes_in_tile)} nodes in this tile:")
        
        for node_idx, node_pos in nodes_in_tile:
            # Count edges for this node
            edge_count = 0
            connected_to_ninja = False
            
            for edge_idx in range(graph_data.num_edges):
                if graph_data.edge_mask[edge_idx] == 0:
                    continue
                
                src = int(graph_data.edge_index[0, edge_idx])
                dst = int(graph_data.edge_index[1, edge_idx])
                
                if src == node_idx or dst == node_idx:
                    edge_count += 1
                    
                    # Check if connected to ninja
                    if (src == ninja_node and dst == node_idx) or (src == node_idx and dst == ninja_node):
                        connected_to_ninja = True
            
            # Calculate distance from ninja
            distance = ((ninja_pos[0] - node_pos[0])**2 + (ninja_pos[1] - node_pos[1])**2)**0.5
            
            connection_status = "🔗 CONNECTED TO NINJA" if connected_to_ninja else "❌ NOT CONNECTED"
            
            print(f"    Node {node_idx}: {node_pos} ({edge_count} edges) | distance: {distance:.1f} | {connection_status}")
    
    # Check if any of the ninja's connected nodes have edges to empty tile nodes
    print(f"\nChecking ninja's connected nodes for escape routes:")
    
    ninja_connected_nodes = []
    
    for edge_idx in range(graph_data.num_edges):
        if graph_data.edge_mask[edge_idx] == 0:
            continue
        
        src = int(graph_data.edge_index[0, edge_idx])
        dst = int(graph_data.edge_index[1, edge_idx])
        
        if src == ninja_node:
            ninja_connected_nodes.append(dst)
        elif dst == ninja_node:
            ninja_connected_nodes.append(src)
    
    # Remove duplicates
    ninja_connected_nodes = list(set(ninja_connected_nodes))
    
    print(f"Ninja is connected to {len(ninja_connected_nodes)} nodes:")
    
    for connected_node in ninja_connected_nodes:
        node_pos = pathfinding_engine._get_node_position(graph_data, connected_node)
        
        # Check what tile this node is in
        node_tile_x = int(node_pos[0] // 24)
        node_tile_y = int(node_pos[1] // 24)
        
        if 0 <= node_tile_x < level_data.width and 0 <= node_tile_y < level_data.height:
            tile_value = level_data.get_tile(node_tile_y, node_tile_x)
            tile_status = "empty" if tile_value == 0 else "solid"
        else:
            tile_value = -1
            tile_status = "out of bounds"
        
        # Count edges from this node to other nodes
        edge_count = 0
        external_connections = []
        
        for edge_idx in range(graph_data.num_edges):
            if graph_data.edge_mask[edge_idx] == 0:
                continue
            
            src = int(graph_data.edge_index[0, edge_idx])
            dst = int(graph_data.edge_index[1, edge_idx])
            
            other_node = None
            if src == connected_node and dst != ninja_node:
                other_node = dst
            elif dst == connected_node and src != ninja_node:
                other_node = src
            
            if other_node is not None:
                edge_count += 1
                other_pos = pathfinding_engine._get_node_position(graph_data, other_node)
                other_tile_x = int(other_pos[0] // 24)
                other_tile_y = int(other_pos[1] // 24)
                
                if 0 <= other_tile_x < level_data.width and 0 <= other_tile_y < level_data.height:
                    other_tile_value = level_data.get_tile(other_tile_y, other_tile_x)
                    other_tile_status = "empty" if other_tile_value == 0 else "solid"
                else:
                    other_tile_value = -1
                    other_tile_status = "out of bounds"
                
                external_connections.append((other_node, other_pos, other_tile_status))
        
        print(f"  Node {connected_node}: {node_pos} in tile ({node_tile_x}, {node_tile_y}) = {tile_value} ({tile_status})")
        print(f"    {edge_count} external connections:")
        
        for other_node, other_pos, other_tile_status in external_connections[:5]:  # Show first 5
            print(f"      -> Node {other_node}: {other_pos} ({other_tile_status})")


if __name__ == '__main__':
    debug_escape_nodes()