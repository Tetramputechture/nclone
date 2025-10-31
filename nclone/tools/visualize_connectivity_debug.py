"""
Visual debug tool for graph connectivity issues.

Creates detailed visualizations showing:
1. Which nodes are being generated
2. Which edges are being created
3. Why specific connections are allowed/blocked
"""

import numpy as np
import pygame
from typing import Dict, Tuple, Set, List
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nclone.graph.reachability.fast_graph_builder import FastGraphBuilder
from nclone.level_loader import LevelLoader


def visualize_graph_debug(map_id: int = 0):
    """Create detailed debug visualization of graph building."""
    
    # Load level
    loader = LevelLoader()
    level_data = loader.load_level(map_id)
    tiles = level_data.tiles
    
    print(f"\n{'='*70}")
    print(f"GRAPH DEBUG VISUALIZATION FOR MAP {map_id}")
    print(f"{'='*70}")
    print(f"Tiles shape: {tiles.shape}")
    
    # Build graph with debug enabled
    builder = FastGraphBuilder(debug=True)
    graph_data = builder.build_graph(level_data, ninja_pos=(100, 100))
    
    adjacency = graph_data["adjacency"]
    
    print(f"\nGraph Statistics:")
    print(f"  Total nodes: {len(adjacency)}")
    
    total_edges = sum(len(neighbors) for neighbors in adjacency.values())
    print(f"  Total edges: {total_edges}")
    print(f"  Average edges per node: {total_edges / len(adjacency) if adjacency else 0:.2f}")
    
    # Analyze nodes by tile type
    print(f"\n{'='*70}")
    print("NODES BY TILE TYPE")
    print(f"{'='*70}")
    
    nodes_by_tile_type = {}
    for node_pos in adjacency.keys():
        # Determine which tile this node is in
        node_x, node_y = node_pos
        tile_x = node_x // 24
        tile_y = node_y // 24
        
        if 0 <= tile_y < tiles.shape[0] and 0 <= tile_x < tiles.shape[1]:
            tile_type = tiles[tile_y, tile_x]
            nodes_by_tile_type[tile_type] = nodes_by_tile_type.get(tile_type, 0) + 1
    
    for tile_type in sorted(nodes_by_tile_type.keys()):
        count = nodes_by_tile_type[tile_type]
        print(f"  Tile {tile_type:2}: {count:4} nodes")
    
    # Find problematic connections
    print(f"\n{'='*70}")
    print("ANALYZING EDGE CONNECTIONS")
    print(f"{'='*70}")
    
    problematic_edges = []
    
    for src_pos, neighbors in adjacency.items():
        src_x, src_y = src_pos
        src_tile_x = src_x // 24
        src_tile_y = src_y // 24
        
        if not (0 <= src_tile_y < tiles.shape[0] and 0 <= src_tile_x < tiles.shape[1]):
            continue
            
        src_tile_type = tiles[src_tile_y, src_tile_x]
        
        for neighbor_pos, cost in neighbors:
            dst_x, dst_y = neighbor_pos
            dst_tile_x = dst_x // 24
            dst_tile_y = dst_y // 24
            
            if not (0 <= dst_tile_y < tiles.shape[0] and 0 <= dst_tile_x < tiles.shape[1]):
                continue
            
            dst_tile_type = tiles[dst_tile_y, dst_tile_x]
            
            # Check if this looks problematic
            # E.g., connecting through what should be solid geometry
            if src_tile_type == 0 and dst_tile_type == 0:
                # Both empty - check if there's solid geometry between them
                dx = dst_tile_x - src_tile_x
                dy = dst_tile_y - src_tile_y
                
                # If moving horizontally/vertically, check intermediate tile
                if abs(dx) + abs(dy) == 1:  # Cardinal direction
                    # This is direct neighbor - should be fine
                    pass
                elif abs(dx) == 1 and abs(dy) == 1:  # Diagonal
                    # Check intermediate tiles
                    side_tile = tiles[src_tile_y, dst_tile_x]
                    vert_tile = tiles[dst_tile_y, src_tile_x]
                    
                    if side_tile == 1 or vert_tile == 1:
                        problematic_edges.append({
                            'src': src_pos,
                            'dst': neighbor_pos,
                            'src_tile': (src_tile_x, src_tile_y, src_tile_type),
                            'dst_tile': (dst_tile_x, dst_tile_y, dst_tile_type),
                            'side_tile': side_tile,
                            'vert_tile': vert_tile,
                            'reason': 'Diagonal through solid intermediate tile'
                        })
    
    if problematic_edges:
        print(f"\nFound {len(problematic_edges)} potentially problematic edges:")
        for i, edge in enumerate(problematic_edges[:10]):  # Show first 10
            print(f"\n  Edge {i+1}:")
            print(f"    From: {edge['src']} (tile {edge['src_tile'][0]},{edge['src_tile'][1]} type {edge['src_tile'][2]})")
            print(f"    To: {edge['dst']} (tile {edge['dst_tile'][0]},{edge['dst_tile'][1]} type {edge['dst_tile'][2]})")
            print(f"    Reason: {edge['reason']}")
            print(f"    Intermediate tiles: side={edge['side_tile']}, vert={edge['vert_tile']}")
    else:
        print("\n✓ No obviously problematic edges found")
    
    return adjacency, problematic_edges


def create_detailed_edge_report(map_id: int = 0, output_file: str = "edge_debug_report.txt"):
    """Create a detailed text report of all edges."""
    
    loader = LevelLoader()
    level_data = loader.load_level(map_id)
    tiles = level_data.tiles
    
    builder = FastGraphBuilder(debug=True)
    graph_data = builder.build_graph(level_data, ninja_pos=(100, 100))
    adjacency = graph_data["adjacency"]
    
    with open(output_file, 'w') as f:
        f.write(f"DETAILED EDGE REPORT FOR MAP {map_id}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Tiles shape: {tiles.shape}\n")
        f.write(f"Total nodes: {len(adjacency)}\n")
        f.write(f"Total edges: {sum(len(neighbors) for neighbors in adjacency.values())}\n\n")
        
        f.write(f"{'='*70}\n")
        f.write("ALL EDGES (DETAILED)\n")
        f.write(f"{'='*70}\n\n")
        
        for src_pos, neighbors in sorted(adjacency.items()):
            src_x, src_y = src_pos
            src_tile_x = src_x // 24
            src_tile_y = src_y // 24
            
            if not (0 <= src_tile_y < tiles.shape[0] and 0 <= src_tile_x < tiles.shape[1]):
                continue
            
            src_tile_type = tiles[src_tile_y, src_tile_x]
            src_sub_x = 0 if (src_x % 24) < 12 else 1
            src_sub_y = 0 if (src_y % 24) < 12 else 1
            
            f.write(f"Node: {src_pos} | Tile: ({src_tile_x},{src_tile_y}) Type {src_tile_type} | Sub: ({src_sub_x},{src_sub_y})\n")
            
            if neighbors:
                for neighbor_pos, cost in neighbors:
                    dst_x, dst_y = neighbor_pos
                    dst_tile_x = dst_x // 24
                    dst_tile_y = dst_y // 24
                    
                    if 0 <= dst_tile_y < tiles.shape[0] and 0 <= dst_tile_x < tiles.shape[1]:
                        dst_tile_type = tiles[dst_tile_y, dst_tile_x]
                        dst_sub_x = 0 if (dst_x % 24) < 12 else 1
                        dst_sub_y = 0 if (dst_y % 24) < 12 else 1
                        
                        dx = dst_tile_x - src_tile_x
                        dy = dst_tile_y - src_tile_y
                        
                        direction = ""
                        if dx == 0 and dy == -1: direction = "N"
                        elif dx == 1 and dy == -1: direction = "NE"
                        elif dx == 1 and dy == 0: direction = "E"
                        elif dx == 1 and dy == 1: direction = "SE"
                        elif dx == 0 and dy == 1: direction = "S"
                        elif dx == -1 and dy == 1: direction = "SW"
                        elif dx == -1 and dy == 0: direction = "W"
                        elif dx == -1 and dy == -1: direction = "NW"
                        else: direction = "SAME"
                        
                        f.write(f"  → {neighbor_pos} | Tile: ({dst_tile_x},{dst_tile_y}) Type {dst_tile_type} | Sub: ({dst_sub_x},{dst_sub_y}) | Dir: {direction} | Cost: {cost:.2f}\n")
            else:
                f.write(f"  (No neighbors)\n")
            
            f.write("\n")
    
    print(f"\nDetailed edge report written to: {output_file}")


if __name__ == "__main__":
    import sys
    
    map_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    # Run debug visualization
    adjacency, problematic = visualize_graph_debug(map_id)
    
    # Create detailed report
    create_detailed_edge_report(map_id, f"edge_debug_report_map{map_id}.txt")
    
    print(f"\n{'='*70}")
    print("DEBUG COMPLETE")
    print(f"{'='*70}")

