"""
Visualize which tile-to-tile edges are being created in the graph.

This creates an ASCII map showing:
- Which tiles have nodes
- Which tile pairs have edges between them
- The direction of cross-tile edges
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from nclone.gym_environment.environment_factory import create_visual_testing_env
from nclone.gym_environment.config import EnvironmentConfig
from nclone.graph.reachability.fast_graph_builder import FastGraphBuilder
from typing import Dict, Tuple, Set


def visualize_tile_edges(map_id: int):
    """Create ASCII visualization of tile-level edges."""
    
    print(f"\n{'='*70}")
    print(f"TILE-LEVEL EDGE VISUALIZATION FOR MAP {map_id}")
    print(f"{'='*70}\n")
    
    # Create environment
    config = EnvironmentConfig.for_visual_testing()
    config.starting_map = map_id
    config.headless = True
    env = create_visual_testing_env(config=config)
    env.reset()
    
    # Get level data
    level_data = env.level_data
    tiles = level_data.tiles
    
    # Build graph
    builder = FastGraphBuilder(debug=False)
    ninja_pos = env.nplay_headless.ninja_position()
    ninja_pos = (int(ninja_pos[0]), int(ninja_pos[1]))
    graph_data = builder.build_graph(level_data, ninja_pos=ninja_pos)
    
    adjacency = graph_data["adjacency"]
    
    # Find which tiles have nodes
    tiles_with_nodes = set()
    for node_pos in adjacency.keys():
        nx, ny = node_pos
        tx, ty = nx // 24, ny // 24
        tiles_with_nodes.add((tx, ty))
    
    # Find cross-tile edges
    tile_edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}
    
    for src_pos, neighbors in adjacency.items():
        sx, sy = src_pos
        src_tx, src_ty = sx // 24, sy // 24
        
        for dst_pos, cost in neighbors:
            dx, dy = dst_pos
            dst_tx, dst_ty = dx // 24, dy // 24
            
            if (src_tx, src_ty) != (dst_tx, dst_ty):
                # Cross-tile edge
                edge_key = ((src_tx, src_ty), (dst_tx, dst_ty))
                tile_edges[edge_key] = tile_edges.get(edge_key, 0) + 1
    
    print(f"Tiles shape: {tiles.shape}")
    print(f"Tiles with nodes: {len(tiles_with_nodes)}")
    print(f"Unique cross-tile edge pairs: {len(tile_edges)}")
    print(f"Total cross-tile edge count: {sum(tile_edges.values())}")
    
    # Print tile map with node indicators
    print(f"\n{'='*70}")
    print("TILE MAP (with node indicators)")
    print(f"{'='*70}")
    print("Legend: . = solid (no nodes), X = has nodes")
    print()
    
    for y in range(tiles.shape[0]):
        line = ""
        for x in range(tiles.shape[1]):
            if (x, y) in tiles_with_nodes:
                line += "X"
            else:
                line += "."
        print(f"  {line}")
    
    # Analyze edge directions
    print(f"\n{'='*70}")
    print("CROSS-TILE EDGE ANALYSIS")
    print(f"{'='*70}")
    
    edge_directions = {'N': 0, 'NE': 0, 'E': 0, 'SE': 0, 'S': 0, 'SW': 0, 'W': 0, 'NW': 0}
    
    for ((src_tx, src_ty), (dst_tx, dst_ty)), count in tile_edges.items():
        dx = dst_tx - src_tx
        dy = dst_ty - src_ty
        
        if dx == 0 and dy == -1:
            edge_directions['N'] += count
        elif dx == 1 and dy == -1:
            edge_directions['NE'] += count
        elif dx == 1 and dy == 0:
            edge_directions['E'] += count
        elif dx == 1 and dy == 1:
            edge_directions['SE'] += count
        elif dx == 0 and dy == 1:
            edge_directions['S'] += count
        elif dx == -1 and dy == 1:
            edge_directions['SW'] += count
        elif dx == -1 and dy == 0:
            edge_directions['W'] += count
        elif dx == -1 and dy == -1:
            edge_directions['NW'] += count
    
    print("\nEdges by direction:")
    for direction, count in edge_directions.items():
        print(f"  {direction:3}: {count:5} edges")
    
    # Find specific problematic connections
    print(f"\n{'='*70}")
    print("CHECKING FOR IMPOSSIBLE CONNECTIONS")
    print(f"{'='*70}")
    
    impossible = []
    
    for ((src_tx, src_ty), (dst_tx, dst_ty)), count in tile_edges.items():
        # Check if tiles are within bounds
        if not (0 <= src_ty < tiles.shape[0] and 0 <= src_tx < tiles.shape[1]):
            continue
        if not (0 <= dst_ty < tiles.shape[0] and 0 <= dst_tx < tiles.shape[1]):
            continue
        
        src_type = tiles[src_ty, src_tx]
        dst_type = tiles[dst_ty, dst_tx]
        
        # Should never connect through solid tiles
        if src_type == 1 or dst_type == 1:
            impossible.append({
                'src': (src_tx, src_ty),
                'dst': (dst_tx, dst_ty),
                'src_type': src_type,
                'dst_type': dst_type,
                'count': count,
                'reason': 'Connects to/from solid tile'
            })
        
        # Check diagonal connections through solid corners
        dx = dst_tx - src_tx
        dy = dst_ty - src_ty
        
        if abs(dx) == 1 and abs(dy) == 1:
            # Diagonal - check intermediate tiles
            side_tx, side_ty = src_tx + dx, src_ty
            vert_tx, vert_ty = src_tx, src_ty + dy
            
            side_type = -1
            vert_type = -1
            
            if 0 <= side_ty < tiles.shape[0] and 0 <= side_tx < tiles.shape[1]:
                side_type = tiles[side_ty, side_tx]
            if 0 <= vert_ty < tiles.shape[0] and 0 <= vert_tx < tiles.shape[1]:
                vert_type = tiles[vert_ty, vert_tx]
            
            # If EITHER intermediate is solid, this is problematic
            if side_type == 1 and vert_type == 1:
                impossible.append({
                    'src': (src_tx, src_ty),
                    'dst': (dst_tx, dst_ty),
                    'src_type': src_type,
                    'dst_type': dst_type,
                    'side_type': side_type,
                    'vert_type': vert_type,
                    'count': count,
                    'reason': f'Diagonal through solid corner (side={side_type}, vert={vert_type})'
                })
    
    if impossible:
        print(f"\n⚠ Found {len(impossible)} impossible connections:")
        for i, conn in enumerate(impossible[:20]):
            print(f"\n  #{i+1}:")
            print(f"    From tile {conn['src']} (type {conn['src_type']})")
            print(f"    To tile {conn['dst']} (type {conn['dst_type']})")
            print(f"    Edge count: {conn['count']}")
            print(f"    Reason: {conn['reason']}")
            if 'side_type' in conn:
                print(f"    Intermediate tiles: side={conn['side_type']}, vert={conn['vert_type']}")
    else:
        print("\n✓ No obviously impossible connections found")
    
    env.close()
    
    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    map_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    visualize_tile_edges(map_id)

