"""
Debug script to identify invalid nodes on solid/wall tiles.

This script specifically checks for nodes that shouldn't exist on outer walls
and other solid tiles.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from nclone.gym_environment.environment_factory import create_visual_testing_env
from nclone.gym_environment.config import EnvironmentConfig
from nclone.graph.reachability.fast_graph_builder import FastGraphBuilder


def debug_invalid_nodes(map_id: int):
    """Find and report all invalid node placements."""
    
    print(f"\n{'='*70}")
    print(f"DEBUGGING INVALID NODE PLACEMENTS FOR MAP {map_id}")
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
    
    print(f"Tiles shape: {tiles.shape}")
    print(f"Unique tile types: {sorted(set(tiles.flatten()))}")
    
    # Count tiles by type
    unique, counts = np.unique(tiles, return_counts=True)
    print(f"\nTile distribution:")
    for tt, count in zip(unique, counts):
        pct = 100 * count / tiles.size
        print(f"  Type {tt:2}: {count:5} tiles ({pct:5.2f}%)")
    
    # Build graph
    builder = FastGraphBuilder(debug=True)
    ninja_pos = env.nplay_headless.ninja_position()
    ninja_pos = (int(ninja_pos[0]), int(ninja_pos[1]))
    
    print(f"\nBuilding graph...")
    graph_data = builder.build_graph(level_data, ninja_pos=ninja_pos)
    
    adjacency = graph_data["adjacency"]
    reachable = graph_data.get("reachable", set())
    
    print(f"\n{'='*70}")
    print("ANALYZING NODE PLACEMENTS")
    print(f"{'='*70}")
    print(f"Total nodes in adjacency: {len(adjacency)}")
    print(f"Total reachable nodes: {len(reachable)}")
    
    # Check each node to see if it's on a valid tile
    nodes_by_tile_type = {}
    invalid_nodes = []
    
    for node_pos in adjacency.keys():
        nx, ny = node_pos
        tx, ty = nx // 24, ny // 24
        
        # Check if tile coordinates are in bounds
        if not (0 <= ty < tiles.shape[0] and 0 <= tx < tiles.shape[1]):
            invalid_nodes.append({
                'pos': node_pos,
                'tile': (tx, ty),
                'reason': 'Out of bounds',
                'tile_type': -1
            })
            continue
        
        tile_type = tiles[ty, tx]
        nodes_by_tile_type[tile_type] = nodes_by_tile_type.get(tile_type, 0) + 1
        
        # Type 1 = solid, should NEVER have nodes
        if tile_type == 1:
            invalid_nodes.append({
                'pos': node_pos,
                'tile': (tx, ty),
                'reason': 'Solid tile (Type 1)',
                'tile_type': tile_type
            })
    
    print(f"\nNodes by tile type:")
    for tt in sorted(nodes_by_tile_type.keys()):
        count = nodes_by_tile_type[tt]
        print(f"  Type {tt:2}: {count:5} nodes")
    
    print(f"\n{'='*70}")
    print("INVALID NODE ANALYSIS")
    print(f"{'='*70}")
    
    if invalid_nodes:
        print(f"\n⚠️  FOUND {len(invalid_nodes)} INVALID NODES!\n")
        
        # Group by reason
        by_reason = {}
        for node in invalid_nodes:
            reason = node['reason']
            by_reason[reason] = by_reason.get(reason, [])
            by_reason[reason].append(node)
        
        for reason, nodes in by_reason.items():
            print(f"\n{reason}: {len(nodes)} nodes")
            # Show first 20 examples
            for i, node in enumerate(nodes[:20]):
                print(f"  {i+1}. Position {node['pos']} on tile {node['tile']} (type {node['tile_type']})")
            if len(nodes) > 20:
                print(f"  ... and {len(nodes) - 20} more")
        
        # Create a visual map showing invalid nodes
        print(f"\n{'='*70}")
        print("VISUAL MAP OF INVALID NODES")
        print(f"{'='*70}")
        print("Legend: . = valid, X = invalid node")
        print()
        
        invalid_tiles = set((node['tile'] for node in invalid_nodes))
        
        for y in range(tiles.shape[0]):
            line = ""
            for x in range(tiles.shape[1]):
                if (x, y) in invalid_tiles:
                    line += "X"
                else:
                    line += "."
            print(f"  {line}")
    else:
        print("\n✅ No invalid nodes found!")
    
    # Check reachability on solid tiles
    print(f"\n{'='*70}")
    print("REACHABILITY ON SOLID TILES")
    print(f"{'='*70}")
    
    reachable_on_solid = []
    for node_pos in reachable:
        nx, ny = node_pos
        tx, ty = nx // 24, ny // 24
        
        if 0 <= ty < tiles.shape[0] and 0 <= tx < tiles.shape[1]:
            if tiles[ty, tx] == 1:  # Solid tile
                reachable_on_solid.append({
                    'pos': node_pos,
                    'tile': (tx, ty)
                })
    
    if reachable_on_solid:
        print(f"\n⚠️  FOUND {len(reachable_on_solid)} REACHABLE NODES ON SOLID TILES!\n")
        for i, node in enumerate(reachable_on_solid[:20]):
            print(f"  {i+1}. Position {node['pos']} on tile {node['tile']}")
        if len(reachable_on_solid) > 20:
            print(f"  ... and {len(reachable_on_solid) - 20} more")
    else:
        print("\n✅ No reachable nodes on solid tiles!")
    
    env.close()
    
    print(f"\n{'='*70}")
    print("DEBUG COMPLETE")
    print(f"{'='*70}\n")
    
    return invalid_nodes, reachable_on_solid


if __name__ == "__main__":
    map_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    invalid_nodes, reachable_on_solid = debug_invalid_nodes(map_id)
    
    if invalid_nodes or reachable_on_solid:
        print("\n" + "="*70)
        print("CRITICAL ISSUE DETECTED")
        print("="*70)
        print(f"Invalid nodes: {len(invalid_nodes)}")
        print(f"Reachable on solid: {len(reachable_on_solid)}")
        print("\nThe graph builder is incorrectly creating nodes on solid tiles!")
        print("="*70)
        sys.exit(1)
    else:
        print("\n✅ All nodes are valid!")
        sys.exit(0)

