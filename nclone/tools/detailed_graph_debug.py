"""
Detailed graph debug tool that analyzes actual graph building from environment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from nclone.gym_environment.environment_factory import create_visual_testing_env
from nclone.graph.reachability.fast_graph_builder import FastGraphBuilder


def analyze_graph_building(map_id: int, verbose: bool = True):
    """Analyze graph building with detailed output."""
    
    print(f"\n{'='*70}")
    print(f"DETAILED GRAPH ANALYSIS FOR MAP {map_id}")
    print(f"{'='*70}\n")
    
    # Create environment
    from nclone.gym_environment.config import EnvironmentConfig
    
    config = EnvironmentConfig.for_visual_testing()
    config.starting_map = map_id
    config.headless = True
    config.enable_graph_updates = True
    
    env = create_visual_testing_env(config=config)
    
    # Reset to initialize
    env.reset()
    
    # Get level data
    level_data = env.level_data
    tiles = level_data.tiles
    
    print(f"Level tiles shape: {tiles.shape}")
    print(f"Tile types present: {sorted(set(tiles.flatten()))}")
    
    # Count tiles
    unique, counts = np.unique(tiles, return_counts=True)
    print(f"\nTile distribution:")
    for tt, count in zip(unique, counts):
        pct = 100 * count / tiles.size
        print(f"  Type {tt:2}: {count:5} ({pct:5.2f}%)")
    
    # Get ninja position
    ninja_pos = env.nplay_headless.ninja_position()
    ninja_pos = (int(ninja_pos[0]), int(ninja_pos[1]))
    print(f"\nNinja start position: {ninja_pos}")
    
    # Build graph with debug enabled
    print(f"\nBuilding graph with debug enabled...")
    builder = FastGraphBuilder(debug=True)
    graph_data = builder.build_graph(level_data, ninja_pos=ninja_pos)
    
    adjacency = graph_data["adjacency"]
    reachable = graph_data.get("reachable", set())
    
    print(f"\n{'='*70}")
    print("GRAPH STATISTICS")
    print(f"{'='*70}")
    print(f"Total nodes: {len(adjacency)}")
    print(f"Reachable nodes: {len(reachable) if reachable else 'N/A'}")
    
    if adjacency:
        total_edges = sum(len(neighbors) for neighbors in adjacency.values())
        print(f"Total edges: {total_edges}")
        print(f"Average degree: {total_edges / len(adjacency):.2f}")
        
        # Detailed edge analysis
        print(f"\n{'='*70}")
        print("EDGE ANALYSIS")
        print(f"{'='*70}")
        
        # Analyze edges that cross tile boundaries
        cross_tile_edges = []
        within_tile_edges = []
        
        for src_pos, neighbors in adjacency.items():
            sx, sy = src_pos
            src_tx, src_ty = sx // 24, sy // 24
            
            for dst_pos, cost in neighbors:
                dx, dy = dst_pos
                dst_tx, dst_ty = dx // 24, dy // 24
                
                if src_tx != dst_tx or src_ty != dst_ty:
                    # Cross-tile edge
                    cross_tile_edges.append({
                        'src': src_pos,
                        'dst': dst_pos,
                        'src_tile': (src_tx, src_ty),
                        'dst_tile': (dst_tx, dst_ty),
                        'cost': cost
                    })
                else:
                    within_tile_edges.append({
                        'src': src_pos,
                        'dst': dst_pos,
                        'tile': (src_tx, src_ty),
                        'cost': cost
                    })
        
        print(f"Within-tile edges: {len(within_tile_edges)}")
        print(f"Cross-tile edges: {len(cross_tile_edges)}")
        
        # Analyze cross-tile edges by tile types
        print(f"\nCross-tile edges by tile type pairs:")
        tile_pair_counts = {}
        for edge in cross_tile_edges:
            src_tx, src_ty = edge['src_tile']
            dst_tx, dst_ty = edge['dst_tile']
            
            if 0 <= src_ty < tiles.shape[0] and 0 <= src_tx < tiles.shape[1]:
                src_type = tiles[src_ty, src_tx]
            else:
                src_type = -1
                
            if 0 <= dst_ty < tiles.shape[0] and 0 <= dst_tx < tiles.shape[1]:
                dst_type = tiles[dst_ty, dst_tx]
            else:
                dst_type = -1
            
            key = (src_type, dst_type)
            tile_pair_counts[key] = tile_pair_counts.get(key, 0) + 1
        
        for (src_type, dst_type), count in sorted(tile_pair_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  Type {src_type:2} → Type {dst_type:2}: {count:4} edges")
        
        # Check for potentially problematic edges
        print(f"\n{'='*70}")
        print("CHECKING FOR PROBLEMATIC EDGES")
        print(f"{'='*70}")
        
        problematic = []
        
        for edge in cross_tile_edges:
            src_tx, src_ty = edge['src_tile']
            dst_tx, dst_ty = edge['dst_tile']
            
            # Get tile types
            if not (0 <= src_ty < tiles.shape[0] and 0 <= src_tx < tiles.shape[1]):
                continue
            if not (0 <= dst_ty < tiles.shape[0] and 0 <= dst_tx < tiles.shape[1]):
                continue
                
            src_type = tiles[src_ty, src_tx]
            dst_type = tiles[dst_ty, dst_tx]
            
            # Check for diagonal movement through solid corners
            dx_tile = dst_tx - src_tx
            dy_tile = dst_ty - src_ty
            
            if abs(dx_tile) == 1 and abs(dy_tile) == 1:
                # Diagonal movement - check intermediate tiles
                side_tx, side_ty = src_tx + dx_tile, src_ty
                vert_tx, vert_ty = src_tx, src_ty + dy_tile
                
                side_type = -1
                vert_type = -1
                
                if 0 <= side_ty < tiles.shape[0] and 0 <= side_tx < tiles.shape[1]:
                    side_type = tiles[side_ty, side_tx]
                if 0 <= vert_ty < tiles.shape[0] and 0 <= vert_tx < tiles.shape[1]:
                    vert_type = tiles[vert_ty, vert_tx]
                
                # Flag if either intermediate is solid
                if side_type == 1 or vert_type == 1:
                    problematic.append({
                        **edge,
                        'src_type': src_type,
                        'dst_type': dst_type,
                        'side_type': side_type,
                        'vert_type': vert_type,
                        'reason': f'Diagonal through solid (side={side_type}, vert={vert_type})'
                    })
        
        if problematic:
            print(f"\nFound {len(problematic)} potentially problematic edges:")
            for i, edge in enumerate(problematic[:20]):  # Show first 20
                print(f"\n  #{i+1}:")
                print(f"    From: {edge['src']} (tile {edge['src_tile']}, type {edge['src_type']})")
                print(f"    To: {edge['dst']} (tile {edge['dst_tile']}, type {edge['dst_type']})")
                print(f"    Reason: {edge['reason']}")
        else:
            print("\n✓ No problematic diagonal edges found")
        
        # Check for edges that shouldn't exist (solid tiles)
        solid_edges = []
        for edge in cross_tile_edges:
            src_tx, src_ty = edge['src_tile']
            dst_tx, dst_ty = edge['dst_tile']
            
            if not (0 <= src_ty < tiles.shape[0] and 0 <= src_tx < tiles.shape[1]):
                continue
            if not (0 <= dst_ty < tiles.shape[0] and 0 <= dst_tx < tiles.shape[1]):
                continue
            
            src_type = tiles[src_ty, src_tx]
            dst_type = tiles[dst_ty, dst_tx]
            
            if src_type == 1 or dst_type == 1:
                solid_edges.append({
                    **edge,
                    'src_type': src_type,
                    'dst_type': dst_type
                })
        
        if solid_edges:
            print(f"\n⚠ Found {len(solid_edges)} edges involving solid tiles (Type 1):")
            for i, edge in enumerate(solid_edges[:10]):
                print(f"  #{i+1}: {edge['src']} (type {edge['src_type']}) → {edge['dst']} (type {edge['dst_type']})")
        else:
            print("\n✓ No edges involving solid tiles found")
    
    env.close()
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
    
    return adjacency, problematic if 'problematic' in locals() else []


if __name__ == "__main__":
    map_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    analyze_graph_building(map_id)

