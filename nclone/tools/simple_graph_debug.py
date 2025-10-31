"""
Simple graph debug tool that works with test_environment
"""

import sys
import os
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nclone.graph.reachability.fast_graph_builder import FastGraphBuilder
from nclone.map_loader import MapLoader


def debug_graph_for_map(map_id: int):
    """Debug graph building for a specific map."""
    
    print(f"\n{'='*70}")
    print(f"DEBUGGING GRAPH FOR MAP {map_id}")
    print(f"{'='*70}\n")
    
    # Load map
    loader = MapLoader()
    level_data = loader.load_map(map_id)
    tiles = level_data["tiles"]
    
    print(f"Tiles shape: {tiles.shape}")
    print(f"Tile types present: {sorted(set(tiles.flatten()))}\n")
    
    # Count tiles by type
    print("Tile distribution:")
    unique, counts = np.unique(tiles, return_counts=True)
    for tile_type, count in zip(unique, counts):
        pct = 100 * count / tiles.size
        print(f"  Type {tile_type:2}: {count:5} tiles ({pct:5.2f}%)")
    
    # Build graph with debug
    builder = FastGraphBuilder(debug=True)
    
    # Use player start position
    start_pos = level_data.get("start_position", (100, 100))
    
    print(f"\nBuilding graph from start position: {start_pos}")
    graph_data = builder.build_graph(level_data, ninja_pos=start_pos)
    
    adjacency = graph_data["adjacency"]
    
    print(f"\n{'='*70}")
    print("GRAPH STATISTICS")
    print(f"{'='*70}")
    print(f"Total nodes: {len(adjacency)}")
    
    if adjacency:
        total_edges = sum(len(neighbors) for neighbors in adjacency.values())
        print(f"Total edges: {total_edges}")
        print(f"Average degree: {total_edges / len(adjacency):.2f}")
        
        # Count nodes by tile type
        nodes_by_type = {}
        for node_pos in adjacency.keys():
            nx, ny = node_pos
            tx, ty = nx // 24, ny // 24
            if 0 <= ty < tiles.shape[0] and 0 <= tx < tiles.shape[1]:
                tt = tiles[ty, tx]
                nodes_by_type[tt] = nodes_by_type.get(tt, 0) + 1
        
        print(f"\nNodes by tile type:")
        for tt in sorted(nodes_by_type.keys()):
            print(f"  Type {tt:2}: {nodes_by_type[tt]:4} nodes")
        
        # Analyze edge directions
        edge_directions = {
            'N': 0, 'NE': 0, 'E': 0, 'SE': 0,
            'S': 0, 'SW': 0, 'W': 0, 'NW': 0, 'SAME': 0
        }
        
        for src_pos, neighbors in adjacency.items():
            sx, sy = src_pos
            for dst_pos, cost in neighbors:
                dx_pix = dst_pos[0] - sx
                dy_pix = dst_pos[1] - sy
                
                # Determine direction
                if dx_pix == 0 and dy_pix == 0:
                    edge_directions['SAME'] += 1
                elif dx_pix == 0 and dy_pix < 0:
                    edge_directions['N'] += 1
                elif dx_pix > 0 and dy_pix < 0:
                    edge_directions['NE'] += 1
                elif dx_pix > 0 and dy_pix == 0:
                    edge_directions['E'] += 1
                elif dx_pix > 0 and dy_pix > 0:
                    edge_directions['SE'] += 1
                elif dx_pix == 0 and dy_pix > 0:
                    edge_directions['S'] += 1
                elif dx_pix < 0 and dy_pix > 0:
                    edge_directions['SW'] += 1
                elif dx_pix < 0 and dy_pix == 0:
                    edge_directions['W'] += 1
                elif dx_pix < 0 and dy_pix < 0:
                    edge_directions['NW'] += 1
        
        print(f"\nEdges by direction:")
        for direction, count in edge_directions.items():
            print(f"  {direction:4}: {count:5} edges")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
    
    return adjacency


if __name__ == "__main__":
    map_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    debug_graph_for_map(map_id)

