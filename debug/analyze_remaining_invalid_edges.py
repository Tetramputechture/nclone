#!/usr/bin/env python3
"""
Analyze the remaining 138 walkable edges in solid tiles to determine if they are legitimate.
"""

import pygame
import numpy as np
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.common import EdgeType
from nclone.graph.precise_collision import PreciseTileCollision

def analyze_remaining_invalid_edges():
    """Analyze the remaining walkable edges in solid tiles."""
    
    print("🔍 ANALYZING REMAINING INVALID WALKABLE EDGES")
    print("=" * 60)
    
    # Initialize pygame
    pygame.init()
    
    # Create environment and reset to load entities
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42,
        custom_map_path=None
    )
    env.reset()
    
    print(f"📍 Map: {env.current_map_name}")
    
    # Build graph
    builder = HierarchicalGraphBuilder()
    level_data = env.level_data
    
    # Get ninja position
    ninja_position = env.nplay_headless.ninja_position()
    ninja_pos = (ninja_position[0], ninja_position[1])
    
    hierarchical_graph = builder.build_graph(level_data, ninja_pos)
    graph = hierarchical_graph.sub_cell_graph
    print(f"📊 Sub-cell Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Initialize collision detector
    collision_detector = PreciseTileCollision()
    
    # Find walkable edges in solid tiles
    invalid_edges = []
    
    for i in range(graph.num_edges):
        if graph.edge_mask[i] == 1 and graph.edge_types[i] == EdgeType.WALK.value:
            src_idx = graph.edge_index[0, i]
            tgt_idx = graph.edge_index[1, i]
            
            # Get positions
            src_x = graph.node_features[src_idx, 0]
            src_y = graph.node_features[src_idx, 1]
            tgt_x = graph.node_features[tgt_idx, 0]
            tgt_y = graph.node_features[tgt_idx, 1]
            
            # Check if both endpoints are in solid tiles
            src_solid = collision_detector.is_position_solid(src_x, src_y)
            tgt_solid = collision_detector.is_position_solid(tgt_x, tgt_y)
            
            if src_solid or tgt_solid:
                invalid_edges.append({
                    'edge_idx': i,
                    'src_idx': src_idx,
                    'tgt_idx': tgt_idx,
                    'src_pos': (src_x, src_y),
                    'tgt_pos': (tgt_x, tgt_y),
                    'src_solid': src_solid,
                    'tgt_solid': tgt_solid,
                    'distance': np.sqrt((tgt_x - src_x)**2 + (tgt_y - src_y)**2)
                })
    
    print(f"🔴 INVALID WALKABLE EDGES FOUND: {len(invalid_edges)}")
    
    # Categorize the invalid edges
    both_solid = [e for e in invalid_edges if e['src_solid'] and e['tgt_solid']]
    src_only_solid = [e for e in invalid_edges if e['src_solid'] and not e['tgt_solid']]
    tgt_only_solid = [e for e in invalid_edges if not e['src_solid'] and e['tgt_solid']]
    
    print(f"  📊 Both endpoints solid: {len(both_solid)}")
    print(f"  📊 Source only solid: {len(src_only_solid)}")
    print(f"  📊 Target only solid: {len(tgt_only_solid)}")
    
    # Analyze distance distribution
    distances = [e['distance'] for e in invalid_edges]
    if distances:
        print(f"  📏 Distance range: {min(distances):.1f} - {max(distances):.1f} pixels")
        print(f"  📏 Average distance: {np.mean(distances):.1f} pixels")
    
    # Sample some edges for detailed analysis
    print(f"\n🔍 DETAILED ANALYSIS OF FIRST 10 INVALID EDGES:")
    
    for i, edge in enumerate(invalid_edges[:10]):
        print(f"\n  Edge {i+1}:")
        print(f"    Index: {edge['edge_idx']}")
        print(f"    Source: {edge['src_pos']} (solid: {edge['src_solid']})")
        print(f"    Target: {edge['tgt_pos']} (solid: {edge['tgt_solid']})")
        print(f"    Distance: {edge['distance']:.1f} pixels")
        
        # Check tile types at both positions
        src_tile = collision_detector.get_tile_at_position(edge['src_pos'][0], edge['src_pos'][1])
        tgt_tile = collision_detector.get_tile_at_position(edge['tgt_pos'][0], edge['tgt_pos'][1])
        print(f"    Source tile: {src_tile}")
        print(f"    Target tile: {tgt_tile}")
        
        # Check if this might be a legitimate boundary crossing
        if edge['src_solid'] != edge['tgt_solid']:
            print(f"    🟡 BOUNDARY CROSSING: One solid, one traversable")
        else:
            print(f"    🔴 BOTH SOLID: Likely invalid edge")
    
    # Check for patterns in tile types
    print(f"\n📊 TILE TYPE ANALYSIS:")
    
    src_tiles = {}
    tgt_tiles = {}
    
    for edge in invalid_edges:
        src_tile = collision_detector.get_tile_at_position(edge['src_pos'][0], edge['src_pos'][1])
        tgt_tile = collision_detector.get_tile_at_position(edge['tgt_pos'][0], edge['tgt_pos'][1])
        
        src_tiles[src_tile] = src_tiles.get(src_tile, 0) + 1
        tgt_tiles[tgt_tile] = tgt_tiles.get(tgt_tile, 0) + 1
    
    print("  Source tile types:")
    for tile_type, count in sorted(src_tiles.items()):
        print(f"    Tile {tile_type}: {count} edges")
    
    print("  Target tile types:")
    for tile_type, count in sorted(tgt_tiles.items()):
        print(f"    Tile {tile_type}: {count} edges")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    boundary_crossings = len(src_only_solid) + len(tgt_only_solid)
    truly_invalid = len(both_solid)
    
    print(f"  🟡 {boundary_crossings} edges are boundary crossings (likely legitimate)")
    print(f"  🔴 {truly_invalid} edges have both endpoints in solid tiles (likely invalid)")
    
    if truly_invalid > 0:
        print(f"  ⚠️  Consider improving collision detection for tile types with both endpoints solid")
    
    if boundary_crossings > 0:
        print(f"  ✅ Boundary crossings may be legitimate for movement between solid and traversable areas")
    
    pygame.quit()
    print("\n✅ Analysis completed")

if __name__ == "__main__":
    analyze_remaining_invalid_edges()