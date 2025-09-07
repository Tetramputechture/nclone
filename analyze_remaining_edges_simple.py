#!/usr/bin/env python3
"""
Simple analysis of remaining walkable edges in solid tiles.
"""

import pygame
import numpy as np
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.common import EdgeType, SUB_CELL_SIZE
from nclone.constants.physics_constants import TILE_PIXEL_SIZE

def analyze_remaining_edges():
    """Analyze the remaining walkable edges in solid tiles."""
    
    print("🔍 ANALYZING REMAINING WALKABLE EDGES")
    print("=" * 50)
    
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
    
    # Get tile data
    tiles = level_data.tiles
    print(f"📐 Level dimensions: {tiles.shape[1]} x {tiles.shape[0]} tiles")
    
    # Find walkable edges and check their tile positions
    walkable_edges = []
    potentially_invalid = []
    
    for i in range(graph.num_edges):
        if graph.edge_mask[i] == 1 and graph.edge_types[i] == EdgeType.WALK.value:
            src_idx = graph.edge_index[0, i]
            tgt_idx = graph.edge_index[1, i]
            
            # Get positions
            src_x = graph.node_features[src_idx, 0]
            src_y = graph.node_features[src_idx, 1]
            tgt_x = graph.node_features[tgt_idx, 0]
            tgt_y = graph.node_features[tgt_idx, 1]
            
            walkable_edges.append({
                'edge_idx': i,
                'src_pos': (src_x, src_y),
                'tgt_pos': (tgt_x, tgt_y),
                'distance': np.sqrt((tgt_x - src_x)**2 + (tgt_y - src_y)**2)
            })
            
            # Check tile types at both positions
            # Convert pixel coordinates to tile coordinates
            src_tile_x = int((src_x - TILE_PIXEL_SIZE) // TILE_PIXEL_SIZE)
            src_tile_y = int((src_y - TILE_PIXEL_SIZE) // TILE_PIXEL_SIZE)
            tgt_tile_x = int((tgt_x - TILE_PIXEL_SIZE) // TILE_PIXEL_SIZE)
            tgt_tile_y = int((tgt_y - TILE_PIXEL_SIZE) // TILE_PIXEL_SIZE)
            
            # Check bounds
            if (0 <= src_tile_x < tiles.shape[1] and 0 <= src_tile_y < tiles.shape[0] and
                0 <= tgt_tile_x < tiles.shape[1] and 0 <= tgt_tile_y < tiles.shape[0]):
                
                src_tile_type = tiles[src_tile_y, src_tile_x]
                tgt_tile_type = tiles[tgt_tile_y, tgt_tile_x]
                
                # Check if either endpoint is in a solid tile (type 1 or >33)
                src_solid = (src_tile_type == 1 or src_tile_type > 33)
                tgt_solid = (tgt_tile_type == 1 or tgt_tile_type > 33)
                
                if src_solid or tgt_solid:
                    potentially_invalid.append({
                        'edge_idx': i,
                        'src_pos': (src_x, src_y),
                        'tgt_pos': (tgt_x, tgt_y),
                        'src_tile': (src_tile_x, src_tile_y, src_tile_type),
                        'tgt_tile': (tgt_tile_x, tgt_tile_y, tgt_tile_type),
                        'src_solid': src_solid,
                        'tgt_solid': tgt_solid,
                        'distance': np.sqrt((tgt_x - src_x)**2 + (tgt_y - src_y)**2)
                    })
    
    print(f"🟢 Total walkable edges: {len(walkable_edges)}")
    print(f"🔴 Potentially invalid edges: {len(potentially_invalid)}")
    
    if len(potentially_invalid) > 0:
        # Categorize the potentially invalid edges
        both_solid = [e for e in potentially_invalid if e['src_solid'] and e['tgt_solid']]
        src_only_solid = [e for e in potentially_invalid if e['src_solid'] and not e['tgt_solid']]
        tgt_only_solid = [e for e in potentially_invalid if not e['src_solid'] and e['tgt_solid']]
        
        print(f"  📊 Both endpoints in solid tiles: {len(both_solid)}")
        print(f"  📊 Source only in solid tile: {len(src_only_solid)}")
        print(f"  📊 Target only in solid tile: {len(tgt_only_solid)}")
        
        # Analyze tile types
        print(f"\n📊 TILE TYPE ANALYSIS:")
        
        tile_type_counts = {}
        for edge in potentially_invalid:
            src_type = edge['src_tile'][2]
            tgt_type = edge['tgt_tile'][2]
            
            tile_type_counts[src_type] = tile_type_counts.get(src_type, 0) + 1
            tile_type_counts[tgt_type] = tile_type_counts.get(tgt_type, 0) + 1
        
        print("  Tile types involved:")
        for tile_type, count in sorted(tile_type_counts.items()):
            solid_status = "SOLID" if (tile_type == 1 or tile_type > 33) else "SHAPED"
            print(f"    Tile {tile_type} ({solid_status}): {count} occurrences")
        
        # Sample some edges for detailed analysis
        print(f"\n🔍 SAMPLE OF POTENTIALLY INVALID EDGES:")
        
        for i, edge in enumerate(potentially_invalid[:5]):
            print(f"\n  Edge {i+1}:")
            print(f"    Source: {edge['src_pos']} -> Tile {edge['src_tile']} (solid: {edge['src_solid']})")
            print(f"    Target: {edge['tgt_pos']} -> Tile {edge['tgt_tile']} (solid: {edge['tgt_solid']})")
            print(f"    Distance: {edge['distance']:.1f} pixels")
            
            if edge['src_solid'] != edge['tgt_solid']:
                print(f"    🟡 BOUNDARY CROSSING: Likely legitimate")
            else:
                print(f"    🔴 BOTH SOLID: Likely invalid")
        
        # Distance analysis
        distances = [e['distance'] for e in potentially_invalid]
        print(f"\n📏 DISTANCE ANALYSIS:")
        print(f"  Range: {min(distances):.1f} - {max(distances):.1f} pixels")
        print(f"  Average: {np.mean(distances):.1f} pixels")
        print(f"  Sub-cell size: {SUB_CELL_SIZE} pixels")
        
        # Recommendations
        print(f"\n💡 ANALYSIS RESULTS:")
        
        boundary_crossings = len(src_only_solid) + len(tgt_only_solid)
        truly_invalid = len(both_solid)
        
        print(f"  🟡 {boundary_crossings} edges are boundary crossings")
        print(f"     These connect solid tiles to traversable areas - likely legitimate")
        
        print(f"  🔴 {truly_invalid} edges have both endpoints in solid tiles")
        print(f"     These are likely invalid and should be filtered out")
        
        improvement_percentage = (436 - len(potentially_invalid)) / 436 * 100
        print(f"  ✅ Overall improvement: {improvement_percentage:.1f}% reduction in invalid edges")
        print(f"     (From 436 to {len(potentially_invalid)} potentially invalid edges)")
        
        if truly_invalid > 0:
            print(f"  ⚠️  Consider additional filtering for edges with both endpoints in solid tiles")
        
    else:
        print("✅ No potentially invalid edges found!")
    
    pygame.quit()
    print("\n✅ Analysis completed")

if __name__ == "__main__":
    analyze_remaining_edges()