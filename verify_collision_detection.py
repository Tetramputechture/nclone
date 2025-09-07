#!/usr/bin/env python3
"""
Verify that the collision detection is working correctly by testing it directly.
"""

import pygame
import numpy as np
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.common import EdgeType, SUB_CELL_SIZE
from nclone.graph.precise_collision import PreciseTileCollision
from nclone.constants.physics_constants import TILE_PIXEL_SIZE

def verify_collision_detection():
    """Verify collision detection by testing it directly on problematic edges."""
    
    print("🔍 VERIFYING COLLISION DETECTION")
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
    
    # Initialize collision detector (same as used in EdgeBuilder)
    collision_detector = PreciseTileCollision()
    tiles = level_data.tiles
    
    # Find walkable edges and test them with collision detection
    walkable_edges = []
    collision_mismatches = []
    
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
                'tgt_pos': (tgt_x, tgt_y)
            })
            
            # Test with collision detector
            is_traversable = collision_detector.is_path_traversable(
                src_x, src_y, tgt_x, tgt_y, tiles
            )
            
            # If the edge exists in the graph but collision detector says it's not traversable,
            # that's a mismatch
            if not is_traversable:
                collision_mismatches.append({
                    'edge_idx': i,
                    'src_pos': (src_x, src_y),
                    'tgt_pos': (tgt_x, tgt_y),
                    'distance': np.sqrt((tgt_x - src_x)**2 + (tgt_y - src_y)**2)
                })
    
    print(f"🟢 Total walkable edges in graph: {len(walkable_edges)}")
    print(f"🔴 Edges that collision detector says are NOT traversable: {len(collision_mismatches)}")
    
    if len(collision_mismatches) > 0:
        print(f"\n⚠️  COLLISION DETECTION MISMATCHES FOUND!")
        print(f"   These edges exist in the graph but collision detector says they're not traversable")
        
        # Sample some mismatches
        print(f"\n🔍 SAMPLE MISMATCHES:")
        for i, edge in enumerate(collision_mismatches[:5]):
            print(f"\n  Mismatch {i+1}:")
            print(f"    Source: {edge['src_pos']}")
            print(f"    Target: {edge['tgt_pos']}")
            print(f"    Distance: {edge['distance']:.1f} pixels")
            
            # Check tile types
            src_x, src_y = edge['src_pos']
            tgt_x, tgt_y = edge['tgt_pos']
            
            src_tile_x = int((src_x - TILE_PIXEL_SIZE) // TILE_PIXEL_SIZE)
            src_tile_y = int((src_y - TILE_PIXEL_SIZE) // TILE_PIXEL_SIZE)
            tgt_tile_x = int((tgt_x - TILE_PIXEL_SIZE) // TILE_PIXEL_SIZE)
            tgt_tile_y = int((tgt_y - TILE_PIXEL_SIZE) // TILE_PIXEL_SIZE)
            
            if (0 <= src_tile_x < tiles.shape[1] and 0 <= src_tile_y < tiles.shape[0] and
                0 <= tgt_tile_x < tiles.shape[1] and 0 <= tgt_tile_y < tiles.shape[0]):
                
                src_tile_type = tiles[src_tile_y, src_tile_x]
                tgt_tile_type = tiles[tgt_tile_y, tgt_tile_x]
                
                print(f"    Source tile: ({src_tile_x}, {src_tile_y}) type {src_tile_type}")
                print(f"    Target tile: ({tgt_tile_x}, {tgt_tile_y}) type {tgt_tile_type}")
        
        print(f"\n💡 This suggests there might be an issue with the collision detection")
        print(f"   or the graph building process is not using it correctly.")
        
    else:
        print(f"\n✅ NO MISMATCHES FOUND!")
        print(f"   All edges in the graph pass collision detection.")
        print(f"   The collision detection system is working correctly.")
    
    # Test some known solid positions
    print(f"\n🧪 TESTING KNOWN SOLID POSITIONS:")
    
    # Test some positions that should definitely be solid
    test_positions = [
        # Center of solid tiles
        (TILE_PIXEL_SIZE * 2, TILE_PIXEL_SIZE * 2),  # Should be solid
        (TILE_PIXEL_SIZE * 3, TILE_PIXEL_SIZE * 3),  # Should be solid
        # Center of empty space
        (TILE_PIXEL_SIZE * 10, TILE_PIXEL_SIZE * 10),  # Should be traversable
    ]
    
    for i, (x, y) in enumerate(test_positions):
        # Test a short movement from this position
        test_x = x + 3  # Move 3 pixels right
        test_y = y
        
        is_traversable = collision_detector.is_path_traversable(x, y, test_x, test_y, tiles)
        
        # Check tile type
        tile_x = int((x - TILE_PIXEL_SIZE) // TILE_PIXEL_SIZE)
        tile_y = int((y - TILE_PIXEL_SIZE) // TILE_PIXEL_SIZE)
        
        if 0 <= tile_x < tiles.shape[1] and 0 <= tile_y < tiles.shape[0]:
            tile_type = tiles[tile_y, tile_x]
            print(f"  Test {i+1}: ({x}, {y}) -> ({test_x}, {test_y})")
            print(f"    Tile type: {tile_type}, Traversable: {is_traversable}")
        else:
            print(f"  Test {i+1}: ({x}, {y}) -> OUT OF BOUNDS")
    
    pygame.quit()
    print("\n✅ Verification completed")

if __name__ == "__main__":
    verify_collision_detection()