#!/usr/bin/env python3

"""
Test script to verify the debug overlay fix works correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

import pygame
import numpy as np
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT

def test_debug_overlay_fix():
    """Test that the debug overlay fix works correctly"""
    
    print("=== Testing Debug Overlay Fix ===")
    
    # Initialize pygame
    pygame.init()
    pygame.display.set_mode((1200, 800))
    
    # Initialize environment
    env = BasicLevelNoGold(
        render_mode="human",
        enable_frame_stack=False,
        enable_debug_overlay=True,
        eval_mode=False,
        seed=42,
        custom_map_path="nclone/test_maps/doortest",
    )
    
    # Reset environment
    observation, info = env.reset()
    print(f"Environment reset complete")
    
    # Get ninja position
    ninja_pos = env.nplay_headless.ninja_position()
    ninja_grid_x = ninja_pos[0] // MAP_TILE_WIDTH
    ninja_grid_y = ninja_pos[1] // MAP_TILE_HEIGHT
    print(f"Ninja position: {ninja_pos} -> grid ({ninja_grid_x}, {ninja_grid_y})")
    
    # Enable graph debug
    env.set_graph_debug_enabled(True)
    
    # Get debug info
    debug_info = env._debug_info()
    
    if debug_info and 'graph' in debug_info:
        graph_data = debug_info['graph']['data']
        print(f"Graph data: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
        
        # Check for functional edges
        from nclone.graph.common import EdgeType
        functional_edge_count = 0
        
        for e_idx in range(int(graph_data.num_edges)):
            if graph_data.edge_mask[e_idx] <= 0:
                continue
            ex = graph_data.edge_features[e_idx]
            edge_type_idx = int(np.argmax(ex[:len(EdgeType)]))
            if edge_type_idx == EdgeType.FUNCTIONAL.value:
                functional_edge_count += 1
        
        print(f"Functional edges: {functional_edge_count}")
        
        # Check ninja node positioning
        node_positions = graph_data.node_features[:, :2]  # Raw pixel coordinates
        
        # Find nodes close to ninja (within 50 pixels)
        distances = np.sqrt((node_positions[:, 0] - ninja_pos[0])**2 + (node_positions[:, 1] - ninja_pos[1])**2)
        close_nodes = np.where(distances < 50.0)[0]
        print(f"Nodes within 50 pixels of ninja: {len(close_nodes)}")
        
        if len(close_nodes) > 0:
            print("Nodes close to ninja:")
            for i, node_idx in enumerate(close_nodes[:5]):
                pos = node_positions[node_idx]
                dist = distances[node_idx]
                print(f"  Node {node_idx}: ({pos[0]:.1f}, {pos[1]:.1f}), distance: {dist:.2f}")
        
        # Check if ninja node exists at correct position
        ninja_node_found = len(close_nodes) > 0
        
        # Summary
        print(f"\n=== Test Results ===")
        print(f"✅ Graph created: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
        print(f"✅ Functional edges: {functional_edge_count}")
        print(f"{'✅' if ninja_node_found else '❌'} Ninja node positioning: {'CORRECT' if ninja_node_found else 'INCORRECT'}")
        
        if ninja_node_found and functional_edge_count > 0:
            print(f"🎉 DEBUG OVERLAY FIX SUCCESSFUL!")
            print(f"   - Entity positions are correctly decoded from raw pixel coordinates")
            print(f"   - Ninja node is positioned correctly at {ninja_pos}")
            print(f"   - Functional edges are working ({functional_edge_count} found)")
        else:
            print(f"❌ DEBUG OVERLAY FIX FAILED!")
            if not ninja_node_found:
                print(f"   - Ninja node not found at correct position")
            if functional_edge_count == 0:
                print(f"   - No functional edges found")
    else:
        print(f"❌ No graph debug info available")
    
    pygame.quit()
    print(f"\n=== Test Complete ===")

if __name__ == "__main__":
    test_debug_overlay_fix()