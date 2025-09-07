#!/usr/bin/env python3

"""
Debug script to compare debug overlay vs isolated graph visualization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

import pygame
import numpy as np
import cv2
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold

def debug_overlay_comparison():
    """Compare debug overlay vs isolated graph visualization"""
    
    print("=== Debug Overlay vs Isolated Graph Comparison ===")
    
    # Initialize pygame
    pygame.init()
    pygame.display.set_mode((1200, 800))
    
    # Initialize environment exactly like test_environment.py
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
    
    # Get ninja position for analysis
    ninja_pos = env.nplay_headless.ninja_position()
    print(f"Ninja position: {ninja_pos}")
    
    # 1. Capture debug overlay frame (with G key functionality)
    print(f"\n1. Capturing debug overlay frame...")
    env.set_graph_debug_enabled(True)
    
    try:
        surface_with_overlay = env.render()
        if surface_with_overlay is not None:
            if hasattr(surface_with_overlay, 'get_size'):
                # Pygame surface
                surface_array = pygame.surfarray.array3d(surface_with_overlay)
                surface_array = surface_array.swapaxes(0, 1)
                surface_bgr = cv2.cvtColor(surface_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite("debug_overlay_frame.png", surface_bgr)
                print(f"✓ Saved debug overlay frame to: debug_overlay_frame.png")
            else:
                # Numpy array
                if len(surface_with_overlay.shape) == 3 and surface_with_overlay.shape[2] == 3:
                    surface_bgr = cv2.cvtColor(surface_with_overlay, cv2.COLOR_RGB2BGR)
                    cv2.imwrite("debug_overlay_frame.png", surface_bgr)
                    print(f"✓ Saved debug overlay frame to: debug_overlay_frame.png")
    except Exception as e:
        print(f"❌ Error capturing debug overlay: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Capture frame without overlay for comparison
    print(f"\n2. Capturing frame without overlay...")
    env.set_graph_debug_enabled(False)
    
    try:
        surface_no_overlay = env.render()
        if surface_no_overlay is not None:
            if hasattr(surface_no_overlay, 'get_size'):
                surface_array = pygame.surfarray.array3d(surface_no_overlay)
                surface_array = surface_array.swapaxes(0, 1)
                surface_bgr = cv2.cvtColor(surface_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite("no_overlay_frame.png", surface_bgr)
                print(f"✓ Saved no overlay frame to: no_overlay_frame.png")
            else:
                if len(surface_no_overlay.shape) == 3 and surface_no_overlay.shape[2] == 3:
                    surface_bgr = cv2.cvtColor(surface_no_overlay, cv2.COLOR_RGB2BGR)
                    cv2.imwrite("no_overlay_frame.png", surface_bgr)
                    print(f"✓ Saved no overlay frame to: no_overlay_frame.png")
    except Exception as e:
        print(f"❌ Error capturing no overlay frame: {e}")
    
    # 3. Create isolated graph visualization using the same approach as before
    print(f"\n3. Creating isolated graph visualization...")
    try:
        # Use the test_environment.py approach but save to file
        import subprocess
        result = subprocess.run([
            sys.executable, "nclone/test_environment.py", 
            "--visualize-graph", "--show-edges", "functional", "--save-graph"
        ], cwd="/workspace/nclone", capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✓ Isolated graph visualization created")
        else:
            print(f"❌ Error creating isolated graph: {result.stderr}")
    except subprocess.TimeoutExpired:
        print(f"✓ Isolated graph visualization process completed (timeout expected)")
    except Exception as e:
        print(f"❌ Error running isolated graph: {e}")
    
    # 4. Analyze debug info to understand the difference
    print(f"\n4. Analyzing debug info...")
    env.set_graph_debug_enabled(True)
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
        
        print(f"Functional edges in debug overlay: {functional_edge_count}")
        
        # Check ninja position in graph
        node_positions = graph_data.node_features[:, :2]
        from nclone.constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
        ninja_grid_x = ninja_pos[0] // MAP_TILE_WIDTH
        ninja_grid_y = ninja_pos[1] // MAP_TILE_HEIGHT
        
        # Find nodes close to ninja
        distances = np.sqrt((node_positions[:, 0] - ninja_grid_x)**2 + (node_positions[:, 1] - ninja_grid_y)**2)
        close_nodes = np.where(distances < 2.0)[0]
        print(f"Nodes within 2 grid units of ninja in debug overlay: {len(close_nodes)}")
        
        if len(close_nodes) > 0:
            print("Closest nodes to ninja in debug overlay:")
            for i, node_idx in enumerate(close_nodes[:3]):
                pos = node_positions[node_idx]
                dist = distances[node_idx]
                print(f"  Node {node_idx}: ({pos[0]:.1f}, {pos[1]:.1f}), distance: {dist:.2f}")
    
    pygame.quit()
    print(f"\n=== Comparison Complete ===")
    print(f"Files created:")
    print(f"  - debug_overlay_frame.png (debug overlay from test_environment.py)")
    print(f"  - no_overlay_frame.png (clean frame without overlay)")
    print(f"  - Any graph visualization files from isolated rendering")

if __name__ == "__main__":
    debug_overlay_comparison()