#!/usr/bin/env python3

"""
Debug script to export a frame from the test environment and analyze entity positions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

import pygame
import numpy as np
import cv2
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold

def debug_frame_export():
    """Export a frame and analyze entity positions"""
    
    print("=== Debug Frame Export and Analysis ===")
    
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
    
    # Enable graph debug
    env.set_graph_debug_enabled(True)
    print(f"Graph debug enabled")
    
    # Get debug info
    debug_info = env._debug_info()
    if debug_info and 'graph' in debug_info:
        graph_data = debug_info['graph']['data']
        print(f"Graph data: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
        
        # Analyze ninja position
        ninja_pos = debug_info.get('ninja_position', 'Unknown')
        ninja_vel = debug_info.get('ninja_velocity', 'Unknown')
        print(f"Ninja position: {ninja_pos}")
        print(f"Ninja velocity: {ninja_vel}")
        
        # Get raw ninja position from simulator
        raw_ninja_pos = env.nplay_headless.ninja_position()
        print(f"Raw ninja position from simulator: {raw_ninja_pos}")
        
        # Check map dimensions
        try:
            map_width = env.nplay_headless.sim.map_width()
            map_height = env.nplay_headless.sim.map_height()
            print(f"Map tile dimensions: {map_width} x {map_height}")
        except:
            print("Could not get map dimensions")
        
        # Analyze entity positions in graph
        print(f"\nAnalyzing entity positions in graph:")
        
        # Get node positions
        node_positions = graph_data.node_features[:, :2]  # First 2 features should be x, y
        print(f"Node position range: X=[{node_positions[:, 0].min():.1f}, {node_positions[:, 0].max():.1f}], Y=[{node_positions[:, 1].min():.1f}, {node_positions[:, 1].max():.1f}]")
        
        # Look for nodes near the ninja position
        ninja_x, ninja_y = raw_ninja_pos
        print(f"\nLooking for nodes near ninja position ({ninja_x:.1f}, {ninja_y:.1f}):")
        
        # Convert ninja position to grid coordinates (like the graph does)
        from nclone.constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
        ninja_grid_x = ninja_x // MAP_TILE_WIDTH
        ninja_grid_y = ninja_y // MAP_TILE_HEIGHT
        print(f"Ninja grid position: ({ninja_grid_x:.1f}, {ninja_grid_y:.1f})")
        
        # Find nodes close to ninja
        distances = np.sqrt((node_positions[:, 0] - ninja_grid_x)**2 + (node_positions[:, 1] - ninja_grid_y)**2)
        close_nodes = np.where(distances < 2.0)[0]  # Within 2 grid units
        print(f"Nodes within 2 grid units of ninja: {len(close_nodes)}")
        
        if len(close_nodes) > 0:
            print("Closest nodes to ninja:")
            for i, node_idx in enumerate(close_nodes[:5]):  # Show first 5
                pos = node_positions[node_idx]
                dist = distances[node_idx]
                print(f"  Node {node_idx}: ({pos[0]:.1f}, {pos[1]:.1f}), distance: {dist:.2f}")
        
        # Count functional edges
        from nclone.graph.common import EdgeType
        functional_edge_count = 0
        functional_edges = []
        
        for e_idx in range(int(graph_data.num_edges)):
            if graph_data.edge_mask[e_idx] <= 0:
                continue
            ex = graph_data.edge_features[e_idx]
            edge_type_idx = int(np.argmax(ex[:len(EdgeType)]))
            if edge_type_idx == EdgeType.FUNCTIONAL.value:
                src = int(graph_data.edge_index[0, e_idx])
                tgt = int(graph_data.edge_index[1, e_idx])
                functional_edges.append((src, tgt))
                functional_edge_count += 1
        
        print(f"\nFunctional edges: {functional_edge_count}")
        if functional_edges:
            print("Functional edge connections:")
            for i, (src, tgt) in enumerate(functional_edges[:8]):  # Show all functional edges
                src_pos = node_positions[src]
                tgt_pos = node_positions[tgt]
                print(f"  Edge {i+1}: Node {src} ({src_pos[0]:.1f}, {src_pos[1]:.1f}) -> Node {tgt} ({tgt_pos[0]:.1f}, {tgt_pos[1]:.1f})")
    
    # Render and export frame
    print(f"\nRendering and exporting frame...")
    try:
        # Render with graph overlay
        surface = env.render()
        
        if surface is not None:
            # Convert pygame surface to numpy array for saving
            if hasattr(surface, 'get_size'):
                # It's a pygame surface
                surface_array = pygame.surfarray.array3d(surface)
                surface_array = surface_array.swapaxes(0, 1)  # Swap x,y axes
                # Convert RGB to BGR for OpenCV
                surface_bgr = cv2.cvtColor(surface_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite("debug_frame_with_graph.png", surface_bgr)
                print(f"Saved frame with graph overlay to: debug_frame_with_graph.png")
            else:
                # It's already a numpy array
                if len(surface.shape) == 3 and surface.shape[2] == 3:
                    surface_bgr = cv2.cvtColor(surface, cv2.COLOR_RGB2BGR)
                    cv2.imwrite("debug_frame_with_graph.png", surface_bgr)
                    print(f"Saved frame with graph overlay to: debug_frame_with_graph.png")
                else:
                    print(f"Unexpected surface format: {surface.shape}")
        
        # Also render without graph overlay for comparison
        env.set_graph_debug_enabled(False)
        surface_no_graph = env.render()
        
        if surface_no_graph is not None:
            if hasattr(surface_no_graph, 'get_size'):
                surface_array = pygame.surfarray.array3d(surface_no_graph)
                surface_array = surface_array.swapaxes(0, 1)
                surface_bgr = cv2.cvtColor(surface_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite("debug_frame_no_graph.png", surface_bgr)
                print(f"Saved frame without graph overlay to: debug_frame_no_graph.png")
            else:
                if len(surface_no_graph.shape) == 3 and surface_no_graph.shape[2] == 3:
                    surface_bgr = cv2.cvtColor(surface_no_graph, cv2.COLOR_RGB2BGR)
                    cv2.imwrite("debug_frame_no_graph.png", surface_bgr)
                    print(f"Saved frame without graph overlay to: debug_frame_no_graph.png")
    
    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()
    
    pygame.quit()
    print("\n=== Debug Frame Export Complete ===")

if __name__ == "__main__":
    debug_frame_export()