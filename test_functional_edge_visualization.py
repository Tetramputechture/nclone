#!/usr/bin/env python3

"""
Test functional edge visualization in the debug overlay.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

import pygame
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.common import EdgeType

def test_functional_edge_visualization():
    """Test functional edge visualization."""
    print("=" * 80)
    print("🔍 TESTING FUNCTIONAL EDGE VISUALIZATION")
    print("=" * 80)
    
    # Initialize pygame
    pygame.init()
    
    # Create environment with debug overlay enabled
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_debug_overlay=True
    )
    env.reset()
    
    print(f"📍 Map: doortest")
    print(f"🎮 Total entities: {len(env.entities)}")
    
    # Check debug overlay configuration
    print(f"🔧 Debug overlay enabled in environment: {env._enable_debug_overlay}")
    print(f"🔧 NPlayHeadless attributes: {[attr for attr in dir(env.nplay_headless) if not attr.startswith('_')]}")
    
    # Check if debug overlay renderer exists
    if hasattr(env.nplay_headless, 'renderer') and hasattr(env.nplay_headless.renderer, 'debug_overlay_renderer'):
        debug_overlay = env.nplay_headless.renderer.debug_overlay_renderer
        print(f"🔧 Debug overlay renderer found: {debug_overlay is not None}")
    else:
        print("❌ Debug overlay renderer not found")
    
    # Render a frame to trigger graph visualization
    print("\n🎨 Rendering frame to test visualization...")
    frame = env.render()
    print(f"📸 Frame rendered: {frame.shape if frame is not None else 'None'}")
    
    # Check if functional edges are in the graph
    if hasattr(env, 'level_data') and env.level_data:
        from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
        
        builder = HierarchicalGraphBuilder()
        ninja_pos = (env.nplay_headless.sim.ninja.xpos, env.nplay_headless.sim.ninja.ypos)
        hierarchical_graph = builder.build_graph(env.level_data, ninja_pos)
        graph = hierarchical_graph.sub_cell_graph
        
        # Count functional edges
        functional_edge_count = 0
        for i in range(graph.num_edges):
            if graph.edge_types[i] == EdgeType.FUNCTIONAL:
                functional_edge_count += 1
        
        print(f"🔗 Functional edges in graph: {functional_edge_count}")
        
        if functional_edge_count > 0:
            print("✅ Functional edges exist in graph data")
            print("🔍 If not visible, issue is in visualization rendering")
        else:
            print("❌ No functional edges found in graph data")
    
    # Test with different debug overlay settings
    print("\n🔧 Testing debug overlay edge type settings...")
    
    if debug_overlay and hasattr(debug_overlay, 'toggle_edge_type'):
        # Try to enable functional edges specifically
        debug_overlay.toggle_edge_type(EdgeType.FUNCTIONAL, True)
        print("✅ Explicitly enabled functional edges")
        
        # Render another frame
        frame2 = env.render()
        print(f"📸 Second frame rendered: {frame2.shape if frame2 is not None else 'None'}")
    
    pygame.quit()
    print("\n✅ Test completed")

if __name__ == "__main__":
    test_functional_edge_visualization()