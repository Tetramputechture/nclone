#!/usr/bin/env python3
"""
Test the full visualization with all edge types enabled after the fix.
"""

import pygame
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.visualization import GraphVisualizer, VisualizationConfig

def test_full_visualization():
    """Test full visualization with all edge types enabled."""
    
    print("🔍 TESTING FULL VISUALIZATION AFTER FIX")
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
    print(f"🎮 Total entities: {len(env.entities)}")
    
    # Build graph
    builder = HierarchicalGraphBuilder()
    level_data = env.level_data
    
    # Get ninja position
    ninja_position = env.nplay_headless.ninja_position()
    ninja_pos = (ninja_position[0], ninja_position[1])
    
    hierarchical_graph = builder.build_graph(level_data, ninja_pos)
    graph = hierarchical_graph.sub_cell_graph
    print(f"📊 Sub-cell Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Create visualization config with ALL edge types enabled
    config = VisualizationConfig(
        show_walk_edges=True,
        show_jump_edges=True,
        show_fall_edges=True,
        show_wall_slide_edges=True,
        show_one_way_edges=True,
        show_functional_edges=True,  # This should now work!
        show_nodes=True,
        show_grid=True,
        alpha=0.8
    )
    
    print("✅ Config: All edge types enabled")
    
    # Create visualizer
    visualizer = GraphVisualizer(config)
    
    try:
        # Create standalone visualization
        width, height = 1200, 800
        surface = visualizer.create_standalone_visualization(
            graph, width=width, height=height
        )
        
        print("✅ Full visualization rendered successfully")
        
        # Save the visualization
        pygame.image.save(surface, "test_full_visualization_fixed.png")
        print("💾 Saved full visualization to test_full_visualization_fixed.png")
        
    except Exception as e:
        print(f"❌ Error rendering full visualization: {e}")
        import traceback
        traceback.print_exc()
    
    pygame.quit()
    print("\n✅ Test completed")

if __name__ == "__main__":
    test_full_visualization()