#!/usr/bin/env python3
"""
Test functional edge visualization by showing ONLY functional edges.
"""

import pygame
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.visualization import GraphVisualizer, VisualizationConfig

def test_functional_edges_only():
    """Test visualization with ONLY functional edges enabled."""
    
    print("🔍 TESTING FUNCTIONAL EDGES ONLY")
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
    entities = env._extract_graph_entities()
    
    # Get ninja position
    ninja_position = env.nplay_headless.ninja_position()
    ninja_pos = (ninja_position[0], ninja_position[1])
    
    hierarchical_graph = builder.build_graph(level_data, ninja_pos)
    
    # Use the sub-cell graph (highest resolution)
    graph = hierarchical_graph.sub_cell_graph
    print(f"📊 Sub-cell Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Count functional edges
    from nclone.graph.common import EdgeType
    functional_edge_count = 0
    functional_edges = []
    
    for i in range(graph.num_edges):
        if graph.edge_mask[i] == 1:  # Valid edge
            edge_type = graph.edge_types[i]
            if edge_type == EdgeType.FUNCTIONAL.value:
                functional_edge_count += 1
                src_idx = graph.edge_index[0, i]
                tgt_idx = graph.edge_index[1, i]
                functional_edges.append((i, src_idx, tgt_idx))
    
    print(f"🟡 FUNCTIONAL EDGES: {functional_edge_count}")
    
    for i, (edge_idx, src_idx, tgt_idx) in enumerate(functional_edges):
        # Extract node positions from node features
        src_x = graph.node_features[src_idx, 0]  # Assuming x is first feature
        src_y = graph.node_features[src_idx, 1]  # Assuming y is second feature
        tgt_x = graph.node_features[tgt_idx, 0]
        tgt_y = graph.node_features[tgt_idx, 1]
        
        print(f"  Edge {i+1}: {src_idx} → {tgt_idx}")
        print(f"    Source: ({src_x:.1f}, {src_y:.1f})")
        print(f"    Target: ({tgt_x:.1f}, {tgt_y:.1f})")
    
    # Create visualization config with ONLY functional edges enabled
    config = VisualizationConfig(
        show_walk_edges=False,      # Disable walkable edges
        show_jump_edges=False,      # Disable jump edges
        show_fall_edges=False,      # Disable fall edges
        show_wall_slide_edges=False, # Disable wall slide edges
        show_one_way_edges=False,   # Disable one-way edges
        show_functional_edges=True, # ONLY functional edges
        show_nodes=True,            # Show nodes for reference
        show_grid=True,             # Show grid for reference
        alpha=1.0                   # Full opacity
    )
    
    print("\n✅ Visualization config (FUNCTIONAL EDGES ONLY):")
    print(f"  show_functional_edges: {config.show_functional_edges}")
    print(f"  show_walk_edges: {config.show_walk_edges}")
    print(f"  alpha: {config.alpha}")
    
    # Create visualizer and render
    visualizer = GraphVisualizer(config)
    
    try:
        # Create standalone visualization
        width, height = 1200, 800
        surface = visualizer.create_standalone_visualization(
            graph, width=width, height=height
        )
        
        print("✅ Functional-only visualization rendered successfully")
        
        # Save the visualization
        pygame.image.save(surface, "debug_functional_edges_only.png")
        print("💾 Saved functional-only visualization to debug_functional_edges_only.png")
        
    except Exception as e:
        print(f"❌ Error rendering functional edges: {e}")
        import traceback
        traceback.print_exc()
    
    pygame.quit()
    print("\n✅ Test completed")

if __name__ == "__main__":
    test_functional_edges_only()