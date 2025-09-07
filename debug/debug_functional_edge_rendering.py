#!/usr/bin/env python3
"""
Debug functional edge rendering by tracing the visualization process.
"""

import pygame
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.visualization import GraphVisualizer, VisualizationConfig
from nclone.graph.common import EdgeType

def debug_functional_edge_rendering():
    """Debug functional edge rendering step by step."""
    
    print("🔍 DEBUGGING FUNCTIONAL EDGE RENDERING")
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
    
    # Find functional edges
    functional_edges = []
    for i in range(graph.num_edges):
        if graph.edge_mask[i] == 1:  # Valid edge
            edge_type = graph.edge_types[i]
            if edge_type == EdgeType.FUNCTIONAL.value:
                src_idx = graph.edge_index[0, i]
                tgt_idx = graph.edge_index[1, i]
                functional_edges.append((i, src_idx, tgt_idx, edge_type))
    
    print(f"🟡 FUNCTIONAL EDGES FOUND: {len(functional_edges)}")
    
    for i, (edge_idx, src_idx, tgt_idx, edge_type) in enumerate(functional_edges):
        src_x = graph.node_features[src_idx, 0]
        src_y = graph.node_features[src_idx, 1]
        tgt_x = graph.node_features[tgt_idx, 0]
        tgt_y = graph.node_features[tgt_idx, 1]
        
        print(f"  Edge {i+1}: idx={edge_idx}, type={edge_type}")
        print(f"    Source: node {src_idx} at ({src_x:.1f}, {src_y:.1f})")
        print(f"    Target: node {tgt_idx} at ({tgt_x:.1f}, {tgt_y:.1f})")
        print(f"    EdgeType.FUNCTIONAL.value = {EdgeType.FUNCTIONAL.value}")
    
    # Create visualization config with ONLY functional edges enabled
    config = VisualizationConfig(
        show_walk_edges=False,
        show_jump_edges=False,
        show_fall_edges=False,
        show_wall_slide_edges=False,
        show_one_way_edges=False,
        show_functional_edges=True,
        show_nodes=True,
        show_grid=True,
        alpha=1.0
    )
    
    print(f"\n✅ Config: show_functional_edges = {config.show_functional_edges}")
    
    # Create visualizer
    visualizer = GraphVisualizer(config)
    
    # Check edge colors
    print(f"🎨 Edge colors available: {list(visualizer.edge_colors.keys())}")
    functional_color = visualizer.edge_colors.get(EdgeType.FUNCTIONAL, None)
    print(f"🟡 FUNCTIONAL edge color: {functional_color}")
    
    # Test _should_show_edge_type method
    should_show = visualizer._should_show_edge_type(EdgeType.FUNCTIONAL)
    print(f"🔍 _should_show_edge_type(FUNCTIONAL): {should_show}")
    
    # Create a custom debug version of _draw_edges
    class DebugGraphVisualizer(GraphVisualizer):
        def _draw_edges(self, surface, graph_data, scale_x, scale_y, offset_x, offset_y, path_result=None):
            """Debug version of _draw_edges that traces functional edge drawing."""
            print(f"\n🔍 _draw_edges called with {graph_data.num_edges} edges")
            
            functional_edges_processed = 0
            functional_edges_drawn = 0
            
            for edge_idx in range(graph_data.num_edges):
                if graph_data.edge_mask[edge_idx] == 0:
                    continue
                
                src_node = graph_data.edge_index[0, edge_idx]
                dst_node = graph_data.edge_index[1, edge_idx]
                edge_type = EdgeType(graph_data.edge_types[edge_idx])
                
                if edge_type == EdgeType.FUNCTIONAL:
                    functional_edges_processed += 1
                    print(f"  🟡 Processing FUNCTIONAL edge {functional_edges_processed}: {edge_idx}")
                    print(f"     Edge type: {edge_type} (value: {edge_type.value})")
                    
                    # Check if this edge type should be shown
                    should_show = self._should_show_edge_type(edge_type)
                    print(f"     Should show: {should_show}")
                    
                    if not should_show:
                        print(f"     ❌ Skipping edge - should_show_edge_type returned False")
                        continue
                    
                    # Get node positions
                    src_x, src_y = self._get_node_position(graph_data, src_node)
                    dst_x, dst_y = self._get_node_position(graph_data, dst_node)
                    
                    src_screen_x = int(src_x * scale_x + offset_x)
                    src_screen_y = int(src_y * scale_y + offset_y)
                    dst_screen_x = int(dst_x * scale_x + offset_x)
                    dst_screen_y = int(dst_y * scale_y + offset_y)
                    
                    print(f"     World pos: ({src_x:.1f}, {src_y:.1f}) -> ({dst_x:.1f}, {dst_y:.1f})")
                    print(f"     Screen pos: ({src_screen_x}, {src_screen_y}) -> ({dst_screen_x}, {dst_screen_y})")
                    
                    # Check sub-pixel skip
                    distance = abs(dst_screen_x - src_screen_x) + abs(dst_screen_y - src_screen_y)
                    print(f"     Screen distance: {distance}")
                    
                    if distance < 1:
                        print(f"     ❌ Skipping edge - sub-pixel distance ({distance})")
                        continue
                    
                    # Get color
                    color = self.edge_colors.get(edge_type, (255, 255, 255, 255))
                    width = int(self.config.edge_width)
                    print(f"     Color: {color}, Width: {width}")
                    
                    # Draw edge
                    pygame.draw.line(
                        surface,
                        color,
                        (src_screen_x, src_screen_y),
                        (dst_screen_x, dst_screen_y),
                        width,
                    )
                    
                    functional_edges_drawn += 1
                    print(f"     ✅ Drew FUNCTIONAL edge {functional_edges_drawn}")
                
                # Call parent method for non-functional edges
                elif self._should_show_edge_type(edge_type):
                    # Get node positions
                    src_x, src_y = self._get_node_position(graph_data, src_node)
                    dst_x, dst_y = self._get_node_position(graph_data, dst_node)
                    
                    src_screen_x = int(src_x * scale_x + offset_x)
                    src_screen_y = int(src_y * scale_y + offset_y)
                    dst_screen_x = int(dst_x * scale_x + offset_x)
                    dst_screen_y = int(dst_y * scale_y + offset_y)
                    
                    # Skip sub-pixel edges
                    if abs(dst_screen_x - src_screen_x) + abs(dst_screen_y - src_screen_y) < 1:
                        continue
                    
                    color = self.edge_colors.get(edge_type, (255, 255, 255, 255))
                    width = int(self.config.edge_width)
                    
                    pygame.draw.line(
                        surface,
                        color,
                        (src_screen_x, src_screen_y),
                        (dst_screen_x, dst_screen_y),
                        width,
                    )
            
            print(f"🟡 FUNCTIONAL EDGES: {functional_edges_processed} processed, {functional_edges_drawn} drawn")
    
    # Use debug visualizer
    debug_visualizer = DebugGraphVisualizer(config)
    
    try:
        # Create standalone visualization
        width, height = 1200, 800
        surface = debug_visualizer.create_standalone_visualization(
            graph, width=width, height=height
        )
        
        print("✅ Debug visualization rendered successfully")
        
        # Save the visualization
        pygame.image.save(surface, "debug_functional_edge_rendering.png")
        print("💾 Saved debug visualization to debug_functional_edge_rendering.png")
        
    except Exception as e:
        print(f"❌ Error rendering debug visualization: {e}")
        import traceback
        traceback.print_exc()
    
    pygame.quit()
    print("\n✅ Debug completed")

if __name__ == "__main__":
    debug_functional_edge_rendering()