#!/usr/bin/env python3

"""
Debug graph visualization issues:
1. Green walkable edges in solid tiles
2. Missing yellow functional edges
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

import pygame
import numpy as np
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.visualization import GraphVisualizer, VisualizationConfig
from nclone.graph.common import EdgeType, NodeType
from nclone.constants import TILE_PIXEL_SIZE

def debug_graph_visualization_issues():
    """Debug both walkable edges in solid tiles and missing functional edges."""
    print("=" * 80)
    print("🔍 DEBUGGING GRAPH VISUALIZATION ISSUES")
    print("=" * 80)
    
    # Initialize pygame
    pygame.init()
    
    # Create environment
    env = BasicLevelNoGold(render_mode="rgb_array")
    env.reset()
    
    print(f"📍 Map: doortest")
    print(f"🗺️  Map size: {env.level_data.width}x{env.level_data.height} tiles")
    print(f"🎮 Total entities: {len(env.entities)}")
    
    # Build graph
    builder = HierarchicalGraphBuilder()
    ninja_pos = (env.nplay_headless.sim.ninja.xpos, env.nplay_headless.sim.ninja.ypos)
    hierarchical_graph = builder.build_graph(env.level_data, ninja_pos)
    graph = hierarchical_graph.sub_cell_graph
    
    print(f"📊 Sub-cell Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Analyze edge types
    print("\n" + "="*60)
    print("🔗 EDGE TYPE ANALYSIS")
    print("="*60)
    
    edge_type_counts = {}
    for i in range(graph.num_edges):
        edge_type = graph.edge_types[i]
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
    
    for edge_type, count in edge_type_counts.items():
        try:
            edge_name = EdgeType(edge_type).name
        except:
            edge_name = f"UNKNOWN_{edge_type}"
        print(f"  {edge_name}: {count} edges")
    
    # Check functional edges specifically
    functional_edges = []
    for i in range(graph.num_edges):
        if graph.edge_types[i] == EdgeType.FUNCTIONAL:
            src_idx = graph.edge_index[0, i]
            dst_idx = graph.edge_index[1, i]
            src_pos = graph.node_features[src_idx][:2]
            dst_pos = graph.node_features[dst_idx][:2]
            functional_edges.append((i, src_idx, dst_idx, src_pos, dst_pos))
    
    print(f"\n🟡 FUNCTIONAL EDGES: {len(functional_edges)}")
    for i, (edge_idx, src_idx, dst_idx, src_pos, dst_pos) in enumerate(functional_edges):
        print(f"  Edge {i+1}: {src_idx} → {dst_idx}")
        print(f"    Source: ({src_pos[0]:.1f}, {src_pos[1]:.1f})")
        print(f"    Target: ({dst_pos[0]:.1f}, {dst_pos[1]:.1f})")
    
    # Analyze walkable edges in solid tiles
    print("\n" + "="*60)
    print("🟢 WALKABLE EDGES IN SOLID TILES ANALYSIS")
    print("="*60)
    
    # Get tile data
    tile_data = env.level_data.tiles
    
    # Check walkable edges
    walkable_edges_in_solid = []
    for i in range(graph.num_edges):
        if graph.edge_types[i] == EdgeType.WALK:
            src_idx = graph.edge_index[0, i]
            dst_idx = graph.edge_index[1, i]
            
            # Skip entity nodes
            if graph.node_types[src_idx] == NodeType.ENTITY or graph.node_types[dst_idx] == NodeType.ENTITY:
                continue
            
            src_pos = graph.node_features[src_idx][:2]
            dst_pos = graph.node_features[dst_idx][:2]
            
            # Convert to tile coordinates
            src_tile_x = int(src_pos[0] // TILE_PIXEL_SIZE)
            src_tile_y = int(src_pos[1] // TILE_PIXEL_SIZE)
            dst_tile_x = int(dst_pos[0] // TILE_PIXEL_SIZE)
            dst_tile_y = int(dst_pos[1] // TILE_PIXEL_SIZE)
            
            # Check if both endpoints are in solid tiles
            src_in_bounds = 0 <= src_tile_x < env.level_data.width and 0 <= src_tile_y < env.level_data.height
            dst_in_bounds = 0 <= dst_tile_x < env.level_data.width and 0 <= dst_tile_y < env.level_data.height
            
            if src_in_bounds and dst_in_bounds:
                src_tile = tile_data[src_tile_y][src_tile_x]
                dst_tile = tile_data[dst_tile_y][dst_tile_x]
                
                # Check if tiles are solid (non-zero values typically indicate solid tiles)
                if src_tile != 0 or dst_tile != 0:
                    walkable_edges_in_solid.append({
                        'edge_idx': i,
                        'src_pos': src_pos,
                        'dst_pos': dst_pos,
                        'src_tile': (src_tile_x, src_tile_y, src_tile),
                        'dst_tile': (dst_tile_x, dst_tile_y, dst_tile)
                    })
    
    print(f"Found {len(walkable_edges_in_solid)} walkable edges in solid tiles")
    
    # Show first few examples
    for i, edge_info in enumerate(walkable_edges_in_solid[:5]):
        print(f"  Edge {i+1}:")
        print(f"    Positions: ({edge_info['src_pos'][0]:.1f}, {edge_info['src_pos'][1]:.1f}) → ({edge_info['dst_pos'][0]:.1f}, {edge_info['dst_pos'][1]:.1f})")
        print(f"    Tiles: ({edge_info['src_tile'][0]}, {edge_info['src_tile'][1]}) = {edge_info['src_tile'][2]} → ({edge_info['dst_tile'][0]}, {edge_info['dst_tile'][1]}) = {edge_info['dst_tile'][2]}")
    
    # Test visualization with all edge types enabled
    print("\n" + "="*60)
    print("🎨 TESTING VISUALIZATION WITH ALL EDGE TYPES")
    print("="*60)
    
    # Create visualization config with all edge types enabled
    config = VisualizationConfig(
        show_walk_edges=True,
        show_jump_edges=True,
        show_fall_edges=True,
        show_wall_slide_edges=True,
        show_one_way_edges=True,
        show_functional_edges=True,  # Explicitly enable functional edges
        show_nodes=True,
        show_grid=True,
        alpha=0.8
    )
    
    print(f"✅ Visualization config created:")
    print(f"  show_functional_edges: {config.show_functional_edges}")
    print(f"  show_walk_edges: {config.show_walk_edges}")
    
    # Create visualizer
    visualizer = GraphVisualizer(config)
    
    # Test rendering
    try:
        # Create a surface for rendering
        width, height = 1000, 600
        surface = pygame.Surface((width, height))
        
        # Render graph using the correct method
        surface = visualizer.create_standalone_visualization(
            graph, width=width, height=height
        )
        
        print("✅ Graph visualization rendered successfully")
        
        # Save the visualization
        pygame.image.save(surface, "/workspace/nclone/debug_graph_visualization.png")
        print("💾 Saved visualization to debug_graph_visualization.png")
        
    except Exception as e:
        print(f"❌ Error rendering graph: {e}")
        import traceback
        traceback.print_exc()
    
    pygame.quit()
    print("\n✅ Debug completed")

if __name__ == "__main__":
    debug_graph_visualization_issues()