#!/usr/bin/env python3
"""
Debug node features for functional edge nodes.
"""

import pygame
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.common import EdgeType, NodeType

def debug_node_features():
    """Debug node features for functional edge nodes."""
    
    print("🔍 DEBUGGING NODE FEATURES FOR FUNCTIONAL EDGES")
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
    
    # Find functional edges and their nodes
    functional_edges = []
    for i in range(graph.num_edges):
        if graph.edge_mask[i] == 1:  # Valid edge
            edge_type = graph.edge_types[i]
            if edge_type == EdgeType.FUNCTIONAL.value:
                src_idx = graph.edge_index[0, i]
                tgt_idx = graph.edge_index[1, i]
                functional_edges.append((i, src_idx, tgt_idx, edge_type))
    
    print(f"🟡 FUNCTIONAL EDGES FOUND: {len(functional_edges)}")
    
    # Calculate sub-grid nodes count
    from nclone.graph.common import SUB_GRID_WIDTH, SUB_GRID_HEIGHT
    sub_grid_nodes_count = SUB_GRID_WIDTH * SUB_GRID_HEIGHT
    print(f"📐 Sub-grid nodes count: {sub_grid_nodes_count}")
    
    for i, (edge_idx, src_idx, tgt_idx, edge_type) in enumerate(functional_edges):
        print(f"\n🟡 FUNCTIONAL EDGE {i+1}:")
        print(f"   Edge index: {edge_idx}")
        print(f"   Source node: {src_idx}, Target node: {tgt_idx}")
        
        # Check if nodes are sub-grid or entity nodes
        src_is_subgrid = src_idx < sub_grid_nodes_count
        tgt_is_subgrid = tgt_idx < sub_grid_nodes_count
        
        print(f"   Source is sub-grid: {src_is_subgrid}")
        print(f"   Target is sub-grid: {tgt_is_subgrid}")
        
        # Analyze source node
        print(f"\n   📍 SOURCE NODE {src_idx}:")
        if src_idx < graph.num_nodes and graph.node_mask[src_idx] == 1:
            src_features = graph.node_features[src_idx]
            src_node_type = graph.node_types[src_idx]
            print(f"      Node type: {src_node_type} ({NodeType(src_node_type).name})")
            print(f"      Features shape: {src_features.shape}")
            print(f"      Features (first 10): {src_features[:10]}")
            
            if not src_is_subgrid:
                # Entity node - check position extraction
                tile_type_dim = 38
                entity_type_dim = 30
                state_offset = tile_type_dim + 4 + entity_type_dim
                print(f"      State offset: {state_offset}")
                
                if len(src_features) > state_offset + 2:
                    norm_x = float(src_features[state_offset + 1])
                    norm_y = float(src_features[state_offset + 2])
                    print(f"      Normalized position: ({norm_x}, {norm_y})")
                    
                    from nclone.constants.physics_constants import FULL_MAP_WIDTH_PX, FULL_MAP_HEIGHT_PX
                    x = norm_x * float(FULL_MAP_WIDTH_PX)
                    y = norm_y * float(FULL_MAP_HEIGHT_PX)
                    print(f"      World position: ({x}, {y})")
                else:
                    print(f"      ❌ Features too short for position extraction")
        else:
            print(f"      ❌ Invalid node or masked")
        
        # Analyze target node
        print(f"\n   🎯 TARGET NODE {tgt_idx}:")
        if tgt_idx < graph.num_nodes and graph.node_mask[tgt_idx] == 1:
            tgt_features = graph.node_features[tgt_idx]
            tgt_node_type = graph.node_types[tgt_idx]
            print(f"      Node type: {tgt_node_type} ({NodeType(tgt_node_type).name})")
            print(f"      Features shape: {tgt_features.shape}")
            print(f"      Features (first 10): {tgt_features[:10]}")
            
            if not tgt_is_subgrid:
                # Entity node - check position extraction
                tile_type_dim = 38
                entity_type_dim = 30
                state_offset = tile_type_dim + 4 + entity_type_dim
                print(f"      State offset: {state_offset}")
                
                if len(tgt_features) > state_offset + 2:
                    norm_x = float(tgt_features[state_offset + 1])
                    norm_y = float(tgt_features[state_offset + 2])
                    print(f"      Normalized position: ({norm_x}, {norm_y})")
                    
                    from nclone.constants.physics_constants import FULL_MAP_WIDTH_PX, FULL_MAP_HEIGHT_PX
                    x = norm_x * float(FULL_MAP_WIDTH_PX)
                    y = norm_y * float(FULL_MAP_HEIGHT_PX)
                    print(f"      World position: ({x}, {y})")
                else:
                    print(f"      ❌ Features too short for position extraction")
        else:
            print(f"      ❌ Invalid node or masked")
    
    pygame.quit()
    print("\n✅ Debug completed")

if __name__ == "__main__":
    debug_node_features()