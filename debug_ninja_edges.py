#!/usr/bin/env python3
"""
Debug ninja edge creation specifically
"""

import os
import sys
import numpy as np

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.constants.entity_types import EntityType
from nclone.graph.common import SUB_CELL_SIZE

def debug_ninja_edges():
    print("=" * 60)
    print("NINJA EDGE CREATION DEBUG")
    print("=" * 60)
    
    # Create environment
    env = BasicLevelNoGold(render_mode="rgb_array", enable_frame_stack=False, enable_debug_overlay=False, eval_mode=False, seed=42)
    env.reset()
    ninja_position = env.nplay_headless.ninja_position()
    level_data = env.level_data
    
    print(f"Ninja position: {ninja_position}")
    print(f"Level size: {level_data.width}x{level_data.height} tiles")
    
    # Build graph
    builder = HierarchicalGraphBuilder()
    hierarchical_graph = builder.build_graph(level_data, ninja_position)
    graph = hierarchical_graph.sub_cell_graph
    
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Find ninja node
    ninja_node = None
    ninja_coords = None
    ninja_candidates = []
    for node_idx in range(graph.num_nodes):
        if graph.node_mask[node_idx] == 1:  # Valid node
            node_x = graph.node_features[node_idx, 0]
            node_y = graph.node_features[node_idx, 1]
            dist = ((node_x - ninja_position[0])**2 + (node_y - ninja_position[1])**2)**0.5
            if dist < 10:  # Close enough
                ninja_candidates.append((node_idx, node_x, node_y, dist))
    
    print(f"Found {len(ninja_candidates)} ninja candidates:")
    for node_idx, x, y, dist in ninja_candidates:
        print(f"  Node {node_idx}: ({x:.1f}, {y:.1f}) distance={dist:.1f}")
    
    if ninja_candidates:
        # Use the closest one
        ninja_candidates.sort(key=lambda x: x[3])  # Sort by distance
        ninja_node, ninja_x, ninja_y, _ = ninja_candidates[0]
        ninja_coords = (ninja_x, ninja_y)
    
    if ninja_node is None:
        print("❌ Could not find ninja node")
        return False
    
    print(f"✅ Found ninja node: {ninja_node} at {ninja_coords}")
    
    # Calculate expected sub-grid position for ninja
    ninja_x, ninja_y = ninja_position
    ninja_sub_col = int(ninja_x // SUB_CELL_SIZE)
    ninja_sub_row = int(ninja_y // SUB_CELL_SIZE)
    
    print(f"Ninja sub-grid position: row={ninja_sub_row}, col={ninja_sub_col}")
    
    # Look for nearby sub-grid nodes
    nearby_nodes = []
    for node_idx in range(graph.num_nodes):
        if graph.node_mask[node_idx] == 1 and node_idx != ninja_node:
            node_x = graph.node_features[node_idx, 0]
            node_y = graph.node_features[node_idx, 1]
            
            # Calculate sub-grid position
            node_sub_col = int(node_x // SUB_CELL_SIZE)
            node_sub_row = int(node_y // SUB_CELL_SIZE)
            
            # Check if it's in the 3x3 area around ninja
            row_diff = abs(node_sub_row - ninja_sub_row)
            col_diff = abs(node_sub_col - ninja_sub_col)
            
            if row_diff <= 1 and col_diff <= 1:
                distance = ((node_x - ninja_x)**2 + (node_y - ninja_y)**2)**0.5
                nearby_nodes.append((node_idx, node_x, node_y, distance, node_sub_row, node_sub_col))
    
    print(f"Found {len(nearby_nodes)} nearby sub-grid nodes:")
    for node_idx, x, y, dist, row, col in nearby_nodes[:10]:  # Show first 10
        print(f"  Node {node_idx}: ({x:.1f}, {y:.1f}) distance={dist:.1f} sub_grid=({row}, {col})")
    
    # Check ninja node connectivity
    ninja_edges = []
    for i in range(graph.num_edges):
        if graph.edge_mask[i] == 1:
            src = graph.edge_index[0, i]
            dst = graph.edge_index[1, i]
            if src == ninja_node:
                ninja_edges.append((src, dst, "outgoing"))
            elif dst == ninja_node:
                ninja_edges.append((src, dst, "incoming"))
    
    print(f"Ninja node has {len(ninja_edges)} edges:")
    for src, dst, direction in ninja_edges:
        if direction == "outgoing":
            target_x = graph.node_features[dst, 0]
            target_y = graph.node_features[dst, 1]
            print(f"  {direction}: {src} -> {dst} at ({target_x:.1f}, {target_y:.1f})")
        else:
            source_x = graph.node_features[src, 0]
            source_y = graph.node_features[src, 1]
            print(f"  {direction}: {src} -> {dst} from ({source_x:.1f}, {source_y:.1f})")
    
    if len(ninja_edges) == 0:
        print("❌ Ninja node is isolated!")
        
        # Check if ninja is in a solid tile
        ninja_tile_x = int(ninja_x // 24)  # TILE_PIXEL_SIZE = 24
        ninja_tile_y = int(ninja_y // 24)
        
        if (0 <= ninja_tile_y < level_data.height and 0 <= ninja_tile_x < level_data.width):
            tile_value = level_data.tiles[ninja_tile_y, ninja_tile_x]
            print(f"Ninja is in tile ({ninja_tile_x}, {ninja_tile_y}) with value {tile_value}")
            if tile_value == 1:
                print("❌ Ninja is in a solid tile!")
            else:
                print("✅ Ninja is in a clear tile")
        else:
            print("❌ Ninja is outside map bounds!")
        
        return False
    else:
        print("✅ Ninja node is connected!")
        return True

if __name__ == "__main__":
    success = debug_ninja_edges()
    if success:
        print("\n🎉 NINJA EDGE DEBUG SUCCESSFUL!")
    else:
        print("\n❌ NINJA EDGE DEBUG FAILED!")
    
    sys.exit(0 if success else 1)