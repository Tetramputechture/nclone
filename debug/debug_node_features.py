#!/usr/bin/env python3
"""
Debug script to investigate node feature extraction issues.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from nclone.nsim import Simulator
from nclone.sim_config import SimConfig
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from validate_with_bblock_test import load_bblock_test_map, create_level_data_from_simulator

def debug_node_features():
    """Debug node feature extraction."""
    print("=== DEBUGGING NODE FEATURES ===")
    
    # Load the bblock_test map
    map_data = load_bblock_test_map()
    if map_data is None:
        return
    
    # Create a simulator to extract level data
    config = SimConfig()
    config.level_name = "bblock_test"
    
    try:
        sim = Simulator(config)
        level_data_obj = create_level_data_from_simulator(sim)
        print(f"✅ Level data: {level_data_obj.tiles.shape} tiles, {len(level_data_obj.entities)} entities")
    except Exception as e:
        print(f"❌ Failed to create simulator: {e}")
        return
    
    # Use default ninja position
    ninja_pos = (36, 564)
    print(f"Ninja position: {ninja_pos}")
    
    # Build graph
    print("Building hierarchical graph...")
    graph_builder = HierarchicalGraphBuilder()
    graph_data = graph_builder.build_graph(level_data_obj, ninja_pos)
    
    # Examine sub-cell graph features
    sub_graph = graph_data.sub_cell_graph
    print(f"\nSub-cell graph:")
    print(f"  Nodes: {np.sum(sub_graph.node_mask)}")
    print(f"  Node features shape: {sub_graph.node_features.shape}")
    print(f"  Feature dimension: {sub_graph.node_features.shape[1]}")
    
    # Check first few valid nodes
    valid_indices = np.where(sub_graph.node_mask)[0][:20]
    print(f"\nFirst 20 valid nodes:")
    for i, idx in enumerate(valid_indices):
        features = sub_graph.node_features[idx]
        pos_x, pos_y = features[0], features[1]
        print(f"  Node {idx}: pos=({pos_x:.1f}, {pos_y:.1f}), features[:5]={features[:5]}")
    
    # Check if positions are reasonable
    all_positions = sub_graph.node_features[valid_indices, :2]
    print(f"\nPosition statistics:")
    print(f"  X range: {np.min(all_positions[:, 0]):.1f} to {np.max(all_positions[:, 0]):.1f}")
    print(f"  Y range: {np.min(all_positions[:, 1]):.1f} to {np.max(all_positions[:, 1]):.1f}")
    print(f"  Unique positions: {len(np.unique(all_positions, axis=0))}")
    
    # Check level data dimensions
    print(f"\nLevel data:")
    print(f"  Tiles shape: {level_data_obj.tiles.shape}")
    print(f"  Tile pixel size: 24px")
    print(f"  Expected map size: {level_data_obj.tiles.shape[0] * 24}x{level_data_obj.tiles.shape[1] * 24} pixels")
    
    # Check some tile values
    print(f"  Sample tiles:")
    for i in range(min(5, level_data_obj.tiles.shape[0])):
        for j in range(min(5, level_data_obj.tiles.shape[1])):
            print(f"    Tile ({i},{j}): {level_data_obj.tiles[i, j]}")

if __name__ == "__main__":
    debug_node_features()