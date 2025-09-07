#!/usr/bin/env python3

"""
Debug script to trace ninja node creation in the graph
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

import numpy as np
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.constants.entity_types import EntityType

def debug_ninja_node():
    """Debug ninja node creation in the graph"""
    
    print("=== Debug Ninja Node Creation ===")
    
    # Initialize environment
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=True,
        eval_mode=False,
        seed=42,
        custom_map_path="nclone/test_maps/doortest",
    )
    
    # Reset environment
    observation, info = env.reset()
    print(f"Environment reset complete")
    
    # Get ninja position
    ninja_pos = env.nplay_headless.ninja_position()
    print(f"Ninja position from simulator: {ninja_pos}")
    
    # Extract entities using the same method as graph construction
    entities = env._extract_graph_entities()
    print(f"\nTotal entities extracted: {len(entities)}")
    
    # Find ninja entity
    ninja_entities = [e for e in entities if e['type'] == EntityType.NINJA]
    print(f"Ninja entities found: {len(ninja_entities)}")
    
    if ninja_entities:
        ninja_entity = ninja_entities[0]
        print(f"Ninja entity: {ninja_entity}")
        
        # Convert to grid coordinates
        from nclone.constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
        ninja_grid_x = ninja_entity['x'] // MAP_TILE_WIDTH
        ninja_grid_y = ninja_entity['y'] // MAP_TILE_HEIGHT
        print(f"Ninja grid position: ({ninja_grid_x}, {ninja_grid_y})")
    
    # Build graph and check if ninja node is included
    print(f"\nBuilding graph...")
    env.set_graph_debug_enabled(True)
    debug_info = env._debug_info()
    
    if debug_info and 'graph' in debug_info:
        graph_data = debug_info['graph']['data']
        print(f"Graph built: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
        
        # Check node features for ninja
        node_positions = graph_data.node_features[:, :2]  # x, y coordinates
        node_types = graph_data.node_features[:, 2:]  # entity type features
        
        print(f"Node position range: X=[{node_positions[:, 0].min():.1f}, {node_positions[:, 0].max():.1f}], Y=[{node_positions[:, 1].min():.1f}, {node_positions[:, 1].max():.1f}]")
        
        # Look for ninja nodes (EntityType.NINJA should be encoded in node features)
        ninja_type_idx = EntityType.NINJA  # This is already an int (0)
        print(f"Looking for ninja nodes (type index {ninja_type_idx})...")
        
        # Check if any nodes have ninja type
        ninja_nodes = []
        for i in range(int(graph_data.num_nodes)):
            # Node type is encoded as one-hot in features starting from index 2
            type_features = node_types[i]
            if len(type_features) > ninja_type_idx and type_features[ninja_type_idx] > 0.5:
                ninja_nodes.append(i)
                pos = node_positions[i]
                print(f"  Found ninja node {i} at position ({pos[0]:.1f}, {pos[1]:.1f})")
        
        print(f"Total ninja nodes found in graph: {len(ninja_nodes)}")
        
        if len(ninja_nodes) == 0:
            print("\n❌ PROBLEM: No ninja nodes found in graph!")
            print("This explains why there are no edges connecting to the ninja position.")
            
            # Check if ninja entity was processed
            if ninja_entities:
                print(f"Ninja entity was extracted but not added to graph.")
                print(f"Expected ninja at grid position: ({ninja_grid_x}, {ninja_grid_y})")
                
                # Look for nodes near expected ninja position
                expected_x, expected_y = ninja_grid_x, ninja_grid_y
                distances = np.sqrt((node_positions[:, 0] - expected_x)**2 + (node_positions[:, 1] - expected_y)**2)
                close_nodes = np.where(distances < 1.0)[0]
                
                print(f"Nodes within 1 grid unit of expected ninja position:")
                for node_idx in close_nodes[:5]:
                    pos = node_positions[node_idx]
                    dist = distances[node_idx]
                    # Check what type this node is
                    type_features = node_types[node_idx]
                    max_type_idx = np.argmax(type_features)
                    print(f"  Node {node_idx}: ({pos[0]:.1f}, {pos[1]:.1f}), distance: {dist:.2f}, type_idx: {max_type_idx}")
        
        # Also check for any nodes at position (0,0) or other suspicious positions
        zero_nodes = np.where((node_positions[:, 0] == 0) & (node_positions[:, 1] == 0))[0]
        if len(zero_nodes) > 0:
            print(f"\n⚠️  Found {len(zero_nodes)} nodes at position (0,0) - these might be misplaced entities:")
            for node_idx in zero_nodes[:5]:
                type_features = node_types[node_idx]
                max_type_idx = np.argmax(type_features)
                print(f"  Node {node_idx}: position (0,0), type_idx: {max_type_idx}")
    
    print("\n=== Debug Ninja Node Complete ===")

if __name__ == "__main__":
    debug_ninja_node()