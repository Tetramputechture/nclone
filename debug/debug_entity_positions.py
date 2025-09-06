#!/usr/bin/env python3
"""
Debug entity positions in level data vs graph features.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
import numpy as np


def debug_entity_positions():
    """Debug entity positions."""
    print("=" * 80)
    print("DEBUGGING ENTITY POSITIONS")
    print("=" * 80)
    
    # Create environment
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42
    )
    
    # Reset to load the map
    env.reset()
    
    # Get level data and ninja position
    level_data = env.level_data
    ninja_pos = env.nplay_headless.ninja_position()
    
    print(f"Ninja position: {ninja_pos}")
    print(f"Total entities in level_data: {len(level_data.entities)}")
    
    # Print all entities from level data
    print("\nEntities from level_data:")
    for i, entity in enumerate(level_data.entities):
        print(f"  Entity {i}: type={entity.get('type')}, x={entity.get('x')}, y={entity.get('y')}")
    
    # Build graph
    graph_builder = HierarchicalGraphBuilder()
    hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
    
    # Use the sub-cell graph
    graph_data = hierarchical_data.sub_cell_graph
    
    print(f"\nGraph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Find all entity nodes and check their features
    from nclone.graph.common import NodeType
    
    entity_node_count = 0
    for node_idx in range(graph_data.num_nodes):
        if graph_data.node_mask[node_idx] == 0:
            continue
            
        if hasattr(graph_data, 'node_types') and graph_data.node_types[node_idx] == NodeType.ENTITY:
            entity_node_count += 1
            node_features = graph_data.node_features[node_idx]
            
            print(f"\nEntity Node {entity_node_count}: Node {node_idx}")
            print(f"  Raw position from features[0:2]: ({node_features[0]:.1f}, {node_features[1]:.1f})")
            
            # Extract normalized position
            tile_type_dim = 38
            entity_type_dim = 30
            state_offset = 2 + tile_type_dim + 4 + entity_type_dim
            
            if len(node_features) > state_offset + 2:
                norm_x = node_features[state_offset + 1]
                norm_y = node_features[state_offset + 2]
                print(f"  Normalized position from features[{state_offset + 1}:{state_offset + 3}]: ({norm_x:.3f}, {norm_y:.3f})")
                
                # Convert back to pixel coordinates
                pixel_x = norm_x * 1056
                pixel_y = norm_y * 600
                print(f"  Converted pixel position: ({pixel_x:.1f}, {pixel_y:.1f})")
            else:
                print(f"  Features too short: {len(node_features)} <= {state_offset + 2}")
            
            # Check entity type
            entity_type_start = 2 + tile_type_dim + 4
            entity_type_end = entity_type_start + entity_type_dim
            entity_type_features = node_features[entity_type_start:entity_type_end]
            entity_type_idx = int(np.argmax(entity_type_features))
            
            entity_type_names = {
                0: "NINJA", 6: "LOCKED_DOOR", 8: "TRAP_DOOR", 11: "ONE_WAY",
                13: "LASER_DRONE", 2: "EXIT_DOOR", 10: "ONE_WAY_PLATFORM"
            }
            entity_type_name = entity_type_names.get(entity_type_idx, f"TYPE_{entity_type_idx}")
            print(f"  Entity type: {entity_type_name} (index {entity_type_idx})")


if __name__ == '__main__':
    debug_entity_positions()