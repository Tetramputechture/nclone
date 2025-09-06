#!/usr/bin/env python3
"""
Debug all entity nodes to see where they are and what types they are.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine
from nclone.constants.entity_types import EntityType


def debug_all_entity_nodes():
    """Debug all entity nodes."""
    print("=" * 80)
    print("DEBUGGING ALL ENTITY NODES")
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
    
    # Build graph
    graph_builder = HierarchicalGraphBuilder()
    hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
    
    # Use the sub-cell graph for pathfinding
    graph_data = hierarchical_data.sub_cell_graph
    
    print(f"Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Find all entity nodes
    entity_nodes = []
    pathfinding_engine = PathfindingEngine()
    
    for node_idx in range(graph_data.num_nodes):
        if graph_data.node_mask[node_idx] == 0:
            continue
            
        # Check if this is an entity node
        from nclone.graph.common import NodeType
        if hasattr(graph_data, 'node_types') and graph_data.node_types[node_idx] == NodeType.ENTITY:
            # Get node position
            node_pos = pathfinding_engine._get_node_position(graph_data, node_idx)
            
            # Try to determine entity type from node features
            node_features = graph_data.node_features[node_idx]
            
            # The entity type should be encoded in the features
            # Let's look at the feature structure to find the entity type
            entity_nodes.append((node_idx, node_pos, node_features))
    
    print(f"\nFound {len(entity_nodes)} entity nodes:")
    
    for i, (node_idx, node_pos, features) in enumerate(entity_nodes):
        print(f"\n  Entity Node {i+1}: Node {node_idx} at {node_pos}")
        
        # Try to decode entity type from features
        # Based on hierarchical_builder.py, the feature layout is:
        # tile_type (38) + 4 + entity_type (30) + state_features
        tile_type_dim = 38
        entity_type_dim = 30
        entity_type_start = tile_type_dim + 4
        entity_type_end = entity_type_start + entity_type_dim
        
        if len(features) > entity_type_end:
            entity_type_features = features[entity_type_start:entity_type_end]
            # Find the index of the maximum value (one-hot encoding)
            entity_type_idx = int(np.argmax(entity_type_features))
            print(f"    Entity type index: {entity_type_idx}")
            
            # Map to entity type name
            entity_type_names = {
                0: "NINJA",
                1: "GOLD",
                2: "EXIT_DOOR",
                3: "EXIT_SWITCH",
                4: "REGULAR_DOOR",
                5: "REGULAR_SWITCH",
                6: "LOCKED_DOOR",
                7: "LOCKED_SWITCH",
                8: "TRAP_DOOR",
                9: "TRAP_SWITCH",
                10: "ONE_WAY_PLATFORM",
                11: "ONE_WAY",
                12: "CHAINGUN_DRONE",
                13: "LASER_DRONE",
                14: "ZAP_DRONE",
                15: "CHASER_DRONE",
                16: "FLOOR_GUARD",
                17: "BOUNCE_BLOCK",
                18: "PLAYER_DRONE",
                19: "MINE",
                20: "THWUMP",
                21: "LOCKED_DOOR_SWITCH",
                22: "LOCKED_DOOR_DOOR",
                23: "ROCKET",
                24: "GAUSS_TURRET",
                25: "DUAL_LASER",
                26: "SEEKER_DRONE",
                27: "MICRO_DRONE",
                28: "ALT_LASER",
                29: "ALT_CHAINGUN"
            }
            
            entity_type_name = entity_type_names.get(entity_type_idx, f"UNKNOWN_{entity_type_idx}")
            print(f"    Entity type: {entity_type_name}")
        
        # Check distance to ninja
        distance = ((node_pos[0] - ninja_pos[0])**2 + (node_pos[1] - ninja_pos[1])**2)**0.5
        print(f"    Distance to ninja: {distance:.1f}")
        
        # Check if this node has edges
        has_outgoing = False
        has_incoming = False
        outgoing_count = 0
        incoming_count = 0
        
        for edge_idx in range(graph_data.num_edges):
            if graph_data.edge_mask[edge_idx] == 0:
                continue
                
            src = int(graph_data.edge_index[0, edge_idx])
            dst = int(graph_data.edge_index[1, edge_idx])
            
            if src == node_idx:
                has_outgoing = True
                outgoing_count += 1
            if dst == node_idx:
                has_incoming = True
                incoming_count += 1
        
        print(f"    Outgoing edges: {outgoing_count}")
        print(f"    Incoming edges: {incoming_count}")


if __name__ == '__main__':
    import numpy as np
    debug_all_entity_nodes()