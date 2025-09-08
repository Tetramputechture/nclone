#!/usr/bin/env python3
"""
Debug script to see what entities are in the doortest map.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.constants.entity_types import EntityType

def debug_entities():
    """Debug what entities are in the doortest map."""
    print("=" * 80)
    print("DEBUGGING ENTITIES IN DOORTEST MAP")
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
    
    # Print all entities
    print("\nAll entities in level_data:")
    for i, entity in enumerate(level_data.entities):
        print(f"  Entity {i}: {entity}")
    
    # Build graph
    graph_builder = HierarchicalGraphBuilder()
    hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
    
    # Use the sub-cell graph for pathfinding
    graph_data = hierarchical_data.sub_cell_graph
    
    print(f"\nGraph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Check what entity types are in the graph nodes
    entity_types_found = set()
    print(f"\nAnalyzing {graph_data.num_nodes} graph nodes:")
    print(f"Node feature dimension: {len(graph_data.node_features[0]) if graph_data.num_nodes > 0 else 'N/A'}")
    
    # Calculate the correct offset for entity type
    tile_type_dim = 38
    entity_type_offset = 2 + tile_type_dim + 4  # position(2) + tile_type(38) + solidity(4)
    
    for node_idx in range(graph_data.num_nodes):
        node_features = graph_data.node_features[node_idx]
        position = (node_features[0], node_features[1])
        
        # Check if this node has entity features
        has_entity = False
        entity_type_found = None
        
        # Check entity type one-hot encoding
        for entity_type in range(30):  # entity_type_dim = 30
            if entity_type_offset + entity_type < len(node_features):
                if node_features[entity_type_offset + entity_type] > 0.5:  # One-hot encoded
                    has_entity = True
                    entity_type_found = entity_type
                    entity_types_found.add(entity_type)
                    break
        
        if has_entity:
            print(f"  Node {node_idx}: entity_type={entity_type_found}, pos={position}")
    
    print(f"\nUnique entity types found in graph nodes: {sorted(entity_types_found)}")
    
    # Map entity type numbers to names
    entity_type_names = {
        EntityType.NINJA: "NINJA",
        EntityType.TOGGLE_MINE: "TOGGLE_MINE",
        EntityType.GOLD: "GOLD",
        EntityType.EXIT_DOOR: "EXIT_DOOR",
        EntityType.EXIT_SWITCH: "EXIT_SWITCH",
        EntityType.REGULAR_DOOR: "REGULAR_DOOR",
        EntityType.LOCKED_DOOR: "LOCKED_DOOR",
        EntityType.TRAP_DOOR: "TRAP_DOOR",
        EntityType.LAUNCH_PAD: "LAUNCH_PAD",
        EntityType.ONE_WAY: "ONE_WAY",
        EntityType.DRONE_ZAP: "DRONE_ZAP",
        EntityType.BOUNCE_BLOCK: "BOUNCE_BLOCK",
        EntityType.THWUMP: "THWUMP",
        EntityType.TOGGLE_MINE_TOGGLED: "TOGGLE_MINE_TOGGLED",
        EntityType.BOOST_PAD: "BOOST_PAD",
        EntityType.DEATH_BALL: "DEATH_BALL",
        EntityType.MINI_DRONE: "MINI_DRONE",
        EntityType.SHWUMP: "SHWUMP",
    }
    
    print("\nEntity type mapping:")
    for entity_type in sorted(entity_types_found):
        name = entity_type_names.get(entity_type, f"UNKNOWN_{entity_type}")
        print(f"  {entity_type}: {name}")

if __name__ == "__main__":
    debug_entities()