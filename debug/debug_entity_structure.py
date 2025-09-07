#!/usr/bin/env python3
"""
Debug the entity structure in doortest map to understand functional edge issues.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.constants.entity_types import EntityType


def debug_entity_structure():
    """Debug the entity structure in doortest map."""
    print("=" * 80)
    print("DEBUGGING ENTITY STRUCTURE IN DOORTEST MAP")
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
    
    # Get level data
    level_data = env.level_data
    
    print(f"Total entities: {len(level_data.entities)}")
    print()
    
    # Group entities by type
    entities_by_type = {}
    for i, entity in enumerate(level_data.entities):
        entity_type = entity.get('type')
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
        entities_by_type[entity_type].append((i, entity))
    
    # Print entity type mapping
    print("Entity Type Mapping:")
    for attr_name in dir(EntityType):
        if not attr_name.startswith('_'):
            value = getattr(EntityType, attr_name)
            if isinstance(value, int):
                print(f"  {value}: {attr_name}")
    print()
    
    # Analyze each entity type
    for entity_type, entities in entities_by_type.items():
        type_name = "UNKNOWN"
        for attr_name in dir(EntityType):
            if not attr_name.startswith('_'):
                if getattr(EntityType, attr_name) == entity_type:
                    type_name = attr_name
                    break
        
        print(f"Entity Type {entity_type} ({type_name}): {len(entities)} entities")
        
        for i, (idx, entity) in enumerate(entities):
            print(f"  Entity {idx}: {entity}")
            
            # For LOCKED_DOOR entities, check if they have switch/door positions
            if entity_type == EntityType.LOCKED_DOOR:
                print(f"    Switch position: ({entity.get('x')}, {entity.get('y')})")
                print(f"    Door position: ({entity.get('door_x')}, {entity.get('door_y')})")
                print(f"    Entity ID: {entity.get('entity_id')}")
                print(f"    State: {entity.get('state')}")
                print(f"    Closed: {entity.get('closed')}")
                
                # Check if positions are in solid tiles
                switch_x, switch_y = entity.get('x'), entity.get('y')
                door_x, door_y = entity.get('door_x'), entity.get('door_y')
                
                # Convert to tile coordinates
                switch_tile_x = int(switch_x // 24)
                switch_tile_y = int(switch_y // 24)
                door_tile_x = int(door_x // 24)
                door_tile_y = int(door_y // 24)
                
                print(f"    Switch tile: ({switch_tile_x}, {switch_tile_y})")
                print(f"    Door tile: ({door_tile_x}, {door_tile_y})")
                
                if (0 <= switch_tile_y < level_data.height and 0 <= switch_tile_x < level_data.width):
                    switch_tile_value = level_data.get_tile(switch_tile_y, switch_tile_x)
                    print(f"    Switch tile value: {switch_tile_value}")
                
                if (0 <= door_tile_y < level_data.height and 0 <= door_tile_x < level_data.width):
                    door_tile_value = level_data.get_tile(door_tile_y, door_tile_x)
                    print(f"    Door tile value: {door_tile_value}")
        
        print()
    
    # Check for potential functional edge pairs
    print("=" * 60)
    print("FUNCTIONAL EDGE ANALYSIS")
    print("=" * 60)
    
    locked_doors = entities_by_type.get(EntityType.LOCKED_DOOR, [])
    trap_doors = entities_by_type.get(EntityType.TRAP_DOOR, [])
    exit_switches = entities_by_type.get(EntityType.EXIT_SWITCH, [])
    exit_doors = entities_by_type.get(EntityType.EXIT_DOOR, [])
    
    print(f"LOCKED_DOOR entities: {len(locked_doors)}")
    print(f"TRAP_DOOR entities: {len(trap_doors)}")
    print(f"EXIT_SWITCH entities: {len(exit_switches)}")
    print(f"EXIT_DOOR entities: {len(exit_doors)}")
    
    # For LOCKED_DOOR entities, they should create functional edges between switch and door positions
    if locked_doors:
        print("\nLOCKED_DOOR functional edges:")
        for idx, (_, entity) in enumerate(locked_doors):
            switch_pos = (entity.get('x'), entity.get('y'))
            door_pos = (entity.get('door_x'), entity.get('door_y'))
            entity_id = entity.get('entity_id')
            
            print(f"  LOCKED_DOOR {idx}: {entity_id}")
            print(f"    Should create functional edge: {switch_pos} -> {door_pos}")
            
            # Calculate distance
            distance = ((door_pos[0] - switch_pos[0])**2 + (door_pos[1] - switch_pos[1])**2)**0.5
            print(f"    Distance: {distance:.1f} pixels")
    
    # Check if there are any entity ID matches that should create functional edges
    print("\nEntity ID analysis:")
    entity_ids = {}
    for i, entity in enumerate(level_data.entities):
        entity_id = entity.get('entity_id')
        if entity_id:
            if entity_id not in entity_ids:
                entity_ids[entity_id] = []
            entity_ids[entity_id].append((i, entity))
    
    for entity_id, entities in entity_ids.items():
        if len(entities) > 1:
            print(f"  Entity ID '{entity_id}' appears {len(entities)} times:")
            for idx, entity in entities:
                print(f"    Entity {idx}: type={entity.get('type')}, pos=({entity.get('x')}, {entity.get('y')})")


if __name__ == '__main__':
    debug_entity_structure()