#!/usr/bin/env python3
"""
Debug script to analyze functional edge creation issues.
"""

import numpy as np
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.common import NodeType, EdgeType
from nclone.constants.entity_types import EntityType
import pygame

def debug_functional_edge_creation():
    """Debug functional edge creation logic."""
    
    print("=== FUNCTIONAL EDGE CREATION DEBUG ===")
    
    # Initialize environment
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_debug_overlay=False,
        custom_map_path="nclone/test_maps/doortest"
    )
    env.reset()
    
    # Get ninja position
    ninja_pos = env.nplay_headless.ninja_position()
    ninja_vel = env.nplay_headless.ninja_velocity()
    
    # Get level data
    level_data = env.level_data
    
    print(f"=== ENTITY ANALYSIS FOR FUNCTIONAL EDGES ===")
    print(f"Total entities: {len(level_data.entities)}")
    
    # Analyze entities for functional edge creation
    switches = []
    doors = []
    
    for i, entity in enumerate(level_data.entities):
        entity_type = entity.get('type')
        entity_id = entity.get('entity_id')
        x, y = entity.get('x'), entity.get('y')
        
        print(f"Entity {i}: type={entity_type}, id='{entity_id}', pos=({x}, {y})")
        
        if entity_type == EntityType.EXIT_SWITCH:
            switches.append(('EXIT_SWITCH', entity))
            print(f"  -> EXIT SWITCH: {entity}")
        elif entity_type == EntityType.EXIT_DOOR:
            doors.append(('EXIT_DOOR', entity))
            print(f"  -> EXIT DOOR: {entity}")
        elif entity_type == EntityType.LOCKED_DOOR:
            # Check if it's a switch or door part
            is_door_part = entity.get('is_door_part', False)
            if is_door_part:
                doors.append(('LOCKED_DOOR', entity))
                print(f"  -> LOCKED DOOR (door part): {entity}")
            else:
                switches.append(('LOCKED_DOOR', entity))
                print(f"  -> LOCKED DOOR (switch part): {entity}")
        elif entity_type == EntityType.TRAP_DOOR:
            # Check if it's a switch or door part
            is_door_part = entity.get('is_door_part', False)
            if is_door_part:
                doors.append(('TRAP_DOOR', entity))
                print(f"  -> TRAP DOOR (door part): {entity}")
            else:
                switches.append(('TRAP_DOOR', entity))
                print(f"  -> TRAP DOOR (switch part): {entity}")
    
    print(f"\n=== SWITCH-DOOR MATCHING ANALYSIS ===")
    print(f"Found {len(switches)} switches and {len(doors)} doors")
    
    # Analyze potential matches
    for switch_type, switch_entity in switches:
        print(f"\nSwitch: {switch_type} at ({switch_entity.get('x')}, {switch_entity.get('y')})")
        print(f"  entity_id: '{switch_entity.get('entity_id')}'")
        
        for door_type, door_entity in doors:
            print(f"  Checking door: {door_type} at ({door_entity.get('x')}, {door_entity.get('y')})")
            print(f"    door entity_id: '{door_entity.get('entity_id')}'")
            print(f"    door switch_entity_id: '{door_entity.get('switch_entity_id')}'")
            
            # Check matching logic from edge_building.py
            is_matching = False
            if (switch_entity.get("type") == EntityType.EXIT_SWITCH and 
                door_entity.get("type") == EntityType.EXIT_DOOR):
                # Match exit switches to exit doors by entity_id
                if switch_entity.get("entity_id") == door_entity.get("switch_entity_id"):
                    is_matching = True
                    print(f"    -> EXIT MATCH: switch_id='{switch_entity.get('entity_id')}' == door_switch_id='{door_entity.get('switch_entity_id')}'")
            elif switch_entity.get("type") == EntityType.LOCKED_DOOR:
                # For LOCKED_DOOR, match switch node to door node of same entity
                if (switch_entity.get("entity_id") == door_entity.get("entity_id") and
                    not switch_entity.get("is_door_part", False) and
                    door_entity.get("is_door_part", False)):
                    is_matching = True
                    print(f"    -> LOCKED DOOR MATCH: same entity_id='{switch_entity.get('entity_id')}'")
            elif switch_entity.get("type") == EntityType.TRAP_DOOR:
                # For TRAP_DOOR, match switch node to door node of same entity
                if (switch_entity.get("entity_id") == door_entity.get("entity_id") and
                    not switch_entity.get("is_door_part", False) and
                    door_entity.get("is_door_part", False)):
                    is_matching = True
                    print(f"    -> TRAP DOOR MATCH: same entity_id='{switch_entity.get('entity_id')}'")
            
            if is_matching:
                print(f"    ✅ FUNCTIONAL EDGE SHOULD BE CREATED!")
            else:
                print(f"    ❌ No match")
    
    # Clean up
    env.close()
    pygame.quit()

if __name__ == "__main__":
    debug_functional_edge_creation()