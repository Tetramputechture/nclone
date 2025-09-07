#!/usr/bin/env python3

"""
Check what maps are available in the environment and their entity types.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.environment import NCloneEnvironment
from nclone.constants.entity_types import EntityType

def check_maps():
    """Check available maps and their entity types."""
    
    # Try to find available maps
    env = NCloneEnvironment()
    
    # Check if there's a way to list available maps
    if hasattr(env, 'available_maps'):
        print("Available maps:", env.available_maps)
    elif hasattr(env, 'maps'):
        print("Available maps:", env.maps)
    else:
        print("No obvious map listing found")
    
    # Try some common map names that might have switches and doors
    test_maps = [
        'doortest',
        'switchtest', 
        'locked_door_test',
        'switch_door_test',
        'tutorial',
        'level1',
        'level_01',
        'test_switch',
        'test_door',
        'simple',
        'basic'
    ]
    
    print("\nTesting maps for switches and doors:")
    
    for map_name in test_maps:
        try:
            env = NCloneEnvironment(map_name=map_name)
            entities = env.entities
            
            # Check entity types
            entity_types = [entity.get('type') for entity in entities]
            unique_types = set(entity_types)
            
            # Check for switches and doors
            has_switches = any(t in [EntityType.EXIT_SWITCH] for t in unique_types)
            has_doors = any(t in [EntityType.LOCKED_DOOR, EntityType.REGULAR_DOOR, EntityType.EXIT_DOOR, EntityType.TRAP_DOOR] for t in unique_types)
            
            print(f"  {map_name}: {len(entities)} entities, types: {sorted(unique_types)}")
            if has_switches or has_doors:
                print(f"    ✅ Has switches: {has_switches}, doors: {has_doors}")
            else:
                print(f"    ❌ No switches or doors")
                
        except Exception as e:
            print(f"  {map_name}: ❌ Failed to load - {e}")
    
    # Also check what entity types are in the current doortest map
    print(f"\nEntity type reference:")
    for attr_name in dir(EntityType):
        if not attr_name.startswith('_'):
            value = getattr(EntityType, attr_name)
            print(f"  {attr_name}: {value}")

if __name__ == "__main__":
    check_maps()