#!/usr/bin/env python3
"""
Simple analysis of the current pathfinding result.
"""

import sys
import os

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold

def simple_path_analysis():
    """Simple analysis of current pathfinding."""
    print("=" * 60)
    print("🔍 SIMPLE PATH ANALYSIS")
    print("=" * 60)
    
    # Load environment
    env = BasicLevelNoGold(render_mode="rgb_array")
    env.reset()
    
    ninja_pos = env.nplay_headless.ninja_position()
    print(f"✅ Ninja position: {ninja_pos}")
    
    # Find entities
    print(f"\n📍 ENTITIES:")
    switches = []
    doors = []
    
    for entity in env.entities:
        if isinstance(entity, dict):
            entity_type = entity.get('entity_type', entity.get('type', 'unknown'))
            x = entity.get('x', 0)
            y = entity.get('y', 0)
        else:
            entity_type = getattr(entity, 'entity_type', getattr(entity, 'type', 'unknown'))
            x = getattr(entity, 'x', 0)
            y = getattr(entity, 'y', 0)
            
        print(f"   Entity type {entity_type} at ({x:.1f}, {y:.1f})")
        if entity_type == 4:  # Switch
            switches.append(entity)
        elif entity_type == 5:  # Door
            doors.append(entity)
    
    if switches:
        def get_x(e):
            return e.get('x', 0) if isinstance(e, dict) else getattr(e, 'x', 0)
        
        leftmost_switch = min(switches, key=get_x)
        switch_x = get_x(leftmost_switch)
        switch_y = leftmost_switch.get('y', 0) if isinstance(leftmost_switch, dict) else getattr(leftmost_switch, 'y', 0)
        print(f"\n🎯 Leftmost switch: ({switch_x:.1f}, {switch_y:.1f})")
        
        # Calculate direct distance
        import math
        direct_distance = math.sqrt(
            (switch_x - ninja_pos[0])**2 + (switch_y - ninja_pos[1])**2
        )
        print(f"📏 Direct distance: {direct_distance:.1f}px")
        
        # Analyze the level structure
        print(f"\n🗺️  LEVEL ANALYSIS:")
        print(f"   Level size: {env.level_data.width}x{env.level_data.height} tiles")
        print(f"   Pixel size: {env.level_data.width*24}x{env.level_data.height*24}px")
        
        # Check if there's a direct line of sight
        print(f"\n👁️  LINE OF SIGHT ANALYSIS:")
        dx = switch_x - ninja_pos[0]
        dy = switch_y - ninja_pos[1]
        print(f"   Horizontal distance: {dx:.1f}px")
        print(f"   Vertical distance: {dy:.1f}px")
        
        if abs(dx) > 300:
            print(f"   ⚠️  Large horizontal gap - should require multiple movements")
        if abs(dy) > 200:
            print(f"   ⚠️  Large vertical gap - should require jumps/falls")
        
        # Check tile types along the path
        print(f"\n🧱 TILE ANALYSIS:")
        ninja_tile_x = int(ninja_pos[0] // 24)
        ninja_tile_y = int(ninja_pos[1] // 24)
        switch_tile_x = int(switch_x // 24)
        switch_tile_y = int(switch_y // 24)
        
        print(f"   Ninja tile: ({ninja_tile_x}, {ninja_tile_y}) = {env.level_data.get_tile(ninja_tile_y, ninja_tile_x)}")
        print(f"   Switch tile: ({switch_tile_x}, {switch_tile_y}) = {env.level_data.get_tile(switch_tile_y, switch_tile_x)}")
        
        # Sample some tiles along the path
        steps = 5
        for i in range(steps + 1):
            t = i / steps
            sample_x = int((ninja_pos[0] + t * dx) // 24)
            sample_y = int((ninja_pos[1] + t * dy) // 24)
            if 0 <= sample_x < env.level_data.width and 0 <= sample_y < env.level_data.height:
                tile_value = env.level_data.get_tile(sample_y, sample_x)
                print(f"   Step {i}: tile ({sample_x}, {sample_y}) = {tile_value}")
                if tile_value != 0:
                    print(f"      ⚠️  Solid tile blocking direct path!")

if __name__ == "__main__":
    simple_path_analysis()