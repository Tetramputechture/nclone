#!/usr/bin/env python3
"""
Analyze coordinate systems between official maps and attract files.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def analyze_coordinate_conversion():
    """Analyze how coordinates convert between systems."""
    
    print("=== COORDINATE ANALYSIS ===")
    
    # Official coordinates (from previous analysis)
    official_entities = [
        {'type': 1, 'x': 4624, 'y': 0},  # Gold
        {'type': 1, 'x': 4176, 'y': 0},  # Gold  
        {'type': 1, 'x': 3756, 'y': 0},  # Gold
        {'type': 3, 'x': 3734, 'y': 0},  # Exit
        {'type': 4, 'x': 3746, 'y': 0},  # Enemy
    ]
    
    # Attract coordinates (from previous analysis)
    attract_entities = [
        {'type': 1, 'x': 16, 'y': 18},   # Gold
        {'type': 1, 'x': 80, 'y': 16},   # Gold
        {'type': 1, 'x': 172, 'y': 14},  # Gold
        {'type': 3, 'x': 150, 'y': 14},  # Exit
        {'type': 4, 'x': 162, 'y': 14},  # Enemy
    ]
    
    official_spawn = (6722, 0)
    attract_spawn = (66, 26)
    
    print("Coordinate conversion analysis:")
    print("Official -> Attract (pixel to tile conversion)")
    
    # Try different conversion factors
    for i in range(min(len(official_entities), len(attract_entities))):
        off_ent = official_entities[i]
        att_ent = attract_entities[i]
        
        if off_ent['type'] == att_ent['type']:
            # Calculate conversion factors
            if att_ent['x'] != 0:
                x_factor = off_ent['x'] / att_ent['x']
            else:
                x_factor = 0
                
            if att_ent['y'] != 0:
                y_factor = off_ent['y'] / att_ent['y']
            else:
                y_factor = 0
                
            print(f"  Entity {i} (type {off_ent['type']}): ({off_ent['x']}, {off_ent['y']}) -> ({att_ent['x']}, {att_ent['y']})")
            print(f"    X factor: {x_factor:.2f}, Y factor: {y_factor:.2f}")
    
    print(f"\nSpawn conversion: ({official_spawn[0]}, {official_spawn[1]}) -> ({attract_spawn[0]}, {attract_spawn[1]})")
    if attract_spawn[0] != 0:
        spawn_x_factor = official_spawn[0] / attract_spawn[0]
        print(f"  Spawn X factor: {spawn_x_factor:.2f}")
    
    # Test if it's a simple pixel-to-tile conversion (24 pixels per tile is common in N++)
    print(f"\nTesting 24-pixel tile size:")
    for i in range(min(len(official_entities), len(attract_entities))):
        off_ent = official_entities[i]
        att_ent = attract_entities[i]
        
        if off_ent['type'] == att_ent['type']:
            tile_x = off_ent['x'] // 24
            tile_y = off_ent['y'] // 24
            print(f"  Entity {i}: official pixels ({off_ent['x']}, {off_ent['y']}) / 24 = tiles ({tile_x}, {tile_y}), attract ({att_ent['x']}, {att_ent['y']})")
    
    spawn_tile_x = official_spawn[0] // 24
    spawn_tile_y = official_spawn[1] // 24
    print(f"  Spawn: official pixels ({official_spawn[0]}, {official_spawn[1]}) / 24 = tiles ({spawn_tile_x}, {spawn_tile_y}), attract ({attract_spawn[0]}, {attract_spawn[1]})")

if __name__ == "__main__":
    analyze_coordinate_conversion()