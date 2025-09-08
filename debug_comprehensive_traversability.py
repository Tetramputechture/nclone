#!/usr/bin/env python3
"""
Comprehensive traversability analysis with proper coordinate system and entity collision
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.constants.entity_types import EntityType
from nclone.tile_definitions import TILE_SEGMENT_DIAG_MAP, TILE_SEGMENT_CIRCULAR_MAP

def analyze_tile_traversability():
    """Analyze which tile types should be traversable"""
    print("=" * 60)
    print("COMPREHENSIVE TRAVERSABILITY ANALYSIS")
    print("=" * 60)
    
    # Create environment
    env = BasicLevelNoGold(render_mode="rgb_array", enable_frame_stack=False, enable_debug_overlay=False, eval_mode=False, seed=42)
    env.reset()
    ninja_position = env.nplay_headless.ninja_position()
    level_data = env.level_data
    
    print(f"Ninja position: {ninja_position}")
    print(f"Level size: {level_data.width}x{level_data.height} tiles")
    
    # Analyze tile type distribution
    tile_counts = defaultdict(int)
    for row in level_data.tiles:
        for tile_value in row:
            tile_counts[tile_value] += 1
    
    print(f"\nTile type distribution:")
    total_tiles = sum(tile_counts.values())
    for tile_type in sorted(tile_counts.keys()):
        count = tile_counts[tile_type]
        percentage = (count / total_tiles) * 100
        traversable = should_tile_be_traversable(tile_type)
        status = "TRAVERSABLE" if traversable else "SOLID"
        print(f"  Type {tile_type:2d}: {count:4d} tiles ({percentage:5.1f}%) - {status}")
    
    # Analyze entities with physical collision
    print(f"\nEntities with physical collision:")
    collision_entities = []
    
    for entity in level_data.entities:
        entity_type = entity.get("type", -1)
        entity_x = entity.get("x", 0)
        entity_y = entity.get("y", 0)
        
        has_collision = entity_has_physical_collision(entity_type, entity)
        if has_collision:
            collision_entities.append((entity_type, entity_x, entity_y, entity))
            print(f"  Type {entity_type:2d} at ({entity_x:3.0f}, {entity_y:3.0f}): {get_entity_name(entity_type)}")
    
    print(f"Total collision entities: {len(collision_entities)}")
    
    # Test traversability with proper coordinate system
    print(f"\nTesting traversability with corrected coordinate system...")
    
    # Build graph to get the traversability checker
    builder = HierarchicalGraphBuilder()
    hierarchical_graph = builder.build_graph(level_data, ninja_position)
    graph_constructor = builder.graph_constructor
    
    # Test a grid of positions
    test_positions = []
    traversable_count = 0
    non_traversable_count = 0
    
    # Test every 6 pixels (quarter tile resolution)
    for x in range(0, level_data.width * 24, 6):
        for y in range(0, level_data.height * 24, 6):
            is_traversable = graph_constructor._is_position_traversable(x, y, level_data.tiles, level_data.entities)
            test_positions.append((x, y, is_traversable))
            
            if is_traversable:
                traversable_count += 1
            else:
                non_traversable_count += 1
    
    total_positions = len(test_positions)
    traversable_percentage = (traversable_count / total_positions) * 100
    
    print(f"Tested {total_positions} positions:")
    print(f"  Traversable: {traversable_count} ({traversable_percentage:.1f}%)")
    print(f"  Non-traversable: {non_traversable_count} ({100-traversable_percentage:.1f}%)")
    
    # Create corrected visualization
    create_corrected_visualization(level_data, ninja_position, test_positions, collision_entities)
    
    # Analyze specific problem areas
    print(f"\nAnalyzing specific problem areas...")
    
    # Check ninja area
    ninja_tile_x = int(ninja_position[0] // 24)
    ninja_tile_y = int(ninja_position[1] // 24)
    ninja_tile_value = level_data.tiles[ninja_tile_y][ninja_tile_x]
    
    print(f"Ninja tile ({ninja_tile_x}, {ninja_tile_y}): type {ninja_tile_value}")
    print(f"Should be traversable: {should_tile_be_traversable(ninja_tile_value)}")
    
    # Test positions around ninja
    print(f"Positions around ninja:")
    for dx in [-12, 0, 12]:
        for dy in [-12, 0, 12]:
            test_x = ninja_position[0] + dx
            test_y = ninja_position[1] + dy
            is_trav = graph_constructor._is_position_traversable(test_x, test_y, level_data.tiles, level_data.entities)
            print(f"  ({test_x:3.0f}, {test_y:3.0f}): {'✓' if is_trav else '✗'}")
    
    # Check switch area
    leftmost_switch = None
    leftmost_x = float('inf')
    
    for entity in level_data.entities:
        if entity.get("type") == EntityType.LOCKED_DOOR:
            switch_x = entity.get("x", 0)
            if switch_x < leftmost_x:
                leftmost_x = switch_x
                leftmost_switch = entity
    
    if leftmost_switch:
        switch_x = leftmost_switch.get("x", 0)
        switch_y = leftmost_switch.get("y", 0)
        switch_tile_x = int(switch_x // 24)
        switch_tile_y = int(switch_y // 24)
        switch_tile_value = level_data.tiles[switch_tile_y][switch_tile_x]
        
        print(f"\nSwitch tile ({switch_tile_x}, {switch_tile_y}): type {switch_tile_value}")
        print(f"Should be traversable: {should_tile_be_traversable(switch_tile_value)}")
        
        # Test positions around switch
        print(f"Positions around switch:")
        for dx in [-12, 0, 12]:
            for dy in [-12, 0, 12]:
                test_x = switch_x + dx
                test_y = switch_y + dy
                is_trav = graph_constructor._is_position_traversable(test_x, test_y, level_data.tiles, level_data.entities)
                print(f"  ({test_x:3.0f}, {test_y:3.0f}): {'✓' if is_trav else '✗'}")

def should_tile_be_traversable(tile_type: int) -> bool:
    """Determine if a tile type should allow traversability"""
    if tile_type == 0:  # Empty
        return True
    elif tile_type == 1:  # Full solid
        return False
    elif 2 <= tile_type <= 33:  # Mixed geometry tiles
        return True  # Should have some traversable areas
    else:  # Unknown or glitched
        return False

def entity_has_physical_collision(entity_type: int, entity: dict) -> bool:
    """Determine if an entity has physical collision that affects traversability"""
    collision_types = {
        EntityType.LOCKED_DOOR: True,      # Solid until switch activated
        EntityType.TRAP_DOOR: True,        # Solid after switch activated  
        EntityType.REGULAR_DOOR: True,     # Always solid
        EntityType.ONE_WAY: True,          # Conditional collision
        EntityType.BOUNCE_BLOCK: True,     # Always solid
        EntityType.THWUMP: True,           # Moving solid obstacle
        EntityType.SHWUMP: True,           # Moving solid obstacle
        EntityType.LAUNCH_PAD: False,      # No collision, just effect
        EntityType.BOOST_PAD: False,       # No collision, just effect
        EntityType.GOLD: False,            # No collision
        EntityType.EXIT_DOOR: False,       # No collision (goal)
        EntityType.EXIT_SWITCH: False,     # No collision
        EntityType.TOGGLE_MINE: False,     # No collision (trigger)
        EntityType.DRONE_ZAP: False,       # No collision (projectile)
        EntityType.DEATH_BALL: False,      # No collision (moving hazard)
        EntityType.MINI_DRONE: False,      # No collision (moving hazard)
    }
    
    return collision_types.get(entity_type, False)

def get_entity_name(entity_type: int) -> str:
    """Get human-readable name for entity type"""
    names = {
        EntityType.NINJA: "Ninja",
        EntityType.LOCKED_DOOR: "Locked Door",
        EntityType.TRAP_DOOR: "Trap Door",
        EntityType.REGULAR_DOOR: "Regular Door",
        EntityType.ONE_WAY: "One Way Platform",
        EntityType.BOUNCE_BLOCK: "Bounce Block",
        EntityType.THWUMP: "Thwump",
        EntityType.SHWUMP: "Shwump",
        EntityType.LAUNCH_PAD: "Launch Pad",
        EntityType.BOOST_PAD: "Boost Pad",
        EntityType.GOLD: "Gold",
        EntityType.EXIT_DOOR: "Exit Door",
        EntityType.EXIT_SWITCH: "Exit Switch",
        EntityType.TOGGLE_MINE: "Toggle Mine",
        EntityType.DRONE_ZAP: "Drone Zap",
        EntityType.DEATH_BALL: "Death Ball",
        EntityType.MINI_DRONE: "Mini Drone",
    }
    
    return names.get(entity_type, f"Unknown_{entity_type}")

def create_corrected_visualization(level_data, ninja_position, test_positions, collision_entities):
    """Create visualization with corrected coordinate system"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left plot: Level rendering with traversability
    # Render level tiles as background
    level_image = np.zeros((level_data.height * 24, level_data.width * 24, 3))
    
    # Color tiles based on type
    for tile_y in range(level_data.height):
        for tile_x in range(level_data.width):
            tile_value = level_data.tiles[tile_y][tile_x]
            
            # Determine tile color
            if tile_value == 0:  # Empty
                color = [0.9, 0.9, 0.9]  # Light gray
            elif tile_value == 1:  # Solid
                color = [0.3, 0.3, 0.3]  # Dark gray
            elif should_tile_be_traversable(tile_value):  # Mixed geometry
                color = [0.7, 0.5, 0.3]  # Brown
            else:  # Unknown
                color = [0.5, 0.0, 0.5]  # Purple
            
            # Fill tile area
            y_start = tile_y * 24
            y_end = (tile_y + 1) * 24
            x_start = tile_x * 24
            x_end = (tile_x + 1) * 24
            
            level_image[y_start:y_end, x_start:x_end] = color
    
    ax1.imshow(level_image, origin='upper', extent=[0, level_data.width * 24, level_data.height * 24, 0])
    
    # Overlay traversability
    traversable_positions = [(x, y) for x, y, is_trav in test_positions if is_trav]
    non_traversable_positions = [(x, y) for x, y, is_trav in test_positions if not is_trav]
    
    if traversable_positions:
        trav_x = [pos[0] for pos in traversable_positions]
        trav_y = [pos[1] for pos in traversable_positions]
        ax1.scatter(trav_x, trav_y, c='lime', alpha=0.6, s=1, label=f'Traversable ({len(traversable_positions)})')
    
    # Sample non-traversable for visibility
    if non_traversable_positions:
        sample_size = min(len(non_traversable_positions), 2000)
        sampled_non_trav = non_traversable_positions[::len(non_traversable_positions)//sample_size]
        
        non_trav_x = [pos[0] for pos in sampled_non_trav]
        non_trav_y = [pos[1] for pos in sampled_non_trav]
        ax1.scatter(non_trav_x, non_trav_y, c='red', alpha=0.3, s=0.5, label=f'Non-traversable (sampled)')
    
    # Mark ninja
    ax1.scatter([ninja_position[0]], [ninja_position[1]], c='blue', s=100, marker='*', 
               label='Ninja', edgecolors='white', linewidth=2, zorder=10)
    
    # Mark collision entities
    if collision_entities:
        entity_x = [e[1] for e in collision_entities]
        entity_y = [e[2] for e in collision_entities]
        ax1.scatter(entity_x, entity_y, c='orange', s=50, marker='D', 
                   label='Collision Entities', edgecolors='black', linewidth=1, zorder=9)
    
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.set_title('Corrected Level Rendering + Traversability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Tile map with proper coordinate system
    # Flip tiles vertically to match pixel coordinate system
    tiles_display = np.flipud(level_data.tiles)
    
    ax2.imshow(tiles_display, cmap='tab20', interpolation='nearest', aspect='auto', alpha=0.8)
    
    # Mark ninja position (convert to tile coordinates with corrected Y)
    ninja_tile_x = int(ninja_position[0] // 24)
    ninja_tile_y = level_data.height - 1 - int(ninja_position[1] // 24)  # Flip Y
    ax2.scatter([ninja_tile_x], [ninja_tile_y], c='blue', s=100, marker='*', 
               label='Ninja', edgecolors='white', linewidth=2)
    
    # Mark collision entities
    if collision_entities:
        entity_tile_x = [int(e[1] // 24) for e in collision_entities]
        entity_tile_y = [level_data.height - 1 - int(e[2] // 24) for e in collision_entities]  # Flip Y
        ax2.scatter(entity_tile_x, entity_tile_y, c='orange', s=50, marker='D', 
                   label='Collision Entities', edgecolors='black', linewidth=1)
    
    ax2.set_xlabel('Tile X')
    ax2.set_ylabel('Tile Y (Corrected)')
    ax2.set_title('Tile Map (Corrected Coordinate System)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/nclone/comprehensive_traversability_debug.png', dpi=150, bbox_inches='tight')
    print("📊 Saved corrected visualization to comprehensive_traversability_debug.png")

if __name__ == "__main__":
    analyze_tile_traversability()
    print("\n🎉 COMPREHENSIVE TRAVERSABILITY ANALYSIS COMPLETE!")