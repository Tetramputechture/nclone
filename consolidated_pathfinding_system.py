#!/usr/bin/env python3
"""
Consolidated N++ Physics-Aware Pathfinding System

This is the single, authoritative script that demonstrates the
consolidated pathfinding system with validation and visualization.
"""

import sys
import os

# Add nclone to path
sys.path.insert(0, "/workspace/nclone")

from nclone.pathfinding import CorePathfinder
from nclone.visualization import PathfindingVisualizer
from nclone.graph.level_data import LevelData
import numpy as np

def create_test_maps():
    """Create the four validation test maps."""
    
    test_maps = {}
    
    # Simple-walk map: 9 tiles wide, single horizontal platform
    tiles = np.zeros((5, 9), dtype=int)
    tiles[3, :] = 1  # Ground platform on row 3
    entities = [
        {"type": 0, "x": 24, "y": 60},     # Ninja at leftmost tile
        {"type": 4, "x": 120, "y": 60},    # Exit switch at middle tile (5th)
        {"type": 3, "x": 192, "y": 60}     # Exit door at rightmost tile
    ]
    test_maps["simple-walk"] = LevelData(tiles, entities)
    
    # Long-walk map: 42 tiles wide, single horizontal platform
    tiles = np.zeros((5, 42), dtype=int)
    tiles[3, :] = 1  # Ground platform on row 3
    entities = [
        {"type": 0, "x": 24, "y": 60},      # Ninja at leftmost tile
        {"type": 4, "x": 960, "y": 60},     # Exit switch at 41st tile
        {"type": 3, "x": 984, "y": 60}      # Exit door at rightmost tile
    ]
    test_maps["long-walk"] = LevelData(tiles, entities)
    
    # Path-jump-required map: elevated switch position
    tiles = np.zeros((5, 9), dtype=int)
    tiles[3, :] = 1  # Ground platform on row 3
    tiles[2, 4] = 1  # Elevated tile for switch at column 4
    entities = [
        {"type": 0, "x": 24, "y": 60},     # Ninja on ground
        {"type": 4, "x": 120, "y": 36},    # Switch on elevated tile
        {"type": 3, "x": 192, "y": 60}     # Door on ground
    ]
    test_maps["path-jump-required"] = LevelData(tiles, entities)
    
    # Only-jump map: vertical corridor
    tiles = np.zeros((8, 3), dtype=int)
    tiles[:, 0] = 1  # Left wall
    tiles[:, 2] = 1  # Right wall
    entities = [
        {"type": 0, "x": 36, "y": 156},    # Ninja at bottom
        {"type": 4, "x": 36, "y": 108},    # Switch in middle
        {"type": 3, "x": 36, "y": 60}      # Door at top
    ]
    test_maps["only-jump"] = LevelData(tiles, entities)
    
    return test_maps

def validate_pathfinding():
    """Validate pathfinding system against test maps."""
    
    print("🧪 Validating Consolidated Pathfinding System")
    print("=" * 50)
    
    pathfinder = CorePathfinder()
    test_maps = create_test_maps()
    
    for map_name, level_data in test_maps.items():
        print(f"\n📍 Testing {map_name}")
        
        # Find entity positions
        ninja_pos = None
        switch_pos = None
        door_pos = None
        
        for entity in level_data.entities:
            entity_type = entity.get("type", 0)
            entity_x = entity.get("x", 0.0)
            entity_y = entity.get("y", 0.0)
            
            if entity_type == 0:  # Ninja
                ninja_pos = (entity_x, entity_y)
            elif entity_type == 4:  # Switch
                switch_pos = (entity_x, entity_y)
            elif entity_type == 3:  # Door
                door_pos = (entity_x, entity_y)
        
        # Create path waypoints
        waypoints = [ninja_pos]
        if switch_pos:
            waypoints.append(switch_pos)
        waypoints.append(door_pos)
        
        # Find path
        path_segments = pathfinder.find_multi_segment_path(level_data, waypoints)
        
        # Print results
        for i, segment in enumerate(path_segments):
            start_pos = segment['start_pos']
            end_pos = segment['end_pos']
            movement_type = segment['movement_type']
            distance = segment['physics_params']['distance']
            is_valid = segment['is_valid']
            
            print(f"  Segment {i+1}: {movement_type.name}")
            print(f"    Distance: {distance:.1f}px")
            print(f"    Valid: {'✅' if is_valid else '❌'}")
        
        # Print summary
        summary = pathfinder.get_path_summary(path_segments)
        print(f"  📊 Summary:")
        print(f"    Total distance: {summary['total_distance']:.1f}px")
        print(f"    Movement types: {summary['movement_type_counts']}")
        print(f"    All valid: {'✅' if summary['all_segments_valid'] else '❌'}")

def create_visualizations():
    """Create visualizations using the consolidated system."""
    
    print("\n🎨 Creating Consolidated Visualizations")
    print("=" * 50)
    
    visualizer = PathfindingVisualizer(tile_size=30)
    visualizer.create_all_visualizations()

def main():
    """Main function to run validation and create visualizations."""
    
    print("🚀 N++ Consolidated Physics-Aware Pathfinding System")
    print("=" * 60)
    print("This script demonstrates the authoritative pathfinding system")
    print("that consolidates all working physics-aware logic.")
    print()
    
    # Run validation
    validate_pathfinding()
    
    # Create visualizations
    create_visualizations()
    
    print("\n🎉 Consolidated pathfinding system validation complete!")
    print("Check the generated PNG files for visual validation.")

if __name__ == "__main__":
    main()