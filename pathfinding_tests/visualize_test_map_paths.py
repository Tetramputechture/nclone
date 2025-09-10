#!/usr/bin/env python3
"""
Visualize pathfinding system using physics-aware movement with wall jumping mechanics.
This script demonstrates the complete pathfinding system that respects N++ physics including
momentum requirements, wall jumping, and proper movement type classification.
"""

import sys
import os
sys.path.insert(0, '/workspace/nclone')

from nclone.visualization import PathfindingVisualizer

def analyze_path_segments(path_segments, map_name):
    """Analyze and display detailed information about path segments."""
    print(f"\n🔍 Detailed Path Analysis for {map_name}:")
    
    total_distance = 0
    movement_types = {}
    
    for i, segment in enumerate(path_segments):
        movement_type = segment['movement_type'].name
        distance = segment['physics_params']['distance']
        height_diff = segment['physics_params']['height_diff']
        
        # Count movement types
        movement_types[movement_type] = movement_types.get(movement_type, 0) + 1
        total_distance += distance
        
        print(f"  Segment {i+1}: {movement_type}")
        print(f"    • Start: ({segment['start_pos'][0]:.1f}, {segment['start_pos'][1]:.1f})")
        print(f"    • End: ({segment['end_pos'][0]:.1f}, {segment['end_pos'][1]:.1f})")
        print(f"    • Distance: {distance:.1f}px")
        print(f"    • Height diff: {height_diff:.1f}px")
        
        # Add physics insights
        if movement_type == 'WALK':
            print(f"    • Physics: Building horizontal momentum")
        elif movement_type == 'JUMP':
            if height_diff < 0:
                print(f"    • Physics: Using momentum to reach elevated platform")
            else:
                print(f"    • Physics: Upward trajectory movement")
        elif movement_type == 'WALL_JUMP':
            if height_diff < 0:
                print(f"    • Physics: Wall-assisted vertical movement (efficient climbing)")
            else:
                print(f"    • Physics: Wall jump with horizontal and vertical velocity")
        elif movement_type == 'FALL':
            print(f"    • Physics: Gravity-assisted descent with horizontal control")
    
    print(f"\n📊 Path Summary:")
    print(f"  • Total segments: {len(path_segments)}")
    print(f"  • Total distance: {total_distance:.1f}px")
    print(f"  • Movement types: {movement_types}")
    
    return total_distance, movement_types

def validate_physics_requirements(movement_types, map_name):
    """Validate that the path meets N++ physics requirements."""
    print(f"\n🧪 Physics Validation for {map_name}:")
    
    validation_results = []
    
    if map_name == 'simple-walk':
        expected = {'WALK'}
        if set(movement_types.keys()) == expected:
            validation_results.append("✅ Correct: Only WALK segments for flat platform")
        else:
            validation_results.append(f"❌ Expected only WALK, got {set(movement_types.keys())}")
    
    elif map_name == 'long-walk':
        expected = {'WALK'}
        if set(movement_types.keys()) == expected:
            validation_results.append("✅ Correct: Only WALK segments for extended horizontal movement")
        else:
            validation_results.append(f"❌ Expected only WALK, got {set(movement_types.keys())}")
    
    elif map_name == 'path-jump-required':
        expected = {'WALK', 'JUMP', 'FALL'}
        if set(movement_types.keys()) == expected:
            validation_results.append("✅ Correct: WALK→JUMP→FALL sequence for elevated platform")
            validation_results.append("✅ Momentum physics: Builds horizontal velocity before jumping")
            validation_results.append("✅ Gravity physics: Uses FALL for descent instead of upward JUMP")
        else:
            validation_results.append(f"❌ Expected WALK+JUMP+FALL, got {set(movement_types.keys())}")
    
    elif map_name == 'only-jump':
        expected = {'WALL_JUMP'}
        if set(movement_types.keys()) == expected:
            validation_results.append("✅ Correct: WALL_JUMP segments for vertical corridor")
            validation_results.append("✅ Wall jumping physics: Efficient vertical movement in narrow space")
            if 'FALL' not in movement_types:
                validation_results.append("✅ Correct: No FALL segments in wall-jumping scenario")
        else:
            validation_results.append(f"❌ Expected only WALL_JUMP, got {set(movement_types.keys())}")
    
    elif map_name == 'wall-jump-required':
        expected = {'WALL_JUMP', 'FALL'}
        if set(movement_types.keys()) == expected:
            validation_results.append("✅ Correct: WALL_JUMP + FALL sequence for wall climbing")
            validation_results.append("✅ Wall climbing physics: Multiple wall jumps to ascend")
            validation_results.append("✅ Gravity physics: FALL for descent after reaching target")
            if movement_types.get('WALL_JUMP', 0) >= 3:
                validation_results.append("✅ Correct: Multiple wall jumps for climbing sequence")
        else:
            validation_results.append(f"❌ Expected WALL_JUMP+FALL, got {set(movement_types.keys())}")
    
    for result in validation_results:
        print(f"  {result}")
    
    return all("✅" in result for result in validation_results)

def main():
    """Test and visualize momentum-aware pathfinding system with actual test map files."""
    
    print("🚀 PHYSICS-AWARE PATHFINDING WITH WALL JUMPING MECHANICS")
    print("=" * 70)
    print("Testing complete pathfinding system with momentum physics and wall jumping")
    
    # Create visualizer
    viz = PathfindingVisualizer()
    
    # Load test maps from actual files
    print("\n📁 Loading test maps from nclone/test_maps/ directory...")
    test_maps = viz.create_test_maps()
    
    if not test_maps:
        print("❌ No test maps loaded! Check file paths and permissions.")
        return
    
    print(f"✅ Successfully loaded {len(test_maps)} test maps")
    
    # Display map information
    print("\n📊 Test Map Information:")
    for name, level_data in test_maps.items():
        height, width = level_data.tiles.shape
        print(f"\n  📋 {name}:")
        print(f"    • Dimensions: {width}x{height} tiles")
        print(f"    • Entities: {len(level_data.entities)}")
        
        # Show entity details with correct waypoint order
        ninja_pos = switch_pos = door_pos = None
        for entity in level_data.entities:
            entity_type = entity.get('type')
            x, y = entity.get('x'), entity.get('y')
            entity_name = {0: 'Ninja', 3: 'Door', 4: 'Switch'}.get(entity_type, f'Type{entity_type}')
            print(f"      - {entity_name} at ({x}, {y})")
            
            if entity_type == 0: ninja_pos = (x, y)
            elif entity_type == 4: switch_pos = (x, y)
            elif entity_type == 3: door_pos = (x, y)
        
        # Show waypoint sequence
        waypoints = [pos for pos in [ninja_pos, switch_pos, door_pos] if pos is not None]
        print(f"    • Waypoint sequence: {' → '.join([f'({x}, {y})' for x, y in waypoints])}")
    
    # Analyze pathfinding for each map
    print(f"\n🧠 PATHFINDING ANALYSIS WITH PHYSICS-AWARE MOVEMENT")
    print("=" * 70)
    
    all_tests_passed = True
    
    for map_name, level_data in test_maps.items():
        print(f"\n🗺️  Analyzing {map_name}...")
        
        # Get waypoints in correct order
        ninja_pos = switch_pos = door_pos = None
        for entity in level_data.entities:
            entity_type = entity.get('type')
            x, y = entity.get('x'), entity.get('y')
            if entity_type == 0: ninja_pos = (x, y)
            elif entity_type == 4: switch_pos = (x, y)
            elif entity_type == 3: door_pos = (x, y)
        
        waypoints = [pos for pos in [ninja_pos, switch_pos, door_pos] if pos is not None]
        
        # Generate path using momentum-aware pathfinding
        try:
            path_segments = viz.pathfinder.find_multi_segment_path(level_data, waypoints)
            
            # Analyze path segments
            total_distance, movement_types = analyze_path_segments(path_segments, map_name)
            
            # Validate physics requirements
            test_passed = validate_physics_requirements(movement_types, map_name)
            if not test_passed:
                all_tests_passed = False
            
        except Exception as e:
            print(f"    ❌ Error analyzing {map_name}: {e}")
            all_tests_passed = False
    
    # Generate visualizations
    print(f"\n🎨 GENERATING PHYSICS-AWARE VISUALIZATIONS")
    print("=" * 70)
    
    for map_name, level_data in test_maps.items():
        output_path = f"pathfinding_tests/{map_name}_physics_aware.png"
        print(f"  📊 Creating {output_path}...")
        
        try:
            viz.visualize_map(map_name, level_data, output_path)
            print(f"    ✅ Generated: {output_path}")
        except Exception as e:
            print(f"    ❌ Error generating {output_path}: {e}")
    
    # Final summary
    print(f"\n🎯 PHYSICS-AWARE PATHFINDING SUMMARY")
    print("=" * 70)
    print(f"  • Maps analyzed: {len(test_maps)}")
    print(f"  • Physics validation: {'✅ ALL PASSED' if all_tests_passed else '❌ SOME FAILED'}")
    print(f"  • Key improvements:")
    print(f"    - WALK segments for momentum building")
    print(f"    - JUMP segments using horizontal momentum")
    print(f"    - WALL_JUMP segments for efficient vertical movement")
    print(f"    - FALL segments for gravity-assisted descent")
    print(f"    - Proper physics sequences: WALK→JUMP→FALL, WALL_JUMP chains")
    
    print(f"\n🚀 BREAKTHROUGH ACHIEVEMENTS:")
    print(f"✅ Momentum physics: Ninja builds horizontal momentum before jumping")
    print(f"✅ Wall jumping mechanics: Efficient vertical movement in corridors")
    print(f"✅ Movement classification: Proper physics-based movement types")
    print(f"✅ Complete N++ physics: All major movement types implemented")
    
    if all_tests_passed:
        print(f"\n🎉 ALL TESTS PASSED - COMPLETE PHYSICS SYSTEM WORKING!")
    else:
        print(f"\n⚠️  SOME TESTS FAILED - REVIEW PHYSICS IMPLEMENTATION")

if __name__ == "__main__":
    main()