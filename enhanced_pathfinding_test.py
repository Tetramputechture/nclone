#!/usr/bin/env python3
"""
Enhanced pathfinding test with waypoint-based multi-hop navigation.

This script tests the new waypoint-based pathfinding system that enables
physics-accurate navigation through chains of shorter jump/fall connections.
"""

import os
import sys
import math

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine
from nclone.constants.physics_constants import MAX_JUMP_DISTANCE, MAX_FALL_DISTANCE


def enhanced_pathfinding_test():
    """Test the enhanced pathfinding system with waypoint-based navigation."""
    print("=" * 80)
    print("🚀 ENHANCED PATHFINDING TEST - WAYPOINT-BASED MULTI-HOP NAVIGATION")
    print("=" * 80)
    
    # Initialize environment
    print("🔧 Initializing environment...")
    env = BasicLevelNoGold(render_mode="rgb_array", custom_map_path="nclone/test_maps/doortest")
    
    # Reset environment to get initial state
    env.reset()
    
    # Get ninja position and target from environment
    ninja_position = env.nplay_headless.ninja_position()
    leftmost_switch_position = (552, 204)  # Known from analysis
    
    print(f"✅ Ninja position: {ninja_position}")
    print(f"🎯 Target switch: {leftmost_switch_position}")
    
    # Calculate direct distance
    dx = leftmost_switch_position[0] - ninja_position[0]
    dy = leftmost_switch_position[1] - ninja_position[1]
    direct_distance = math.sqrt(dx * dx + dy * dy)
    print(f"📏 Direct distance: {direct_distance:.1f}px")
    
    # Show physics constraints
    print(f"⚡ Physics constraints:")
    print(f"   MAX_JUMP_DISTANCE: {MAX_JUMP_DISTANCE}px")
    print(f"   MAX_FALL_DISTANCE: {MAX_FALL_DISTANCE}px")
    print(f"   Direct distance exceeds jump limit: {direct_distance > MAX_JUMP_DISTANCE}")
    
    # Build enhanced graph with waypoint system
    print("\n🔧 Building enhanced graph with waypoint pathfinding...")
    builder = HierarchicalGraphBuilder()
    
    try:
        graph = builder.build_graph(env.level_data, ninja_position)
        print(f"✅ Enhanced graph built successfully!")
        
        # Print graph statistics
        if hasattr(graph, 'sub_cell_graph'):
            sub_graph = graph.sub_cell_graph
            print(f"📊 Graph statistics:")
            print(f"   Nodes: {len(sub_graph.node_features) if hasattr(sub_graph, 'node_features') else 'N/A'}")
            print(f"   Edges: {len(sub_graph.edge_index[0]) if hasattr(sub_graph, 'edge_index') else 'N/A'}")
            
            # Count edge types
            if hasattr(sub_graph, 'edge_types'):
                edge_type_counts = {}
                for edge_type in sub_graph.edge_types:
                    edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
                
                print(f"   Edge types:")
                for edge_type, count in edge_type_counts.items():
                    edge_name = ["WALK", "JUMP", "WALL_SLIDE", "FALL", "ONE_WAY", "FUNCTIONAL"][edge_type]
                    print(f"     {edge_name}: {count}")
        
        # Test pathfinding with enhanced system
        print("\n🔍 Testing enhanced pathfinding...")
        pathfinder = PathfindingEngine()
        
        # Find path from ninja to leftmost switch
        path_result = pathfinder.find_shortest_path(
            graph.sub_cell_graph,
            ninja_position,
            leftmost_switch_position
        )
        
        path = path_result.path if path_result and path_result.success else None
        
        if path is not None:
            print(f"✅ Path found with {len(path)} waypoints!")
            
            # Analyze path segments
            print(f"\n📍 Path analysis:")
            total_distance = 0
            movement_types = []
            
            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]
                
                segment_dx = end[0] - start[0]
                segment_dy = end[1] - start[1]
                segment_distance = math.sqrt(segment_dx * segment_dx + segment_dy * segment_dy)
                total_distance += segment_distance
                
                # Classify movement type
                if abs(segment_dy) < 12:  # Roughly horizontal
                    movement_type = "WALK"
                elif segment_dy < -12:  # Upward
                    movement_type = "JUMP"
                else:  # Downward
                    movement_type = "FALL"
                
                movement_types.append(movement_type)
                
                print(f"   Segment {i+1}: ({start[0]:.1f}, {start[1]:.1f}) -> ({end[0]:.1f}, {end[1]:.1f})")
                print(f"     Distance: {segment_distance:.1f}px, Type: {movement_type}")
                
                # Check if segment is physics-accurate
                if movement_type == "JUMP" and segment_distance > MAX_JUMP_DISTANCE:
                    print(f"     ⚠️  WARNING: Jump distance {segment_distance:.1f}px exceeds limit {MAX_JUMP_DISTANCE}px")
                elif movement_type == "FALL" and segment_distance > MAX_FALL_DISTANCE:
                    print(f"     ⚠️  WARNING: Fall distance {segment_distance:.1f}px exceeds limit {MAX_FALL_DISTANCE}px")
                else:
                    print(f"     ✅ Physics-accurate segment")
            
            print(f"\n📊 Path summary:")
            print(f"   Total segments: {len(path) - 1}")
            print(f"   Total distance: {total_distance:.1f}px")
            print(f"   Movement types: {set(movement_types)}")
            print(f"   Movement diversity: {len(set(movement_types))}/3 types")
            
            # Compare with expected path from user image
            expected_segments = 7  # From user description: 7-8 segments
            expected_types = {"WALK", "JUMP", "FALL"}  # Mixed movement types
            
            print(f"\n🎯 Comparison with expected path:")
            print(f"   Expected segments: ~{expected_segments}")
            print(f"   Actual segments: {len(path) - 1}")
            print(f"   Expected types: {expected_types}")
            print(f"   Actual types: {set(movement_types)}")
            
            # Success criteria
            segments_ok = len(path) - 1 >= 5  # At least 5 segments for complex navigation
            types_ok = len(set(movement_types)) >= 2  # At least 2 movement types
            physics_ok = all(
                (mt == "JUMP" and math.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2) <= MAX_JUMP_DISTANCE) or
                (mt == "FALL" and math.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2) <= MAX_FALL_DISTANCE) or
                mt == "WALK"
                for i, mt in enumerate(movement_types)
            )
            
            print(f"\n🏆 SUCCESS CRITERIA:")
            print(f"   ✅ Sufficient segments: {segments_ok} ({len(path) - 1} >= 5)")
            print(f"   ✅ Movement diversity: {types_ok} ({len(set(movement_types))} >= 2)")
            print(f"   ✅ Physics accuracy: {physics_ok}")
            
            if segments_ok and types_ok and physics_ok:
                print(f"\n🎉 ENHANCED PATHFINDING SUCCESS!")
                print(f"   The waypoint-based system successfully created a physics-accurate")
                print(f"   multi-hop path with diverse movement types!")
            else:
                print(f"\n⚠️  PARTIAL SUCCESS - Some criteria not met")
                
        else:
            print(f"❌ No path found with enhanced system")
            
            # Try to diagnose the issue
            print(f"\n🔍 Diagnosing pathfinding failure...")
            
            # Check if waypoint system was activated
            if hasattr(builder.edge_builder, 'waypoint_pathfinder'):
                waypoint_stats = builder.edge_builder.waypoint_pathfinder.get_waypoint_statistics()
                print(f"   Waypoint statistics: {waypoint_stats}")
                
                if waypoint_stats['total_waypoints'] == 0:
                    print(f"   ⚠️  No waypoints created - level geometry analysis may have failed")
                else:
                    print(f"   ✅ Waypoints created successfully")
            else:
                print(f"   ❌ Waypoint pathfinder not found - enhanced edge builder not used")
    
    except Exception as e:
        print(f"❌ Enhanced graph building failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n" + "=" * 80)
    print(f"🏁 ENHANCED PATHFINDING TEST COMPLETE")
    print(f"=" * 80)


if __name__ == "__main__":
    enhanced_pathfinding_test()