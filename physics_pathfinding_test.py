#!/usr/bin/env python3
"""
Comprehensive physics-accurate pathfinding test.

This script validates the complete physics-enhanced waypoint pathfinding system
for N++ multi-hop navigation with all game physics constraints.
"""

import os
import sys
import math

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine
from nclone.constants.physics_constants import MAX_JUMP_DISTANCE, MAX_FALL_DISTANCE, NINJA_RADIUS


def physics_pathfinding_test():
    """Test the complete physics-enhanced pathfinding system."""
    
    print("=" * 80)
    print("🚀 PHYSICS-ACCURATE PATHFINDING TEST - COMPREHENSIVE VALIDATION")
    print("=" * 80)
    
    try:
        # Initialize environment
        print("🔧 Initializing N++ environment...")
        env = BasicLevelNoGold(render_mode="rgb_array")
        
        # Get ninja and target positions
        ninja_position = (132.0, 444.0)  # Known ninja spawn position
        leftmost_switch_position = (552.0, 204.0)  # Known leftmost switch position
        
        print(f"✅ Ninja position: {ninja_position}")
        print(f"🎯 Target switch: {leftmost_switch_position}")
        
        # Calculate direct distance and physics constraints
        dx = leftmost_switch_position[0] - ninja_position[0]
        dy = leftmost_switch_position[1] - ninja_position[1]
        direct_distance = math.sqrt(dx * dx + dy * dy)
        
        print(f"📏 Direct distance: {direct_distance:.1f}px")
        print(f"⚡ Physics constraints:")
        print(f"   MAX_JUMP_DISTANCE: {MAX_JUMP_DISTANCE:.1f}px")
        print(f"   MAX_FALL_DISTANCE: {MAX_FALL_DISTANCE:.1f}px")
        print(f"   NINJA_RADIUS: {NINJA_RADIUS:.1f}px")
        print(f"   Direct distance exceeds jump limit: {direct_distance > MAX_JUMP_DISTANCE}")
        
        # Build physics-enhanced graph
        print("\n🔧 Building physics-enhanced graph with comprehensive waypoint pathfinding...")
        builder = HierarchicalGraphBuilder()
        graph = builder.build_graph(env.level_data, ninja_position)
        
        print("✅ Physics-enhanced graph built successfully!")
        
        # Display graph statistics
        print(f"📊 Graph statistics:")
        print(f"   Sub-cell nodes: {graph.sub_cell_graph.num_nodes}")
        print(f"   Sub-cell edges: {graph.sub_cell_graph.num_edges}")
        
        # Count edge types in sub-cell graph
        edge_type_counts = {}
        for i in range(graph.sub_cell_graph.num_edges):
            if graph.sub_cell_graph.edge_mask[i]:
                edge_type = graph.sub_cell_graph.edge_types[i]
                edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        print(f"   Edge types:")
        from nclone.graph.common import EdgeType
        for edge_type_val, count in edge_type_counts.items():
            edge_type_name = EdgeType(edge_type_val).name
            print(f"     {edge_type_name}: {count}")
        
        # Test physics-accurate pathfinding
        print("\n🔍 Testing physics-accurate pathfinding...")
        pathfinder = PathfindingEngine()
        
        # Find path from ninja to leftmost switch
        path_result = pathfinder.find_shortest_path(
            graph.sub_cell_graph,
            ninja_position,
            leftmost_switch_position
        )
        
        path = path_result.path if path_result and path_result.success else None
        
        if path is not None:
            print(f"✅ Physics-accurate path found!")
            print(f"   Path length: {len(path)} segments")
            
            # Analyze path physics
            print(f"📊 Path analysis:")
            total_path_distance = 0.0
            movement_types = []
            
            for i in range(len(path) - 1):
                current_pos = path[i]
                next_pos = path[i + 1]
                
                segment_dx = next_pos[0] - current_pos[0]
                segment_dy = next_pos[1] - current_pos[1]
                segment_distance = math.sqrt(segment_dx * segment_dx + segment_dy * segment_dy)
                total_path_distance += segment_distance
                
                # Determine movement type
                if segment_dy <= 0:  # Upward or horizontal
                    movement_type = "JUMP" if segment_distance > 50 else "WALK"
                else:  # Downward
                    movement_type = "FALL" if segment_distance > 50 else "WALK"
                
                movement_types.append(movement_type)
                
                print(f"   Segment {i+1}: ({current_pos[0]:.1f}, {current_pos[1]:.1f}) -> ({next_pos[0]:.1f}, {next_pos[1]:.1f})")
                print(f"     Distance: {segment_distance:.1f}px, Type: {movement_type}")
                
                # Validate physics constraints
                if movement_type == "JUMP" and segment_distance > MAX_JUMP_DISTANCE:
                    print(f"     ⚠️  Physics violation: Jump distance {segment_distance:.1f}px exceeds limit {MAX_JUMP_DISTANCE:.1f}px")
                elif movement_type == "FALL" and segment_distance > MAX_FALL_DISTANCE:
                    print(f"     ⚠️  Physics violation: Fall distance {segment_distance:.1f}px exceeds limit {MAX_FALL_DISTANCE:.1f}px")
                else:
                    print(f"     ✅ Physics constraint satisfied")
            
            print(f"\n📈 Path summary:")
            print(f"   Total path distance: {total_path_distance:.1f}px")
            print(f"   Direct distance: {direct_distance:.1f}px")
            print(f"   Path efficiency: {(direct_distance / total_path_distance * 100):.1f}%")
            
            # Count movement types
            movement_counts = {}
            for movement in movement_types:
                movement_counts[movement] = movement_counts.get(movement, 0) + 1
            
            print(f"   Movement type distribution:")
            for movement, count in movement_counts.items():
                print(f"     {movement}: {count} segments")
            
            # Validate expected path characteristics
            unique_movements = len(set(movement_types))
            print(f"   Movement diversity: {unique_movements} different types")
            
            # Success criteria validation
            success_criteria = {
                "Path found": path is not None,
                "Multi-segment path": len(path) >= 7,  # Expected 7-8 segments
                "Movement diversity": unique_movements >= 2,  # At least WALK and JUMP
                "Physics compliance": all(
                    (mt != "JUMP" or dist <= MAX_JUMP_DISTANCE) and 
                    (mt != "FALL" or dist <= MAX_FALL_DISTANCE)
                    for mt, dist in zip(movement_types, [
                        math.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)
                        for i in range(len(path) - 1)
                    ])
                )
            }
            
            print(f"\n🎯 Success criteria validation:")
            all_passed = True
            for criterion, passed in success_criteria.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"   {criterion}: {status}")
                if not passed:
                    all_passed = False
            
            if all_passed:
                print(f"\n🏆 ALL SUCCESS CRITERIA MET - PHYSICS-ACCURATE PATHFINDING WORKING!")
            else:
                print(f"\n⚠️  Some success criteria not met - further optimization needed")
                
        else:
            print("❌ No physics-accurate path found")
            print("🔍 Diagnosing pathfinding failure...")
            
            # Get waypoint statistics from the builder
            if hasattr(builder.edge_builder, 'waypoint_pathfinder'):
                stats = builder.edge_builder.waypoint_pathfinder.get_physics_statistics()
                print(f"   Waypoint statistics: {stats}")
            
    except Exception as e:
        print(f"❌ Physics pathfinding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("🏁 PHYSICS-ACCURATE PATHFINDING TEST COMPLETE")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    physics_pathfinding_test()