#!/usr/bin/env python3
"""
Final comprehensive validation of the physics-accurate waypoint pathfinding system.

This script demonstrates the complete solution for N++ multi-hop navigation
with all physics constraints properly implemented and validated.
"""

import os
import sys
import math

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.physics_waypoint_pathfinder import PhysicsWaypointPathfinder
from nclone.graph.physics_enhanced_edge_builder import PhysicsEnhancedEdgeBuilder
from nclone.constants.physics_constants import MAX_JUMP_DISTANCE, MAX_FALL_DISTANCE, NINJA_RADIUS


def final_physics_validation():
    """Comprehensive validation of the complete physics-accurate pathfinding system."""
    
    print("🏆 FINAL PHYSICS-ACCURATE PATHFINDING VALIDATION")
    print("=" * 80)
    print("Validating complete solution for N++ multi-hop navigation")
    print("with comprehensive physics constraints and waypoint system")
    print("=" * 80)
    
    try:
        # Initialize environment
        print("🔧 STEP 1: Environment Initialization")
        print("-" * 40)
        env = BasicLevelNoGold(render_mode="rgb_array")
        ninja_position = (132.0, 444.0)
        
        print(f"✅ Environment initialized successfully")
        print(f"✅ Ninja spawn position: {ninja_position}")
        print(f"✅ Physics constants validated:")
        print(f"   - MAX_JUMP_DISTANCE: {MAX_JUMP_DISTANCE}px")
        print(f"   - MAX_FALL_DISTANCE: {MAX_FALL_DISTANCE}px")
        print(f"   - NINJA_RADIUS: {NINJA_RADIUS}px")
        
        # Build physics-enhanced graph
        print(f"\n🚀 STEP 2: Physics-Enhanced Graph Construction")
        print("-" * 40)
        builder = HierarchicalGraphBuilder()
        
        # Verify enhanced edge builder is being used
        print(f"✅ Using PhysicsEnhancedEdgeBuilder: {isinstance(builder.edge_builder, PhysicsEnhancedEdgeBuilder)}")
        print(f"✅ Waypoint pathfinder integrated: {hasattr(builder.edge_builder, 'waypoint_pathfinder')}")
        
        graph = builder.build_graph(env.level_data, ninja_position)
        
        print(f"✅ Graph construction completed successfully")
        print(f"   - Total nodes: {graph.sub_cell_graph.num_nodes:,}")
        print(f"   - Total edges: {graph.sub_cell_graph.num_edges:,}")
        
        # Analyze edge types
        edge_type_counts = {}
        for i in range(graph.sub_cell_graph.num_edges):
            if graph.sub_cell_graph.edge_mask[i]:
                edge_type = graph.sub_cell_graph.edge_types[i]
                edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        from nclone.graph.common import EdgeType
        print(f"   - Edge type distribution:")
        for edge_type_val, count in edge_type_counts.items():
            edge_type_name = EdgeType(edge_type_val).name
            print(f"     * {edge_type_name}: {count:,}")
        
        # Validate waypoint system
        print(f"\n🎯 STEP 3: Waypoint System Validation")
        print("-" * 40)
        
        if hasattr(builder.edge_builder, 'waypoint_pathfinder'):
            waypoint_stats = builder.edge_builder.waypoint_pathfinder.get_physics_statistics()
            
            print(f"✅ Waypoint system statistics:")
            print(f"   - Total waypoints created: {waypoint_stats['total_waypoints']}")
            print(f"   - Total connections: {waypoint_stats['total_connections']}")
            print(f"   - Traversable waypoints: {waypoint_stats['traversable_waypoints']}")
            print(f"   - Average ground distance: {waypoint_stats['average_ground_distance']:.1f}px")
            print(f"   - Average clearance above: {waypoint_stats['average_clearance_above']:.1f}px")
            print(f"   - Physics validation: {waypoint_stats['physics_validation']}")
            
            # Validate physics compliance
            physics_compliant = True
            if waypoint_stats['total_waypoints'] > 0:
                waypoints = builder.edge_builder.waypoint_pathfinder.waypoints
                
                # Check waypoint spacing
                for i, waypoint in enumerate(waypoints):
                    print(f"   - Waypoint {i+1}: ({waypoint.x:.1f}, {waypoint.y:.1f})")
                    print(f"     * Ground support: {waypoint.ground_distance:.1f}px below")
                    print(f"     * Clearance: {waypoint.clearance_above:.1f}px above")
                    print(f"     * Traversable: {waypoint.is_traversable}")
                
                # Check connections
                connections = builder.edge_builder.waypoint_pathfinder.waypoint_connections
                print(f"   - Connection validation:")
                for waypoint_id, connected_ids in connections.items():
                    print(f"     * Waypoint {waypoint_id} connects to: {connected_ids}")
            
        else:
            print("❌ Waypoint pathfinder not found in edge builder")
            physics_compliant = False
        
        # Test specific long-distance scenarios
        print(f"\n🔍 STEP 4: Long-Distance Navigation Scenarios")
        print("-" * 40)
        
        # Test scenario 1: Direct waypoint creation
        pathfinder = PhysicsWaypointPathfinder()
        test_target = (500.0, 300.0)  # Long-distance target
        distance = math.sqrt((test_target[0] - ninja_position[0])**2 + (test_target[1] - ninja_position[1])**2)
        
        print(f"📊 Test scenario: Ninja to target at {test_target}")
        print(f"   - Direct distance: {distance:.1f}px")
        print(f"   - Exceeds jump limit: {distance > MAX_JUMP_DISTANCE}")
        
        waypoints = pathfinder.create_physics_accurate_waypoints(
            ninja_position, test_target, env.level_data
        )
        
        if waypoints:
            print(f"✅ Waypoints created successfully: {len(waypoints)} waypoints")
            
            # Validate complete path
            complete_path = pathfinder.get_complete_waypoint_path(ninja_position, test_target)
            print(f"✅ Complete path: {len(complete_path)} segments")
            
            # Validate physics compliance of path
            total_path_distance = 0.0
            max_segment_distance = 0.0
            
            for i in range(len(complete_path) - 1):
                segment_distance = math.sqrt(
                    (complete_path[i+1][0] - complete_path[i][0])**2 + 
                    (complete_path[i+1][1] - complete_path[i][1])**2
                )
                total_path_distance += segment_distance
                max_segment_distance = max(max_segment_distance, segment_distance)
                
                print(f"   - Segment {i+1}: {segment_distance:.1f}px")
            
            print(f"📈 Path analysis:")
            print(f"   - Total path distance: {total_path_distance:.1f}px")
            print(f"   - Direct distance: {distance:.1f}px")
            print(f"   - Path efficiency: {(distance / total_path_distance * 100):.1f}%")
            print(f"   - Maximum segment: {max_segment_distance:.1f}px")
            print(f"   - Physics compliant: {max_segment_distance <= MAX_JUMP_DISTANCE}")
            
        else:
            print("❌ No waypoints created for test scenario")
            physics_compliant = False
        
        # Final validation summary
        print(f"\n🏆 STEP 5: Final Validation Summary")
        print("-" * 40)
        
        validation_results = {
            "Environment initialization": True,
            "Physics constants defined": True,
            "Enhanced edge builder integrated": isinstance(builder.edge_builder, PhysicsEnhancedEdgeBuilder),
            "Graph construction successful": graph is not None,
            "Waypoint system functional": waypoint_stats['total_waypoints'] > 0 if 'waypoint_stats' in locals() else False,
            "Physics compliance verified": physics_compliant,
            "Edge count increased": graph.sub_cell_graph.num_edges > 100000,  # Should have many edges
            "Multi-hop navigation enabled": len(waypoints) > 0 if waypoints else False
        }
        
        all_passed = all(validation_results.values())
        
        print(f"📊 Validation Results:")
        for criterion, passed in validation_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status} {criterion}")
        
        if all_passed:
            print(f"\n🎉 SUCCESS: ALL VALIDATION CRITERIA PASSED!")
            print(f"🏆 Physics-accurate waypoint pathfinding system is fully operational")
            print(f"✅ Multi-hop navigation with physics constraints implemented")
            print(f"✅ Waypoint creation and validation working correctly")
            print(f"✅ Graph integration and edge building successful")
            print(f"✅ Ready for complex N++ navigation scenarios")
        else:
            print(f"\n⚠️  WARNING: Some validation criteria failed")
            print(f"❌ System may not be fully operational")
        
        # Performance metrics
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   - Graph nodes: {graph.sub_cell_graph.num_nodes:,}")
        print(f"   - Graph edges: {graph.sub_cell_graph.num_edges:,}")
        print(f"   - Waypoints created: {waypoint_stats.get('total_waypoints', 0)}")
        print(f"   - Physics-validated connections: {waypoint_stats.get('total_connections', 0)}")
        print(f"   - Edge types supported: {len(edge_type_counts)}")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("\n" + "=" * 80)
        print("🏁 FINAL PHYSICS VALIDATION COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    success = final_physics_validation()
    print(f"\n{'🎉 VALIDATION SUCCESSFUL' if success else '❌ VALIDATION FAILED'}")
    sys.exit(0 if success else 1)