#!/usr/bin/env python3
"""
Simple physics pathfinding test to validate the waypoint system.
"""

import os
import sys
import math

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine


def simple_physics_test():
    """Simple test of the physics pathfinding system."""
    
    print("🚀 SIMPLE PHYSICS PATHFINDING TEST")
    print("=" * 50)
    
    try:
        # Initialize environment
        print("🔧 Initializing environment...")
        env = BasicLevelNoGold(render_mode="rgb_array")
        
        # Get positions
        ninja_position = (132.0, 444.0)
        target_position = (396.0, 228.0)  # Switch found by the system
        
        distance = math.sqrt((target_position[0] - ninja_position[0])**2 + (target_position[1] - ninja_position[1])**2)
        print(f"✅ Ninja: {ninja_position}, Target: {target_position}")
        print(f"📏 Distance: {distance:.1f}px")
        
        # Build graph
        print("🔧 Building graph...")
        builder = HierarchicalGraphBuilder()
        graph = builder.build_graph(env.level_data, ninja_position)
        
        print(f"✅ Graph built: {graph.sub_cell_graph.num_nodes} nodes, {graph.sub_cell_graph.num_edges} edges")
        
        # Test pathfinding
        print("🔍 Testing pathfinding...")
        pathfinder = PathfindingEngine()
        
        path_result = pathfinder.find_shortest_path(
            graph.sub_cell_graph,
            ninja_position,
            target_position
        )
        
        if path_result and path_result.success and path_result.path:
            path = path_result.path
            print(f"✅ Path found with {len(path)} segments!")
            
            # Analyze path
            total_distance = 0.0
            for i in range(len(path) - 1):
                segment_distance = math.sqrt(
                    (path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2
                )
                total_distance += segment_distance
                print(f"   Segment {i+1}: {segment_distance:.1f}px")
            
            print(f"📊 Total path distance: {total_distance:.1f}px")
            print(f"📊 Path efficiency: {(distance / total_distance * 100):.1f}%")
            
            # Check if path meets criteria
            success = len(path) >= 5  # Multi-segment path
            print(f"🎯 Success: {success} (path has {len(path)} segments)")
            
        else:
            print("❌ No path found")
            
            # Check waypoint statistics
            if hasattr(builder.edge_builder, 'waypoint_pathfinder'):
                stats = builder.edge_builder.waypoint_pathfinder.get_physics_statistics()
                print(f"📊 Waypoint stats: {stats}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("🏁 Test complete")
    return True


if __name__ == "__main__":
    simple_physics_test()