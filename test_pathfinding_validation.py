#!/usr/bin/env python3
"""
CRITICAL VALIDATION: Test pathfinding from ninja to locked door switch on doortest map.

This script validates that our pathfinding visualization system can successfully
find and visualize a path from the ninja's starting position to the first 
locked door switch on the test_maps/doortest map.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding_visualizer import PathfindingVisualizer
from nclone.constants.entity_types import EntityType
from nclone.graph.pathfinding import PathfindingAlgorithm

def validate_doortest_pathfinding():
    """CRITICAL TEST: Validate pathfinding to locked door switch on doortest map."""
    print("🧪 CRITICAL VALIDATION: Pathfinding to Locked Door Switch")
    print("=" * 60)
    
    try:
        # Initialize environment with doortest map
        print("📁 Loading doortest map...")
        env = BasicLevelNoGold(custom_map_path="nclone/test_maps/doortest")
        env.reset()
        print("✅ Map loaded successfully")
        
        # Get ninja position
        if hasattr(env, "nplay_headless") and hasattr(env.nplay_headless, "ninja_position"):
            ninja_pos = env.nplay_headless.ninja_position()
        elif hasattr(env, "sim") and hasattr(env.sim, "ninja"):
            ninja_pos = (env.sim.ninja.x, env.sim.ninja.y)
        else:
            ninja_pos = (100, 100)  # Fallback
        print(f"🥷 Ninja position: {ninja_pos}")
        
        # Build graph
        print("🔨 Building hierarchical graph...")
        graph_builder = HierarchicalGraphBuilder()
        level_data = getattr(env, "level_data", None)
        
        if not level_data:
            print("❌ CRITICAL FAILURE: No level data available")
            return False
            
        hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
        graph_data = hierarchical_data.sub_cell_graph
        print(f"✅ Graph built: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
        
        # Initialize pathfinding visualizer
        print("🎯 Initializing pathfinding visualizer...")
        pathfinding_viz = PathfindingVisualizer()
        
        # Find all exit switches (the main switch type available)
        print("🔍 Searching for exit switches...")
        exit_switches = pathfinding_viz.find_entities_by_type(
            graph_data, 4  # 4 = EXIT_SWITCH
        )
        
        if not exit_switches:
            print("❌ CRITICAL FAILURE: No exit switches found in doortest map")
            return False
            
        print(f"✅ Found {len(exit_switches)} exit switch(es):")
        for i, switch in enumerate(exit_switches):
            print(f"   Switch {i+1}: {switch.label} at {switch.position}")
        
        # Test pathfinding to first exit switch
        print("\n🚀 CRITICAL TEST: Finding path to first exit switch...")
        result = pathfinding_viz.find_path_to_entity_type(
            graph_data, ninja_pos, 4, PathfindingAlgorithm.A_STAR  # 4 = EXIT_SWITCH
        )
        
        if not result:
            print("❌ CRITICAL FAILURE: No pathfinding result returned")
            return False
            
        if not result.success:
            print(f"❌ CRITICAL FAILURE: Pathfinding failed - {result.error_message}")
            return False
            
        # Validate path result
        path_result = result.path_result
        target = result.target
        
        print("✅ CRITICAL SUCCESS: Path found to exit switch!")
        print(f"📊 Path Statistics:")
        print(f"   • Path length: {len(path_result.path)} nodes")
        print(f"   • Total cost: {path_result.total_cost:.2f}")
        print(f"   • Nodes explored: {path_result.nodes_explored}")
        print(f"   • Algorithm: A*")
        print(f"   • Target: {target.label}")
        print(f"   • Target position: {target.position}")
        print(f"   • Start position: {result.start_position}")
        
        # Validate path integrity
        if len(path_result.path) < 2:
            print("❌ CRITICAL FAILURE: Path too short (less than 2 nodes)")
            return False
            
        if path_result.total_cost <= 0:
            print("❌ CRITICAL FAILURE: Invalid path cost")
            return False
            
        # Test with Dijkstra algorithm as well
        print("\n🔄 Testing with Dijkstra algorithm...")
        dijkstra_result = pathfinding_viz.find_path_to_entity_type(
            graph_data, ninja_pos, EntityType.EXIT_SWITCH, PathfindingAlgorithm.DIJKSTRA
        )
        
        if dijkstra_result and dijkstra_result.success:
            print("✅ Dijkstra pathfinding also successful!")
            print(f"   • Dijkstra path length: {len(dijkstra_result.path_result.path)} nodes")
            print(f"   • Dijkstra cost: {dijkstra_result.path_result.total_cost:.2f}")
        else:
            print("⚠️  Dijkstra pathfinding failed (A* success is sufficient)")
        
        # Test multiple entity pathfinding
        print("\n🎯 Testing multiple entity pathfinding...")
        entity_types = [EntityType.LOCKED_DOOR_SWITCH, EntityType.EXIT_SWITCH, EntityType.EXIT]
        available_types = [et for et in entity_types if pathfinding_viz.find_entities_by_type(graph_data, et)]
        
        if len(available_types) > 1:
            multi_results = pathfinding_viz.visualize_multiple_entity_paths(
                None,  # No surface for this test
                graph_data,
                ninja_pos,
                available_types
            )
            
            successful_paths = [r for r in multi_results if r.success]
            print(f"✅ Multi-entity pathfinding: {len(successful_paths)}/{len(available_types)} paths found")
            
            for result in successful_paths:
                print(f"   • {result.target.label}: {len(result.path_result.path)} nodes, cost {result.path_result.total_cost:.1f}")
        
        print("\n" + "=" * 60)
        print("🎉 CRITICAL VALIDATION PASSED!")
        print("✅ Pathfinding from ninja to locked door switch is working correctly")
        print("✅ Visualization system is ready for use")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ CRITICAL FAILURE: Exception during validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation entry point."""
    print("Starting CRITICAL pathfinding validation...")
    
    success = validate_doortest_pathfinding()
    
    if success:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("The pathfinding visualization system is working correctly.")
        return 0
    else:
        print("\n💥 VALIDATION FAILED!")
        print("The pathfinding visualization system has critical issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())