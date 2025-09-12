#!/usr/bin/env python3
"""
Test script to validate graph visualization fixes using the doortest map.

This script uses the BaseEnvironment map loading logic to properly load the doortest map
and test all three graph visualization issues:
1. Functional edges between switches and doors
2. Walkable edges in solid tiles (ninja radius clearance)
3. Pathfinding on traversable paths

The doortest map is the one shown in the user's screenshots.
"""

import os
import sys
import numpy as np

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.navigation import PathfindingEngine
from nclone.graph.common import EdgeType
from nclone.constants import TILE_PIXEL_SIZE


def test_doortest_map_issues():
    """Test all three graph visualization issues using the doortest map."""
    print("=" * 80)
    print("TESTING DOORTEST MAP GRAPH VISUALIZATION")
    print("=" * 80)
    print("Loading doortest map using BaseEnvironment logic...")
    
    # Create environment with doortest map (it's hardcoded to load doortest)
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42
    )
    
    # Reset environment to load the map
    env.reset()
    
    # Extract level data using BaseEnvironment method
    level_data = env.level_data
    print(f"✅ Loaded doortest map: ({level_data.width}, {level_data.height}) tiles, {len(level_data.entities)} entities")
    
    # Get ninja position
    ninja_pos = env.nplay_headless.ninja_position()
    print(f"✅ Ninja position: {ninja_pos}")
    
    # Build graph using HierarchicalGraphBuilder
    print("\nBuilding hierarchical graph...")
    graph_builder = HierarchicalGraphBuilder()
    graph_data = graph_builder.build_hierarchical_graph(level_data, ninja_pos)
    
    print(f"✅ Built graph with {graph_data.sub_cell_graph.num_nodes} nodes and {graph_data.sub_cell_graph.num_edges} edges")
    
    # Test Issue #1: Functional edges
    print("\n" + "=" * 60)
    print("=== ISSUE #1: FUNCTIONAL EDGES ===")
    print("=" * 60)
    
    edge_types = graph_data.sub_cell_graph.edge_types
    functional_edges = np.sum(edge_types == EdgeType.FUNCTIONAL.value)
    
    print(f"Functional edges found: {functional_edges}")
    
    if functional_edges > 0:
        print("✅ ISSUE #1 RESOLVED: Functional edges are present")
        
        # Show some functional edge details
        edge_index = graph_data.sub_cell_graph.edge_index
        node_features = graph_data.sub_cell_graph.node_features
        
        functional_count = 0
        for i in range(graph_data.sub_cell_graph.num_edges):
            if edge_types[i] == EdgeType.FUNCTIONAL.value:
                node1_id = edge_index[0, i]
                node2_id = edge_index[1, i]
                
                node1_pos = (float(node_features[node1_id, 0]), float(node_features[node1_id, 1]))
                node2_pos = (float(node_features[node2_id, 0]), float(node_features[node2_id, 1]))
                
                print(f"  Functional edge {functional_count + 1}: {node1_pos} -> {node2_pos}")
                functional_count += 1
                if functional_count >= 5:  # Show first 5
                    break
    else:
        print("❌ ISSUE #1 NOT RESOLVED: No functional edges found")
    
    # Test Issue #2: Walkable edges in solid tiles
    print("\n" + "=" * 60)
    print("=== ISSUE #2: WALKABLE EDGES IN SOLID TILES ===")
    print("=" * 60)
    
    edge_index = graph_data.sub_cell_graph.edge_index
    edge_types = graph_data.sub_cell_graph.edge_types
    node_features = graph_data.sub_cell_graph.node_features
    
    walkable_edges_in_solid = 0
    solid_tile_violations = []
    
    for i in range(graph_data.sub_cell_graph.num_edges):
        if edge_types[i] == EdgeType.WALK.value:
            node1_id = edge_index[0, i]
            node2_id = edge_index[1, i]
            
            # Get node positions
            node1_pos = (float(node_features[node1_id, 0]), float(node_features[node1_id, 1]))
            node2_pos = (float(node_features[node2_id, 0]), float(node_features[node2_id, 1]))
            
            # Check if either node is in a solid tile
            for pos in [node1_pos, node2_pos]:
                tile_x = int(pos[0] // TILE_PIXEL_SIZE)
                tile_y = int(pos[1] // TILE_PIXEL_SIZE)
                
                if (0 <= tile_y < level_data.height and 0 <= tile_x < level_data.width and
                    level_data.get_tile(tile_y, tile_x) == 1):
                    walkable_edges_in_solid += 1
                    solid_tile_violations.append({
                        'edge_id': i,
                        'node1_pos': node1_pos,
                        'node2_pos': node2_pos,
                        'tile_pos': (tile_x, tile_y),
                        'tile_value': level_data.get_tile(tile_y, tile_x)
                    })
                    break
    
    print(f"Walkable edges in solid tiles: {walkable_edges_in_solid}")
    
    if walkable_edges_in_solid == 0:
        print("✅ ISSUE #2 RESOLVED: Walkable edges in solid tiles have been filtered out")
    else:
        print("❌ ISSUE #2 NOT RESOLVED: Found walkable edges in solid tiles")
        print("First few violations:")
        for i, violation in enumerate(solid_tile_violations[:5]):
            print(f"  Violation {i+1}: Edge {violation['edge_id']} at tile {violation['tile_pos']} (value={violation['tile_value']})")
            print(f"    Node positions: {violation['node1_pos']} -> {violation['node2_pos']}")
    
    # Test Issue #3: Pathfinding
    print("\n" + "=" * 60)
    print("=== ISSUE #3: PATHFINDING ===")
    print("=" * 60)
    
    navigation_engine = PathfindingEngine()
    
    # Test navigation between nearby positions
    start_pos = ninja_pos
    # Choose an end position that should be reachable
    end_pos = (ninja_pos[0] + 50, ninja_pos[1])  # 50 pixels to the right
    
    print(f"Testing navigation from {start_pos} to {end_pos}")
    
    # Find nodes at these positions
    start_node = navigation_engine._find_node_at_position(graph_data.sub_cell_graph, start_pos)
    end_node = navigation_engine._find_node_at_position(graph_data.sub_cell_graph, end_pos)
    
    print(f"Start node: {start_node}, End node: {end_node}")
    
    if start_node is not None and end_node is not None:
        # Test A* navigation
        path_result = navigation_engine.find_shortest_path(graph_data.sub_cell_graph, start_node, end_node)
        
        print("A* navigation result:")
        print(f"  Success: {path_result.success}")
        print(f"  Path length: {len(path_result.path)} nodes")
        print(f"  Total cost: {path_result.total_cost:.2f}")
        print(f"  Nodes explored: {path_result.nodes_explored}")
        
        if path_result.success:
            print("✅ ISSUE #3 RESOLVED: Pathfinding is working correctly")
            
            # Show path details
            if len(path_result.path) > 0:
                print("Path nodes:")
                for i, node_id in enumerate(path_result.path[:5]):  # Show first 5 nodes
                    node_pos = (float(node_features[node_id, 0]), float(node_features[node_id, 1]))
                    print(f"  Node {i}: {node_id} at {node_pos}")
                if len(path_result.path) > 5:
                    print(f"  ... and {len(path_result.path) - 5} more nodes")
        else:
            print("❌ ISSUE #3 NOT RESOLVED: Pathfinding failed")
    else:
        print("❌ ISSUE #3 NOT RESOLVED: Could not find nodes at specified positions")
        print(f"  Start node search result: {start_node}")
        print(f"  End node search result: {end_node}")
    
    # Additional navigation tests with different positions
    print("\nTesting additional navigation scenarios...")
    
    # Test with positions that should definitely be reachable
    test_positions = [
        (ninja_pos[0] + 25, ninja_pos[1]),      # 25 pixels right
        (ninja_pos[0] - 25, ninja_pos[1]),      # 25 pixels left  
        (ninja_pos[0], ninja_pos[1] + 25),      # 25 pixels down
        (ninja_pos[0], ninja_pos[1] - 25),      # 25 pixels up
    ]
    
    successful_paths = 0
    for i, test_pos in enumerate(test_positions):
        test_node = navigation_engine._find_node_at_position(graph_data.sub_cell_graph, test_pos)
        if start_node is not None and test_node is not None:
            path_result = navigation_engine.find_shortest_path(graph_data.sub_cell_graph, start_node, test_node)
            if path_result.success:
                successful_paths += 1
                print(f"  Test {i+1}: ✅ Path found to {test_pos} ({len(path_result.path)} nodes, cost {path_result.total_cost:.2f})")
            else:
                print(f"  Test {i+1}: ❌ No path to {test_pos}")
        else:
            print(f"  Test {i+1}: ❌ Could not find nodes for {test_pos}")
    
    print(f"\nPathfinding success rate: {successful_paths}/{len(test_positions)} ({100*successful_paths/len(test_positions):.1f}%)")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY (DOORTEST MAP)")
    print("=" * 80)
    
    issue1_resolved = functional_edges > 0
    issue2_resolved = walkable_edges_in_solid == 0
    issue3_resolved = successful_paths > 0
    
    print(f"{'✅' if issue1_resolved else '❌'} RESOLVED: Issue #1: Functional edges between switches and doors")
    print(f"{'✅' if issue2_resolved else '❌'} RESOLVED: Issue #2: Walkable edges in solid tiles")
    print(f"{'✅' if issue3_resolved else '❌'} RESOLVED: Issue #3: Pathfinding not working on traversable paths")
    
    if issue1_resolved and issue2_resolved and issue3_resolved:
        print("\n🎉 ALL ISSUES RESOLVED! The graph visualization system is working correctly.")
    else:
        print(f"\n⚠️  {sum([issue1_resolved, issue2_resolved, issue3_resolved])}/3 issues resolved. More work needed.")
    
    print("=" * 80)
    
    return {
        'functional_edges': functional_edges,
        'walkable_edges_in_solid': walkable_edges_in_solid,
        'navigation_success_rate': successful_paths / len(test_positions),
        'all_resolved': issue1_resolved and issue2_resolved and issue3_resolved
    }


if __name__ == '__main__':
    try:
        results = test_doortest_map_issues()
        
        # Exit with appropriate code
        if results['all_resolved']:
            print("✅ All tests passed!")
            sys.exit(0)
        else:
            print("❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)