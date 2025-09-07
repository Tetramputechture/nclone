#!/usr/bin/env python3
"""
Test the new corridor connections system for long-distance pathfinding.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine


def test_corridor_connections():
    """Test the corridor connections system."""
    print("=" * 80)
    print("TESTING CORRIDOR CONNECTIONS FOR LONG-DISTANCE PATHFINDING")
    print("=" * 80)
    
    # Create environment
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42
    )
    
    # Reset to load the map
    env.reset()
    
    # Get level data and ninja position
    level_data = env.level_data
    ninja_pos = env.nplay_headless.ninja_position()
    
    print(f"Map: {level_data.width}x{level_data.height} tiles")
    print(f"Ninja: {ninja_pos}")
    
    # Build graph with new corridor connections
    print(f"\nBuilding graph with corridor connections...")
    graph_builder = HierarchicalGraphBuilder()
    hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
    graph_data = hierarchical_data.sub_cell_graph
    pathfinding_engine = PathfindingEngine()
    
    print(f"Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Find ninja node
    ninja_node = pathfinding_engine._find_node_at_position(graph_data, ninja_pos)
    print(f"Ninja node: {ninja_node}")
    
    if ninja_node is None:
        print("❌ Ninja node not found!")
        return
    
    # Test long-distance pathfinding to various targets
    print(f"\n" + "=" * 60)
    print("TESTING LONG-DISTANCE PATHFINDING")
    print("=" * 60)
    
    # Test targets in different empty tile clusters
    distant_targets = [
        (468, 216),  # Cluster 1 center (largest cluster)
        (636, 216),  # Cluster 2 center
        (240, 288),  # Cluster 3 center
        (324, 276),  # Cluster 4 center
        (168, 324),  # Cluster 5 center
        (192, 396),  # Cluster 6 center
        (120, 420),  # Cluster 7 center
        (156, 252),  # Cluster 8 center
        (204, 348),  # Cluster 9 center
    ]
    
    successful_paths = 0
    total_tests = len(distant_targets)
    
    print(f"Testing pathfinding to {total_tests} distant cluster centers:")
    
    for i, target_pos in enumerate(distant_targets, 1):
        target_x, target_y = target_pos
        target_node = pathfinding_engine._find_node_at_position(graph_data, target_pos)
        
        if target_node is None:
            print(f"  {i:2d}. {target_pos} -> ❌ Target node not found")
            continue
        
        # Calculate distance
        distance = ((ninja_pos[0] - target_x)**2 + (ninja_pos[1] - target_y)**2)**0.5
        
        # Attempt pathfinding
        path_result = pathfinding_engine.find_shortest_path(graph_data, ninja_node, target_node)
        
        if path_result and path_result.success and len(path_result.path) > 0:
            print(f"  {i:2d}. {target_pos} (dist: {distance:5.1f}) -> ✅ Path found! ({len(path_result.path)} nodes, cost: {path_result.total_cost:.1f})")
            successful_paths += 1
        else:
            print(f"  {i:2d}. {target_pos} (dist: {distance:5.1f}) -> ❌ No path found")
    
    success_rate = (successful_paths / total_tests) * 100
    print(f"\nLong-distance pathfinding success rate: {successful_paths}/{total_tests} ({success_rate:.1f}%)")
    
    # Analyze ninja's connectivity
    print(f"\n" + "=" * 60)
    print("ANALYZING NINJA'S CONNECTIVITY")
    print("=" * 60)
    
    # Find ninja's connected component
    visited = set()
    stack = [ninja_node]
    ninja_component = []
    
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        
        visited.add(current)
        ninja_component.append(current)
        
        # Find neighbors
        for edge_idx in range(graph_data.num_edges):
            if graph_data.edge_mask[edge_idx] == 0:
                continue
            
            src = int(graph_data.edge_index[0, edge_idx])
            dst = int(graph_data.edge_index[1, edge_idx])
            
            if src == current and dst not in visited:
                stack.append(dst)
            elif dst == current and src not in visited:
                stack.append(src)
    
    print(f"Ninja's connected component: {len(ninja_component)} nodes")
    
    # Calculate improvement
    original_component_size = 44  # From previous tests
    improvement = len(ninja_component) - original_component_size
    improvement_percent = (improvement / original_component_size) * 100 if original_component_size > 0 else 0
    
    print(f"Improvement: +{improvement} nodes ({improvement_percent:+.1f}%)")
    
    # Test specific empty tile areas for reachability
    print(f"\n" + "=" * 60)
    print("TESTING REACHABILITY TO SPECIFIC EMPTY AREAS")
    print("=" * 60)
    
    # Test positions in each cluster
    cluster_tests = [
        ("Cluster 1 (largest)", [(492, 180), (516, 180), (540, 180)]),
        ("Cluster 2", [(612, 180), (636, 180), (660, 180)]),
        ("Cluster 3", [(228, 276), (252, 276), (240, 288)]),
        ("Cluster 4", [(300, 276), (324, 276), (348, 276)]),
        ("Cluster 5", [(156, 324), (180, 324)]),
    ]
    
    for cluster_name, positions in cluster_tests:
        print(f"\n{cluster_name}:")
        cluster_reachable = 0
        
        for pos in positions:
            target_node = pathfinding_engine._find_node_at_position(graph_data, pos)
            if target_node is None:
                print(f"  {pos} -> ❌ No node")
                continue
            
            path_result = pathfinding_engine.find_shortest_path(graph_data, ninja_node, target_node)
            if path_result and path_result.success:
                distance = ((ninja_pos[0] - pos[0])**2 + (ninja_pos[1] - pos[1])**2)**0.5
                print(f"  {pos} -> ✅ Reachable (dist: {distance:.1f}, cost: {path_result.total_cost:.1f})")
                cluster_reachable += 1
            else:
                print(f"  {pos} -> ❌ Not reachable")
        
        cluster_rate = (cluster_reachable / len(positions)) * 100 if positions else 0
        print(f"  Cluster reachability: {cluster_reachable}/{len(positions)} ({cluster_rate:.1f}%)")
    
    # Final assessment
    print(f"\n" + "=" * 60)
    print("CORRIDOR CONNECTIONS ASSESSMENT")
    print("=" * 60)
    
    if success_rate >= 70:
        print("🎉 EXCELLENT: Corridor connections dramatically improved long-distance pathfinding!")
        print("   ✅ Issue #3 is now fully resolved!")
    elif success_rate >= 40:
        print("✅ GOOD: Corridor connections significantly improved pathfinding!")
        print("   🔧 Some additional improvements possible")
    elif success_rate >= 20:
        print("⚠️  MODERATE: Some improvement but more work needed")
        print("   🔧 Consider expanding corridor detection or reducing restrictions")
    else:
        print("❌ LIMITED: Corridor connections didn't provide significant improvement")
        print("   🔧 Need to investigate alternative approaches")
    
    if len(ninja_component) > original_component_size * 2:
        print(f"🚀 CONNECTIVITY BOOST: Ninja's reachable area increased by {improvement_percent:+.1f}%!")
    
    return {
        'success_rate': success_rate,
        'successful_paths': successful_paths,
        'total_tests': total_tests,
        'ninja_component_size': len(ninja_component),
        'connectivity_improvement': improvement_percent
    }


if __name__ == '__main__':
    results = test_corridor_connections()
    print(f"\nCorridor connections test completed. Results: {results}")