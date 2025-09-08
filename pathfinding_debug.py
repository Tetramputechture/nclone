#!/usr/bin/env python3
"""
Debug pathfinding issues
"""

import os
import sys
import numpy as np

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm

def debug_pathfinding():
    print("=" * 60)
    print("PATHFINDING DEBUG")
    print("=" * 60)
    
    # Create environment
    env = BasicLevelNoGold(render_mode="rgb_array", enable_frame_stack=False, enable_debug_overlay=False, eval_mode=False, seed=42)
    env.reset()
    ninja_position = env.nplay_headless.ninja_position()
    level_data = env.level_data
    
    # Build graph
    builder = HierarchicalGraphBuilder()
    hierarchical_graph = builder.build_graph(level_data, ninja_position)
    graph = hierarchical_graph.sub_cell_graph
    
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Find two nearby nodes
    valid_nodes = []
    for i in range(graph.num_nodes):
        if graph.node_mask[i] == 1:
            node_x = graph.node_features[i, 0]
            node_y = graph.node_features[i, 1]
            valid_nodes.append((i, node_x, node_y))
    
    print(f"Found {len(valid_nodes)} valid nodes")
    
    if len(valid_nodes) < 2:
        print("❌ Not enough valid nodes")
        return False
    
    # Try pathfinding between first two nodes
    start_node = valid_nodes[0][0]
    goal_node = valid_nodes[1][0]
    
    start_pos = (valid_nodes[0][1], valid_nodes[0][2])
    goal_pos = (valid_nodes[1][1], valid_nodes[1][2])
    
    print(f"Testing path from node {start_node} at {start_pos} to node {goal_node} at {goal_pos}")
    
    # Check if nodes have edges
    start_edges = 0
    goal_edges = 0
    for i in range(graph.num_edges):
        if graph.edge_mask[i] == 1:
            src = graph.edge_index[0, i]
            dst = graph.edge_index[1, i]
            if src == start_node or dst == start_node:
                start_edges += 1
            if src == goal_node or dst == goal_node:
                goal_edges += 1
    
    print(f"Start node {start_node} has {start_edges} edges")
    print(f"Goal node {goal_node} has {goal_edges} edges")
    
    if start_edges == 0:
        print("❌ Start node has no edges - isolated!")
        return False
    if goal_edges == 0:
        print("❌ Goal node has no edges - isolated!")
        return False
    
    # Try pathfinding
    try:
        pathfinding_engine = PathfindingEngine()
        path_result = pathfinding_engine.find_shortest_path(
            graph, start_node, goal_node, PathfindingAlgorithm.A_STAR
        )
        
        if path_result.success:
            print(f"✅ Pathfinding SUCCESS: {len(path_result.path)} nodes, cost {path_result.total_cost:.2f}")
            return True
        else:
            print(f"❌ Pathfinding FAILED")
            
            # Try with Dijkstra
            print("🔄 Trying with Dijkstra...")
            path_result2 = pathfinding_engine.find_shortest_path(
                graph, start_node, goal_node, PathfindingAlgorithm.DIJKSTRA
            )
            
            if path_result2.success:
                print(f"✅ Dijkstra SUCCESS: {len(path_result2.path)} nodes, cost {path_result2.total_cost:.2f}")
                return True
            else:
                print(f"❌ Dijkstra also FAILED")
                return False
                
    except Exception as e:
        print(f"❌ Pathfinding exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_pathfinding()
    if success:
        print("\n🎉 PATHFINDING DEBUG SUCCESSFUL!")
    else:
        print("\n❌ PATHFINDING DEBUG FAILED!")
    
    sys.exit(0 if success else 1)