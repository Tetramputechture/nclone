#!/usr/bin/env python3
"""
Debug pathfinding connectivity issues
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
    print("PATHFINDING CONNECTIVITY DEBUG")
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
    
    # Find ninja node (use closest match)
    ninja_candidates = []
    for node_idx in range(graph.num_nodes):
        if graph.node_mask[node_idx] == 1:  # Valid node
            node_x = graph.node_features[node_idx, 0]
            node_y = graph.node_features[node_idx, 1]
            dist = ((node_x - ninja_position[0])**2 + (node_y - ninja_position[1])**2)**0.5
            if dist < 10:  # Close enough
                ninja_candidates.append((node_idx, node_x, node_y, dist))
    
    ninja_node = None
    if ninja_candidates:
        # Use the closest one
        ninja_candidates.sort(key=lambda x: x[3])  # Sort by distance
        ninja_node = ninja_candidates[0][0]
    
    if ninja_node is None:
        print("❌ Could not find ninja node")
        return False
    
    print(f"✅ Found ninja node: {ninja_node}")
    
    # Check ninja node connectivity
    ninja_edges = 0
    connected_nodes = []
    for i in range(graph.num_edges):
        if graph.edge_mask[i] == 1:
            src = graph.edge_index[0, i]
            dst = graph.edge_index[1, i]
            if src == ninja_node:
                ninja_edges += 1
                connected_nodes.append(dst)
            elif dst == ninja_node:
                ninja_edges += 1
                connected_nodes.append(src)
    
    print(f"Ninja node has {ninja_edges} edges, connected to {len(set(connected_nodes))} unique nodes")
    
    if ninja_edges == 0:
        print("❌ Ninja node is isolated - no edges!")
        return False
    
    # Try pathfinding to a nearby connected node
    if connected_nodes:
        target_node = connected_nodes[0]  # Pick first connected node
        target_x = graph.node_features[target_node, 0]
        target_y = graph.node_features[target_node, 1]
        
        print(f"Testing path to connected node {target_node} at ({target_x}, {target_y})")
        
        try:
            pathfinding_engine = PathfindingEngine()
            path_result = pathfinding_engine.find_shortest_path(
                graph, ninja_node, target_node, PathfindingAlgorithm.A_STAR
            )
            
            if path_result.success:
                print(f"✅ SUCCESS: Path to connected node - {len(path_result.path)} nodes, cost {path_result.total_cost:.2f}")
            else:
                print(f"❌ FAILED: Path to connected node failed")
                return False
                
        except Exception as e:
            print(f"❌ Exception during pathfinding: {e}")
            return False
    
    # Now try pathfinding to the leftmost locked door switch
    locked_door_switches = []
    for entity in level_data.entities:
        if entity.get("type", 0) == 6:  # LOCKED_DOOR
            entity_x, entity_y = entity.get("x", 0), entity.get("y", 0)
            locked_door_switches.append((entity_x, entity_y))
    
    if not locked_door_switches:
        print("❌ No locked door switches found")
        return False
    
    leftmost_switch = min(locked_door_switches, key=lambda pos: pos[0])
    target_x, target_y = leftmost_switch
    
    # Find closest node to leftmost switch
    target_node = None
    closest_dist = float('inf')
    for node_idx in range(graph.num_nodes):
        if graph.node_mask[node_idx] == 1:
            node_x = graph.node_features[node_idx, 0]
            node_y = graph.node_features[node_idx, 1]
            dist = ((node_x - target_x)**2 + (node_y - target_y)**2)**0.5
            if dist < closest_dist:
                closest_dist = dist
                target_node = node_idx
    
    print(f"Leftmost switch at ({target_x}, {target_y}), closest node {target_node} at distance {closest_dist:.1f}")
    
    # Check target node connectivity
    target_edges = 0
    for i in range(graph.num_edges):
        if graph.edge_mask[i] == 1:
            src = graph.edge_index[0, i]
            dst = graph.edge_index[1, i]
            if src == target_node or dst == target_node:
                target_edges += 1
    
    print(f"Target node has {target_edges} edges")
    
    if target_edges == 0:
        print("❌ Target node is isolated - no edges!")
        return False
    
    # Try pathfinding to target
    try:
        pathfinding_engine = PathfindingEngine()
        path_result = pathfinding_engine.find_shortest_path(
            graph, ninja_node, target_node, PathfindingAlgorithm.A_STAR
        )
        
        if path_result.success:
            print(f"✅ SUCCESS: Path to leftmost switch - {len(path_result.path)} nodes, cost {path_result.total_cost:.2f}")
            return True
        else:
            print(f"❌ FAILED: Path to leftmost switch failed")
            
            # Try with Dijkstra
            print("🔄 Trying with Dijkstra...")
            path_result2 = pathfinding_engine.find_shortest_path(
                graph, ninja_node, target_node, PathfindingAlgorithm.DIJKSTRA
            )
            
            if path_result2.success:
                print(f"✅ Dijkstra SUCCESS: {len(path_result2.path)} nodes, cost {path_result2.total_cost:.2f}")
                return True
            else:
                print(f"❌ Dijkstra also FAILED")
                return False
                
    except Exception as e:
        print(f"❌ Exception during pathfinding: {e}")
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