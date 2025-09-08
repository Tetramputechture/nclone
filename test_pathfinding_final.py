#!/usr/bin/env python3
"""
Final pathfinding test with improved traversability
"""

import os
import sys
import numpy as np

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding_engine import PathfindingEngine
from nclone.constants.entity_types import EntityType

def test_pathfinding():
    """Test pathfinding from ninja to leftmost locked door switch"""
    print("=" * 60)
    print("FINAL PATHFINDING TEST")
    print("=" * 60)
    
    # Create environment
    env = BasicLevelNoGold(render_mode="rgb_array", enable_frame_stack=False, enable_debug_overlay=False, eval_mode=False, seed=42)
    env.reset()
    ninja_position = env.nplay_headless.ninja_position()
    level_data = env.level_data
    
    print(f"Ninja position: {ninja_position}")
    print(f"Level size: {level_data.width}x{level_data.height} tiles")
    
    # Find leftmost locked door switch
    leftmost_switch = None
    leftmost_x = float('inf')
    
    for entity in level_data.entities:
        if entity.get("type") == EntityType.LOCKED_DOOR:
            switch_x = entity.get("x", 0)
            if switch_x < leftmost_x:
                leftmost_x = switch_x
                leftmost_switch = entity
    
    if not leftmost_switch:
        print("❌ No locked door switch found!")
        return
    
    target_position = (leftmost_switch.get("x", 0), leftmost_switch.get("y", 0))
    print(f"Target switch position: {target_position}")
    
    # Build graph
    print(f"\n🔧 Building hierarchical graph...")
    builder = HierarchicalGraphBuilder()
    hierarchical_graph = builder.build_graph(level_data, ninja_position)
    
    print(f"Graph stats:")
    print(f"  Nodes: {hierarchical_graph.num_nodes}")
    print(f"  Edges: {hierarchical_graph.num_edges}")
    
    # Test pathfinding
    print(f"\n🎯 Testing pathfinding...")
    pathfinding_engine = PathfindingEngine()
    
    try:
        path = pathfinding_engine.find_path(
            hierarchical_graph, 
            ninja_position, 
            target_position
        )
        
        if path and len(path) > 0:
            print(f"✅ SUCCESS! Found path with {len(path)} nodes")
            print(f"Path cost: {path[-1].cost:.2f}")
            
            # Show first few and last few path nodes
            print(f"\nPath preview:")
            for i, node in enumerate(path[:3]):
                print(f"  {i+1}: ({node.x:.1f}, {node.y:.1f}) cost={node.cost:.2f}")
            
            if len(path) > 6:
                print(f"  ... ({len(path)-6} intermediate nodes)")
                
            for i, node in enumerate(path[-3:]):
                idx = len(path) - 3 + i
                print(f"  {idx+1}: ({node.x:.1f}, {node.y:.1f}) cost={node.cost:.2f}")
                
        else:
            print(f"❌ FAILED! No path found")
            
    except Exception as e:
        print(f"❌ ERROR during pathfinding: {e}")
        import traceback
        traceback.print_exc()
    
    # Test connectivity analysis
    print(f"\n🔍 Connectivity analysis...")
    
    # Check if ninja and target are in same connected component
    from collections import deque
    
    # Find ninja node
    ninja_node = None
    target_node = None
    min_ninja_dist = float('inf')
    min_target_dist = float('inf')
    
    for node_id in range(hierarchical_graph.num_nodes):
        node_x = hierarchical_graph.node_features[node_id][0]
        node_y = hierarchical_graph.node_features[node_id][1]
        
        ninja_dist = ((node_x - ninja_position[0])**2 + (node_y - ninja_position[1])**2)**0.5
        target_dist = ((node_x - target_position[0])**2 + (node_y - target_position[1])**2)**0.5
        
        if ninja_dist < min_ninja_dist:
            min_ninja_dist = ninja_dist
            ninja_node = node_id
            
        if target_dist < min_target_dist:
            min_target_dist = target_dist
            target_node = node_id
    
    print(f"Closest ninja node: {ninja_node} (distance: {min_ninja_dist:.1f}px)")
    print(f"Closest target node: {target_node} (distance: {min_target_dist:.1f}px)")
    
    if ninja_node is not None and target_node is not None:
        # BFS to check connectivity
        visited = set()
        queue = deque([ninja_node])
        visited.add(ninja_node)
        
        while queue:
            current = queue.popleft()
            
            if current == target_node:
                print(f"✅ Ninja and target are in the same connected component!")
                break
                
            # Get neighbors
            for edge_idx in range(hierarchical_graph.num_edges):
                src = hierarchical_graph.edge_index[0][edge_idx]
                dst = hierarchical_graph.edge_index[1][edge_idx]
                
                if src == current and dst not in visited:
                    visited.add(dst)
                    queue.append(dst)
                elif dst == current and src not in visited:
                    visited.add(src)
                    queue.append(src)
        else:
            print(f"❌ Ninja and target are in DIFFERENT connected components!")
            print(f"Ninja component size: {len(visited)}")
    
    print(f"\n🎉 PATHFINDING TEST COMPLETE!")

if __name__ == "__main__":
    test_pathfinding()