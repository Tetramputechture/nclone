#!/usr/bin/env python3
"""
Debug corridor connections to understand why pathfinding is failing.
"""

import sys
import os

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.common import EdgeType

def debug_corridor_connections():
    """Debug corridor connections in doortest map."""
    print("=" * 80)
    print("🔍 DEBUGGING CORRIDOR CONNECTIONS")
    print("=" * 80)
    
    # Load doortest map
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42
    )
    
    env.reset()
    level_data = env.level_data
    ninja_position = env.nplay_headless.ninja_position()
    
    print(f"🗺️  Map: {level_data.width}x{level_data.height} tiles")
    print(f"🥷 Ninja: {ninja_position}")
    
    # Build graph
    builder = HierarchicalGraphBuilder()
    hierarchical_graph = builder.build_graph(level_data, ninja_position)
    graph = hierarchical_graph.sub_cell_graph
    
    print(f"📊 Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Analyze edge types
    edge_counts = {}
    corridor_edges = []
    
    for edge_idx in range(graph.num_edges):
        if graph.edge_mask[edge_idx] == 1:
            # Find edge type
            edge_type = None
            for et in EdgeType:
                if graph.edge_features[edge_idx, et] > 0.5:
                    edge_type = et
                    break
            
            if edge_type:
                edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1
                
                # Check if this is a corridor edge (long distance)
                src_node = graph.edge_index[0, edge_idx]
                dst_node = graph.edge_index[1, edge_idx]
                
                src_x = graph.node_features[src_node, 0]
                src_y = graph.node_features[src_node, 1]
                dst_x = graph.node_features[dst_node, 0]
                dst_y = graph.node_features[dst_node, 1]
                
                distance = ((dst_x - src_x)**2 + (dst_y - src_y)**2)**0.5
                
                if distance > 50:  # Corridor threshold
                    corridor_edges.append({
                        'type': edge_type,
                        'distance': distance,
                        'src': (src_x, src_y),
                        'dst': (dst_x, dst_y)
                    })
    
    print(f"\n📈 Edge Type Counts:")
    for edge_type, count in edge_counts.items():
        print(f"   {edge_type.name}: {count}")
    
    print(f"\n🛤️  Corridor Edges (>50px): {len(corridor_edges)}")
    for i, edge in enumerate(corridor_edges[:10]):  # Show first 10
        print(f"   {i+1}: {edge['type'].name} {edge['distance']:.1f}px {edge['src']} → {edge['dst']}")
    
    # Check connectivity from ninja
    ninja_node = None
    min_dist = float('inf')
    
    for node_idx in range(graph.num_nodes):
        if graph.node_mask[node_idx] == 1:
            node_x = graph.node_features[node_idx, 0]
            node_y = graph.node_features[node_idx, 1]
            dist = ((node_x - ninja_position[0])**2 + (node_y - ninja_position[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                ninja_node = node_idx
    
    if ninja_node is not None:
        # BFS from ninja node
        visited = set()
        queue = [ninja_node]
        visited.add(ninja_node)
        reachable_positions = []
        
        while queue:
            current = queue.pop(0)
            node_x = graph.node_features[current, 0]
            node_y = graph.node_features[current, 1]
            reachable_positions.append((node_x, node_y))
            
            # Find edges from current node
            for edge_idx in range(graph.num_edges):
                if graph.edge_mask[edge_idx] == 1:
                    src = graph.edge_index[0, edge_idx]
                    dst = graph.edge_index[1, edge_idx]
                    
                    if src == current and dst not in visited:
                        visited.add(dst)
                        queue.append(dst)
        
        print(f"\n🎯 Ninja Connectivity:")
        print(f"   Ninja node: {ninja_node} at {graph.node_features[ninja_node, 0]:.1f}, {graph.node_features[ninja_node, 1]:.1f}")
        print(f"   Reachable nodes: {len(reachable_positions)}")
        
        # Show some reachable positions
        print(f"   Sample positions:")
        for i, pos in enumerate(reachable_positions[:10]):
            print(f"     {i+1}: ({pos[0]:.1f}, {pos[1]:.1f})")
        
        # Check if we can reach the target area
        target_area_nodes = 0
        for pos in reachable_positions:
            if pos[0] > 300:  # Right side of map where target is
                target_area_nodes += 1
        
        print(f"   Nodes in target area (x>300): {target_area_nodes}")
        
        if target_area_nodes == 0:
            print("   ❌ Cannot reach target area - graph fragmentation confirmed")
        else:
            print("   ✅ Can reach target area")

if __name__ == "__main__":
    debug_corridor_connections()