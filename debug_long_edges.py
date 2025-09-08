#!/usr/bin/env python3
"""
Debug script to find unrealistically long edges.
"""

import sys
import os
import math

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder

def debug_long_edges():
    """Find and analyze unrealistically long edges."""
    print("=" * 60)
    print("🔍 LONG EDGE ANALYSIS")
    print("=" * 60)
    
    # Load environment
    env = BasicLevelNoGold(render_mode="rgb_array")
    env.reset()
    
    ninja_pos = env.nplay_headless.ninja_position()
    print(f"✅ Ninja position: {ninja_pos}")
    
    # Build graph
    print("🔧 Building graph...")
    builder = HierarchicalGraphBuilder()
    graph_data = builder.build_graph(env.level_data, ninja_pos)
    
    # Get the 6px resolution graph
    graph = graph_data.sub_cell_graph
    print(f"✅ Graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # Create node position lookup
    node_positions = {}
    for node in graph.nodes:
        node_positions[node.id] = (node.x, node.y)
    
    # Find long edges
    long_edges = []
    
    for edge in graph.edges:
        src_pos = node_positions[edge.source]
        dst_pos = node_positions[edge.target]
        
        distance = math.sqrt(
            (dst_pos[0] - src_pos[0])**2 + (dst_pos[1] - src_pos[1])**2
        )
        
        # Flag edges that are unrealistically long for their type
        is_long = False
        if edge.type.name == 'WALK' and distance > 50:
            is_long = True
        elif edge.type.name == 'JUMP' and distance > 150:
            is_long = True
        elif edge.type.name == 'FALL' and distance > 300:
            is_long = True
        
        if is_long:
            long_edges.append((edge, distance, src_pos, dst_pos))
    
    # Sort by distance
    long_edges.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n⚠️  FOUND {len(long_edges)} UNREALISTICALLY LONG EDGES:")
    
    for i, (edge, distance, src_pos, dst_pos) in enumerate(long_edges[:10]):  # Show top 10
        print(f"   {i+1}. {edge.type.name} edge: {distance:.1f}px")
        print(f"      From: ({src_pos[0]:.1f}, {src_pos[1]:.1f})")
        print(f"      To: ({dst_pos[0]:.1f}, {dst_pos[1]:.1f})")
        print(f"      Nodes: {edge.source} -> {edge.target}")
        
        # Check if this involves the ninja
        ninja_distance_src = math.sqrt((src_pos[0] - ninja_pos[0])**2 + (src_pos[1] - ninja_pos[1])**2)
        ninja_distance_dst = math.sqrt((dst_pos[0] - ninja_pos[0])**2 + (dst_pos[1] - ninja_pos[1])**2)
        
        if ninja_distance_src < 20 or ninja_distance_dst < 20:
            print(f"      🥷 INVOLVES NINJA!")
        print()

if __name__ == "__main__":
    debug_long_edges()