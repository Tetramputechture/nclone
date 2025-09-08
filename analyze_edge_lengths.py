#!/usr/bin/env python3
"""
Analyze edge lengths to find unrealistic long-distance connections.
"""

import sys
import os
import math

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder

def analyze_edge_lengths():
    """Analyze edge lengths to find unrealistic connections."""
    print("=" * 60)
    print("🔍 EDGE LENGTH ANALYSIS")
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
    
    # Get the 6px resolution graph (finest detail)
    graph = graph_data.sub_cell_graph
    print(f"✅ Graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # Create node position lookup
    node_positions = {}
    for node in graph.nodes:
        node_positions[node.id] = (node.x, node.y)
    
    # Analyze edge lengths by type
    edge_lengths = {'WALK': [], 'JUMP': [], 'FALL': [], 'FUNCTIONAL': [], 'OTHER': []}
    
    for edge in graph.edges:
        src_pos = node_positions[edge.source]
        dst_pos = node_positions[edge.target]
        
        distance = math.sqrt(
            (dst_pos[0] - src_pos[0])**2 + (dst_pos[1] - src_pos[1])**2
        )
        
        edge_type = edge.type.name
        if edge_type in edge_lengths:
            edge_lengths[edge_type].append(distance)
        else:
            edge_lengths['OTHER'].append(distance)
    
    # Report statistics
    print(f"\n📊 EDGE LENGTH STATISTICS:")
    for edge_type, lengths in edge_lengths.items():
        if lengths:
            avg_length = sum(lengths) / len(lengths)
            max_length = max(lengths)
            min_length = min(lengths)
            print(f"   {edge_type}: {len(lengths)} edges")
            print(f"      Average: {avg_length:.1f}px")
            print(f"      Range: {min_length:.1f}px - {max_length:.1f}px")
            
            # Flag unrealistic edges
            if edge_type == 'WALK' and max_length > 50:
                print(f"      ⚠️  Long WALK edges detected!")
                long_walks = [l for l in lengths if l > 50]
                print(f"      {len(long_walks)} WALK edges > 50px")
            elif edge_type == 'JUMP' and max_length > 150:
                print(f"      ⚠️  Very long JUMP edges detected!")
            elif edge_type == 'FALL' and max_length > 300:
                print(f"      ⚠️  Very long FALL edges detected!")
    
    # Find the longest edges of each type
    print(f"\n🔍 LONGEST EDGES BY TYPE:")
    for edge_type, lengths in edge_lengths.items():
        if lengths:
            max_length = max(lengths)
            # Find the edge with this length
            for edge in graph.edges:
                src_pos = node_positions[edge.source]
                dst_pos = node_positions[edge.target]
                distance = math.sqrt(
                    (dst_pos[0] - src_pos[0])**2 + (dst_pos[1] - src_pos[1])**2
                )
                
                if edge.type.name == edge_type and abs(distance - max_length) < 0.1:
                    print(f"   {edge_type}: {distance:.1f}px")
                    print(f"      From: ({src_pos[0]:.1f}, {src_pos[1]:.1f})")
                    print(f"      To: ({dst_pos[0]:.1f}, {dst_pos[1]:.1f})")
                    break

if __name__ == "__main__":
    analyze_edge_lengths()