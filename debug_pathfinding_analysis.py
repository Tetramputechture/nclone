#!/usr/bin/env python3
"""
Debug script to analyze current pathfinding issues and understand why movement diversity is low.
"""

import sys
import os
import math

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine
from nclone.graph.common import EdgeType

def analyze_pathfinding_issues():
    """Analyze current pathfinding issues and edge building problems."""
    print("=" * 80)
    print("🔍 PATHFINDING ANALYSIS - UNDERSTANDING MOVEMENT DIVERSITY ISSUES")
    print("=" * 80)
    
    # Load environment
    print("📁 Loading doortest environment...")
    env = BasicLevelNoGold(render_mode="rgb_array")
    env.reset()
    
    ninja_pos = env.nplay_headless.ninja_position()
    print(f"✅ Ninja position: {ninja_pos}")
    
    # Build graph
    print("🔧 Building graph...")
    builder = HierarchicalGraphBuilder()
    graph = builder.build_graph(env.level_data, ninja_pos)
    
    print(f"✅ Graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # Analyze edge types
    edge_type_counts = {}
    for edge in graph.edges:
        edge_type = edge.type
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
    
    print("\n📊 EDGE TYPE ANALYSIS:")
    for edge_type, count in edge_type_counts.items():
        print(f"   {edge_type.name}: {count} edges")
    
    # Find ninja node
    ninja_node = None
    min_dist = float('inf')
    for node in graph.nodes:
        dist = math.sqrt((node.x - ninja_pos[0])**2 + (node.y - ninja_pos[1])**2)
        if dist < min_dist:
            min_dist = dist
            ninja_node = node
    
    print(f"\n🥷 Ninja node: {ninja_node.id} at ({ninja_node.x:.1f}, {ninja_node.y:.1f})")
    
    # Find switch nodes (leftmost switch)
    switch_nodes = []
    for node in graph.nodes:
        if hasattr(node, 'entity_type') and node.entity_type == 4:  # Switch
            switch_nodes.append(node)
    
    if switch_nodes:
        # Find leftmost switch
        leftmost_switch = min(switch_nodes, key=lambda n: n.x)
        print(f"🔘 Leftmost switch: {leftmost_switch.id} at ({leftmost_switch.x:.1f}, {leftmost_switch.y:.1f})")
        
        # Analyze edges from ninja node
        ninja_edges = [edge for edge in graph.edges if edge.source == ninja_node.id]
        print(f"\n🔗 Ninja outgoing edges: {len(ninja_edges)}")
        
        ninja_edge_types = {}
        for edge in ninja_edges:
            edge_type = edge.type
            ninja_edge_types[edge_type] = ninja_edge_types.get(edge_type, 0) + 1
        
        for edge_type, count in ninja_edge_types.items():
            print(f"   {edge_type.name}: {count} edges")
        
        # Try pathfinding
        print(f"\n🎯 PATHFINDING ANALYSIS:")
        pathfinder = PathfindingEngine()
        
        try:
            path = pathfinder.find_path(
                graph, ninja_node.id, leftmost_switch.id, algorithm='dijkstra'
            )
            
            if path:
                print(f"✅ Path found: {len(path.nodes)} nodes")
                print(f"   Path cost: {path.total_cost:.1f}px")
                
                # Analyze movement types in path
                movement_types = {}
                for i in range(len(path.edges)):
                    edge = path.edges[i]
                    movement_type = edge.type
                    movement_types[movement_type] = movement_types.get(movement_type, 0) + 1
                
                print(f"   Movement types: {movement_types}")
                
                # Show path details
                print(f"\n📍 PATH DETAILS:")
                for i, node_id in enumerate(path.nodes):
                    node = next(n for n in graph.nodes if n.id == node_id)
                    print(f"   {i}: Node {node_id} at ({node.x:.1f}, {node.y:.1f})")
                    if i < len(path.edges):
                        edge = path.edges[i]
                        print(f"      -> {edge.type.name} edge (cost: {edge.cost:.1f})")
                
            else:
                print("❌ No path found!")
                
        except Exception as e:
            print(f"❌ Pathfinding failed: {e}")
    
    # Analyze potential jump/fall opportunities
    print(f"\n🦘 JUMP/FALL OPPORTUNITY ANALYSIS:")
    
    # Look for nodes at different heights that could be connected
    nodes_by_height = {}
    for node in graph.nodes:
        height_bucket = int(node.y // 24)  # Group by 24px height buckets
        if height_bucket not in nodes_by_height:
            nodes_by_height[height_bucket] = []
        nodes_by_height[height_bucket].append(node)
    
    print(f"   Height levels: {len(nodes_by_height)}")
    for height, nodes in sorted(nodes_by_height.items()):
        print(f"   Level {height} (y={height*24}-{(height+1)*24}): {len(nodes)} nodes")
    
    # Check for missing jump/fall connections
    ninja_height = int(ninja_node.y // 24)
    print(f"\n   Ninja at height level {ninja_height}")
    
    # Look for nodes at different heights within jump/fall range
    potential_targets = []
    for node in graph.nodes:
        if node.id == ninja_node.id:
            continue
            
        dx = node.x - ninja_node.x
        dy = node.y - ninja_node.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check if this could be a jump or fall target
        if distance <= 150 and abs(dy) > 24:  # Within jump range and significant height diff
            movement_type = "JUMP" if dy < 0 else "FALL"
            potential_targets.append((node, distance, movement_type))
    
    potential_targets.sort(key=lambda x: x[1])  # Sort by distance
    
    print(f"   Potential jump/fall targets: {len(potential_targets)}")
    for i, (node, distance, movement_type) in enumerate(potential_targets[:5]):
        print(f"   {i+1}: {movement_type} to node {node.id} at ({node.x:.1f}, {node.y:.1f}) - {distance:.1f}px")
        
        # Check if edge exists
        edge_exists = any(
            edge.source == ninja_node.id and edge.target == node.id 
            for edge in graph.edges
        )
        print(f"      Edge exists: {edge_exists}")

if __name__ == "__main__":
    analyze_pathfinding_issues()