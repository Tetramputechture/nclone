#!/usr/bin/env python3
"""
Analyze the current 4-node path to understand why it's so direct.
"""

import sys
import os
import math

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine

def analyze_current_path():
    """Analyze the current pathfinding result in detail."""
    print("=" * 80)
    print("🔍 ANALYZING CURRENT 4-NODE PATH")
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
    graph_data = builder.build_graph(env.level_data, ninja_pos)
    
    # Get the 6px resolution graph (finest detail)
    graph = graph_data.sub_cell_graph
    print(f"✅ Graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # Find ninja and target nodes
    ninja_node = None
    target_node = None
    
    for node in graph.nodes:
        # Find ninja node (closest to ninja position)
        if ninja_node is None:
            ninja_node = node
        else:
            ninja_dist = math.sqrt((ninja_node.x - ninja_pos[0])**2 + (ninja_node.y - ninja_pos[1])**2)
            node_dist = math.sqrt((node.x - ninja_pos[0])**2 + (node.y - ninja_pos[1])**2)
            if node_dist < ninja_dist:
                ninja_node = node
        
        # Find leftmost switch node (entity type 4)
        if hasattr(node, 'entity_type') and node.entity_type == 4:  # Switch
            if target_node is None or node.x < target_node.x:
                target_node = node
    
    print(f"🥷 Ninja node: {ninja_node.id} at ({ninja_node.x:.1f}, {ninja_node.y:.1f})")
    print(f"🎯 Target node: {target_node.id} at ({target_node.x:.1f}, {target_node.y:.1f})")
    
    # Calculate direct distance
    direct_distance = math.sqrt(
        (target_node.x - ninja_node.x)**2 + (target_node.y - ninja_node.y)**2
    )
    print(f"📏 Direct distance: {direct_distance:.1f}px")
    
    # Try pathfinding
    print(f"\n🎯 PATHFINDING ANALYSIS:")
    pathfinder = PathfindingEngine()
    
    try:
        path = pathfinder.find_path(
            graph, ninja_node.id, target_node.id, algorithm='dijkstra'
        )
        
        if path:
            print(f"✅ Path found: {len(path.nodes)} nodes")
            print(f"   Path cost: {path.total_cost:.1f}px")
            
            # Analyze each step in detail
            print(f"\n📍 DETAILED PATH ANALYSIS:")
            for i, node_id in enumerate(path.nodes):
                node = next(n for n in graph.nodes if n.id == node_id)
                print(f"   Step {i}: Node {node_id} at ({node.x:.1f}, {node.y:.1f})")
                
                if hasattr(node, 'entity_type'):
                    print(f"           Entity type: {node.entity_type}")
                
                if i < len(path.edges):
                    edge = path.edges[i]
                    print(f"           -> {edge.type.name} edge (cost: {edge.cost:.1f})")
                    
                    # Calculate step distance
                    if i + 1 < len(path.nodes):
                        next_node_id = path.nodes[i + 1]
                        next_node = next(n for n in graph.nodes if n.id == next_node_id)
                        step_distance = math.sqrt(
                            (next_node.x - node.x)**2 + (next_node.y - node.y)**2
                        )
                        print(f"           Step distance: {step_distance:.1f}px")
            
            # Check if this path makes physical sense
            print(f"\n🔬 PHYSICS ANALYSIS:")
            total_path_distance = 0
            for i in range(len(path.nodes) - 1):
                node1 = next(n for n in graph.nodes if n.id == path.nodes[i])
                node2 = next(n for n in graph.nodes if n.id == path.nodes[i + 1])
                step_distance = math.sqrt(
                    (node2.x - node1.x)**2 + (node2.y - node1.y)**2
                )
                total_path_distance += step_distance
            
            print(f"   Total path distance: {total_path_distance:.1f}px")
            print(f"   Direct distance: {direct_distance:.1f}px")
            print(f"   Path efficiency: {direct_distance/total_path_distance:.2f}")
            
            # Check for impossible movements
            for i, edge in enumerate(path.edges):
                node1 = next(n for n in graph.nodes if n.id == path.nodes[i])
                node2 = next(n for n in graph.nodes if n.id == path.nodes[i + 1])
                
                dx = node2.x - node1.x
                dy = node2.y - node1.y
                distance = math.sqrt(dx*dx + dy*dy)
                
                print(f"   Edge {i}: {edge.type.name} - {distance:.1f}px")
                
                # Check if movement type makes sense
                if edge.type.name == "WALK" and distance > 50:
                    print(f"      ⚠️  Long WALK distance ({distance:.1f}px) - should be JUMP/FALL?")
                elif edge.type.name == "JUMP" and dy > 0:
                    print(f"      ⚠️  JUMP going downward ({dy:.1f}px) - should be FALL?")
                elif edge.type.name == "FALL" and dy < 0:
                    print(f"      ⚠️  FALL going upward ({dy:.1f}px) - should be JUMP?")
        
        else:
            print("❌ No path found!")
            
    except Exception as e:
        print(f"❌ Pathfinding failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_current_path()