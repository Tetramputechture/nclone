#!/usr/bin/env python3
"""
Test complex pathfinding with multiple movement types matching the user's expected path.
"""

import sys
import os
import math

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm

def test_complex_pathfinding():
    """Test pathfinding with focus on movement type diversity."""
    print("=" * 70)
    print("🎯 COMPLEX PATHFINDING TEST")
    print("=" * 70)
    
    # Load environment
    env = BasicLevelNoGold(render_mode="rgb_array")
    env.reset()
    
    ninja_pos = env.nplay_headless.ninja_position()
    print(f"✅ Ninja position: {ninja_pos}")
    
    # Find the leftmost switch
    switches = []
    for entity in env.entities:
        if isinstance(entity, dict):
            entity_type = entity.get('entity_type', entity.get('type', 'unknown'))
            x = entity.get('x', 0)
            y = entity.get('y', 0)
        else:
            entity_type = getattr(entity, 'entity_type', getattr(entity, 'type', 'unknown'))
            x = getattr(entity, 'x', 0)
            y = getattr(entity, 'y', 0)
            
        if entity_type == 4:  # Switch
            switches.append((x, y))
    
    if switches:
        leftmost_switch = min(switches, key=lambda s: s[0])
        print(f"🎯 Target switch: {leftmost_switch}")
    else:
        print("❌ No switches found!")
        return
    
    # Build graph with reduced sampling for faster testing
    print(f"\n🔧 Building graph (reduced sampling)...")
    builder = HierarchicalGraphBuilder()
    
    # Temporarily reduce sampling for testing
    original_sampling = True
    try:
        graph_data = builder.build_graph(env.level_data, ninja_pos)
        graph = graph_data.sub_cell_graph
        print(f"✅ Graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    except Exception as e:
        print(f"❌ Graph building failed: {e}")
        return
    
    # Analyze edge types
    edge_types = {}
    for edge in graph.edges:
        edge_type = edge.type.name
        if edge_type not in edge_types:
            edge_types[edge_type] = 0
        edge_types[edge_type] += 1
    
    print(f"\n📊 EDGE TYPE DISTRIBUTION:")
    for edge_type, count in edge_types.items():
        print(f"   {edge_type}: {count} edges")
    
    # Calculate movement diversity
    movement_types = [t for t in edge_types.keys() if t in ['WALK', 'JUMP', 'FALL']]
    movement_diversity = len(movement_types) / 3.0  # 3 expected types
    print(f"\n🎲 Movement diversity: {movement_diversity:.3f} ({len(movement_types)}/3 types)")
    
    # Test pathfinding
    print(f"\n🗺️  PATHFINDING TEST:")
    
    # Find ninja node
    ninja_node = None
    min_distance = float('inf')
    for node in graph.nodes:
        distance = math.sqrt((node.x - ninja_pos[0])**2 + (node.y - ninja_pos[1])**2)
        if distance < min_distance:
            min_distance = distance
            ninja_node = node
    
    if not ninja_node:
        print("❌ Ninja node not found!")
        return
    
    print(f"   Ninja node: {ninja_node.id} at ({ninja_node.x:.1f}, {ninja_node.y:.1f})")
    
    # Find target node
    target_node = None
    min_distance = float('inf')
    for node in graph.nodes:
        distance = math.sqrt((node.x - leftmost_switch[0])**2 + (node.y - leftmost_switch[1])**2)
        if distance < min_distance:
            min_distance = distance
            target_node = node
    
    if not target_node:
        print("❌ Target node not found!")
        return
    
    print(f"   Target node: {target_node.id} at ({target_node.x:.1f}, {target_node.y:.1f})")
    
    # Run pathfinding
    pathfinder = PathfindingEngine()
    path_result = pathfinder.find_shortest_path(
        graph_data, ninja_node.id, target_node.id, 
        algorithm=PathfindingAlgorithm.DIJKSTRA
    )
    path = path_result.path if path_result.success else None
    
    if path:
        print(f"✅ Path found: {len(path)} nodes")
        
        # Analyze path
        total_distance = 0
        path_movement_types = set()
        
        node_positions = {node.id: (node.x, node.y) for node in graph.nodes}
        
        for i in range(len(path) - 1):
            src_id = path[i]
            dst_id = path[i + 1]
            
            # Find the edge
            edge_found = None
            for edge in graph.edges:
                if edge.source == src_id and edge.target == dst_id:
                    edge_found = edge
                    break
            
            if edge_found:
                path_movement_types.add(edge_found.type.name)
                
                # Calculate distance
                src_pos = node_positions[src_id]
                dst_pos = node_positions[dst_id]
                segment_distance = math.sqrt(
                    (dst_pos[0] - src_pos[0])**2 + (dst_pos[1] - src_pos[1])**2
                )
                total_distance += segment_distance
                
                print(f"   Segment {i+1}: {edge_found.type.name} ({segment_distance:.1f}px)")
        
        print(f"\n📏 Path analysis:")
        print(f"   Total distance: {total_distance:.1f}px")
        print(f"   Movement types used: {sorted(path_movement_types)}")
        print(f"   Path complexity: {len(path)} nodes, {len(path_movement_types)} movement types")
        
        # Check if path meets expectations
        expected_min_nodes = 6  # Based on user image
        expected_min_types = 2  # At least WALK and JUMP
        
        success_criteria = [
            len(path) >= expected_min_nodes,
            len(path_movement_types) >= expected_min_types,
            'WALK' in path_movement_types,
            total_distance > 300  # Should be a complex path
        ]
        
        passed_criteria = sum(success_criteria)
        print(f"\n✅ Success criteria: {passed_criteria}/4")
        print(f"   Path length ≥ {expected_min_nodes}: {'✅' if success_criteria[0] else '❌'}")
        print(f"   Movement types ≥ {expected_min_types}: {'✅' if success_criteria[1] else '❌'}")
        print(f"   Contains WALK: {'✅' if success_criteria[2] else '❌'}")
        print(f"   Complex path (>300px): {'✅' if success_criteria[3] else '❌'}")
        
        if passed_criteria >= 3:
            print(f"\n🎉 PATHFINDING SUCCESS! Complex path with diverse movements.")
        else:
            print(f"\n⚠️  PATHFINDING NEEDS IMPROVEMENT. Path too simple.")
            
    else:
        print("❌ No path found!")

if __name__ == "__main__":
    test_complex_pathfinding()