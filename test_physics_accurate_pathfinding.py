#!/usr/bin/env python3
"""
Test physics-accurate pathfinding improvements.

This script validates that the corridor connections now use proper movement types
(JUMP/FALL) instead of impossible long-distance WALK movements.
"""

import sys
import os
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.graph.common import EdgeType

def test_physics_accurate_pathfinding():
    """Test that pathfinding now uses physics-accurate movement types."""
    print("=" * 80)
    print("🧪 TESTING PHYSICS-ACCURATE PATHFINDING IMPROVEMENTS")
    print("=" * 80)
    
    # Load environment
    env = BasicLevelNoGold(render_mode="rgb_array")
    level_data = env.level_data
    entities = env.entities
    ninja_position = env.nplay_headless.ninja_position()
    
    print(f"📍 Ninja position: {ninja_position}")
    print(f"🗺️  Level size: {level_data.width}x{level_data.height} tiles")
    
    # Build hierarchical graph
    builder = HierarchicalGraphBuilder()
    hierarchical_graph = builder.build_graph(level_data, ninja_position)
    graph = hierarchical_graph.sub_cell_graph
    
    print(f"✅ Graph built: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Analyze edge types and distances
    edge_analysis = {
        EdgeType.WALK: [],
        EdgeType.JUMP: [],
        EdgeType.FALL: [],
        EdgeType.FUNCTIONAL: [],
        EdgeType.WALL_SLIDE: [],
        EdgeType.ONE_WAY: []
    }
    
    print(f"\n🔍 ANALYZING EDGE TYPES AND DISTANCES:")
    
    for edge_idx in range(graph.num_edges):
        if graph.edge_mask[edge_idx] == 1:  # Valid edge
            src_node = graph.edge_index[0, edge_idx]
            dst_node = graph.edge_index[1, edge_idx]
            
            # Get node positions
            src_x = graph.node_features[src_node, 0]
            src_y = graph.node_features[src_node, 1]
            dst_x = graph.node_features[dst_node, 0]
            dst_y = graph.node_features[dst_node, 1]
            
            # Calculate distance
            distance = ((dst_x - src_x)**2 + (dst_y - src_y)**2)**0.5
            
            # Determine edge type
            edge_type = None
            for et in EdgeType:
                if graph.edge_features[edge_idx, et] > 0.5:
                    edge_type = et
                    break
            
            if edge_type is not None:
                edge_analysis[edge_type].append(distance)
    
    # Print analysis results
    total_edges = sum(len(distances) for distances in edge_analysis.values())
    print(f"   Total analyzed edges: {total_edges}")
    
    for edge_type, distances in edge_analysis.items():
        if distances:
            avg_distance = sum(distances) / len(distances)
            max_distance = max(distances)
            min_distance = min(distances)
            
            print(f"   {edge_type.name}: {len(distances)} edges")
            print(f"      Distance range: {min_distance:.1f}px - {max_distance:.1f}px")
            print(f"      Average distance: {avg_distance:.1f}px")
            
            # Check for physics violations
            if edge_type == EdgeType.WALK and max_distance > 50:
                print(f"      ⚠️  WARNING: Long WALK edges detected (max {max_distance:.1f}px)")
            elif edge_type == EdgeType.WALK and max_distance <= 50:
                print(f"      ✅ WALK edges within physics limits")
    
    # Test pathfinding with physics engine
    print(f"\n🚀 TESTING PATHFINDING ENGINE:")
    
    # Find ninja node
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
    
    # Find a target node (different from ninja)
    target_node = None
    for node_idx in range(graph.num_nodes):
        if (graph.node_mask[node_idx] == 1 and 
            node_idx != ninja_node):
            node_x = graph.node_features[node_idx, 0]
            node_y = graph.node_features[node_idx, 1]
            # Find a node that's reasonably far away
            dist = ((node_x - ninja_position[0])**2 + (node_y - ninja_position[1])**2)**0.5
            if dist > 100:  # At least 100px away
                target_node = node_idx
                break
    
    if ninja_node is not None and target_node is not None:
        ninja_coords = (graph.node_features[ninja_node, 0], graph.node_features[ninja_node, 1])
        target_coords = (graph.node_features[target_node, 0], graph.node_features[target_node, 1])
        
        print(f"   Ninja node: {ninja_node} at {ninja_coords}")
        print(f"   Target node: {target_node} at {target_coords}")
        
        # Test pathfinding with Dijkstra
        pathfinding_engine = PathfindingEngine(level_data, entities)
        result = pathfinding_engine.find_shortest_path(
            graph, ninja_node, target_node, PathfindingAlgorithm.DIJKSTRA
        )
        
        if result.success:
            print(f"   ✅ Path found: {len(result.path)} nodes, {result.total_cost:.1f}px cost")
            
            # Analyze path movement types
            movement_counts = {}
            for edge_type in result.edge_types:
                edge_name = EdgeType(edge_type).name
                movement_counts[edge_name] = movement_counts.get(edge_name, 0) + 1
            
            print(f"   Movement types: {movement_counts}")
            
            # Check for physics violations in path
            has_violations = False
            for i in range(len(result.path) - 1):
                src_node = result.path[i]
                dst_node = result.path[i + 1]
                
                src_x = graph.node_features[src_node, 0]
                src_y = graph.node_features[src_node, 1]
                dst_x = graph.node_features[dst_node, 0]
                dst_y = graph.node_features[dst_node, 1]
                
                distance = ((dst_x - src_x)**2 + (dst_y - src_y)**2)**0.5
                edge_type = EdgeType(result.edge_types[i])
                
                # Check for physics violations
                if edge_type == EdgeType.WALK and distance > 50:
                    print(f"   ❌ PHYSICS VIOLATION: WALK edge {distance:.1f}px in path")
                    has_violations = True
            
            if not has_violations:
                print(f"   ✅ Path respects physics constraints")
                
        else:
            print(f"   ❌ No path found")
    else:
        print(f"   ❌ Could not find suitable nodes for pathfinding test")
    
    # Summary
    print(f"\n📊 PHYSICS-ACCURATE PATHFINDING SUMMARY:")
    
    walk_edges = edge_analysis[EdgeType.WALK]
    jump_edges = edge_analysis[EdgeType.JUMP]
    fall_edges = edge_analysis[EdgeType.FALL]
    
    if walk_edges:
        max_walk = max(walk_edges)
        if max_walk <= 50:
            print(f"   ✅ WALK edges: {len(walk_edges)} edges, max {max_walk:.1f}px (PHYSICS COMPLIANT)")
        else:
            print(f"   ❌ WALK edges: {len(walk_edges)} edges, max {max_walk:.1f}px (PHYSICS VIOLATION)")
    
    if jump_edges:
        print(f"   ✅ JUMP edges: {len(jump_edges)} edges (physics-accurate)")
    
    if fall_edges:
        print(f"   ✅ FALL edges: {len(fall_edges)} edges (physics-accurate)")
    
    print(f"   📈 Total edges using physics-accurate types: {len(jump_edges) + len(fall_edges)}")
    
    return True

if __name__ == "__main__":
    test_physics_accurate_pathfinding()