#!/usr/bin/env python3
"""
Comprehensive test of doortest map fixes to validate all three issues are resolved.
"""

import os
import sys
import pygame
import numpy as np

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine
from nclone.graph.common import EdgeType


def test_doortest_fixes():
    """Test all fixes on the actual doortest map."""
    print("=" * 80)
    print("COMPREHENSIVE DOORTEST MAP VALIDATION")
    print("=" * 80)
    
    # Create environment with doortest map
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42
    )
    
    # Reset to load the map
    print("Loading doortest map...")
    env.reset()
    
    # Get level data and ninja position
    level_data = env.level_data
    ninja_pos = env.nplay_headless.ninja_position()
    
    print(f"✅ Map loaded: {level_data.width}x{level_data.height} tiles")
    print(f"✅ Ninja position: {ninja_pos}")
    print(f"✅ Entities: {len(level_data.entities)}")
    
    # Build graph
    print("\nBuilding hierarchical graph...")
    graph_builder = HierarchicalGraphBuilder()
    hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
    graph_data = hierarchical_data.sub_cell_graph
    pathfinding_engine = PathfindingEngine()
    
    print(f"✅ Graph built: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Test Issue #1: Functional Edges
    print("\n" + "=" * 60)
    print("ISSUE #1: FUNCTIONAL EDGES TEST")
    print("=" * 60)
    
    functional_edges = []
    for edge_idx in range(graph_data.num_edges):
        if graph_data.edge_mask[edge_idx] == 0:
            continue
        
        edge_type = int(graph_data.edge_types[edge_idx])
        if edge_type == EdgeType.FUNCTIONAL:
            src_idx = int(graph_data.edge_index[0, edge_idx])
            dst_idx = int(graph_data.edge_index[1, edge_idx])
            
            src_pos = pathfinding_engine._get_node_position(graph_data, src_idx)
            dst_pos = pathfinding_engine._get_node_position(graph_data, dst_idx)
            
            distance = ((src_pos[0] - dst_pos[0])**2 + (src_pos[1] - dst_pos[1])**2)**0.5
            functional_edges.append((src_pos, dst_pos, distance))
    
    print(f"Found {len(functional_edges)} functional edges:")
    for i, (src_pos, dst_pos, distance) in enumerate(functional_edges, 1):
        print(f"  {i}. {src_pos} -> {dst_pos} (distance: {distance:.1f})")
    
    if len(functional_edges) >= 2:
        print("✅ ISSUE #1 RESOLVED: Functional edges are working correctly")
    else:
        print("❌ ISSUE #1 NOT RESOLVED: Expected at least 2 functional edges")
    
    # Test Issue #2: Walkable Edges in Solid Tiles
    print("\n" + "=" * 60)
    print("ISSUE #2: WALKABLE EDGES IN SOLID TILES TEST")
    print("=" * 60)
    
    solid_violations = []
    ninja_escape_edges = 0
    
    for edge_idx in range(graph_data.num_edges):
        if graph_data.edge_mask[edge_idx] == 0:
            continue
        
        edge_type = int(graph_data.edge_types[edge_idx])
        if edge_type == EdgeType.WALK:
            src_idx = int(graph_data.edge_index[0, edge_idx])
            dst_idx = int(graph_data.edge_index[1, edge_idx])
            
            src_pos = pathfinding_engine._get_node_position(graph_data, src_idx)
            dst_pos = pathfinding_engine._get_node_position(graph_data, dst_idx)
            
            # Check if source node is in solid tile
            src_tile_x = int(src_pos[0] // 24)
            src_tile_y = int(src_pos[1] // 24)
            
            if (0 <= src_tile_x < level_data.width and 
                0 <= src_tile_y < level_data.height):
                tile_value = level_data.get_tile(src_tile_y, src_tile_x)
                
                if tile_value == 1:  # Solid tile
                    # Check if this is a ninja escape edge
                    ninja_distance_src = ((src_pos[0] - ninja_pos[0])**2 + (src_pos[1] - ninja_pos[1])**2)**0.5
                    ninja_distance_dst = ((dst_pos[0] - ninja_pos[0])**2 + (dst_pos[1] - ninja_pos[1])**2)**0.5
                    
                    if ninja_distance_src <= 24 or ninja_distance_dst <= 24:  # Within ninja escape range
                        ninja_escape_edges += 1
                    else:
                        solid_violations.append((src_pos, dst_pos, tile_value))
    
    print(f"Total walkable edges: {sum(1 for i in range(graph_data.num_edges) if graph_data.edge_mask[i] == 1 and int(graph_data.edge_types[i]) == EdgeType.WALK)}")
    print(f"Ninja escape edges (intentional): {ninja_escape_edges}")
    print(f"Invalid solid tile violations: {len(solid_violations)}")
    
    if len(solid_violations) == 0:
        print("✅ ISSUE #2 RESOLVED: No invalid walkable edges in solid tiles")
    else:
        print("❌ ISSUE #2 NOT RESOLVED: Found invalid edges in solid tiles")
        for i, (src_pos, dst_pos, tile_value) in enumerate(solid_violations[:5], 1):
            print(f"  {i}. {src_pos} -> {dst_pos} (tile value: {tile_value})")
    
    # Test Issue #3: Ninja Pathfinding
    print("\n" + "=" * 60)
    print("ISSUE #3: NINJA PATHFINDING TEST")
    print("=" * 60)
    
    # Find ninja node
    ninja_node = pathfinding_engine._find_node_at_position(graph_data, ninja_pos)
    print(f"Ninja node: {ninja_node}")
    
    if ninja_node is None:
        print("❌ ISSUE #3 NOT RESOLVED: Ninja node not found")
        return
    
    # Test pathfinding to various targets in empty areas
    test_targets = [
        (156, 252),  # Empty area
        (228, 276),  # Empty area  
        (204, 300),  # Empty area
        (156, 324),  # Empty area
        (180, 324),  # Empty area
    ]
    
    successful_paths = 0
    total_tests = len(test_targets)
    
    print(f"Testing pathfinding from ninja position {ninja_pos}:")
    
    for i, target_pos in enumerate(test_targets, 1):
        target_node = pathfinding_engine._find_node_at_position(graph_data, target_pos)
        
        if target_node is None:
            print(f"  Test {i}: {target_pos} -> ❌ Target node not found")
            continue
        
        # Attempt pathfinding
        path_result = pathfinding_engine.find_shortest_path(graph_data, ninja_node, target_node)
        
        if path_result and path_result.success and len(path_result.path) > 0:
            print(f"  Test {i}: {target_pos} -> ✅ Path found ({len(path_result.path)} nodes, cost: {path_result.total_cost:.1f})")
            successful_paths += 1
        else:
            print(f"  Test {i}: {target_pos} -> ❌ No path found")
    
    success_rate = (successful_paths / total_tests) * 100
    print(f"\nPathfinding success rate: {successful_paths}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 50:  # At least 50% success rate indicates ninja can move around
        print("✅ ISSUE #3 RESOLVED: Ninja pathfinding is working")
    else:
        print("❌ ISSUE #3 NOT RESOLVED: Ninja pathfinding success rate too low")
    
    # Test ninja's connected component size
    print(f"\nAnalyzing ninja's connectivity...")
    
    visited = set()
    stack = [ninja_node]
    component_size = 0
    
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        
        visited.add(current)
        component_size += 1
        
        # Find neighbors
        for edge_idx in range(graph_data.num_edges):
            if graph_data.edge_mask[edge_idx] == 0:
                continue
            
            src = int(graph_data.edge_index[0, edge_idx])
            dst = int(graph_data.edge_index[1, edge_idx])
            
            if src == current and dst not in visited:
                stack.append(dst)
            elif dst == current and src not in visited:
                stack.append(src)
    
    print(f"Ninja's connected component: {component_size} nodes")
    
    if component_size >= 20:  # Significant improvement from original 5 nodes
        print("✅ Ninja connectivity significantly improved")
    else:
        print("❌ Ninja connectivity still limited")
    
    # Generate visualization
    print(f"\n" + "=" * 60)
    print("GENERATING VISUALIZATION")
    print("=" * 60)
    
    try:
        # Get the current frame with graph overlay
        frame = env.render()
        
        if frame is not None:
            # Save the frame
            pygame.image.save(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), 
                            "doortest_validation.png")
            print("✅ Visualization saved as doortest_validation.png")
        else:
            print("❌ Could not generate visualization")
    except Exception as e:
        print(f"❌ Visualization error: {e}")
    
    # Final Summary
    print(f"\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)
    
    issue1_status = "✅ RESOLVED" if len(functional_edges) >= 2 else "❌ NOT RESOLVED"
    issue2_status = "✅ RESOLVED" if len(solid_violations) == 0 else "❌ NOT RESOLVED"
    issue3_status = "✅ RESOLVED" if success_rate >= 50 else "❌ NOT RESOLVED"
    
    print(f"Issue #1 (Functional edges): {issue1_status}")
    print(f"Issue #2 (Solid tile edges): {issue2_status}")
    print(f"Issue #3 (Ninja pathfinding): {issue3_status}")
    
    resolved_count = sum([
        len(functional_edges) >= 2,
        len(solid_violations) == 0,
        success_rate >= 50
    ])
    
    print(f"\nOverall: {resolved_count}/3 issues resolved")
    
    if resolved_count == 3:
        print("🎉 ALL ISSUES SUCCESSFULLY RESOLVED!")
    else:
        print("⚠️  Some issues still need attention")
    
    return {
        'functional_edges': len(functional_edges),
        'solid_violations': len(solid_violations),
        'ninja_escape_edges': ninja_escape_edges,
        'pathfinding_success_rate': success_rate,
        'ninja_component_size': component_size,
        'issues_resolved': resolved_count
    }


if __name__ == '__main__':
    results = test_doortest_fixes()
    print(f"\nTest completed. Results: {results}")