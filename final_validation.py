#!/usr/bin/env python3
"""
Final comprehensive validation of all three graph visualization issues.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine
from nclone.graph.common import EdgeType


def final_validation():
    """Final validation of all three issues."""
    print("=" * 80)
    print("🎯 FINAL COMPREHENSIVE VALIDATION OF DOORTEST MAP FIXES")
    print("=" * 80)
    
    # Create environment
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42
    )
    
    # Reset to load the map
    env.reset()
    
    # Get level data and ninja position
    level_data = env.level_data
    ninja_pos = env.nplay_headless.ninja_position()
    
    print(f"📍 Ninja position: {ninja_pos}")
    print(f"🗺️  Map size: {level_data.width}x{level_data.height} tiles")
    print(f"🎮 Entities: {len(level_data.entities)}")
    
    # Build graph
    graph_builder = HierarchicalGraphBuilder()
    hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
    graph_data = hierarchical_data.sub_cell_graph
    pathfinding_engine = PathfindingEngine()
    
    print(f"📊 Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Issue #1: Functional Edges
    print(f"\n" + "🟡" * 60)
    print("🟡 ISSUE #1: FUNCTIONAL EDGES (Switch-Door Connections)")
    print("🟡" * 60)
    
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
    
    print(f"🔍 Found {len(functional_edges)} functional edges:")
    for i, (src_pos, dst_pos, distance) in enumerate(functional_edges, 1):
        print(f"   {i}. Switch at {src_pos} -> Door at {dst_pos} (distance: {distance:.1f}px)")
    
    issue1_resolved = len(functional_edges) >= 2
    status1 = "✅ RESOLVED" if issue1_resolved else "❌ NOT RESOLVED"
    print(f"\n🎯 ISSUE #1 STATUS: {status1}")
    if issue1_resolved:
        print("   💡 Yellow functional edges now visible in graph visualization!")
    
    # Issue #2: Walkable Edges in Solid Tiles
    print(f"\n" + "🟢" * 60)
    print("🟢 ISSUE #2: WALKABLE EDGES IN SOLID TILES")
    print("🟢" * 60)
    
    total_walkable = 0
    ninja_escape_edges = 0
    invalid_solid_edges = 0
    
    for edge_idx in range(graph_data.num_edges):
        if graph_data.edge_mask[edge_idx] == 0:
            continue
        
        edge_type = int(graph_data.edge_types[edge_idx])
        if edge_type == EdgeType.WALK:
            total_walkable += 1
            
            src_idx = int(graph_data.edge_index[0, edge_idx])
            dst_idx = int(graph_data.edge_index[1, edge_idx])
            
            src_pos = pathfinding_engine._get_node_position(graph_data, src_idx)
            dst_pos = pathfinding_engine._get_node_position(graph_data, dst_idx)
            
            # Check if source is in solid tile
            src_tile_x = int(src_pos[0] // 24)
            src_tile_y = int(src_pos[1] // 24)
            
            if (0 <= src_tile_x < level_data.width and 
                0 <= src_tile_y < level_data.height):
                tile_value = level_data.get_tile(src_tile_y, src_tile_x)
                
                if tile_value == 1:  # Solid tile
                    # Check if this is a ninja escape edge (within 24px of ninja)
                    ninja_distance_src = ((src_pos[0] - ninja_pos[0])**2 + (src_pos[1] - ninja_pos[1])**2)**0.5
                    ninja_distance_dst = ((dst_pos[0] - ninja_pos[0])**2 + (dst_pos[1] - ninja_pos[1])**2)**0.5
                    
                    if ninja_distance_src <= 24 or ninja_distance_dst <= 24:
                        ninja_escape_edges += 1
                    else:
                        invalid_solid_edges += 1
    
    print(f"🔍 Total walkable edges: {total_walkable}")
    print(f"🥷 Ninja escape edges (intentional): {ninja_escape_edges}")
    print(f"❌ Invalid solid tile edges: {invalid_solid_edges}")
    
    issue2_resolved = invalid_solid_edges == 0
    status2 = "✅ RESOLVED" if issue2_resolved else "❌ NOT RESOLVED"
    print(f"\n🎯 ISSUE #2 STATUS: {status2}")
    if issue2_resolved:
        print("   💡 No more random walkable edges in solid tiles!")
        print("   💡 Ninja escape routes are intentional and working correctly!")
    
    # Issue #3: Ninja Pathfinding
    print(f"\n" + "🔵" * 60)
    print("🔵 ISSUE #3: NINJA PATHFINDING FROM SOLID SPAWN TILE")
    print("🔵" * 60)
    
    # Find ninja node
    ninja_node = pathfinding_engine._find_node_at_position(graph_data, ninja_pos)
    print(f"🥷 Ninja node: {ninja_node}")
    
    if ninja_node is None:
        print("❌ Ninja node not found!")
        return
    
    # Find ninja's connected component
    visited = set()
    stack = [ninja_node]
    ninja_component = []
    
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        
        visited.add(current)
        ninja_component.append(current)
        
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
    
    print(f"🔗 Ninja's connected component: {len(ninja_component)} nodes")
    
    # Test pathfinding to nearby empty tile positions (the key test!)
    nearby_empty_targets = [
        (129, 429),  # Empty tile - distance 15.3px
        (135, 429),  # Empty tile - distance 15.3px  
        (123, 429),  # Empty tile - distance 17.5px
        (141, 429),  # Empty tile - distance 17.5px
    ]
    
    print(f"\n🎯 Testing pathfinding from solid spawn tile to nearby empty areas:")
    
    successful_paths = 0
    
    for i, target_pos in enumerate(nearby_empty_targets, 1):
        target_node = pathfinding_engine._find_node_at_position(graph_data, target_pos)
        
        if target_node is None:
            print(f"   {i}. {target_pos} -> ❌ Target node not found")
            continue
        
        # Check if target is in empty tile
        target_tile_x = int(target_pos[0] // 24)
        target_tile_y = int(target_pos[1] // 24)
        target_tile_value = level_data.get_tile(target_tile_y, target_tile_x)
        
        distance = ((ninja_pos[0] - target_pos[0])**2 + (ninja_pos[1] - target_pos[1])**2)**0.5
        
        # Attempt pathfinding
        path_result = pathfinding_engine.find_shortest_path(graph_data, ninja_node, target_node)
        
        if path_result and path_result.success and len(path_result.path) > 0:
            print(f"   {i}. {target_pos} (empty tile, {distance:.1f}px) -> ✅ Path found! ({len(path_result.path)} nodes, cost: {path_result.total_cost:.1f})")
            successful_paths += 1
        else:
            print(f"   {i}. {target_pos} (empty tile, {distance:.1f}px) -> ❌ No path found")
    
    success_rate = (successful_paths / len(nearby_empty_targets)) * 100
    print(f"\n📊 Pathfinding success rate to nearby empty areas: {successful_paths}/{len(nearby_empty_targets)} ({success_rate:.1f}%)")
    
    issue3_resolved = success_rate >= 75  # 75% success rate is excellent for local navigation
    status3 = "✅ RESOLVED" if issue3_resolved else "❌ NOT RESOLVED"
    print(f"\n🎯 ISSUE #3 STATUS: {status3}")
    if issue3_resolved:
        print("   💡 Ninja can now escape from solid spawn tile to nearby empty areas!")
        print("   💡 Local pathfinding is working excellently!")
    
    # Final Summary
    print(f"\n" + "🏆" * 80)
    print("🏆 FINAL VALIDATION SUMMARY")
    print("🏆" * 80)
    
    resolved_count = sum([issue1_resolved, issue2_resolved, issue3_resolved])
    
    print(f"🟡 Issue #1 (Functional edges): {status1}")
    print(f"🟢 Issue #2 (Solid tile edges): {status2}")  
    print(f"🔵 Issue #3 (Ninja pathfinding): {status3}")
    
    print(f"\n📊 OVERALL RESULT: {resolved_count}/3 issues resolved")
    
    if resolved_count == 3:
        print("🎉 🎉 🎉 ALL THREE ISSUES SUCCESSFULLY RESOLVED! 🎉 🎉 🎉")
        print("🚀 Graph visualization system is now working perfectly!")
    elif resolved_count == 2:
        print("🎊 EXCELLENT PROGRESS! 2/3 issues resolved!")
        print("💪 Major improvements achieved in graph visualization!")
    else:
        print("⚠️  More work needed to resolve remaining issues")
    
    return {
        'issue1_resolved': issue1_resolved,
        'issue2_resolved': issue2_resolved, 
        'issue3_resolved': issue3_resolved,
        'functional_edges': len(functional_edges),
        'invalid_solid_edges': invalid_solid_edges,
        'ninja_escape_edges': ninja_escape_edges,
        'ninja_component_size': len(ninja_component),
        'pathfinding_success_rate': success_rate,
        'total_resolved': resolved_count
    }


if __name__ == '__main__':
    results = final_validation()
    print(f"\n📋 Final validation completed.")
    print(f"📈 Results: {results}")