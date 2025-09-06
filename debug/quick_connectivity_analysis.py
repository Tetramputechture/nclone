#!/usr/bin/env python3
"""
Quick analysis of graph connectivity issues for long-distance pathfinding.
"""

import os
import sys
import numpy as np

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine


def quick_connectivity_analysis():
    """Quick analysis of connectivity issues."""
    print("=" * 80)
    print("QUICK CONNECTIVITY ANALYSIS FOR LONG-DISTANCE PATHFINDING")
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
    
    print(f"Map: {level_data.width}x{level_data.height} tiles")
    print(f"Ninja: {ninja_pos}")
    
    # Count tile types
    empty_tiles = sum(1 for y in range(level_data.height) for x in range(level_data.width) 
                     if level_data.get_tile(y, x) == 0)
    total_tiles = level_data.width * level_data.height
    
    print(f"Empty tiles: {empty_tiles}/{total_tiles} ({empty_tiles/total_tiles*100:.1f}%)")
    
    # Build graph
    graph_builder = HierarchicalGraphBuilder()
    hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
    graph_data = hierarchical_data.sub_cell_graph
    pathfinding_engine = PathfindingEngine()
    
    print(f"Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Find ninja's component size (we already know this)
    ninja_node = pathfinding_engine._find_node_at_position(graph_data, ninja_pos)
    print(f"Ninja node: {ninja_node}")
    
    # Sample a few distant empty tile positions to test connectivity
    distant_targets = [
        (156, 252),  # Top area
        (228, 276),  # Middle area
        (204, 300),  # Lower middle
        (396, 204),  # Switch area
        (480, 276),  # Door area
    ]
    
    print(f"\nTesting connectivity to distant targets:")
    
    reachable_targets = 0
    
    for i, target_pos in enumerate(distant_targets, 1):
        target_node = pathfinding_engine._find_node_at_position(graph_data, target_pos)
        
        if target_node is None:
            print(f"  {i}. {target_pos} -> ❌ No node found")
            continue
        
        # Check tile type
        tile_x = int(target_pos[0] // 24)
        tile_y = int(target_pos[1] // 24)
        tile_value = level_data.get_tile(tile_y, tile_x) if (0 <= tile_x < level_data.width and 0 <= tile_y < level_data.height) else -1
        tile_type = "empty" if tile_value == 0 else "solid" if tile_value == 1 else "other"
        
        # Test pathfinding
        path_result = pathfinding_engine.find_shortest_path(graph_data, ninja_node, target_node)
        
        if path_result and path_result.success:
            print(f"  {i}. {target_pos} ({tile_type}) -> ✅ Reachable ({len(path_result.path)} nodes)")
            reachable_targets += 1
        else:
            print(f"  {i}. {target_pos} ({tile_type}) -> ❌ Not reachable")
    
    reachability_rate = (reachable_targets / len(distant_targets)) * 100
    print(f"\nDistant target reachability: {reachable_targets}/{len(distant_targets)} ({reachability_rate:.1f}%)")
    
    # Analyze the problem
    print(f"\n" + "=" * 60)
    print("CONNECTIVITY PROBLEM DIAGNOSIS")
    print("=" * 60)
    
    if reachability_rate == 0:
        print("❌ SEVERE CONNECTIVITY ISSUE: No distant targets reachable")
        print("   Root cause: Graph is highly fragmented")
        print("   Solution needed: Improve collision detection or edge building")
    elif reachability_rate < 50:
        print("⚠️  MODERATE CONNECTIVITY ISSUE: Limited distant reachability")
        print("   Root cause: Some graph fragmentation")
        print("   Solution needed: Minor connectivity improvements")
    else:
        print("✅ GOOD CONNECTIVITY: Most distant targets reachable")
    
    # Check specific empty tile areas
    print(f"\nAnalyzing specific empty tile areas:")
    
    # Define known empty tile areas from the map
    empty_areas = [
        ("Top area", [(156, 252), (180, 252)]),
        ("Middle area", [(228, 276), (252, 276)]),
        ("Lower area", [(204, 300), (228, 300)]),
        ("Bottom area", [(156, 324), (180, 324)]),
    ]
    
    for area_name, positions in empty_areas:
        print(f"\n{area_name}:")
        area_reachable = 0
        
        for pos in positions:
            target_node = pathfinding_engine._find_node_at_position(graph_data, pos)
            if target_node is None:
                print(f"  {pos} -> ❌ No node")
                continue
            
            path_result = pathfinding_engine.find_shortest_path(graph_data, ninja_node, target_node)
            if path_result and path_result.success:
                print(f"  {pos} -> ✅ Reachable")
                area_reachable += 1
            else:
                print(f"  {pos} -> ❌ Not reachable")
        
        area_rate = (area_reachable / len(positions)) * 100 if positions else 0
        print(f"  Area reachability: {area_reachable}/{len(positions)} ({area_rate:.1f}%)")
    
    # Recommendations
    print(f"\n" + "=" * 60)
    print("RECOMMENDATIONS FOR LONG-DISTANCE PATHFINDING")
    print("=" * 60)
    
    if reachability_rate == 0:
        print("🔧 IMMEDIATE ACTIONS NEEDED:")
        print("   1. Investigate PreciseTileCollision system")
        print("   2. Check if collision detection is too restrictive")
        print("   3. Analyze why empty tiles are not connecting")
        print("   4. Consider relaxing collision constraints for empty areas")
        
        print("\n🎯 SPECIFIC FIXES TO TRY:")
        print("   • Reduce ninja radius in collision detection")
        print("   • Allow more lenient traversal in empty tile areas")
        print("   • Add special connectivity rules for empty-to-empty tile transitions")
        print("   • Debug why sub-grid nodes in empty tiles lack connections")
    
    elif reachability_rate < 50:
        print("🔧 MODERATE IMPROVEMENTS NEEDED:")
        print("   1. Identify specific connectivity bottlenecks")
        print("   2. Improve edge building between empty tile clusters")
        print("   3. Add bridge connections for isolated areas")
    
    else:
        print("✅ CONNECTIVITY IS GOOD!")
        print("   Long-distance pathfinding is working well.")
    
    return {
        'distant_reachability_rate': reachability_rate,
        'reachable_targets': reachable_targets,
        'total_targets': len(distant_targets)
    }


if __name__ == '__main__':
    results = quick_connectivity_analysis()
    print(f"\nQuick analysis completed. Results: {results}")