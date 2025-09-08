#!/usr/bin/env python3
"""
Create waypoint-based pathfinding that matches the expected path from the user image.
"""

import sys
import os
import math

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder

def create_waypoint_path():
    """Create a waypoint-based path that matches the expected route."""
    print("=" * 70)
    print("🗺️  WAYPOINT PATH CREATION")
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
    
    # Define expected waypoints based on the user's image
    # These represent the key transition points in the expected path
    expected_waypoints = [
        ninja_pos,  # Start: ninja position
        (200, 444),  # Waypoint 1: Move right along bottom platform (WALK)
        (280, 380),  # Waypoint 2: Jump up to next level (JUMP)
        (360, 380),  # Waypoint 3: Walk along middle platform (WALK)
        (420, 320),  # Waypoint 4: Jump up to higher level (JUMP)
        (480, 280),  # Waypoint 5: Navigate complex geometry (MIXED)
        (520, 220),  # Waypoint 6: Approach target area (JUMP)
        leftmost_switch  # End: target switch
    ]
    
    print(f"\n📍 EXPECTED WAYPOINTS:")
    for i, (x, y) in enumerate(expected_waypoints):
        print(f"   {i}: ({x:.1f}, {y:.1f})")
    
    # Build graph
    print(f"\n🔧 Building graph...")
    builder = HierarchicalGraphBuilder()
    graph_data = builder.build_graph(env.level_data, ninja_pos)
    graph = graph_data.sub_cell_graph
    
    print(f"✅ Graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # Find nodes near each waypoint
    waypoint_nodes = []
    node_positions = {node.id: (node.x, node.y) for node in graph.nodes}
    
    for i, (wx, wy) in enumerate(expected_waypoints):
        closest_node = None
        min_distance = float('inf')
        
        for node in graph.nodes:
            distance = math.sqrt((node.x - wx)**2 + (node.y - wy)**2)
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        
        if closest_node:
            waypoint_nodes.append((i, closest_node, min_distance))
            print(f"   Waypoint {i}: Node {closest_node.id} at ({closest_node.x:.1f}, {closest_node.y:.1f}) - distance {min_distance:.1f}px")
        else:
            print(f"   Waypoint {i}: No nearby node found!")
    
    # Analyze connections between waypoints
    print(f"\n🔗 WAYPOINT CONNECTIONS:")
    
    for i in range(len(waypoint_nodes) - 1):
        curr_waypoint, curr_node, _ = waypoint_nodes[i]
        next_waypoint, next_node, _ = waypoint_nodes[i + 1]
        
        # Check if there's a direct edge between these nodes
        direct_edge = None
        for edge in graph.edges:
            if edge.source == curr_node.id and edge.target == next_node.id:
                direct_edge = edge
                break
        
        if direct_edge:
            print(f"   {curr_waypoint} -> {next_waypoint}: Direct {direct_edge.type.name} edge")
        else:
            print(f"   {curr_waypoint} -> {next_waypoint}: No direct edge - needs pathfinding")
            
            # Calculate expected movement type
            dx = next_node.x - curr_node.x
            dy = next_node.y - curr_node.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if abs(dy) < 12:
                expected_type = "WALK"
            elif dy < -12:
                expected_type = "JUMP"
            elif dy > 12:
                expected_type = "FALL"
            else:
                expected_type = "MIXED"
            
            print(f"      Expected: {expected_type} ({distance:.1f}px, dx={dx:.1f}, dy={dy:.1f})")
    
    # Check if we have the necessary edge types
    edge_types = {}
    for edge in graph.edges:
        edge_type = edge.type.name
        if edge_type not in edge_types:
            edge_types[edge_type] = 0
        edge_types[edge_type] += 1
    
    print(f"\n📊 EDGE TYPE DISTRIBUTION:")
    for edge_type, count in edge_types.items():
        print(f"   {edge_type}: {count} edges")
    
    # Identify missing connections
    print(f"\n⚠️  MISSING CONNECTIONS ANALYSIS:")
    
    if edge_types.get('JUMP', 0) < 10:
        print(f"   - Insufficient JUMP edges ({edge_types.get('JUMP', 0)}) for vertical navigation")
    
    if edge_types.get('FALL', 0) < 5:
        print(f"   - Insufficient FALL edges ({edge_types.get('FALL', 0)}) for downward movement")
    
    total_movement_types = len([t for t in edge_types.keys() if t in ['WALK', 'JUMP', 'FALL']])
    if total_movement_types < 3:
        print(f"   - Only {total_movement_types} movement types available, need 3+")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Increase node density around waypoint areas")
    print(f"   2. Create more short-range JUMP/FALL connections")
    print(f"   3. Reduce reliance on long-distance corridor connections")
    print(f"   4. Ensure pathfinding follows level geometry")

if __name__ == "__main__":
    create_waypoint_path()