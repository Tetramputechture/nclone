#!/usr/bin/env python3
"""
Analyze the current pathfinding results to understand why movements are physically impossible.
"""

import sys
import os
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.constants.physics_constants import TILE_PIXEL_SIZE
from nclone.graph.common import EdgeType

def analyze_current_path():
    """Analyze the current pathfinding result to understand movement issues."""
    print("=" * 80)
    print("🔍 ANALYZING CURRENT PATHFINDING PHYSICS")
    print("=" * 80)
    
    # Load environment
    env = BasicLevelNoGold(render_mode="rgb_array")
    level_data = env.level_data
    entities = env.entities
    
    # Get ninja position from entities
    ninja_position = None
    for entity in entities:
        if entity.get('type') == 'ninja':
            ninja_position = (entity['x'], entity['y'])
            break
    
    if ninja_position is None:
        ninja_position = (132, 444)  # Default known position
    
    print(f"📍 Ninja position: {ninja_position}")
    print(f"🗺️  Level size: {level_data.width}x{level_data.height} tiles")
    
    # Build graph
    builder = HierarchicalGraphBuilder()
    hierarchical_graph = builder.build_graph(level_data, ninja_position)
    
    # Use the tile-level graph for pathfinding
    graph = hierarchical_graph.tile_graph
    
    print(f"📊 Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Find ninja and target nodes
    ninja_node = None
    target_node = None
    
    # Find ninja node (closest to ninja position)
    min_distance = float('inf')
    for i in range(graph.num_nodes):
        if graph.node_mask[i] == 1:
            node_x = graph.node_features[i, 0]
            node_y = graph.node_features[i, 1]
            distance = ((node_x - ninja_position[0])**2 + (node_y - ninja_position[1])**2)**0.5
            if distance < min_distance:
                min_distance = distance
                ninja_node = i
    
    # Find target node (leftmost locked door switch at y=204)
    target_switches = []
    for i in range(graph.num_nodes):
        if graph.node_mask[i] == 1:
            node_x = graph.node_features[i, 0]
            node_y = graph.node_features[i, 1]
            # Look for switches around y=204 (locked door switches)
            if abs(node_y - 204) < 10:
                target_switches.append((i, node_x, node_y))
    
    # Sort by x coordinate and take leftmost
    target_switches.sort(key=lambda x: x[1])
    if target_switches:
        target_node = target_switches[0][0]
        target_x = target_switches[0][1]
        target_y = target_switches[0][2]
    else:
        print("❌ No target switches found!")
        return
    
    print(f"🥷 Ninja node: {ninja_node} at ({graph.node_features[ninja_node, 0]:.1f}, {graph.node_features[ninja_node, 1]:.1f})")
    print(f"🎯 Target node: {target_node} at ({target_x:.1f}, {target_y:.1f})")
    
    # Find path
    pathfinding_engine = PathfindingEngine()
    result = pathfinding_engine.find_shortest_path(
        graph, ninja_node, target_node, PathfindingAlgorithm.DIJKSTRA
    )
    
    if not result.success:
        print("❌ No path found!")
        return
    
    print(f"\n🛤️  PATH ANALYSIS:")
    print(f"   Path length: {len(result.path)} nodes")
    print(f"   Total cost: {result.total_cost:.1f}px")
    print(f"   Movement types: {[EdgeType(et).name for et in result.edge_types]}")
    
    # Analyze each segment of the path
    print(f"\n📋 SEGMENT-BY-SEGMENT ANALYSIS:")
    for i in range(len(result.path) - 1):
        src_node = result.path[i]
        dst_node = result.path[i + 1]
        movement_type = EdgeType(result.edge_types[i])
        
        src_x = graph.node_features[src_node, 0]
        src_y = graph.node_features[src_node, 1]
        dst_x = graph.node_features[dst_node, 0]
        dst_y = graph.node_features[dst_node, 1]
        
        dx = dst_x - src_x
        dy = dst_y - src_y
        distance = (dx**2 + dy**2)**0.5
        
        print(f"\n   Segment {i+1}: Node {src_node} → Node {dst_node}")
        print(f"   Position: ({src_x:.1f}, {src_y:.1f}) → ({dst_x:.1f}, {dst_y:.1f})")
        print(f"   Movement: {movement_type.name} (Δx={dx:.1f}, Δy={dy:.1f}, dist={distance:.1f}px)")
        
        # Analyze if this movement makes physical sense
        analyze_movement_physics(src_x, src_y, dst_x, dst_y, movement_type, level_data)

def analyze_movement_physics(src_x, src_y, dst_x, dst_y, movement_type, level_data):
    """Analyze if a movement segment is physically possible."""
    dx = dst_x - src_x
    dy = dst_y - src_y
    distance = (dx**2 + dy**2)**0.5
    
    # Check what tiles are between source and destination
    blocking_tiles = check_path_for_blocking_tiles(src_x, src_y, dst_x, dst_y, level_data)
    
    print(f"   🔍 Physics Analysis:")
    
    # Analyze based on movement type
    if movement_type == EdgeType.WALK:
        if abs(dy) > 12:  # Walking shouldn't have large vertical changes
            print(f"   ⚠️  WARNING: WALK movement has large vertical change ({dy:.1f}px)")
        if distance > 50:  # Walking shouldn't be too far
            print(f"   ⚠️  WARNING: WALK movement is very long ({distance:.1f}px)")
    
    elif movement_type == EdgeType.JUMP:
        if dy > 0:  # Jumping should go up (negative y)
            print(f"   ❌ ERROR: JUMP movement goes down ({dy:.1f}px) - should go up!")
        if distance > 120:  # Jump distance limit
            print(f"   ⚠️  WARNING: JUMP movement is very long ({distance:.1f}px)")
    
    elif movement_type == EdgeType.FALL:
        if dy < 0:  # Falling should go down (positive y)
            print(f"   ❌ ERROR: FALL movement goes up ({dy:.1f}px) - should go down!")
        if abs(dx) > 60:  # Falls shouldn't have huge horizontal movement
            print(f"   ⚠️  WARNING: FALL movement has large horizontal change ({dx:.1f}px)")
    
    # Check for blocking tiles
    if blocking_tiles > 0:
        print(f"   ❌ ERROR: Path blocked by {blocking_tiles} solid tiles!")
    else:
        print(f"   ✅ Path appears clear of solid tiles")

def check_path_for_blocking_tiles(x1, y1, x2, y2, level_data):
    """Check how many solid tiles block the path between two points."""
    import math
    
    # Sample points along the line
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    num_samples = max(5, int(distance / 8))  # Sample every 8 pixels
    
    blocking_tiles = set()
    
    for i in range(1, num_samples):  # Skip endpoints
        t = i / num_samples
        sample_x = x1 + t * (x2 - x1)
        sample_y = y1 + t * (y2 - y1)
        
        # Convert to tile coordinates (accounting for padding)
        tile_x = int((sample_x - TILE_PIXEL_SIZE) // TILE_PIXEL_SIZE)
        tile_y = int((sample_y - TILE_PIXEL_SIZE) // TILE_PIXEL_SIZE)
        
        # Check if tile is in bounds and solid
        if (0 <= tile_x < level_data.width and 0 <= tile_y < level_data.height):
            tile_value = level_data.tiles[tile_y][tile_x]
            if tile_value == 1:  # Solid tile
                blocking_tiles.add((tile_x, tile_y))
    
    return len(blocking_tiles)

if __name__ == "__main__":
    analyze_current_path()