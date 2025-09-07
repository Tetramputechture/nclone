#!/usr/bin/env python3
"""
Debug script to analyze player radius collision detection issues.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.constants.physics_constants import TILE_PIXEL_SIZE, NINJA_RADIUS
from nclone.constants.entity_types import EntityType

def analyze_doortest_map():
    """Analyze the doortest map and player position."""
    print("=== Analyzing doortest map ===")
    
    # Load the map using the environment
    map_path = "nclone/test_maps/doortest"
    if not os.path.exists(map_path):
        print(f"❌ Map file not found: {map_path}")
        return
        
    print(f"✅ Loading map: {map_path}")
    
    # Create environment with doortest map
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42,
        custom_map_path=map_path,
    )
    env.reset()
    
    level_data = env.level_data
    ninja_pos = env.nplay_headless.ninja_position()
    
    print(f"Map dimensions: {level_data.width}x{level_data.height} tiles")
    print(f"Tile pixel size: {TILE_PIXEL_SIZE}px")
    print(f"Map pixel dimensions: {level_data.width * TILE_PIXEL_SIZE}x{level_data.height * TILE_PIXEL_SIZE}px")
    print(f"Initial ninja position: {ninja_pos}")
    print(f"Ninja radius: {NINJA_RADIUS}px")
    
    # Check tiles around ninja position
    print(f"\n=== Tiles around ninja position ===")
    ninja_x, ninja_y = ninja_pos
    
    # Check a 5x5 grid around the ninja
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            tile_x = int((ninja_x + dx * 12) // 24)
            tile_y = int((ninja_y + dy * 12) // 24)
            
            if 0 <= tile_x < level_data.width and 0 <= tile_y < level_data.height:
                tile_id = level_data.tiles[tile_y, tile_x]
                pixel_x = ninja_x + dx * 12
                pixel_y = ninja_y + dy * 12
                print(f"  Tile ({tile_x:2d}, {tile_y:2d}) at pixel ({pixel_x:3.0f}, {pixel_y:3.0f}): tile_id={tile_id}")
            else:
                print(f"  Tile ({tile_x:2d}, {tile_y:2d}) OUT OF BOUNDS")
    
    # Specifically check the ninja's collision circle
    print(f"\n=== Ninja collision circle analysis ===")
    print(f"Ninja center: ({ninja_x}, {ninja_y})")
    print(f"Ninja radius: {NINJA_RADIUS}px")
    print(f"Ninja bounding box: x=[{ninja_x-NINJA_RADIUS}, {ninja_x+NINJA_RADIUS}], y=[{ninja_y-NINJA_RADIUS}, {ninja_y+NINJA_RADIUS}]")
    
    # Check the tiles that the ninja circle overlaps
    import math
    for angle in [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]:
        check_x = ninja_x + NINJA_RADIUS * math.cos(angle)
        check_y = ninja_y + NINJA_RADIUS * math.sin(angle)
        tile_x = int(check_x // 24)
        tile_y = int(check_y // 24)
        
        if 0 <= tile_x < level_data.width and 0 <= tile_y < level_data.height:
            tile_id = level_data.tiles[tile_y, tile_x]
            print(f"  Angle {angle:4.2f}: pos ({check_x:5.1f}, {check_y:5.1f}) -> tile ({tile_x:2d}, {tile_y:2d}) -> tile_id={tile_id} {'SOLID' if tile_id > 0 else 'CLEAR'}")
        else:
            print(f"  Angle {angle:4.2f}: pos ({check_x:5.1f}, {check_y:5.1f}) -> OUT OF BOUNDS")
    
    # Create entity type mapping using the EntityType constants
    entity_type_names = {
        EntityType.NINJA: "NINJA",
        EntityType.TOGGLE_MINE: "TOGGLE_MINE", 
        EntityType.GOLD: "GOLD",
        EntityType.EXIT_DOOR: "EXIT_DOOR",
        EntityType.EXIT_SWITCH: "EXIT_SWITCH",
        EntityType.REGULAR_DOOR: "REGULAR_DOOR",
        EntityType.LOCKED_DOOR: "LOCKED_DOOR",
        EntityType.TRAP_DOOR: "TRAP_DOOR",
        EntityType.LAUNCH_PAD: "LAUNCH_PAD",
        EntityType.ONE_WAY: "ONE_WAY",
        EntityType.DRONE_ZAP: "DRONE_ZAP",
        EntityType.BOUNCE_BLOCK: "BOUNCE_BLOCK",
        EntityType.THWUMP: "THWUMP",
        EntityType.TOGGLE_MINE_TOGGLED: "TOGGLE_MINE_TOGGLED",
        EntityType.BOOST_PAD: "BOOST_PAD",
        EntityType.DEATH_BALL: "DEATH_BALL",
        EntityType.MINI_DRONE: "MINI_DRONE",
        EntityType.SHWUMP: "SHWUMP",
        # Additional types that might exist but aren't in EntityType class
        7: "LOCKED_SWITCH",
        9: "TRAP_SWITCH", 
        21: "LOCKED_DOOR_SWITCH",
        22: "LOCKED_DOOR_DOOR",
    }

    # Find entities
    print(f"\n=== Entities in map ===")
    print(f"Total entities: {len(level_data.entities)}")
    
    # Check if we're missing entities by looking at the raw level data
    print(f"Level data attributes: {dir(level_data)}")
    
    # Track unknown entity types
    unknown_types = set()
    
    for i, entity in enumerate(level_data.entities):
        entity_type_id = entity.get('type', -1)
        entity_type_name = entity_type_names.get(entity_type_id, f"UNKNOWN_{entity_type_id}")
        entity_x = entity.get('x', 0)
        entity_y = entity.get('y', 0)
        print(f"Entity {i}: type={entity_type_name} ({entity_type_id}), pos=({entity_x}, {entity_y})")
        
        # Track unknown types
        if entity_type_id not in entity_type_names:
            unknown_types.add(entity_type_id)
            print(f"  *** UNKNOWN ENTITY TYPE: {entity_type_id} ***")
            print(f"  Entity keys: {list(entity.keys())}")
        
        # Calculate distance from ninja to entity
        dx = entity_x - ninja_pos[0]
        dy = entity_y - ninja_pos[1]
        distance = (dx*dx + dy*dy)**0.5
        print(f"  Distance from ninja: {distance:.1f}px")
        
        # Check if ninja can reach this entity (considering ninja radius)
        # For switches, we need to check if ninja circle can overlap with entity
        if 'SWITCH' in entity_type_name:
            # Assuming entity has some trigger radius (need to check this)
            entity_radius = 12  # Typical switch radius, need to verify
            can_reach = distance <= (NINJA_RADIUS + entity_radius)  # ninja radius + entity radius
            print(f"  Can reach (with {NINJA_RADIUS}px ninja radius): {can_reach}")
    
    if unknown_types:
        print(f"\n*** Found unknown entity types: {sorted(unknown_types)} ***")
        print("These might be the missing switch entities!")
    
    # Check if there are other entity sources
    if hasattr(level_data, 'objects'):
        print(f"\nLevel data objects: {len(level_data.objects) if level_data.objects else 0}")
    if hasattr(level_data, 'switches'):
        print(f"Level data switches: {len(level_data.switches) if level_data.switches else 0}")
    if hasattr(level_data, 'doors'):
        print(f"Level data doors: {len(level_data.doors) if level_data.doors else 0}")
    
    # Build graph with current implementation
    print(f"\n=== Building graph with current implementation ===")
    graph_builder = HierarchicalGraphBuilder()
    hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
    graph_data = hierarchical_data.sub_cell_graph
    
    print(f"Graph nodes: {graph_data.num_nodes}")
    print(f"Graph edges: {graph_data.num_edges}")
    
    # Try pathfinding to leftmost switch
    print(f"\n=== Testing pathfinding to leftmost switch ===")
    
    # Extract switch positions from door entities
    print(f"\n=== Extracting switches from door entities ===")
    switches = []
    
    # Look for EXIT_SWITCH entities
    for entity in level_data.entities:
        if entity.get('type') == EntityType.EXIT_SWITCH:
            switches.append({
                'type': 'EXIT_SWITCH',
                'x': entity.get('x'),
                'y': entity.get('y'),
                'entity': entity
            })
    
    # Look for LOCKED_DOOR and TRAP_DOOR entities and extract their switch positions
    for entity in level_data.entities:
        entity_type = entity.get('type')
        if entity_type == EntityType.LOCKED_DOOR:
            # The switch coordinates should be in the entity data
            # Based on entity_factory.py, they're at index + 6 and index + 7 in map_data
            # But in the entity dict, they might be stored differently
            print(f"Found LOCKED_DOOR at ({entity.get('x')}, {entity.get('y')})")
            print(f"  Entity keys: {list(entity.keys())}")
            
            # Check if this is the door part or switch part
            is_door_part = entity.get('is_door_part', True)
            print(f"  is_door_part: {is_door_part}")
            
            if not is_door_part:
                # This is the switch part - the x,y coordinates are the switch position
                switch_x = entity.get('x')
                switch_y = entity.get('y')
                print(f"  Found switch part at ({switch_x}, {switch_y})")
            else:
                # This is the door part - need to find the corresponding switch
                # The switch should be at the entity's original position
                entity_ref = entity.get('entity_ref')
                if entity_ref and hasattr(entity_ref, 'sw_xpos') and hasattr(entity_ref, 'sw_ypos'):
                    switch_x = entity_ref.sw_xpos
                    switch_y = entity_ref.sw_ypos
                    print(f"  Found switch coordinates from entity_ref: ({switch_x}, {switch_y})")
                else:
                    print(f"  Could not find switch coordinates for door part")
                    continue
                
            switches.append({
                'type': 'LOCKED_DOOR_SWITCH',
                'x': switch_x,
                'y': switch_y,
                'door_entity': entity
            })
            
        elif entity_type == EntityType.TRAP_DOOR:
            print(f"Found TRAP_DOOR at ({entity.get('x')}, {entity.get('y')})")
            print(f"  Entity keys: {list(entity.keys())}")
            
            # Check if this is the door part or switch part
            is_door_part = entity.get('is_door_part', True)
            print(f"  is_door_part: {is_door_part}")
            
            if not is_door_part:
                # This is the switch part - the x,y coordinates are the switch position
                switch_x = entity.get('x')
                switch_y = entity.get('y')
                print(f"  Found switch part at ({switch_x}, {switch_y})")
            else:
                # This is the door part - need to find the corresponding switch
                # The switch should be at the entity's original position
                entity_ref = entity.get('entity_ref')
                if entity_ref and hasattr(entity_ref, 'sw_xpos') and hasattr(entity_ref, 'sw_ypos'):
                    switch_x = entity_ref.sw_xpos
                    switch_y = entity_ref.sw_ypos
                    print(f"  Found switch coordinates from entity_ref: ({switch_x}, {switch_y})")
                else:
                    print(f"  Could not find switch coordinates for door part")
                    continue
                
            switches.append({
                'type': 'TRAP_DOOR_SWITCH',
                'x': switch_x,
                'y': switch_y,
                'door_entity': entity
            })
    
    print(f"Found {len(switches)} switches total")
    for i, switch in enumerate(switches):
        print(f"Switch {i}: type={switch['type']}, pos=({switch['x']}, {switch['y']})")
        
        # Calculate distance from ninja to switch
        dx = switch['x'] - ninja_pos[0]
        dy = switch['y'] - ninja_pos[1]
        distance = (dx*dx + dy*dy)**0.5
        print(f"  Distance from ninja: {distance:.1f}px")
        
        # Check if ninja can reach this switch (considering ninja radius)
        entity_radius = 12  # Typical switch radius
        can_reach = distance <= (NINJA_RADIUS + entity_radius)
        print(f"  Can reach (with {NINJA_RADIUS}px ninja radius): {can_reach}")
    
    # Find leftmost switch
    if switches:
        leftmost_switch = min(switches, key=lambda s: s['x'])
        print(f"\nLeftmost switch: type={leftmost_switch['type']}, pos=({leftmost_switch['x']}, {leftmost_switch['y']})")
        
        # Find ninja node and switch node in graph
        from nclone.graph.visualization_api import GraphVisualizationAPI
        api = GraphVisualizationAPI()
        
        ninja_node = api._find_closest_node(graph_data, ninja_pos)
        switch_node = api._find_closest_node(graph_data, (leftmost_switch['x'], leftmost_switch['y']))
        
        print(f"Ninja node: {ninja_node}")
        print(f"Switch node: {switch_node}")
        
        if ninja_node is not None and switch_node is not None:
            # Try pathfinding
            pathfinding_engine = PathfindingEngine(level_data, level_data.entities)
            path_result = pathfinding_engine.find_shortest_path(graph_data, ninja_node, switch_node, PathfindingAlgorithm.A_STAR)
            
            if path_result and path_result.path:
                print(f"✅ Path found: {len(path_result.path)} nodes, cost: {path_result.cost:.1f}")
            else:
                print(f"❌ No path found from ninja to leftmost switch")
                
                # Debug why no path exists
                print(f"\n=== Debugging path failure ===")
                
                # Check if nodes are in same connected component
                visited = set()
                queue = [ninja_node]
                visited.add(ninja_node)
                
                while queue:
                    current = queue.pop(0)
                    for edge_idx in range(graph_data.num_edges):
                        if (graph_data.edge_index[0, edge_idx] == current and 
                            graph_data.edge_index[1, edge_idx] not in visited):
                            visited.add(graph_data.edge_index[1, edge_idx])
                            queue.append(graph_data.edge_index[1, edge_idx])
                        elif (graph_data.edge_index[1, edge_idx] == current and 
                              graph_data.edge_index[0, edge_idx] not in visited):
                            visited.add(graph_data.edge_index[0, edge_idx])
                            queue.append(graph_data.edge_index[0, edge_idx])
                
                if switch_node in visited:
                    print(f"✅ Switch node is reachable from ninja node (connected component)")
                else:
                    print(f"❌ Switch node is NOT reachable from ninja node (disconnected)")
                    print(f"Ninja component size: {len(visited)} nodes")
        else:
            print(f"❌ Could not find nodes in graph")
    else:
        print(f"❌ No switches found in map")

if __name__ == "__main__":
    analyze_doortest_map()