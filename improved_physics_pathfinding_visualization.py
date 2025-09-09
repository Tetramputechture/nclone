#!/usr/bin/env python3
"""
Improved physics-accurate pathfinding visualization for doortest level.

This script creates a tile-accurate visualization with proper physics-based pathfinding
that uses JUMP and FALL movements as shown in the user's reference image.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, Polygon, Wedge
import math

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.constants.entity_types import EntityType
from nclone.constants.physics_constants import (
    TILE_PIXEL_SIZE, NINJA_RADIUS, MAP_PADDING, 
    MAX_JUMP_DISTANCE, MAX_FALL_DISTANCE, MAX_HOR_SPEED
)
from nclone.graph.common import EdgeType

# Movement type colors matching the user's reference image
MOVEMENT_COLORS = {
    EdgeType.WALK: '#0080FF',      # Blue - walking (matches user image)
    EdgeType.JUMP: '#FF8000',      # Orange - jumping (matches user image)  
    EdgeType.FALL: '#0080FF',      # Blue - falling
    EdgeType.WALL_SLIDE: '#FF00FF', # Magenta - wall sliding
    EdgeType.ONE_WAY: '#FFFF00',   # Yellow - one way platform
    EdgeType.FUNCTIONAL: '#FF0000'  # Red - functional (switch/door)
}

def create_accurate_tile_patches(tile_value, tile_x, tile_y):
    """Create tile patches using exact game tile rendering logic."""
    patches_list = []
    tilesize = TILE_PIXEL_SIZE  # 24 pixels
    
    if tile_value == 0:
        # Empty tile - no visual representation
        return patches_list
    elif tile_value == 1 or tile_value > 33:
        # Full solid tile
        rect = Rectangle((tile_x, tile_y), tilesize, tilesize, 
                        facecolor='#C0C0C0', edgecolor='#808080', linewidth=0.5)
        patches_list.append(rect)
    elif tile_value < 6:
        # Half tiles
        dx = tilesize / 2 if tile_value == 3 else 0
        dy = tilesize / 2 if tile_value == 4 else 0
        w = tilesize if tile_value % 2 == 0 else tilesize / 2
        h = tilesize / 2 if tile_value % 2 == 0 else tilesize
        
        rect = Rectangle((tile_x + dx, tile_y + dy), w, h, 
                        facecolor='#C0C0C0', edgecolor='#808080', linewidth=0.5)
        patches_list.append(rect)
    else:
        # Complex tiles - simplified for visualization
        if tile_value < 10:
            # 45-degree slopes
            dx1 = 0
            dy1 = tilesize if tile_value == 8 else 0
            dx2 = 0 if tile_value == 9 else tilesize
            dy2 = tilesize if tile_value == 9 else 0
            dx3 = 0 if tile_value == 6 else tilesize
            dy3 = tilesize
            
            triangle = Polygon([(tile_x + dx1, tile_y + dy1),
                              (tile_x + dx2, tile_y + dy2),
                              (tile_x + dx3, tile_y + dy3)], 
                             facecolor='#C0C0C0', edgecolor='#808080', linewidth=0.5)
            patches_list.append(triangle)
        else:
            # Other complex tiles - simplified as full tiles
            rect = Rectangle((tile_x, tile_y), tilesize, tilesize, 
                            facecolor='#C0C0C0', edgecolor='#808080', linewidth=0.5)
            patches_list.append(rect)
    
    return patches_list

def create_entity_representation(entity_type, x, y):
    """Create visual representation of entities."""
    if entity_type == 0:  # Ninja
        return Circle((x, y), NINJA_RADIUS, facecolor='black', edgecolor='white', linewidth=2, zorder=15)
    elif entity_type == 4:  # Exit switch
        return Circle((x, y), 8, facecolor='#FFD700', edgecolor='black', linewidth=2, zorder=12)
    elif entity_type == 3:  # Exit door
        return Circle((x, y), 8, facecolor='#FFA500', edgecolor='black', linewidth=2, zorder=12)
    elif entity_type == 6:  # Locked door switch
        return Circle((x, y), 8, facecolor='black', edgecolor='white', linewidth=2, zorder=12)
    elif entity_type == 8:  # Some other entity
        return Rectangle((x-8, y-8), 16, 16, facecolor='magenta', edgecolor='black', linewidth=2, zorder=12)
    elif entity_type == 11:  # Some other entity
        return Rectangle((x-12, y-6), 24, 12, facecolor='#00FF00', edgecolor='black', linewidth=2, zorder=12)
    else:
        return Circle((x, y), 5, facecolor='gray', edgecolor='black', linewidth=1, zorder=10)

def find_physics_based_path_with_analysis(graph, ninja_node, target_node, level_data, entities):
    """Find physics-based path with detailed analysis of available edge types."""
    print(f"🔍 Analyzing available edge types from ninja node {ninja_node}...")
    
    # Analyze edges from ninja node
    ninja_edges = []
    for edge_idx in range(graph.num_edges):
        if graph.edge_mask[edge_idx] == 1:  # Valid edge
            source_node = graph.edge_index[0, edge_idx]
            target_node_edge = graph.edge_index[1, edge_idx]
            edge_type = graph.edge_features[edge_idx, 0]  # Edge type is first feature
            
            if source_node == ninja_node:
                ninja_edges.append((target_node_edge, EdgeType(int(edge_type))))
    
    print(f"📊 Found {len(ninja_edges)} edges from ninja node:")
    edge_type_counts = {}
    for _, edge_type in ninja_edges:
        edge_type_counts[edge_type.name] = edge_type_counts.get(edge_type.name, 0) + 1
    
    for edge_type_name, count in sorted(edge_type_counts.items()):
        print(f"   {edge_type_name}: {count} edges")
    
    # Try different pathfinding algorithms
    pathfinding_engine = PathfindingEngine(level_data, entities)
    
    print(f"\n🚀 Trying different pathfinding algorithms...")
    
    # Try A* first
    print("🔍 Trying A* algorithm...")
    result_astar = pathfinding_engine.find_shortest_path(
        graph, ninja_node, target_node, PathfindingAlgorithm.A_STAR
    )
    
    if result_astar.success:
        print(f"✅ A* found path: {len(result_astar.path)} nodes, {result_astar.total_cost:.1f}px")
        movement_summary = {}
        for edge_type in result_astar.edge_types:
            movement_name = EdgeType(edge_type).name
            movement_summary[movement_name] = movement_summary.get(movement_name, 0) + 1
        print(f"🎯 A* Movement types: {movement_summary}")
    else:
        print("❌ A* failed")
    
    # Try Dijkstra
    print("\n🔍 Trying Dijkstra algorithm...")
    result_dijkstra = pathfinding_engine.find_shortest_path(
        graph, ninja_node, target_node, PathfindingAlgorithm.DIJKSTRA
    )
    
    if result_dijkstra.success:
        print(f"✅ Dijkstra found path: {len(result_dijkstra.path)} nodes, {result_dijkstra.total_cost:.1f}px")
        movement_summary = {}
        for edge_type in result_dijkstra.edge_types:
            movement_name = EdgeType(edge_type).name
            movement_summary[movement_name] = movement_summary.get(movement_name, 0) + 1
        print(f"🎯 Dijkstra Movement types: {movement_summary}")
    else:
        print("❌ Dijkstra failed")
    
    # Choose the best result (prefer one with more movement diversity)
    best_result = None
    if result_astar.success and result_dijkstra.success:
        # Count unique movement types
        astar_types = len(set(result_astar.edge_types))
        dijkstra_types = len(set(result_dijkstra.edge_types))
        
        if astar_types >= dijkstra_types:
            best_result = result_astar
            print(f"\n✅ Selected A* path (more diverse movements: {astar_types} types)")
        else:
            best_result = result_dijkstra
            print(f"\n✅ Selected Dijkstra path (more diverse movements: {dijkstra_types} types)")
    elif result_astar.success:
        best_result = result_astar
        print(f"\n✅ Selected A* path (only successful option)")
    elif result_dijkstra.success:
        best_result = result_dijkstra
        print(f"\n✅ Selected Dijkstra path (only successful option)")
    else:
        print(f"\n❌ No path found with either algorithm")
        return None, None
    
    return best_result.path, best_result.edge_types

def create_improved_physics_visualization():
    """Create improved physics-accurate pathfinding visualization."""
    print("=" * 80)
    print("🎯 CREATING IMPROVED PHYSICS-ACCURATE PATHFINDING VISUALIZATION")
    print("=" * 80)
    
    # Load doortest environment
    try:
        print("📁 Loading doortest environment...")
        env = BasicLevelNoGold(render_mode="rgb_array")
        env.reset()
        ninja_position = env.nplay_headless.ninja_position()
        level_data = env.level_data
        entities = level_data.entities
        
        print(f"✅ Ninja position: {ninja_position}")
        print(f"✅ Level size: {level_data.width}x{level_data.height} tiles")
        print(f"✅ Found {len(entities)} entities")
        
    except Exception as e:
        print(f"❌ Failed to load environment: {e}")
        return False
    
    # Build hierarchical graph with physics enhancement
    try:
        print("\n🔧 Building hierarchical graph with physics enhancement...")
        builder = HierarchicalGraphBuilder()
        hierarchical_graph = builder.build_graph(level_data, ninja_position)
        graph = hierarchical_graph.sub_cell_graph
        print(f"✅ Graph built: {graph.num_nodes} nodes, {graph.num_edges} edges")
    except Exception as e:
        print(f"❌ Failed to build graph: {e}")
        return False
    
    # Find ninja and target nodes
    try:
        print("\n🔍 Finding ninja and target nodes...")
        
        # Find ninja node (closest to ninja position)
        ninja_node = None
        min_ninja_dist = float('inf')
        
        for node_idx in range(graph.num_nodes):
            if graph.node_mask[node_idx] == 1:
                node_x = graph.node_features[node_idx, 0]
                node_y = graph.node_features[node_idx, 1]
                dist = ((node_x - ninja_position[0])**2 + (node_y - ninja_position[1])**2)**0.5
                if dist < min_ninja_dist:
                    min_ninja_dist = dist
                    ninja_node = node_idx
        
        ninja_coords = (graph.node_features[ninja_node, 0], graph.node_features[ninja_node, 1])
        print(f"✅ Ninja node: {ninja_node} at {ninja_coords}")
        
        # Find leftmost locked door switch (type 6)
        locked_door_switches = []
        for entity in entities:
            if entity.get("type") == 6:  # Locked door switch
                entity_x = entity.get("x", 0)
                entity_y = entity.get("y", 0)
                locked_door_switches.append((entity_x, entity_y))
        
        if not locked_door_switches:
            print("❌ No locked door switches found!")
            return False
        
        leftmost_switch = min(locked_door_switches, key=lambda pos: pos[0])
        target_x, target_y = leftmost_switch
        print(f"🎯 Targeting leftmost locked door switch at ({target_x}, {target_y})")
        
        # Find closest graph node to the leftmost switch
        target_node = None
        min_target_dist = float('inf')
        
        for node_idx in range(graph.num_nodes):
            if graph.node_mask[node_idx] == 1:
                node_x = graph.node_features[node_idx, 0]
                node_y = graph.node_features[node_idx, 1]
                dist = ((node_x - target_x)**2 + (node_y - target_y)**2)**0.5
                if dist < min_target_dist:
                    min_target_dist = dist
                    target_node = node_idx
        
        target_coords = (graph.node_features[target_node, 0], graph.node_features[target_node, 1])
        print(f"✅ Target node: {target_node} at {target_coords}")
        
    except Exception as e:
        print(f"❌ Failed to find nodes: {e}")
        return False
    
    # Find physics-accurate path with analysis
    try:
        print("\n🚀 Finding physics-accurate path with detailed analysis...")
        path_nodes, edge_types = find_physics_based_path_with_analysis(graph, ninja_node, target_node, level_data, entities)
        
        if path_nodes is None:
            print("❌ No path found!")
            return False
            
    except Exception as e:
        print(f"❌ Failed to find path: {e}")
        return False
    
    # Create visualization
    try:
        print("\n🎨 Creating improved tile-accurate visualization...")
        
        # Create figure with proper aspect ratio
        level_width_px = level_data.width * TILE_PIXEL_SIZE
        level_height_px = level_data.height * TILE_PIXEL_SIZE
        
        fig, ax = plt.subplots(1, 1, figsize=(18, 14))
        ax.set_xlim(0, level_width_px)
        ax.set_ylim(0, level_height_px)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Match game coordinates
        
        # Draw tiles
        print("🎨 Rendering tiles...")
        for row in range(level_data.height):
            for col in range(level_data.width):
                tile_value = level_data.tiles[row][col]
                if tile_value != 0:  # Skip empty tiles
                    tile_x = col * TILE_PIXEL_SIZE
                    tile_y = row * TILE_PIXEL_SIZE
                    
                    tile_patches = create_accurate_tile_patches(tile_value, tile_x, tile_y)
                    for patch in tile_patches:
                        ax.add_patch(patch)
        
        # Draw entities
        print("🎨 Rendering entities...")
        for entity in entities:
            entity_type = entity.get("type", 0)
            entity_x = entity.get("x", 0)
            entity_y = entity.get("y", 0)
            
            entity_patch = create_entity_representation(entity_type, entity_x, entity_y)
            ax.add_patch(entity_patch)
            
            # Label important entities
            if entity_type == 0:  # Ninja
                ax.text(entity_x + 15, entity_y - 15, 'NINJA\nSTART', fontsize=10, fontweight='bold', color='blue')
            elif entity_type == 6 and entity_x == target_x:  # Target switch
                ax.text(entity_x + 15, entity_y - 15, 'TARGET\nSWITCH', fontsize=10, fontweight='bold', color='red')
        
        # Draw path with enhanced visualization
        if path_nodes and edge_types:
            print("🎨 Rendering physics-accurate path...")
            
            total_distance = 0
            movement_counts = {}
            
            for i in range(len(path_nodes) - 1):
                start_node = path_nodes[i]
                end_node = path_nodes[i + 1]
                edge_type = edge_types[i]
                
                start_x = graph.node_features[start_node, 0]
                start_y = graph.node_features[start_node, 1]
                end_x = graph.node_features[end_node, 0]
                end_y = graph.node_features[end_node, 1]
                
                # Calculate segment distance
                segment_distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                total_distance += segment_distance
                
                # Count movement types
                movement_name = EdgeType(edge_type).name
                movement_counts[movement_name] = movement_counts.get(movement_name, 0) + 1
                
                # Get color for movement type
                color = MOVEMENT_COLORS.get(edge_type, '#000000')
                
                # Draw path segment with enhanced styling
                linewidth = 6 if edge_type in [EdgeType.WALK, EdgeType.JUMP] else 4
                alpha = 0.9 if edge_type in [EdgeType.WALK, EdgeType.JUMP] else 0.7
                
                ax.plot([start_x, end_x], [start_y, end_y], color=color, linewidth=linewidth, alpha=alpha, zorder=10)
                
                # Add movement type label with better positioning
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                
                # Offset label to avoid overlap
                label_offset_x = 0
                label_offset_y = -15 if i % 2 == 0 else 15
                
                ax.text(mid_x + label_offset_x, mid_y + label_offset_y, 
                       f'{movement_name}\n{segment_distance:.0f}px', 
                       fontsize=9, ha='center', color=color, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor=color))
            
            # Add path summary
            path_summary = f"Path: {len(path_nodes)} nodes, {total_distance:.1f}px total"
            movement_summary = ", ".join([f"{name}: {count}" for name, count in movement_counts.items()])
            
            ax.text(50, level_height_px - 50, f"{path_summary}\nMovements: {movement_summary}", 
                   fontsize=12, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
        
        # Add physics constraints info
        ax.text(50, 50, f'MAX JUMP: {MAX_JUMP_DISTANCE}px', fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
        ax.text(50, 80, f'MAX FALL: {MAX_FALL_DISTANCE}px', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax.text(50, 110, f'NINJA RADIUS: {NINJA_RADIUS}px', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        # Add enhanced legend
        legend_elements = [
            plt.Line2D([0], [0], color='black', marker='o', markersize=10, label='Ninja Start', linestyle='None'),
            plt.Line2D([0], [0], color='red', marker='o', markersize=10, label='Target Switch', linestyle='None'),
            plt.Line2D([0], [0], color=MOVEMENT_COLORS[EdgeType.WALK], linewidth=6, label='Walk Movement'),
            plt.Line2D([0], [0], color=MOVEMENT_COLORS[EdgeType.JUMP], linewidth=6, label='Jump Movement'),
            plt.Line2D([0], [0], color=MOVEMENT_COLORS[EdgeType.FALL], linewidth=4, label='Fall Movement')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
        
        # Add title and info
        title = "Improved Physics-Accurate Doortest Pathfinding"
        subtitle = f"Graph: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges"
        
        ax.set_title(f"{title}\n{subtitle}", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Position (pixels)', fontsize=12)
        ax.set_ylabel('Y Position (pixels)', fontsize=12)
        
        # Add grid for better reference
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Save visualization
        output_path = '/workspace/nclone/improved_physics_pathfinding_visualization.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Improved visualization saved to: {output_path}")
        return True, output_path
        
    except Exception as e:
        print(f"❌ Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, output_path = create_improved_physics_visualization()
    
    if success:
        print(f"\n🎉 SUCCESS: Improved physics pathfinding visualization created!")
        print(f"📊 Visualization saved: {output_path}")
    else:
        print(f"\n❌ FAILURE: Visualization creation failed")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("🏁 IMPROVED PHYSICS PATHFINDING VISUALIZATION COMPLETE")
    print("=" * 80)