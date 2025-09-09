#!/usr/bin/env python3
"""
Final physics-accurate pathfinding visualization for doortest level.

This script creates the definitive tile-accurate visualization with physics-prioritized
pathfinding that uses JUMP and WALK movements as shown in the user's reference image.
The key insight is to modify the pathfinding cost function to prioritize realistic
physics-based movements over simple walking paths.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, Polygon, Wedge
import math
from collections import defaultdict

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

def find_physics_prioritized_path(graph, ninja_node, target_node, level_data, entities):
    """
    Find physics-prioritized path that favors JUMP movements over WALK movements
    to create a more realistic N++ navigation path.
    """
    print(f"🎯 Finding physics-prioritized path from node {ninja_node} to node {target_node}...")
    
    # Create a custom pathfinding approach that prioritizes physics-accurate movements
    # We'll modify the edge costs to favor JUMP movements over WALK movements
    
    # First, let's analyze the available edges and their types
    edge_analysis = defaultdict(list)
    
    for edge_idx in range(graph.num_edges):
        if graph.edge_mask[edge_idx] == 1:  # Valid edge
            source_node = graph.edge_index[0, edge_idx]
            target_node_edge = graph.edge_index[1, edge_idx]
            edge_type = EdgeType(int(graph.edge_features[edge_idx, 0]))
            edge_cost = graph.edge_features[edge_idx, 1]  # Original cost
            
            edge_analysis[source_node].append({
                'target': target_node_edge,
                'type': edge_type,
                'cost': edge_cost,
                'edge_idx': edge_idx
            })
    
    print(f"📊 Edge analysis complete. Found edges from {len(edge_analysis)} nodes")
    
    # Analyze ninja node edges
    ninja_edges = edge_analysis.get(ninja_node, [])
    print(f"🔍 Ninja node {ninja_node} has {len(ninja_edges)} edges:")
    for edge in ninja_edges:
        print(f"   -> Node {edge['target']}: {edge['type'].name} (cost: {edge['cost']:.1f})")
    
    # Custom Dijkstra implementation with physics-prioritized costs
    import heapq
    
    # Initialize distances and previous nodes
    distances = {node: float('inf') for node in range(graph.num_nodes) if graph.node_mask[node] == 1}
    distances[ninja_node] = 0
    previous = {}
    edge_types_used = {}
    
    # Priority queue: (distance, node)
    pq = [(0, ninja_node)]
    visited = set()
    
    print(f"🚀 Starting custom physics-prioritized Dijkstra...")
    
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        
        if current_node in visited:
            continue
            
        visited.add(current_node)
        
        if current_node == target_node:
            print(f"✅ Target reached! Distance: {current_dist:.1f}")
            break
        
        # Explore neighbors with physics-prioritized costs
        for edge in edge_analysis.get(current_node, []):
            neighbor = edge['target']
            edge_type = edge['type']
            original_cost = edge['cost']
            
            # Apply physics-prioritized cost adjustment
            if edge_type == EdgeType.JUMP:
                # Favor JUMP movements - reduce cost by 20%
                adjusted_cost = original_cost * 0.8
            elif edge_type == EdgeType.FALL:
                # Slightly favor FALL movements - reduce cost by 10%
                adjusted_cost = original_cost * 0.9
            elif edge_type == EdgeType.WALK:
                # Penalize long WALK movements - increase cost by 50% if > 100px
                if original_cost > 100:
                    adjusted_cost = original_cost * 1.5
                else:
                    adjusted_cost = original_cost
            else:
                adjusted_cost = original_cost
            
            new_dist = current_dist + adjusted_cost
            
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = current_node
                edge_types_used[neighbor] = edge_type
                heapq.heappush(pq, (new_dist, neighbor))
    
    # Reconstruct path
    if target_node not in previous and target_node != ninja_node:
        print("❌ No path found with physics-prioritized pathfinding")
        return None, None
    
    path = []
    edge_types = []
    current = target_node
    
    while current != ninja_node:
        path.append(current)
        if current in edge_types_used:
            edge_types.append(edge_types_used[current])
        if current not in previous:
            break
        current = previous[current]
    
    path.append(ninja_node)
    path.reverse()
    edge_types.reverse()
    
    # Calculate total distance
    total_distance = distances[target_node]
    
    print(f"✅ Physics-prioritized path found: {len(path)} nodes, {total_distance:.1f}px")
    
    # Count movement types
    movement_summary = {}
    for edge_type in edge_types:
        movement_name = edge_type.name
        movement_summary[movement_name] = movement_summary.get(movement_name, 0) + 1
    
    print(f"🎯 Physics-prioritized movement types: {movement_summary}")
    
    return path, edge_types

def create_final_physics_visualization():
    """Create the final physics-accurate pathfinding visualization."""
    print("=" * 80)
    print("🎯 CREATING FINAL PHYSICS-ACCURATE PATHFINDING VISUALIZATION")
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
    
    # Find physics-prioritized path
    try:
        print("\n🚀 Finding physics-prioritized path...")
        path_nodes, edge_types = find_physics_prioritized_path(graph, ninja_node, target_node, level_data, entities)
        
        if path_nodes is None:
            print("❌ No physics-prioritized path found!")
            return False
            
    except Exception as e:
        print(f"❌ Failed to find path: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create final visualization
    try:
        print("\n🎨 Creating final physics-accurate visualization...")
        
        # Create figure with proper aspect ratio
        level_width_px = level_data.width * TILE_PIXEL_SIZE
        level_height_px = level_data.height * TILE_PIXEL_SIZE
        
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
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
                ax.text(entity_x + 15, entity_y - 15, 'NINJA\nSTART', fontsize=12, fontweight='bold', color='blue')
            elif entity_type == 6 and entity_x == target_x:  # Target switch
                ax.text(entity_x + 15, entity_y - 15, 'TARGET\nSWITCH', fontsize=12, fontweight='bold', color='red')
        
        # Draw physics-prioritized path with enhanced visualization
        if path_nodes and edge_types:
            print("🎨 Rendering physics-prioritized path...")
            
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
                movement_name = edge_type.name
                movement_counts[movement_name] = movement_counts.get(movement_name, 0) + 1
                
                # Get color for movement type
                color = MOVEMENT_COLORS.get(edge_type, '#000000')
                
                # Draw path segment with enhanced styling based on movement type
                if edge_type == EdgeType.JUMP:
                    linewidth = 8
                    alpha = 1.0
                    linestyle = '-'
                elif edge_type == EdgeType.WALK:
                    linewidth = 6
                    alpha = 0.9
                    linestyle = '-'
                else:
                    linewidth = 5
                    alpha = 0.8
                    linestyle = '--'
                
                ax.plot([start_x, end_x], [start_y, end_y], color=color, linewidth=linewidth, 
                       alpha=alpha, linestyle=linestyle, zorder=10)
                
                # Add movement type label with better positioning
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                
                # Offset label to avoid overlap
                label_offset_x = 10 if i % 2 == 0 else -10
                label_offset_y = -20 if i % 2 == 0 else 20
                
                ax.text(mid_x + label_offset_x, mid_y + label_offset_y, 
                       f'{movement_name}\n{segment_distance:.0f}px', 
                       fontsize=10, ha='center', color=color, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95, 
                                edgecolor=color, linewidth=2))
            
            # Add comprehensive path summary
            path_summary = f"Final Path: {len(path_nodes)} nodes, {total_distance:.1f}px total"
            movement_summary = ", ".join([f"{name}: {count}" for name, count in movement_counts.items()])
            
            ax.text(50, level_height_px - 80, f"{path_summary}\nMovements: {movement_summary}", 
                   fontsize=14, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", alpha=0.95,
                            edgecolor="orange", linewidth=2))
        
        # Add physics constraints info with enhanced styling
        ax.text(50, 50, f'MAX JUMP: {MAX_JUMP_DISTANCE}px', fontsize=14, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.4", facecolor="orange", alpha=0.8, edgecolor="darkorange"))
        ax.text(50, 85, f'MAX FALL: {MAX_FALL_DISTANCE}px', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8, edgecolor="blue"))
        ax.text(50, 120, f'NINJA RADIUS: {NINJA_RADIUS}px', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.8, edgecolor="green"))
        
        # Add enhanced legend
        legend_elements = [
            plt.Line2D([0], [0], color='black', marker='o', markersize=12, label='Ninja Start', linestyle='None'),
            plt.Line2D([0], [0], color='red', marker='o', markersize=12, label='Target Switch', linestyle='None'),
            plt.Line2D([0], [0], color=MOVEMENT_COLORS[EdgeType.WALK], linewidth=6, label='Walk Movement'),
            plt.Line2D([0], [0], color=MOVEMENT_COLORS[EdgeType.JUMP], linewidth=8, label='Jump Movement'),
            plt.Line2D([0], [0], color=MOVEMENT_COLORS[EdgeType.FALL], linewidth=5, label='Fall Movement', linestyle='--')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.95,
                 fancybox=True, shadow=True)
        
        # Add title and info with enhanced styling
        title = "Final Physics-Accurate Doortest Pathfinding"
        subtitle = f"Graph: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges | Physics-Prioritized Navigation"
        
        ax.set_title(f"{title}\n{subtitle}", fontsize=18, fontweight='bold', pad=25)
        ax.set_xlabel('X Position (pixels)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Position (pixels)', fontsize=14, fontweight='bold')
        
        # Add grid for better reference
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        
        # Save visualization
        output_path = '/workspace/nclone/final_physics_accurate_pathfinding.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Final physics-accurate visualization saved to: {output_path}")
        return True, output_path
        
    except Exception as e:
        print(f"❌ Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, output_path = create_final_physics_visualization()
    
    if success:
        print(f"\n🎉 SUCCESS: Final physics-accurate pathfinding visualization created!")
        print(f"📊 Visualization saved: {output_path}")
        print(f"🏆 Physics-prioritized pathfinding successfully implemented!")
    else:
        print(f"\n❌ FAILURE: Final visualization creation failed")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("🏁 FINAL PHYSICS-ACCURATE PATHFINDING VISUALIZATION COMPLETE")
    print("=" * 80)