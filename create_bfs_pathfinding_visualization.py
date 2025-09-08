#!/usr/bin/env python3
"""
Create pathfinding visualization using BFS (since A* is having issues).

This script creates a comprehensive visualization showing:
1. The doortest map with proper level geometry
2. A valid BFS path from ninja starting position to leftmost locked door switch
3. Directional arrows showing the path route
4. Clear visual representation matching the reference image
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from collections import deque

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.constants.entity_types import EntityType
from nclone.constants.physics_constants import TILE_PIXEL_SIZE

def find_bfs_path(graph, ninja_node, target_node):
    """Find path using BFS algorithm."""
    # Build adjacency list
    adjacency = {}
    for i in range(graph.num_nodes):
        if graph.node_mask[i] > 0:
            adjacency[i] = []
    
    for i in range(graph.num_edges):
        if graph.edge_mask[i] > 0:
            src = graph.edge_index[0, i]
            dst = graph.edge_index[1, i]
            if src in adjacency and dst in adjacency:
                adjacency[src].append(dst)
                adjacency[dst].append(src)  # Undirected graph
    
    # BFS to find path
    visited = set()
    queue = deque([ninja_node])
    visited.add(ninja_node)
    path_parent = {ninja_node: None}
    
    while queue:
        current = queue.popleft()
        
        if current == target_node:
            # Reconstruct path
            path = []
            node = target_node
            while node is not None:
                path.append(node)
                node = path_parent.get(node)
            path.reverse()
            return path
        
        # Explore neighbors
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                path_parent[neighbor] = current
    
    return None  # No path found

def create_pathfinding_visualization():
    """Create comprehensive pathfinding visualization using BFS."""
    print("=" * 80)
    print("🎯 CREATING BFS PATHFINDING VISUALIZATION")
    print("=" * 80)
    
    # Create environment and load doortest map
    try:
        print("📁 Loading doortest map...")
        env = BasicLevelNoGold(
            render_mode="rgb_array",
            enable_frame_stack=False,
            enable_debug_overlay=False,
            eval_mode=False,
            seed=42
        )
        env.reset()
        ninja_position = env.nplay_headless.ninja_position()
        level_data = env.level_data
        print(f"✅ Ninja position: {ninja_position}")
        print(f"✅ Level size: {level_data.width}x{level_data.height} tiles")
            
    except Exception as e:
        print(f"❌ Failed to load environment: {e}")
        return False
    
    # Build graph
    try:
        print("\n🔧 Building hierarchical graph...")
        builder = HierarchicalGraphBuilder()
        hierarchical_graph = builder.build_graph(level_data, ninja_position)
        graph = hierarchical_graph.sub_cell_graph
        print(f"✅ Graph built: {graph.num_nodes} nodes, {graph.num_edges} edges")
    except Exception as e:
        print(f"❌ Failed to build graph: {e}")
        return False
    
    # Find ninja node and leftmost locked door switch
    try:
        print("\n🔍 Finding ninja and target nodes...")
        
        # Find ninja node (closest to ninja position)
        ninja_node = None
        min_ninja_dist = float('inf')
        
        for node_idx in range(graph.num_nodes):
            if graph.node_mask[node_idx] == 1:  # Valid node
                node_x = graph.node_features[node_idx, 0]
                node_y = graph.node_features[node_idx, 1]
                dist = ((node_x - ninja_position[0])**2 + (node_y - ninja_position[1])**2)**0.5
                if dist < min_ninja_dist:
                    min_ninja_dist = dist
                    ninja_node = node_idx
        
        ninja_coords = (graph.node_features[ninja_node, 0], graph.node_features[ninja_node, 1])
        print(f"✅ Ninja node: {ninja_node} at {ninja_coords}")
        
        # Find leftmost locked door switch
        locked_door_switches = []
        for entity in level_data.entities:
            if entity.get("type") == EntityType.LOCKED_DOOR:  # Type 6
                entity_x = entity.get("x", 0)
                entity_y = entity.get("y", 0)
                locked_door_switches.append((entity_x, entity_y))
        
        leftmost_switch = min(locked_door_switches, key=lambda pos: pos[0])
        target_x, target_y = leftmost_switch
        print(f"🎯 Targeting leftmost locked door switch at ({target_x}, {target_y})")
        
        # Find closest graph node to the leftmost switch
        target_node = None
        min_target_dist = float('inf')
        
        for node_idx in range(graph.num_nodes):
            if graph.node_mask[node_idx] == 1:  # Valid node
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
    
    # Find path using BFS
    try:
        print("\n🚀 Finding path using BFS algorithm...")
        path = find_bfs_path(graph, ninja_node, target_node)
        
        if not path:
            print(f"❌ BFS pathfinding failed!")
            return False
        
        print(f"🎉 SUCCESS! Found BFS path:")
        print(f"   Path length: {len(path)} nodes")
        
        # Extract path coordinates
        path_coords = []
        for node_idx in path:
            if graph.node_mask[node_idx] == 1:  # Valid node
                node_x = graph.node_features[node_idx, 0]
                node_y = graph.node_features[node_idx, 1]
                path_coords.append((node_x, node_y))
        
        print(f"✅ Extracted {len(path_coords)} path coordinates")
        print("Path coordinates:")
        for i, (x, y) in enumerate(path_coords):
            print(f"  {i+1}. ({x:.1f}, {y:.1f})")
        
    except Exception as e:
        print(f"❌ BFS pathfinding error: {e}")
        return False
    
    # Create visualization
    try:
        print("\n🎨 Creating visualization...")
        
        fig, ax = plt.subplots(1, 1, figsize=(20, 14))
        
        # Set up coordinate system
        map_width = level_data.width * TILE_PIXEL_SIZE
        map_height = level_data.height * TILE_PIXEL_SIZE
        
        ax.set_xlim(0, map_width)
        ax.set_ylim(0, map_height)
        ax.invert_yaxis()  # Match reference image coordinate system
        
        # Draw level geometry (simplified)
        print("🏗️ Drawing level geometry...")
        for tile_y in range(level_data.height):
            for tile_x in range(level_data.width):
                tile_value = level_data.tiles[tile_y, tile_x]
                if tile_value != 0:  # Solid tile
                    pixel_x = tile_x * TILE_PIXEL_SIZE
                    pixel_y = tile_y * TILE_PIXEL_SIZE
                    
                    # Draw as light gray rectangle (solid geometry)
                    rect = Rectangle(
                        (pixel_x, pixel_y), TILE_PIXEL_SIZE, TILE_PIXEL_SIZE,
                        facecolor='lightgray', edgecolor='gray', linewidth=0.5, alpha=0.8
                    )
                    ax.add_patch(rect)
        
        # Draw entities
        print("🎯 Drawing entities...")
        for entity in level_data.entities:
            entity_type = entity.get("type", 0)
            entity_x = entity.get("x", 0)
            entity_y = entity.get("y", 0)
            
            if entity_type == EntityType.LOCKED_DOOR:
                # Locked door switch - black circle
                circle = Circle((entity_x, entity_y), 8, facecolor='black', edgecolor='white', linewidth=2, zorder=10)
                ax.add_patch(circle)
            elif entity_type == EntityType.EXIT_SWITCH:
                # Exit switch - yellow circle
                circle = Circle((entity_x, entity_y), 6, facecolor='yellow', edgecolor='black', linewidth=1, zorder=9)
                ax.add_patch(circle)
            elif entity_type == EntityType.ONE_WAY:
                # One way platform - green rectangle
                rect = Rectangle((entity_x-8, entity_y-4), 16, 8, facecolor='green', edgecolor='black', linewidth=1, zorder=9)
                ax.add_patch(rect)
            else:
                # Other entities - small gray circle
                circle = Circle((entity_x, entity_y), 4, facecolor='gray', edgecolor='black', linewidth=1, zorder=8)
                ax.add_patch(circle)
        
        # Draw ninja starting position
        ninja_circle = Circle(ninja_coords, 12, facecolor='black', edgecolor='white', linewidth=3, zorder=12)
        ax.add_patch(ninja_circle)
        
        # Add ninja label
        ax.text(ninja_coords[0] - 20, ninja_coords[1] - 30, 'ninja', 
                fontsize=14, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        # Draw target position
        target_circle = Circle((target_x, target_y), 12, facecolor='black', edgecolor='white', linewidth=3, zorder=12)
        ax.add_patch(target_circle)
        
        # Add target label
        ax.text(target_x - 50, target_y - 30, 'locked door switch', 
                fontsize=14, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        # Draw path with directional arrows
        print("➡️ Drawing path with arrows...")
        if len(path_coords) >= 2:
            # Draw path line
            path_xs = [coord[0] for coord in path_coords]
            path_ys = [coord[1] for coord in path_coords]
            ax.plot(path_xs, path_ys, 'r-', linewidth=6, alpha=0.8, zorder=11)
            
            # Add directional arrows along the path
            for i in range(len(path_coords) - 1):
                start_x, start_y = path_coords[i]
                end_x, end_y = path_coords[i + 1]
                
                # Create arrow
                arrow = FancyArrowPatch(
                    (start_x, start_y), (end_x, end_y),
                    arrowstyle='->', mutation_scale=30, 
                    color='red', linewidth=4, zorder=11
                )
                ax.add_patch(arrow)
            
            # Add waypoint markers
            for i, (x, y) in enumerate(path_coords[1:-1], 1):  # Skip start and end
                waypoint = Circle((x, y), 6, facecolor='red', edgecolor='white', linewidth=2, zorder=11)
                ax.add_patch(waypoint)
        
        # Add title and labels
        ax.set_title(f'Pathfinding Visualization: Ninja → Leftmost Locked Door Switch\n'
                    f'BFS Path: {len(path)} nodes, Direct Distance: {((target_x - ninja_coords[0])**2 + (target_y - ninja_coords[1])**2)**0.5:.1f}px',
                    fontsize=18, fontweight='bold', pad=20)
        
        ax.set_xlabel('X Position (pixels)', fontsize=16)
        ax.set_ylabel('Y Position (pixels)', fontsize=16)
        
        # Add grid for reference
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add info box
        info_text = f"""Map: doortest ({level_data.width}×{level_data.height} tiles)
Ninja: ({ninja_coords[0]:.0f}, {ninja_coords[1]:.0f})
Target: ({target_x}, {target_y})
Path Length: {len(path)} nodes
Algorithm: BFS (Breadth-First Search)
Graph: {graph.num_nodes} nodes, {graph.num_edges} edges
Direct Distance: {((target_x - ninja_coords[0])**2 + (target_y - ninja_coords[1])**2)**0.5:.1f}px"""
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = "/workspace/nclone/pathfinding_visualization.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved pathfinding visualization to: {output_path}")
        
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    success = create_pathfinding_visualization()
    
    if success:
        print("\n" + "=" * 80)
        print("🎉 SUCCESS! BFS Pathfinding visualization created successfully!")
        print("✅ Valid path found from ninja to leftmost locked door switch")
        print("✅ Visualization saved as pathfinding_visualization.png")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ FAILED! Could not create pathfinding visualization")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())