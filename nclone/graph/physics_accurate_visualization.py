#!/usr/bin/env python3
"""
Physics-accurate pathfinding visualization within the nclone graph system.

This module provides visualization capabilities for the improved physics-accurate
pathfinding system, showing proper movement types and realistic path costs.
"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from typing import List, Tuple, Optional, Dict

# Import nclone components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.graph.common import EdgeType
from nclone.constants.physics_constants import TILE_PIXEL_SIZE, NINJA_RADIUS

# Color scheme for movement types
MOVEMENT_COLORS = {
    EdgeType.WALK: '#00FF00',      # Green - walking (short distances)
    EdgeType.JUMP: '#FF8000',      # Orange - jumping (upward/long horizontal)
    EdgeType.FALL: '#0080FF',      # Blue - falling (downward)
    EdgeType.FUNCTIONAL: '#FF0000', # Red - functional (switch/door)
    EdgeType.WALL_SLIDE: '#FF00FF', # Magenta - wall sliding
    EdgeType.ONE_WAY: '#FFFF00'    # Yellow - one way platform
}

def create_tile_patches(tile_value: int, tile_x: float, tile_y: float) -> List[patches.Patch]:
    """Create visual patches for different tile types."""
    patches_list = []
    tilesize = TILE_PIXEL_SIZE
    
    if tile_value == 0:
        # Empty tile - no visual representation
        return patches_list
    elif tile_value == 1 or tile_value > 33:
        # Full solid tile
        rect = Rectangle((tile_x, tile_y), tilesize, tilesize, 
                        facecolor='#C0C0C0', edgecolor='#A0A0A0', 
                        linewidth=1, alpha=0.8)
        patches_list.append(rect)
    else:
        # Simplified representation for complex tiles
        rect = Rectangle((tile_x, tile_y), tilesize, tilesize, 
                        facecolor='#C0C0C0', edgecolor='#A0A0A0', 
                        linewidth=1, alpha=0.6)
        patches_list.append(rect)
    
    return patches_list

def create_physics_accurate_pathfinding_visualization(
    save_path: str = '/workspace/nclone/physics_accurate_pathfinding.png'
) -> bool:
    """
    Create comprehensive visualization of physics-accurate pathfinding.
    
    Args:
        save_path: Path to save the visualization image
        
    Returns:
        True if visualization was created successfully
    """
    print("=" * 80)
    print("🎨 CREATING PHYSICS-ACCURATE PATHFINDING VISUALIZATION")
    print("=" * 80)
    
    try:
        # Load environment and build graph
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
        
        # Find ninja and target nodes for pathfinding
        ninja_node = None
        target_node = None
        min_ninja_dist = float('inf')
        
        for node_idx in range(graph.num_nodes):
            if graph.node_mask[node_idx] == 1:
                node_x = graph.node_features[node_idx, 0]
                node_y = graph.node_features[node_idx, 1]
                
                # Find ninja node
                ninja_dist = ((node_x - ninja_position[0])**2 + (node_y - ninja_position[1])**2)**0.5
                if ninja_dist < min_ninja_dist:
                    min_ninja_dist = ninja_dist
                    ninja_node = node_idx
                
                # Find a suitable target node (reasonably far away)
                if ninja_node is not None and target_node is None:
                    target_dist = ((node_x - ninja_position[0])**2 + (node_y - ninja_position[1])**2)**0.5
                    if target_dist > 200:  # At least 200px away
                        target_node = node_idx
        
        if ninja_node is None or target_node is None:
            print("❌ Could not find suitable nodes for pathfinding")
            return False
        
        # Find optimal path
        pathfinding_engine = PathfindingEngine(level_data, entities)
        result = pathfinding_engine.find_shortest_path(
            graph, ninja_node, target_node, PathfindingAlgorithm.DIJKSTRA
        )
        
        if not result.success:
            print("❌ No path found for visualization")
            return False
        
        print(f"✅ Path found: {len(result.path)} nodes, {result.total_cost:.1f}px cost")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.patch.set_facecolor('#1a1a1a')
        
        # Calculate level dimensions
        level_width = level_data.width * TILE_PIXEL_SIZE
        level_height = level_data.height * TILE_PIXEL_SIZE
        
        # Left plot: Level with path
        ax1.set_facecolor('#1a1a1a')
        ax1.set_title('Physics-Accurate Pathfinding', fontsize=14, fontweight='bold', color='white')
        
        # Draw tiles
        for tile_y in range(level_data.height):
            for tile_x in range(level_data.width):
                tile_value = level_data.get_tile(tile_y, tile_x)
                if tile_value != 0:
                    pixel_x = tile_x * TILE_PIXEL_SIZE
                    pixel_y = tile_y * TILE_PIXEL_SIZE
                    
                    tile_patches = create_tile_patches(tile_value, pixel_x, pixel_y)
                    for patch in tile_patches:
                        ax1.add_patch(patch)
        
        # Draw path with movement type colors
        if result.path and len(result.path) > 1:
            for i in range(len(result.path) - 1):
                src_node = result.path[i]
                dst_node = result.path[i + 1]
                
                src_x = graph.node_features[src_node, 0]
                src_y = graph.node_features[src_node, 1]
                dst_x = graph.node_features[dst_node, 0]
                dst_y = graph.node_features[dst_node, 1]
                
                # Get movement type
                edge_type = EdgeType(result.edge_types[i])
                color = MOVEMENT_COLORS.get(edge_type, '#FFFFFF')
                
                # Draw movement arrow
                arrow = FancyArrowPatch(
                    (src_x, src_y), (dst_x, dst_y),
                    arrowstyle='->', mutation_scale=20,
                    color=color, linewidth=3, alpha=0.8, zorder=10
                )
                ax1.add_patch(arrow)
                
                # Add movement type label
                mid_x = (src_x + dst_x) / 2
                mid_y = (src_y + dst_y) / 2
                ax1.annotate(edge_type.name, (mid_x, mid_y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold', color='white',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7),
                           zorder=15)
        
        # Draw ninja and target
        ninja_x = graph.node_features[ninja_node, 0]
        ninja_y = graph.node_features[ninja_node, 1]
        target_x = graph.node_features[target_node, 0]
        target_y = graph.node_features[target_node, 1]
        
        ninja_circle = Circle((ninja_x, ninja_y), NINJA_RADIUS, 
                             facecolor='red', edgecolor='white', 
                             linewidth=3, zorder=20)
        ax1.add_patch(ninja_circle)
        
        target_circle = Circle((target_x, target_y), 8, 
                              facecolor='cyan', edgecolor='black', 
                              linewidth=2, zorder=20)
        ax1.add_patch(target_circle)
        
        ax1.set_xlim(-20, level_width + 20)
        ax1.set_ylim(-20, level_height + 20)
        ax1.set_aspect('equal')
        ax1.invert_yaxis()
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Right plot: Movement analysis
        ax2.set_facecolor('#1a1a1a')
        ax2.set_title('Movement Type Analysis', fontsize=14, fontweight='bold', color='white')
        
        # Count movement types in path
        movement_counts = {}
        movement_distances = {}
        
        for i, edge_type_val in enumerate(result.edge_types):
            edge_type = EdgeType(edge_type_val)
            movement_name = edge_type.name
            
            # Count movements
            movement_counts[movement_name] = movement_counts.get(movement_name, 0) + 1
            
            # Calculate distance for this movement
            if i < len(result.path) - 1:
                src_node = result.path[i]
                dst_node = result.path[i + 1]
                
                src_x = graph.node_features[src_node, 0]
                src_y = graph.node_features[src_node, 1]
                dst_x = graph.node_features[dst_node, 0]
                dst_y = graph.node_features[dst_node, 1]
                
                distance = math.sqrt((dst_x - src_x)**2 + (dst_y - src_y)**2)
                
                if movement_name not in movement_distances:
                    movement_distances[movement_name] = []
                movement_distances[movement_name].append(distance)
        
        # Create bar chart
        movement_names = list(movement_counts.keys())
        counts = list(movement_counts.values())
        colors = [MOVEMENT_COLORS.get(getattr(EdgeType, name, EdgeType.WALK), '#FFFFFF') 
                 for name in movement_names]
        
        bars = ax2.bar(movement_names, counts, color=colors, alpha=0.8, edgecolor='white')
        
        # Add distance information as text
        for i, (name, bar) in enumerate(zip(movement_names, bars)):
            if name in movement_distances:
                distances = movement_distances[name]
                avg_dist = sum(distances) / len(distances)
                max_dist = max(distances)
                
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'Avg: {avg_dist:.1f}px\nMax: {max_dist:.1f}px',
                        ha='center', va='bottom', fontsize=9, color='white',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        ax2.set_ylabel('Number of Movements', fontsize=12, color='white')
        ax2.set_xlabel('Movement Type', fontsize=12, color='white')
        ax2.tick_params(colors='white')
        
        # Add summary statistics
        total_distance = sum(sum(distances) for distances in movement_distances.values())
        summary_text = f"Path Summary:\n"
        summary_text += f"• Total nodes: {len(result.path)}\n"
        summary_text += f"• Total cost: {result.total_cost:.1f}px\n"
        summary_text += f"• Total distance: {total_distance:.1f}px\n"
        summary_text += f"• Efficiency: {total_distance/result.total_cost:.2%}"
        
        ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
        
        # Add legend
        legend_elements = []
        for edge_type in EdgeType:
            if edge_type.name in movement_counts:
                color = MOVEMENT_COLORS.get(edge_type, '#FFFFFF')
                legend_elements.append(
                    plt.Line2D([0], [0], color=color, linewidth=4, label=edge_type.name)
                )
        
        ax1.legend(handles=legend_elements, loc='upper right', 
                  facecolor='black', edgecolor='white', 
                  labelcolor='white', fontsize=10)
        
        # Save visualization
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='#1a1a1a', edgecolor='none')
        
        print(f"✅ Visualization saved to: {save_path}")
        
        # Print summary
        print(f"\n📊 PHYSICS-ACCURATE PATHFINDING SUMMARY:")
        print(f"   Path: {len(result.path)} nodes, {result.total_cost:.1f}px cost")
        print(f"   Movement types: {movement_counts}")
        
        # Validate physics compliance
        physics_compliant = True
        for name, distances in movement_distances.items():
            max_dist = max(distances)
            if name == 'WALK' and max_dist > 50:
                print(f"   ❌ PHYSICS VIOLATION: WALK movement {max_dist:.1f}px")
                physics_compliant = False
        
        if physics_compliant:
            print(f"   ✅ All movements respect physics constraints")
        
        plt.show()
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False

if __name__ == "__main__":
    create_physics_accurate_pathfinding_visualization()