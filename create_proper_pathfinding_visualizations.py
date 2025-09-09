#!/usr/bin/env python3
"""
Create pathfinding visualizations with accurate tile rendering and path overlays.

This script creates visual validation of the physics-aware pathfinding system
using the proper environment loading system and accurate tile rendering.
"""

import os
import sys
import numpy as np
import cairo
import math
from typing import Dict, List, Tuple, Optional

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.constants.entity_types import EntityType
from nclone.constants.physics_constants import TILE_PIXEL_SIZE, MAP_PADDING
from nclone.graph.common import EdgeType
# Define colors directly
BGCOLOR_RGB = (0.1, 0.1, 0.1)  # Dark gray background
TILECOLOR_RGB = (0.75, 0.75, 0.75)  # Light gray tiles

# Movement type colors
MOVEMENT_COLORS = {
    EdgeType.WALK: (0, 1, 0),      # Green - walking
    EdgeType.JUMP: (1, 0.5, 0),   # Orange - jumping  
    EdgeType.FALL: (0, 0.5, 1),   # Blue - falling
    EdgeType.WALL_SLIDE: (1, 0, 1), # Magenta - wall sliding
    EdgeType.ONE_WAY: (1, 1, 0),   # Yellow - one way platform
    EdgeType.FUNCTIONAL: (1, 0, 0)  # Red - functional (switch/door)
}

class PathfindingVisualizer:
    """Creates pathfinding visualizations with accurate tile rendering."""
    
    def __init__(self):
        self.tile_size = TILE_PIXEL_SIZE  # 24 pixels per tile
        
    def load_test_map(self, map_path: str):
        """Load a test map using the proper environment system."""
        print(f"Loading test map: {map_path}")
        
        # Create environment with custom map path
        env = BasicLevelNoGold(
            render_mode="rgb_array",
            enable_frame_stack=False,
            enable_debug_overlay=False,
            eval_mode=False,
            seed=42,
            custom_map_path=map_path
        )
        
        # Reset to load the map
        env.reset()
        
        # Get level data and ninja position
        level_data = env.level_data
        ninja_position = env.nplay_headless.ninja_position()
        
        print(f"✅ Map loaded: {level_data.width}x{level_data.height} tiles")
        print(f"✅ Ninja position: {ninja_position}")
        print(f"✅ Found {len(level_data.entities)} entities")
        
        # Verify entities are in empty tiles (accounting for 1-tile padding)
        for i, entity in enumerate(level_data.entities):
            entity_type = entity.get("type", 0)
            entity_x = entity.get("x", 0)
            entity_y = entity.get("y", 0)
            
            # Convert to tile coordinates and account for 1-tile padding offset
            tile_x = int(entity_x // TILE_PIXEL_SIZE) - 1
            tile_y = int(entity_y // TILE_PIXEL_SIZE) - 1
            
            # Check if entity is in an empty tile (type 0)
            if 0 <= tile_y < level_data.height and 0 <= tile_x < level_data.width:
                tile_type = level_data.tiles[tile_y, tile_x]
                print(f"Entity {i}: type={entity_type}, pos=({entity_x}, {entity_y}), tile=({tile_x}, {tile_y}), tile_type={tile_type}")
                
                if tile_type != 0:
                    print(f"⚠️  WARNING: Entity {i} is NOT in an empty tile (tile_type={tile_type})")
                else:
                    print(f"✅ Entity {i} is correctly placed in empty tile")
            else:
                print(f"⚠️  WARNING: Entity {i} is outside map bounds")
        
        env.close()
        return level_data, ninja_position
    
    def draw_accurate_tile(self, ctx, tile_type: int, tile_x: int, tile_y: int):
        """Draw a tile using accurate rendering logic from TileRenderer."""
        if tile_type == 0:
            return  # Empty tile
            
        # Convert tile coordinates to pixel coordinates
        x = tile_x * self.tile_size
        y = tile_y * self.tile_size
        
        if tile_type == 1 or tile_type > 33:
            # Full solid tile
            ctx.rectangle(x, y, self.tile_size, self.tile_size)
            ctx.fill()
        elif tile_type < 6:
            # Half tiles
            dx = self.tile_size / 2 if tile_type == 3 else 0
            dy = self.tile_size / 2 if tile_type == 4 else 0
            w = self.tile_size if tile_type % 2 == 0 else self.tile_size / 2
            h = self.tile_size / 2 if tile_type % 2 == 0 else self.tile_size
            
            ctx.rectangle(x + dx, y + dy, w, h)
            ctx.fill()
        elif tile_type < 10:
            # 45-degree slopes
            dx1 = 0
            dy1 = self.tile_size if tile_type == 8 else 0
            dx2 = 0 if tile_type == 9 else self.tile_size
            dy2 = self.tile_size if tile_type == 9 else 0
            dx3 = 0 if tile_type == 6 else self.tile_size
            dy3 = self.tile_size
            
            ctx.move_to(x + dx1, y + dy1)
            ctx.line_to(x + dx2, y + dy2)
            ctx.line_to(x + dx3, y + dy3)
            ctx.close_path()
            ctx.fill()
        elif tile_type < 14:
            # Quarter circles
            dx = self.tile_size if (tile_type == 11 or tile_type == 12) else 0
            dy = self.tile_size if (tile_type == 12 or tile_type == 13) else 0
            a1 = (math.pi / 2) * (tile_type - 10)
            a2 = (math.pi / 2) * (tile_type - 9)
            
            ctx.arc(x + dx, y + dy, self.tile_size, a1, a2)
            ctx.line_to(x + dx, y + dy)
            ctx.close_path()
            ctx.fill()
        else:
            # Other complex tiles - simplified as rectangles for now
            ctx.rectangle(x, y, self.tile_size, self.tile_size)
            ctx.fill()
    
    def create_visualization(self, map_path: str, output_path: str):
        """Create a complete pathfinding visualization for a test map."""
        print(f"\nCreating visualization for {map_path}")
        
        # Load the map using proper environment system
        try:
            level_data, ninja_position = self.load_test_map(map_path)
        except Exception as e:
            print(f"Error loading map {map_path}: {e}")
            return
        
        # Calculate canvas dimensions
        width = level_data.width
        height = level_data.height
        padding = 2
        canvas_width = (width + 2 * padding) * self.tile_size
        canvas_height = (height + 2 * padding) * self.tile_size
        
        # Create Cairo surface and context
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, canvas_width, canvas_height)
        ctx = cairo.Context(surface)
        
        # Fill background
        ctx.set_source_rgb(*BGCOLOR_RGB)
        ctx.paint()
        
        # Draw tiles with accurate rendering
        ctx.set_source_rgb(*TILECOLOR_RGB)
        for y in range(height):
            for x in range(width):
                tile_type = level_data.tiles[y, x]
                if tile_type != 0:  # Skip empty tiles
                    adjusted_x = x + padding
                    adjusted_y = y + padding
                    self.draw_accurate_tile(ctx, tile_type, adjusted_x, adjusted_y)
        
        # Draw entities
        for entity in level_data.entities:
            entity_type = entity.get("type", 0)
            entity_x = entity.get("x", 0)
            entity_y = entity.get("y", 0)
            
            # Move entities left and down by 24px (1 tile) to account for padding
            corrected_x = entity_x - TILE_PIXEL_SIZE  # Move left by 24px
            corrected_y = entity_y - TILE_PIXEL_SIZE  # Move down by 24px
            
            # Convert to canvas coordinates directly
            adj_x = (corrected_x / TILE_PIXEL_SIZE + padding) * self.tile_size + self.tile_size / 2
            adj_y = (corrected_y / TILE_PIXEL_SIZE + padding) * self.tile_size + self.tile_size / 2
            
            # Draw entity based on type
            if entity_type == EntityType.NINJA:
                ctx.set_source_rgb(0, 0, 0)  # Black
                ctx.arc(adj_x, adj_y, 8, 0, 2 * math.pi)
                ctx.fill()
                ctx.set_source_rgb(1, 1, 1)  # White outline
                ctx.arc(adj_x, adj_y, 8, 0, 2 * math.pi)
                ctx.set_line_width(2)
                ctx.stroke()
            elif entity_type == EntityType.EXIT_SWITCH:
                ctx.set_source_rgb(1, 0.84, 0)  # Gold
                ctx.arc(adj_x, adj_y, 6, 0, 2 * math.pi)
                ctx.fill()
                ctx.set_source_rgb(0, 0, 0)  # Black outline
                ctx.arc(adj_x, adj_y, 6, 0, 2 * math.pi)
                ctx.set_line_width(1)
                ctx.stroke()
            elif entity_type == EntityType.EXIT_DOOR:
                ctx.set_source_rgb(1, 0.65, 0)  # Orange
                ctx.arc(adj_x, adj_y, 6, 0, 2 * math.pi)
                ctx.fill()
                ctx.set_source_rgb(0, 0, 0)  # Black outline
                ctx.arc(adj_x, adj_y, 6, 0, 2 * math.pi)
                ctx.set_line_width(1)
                ctx.stroke()
        
        # Build graph and find path
        try:
            print("Building graph...")
            builder = HierarchicalGraphBuilder()
            graph_data = builder.build_graph(level_data, ninja_position)
            
            # Find entity positions for pathfinding
            ninja_entity_pos = None
            switch_entity_pos = None
            exit_entity_pos = None
            
            for entity in level_data.entities:
                entity_type = entity.get("type", 0)
                entity_x = entity.get("x", 0)
                entity_y = entity.get("y", 0)
                
                if entity_type == EntityType.NINJA:
                    ninja_entity_pos = (entity_x, entity_y)
                elif entity_type == EntityType.EXIT_SWITCH:
                    switch_entity_pos = (entity_x, entity_y)
                elif entity_type == EntityType.EXIT_DOOR:
                    exit_entity_pos = (entity_x, entity_y)
            
            if not ninja_entity_pos or not exit_entity_pos:
                print(f"Could not find ninja or exit positions in {map_path}")
                return
            
            # Find closest nodes to entity positions
            def find_closest_node(pos, graph_data):
                min_dist = float('inf')
                closest_node = None
                
                # Use the sub_cell_graph from hierarchical data
                sub_graph = graph_data.sub_cell_graph
                
                for node_idx in range(sub_graph.num_nodes):
                    if sub_graph.node_mask[node_idx] == 1:  # Valid node
                        node_x = sub_graph.node_features[node_idx, 0]
                        node_y = sub_graph.node_features[node_idx, 1]
                        dist = ((node_x - pos[0])**2 + (node_y - pos[1])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            closest_node = node_idx
                return closest_node
            
            ninja_node = find_closest_node(ninja_entity_pos, graph_data)
            exit_node = find_closest_node(exit_entity_pos, graph_data)
            switch_node = find_closest_node(switch_entity_pos, graph_data) if switch_entity_pos else None
            
            # Create pathfinding engine
            pathfinder = PathfindingEngine(level_data, level_data.entities)
            algorithm = PathfindingAlgorithm.DIJKSTRA
            
            # Find paths
            paths_to_draw = []
            
            if switch_node:
                # Path from ninja to switch
                result1 = pathfinder.find_shortest_path(
                    graph_data=graph_data.sub_cell_graph,
                    start_node=ninja_node,
                    goal_node=switch_node,
                    algorithm=algorithm
                )
                
                if result1.success:
                    paths_to_draw.append(("Ninja to Switch", result1))
                
                # Path from switch to exit
                result2 = pathfinder.find_shortest_path(
                    graph_data=graph_data.sub_cell_graph,
                    start_node=switch_node,
                    goal_node=exit_node,
                    algorithm=algorithm
                )
                
                if result2.success:
                    paths_to_draw.append(("Switch to Exit", result2))
            else:
                # Direct path from ninja to exit
                result = pathfinder.find_shortest_path(
                    graph_data=graph_data.sub_cell_graph,
                    start_node=ninja_node,
                    goal_node=exit_node,
                    algorithm=algorithm
                )
                
                if result.success:
                    paths_to_draw.append(("Ninja to Exit", result))
            
            # Draw paths
            for path_name, result in paths_to_draw:
                print(f"Drawing {path_name}: {len(result.path_coordinates)} waypoints")
                
                # Draw path segments with movement type colors
                for i in range(len(result.path_coordinates) - 1):
                    start_pos = result.path_coordinates[i]
                    end_pos = result.path_coordinates[i + 1]
                    
                    # Get movement type for this segment
                    edge_type = result.edge_types[i] if i < len(result.edge_types) else EdgeType.WALK
                    color = MOVEMENT_COLORS.get(edge_type, (0.5, 0.5, 0.5))
                    
                    # Move path coordinates left and down by 24px and convert to canvas space
                    corrected_start_x = start_pos[0] - TILE_PIXEL_SIZE
                    corrected_start_y = start_pos[1] - TILE_PIXEL_SIZE
                    corrected_end_x = end_pos[0] - TILE_PIXEL_SIZE
                    corrected_end_y = end_pos[1] - TILE_PIXEL_SIZE
                    
                    start_x = corrected_start_x / TILE_PIXEL_SIZE + padding
                    start_y = corrected_start_y / TILE_PIXEL_SIZE + padding
                    end_x = corrected_end_x / TILE_PIXEL_SIZE + padding
                    end_y = corrected_end_y / TILE_PIXEL_SIZE + padding
                    
                    start_x *= self.tile_size
                    start_y *= self.tile_size
                    end_x *= self.tile_size
                    end_y *= self.tile_size
                    
                    # Draw path segment
                    ctx.set_source_rgb(*color)
                    ctx.set_line_width(3)
                    ctx.move_to(start_x, start_y)
                    ctx.line_to(end_x, end_y)
                    ctx.stroke()
                    
                    # Draw movement type indicator
                    mid_x = (start_x + end_x) / 2
                    mid_y = (start_y + end_y) / 2
                    ctx.set_source_rgb(*color)
                    ctx.arc(mid_x, mid_y, 2, 0, 2 * math.pi)
                    ctx.fill()
                
                # Print movement summary
                movement_summary = {}
                for edge_type in result.edge_types:
                    movement_name = EdgeType(edge_type).name
                    movement_summary[movement_name] = movement_summary.get(movement_name, 0) + 1
                
                print(f"Movement types in {path_name}: {movement_summary}")
            
        except Exception as e:
            print(f"Error finding path for {map_path}: {e}")
        
        # Save the visualization
        surface.write_to_png(output_path)
        print(f"Visualization saved to {output_path}")

def main():
    """Create visualizations for all test maps."""
    print("Creating pathfinding visualizations with accurate tile rendering...")
    print("=" * 60)
    
    visualizer = PathfindingVisualizer()
    
    test_maps = [
        ("nclone/test_maps/simple-walk", "simple_walk_pathfinding.png"),
        ("nclone/test_maps/long-walk", "long_walk_pathfinding.png"),
        ("nclone/test_maps/path-jump-required", "path_jump_required_pathfinding.png"),
        ("nclone/test_maps/only-jump", "only_jump_pathfinding.png"),
    ]
    
    for map_path, output_path in test_maps:
        if os.path.exists(map_path):
            visualizer.create_visualization(map_path, output_path)
        else:
            print(f"Map not found: {map_path}")
    
    print("\nVisualization complete! Check the generated PNG files for visual validation.")

if __name__ == "__main__":
    main()