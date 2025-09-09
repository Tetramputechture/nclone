#!/usr/bin/env python3
"""
Create visual validation images for pathfinding test maps using accurate tile rendering.
"""

import sys
import os
import cairo
import math
import json
import numpy as np
from typing import List, Tuple, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.graph.level_data import LevelData
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.graph.movement_classifier import MovementType
from nclone import render_utils

class PathfindingVisualizer:
    """Creates accurate visual representations of pathfinding results on test maps."""
    
    def __init__(self, tile_size: int = 24):
        self.tile_size = tile_size
        self.pathfinder = PathfindingEngine()
        
        # Colors for different movement types
        self.movement_colors = {
            MovementType.WALK: (0.2, 0.8, 0.2, 0.8),      # Green
            MovementType.JUMP: (0.8, 0.2, 0.2, 0.8),      # Red  
            MovementType.FALL: (0.2, 0.2, 0.8, 0.8),      # Blue
            MovementType.WALL_SLIDE: (0.8, 0.8, 0.2, 0.8), # Yellow
            MovementType.COMBO: (0.8, 0.2, 0.8, 0.8),     # Magenta
        }
        
        # Entity colors
        self.entity_colors = {
            'NINJA': (0.0, 0.0, 0.0, 1.0),           # Black
            'EXIT_SWITCH': (0.8, 0.8, 0.2, 1.0),     # Yellow
            'EXIT_DOOR': (0.2, 0.8, 0.2, 1.0),       # Green
        }

    def load_test_map(self, map_path: str) -> LevelData:
        """Load a test map from binary file using the same logic as the game."""
        # Load binary map data
        with open(map_path, "rb") as map_file:
            map_data = [int(b) for b in map_file.read()]
        
        # Extract tile data (same logic as MapLoader)
        tile_data = map_data[184:1150]
        tiles = {}
        
        # Map each tile to its cell (same as MapLoader.load_map_tiles)
        for x_coord_tile in range(42):
            for y_coord_tile in range(23):
                tile_value = tile_data[x_coord_tile + y_coord_tile * 42]
                if tile_value != 0:  # Only store non-empty tiles
                    tiles[(x_coord_tile + 1, y_coord_tile + 1)] = tile_value
        
        # Set outer edges to tile type 1 (full tile) - same as MapLoader
        for x_coord_tile in range(44):
            tiles[(x_coord_tile, 0)] = 1
            tiles[(x_coord_tile, 24)] = 1
        for y_coord_tile in range(25):
            tiles[(0, y_coord_tile)] = 1
            tiles[(43, y_coord_tile)] = 1
        
        # Extract entity data (entities start at byte 1230, same as MapLoader)
        entities = []
        
        # First, add the ninja from its spawn position (bytes 1150-1151)
        ninja_x = map_data[1150]
        ninja_y = map_data[1151]
        
        from nclone.constants.entity_types import EntityType
        
        class Entity:
            def __init__(self, entity_type_const, entity_type_name, x, y):
                self.entity_type = type('EntityType', (), {'name': entity_type_name, 'value': entity_type_const})()
                self.x = x * 24  # Convert tile coordinates to pixel coordinates
                self.y = y * 24
                
            def get(self, key, default=None):
                """Support dict-like access for compatibility."""
                return getattr(self, key, default)
        
        # Add ninja entity
        entities.append(Entity(EntityType.NINJA, 'NINJA', ninja_x, ninja_y))
        
        # Parse other entities starting at byte 1230 (same as MapLoader.load_map_entities)
        index = 1230
        while index < len(map_data) - 4:
            entity_type_id = map_data[index]
            if entity_type_id == 0:  # Skip empty entity slots
                index += 5
                continue
                
            x = map_data[index + 1]
            y = map_data[index + 2]
            
            # Map entity type IDs to EntityType constants and names
            entity_type_map = {
                3: (EntityType.EXIT_DOOR, 'EXIT_DOOR'),
                4: (EntityType.EXIT_SWITCH, 'EXIT_SWITCH'), 
                # Add more entity types as needed
            }
            
            if entity_type_id in entity_type_map:
                entity_type_const, entity_type_name = entity_type_map[entity_type_id]
                entities.append(Entity(entity_type_const, entity_type_name, x, y))
            
            index += 5
        
        # Convert tiles dictionary to NumPy array (44x25 as per MapLoader)
        tile_array = np.zeros((25, 44), dtype=int)  # height x width
        for (x, y), tile_type in tiles.items():
            if 0 <= x < 44 and 0 <= y < 25:
                tile_array[y, x] = tile_type
        
        return LevelData(tile_array, entities)

    def draw_accurate_tile(self, ctx: cairo.Context, tile_type: int, x: int, y: int):
        """Draw a tile using the exact same logic as the game's tile renderer."""
        tilesize = self.tile_size
        
        if tile_type == 1 or tile_type > 33:
            # Full solid tiles
            ctx.rectangle(x * tilesize, y * tilesize, tilesize, tilesize)
            ctx.fill()
        elif tile_type < 6:
            # Half tiles (2-5)
            dx = tilesize / 2 if tile_type == 3 else 0
            dy = tilesize / 2 if tile_type == 4 else 0
            w = tilesize if tile_type % 2 == 0 else tilesize / 2
            h = tilesize / 2 if tile_type % 2 == 0 else tilesize
            ctx.rectangle(x * tilesize + dx, y * tilesize + dy, w, h)
            ctx.fill()
        else:
            # Complex tiles (6-33)
            self._draw_complex_tile(ctx, tile_type, x, y, tilesize)

    def _draw_complex_tile(self, ctx: cairo.Context, tile_type: int, x: int, y: int, tilesize: int):
        """Draw complex tile shapes using exact game logic."""
        if tile_type < 10:
            # 45-degree slopes (6-9)
            dx1 = 0
            dy1 = tilesize if tile_type == 8 else 0
            dx2 = 0 if tile_type == 9 else tilesize
            dy2 = tilesize if tile_type == 9 else 0
            dx3 = 0 if tile_type == 6 else tilesize
            dy3 = tilesize
            ctx.move_to(x * tilesize + dx1, y * tilesize + dy1)
            ctx.line_to(x * tilesize + dx2, y * tilesize + dy2)
            ctx.line_to(x * tilesize + dx3, y * tilesize + dy3)
            ctx.close_path()
            ctx.fill()
        elif tile_type < 14:
            # Quarter moon curves (10-13)
            dx = tilesize if (tile_type == 11 or tile_type == 12) else 0
            dy = tilesize if (tile_type == 12 or tile_type == 13) else 0
            a1 = (math.pi / 2) * (tile_type - 10)
            a2 = (math.pi / 2) * (tile_type - 9)
            ctx.move_to(x * tilesize + dx, y * tilesize + dy)
            ctx.arc(x * tilesize + dx, y * tilesize + dy, tilesize, a1, a2)
            ctx.close_path()
            ctx.fill()
        elif tile_type < 18:
            # Inverted quarter moon curves (14-17)
            dx1 = tilesize if (tile_type == 15 or tile_type == 16) else 0
            dy1 = tilesize if (tile_type == 16 or tile_type == 17) else 0
            dx2 = tilesize if (tile_type == 14 or tile_type == 17) else 0
            dy2 = tilesize if (tile_type == 14 or tile_type == 15) else 0
            a1 = math.pi + (math.pi / 2) * (tile_type - 10)
            a2 = math.pi + (math.pi / 2) * (tile_type - 9)
            ctx.move_to(x * tilesize + dx1, y * tilesize + dy1)
            ctx.arc(x * tilesize + dx2, y * tilesize + dy2, tilesize, a1, a2)
            ctx.close_path()
            ctx.fill()
        elif tile_type < 22:
            # Steep slopes (18-21)
            dx1 = 0
            dy1 = tilesize if (tile_type == 20 or tile_type == 21) else 0
            dx2 = tilesize
            dy2 = tilesize if (tile_type == 20 or tile_type == 21) else 0
            dx3 = tilesize if (tile_type == 19 or tile_type == 20) else 0
            dy3 = tilesize / 2
            ctx.move_to(x * tilesize + dx1, y * tilesize + dy1)
            ctx.line_to(x * tilesize + dx2, y * tilesize + dy2)
            ctx.line_to(x * tilesize + dx3, y * tilesize + dy3)
            ctx.close_path()
            ctx.fill()
        elif tile_type < 26:
            # Gentle slopes (22-25)
            dx1 = 0
            dy1 = tilesize / 2 if (tile_type == 23 or tile_type == 24) else 0
            dx2 = 0 if tile_type == 23 else tilesize
            dy2 = tilesize / 2 if tile_type == 25 else 0
            dx3 = tilesize
            dy3 = (tilesize / 2 if tile_type == 22 else 0) if tile_type < 24 else tilesize
            dx4 = tilesize if tile_type == 23 else 0
            dy4 = tilesize
            ctx.move_to(x * tilesize + dx1, y * tilesize + dy1)
            ctx.line_to(x * tilesize + dx2, y * tilesize + dy2)
            ctx.line_to(x * tilesize + dx3, y * tilesize + dy3)
            ctx.line_to(x * tilesize + dx4, y * tilesize + dy4)
            ctx.close_path()
            ctx.fill()
        elif tile_type < 30:
            # More slopes (26-29)
            dx1 = tilesize / 2
            dy1 = tilesize if (tile_type == 28 or tile_type == 29) else 0
            dx2 = tilesize if (tile_type == 27 or tile_type == 28) else 0
            dy2 = 0
            dx3 = tilesize if (tile_type == 27 or tile_type == 28) else 0
            dy3 = tilesize
            ctx.move_to(x * tilesize + dx1, y * tilesize + dy1)
            ctx.line_to(x * tilesize + dx2, y * tilesize + dy2)
            ctx.line_to(x * tilesize + dx3, y * tilesize + dy3)
            ctx.close_path()
            ctx.fill()
        elif tile_type < 34:
            # Complex slopes (30-33)
            dx1 = tilesize / 2
            dy1 = tilesize if (tile_type == 30 or tile_type == 31) else 0
            dx2 = tilesize if (tile_type == 31 or tile_type == 33) else 0
            dy2 = tilesize
            dx3 = tilesize if (tile_type == 31 or tile_type == 32) else 0
            dy3 = tilesize if (tile_type == 32 or tile_type == 33) else 0
            dx4 = tilesize if (tile_type == 30 or tile_type == 32) else 0
            dy4 = 0
            ctx.move_to(x * tilesize + dx1, y * tilesize + dy1)
            ctx.line_to(x * tilesize + dx2, y * tilesize + dy2)
            ctx.line_to(x * tilesize + dx3, y * tilesize + dy3)
            ctx.line_to(x * tilesize + dx4, y * tilesize + dy4)
            ctx.close_path()
            ctx.fill()

    def draw_entity(self, ctx: cairo.Context, entity, radius: float = 6.0):
        """Draw an entity with appropriate color and shape."""
        entity_type = entity.entity_type.name
        color = self.entity_colors.get(entity_type, (0.5, 0.5, 0.5, 1.0))
        
        ctx.set_source_rgba(*color)
        
        if entity_type == 'NINJA':
            # Draw ninja as a filled circle
            ctx.arc(entity.x, entity.y, radius, 0, 2 * math.pi)
            ctx.fill()
        elif entity_type == 'EXIT_SWITCH':
            # Draw switch as a diamond
            ctx.move_to(entity.x, entity.y - radius)
            ctx.line_to(entity.x + radius, entity.y)
            ctx.line_to(entity.x, entity.y + radius)
            ctx.line_to(entity.x - radius, entity.y)
            ctx.close_path()
            ctx.fill()
        elif entity_type == 'EXIT_DOOR':
            # Draw door as a rectangle
            ctx.rectangle(entity.x - radius, entity.y - radius, radius * 2, radius * 2)
            ctx.fill()
        else:
            # Default: draw as circle
            ctx.arc(entity.x, entity.y, radius, 0, 2 * math.pi)
            ctx.fill()

    def draw_path_segment(self, ctx: cairo.Context, start_pos: Tuple[float, float], 
                         end_pos: Tuple[float, float], movement_type: MovementType, 
                         segment_index: int):
        """Draw a path segment with appropriate color and style."""
        color = self.movement_colors.get(movement_type, (0.5, 0.5, 0.5, 0.8))
        ctx.set_source_rgba(*color)
        
        # Set line width based on movement type
        line_widths = {
            MovementType.WALK: 3.0,
            MovementType.JUMP: 4.0,
            MovementType.FALL: 2.0,
            MovementType.WALL_SLIDE: 3.0,
            MovementType.COMBO: 5.0,
        }
        ctx.set_line_width(line_widths.get(movement_type, 3.0))
        
        # Draw the path line
        ctx.move_to(start_pos[0], start_pos[1])
        ctx.line_to(end_pos[0], end_pos[1])
        ctx.stroke()
        
        # Add movement type indicators
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2
        
        # Draw movement type symbol
        ctx.set_source_rgba(*color)
        if movement_type == MovementType.JUMP:
            # Draw upward arrow for jumps
            ctx.move_to(mid_x, mid_y + 4)
            ctx.line_to(mid_x - 3, mid_y - 2)
            ctx.line_to(mid_x + 3, mid_y - 2)
            ctx.close_path()
            ctx.fill()
        elif movement_type == MovementType.FALL:
            # Draw downward arrow for falls
            ctx.move_to(mid_x, mid_y - 4)
            ctx.line_to(mid_x - 3, mid_y + 2)
            ctx.line_to(mid_x + 3, mid_y + 2)
            ctx.close_path()
            ctx.fill()
        elif movement_type == MovementType.WALK:
            # Draw horizontal line for walks
            ctx.move_to(mid_x - 4, mid_y)
            ctx.line_to(mid_x + 4, mid_y)
            ctx.set_line_width(2.0)
            ctx.stroke()

    def create_visualization(self, map_path: str, output_path: str, 
                           algorithm: PathfindingAlgorithm = PathfindingAlgorithm.DIJKSTRA):
        """Create a complete visualization of pathfinding results for a test map."""
        print(f"Creating visualization for {map_path}")
        
        # Load the test map
        level_data = self.load_test_map(map_path)
        
        # Calculate map dimensions from NumPy array
        if level_data.tiles.size == 0:
            print(f"No tiles found in {map_path}")
            return
            
        height, width = level_data.tiles.shape
        min_x, max_x = 0, width - 1
        min_y, max_y = 0, height - 1
        
        # Add padding
        padding = 2
        canvas_width = (max_x - min_x + 1 + 2 * padding) * self.tile_size
        canvas_height = (max_y - min_y + 1 + 2 * padding) * self.tile_size
        
        # Create Cairo surface
        surface = cairo.ImageSurface(cairo.Format.ARGB32, canvas_width, canvas_height)
        ctx = cairo.Context(surface)
        
        # Set background color
        ctx.set_source_rgb(*render_utils.BGCOLOR_RGB)
        ctx.paint()
        
        # Draw tiles with accurate rendering
        ctx.set_source_rgb(*render_utils.TILECOLOR_RGB)
        for y in range(height):
            for x in range(width):
                tile_type = level_data.tiles[y, x]
                if tile_type != 0:  # Skip empty tiles
                    adjusted_x = x - min_x + padding
                    adjusted_y = y - min_y + padding
                    self.draw_accurate_tile(ctx, tile_type, adjusted_x, adjusted_y)
        
        # Find ninja and exit positions
        ninja_pos = None
        exit_pos = None
        switch_pos = None
        
        for entity in level_data.entities:
            entity_type = entity.entity_type.name
            # Adjust entity positions to canvas coordinates
            adj_x = (entity.x / self.tile_size - min_x + padding) * self.tile_size + self.tile_size / 2
            adj_y = (entity.y / self.tile_size - min_y + padding) * self.tile_size + self.tile_size / 2
            
            if entity_type == 'NINJA':
                ninja_pos = (adj_x, adj_y)
            elif entity_type == 'EXIT_DOOR':
                exit_pos = (adj_x, adj_y)
            elif entity_type == 'EXIT_SWITCH':
                switch_pos = (adj_x, adj_y)
        
        if not ninja_pos or not exit_pos:
            print(f"Could not find ninja or exit positions in {map_path}")
            return
        
        # Build graph and find path
        try:
            # Get ninja position for graph building
            ninja_entity_pos = None
            for entity in level_data.entities:
                if entity.entity_type.name == 'NINJA':
                    ninja_entity_pos = (entity.x, entity.y)
                    break
            
            if not ninja_entity_pos:
                print(f"No ninja found in {map_path}")
                return
                
            graph_data = self.pathfinder.graph_builder.build_graph(level_data, ninja_entity_pos)
            
            # Find path from ninja to switch (if exists), then to exit
            paths_to_draw = []
            
            # Get actual entity positions for pathfinding
            ninja_entity_pos = None
            switch_entity_pos = None
            exit_entity_pos = None
            
            for entity in level_data.entities:
                if entity.entity_type.name == 'NINJA':
                    ninja_entity_pos = (entity.x, entity.y)
                elif entity.entity_type.name == 'EXIT_SWITCH':
                    switch_entity_pos = (entity.x, entity.y)
                elif entity.entity_type.name == 'EXIT_DOOR':
                    exit_entity_pos = (entity.x, entity.y)
            
            # Find closest nodes to entity positions
            def find_closest_node(pos, graph_data):
                min_dist = float('inf')
                closest_node = None
                for node_id, node_data in graph_data.nodes.items():
                    dist = ((node_data.x - pos[0])**2 + (node_data.y - pos[1])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_node = node_id
                return closest_node
            
            ninja_node = find_closest_node(ninja_entity_pos, graph_data)
            exit_node = find_closest_node(exit_entity_pos, graph_data)
            switch_node = find_closest_node(switch_entity_pos, graph_data) if switch_entity_pos else None
            
            if switch_node:
                # Path from ninja to switch
                result1 = self.pathfinder.find_shortest_path(
                    graph_data=graph_data,
                    start_node=ninja_node,
                    goal_node=switch_node,
                    algorithm=algorithm
                )
                
                if result1.success:
                    paths_to_draw.append(("Ninja to Switch", result1, ninja_pos, switch_pos))
                
                # Path from switch to exit
                result2 = self.pathfinder.find_shortest_path(
                    graph_data=graph_data,
                    start_node=switch_node,
                    goal_node=exit_node,
                    algorithm=algorithm
                )
                
                if result2.success:
                    paths_to_draw.append(("Switch to Exit", result2, switch_pos, exit_pos))
            else:
                # Direct path from ninja to exit
                result = self.pathfinder.find_shortest_path(
                    graph_data=graph_data,
                    start_node=ninja_node,
                    goal_node=exit_node,
                    algorithm=algorithm
                )
                
                if result.success:
                    paths_to_draw.append(("Ninja to Exit", result, ninja_pos, exit_pos))
            
            # Draw paths
            for path_name, result, start_canvas_pos, end_canvas_pos in paths_to_draw:
                print(f"Drawing {path_name}: {len(result.path_coordinates)} waypoints")
                
                # Convert path coordinates to canvas coordinates
                canvas_path = []
                for x, y in result.path_coordinates:
                    canvas_x = (x / self.tile_size - min_x + padding) * self.tile_size + self.tile_size / 2
                    canvas_y = (y / self.tile_size - min_y + padding) * self.tile_size + self.tile_size / 2
                    canvas_path.append((canvas_x, canvas_y))
                
                # Draw path segments
                for i in range(len(canvas_path) - 1):
                    start_pos = canvas_path[i]
                    end_pos = canvas_path[i + 1]
                    
                    # Get movement type for this segment
                    if i < len(result.edge_types):
                        movement_type = MovementType(result.edge_types[i])
                    else:
                        movement_type = MovementType.WALK  # Default
                    
                    self.draw_path_segment(ctx, start_pos, end_pos, movement_type, i)
        
        except Exception as e:
            print(f"Error finding path for {map_path}: {e}")
        
        # Draw entities on top
        for entity in level_data.entities:
            adj_x = (entity.x / self.tile_size - min_x + padding) * self.tile_size + self.tile_size / 2
            adj_y = (entity.y / self.tile_size - min_y + padding) * self.tile_size + self.tile_size / 2
            
            # Create temporary entity with adjusted position
            class AdjustedEntity:
                def __init__(self, original_entity, x, y):
                    self.entity_type = original_entity.entity_type
                    self.x = x
                    self.y = y
            
            adjusted_entity = AdjustedEntity(entity, adj_x, adj_y)
            self.draw_entity(ctx, adjusted_entity)
        
        # Add legend
        self.draw_legend(ctx, canvas_width, canvas_height)
        
        # Save the image
        surface.write_to_png(output_path)
        print(f"Visualization saved to {output_path}")

    def draw_legend(self, ctx: cairo.Context, canvas_width: int, canvas_height: int):
        """Draw a legend explaining the visualization."""
        legend_x = canvas_width - 200
        legend_y = 20
        
        # Legend background
        ctx.set_source_rgba(1.0, 1.0, 1.0, 0.9)
        ctx.rectangle(legend_x - 10, legend_y - 10, 190, 150)
        ctx.fill()
        
        # Legend border
        ctx.set_source_rgba(0.0, 0.0, 0.0, 1.0)
        ctx.rectangle(legend_x - 10, legend_y - 10, 190, 150)
        ctx.set_line_width(1.0)
        ctx.stroke()
        
        # Legend title
        ctx.set_source_rgba(0.0, 0.0, 0.0, 1.0)
        ctx.move_to(legend_x, legend_y)
        ctx.show_text("Movement Types:")
        
        # Movement type legend entries
        y_offset = 20
        for movement_type, color in self.movement_colors.items():
            ctx.set_source_rgba(*color)
            ctx.move_to(legend_x, legend_y + y_offset)
            ctx.line_to(legend_x + 20, legend_y + y_offset)
            ctx.set_line_width(3.0)
            ctx.stroke()
            
            ctx.set_source_rgba(0.0, 0.0, 0.0, 1.0)
            ctx.move_to(legend_x + 25, legend_y + y_offset + 3)
            ctx.show_text(movement_type.name)
            
            y_offset += 15
        
        # Entity legend
        y_offset += 10
        ctx.set_source_rgba(0.0, 0.0, 0.0, 1.0)
        ctx.move_to(legend_x, legend_y + y_offset)
        ctx.show_text("Entities:")
        
        y_offset += 15
        for entity_type, color in self.entity_colors.items():
            ctx.set_source_rgba(*color)
            ctx.arc(legend_x + 10, legend_y + y_offset, 5, 0, 2 * math.pi)
            ctx.fill()
            
            ctx.set_source_rgba(0.0, 0.0, 0.0, 1.0)
            ctx.move_to(legend_x + 20, legend_y + y_offset + 3)
            ctx.show_text(entity_type.replace('_', ' '))
            
            y_offset += 15

def main():
    """Create visualizations for all test maps."""
    visualizer = PathfindingVisualizer()
    
    test_maps = [
        ("nclone/test_maps/simple-walk", "simple_walk_pathfinding.png"),
        ("nclone/test_maps/long-walk", "long_walk_pathfinding.png"),
        ("nclone/test_maps/path-jump-required", "path_jump_required_pathfinding.png"),
        ("nclone/test_maps/only-jump", "only_jump_pathfinding.png"),
    ]
    
    print("Creating pathfinding visualizations with accurate tile rendering...")
    print("=" * 60)
    
    for map_path, output_path in test_maps:
        if os.path.exists(map_path):
            visualizer.create_visualization(map_path, output_path)
        else:
            print(f"Test map not found: {map_path}")
    
    print("\nVisualization complete! Check the generated PNG files for visual validation.")

if __name__ == "__main__":
    main()