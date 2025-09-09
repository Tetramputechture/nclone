"""
Authoritative pathfinding visualization for N++ using the consolidated system.

This module provides the single, correct visualization implementation
that uses the same pathfinding system as the validation tests.
"""

import os
import math
import cairo
import numpy as np
from typing import Dict, List, Tuple, Optional

from ..pathfinding import CorePathfinder, MovementType
from ..pathfinding.movement_types import MOVEMENT_COLORS
from ..graph.level_data import LevelData
from ..constants.entity_types import EntityType
from ..constants.physics_constants import TILE_PIXEL_SIZE

# Entity colors
ENTITY_COLORS = {
    EntityType.NINJA: (1.0, 1.0, 1.0),       # White
    EntityType.EXIT_SWITCH: (1.0, 0.84, 0.0), # Gold
    EntityType.EXIT_DOOR: (1.0, 0.0, 0.0),   # Red
}

# Background and tile colors
BGCOLOR_RGB = (0.1, 0.1, 0.1)  # Dark gray background
TILECOLOR_RGB = (0.75, 0.75, 0.75)  # Light gray tiles

class PathfindingVisualizer:
    """
    Authoritative pathfinding visualizer using the consolidated system.
    
    This class creates visualizations that accurately represent the
    physics-aware pathfinding results.
    """
    
    def __init__(self, tile_size: int = 30):
        self.tile_size = tile_size
        self.pathfinder = CorePathfinder()
    
    def create_test_maps(self) -> Dict[str, LevelData]:
        """Create the four validation test maps."""
        
        test_maps = {}
        
        # Simple-walk map: 9 tiles wide, single horizontal platform
        tiles = np.zeros((5, 9), dtype=int)
        tiles[3, :] = 1  # Ground platform on row 3
        entities = [
            {"type": 0, "x": 24, "y": 60},     # Ninja at leftmost tile
            {"type": 4, "x": 120, "y": 60},    # Exit switch at middle tile (5th)
            {"type": 3, "x": 192, "y": 60}     # Exit door at rightmost tile
        ]
        test_maps["simple-walk"] = LevelData(tiles, entities)
        
        # Long-walk map: 42 tiles wide, single horizontal platform
        tiles = np.zeros((5, 42), dtype=int)
        tiles[3, :] = 1  # Ground platform on row 3
        entities = [
            {"type": 0, "x": 24, "y": 60},      # Ninja at leftmost tile
            {"type": 4, "x": 960, "y": 60},     # Exit switch at 41st tile
            {"type": 3, "x": 984, "y": 60}      # Exit door at rightmost tile
        ]
        test_maps["long-walk"] = LevelData(tiles, entities)
        
        # Path-jump-required map: elevated switch position
        tiles = np.zeros((5, 9), dtype=int)
        tiles[3, :] = 1  # Ground platform on row 3
        tiles[2, 4] = 1  # Elevated tile for switch at column 4
        entities = [
            {"type": 0, "x": 24, "y": 60},     # Ninja on ground
            {"type": 4, "x": 120, "y": 36},    # Switch on elevated tile
            {"type": 3, "x": 192, "y": 60}     # Door on ground
        ]
        test_maps["path-jump-required"] = LevelData(tiles, entities)
        
        # Only-jump map: vertical corridor
        tiles = np.zeros((8, 3), dtype=int)
        tiles[:, 0] = 1  # Left wall
        tiles[:, 2] = 1  # Right wall
        entities = [
            {"type": 0, "x": 36, "y": 156},    # Ninja at bottom
            {"type": 4, "x": 36, "y": 108},    # Switch in middle
            {"type": 3, "x": 36, "y": 60}      # Door at top
        ]
        test_maps["only-jump"] = LevelData(tiles, entities)
        
        return test_maps
    
    def visualize_map(self, map_name: str, level_data: LevelData, output_path: str):
        """Create visualization for a single map."""
        
        print(f"\n=== Visualizing {map_name} ===")
        
        # Calculate canvas size
        height, width = level_data.tiles.shape
        canvas_width = width * self.tile_size + 150  # Extra space for legend
        canvas_height = height * self.tile_size + 100
        
        # Create Cairo surface
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, canvas_width, canvas_height)
        ctx = cairo.Context(surface)
        
        # Fill background
        ctx.set_source_rgb(*BGCOLOR_RGB)
        ctx.paint()
        
        # Draw tiles
        self._draw_tiles(ctx, level_data.tiles)
        
        # Draw entities and get positions
        entity_positions = self._draw_entities(ctx, level_data.entities)
        
        # Find and draw paths
        self._draw_paths(ctx, level_data, entity_positions)
        
        # Draw legend
        self._draw_legend(ctx, canvas_width, canvas_height)
        
        # Save image
        surface.write_to_png(output_path)
        print(f"✅ Saved visualization to {output_path}")
    
    def _draw_tiles(self, ctx, tiles):
        """Draw the tile grid."""
        
        height, width = tiles.shape
        
        for row in range(height):
            for col in range(width):
                if tiles[row, col] == 1:  # Solid tile
                    x = col * self.tile_size
                    y = row * self.tile_size
                    
                    # Fill tile
                    ctx.set_source_rgb(*TILECOLOR_RGB)
                    ctx.rectangle(x, y, self.tile_size, self.tile_size)
                    ctx.fill()
                    
                    # Draw tile border
                    ctx.set_source_rgb(0.5, 0.5, 0.5)
                    ctx.rectangle(x, y, self.tile_size, self.tile_size)
                    ctx.set_line_width(1)
                    ctx.stroke()
    
    def _draw_entities(self, ctx, entities) -> Dict[str, Tuple[float, float]]:
        """Draw entities and return their positions."""
        
        entity_positions = {}
        
        for entity in entities:
            entity_type = entity.get("type", 0)
            entity_x = entity.get("x", 0.0)
            entity_y = entity.get("y", 0.0)
            
            # Convert to canvas coordinates
            canvas_x = (entity_x / TILE_PIXEL_SIZE) * self.tile_size
            canvas_y = (entity_y / TILE_PIXEL_SIZE) * self.tile_size
            
            # Store position for pathfinding
            if entity_type == 0:  # Ninja
                entity_positions["ninja"] = (entity_x, entity_y)
                color = ENTITY_COLORS[EntityType.NINJA]
                radius = 8
                outline_width = 2
            elif entity_type == 4:  # Switch
                entity_positions["switch"] = (entity_x, entity_y)
                color = ENTITY_COLORS[EntityType.EXIT_SWITCH]
                radius = 6
                outline_width = 1
            elif entity_type == 3:  # Door
                entity_positions["door"] = (entity_x, entity_y)
                color = ENTITY_COLORS[EntityType.EXIT_DOOR]
                radius = 6
                outline_width = 1
            else:
                continue
            
            # Draw entity
            ctx.set_source_rgb(*color)
            ctx.arc(canvas_x, canvas_y, radius, 0, 2 * math.pi)
            ctx.fill()
            
            # Draw outline
            ctx.set_source_rgb(0, 0, 0)
            ctx.arc(canvas_x, canvas_y, radius, 0, 2 * math.pi)
            ctx.set_line_width(outline_width)
            ctx.stroke()
        
        return entity_positions
    
    def _draw_paths(self, ctx, level_data: LevelData, entity_positions: Dict[str, Tuple[float, float]]):
        """Find and draw pathfinding paths."""
        
        ninja_pos = entity_positions.get("ninja")
        switch_pos = entity_positions.get("switch")
        door_pos = entity_positions.get("door")
        
        if not ninja_pos or not door_pos:
            print("⚠️  Missing ninja or door position")
            return
        
        # Create path waypoints
        waypoints = [ninja_pos]
        if switch_pos:
            waypoints.append(switch_pos)
        waypoints.append(door_pos)
        
        # Find path using consolidated pathfinder
        path_segments = self.pathfinder.find_multi_segment_path(level_data, waypoints)
        
        # Draw each path segment
        for i, segment in enumerate(path_segments):
            start_pos = segment['start_pos']
            end_pos = segment['end_pos']
            movement_type = segment['movement_type']
            
            # Get color for movement type
            color = MOVEMENT_COLORS.get(movement_type, (0.5, 0.5, 0.5))
            
            # Convert to canvas coordinates
            start_canvas_x = (start_pos[0] / TILE_PIXEL_SIZE) * self.tile_size
            start_canvas_y = (start_pos[1] / TILE_PIXEL_SIZE) * self.tile_size
            end_canvas_x = (end_pos[0] / TILE_PIXEL_SIZE) * self.tile_size
            end_canvas_y = (end_pos[1] / TILE_PIXEL_SIZE) * self.tile_size
            
            # Draw path segment
            ctx.set_source_rgb(*color)
            ctx.set_line_width(4)
            ctx.move_to(start_canvas_x, start_canvas_y)
            ctx.line_to(end_canvas_x, end_canvas_y)
            ctx.stroke()
            
            # Print segment info
            distance = segment['physics_params']['distance']
            print(f"  Segment {i+1}: {movement_type.name} - {distance:.1f}px")
        
        # Print path summary
        summary = self.pathfinder.get_path_summary(path_segments)
        print(f"  Total distance: {summary['total_distance']:.1f}px")
        print(f"  Movement types: {summary['movement_type_counts']}")
    
    def _draw_legend(self, ctx, canvas_width: int, canvas_height: int):
        """Draw legend showing entity types and movement types."""
        
        legend_x = canvas_width - 140
        legend_y = 10
        
        # Legend background
        ctx.set_source_rgba(0, 0, 0, 0.8)
        ctx.rectangle(legend_x - 5, legend_y - 5, 135, 220)
        ctx.fill()
        
        # Legend title
        ctx.set_source_rgb(1, 1, 1)
        ctx.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(14)
        ctx.move_to(legend_x, legend_y + 15)
        ctx.show_text("Legend")
        
        y_offset = 35
        
        # Entity types section
        ctx.set_font_size(12)
        ctx.move_to(legend_x, legend_y + y_offset)
        ctx.show_text("Entities:")
        y_offset += 20
        
        # Draw entity legend items
        entity_items = [
            ("Ninja", ENTITY_COLORS[EntityType.NINJA]),
            ("Switch", ENTITY_COLORS[EntityType.EXIT_SWITCH]),
            ("Door", ENTITY_COLORS[EntityType.EXIT_DOOR])
        ]
        
        ctx.set_font_size(10)
        for name, color in entity_items:
            ctx.set_source_rgb(*color)
            ctx.arc(legend_x + 10, legend_y + y_offset, 4, 0, 2 * math.pi)
            ctx.fill()
            
            ctx.set_source_rgb(1, 1, 1)
            ctx.move_to(legend_x + 20, legend_y + y_offset + 3)
            ctx.show_text(name)
            y_offset += 18
        
        y_offset += 10
        
        # Movement types section
        ctx.set_font_size(12)
        ctx.set_source_rgb(1, 1, 1)
        ctx.move_to(legend_x, legend_y + y_offset)
        ctx.show_text("Movements:")
        y_offset += 20
        
        # Draw movement type legend items
        ctx.set_font_size(10)
        for movement_type, color in MOVEMENT_COLORS.items():
            ctx.set_source_rgb(*color)
            ctx.set_line_width(4)
            ctx.move_to(legend_x + 5, legend_y + y_offset)
            ctx.line_to(legend_x + 15, legend_y + y_offset)
            ctx.stroke()
            
            ctx.set_source_rgb(1, 1, 1)
            ctx.move_to(legend_x + 20, legend_y + y_offset + 3)
            ctx.show_text(movement_type.name)
            y_offset += 18
    
    def create_all_visualizations(self):
        """Create visualizations for all test maps."""
        
        test_maps = self.create_test_maps()
        
        print("🎨 Creating consolidated pathfinding visualizations...")
        
        for map_name, level_data in test_maps.items():
            output_path = f"{map_name}_consolidated.png"
            self.visualize_map(map_name, level_data, output_path)
        
        print("\n✅ All consolidated visualizations created successfully!")