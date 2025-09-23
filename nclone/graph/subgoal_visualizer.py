"""
Subgoal Visualization System for N++ Environment

This module provides comprehensive visualization capabilities for subgoals,
reachability analysis, and strategic waypoints in the N++ simulation environment.
It can be used as a debug overlay in realtime or for exporting static images.

Features:
- Real-time subgoal visualization overlay
- Reachability area highlighting
- Strategic waypoint display
- Path visualization between subgoals
- Export functionality for static analysis
- Customizable visual styles and colors
"""

import pygame
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from enum import Enum

from .subgoal_types import Subgoal, SubgoalPlan
from .common import SUB_CELL_SIZE
from ..constants.physics_constants import TILE_PIXEL_SIZE
from ..shared_tile_renderer import SharedTileRenderer


class SubgoalVisualizationMode(Enum):
    """Different visualization modes for subgoals."""

    BASIC = "basic"  # Simple markers for subgoals
    DETAILED = "detailed"  # Detailed info with labels and connections
    REACHABILITY = "reachability"  # Focus on reachability analysis


@dataclass
class SubgoalVisualizationConfig:
    """Configuration for subgoal visualization."""

    # Display modes
    mode: SubgoalVisualizationMode = SubgoalVisualizationMode.DETAILED
    show_labels: bool = True
    show_connections: bool = True
    show_reachability: bool = True
    show_priorities: bool = True

    # Colors (RGBA)
    subgoal_colors: Dict[str, Tuple[int, int, int, int]] = None
    reachability_color: Tuple[int, int, int, int] = (0, 255, 0, 80)
    unreachable_color: Tuple[int, int, int, int] = (255, 0, 0, 80)
    path_color: Tuple[int, int, int, int] = (255, 255, 0, 150)
    connection_color: Tuple[int, int, int, int] = (255, 255, 255, 100)

    # Sizes
    subgoal_radius: int = 12
    path_width: int = 3
    connection_width: int = 2
    font_size: int = 14

    # Animation
    animate_subgoals: bool = True
    pulse_speed: float = 2.0

    def __post_init__(self):
        """Initialize default colors if not provided."""
        if self.subgoal_colors is None:
            self.subgoal_colors = {
                "exit": (0, 255, 0, 200),  # Green
                "exit_switch": (255, 165, 0, 200),  # Orange
                "locked_door_switch": (255, 0, 0, 200),  # Red
                "trap_door_switch": (255, 0, 255, 200),  # Magenta
                "default": (128, 128, 128, 200),  # Gray
            }


class SubgoalVisualizer:
    """
    Comprehensive subgoal visualization system.

    This class handles all aspects of subgoal visualization including:
    - Rendering subgoals with different visual styles
    - Showing reachability areas
    - Displaying connections and dependencies
    - Animating elements for better visibility
    - Exporting visualization to images
    """

    def __init__(self, config: Optional[SubgoalVisualizationConfig] = None):
        """
        Initialize the subgoal visualizer.

        Args:
            config: Visualization configuration. Uses defaults if None.
        """
        self.config = config or SubgoalVisualizationConfig()
        self.font = None
        self.animation_time = 0.0

        # Cache for expensive calculations
        self._reachability_cache = {}
        self._path_cache = {}

        # Initialize pygame font if available
        try:
            pygame.font.init()
            self.font = pygame.font.Font(None, self.config.font_size)
        except (pygame.error, AttributeError):
            self.font = None

    def render_subgoals_overlay(
        self,
        surface: pygame.Surface,
        subgoals: List[Subgoal],
        ninja_position: Tuple[float, float],
        reachable_positions: Optional[Set[Tuple[int, int]]] = None,
        current_plan: Optional[SubgoalPlan] = None,
        tile_x_offset: float = 0,
        tile_y_offset: float = 0,
        adjust: float = 1.0,
    ) -> pygame.Surface:
        """
        Render subgoals overlay on the provided surface.

        Args:
            surface: Pygame surface to render on
            subgoals: List of subgoals to visualize
            ninja_position: Current ninja position (x, y)
            reachable_positions: Set of reachable positions for highlighting
            current_plan: Current subgoal plan for showing execution order
            tile_x_offset: X offset for tile positioning
            tile_y_offset: Y offset for tile positioning
            adjust: Scale adjustment factor

        Returns:
            Modified surface with subgoal visualization
        """
        if not subgoals:
            return surface

        # Create overlay surface
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

        # Update animation time
        self.animation_time += 0.016  # Assume ~60 FPS

        # Render reachability areas first (background layer)
        if self.config.show_reachability and reachable_positions:
            self._render_reachability_areas(
                overlay, reachable_positions, tile_x_offset, tile_y_offset, adjust
            )

        # Render connections between subgoals
        if self.config.show_connections and current_plan:
            self._render_subgoal_connections(
                overlay, subgoals, current_plan, tile_x_offset, tile_y_offset, adjust
            )

        # Render individual subgoals
        for i, subgoal in enumerate(subgoals):
            self._render_single_subgoal(
                overlay,
                subgoal,
                i,
                ninja_position,
                current_plan,
                tile_x_offset,
                tile_y_offset,
                adjust,
            )

        # Render ninja position
        self._render_ninja_position(
            overlay, ninja_position, tile_x_offset, tile_y_offset, adjust
        )

        # Blit overlay onto main surface
        surface.blit(overlay, (0, 0))
        return surface

    def _render_reachability_areas(
        self,
        surface: pygame.Surface,
        reachable_positions: Set[Tuple[int, int]],
        tile_x_offset: float,
        tile_y_offset: float,
        adjust: float,
    ):
        """Render reachability areas as colored overlays."""
        cell_size = SUB_CELL_SIZE * adjust

        for row, col in reachable_positions:
            x = col * cell_size + tile_x_offset
            y = row * cell_size + tile_y_offset
            
            # Fix Y-axis inversion for reachability areas
            y = surface.get_height() - y - cell_size

            rect = pygame.Rect(x, y, cell_size, cell_size)
            pygame.draw.rect(surface, self.config.reachability_color, rect)

    def _render_subgoal_connections(
        self,
        surface: pygame.Surface,
        subgoals: List[Subgoal],
        plan: SubgoalPlan,
        tile_x_offset: float,
        tile_y_offset: float,
        adjust: float,
    ):
        """Render connections between subgoals based on execution order."""
        if len(plan.execution_order) < 2:
            return

        # Draw connections between consecutive subgoals in execution order
        for i in range(len(plan.execution_order) - 1):
            current_idx = plan.execution_order[i]
            next_idx = plan.execution_order[i + 1]

            if current_idx < len(subgoals) and next_idx < len(subgoals):
                current_subgoal = subgoals[current_idx]
                next_subgoal = subgoals[next_idx]

                start_pos = self._subgoal_to_screen_pos(
                    current_subgoal, tile_x_offset, tile_y_offset, adjust, surface.get_height()
                )
                end_pos = self._subgoal_to_screen_pos(
                    next_subgoal, tile_x_offset, tile_y_offset, adjust, surface.get_height()
                )

                # Draw connection line with arrow
                pygame.draw.line(
                    surface,
                    self.config.connection_color,
                    start_pos,
                    end_pos,
                    self.config.connection_width,
                )

                # Draw arrow at end
                self._draw_arrow(
                    surface, start_pos, end_pos, self.config.connection_color
                )

    def _render_single_subgoal(
        self,
        surface: pygame.Surface,
        subgoal: Subgoal,
        index: int,
        ninja_position: Tuple[float, float],
        plan: Optional[SubgoalPlan],
        tile_x_offset: float,
        tile_y_offset: float,
        adjust: float,
    ):
        """Render a single subgoal with all its visual elements."""
        screen_pos = self._subgoal_to_screen_pos(
            subgoal, tile_x_offset, tile_y_offset, adjust, surface.get_height()
        )

        # Get color for this subgoal type
        color = self.config.subgoal_colors.get(
            subgoal.goal_type, self.config.subgoal_colors["default"]
        )

        # Apply animation if enabled
        radius = self.config.subgoal_radius
        if self.config.animate_subgoals:
            pulse = 1.0 + 0.2 * np.sin(
                self.animation_time * self.config.pulse_speed + index
            )
            radius = int(radius * pulse)

        # Draw main subgoal circle
        pygame.draw.circle(surface, color, screen_pos, radius)
        pygame.draw.circle(surface, (255, 255, 255, 255), screen_pos, radius, 2)

        # Draw priority indicator if enabled
        if self.config.show_priorities and hasattr(subgoal, "priority"):
            self._draw_priority_indicator(surface, screen_pos, subgoal.priority, radius)

        # Draw label if enabled
        if self.config.show_labels and self.font:
            self._draw_subgoal_label(surface, screen_pos, subgoal, radius)

        # Highlight if this is the next subgoal in plan
        if plan and plan.execution_order and plan.execution_order[0] == index:
            self._draw_next_subgoal_highlight(surface, screen_pos, radius)

    def _render_ninja_position(
        self,
        surface: pygame.Surface,
        ninja_position: Tuple[float, float],
        tile_x_offset: float,
        tile_y_offset: float,
        adjust: float,
    ):
        """Render ninja position marker."""
        x = ninja_position[0] * adjust + tile_x_offset
        # Use same coordinate system as normal entity renderer (no Y-flipping)
        # Ninja coordinates are already in pygame/screen coordinate system
        y = ninja_position[1] * adjust + tile_y_offset

        # Draw ninja marker
        ninja_color = (0, 255, 255, 255)  # Cyan
        pygame.draw.circle(surface, ninja_color, (int(x), int(y)), 8)
        pygame.draw.circle(surface, (255, 255, 255, 255), (int(x), int(y)), 8, 2)

        # Draw ninja label
        if self.font:
            text = self.font.render("NINJA", True, (255, 255, 255))
            text_rect = text.get_rect(center=(int(x), int(y) - 20))
            surface.blit(text, text_rect)

    def _subgoal_to_screen_pos(
        self,
        subgoal: Subgoal,
        tile_x_offset: float,
        tile_y_offset: float,
        adjust: float,
        surface_height: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Convert subgoal position to screen coordinates."""
        # Use actual entity position if available (more accurate)
        if hasattr(subgoal, 'entity_position') and subgoal.entity_position is not None:
            x = subgoal.entity_position[0] * adjust + tile_x_offset
            # Use same coordinate system as entities and ninja (no Y-flipping)
            y = subgoal.entity_position[1] * adjust + tile_y_offset
        else:
            # Fallback to sub-grid position
            sub_row, sub_col = subgoal.position
            x = (
                sub_col * SUB_CELL_SIZE * adjust
                + SUB_CELL_SIZE * adjust // 2
                + tile_x_offset
            )
            # Use same coordinate system as entities and ninja (no Y-flipping)
            y = (
                sub_row * SUB_CELL_SIZE * adjust
                + SUB_CELL_SIZE * adjust // 2
                + tile_y_offset
            )
            
        return (int(x), int(y))

    def _draw_arrow(
        self,
        surface: pygame.Surface,
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: Tuple[int, int, int, int],
    ):
        """Draw an arrow from start to end position."""
        # Calculate arrow direction
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx * dx + dy * dy)

        if length < 1:
            return

        # Normalize direction
        dx /= length
        dy /= length

        # Arrow head size
        head_size = 8

        # Calculate arrow head points
        head_x = end[0] - head_size * dx
        head_y = end[1] - head_size * dy

        # Perpendicular vector for arrow head
        perp_x = -dy * head_size * 0.5
        perp_y = dx * head_size * 0.5

        # Arrow head points
        p1 = (int(head_x + perp_x), int(head_y + perp_y))
        p2 = (int(head_x - perp_x), int(head_y - perp_y))

        # Draw arrow head
        pygame.draw.polygon(surface, color[:3], [end, p1, p2])

    def _draw_priority_indicator(
        self,
        surface: pygame.Surface,
        center: Tuple[int, int],
        priority: int,
        radius: int,
    ):
        """Draw priority indicator on subgoal."""
        # Draw priority number in top-right corner of subgoal
        if self.font:
            text = self.font.render(str(priority), True, (255, 255, 255))
            text_pos = (center[0] + radius // 2, center[1] - radius // 2)
            surface.blit(text, text_pos)

    def _draw_subgoal_label(
        self,
        surface: pygame.Surface,
        center: Tuple[int, int],
        subgoal: Subgoal,
        radius: int,
    ):
        """Draw label for subgoal."""
        # Create readable label
        label_text = subgoal.goal_type.replace("_", " ").title()
        text = self.font.render(label_text, True, (255, 255, 255))

        # Position label below subgoal
        text_rect = text.get_rect(center=(center[0], center[1] + radius + 15))

        # Draw background for better readability
        bg_rect = text_rect.inflate(4, 2)
        pygame.draw.rect(surface, (0, 0, 0, 180), bg_rect)

        surface.blit(text, text_rect)

    def _draw_next_subgoal_highlight(
        self, surface: pygame.Surface, center: Tuple[int, int], radius: int
    ):
        """Draw highlight for the next subgoal to execute."""
        # Animated highlight ring
        highlight_radius = radius + 5 + int(3 * np.sin(self.animation_time * 4))
        pygame.draw.circle(surface, (255, 255, 0, 150), center, highlight_radius, 3)

    def export_subgoal_visualization(
        self,
        subgoals: List[Subgoal],
        ninja_position: Tuple[float, float],
        level_dimensions: Tuple[int, int],
        reachable_positions: Optional[Set[Tuple[int, int]]] = None,
        current_plan: Optional[SubgoalPlan] = None,
        filename: str = "subgoal_visualization.png",
        level_data=None,
        entities=None,
    ) -> bool:
        """
        Export subgoal visualization to an image file with actual level rendering.

        Args:
            subgoals: List of subgoals to visualize
            ninja_position: Current ninja position
            level_dimensions: (width, height) of the level in tiles
            reachable_positions: Set of reachable positions
            current_plan: Current subgoal plan
            filename: Output filename
            level_data: Level data for rendering tiles
            entities: Entity data for rendering entities

        Returns:
            True if export successful, False otherwise
        """
        try:
            # Calculate image dimensions - note Y=0 is at top-left
            width = level_dimensions[0] * TILE_PIXEL_SIZE
            height = level_dimensions[1] * TILE_PIXEL_SIZE

            # Create surface for export
            export_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            export_surface.fill((30, 30, 30, 255))  # Dark background

            # Render level background if available
            if level_data is not None:
                self._render_level_background(export_surface, level_data, level_dimensions)

            # Render entities if available
            if entities is not None:
                self._render_entities(export_surface, entities, level_dimensions, 
                                    tile_x_offset=0, tile_y_offset=0, adjust=1.0)

            # Render subgoals overlay
            self.render_subgoals_overlay(
                export_surface,
                subgoals,
                ninja_position,
                reachable_positions,
                current_plan,
                tile_x_offset=0,
                tile_y_offset=0,
                adjust=1.0,
            )

            # Save to file
            pygame.image.save(export_surface, filename)
            return True

        except Exception as e:
            print(f"Error exporting subgoal visualization: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _render_level_background(self, surface, level_data, level_dimensions):
        """Render level tiles as background using proper tile rendering."""
        try:
            # Handle different level_data formats
            if isinstance(level_data, dict):
                # level_data is tile_dic - use shared tile renderer for pixel-perfect rendering
                tile_surface = SharedTileRenderer.render_tiles_to_pygame_surface(
                    level_data, surface.get_width(), surface.get_height(), tile_size=TILE_PIXEL_SIZE
                )
                surface.blit(tile_surface, (0, 0))
                
            elif hasattr(level_data, 'shape'):
                # level_data is numpy array - fallback to simple rendering
                tiles = level_data
                tile_count = 0
                
                # Render tiles - Fix Y-axis inversion
                for y in range(tiles.shape[0]):
                    for x in range(tiles.shape[1]):
                        tile_value = tiles[y, x]
                        if tile_value > 0:  # Non-empty tile
                            tile_count += 1
                            # Convert tile coordinates to pixel coordinates
                            pixel_x = x * TILE_PIXEL_SIZE
                            # Fix Y-axis inversion: flip Y coordinate
                            pixel_y = surface.get_height() - (y + 1) * TILE_PIXEL_SIZE
                            
                            # Draw tile as a rectangle
                            tile_rect = pygame.Rect(pixel_x, pixel_y, TILE_PIXEL_SIZE, TILE_PIXEL_SIZE)
                            pygame.draw.rect(surface, (100, 100, 100, 255), tile_rect)
                            pygame.draw.rect(surface, (150, 150, 150, 255), tile_rect, 1)
            
        except Exception as e:
            print(f"Error rendering level background: {e}")
            import traceback
            traceback.print_exc()

    def _render_entities(self, surface, entities, level_dimensions, tile_x_offset=0, tile_y_offset=0, adjust=1.0):
        """Render entities on the level."""
        try:
            from ..constants.entity_types import EntityType
            
            # Entity colors
            entity_colors = {
                EntityType.NINJA: (0, 255, 255, 255),  # Cyan
                EntityType.EXIT_SWITCH: (255, 255, 0, 255),  # Yellow
                EntityType.EXIT_DOOR: (0, 255, 0, 255),  # Green
                EntityType.LOCKED_DOOR: (255, 0, 0, 255),  # Red
                EntityType.REGULAR_DOOR: (0, 200, 0, 255),  # Dark green
                EntityType.ONE_WAY: (128, 128, 128, 255),  # Gray
                EntityType.GOLD: (255, 215, 0, 255),  # Gold
                EntityType.TOGGLE_MINE: (255, 100, 100, 255),  # Light red
                EntityType.DRONE_ZAP: (255, 0, 255, 255),  # Magenta
                EntityType.THWUMP: (139, 69, 19, 255),  # Brown
            }
            
            entity_count = 0
            for entity in entities:
                # Try different ways to get entity data
                if isinstance(entity, dict):
                    entity_type = entity.get('type')
                    x = entity.get('x', 0)
                    y = entity.get('y', 0)
                else:
                    entity_type = getattr(entity, 'type', None)
                    # Use the same position attributes as entity_renderer.py
                    if hasattr(entity, 'xpos') and hasattr(entity, 'ypos'):
                        x = entity.xpos
                        y = entity.ypos
                    elif hasattr(entity, 'position'):
                        pos = entity.position
                        x, y = pos[0], pos[1] if len(pos) > 1 else (pos, 0)
                    elif hasattr(entity, 'x') and hasattr(entity, 'y'):
                        x = entity.x
                        y = entity.y
                    elif hasattr(entity, 'pos_x') and hasattr(entity, 'pos_y'):
                        x = entity.pos_x
                        y = entity.pos_y
                    else:
                        x, y = 0, 0
                
                # Skip entities at origin (likely uninitialized)
                if x == 0 and y == 0:
                    entity_count += 1
                    continue
                
                # Get entity color
                color = entity_colors.get(entity_type, (255, 255, 255, 255))  # Default white
                
                # Use same coordinate system as normal entity renderer (no Y-flipping)
                # Entity coordinates are already in pygame/screen coordinate system
                screen_x = int(x * adjust + tile_x_offset)
                screen_y = int(y * adjust + tile_y_offset)
                center = (screen_x, screen_y)
                radius = 8
                pygame.draw.circle(surface, color, center, radius)
                pygame.draw.circle(surface, (255, 255, 255, 255), center, radius, 2)
                
                entity_count += 1
                
        except Exception as e:
            print(f"Error rendering entities: {e}")
            import traceback
            traceback.print_exc()

    def update_config(self, **kwargs):
        """Update visualization configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def set_mode(self, mode: SubgoalVisualizationMode):
        """Set visualization mode."""
        self.config.mode = mode

        # Adjust settings based on mode
        if mode == SubgoalVisualizationMode.BASIC:
            self.config.show_labels = False
            self.config.show_connections = False
            self.config.animate_subgoals = False
        elif mode == SubgoalVisualizationMode.DETAILED:
            self.config.show_labels = True
            self.config.show_connections = True
            self.config.animate_subgoals = True
        elif mode == SubgoalVisualizationMode.REACHABILITY:
            self.config.show_reachability = True
            self.config.show_connections = False
        elif mode == SubgoalVisualizationMode.PATHFINDING:
            self.config.show_connections = True
            self.config.show_labels = True

    def clear_cache(self):
        """Clear internal caches."""
        self._reachability_cache.clear()
        self._path_cache.clear()


# Utility functions for integration with existing systems


def create_subgoals_from_reachability(reachability_state, level_data) -> List[Subgoal]:
    """
    Create Subgoal objects from reachability analysis results.

    Args:
        reachability_state: Reachability analysis results
        level_data: Level data for position calculations

    Returns:
        List of Subgoal objects
    """
    subgoals = []

    # Extract subgoals from reachability state
    if hasattr(reachability_state, "subgoals"):
        for sub_row, sub_col, goal_type in reachability_state.subgoals:
            subgoal = Subgoal(
                goal_type=goal_type,
                position=(sub_row, sub_col),
                priority=_get_priority_for_goal_type(goal_type),
            )
            subgoals.append(subgoal)

    return subgoals


def _get_priority_for_goal_type(goal_type: str) -> int:
    """Get priority value for a goal type."""
    priority_map = {
        "exit": 1,
        "exit_switch": 2,
        "locked_door_switch": 3,
        "trap_door_switch": 4,
    }
    return priority_map.get(goal_type, 999)
