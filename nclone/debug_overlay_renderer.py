import pygame
import numpy as np
from . import render_utils
from typing import Optional
from .constants.physics_constants import (
    TILE_PIXEL_SIZE,
    FULL_MAP_WIDTH,
    FULL_MAP_HEIGHT,
)
from .graph.hierarchical_builder import HierarchicalGraphBuilder
from .graph.common import EdgeType, GraphData


class DebugOverlayRenderer:
    def __init__(self, sim, screen, adjust, tile_x_offset, tile_y_offset):
        self.sim = sim
        self.screen = screen
        self.adjust = adjust
        self.tile_x_offset = tile_x_offset
        self.tile_y_offset = tile_y_offset
        pygame.font.init()

        # Colors for Entity Grid visualization
        self.ENTITY_GRID_CELL_COLOR = (255, 165, 0, 100)  # Orange, semi-transparent
        self.ENTITY_GRID_TEXT_COLOR = (255, 255, 255, 200)  # White, semi-transparent

        self.GRAPH_NODE_COLOR_GRID = (240, 240, 240, 220)
        self.GRAPH_NODE_COLOR_ENTITY = (255, 90, 90, 240)
        self.GRAPH_NODE_COLOR_NINJA = (60, 220, 255, 255)
        self.GRAPH_BG_DIM = (0, 0, 0, 140)
        self._graph_builder_for_dims = HierarchicalGraphBuilder()

    def update_params(self, adjust, tile_x_offset, tile_y_offset):
        self.adjust = adjust
        self.tile_x_offset = tile_x_offset
        self.tile_y_offset = tile_y_offset

    def _get_area_color(
        self,
        base_color: tuple[int, int, int],
        index: int,
        max_index: int,
        opacity: int = 192,
    ) -> tuple[int, int, int, int]:
        """Calculate color based on area index, making it darker as index increases."""
        # Calculate brightness factor (0.3 to 1.0)
        brightness = 1.0 - (0.7 * index / max_index if max_index > 0 else 0)
        return (
            int(base_color[0] * brightness),
            int(base_color[1] * brightness),
            int(base_color[2] * brightness),
            opacity,
        )

    def _draw_exploration_grid(self, debug_info: dict) -> pygame.Surface:
        """Draw the exploration grid overlay."""
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        if "exploration" not in debug_info:
            return surface  # Return an empty surface if no exploration data

        exploration = debug_info["exploration"]

        # Get exploration data
        visited_cells = exploration.get("visited_cells")
        visited_4x4 = exploration.get("visited_4x4")
        visited_8x8 = exploration.get("visited_8x8")
        visited_16x16 = exploration.get("visited_16x16")

        if not all(
            isinstance(arr, np.ndarray)
            for arr in [visited_cells, visited_4x4, visited_8x8, visited_16x16]
        ):
            return surface  # Return empty if any data is missing or not an ndarray

        if not all(
            arr is not None
            for arr in [visited_cells, visited_4x4, visited_8x8, visited_16x16]
        ):
            return surface

        # Calculate cell size based on grid dimensions
        cell_size = 24 * self.adjust  # 24 pixels per cell
        quarter_size = cell_size / 2  # Size of each quarter of the cell

        # Draw individual cells with subdivisions
        for y in range(visited_cells.shape[0]):
            for x in range(visited_cells.shape[1]):
                base_x = x * cell_size + self.tile_x_offset
                base_y = y * cell_size + self.tile_y_offset

                if visited_cells[y, x]:
                    # Draw the four quarters of each cell
                    # Top-left (cell color - green)
                    rect_cell = pygame.Rect(base_x, base_y, quarter_size, quarter_size)
                    pygame.draw.rect(
                        surface,
                        render_utils.EXPLORATION_COLORS["cell_visited"],
                        rect_cell,
                    )

                    # Top-right (4x4 area color - red)
                    area_4x4_x, area_4x4_y = x // 4, y // 4
                    rect_4x4 = pygame.Rect(
                        base_x + quarter_size, base_y, quarter_size, quarter_size
                    )
                    if (
                        0 <= area_4x4_y < visited_4x4.shape[0]
                        and 0 <= area_4x4_x < visited_4x4.shape[1]
                        and visited_4x4[area_4x4_y, area_4x4_x]
                    ):
                        index_4x4 = area_4x4_y * visited_4x4.shape[1] + area_4x4_x
                        max_index_4x4 = visited_4x4.shape[0] * visited_4x4.shape[1] - 1
                        color_4x4 = self._get_area_color(
                            render_utils.AREA_BASE_COLORS["4x4"],
                            index_4x4,
                            max_index_4x4,
                        )
                        pygame.draw.rect(surface, color_4x4, rect_4x4)

                    # Bottom-left (8x8 area color - blue)
                    area_8x8_x, area_8x8_y = x // 8, y // 8
                    rect_8x8 = pygame.Rect(
                        base_x, base_y + quarter_size, quarter_size, quarter_size
                    )
                    if (
                        0 <= area_8x8_y < visited_8x8.shape[0]
                        and 0 <= area_8x8_x < visited_8x8.shape[1]
                        and visited_8x8[area_8x8_y, area_8x8_x]
                    ):
                        index_8x8 = area_8x8_y * visited_8x8.shape[1] + area_8x8_x
                        max_index_8x8 = visited_8x8.shape[0] * visited_8x8.shape[1] - 1
                        color_8x8 = self._get_area_color(
                            render_utils.AREA_BASE_COLORS["8x8"],
                            index_8x8,
                            max_index_8x8,
                        )
                        pygame.draw.rect(surface, color_8x8, rect_8x8)

                    # Bottom-right (16x16 area color - grey)
                    area_16x16_x, area_16x16_y = x // 16, y // 16
                    rect_16x16 = pygame.Rect(
                        base_x + quarter_size,
                        base_y + quarter_size,
                        quarter_size,
                        quarter_size,
                    )
                    if (
                        0 <= area_16x16_y < visited_16x16.shape[0]
                        and 0 <= area_16x16_x < visited_16x16.shape[1]
                        and visited_16x16[area_16x16_y, area_16x16_x]
                    ):
                        index_16x16 = (
                            area_16x16_y * visited_16x16.shape[1] + area_16x16_x
                        )
                        max_index_16x16 = (
                            visited_16x16.shape[0] * visited_16x16.shape[1] - 1
                        )
                        color_16x16 = self._get_area_color(
                            render_utils.AREA_BASE_COLORS["16x16"],
                            index_16x16,
                            max_index_16x16,
                        )
                        pygame.draw.rect(surface, color_16x16, rect_16x16)

                    # Draw cell grid
                    rect_full = pygame.Rect(base_x, base_y, cell_size, cell_size)
                    pygame.draw.rect(
                        surface,
                        render_utils.EXPLORATION_COLORS["grid_cell"],
                        rect_full,
                        1,
                    )

        return surface

    def _draw_entity_grid(self) -> pygame.Surface:
        """Draw the entity grid overlay, showing cells with entities and their counts."""
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        cell_size_pixels = 24  # Standard N cell size
        adjusted_cell_size = cell_size_pixels * self.adjust

        try:
            font = pygame.font.Font(None, int(16 * self.adjust))  # Adjust font size
        except pygame.error:
            font = pygame.font.SysFont("arial", int(14 * self.adjust))

        for cell_coord, entities_in_cell in self.sim.grid_entity.items():
            if not entities_in_cell:
                continue

            cell_x, cell_y = cell_coord
            num_entities = len(entities_in_cell)

            # Calculate screen coordinates for the cell rectangle
            rect_x = cell_x * adjusted_cell_size + self.tile_x_offset
            rect_y = cell_y * adjusted_cell_size + self.tile_y_offset

            # Draw the cell rectangle
            cell_rect = pygame.Rect(
                rect_x, rect_y, adjusted_cell_size, adjusted_cell_size
            )
            pygame.draw.rect(surface, self.ENTITY_GRID_CELL_COLOR, cell_rect)
            pygame.draw.rect(
                surface, (255, 255, 255, 150), cell_rect, 1
            )  # Border for clarity

            # Render the entity count text
            if num_entities > 0:
                text_surf = font.render(
                    str(num_entities), True, self.ENTITY_GRID_TEXT_COLOR
                )
                text_rect = text_surf.get_rect(
                    center=(
                        rect_x + adjusted_cell_size / 2,
                        rect_y + adjusted_cell_size / 2,
                    )
                )
                surface.blit(text_surf, text_rect)

        return surface

    def _draw_grid_outline(self) -> pygame.Surface:
        """Draw the grid outline overlay showing 42x23 map boundaries."""
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        # Calculate the actual size on screen
        cell_size = TILE_PIXEL_SIZE * self.adjust

        # Grid line color - light grey with transparency
        grid_color = (150, 150, 150, 150)

        # Draw vertical lines
        for x in range(FULL_MAP_WIDTH + 1):
            line_x = self.tile_x_offset + x * cell_size
            start_y = self.tile_y_offset
            end_y = self.tile_y_offset + FULL_MAP_HEIGHT * cell_size
            pygame.draw.line(surface, grid_color, (line_x, start_y), (line_x, end_y), 1)

        # Draw horizontal lines
        for y in range(FULL_MAP_HEIGHT + 1):
            line_y = self.tile_y_offset + y * cell_size
            start_x = self.tile_x_offset
            end_x = self.tile_x_offset + FULL_MAP_WIDTH * cell_size
            pygame.draw.line(surface, grid_color, (start_x, line_y), (end_x, line_y), 1)

        return surface

    def draw_debug_overlay(self, debug_info: dict = None) -> pygame.Surface:
        """Helper method to draw debug overlay with nested dictionary support.

        Args:
            debug_info: Optional dictionary containing debug information to display

        Returns:
            pygame.Surface: Surface containing the rendered debug text and exploration grid.
        """
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        if not debug_info:
            return surface  # Return empty surface if no debug info

        # Draw exploration grid if available
        exploration_surface = self._draw_exploration_grid(debug_info)
        if exploration_surface:
            surface.blit(exploration_surface, (0, 0))

        # Draw grid outline if provided
        if debug_info and "grid_outline" in debug_info:
            grid_surface = self._draw_grid_outline()
            surface.blit(grid_surface, (0, 0))

        # Base font and settings
        try:
            font = pygame.font.Font(None, 20)  # Small font size
        except pygame.error:
            # Fallback if default font is not found (e.g. pygame not fully initialized elsewhere)
            font = pygame.font.SysFont("arial", 18)

        line_height = 12  # Reduced from 20 to 16 for tighter spacing
        base_color = (255, 255, 255, 191)  # White with 75% opacity

        # Calculate total height needed for text
        def calc_text_height(d: dict, level: int = 0) -> int:
            height = 0
            for key, value in d.items():
                if key == "exploration":  # Don't count exploration dict for text height
                    continue
                if (
                    key == "grid_outline"
                ):  # Don't count grid outline dict for text height, it's visual
                    continue
                height += line_height
                if isinstance(value, dict):
                    height += calc_text_height(value, level + 1)
            return height

        total_height = line_height  # For frame number or a general padding
        if debug_info:
            total_height += calc_text_height(debug_info)

        # Calculate starting position (bottom right)
        x_pos = self.screen.get_width() - 250  # Fixed width from right edge
        y_pos = self.screen.get_height() - total_height - 5  # 5px padding from bottom

        def format_value(value):
            """Format value with rounding for numbers."""
            if isinstance(value, (float, np.float32, np.float64)):
                return f"{value:.3f}"
            elif isinstance(value, tuple) and all(
                isinstance(x, (int, float, np.float32, np.float64)) for x in value
            ):
                return tuple(
                    round(x, 2) if isinstance(x, (float, np.float32, np.float64)) else x
                    for x in value
                )
            elif isinstance(value, np.ndarray):
                return f"Array({value.shape})"
            return value

        def render_dict(d: dict, indent_level: int = 0):
            nonlocal y_pos
            indent = "  " * indent_level

            for key, value in d.items():
                if key == "exploration":  # Skip rendering exploration data as text
                    continue
                if key == "graph":  # Skip rendering graph data as text
                    continue
                if key == "grid_outline":  # Skip rendering grid outline data as text
                    continue
                if isinstance(value, dict):
                    # Render dictionary key as a header
                    text_surface = font.render(f"{indent}{key}:", True, base_color)
                    surface.blit(text_surface, (x_pos, y_pos))
                    y_pos += line_height
                    # Recursively render nested dictionary
                    render_dict(value, indent_level + 1)
                else:
                    # Format and render key-value pair
                    formatted_value = format_value(value)
                    text_surface = font.render(
                        f"{indent}{key}: {formatted_value}", True, base_color
                    )
                    surface.blit(text_surface, (x_pos, y_pos))
                    y_pos += line_height

        # Render debug info text if provided
        if debug_info:
            render_dict(debug_info)

        return surface
