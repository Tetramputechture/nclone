import pygame
import numpy as np
from . import render_utils
from typing import Optional, List, Set, Tuple
from .constants.physics_constants import (
    TILE_PIXEL_SIZE,
    FULL_MAP_WIDTH,
    FULL_MAP_HEIGHT,
)
from .constants.entity_types import EntityType
from .graph.subgoal_visualizer import SubgoalVisualizer
from .planning import Subgoal, SubgoalPlan


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

        # Initialize subgoal visualizer
        self.subgoal_visualizer = SubgoalVisualizer()
        self.subgoal_debug_enabled = False
        self.current_subgoals = []
        self.current_subgoal_plan = None
        self.current_reachable_positions = None

        # Pathfinding cache: use shared visualization cache
        from nclone.graph.reachability.path_visualization_cache import (
            PathVisualizationCache,
        )

        self._path_visualization_cache = PathVisualizationCache()

        self.frame_throttle_interval = 5
        self.last_mine_predictor_frame = -999
        self.last_death_prob_frame = -999
        self.last_terminal_velocity_prob_frame = -999
        self.last_exploration_frame = -999

        # Surface caching for expensive overlays
        self.cached_mine_surface = None
        self.cached_death_surface = None
        self.cached_terminal_velocity_surface = None
        self.cached_exploration_surface = None

        # Text rendering cache
        self.text_cache = {}  # (text, font_size, color) -> surface

    def update_params(self, adjust, tile_x_offset, tile_y_offset):
        self.adjust = adjust
        self.tile_x_offset = tile_x_offset
        self.tile_y_offset = tile_y_offset

    def clear_pathfinding_cache(self):
        """Clear pathfinding cache. Call this when level changes or ninja position resets."""
        if hasattr(self, "_path_visualization_cache"):
            self._path_visualization_cache.clear_cache()

    def _render_text_cached(self, font, text, color):
        """
        Render text with caching to avoid repeated font.render() calls.

        PERFORMANCE: Eliminates 3.6s of font rendering per 858 steps (~45,270 calls)
        """
        # Convert color to hashable tuple
        if isinstance(color, list):
            color = tuple(color)

        # Create cache key
        font_size = font.get_height()
        cache_key = (text, font_size, color)

        # Check cache
        if cache_key not in self.text_cache:
            # Cache miss - render and store
            self.text_cache[cache_key] = font.render(text, True, color)

        return self.text_cache[cache_key]

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

    def _draw_tile_types(self) -> pygame.Surface:
        """Draw tile type numbers in the center of each cell (excluding type 0)."""
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        # Calculate cell size
        cell_size = TILE_PIXEL_SIZE * self.adjust

        # Create font for rendering tile types
        try:
            font = pygame.font.Font(None, int(14 * self.adjust))
        except pygame.error:
            font = pygame.font.SysFont("arial", int(12 * self.adjust))

        # Black color for text
        text_color = (0, 0, 0, 255)

        # Iterate through all tiles in the simulator's tile dictionary
        if hasattr(self.sim, "tile_dic"):
            for (tile_x, tile_y), tile_type in self.sim.tile_dic.items():
                # Skip empty tiles (type 0)
                if tile_type == 0:
                    continue

                # Calculate screen position for this tile
                screen_x = tile_x * cell_size + self.tile_x_offset
                screen_y = tile_y * cell_size + self.tile_y_offset

                # Render the tile type number
                text_surf = font.render(str(tile_type), True, text_color)
                text_rect = text_surf.get_rect(
                    center=(screen_x + cell_size / 2, screen_y + cell_size / 2)
                )
                surface.blit(text_surf, text_rect)

        return surface

    def _draw_mine_predictor(
        self, mine_predictor_info: dict
    ) -> Optional[pygame.Surface]:
        """Draw mine death predictor visualization (danger zones, mines, thresholds).

        Args:
            mine_predictor_info: Dictionary containing:
                - mine_positions: list of (x, y) mine positions
                - danger_zone_cells: set of (cell_x, cell_y) danger zone grid cells
                - stats: HybridPredictorStats object
                - ninja_position: tuple of (x, y) ninja position

        Returns:
            pygame.Surface with mine predictor visualization

        PERFORMANCE: Frame throttling reduces cost from 8.2s to <1.5s per 858 steps
        """
        # Frame throttling: Only redraw every N frames
        current_frame = self.sim.frame
        if (
            current_frame - self.last_mine_predictor_frame
            < self.frame_throttle_interval
            and self.cached_mine_surface is not None
        ):
            return self.cached_mine_surface

        self.last_mine_predictor_frame = current_frame

        from .constants import (
            MINE_DANGER_ZONE_CELL_SIZE,
            MINE_DANGER_ZONE_RADIUS,
            MINE_DANGER_THRESHOLD,
        )

        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        mine_positions = mine_predictor_info.get("mine_positions", [])
        danger_zone_cells = mine_predictor_info.get("danger_zone_cells", set())

        # Colors
        DANGER_ZONE_COLOR = (255, 140, 0, 40)  # Orange, very transparent for grid
        DANGER_RADIUS_COLOR = (255, 140, 0, 80)  # Orange, semi-transparent for radius
        THRESHOLD_RADIUS_COLOR = (255, 255, 0, 60)  # Yellow, semi-transparent
        MINE_COLOR = (255, 0, 0, 220)  # Red, mostly opaque
        MINE_BORDER_COLOR = (150, 0, 0, 255)  # Dark red border
        TEXT_COLOR = (255, 255, 255, 255)  # White text

        # Draw danger zone grid cells (Tier 1)
        for cell_x, cell_y in danger_zone_cells:
            # Convert cell coordinates to screen pixels
            screen_x = (
                int(cell_x * MINE_DANGER_ZONE_CELL_SIZE * self.adjust)
                + self.tile_x_offset
            )
            screen_y = (
                int(cell_y * MINE_DANGER_ZONE_CELL_SIZE * self.adjust)
                + self.tile_y_offset
            )
            cell_size = int(MINE_DANGER_ZONE_CELL_SIZE * self.adjust)

            # Draw semi-transparent rectangle for danger zone cell
            pygame.draw.rect(
                surface,
                DANGER_ZONE_COLOR,
                (screen_x, screen_y, cell_size, cell_size),
            )

        # Draw mine positions and threshold circles
        for mine_x, mine_y in mine_positions:
            screen_x = int(mine_x * self.adjust) + self.tile_x_offset
            screen_y = int(mine_y * self.adjust) + self.tile_y_offset

            # Draw danger zone radius (80px, Tier 1 boundary)
            danger_radius = int(MINE_DANGER_ZONE_RADIUS * self.adjust)
            pygame.draw.circle(
                surface, DANGER_RADIUS_COLOR, (screen_x, screen_y), danger_radius, 2
            )

            # Draw threshold radius (30px, Tier 2 boundary)
            threshold_radius = int(MINE_DANGER_THRESHOLD * self.adjust)
            pygame.draw.circle(
                surface,
                THRESHOLD_RADIUS_COLOR,
                (screen_x, screen_y),
                threshold_radius,
                2,
            )

            # Draw mine position (filled circle)
            mine_radius = int(6 * self.adjust)
            pygame.draw.circle(surface, MINE_COLOR, (screen_x, screen_y), mine_radius)
            pygame.draw.circle(
                surface, MINE_BORDER_COLOR, (screen_x, screen_y), mine_radius, 1
            )
        # Cache the surface for next frame
        self.cached_mine_surface = surface
        return surface

    def _draw_death_probability(
        self, death_prob_info: dict
    ) -> Optional[pygame.Surface]:
        """Draw mine death probability visualization showing action masks and probabilities.

        Args:
            death_prob_info: Dictionary containing:
                - result: DeathProbabilityResult object
                - ninja_position: tuple of (x, y) ninja position

        Returns:
            pygame.Surface with mine death probability visualization

        PERFORMANCE: Frame throttling reduces cost from 7.8s to <1.5s per 858 steps
        """
        # Frame throttling: Only redraw every N frames
        current_frame = self.sim.frame
        if (
            current_frame - self.last_death_prob_frame < self.frame_throttle_interval
            and self.cached_death_surface is not None
        ):
            return self.cached_death_surface

        self.last_death_prob_frame = current_frame

        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        result = death_prob_info.get("result")
        ninja_pos = death_prob_info.get("ninja_position", (0, 0))

        if not result:
            return surface

        # Action names for display (compact)
        action_names = ["NO", "LT", "RT", "JP", "JL", "JR"]

        # Colors
        SAFE_COLOR = (100, 255, 100, 255)  # Green for safe actions
        DANGER_COLOR = (255, 200, 0, 255)  # Yellow for risky actions
        DEADLY_COLOR = (255, 50, 50, 255)  # Red for masked actions
        TEXT_COLOR = (255, 255, 255, 255)  # White text
        BG_COLOR = (0, 0, 0, 220)  # Semi-transparent black background

        # Calculate overall death probability (minimum of non-masked actions)
        non_masked_probs = [
            prob
            for i, prob in enumerate(result.action_death_probs)
            if i not in result.masked_actions
        ]
        overall_prob = min(non_masked_probs) if non_masked_probs else 1.0

        # Determine which quadrant ninja is in and place panel in opposite quadrant
        ninja_screen_x = int(ninja_pos[0] * self.adjust) + self.tile_x_offset
        ninja_screen_y = int(ninja_pos[1] * self.adjust) + self.tile_y_offset

        screen_center_x = self.screen.get_width() / 2
        screen_center_y = self.screen.get_height() / 2

        # Determine ninja quadrant
        ninja_in_left = ninja_screen_x < screen_center_x
        ninja_in_top = ninja_screen_y < screen_center_y

        # Compact panel dimensions
        panel_padding = 8
        bar_width = 50
        bar_height = 10
        bar_spacing = 2
        panel_width = 140

        # Calculate panel height
        header_height = 55  # Title + overall bar + distance
        action_section_height = len(action_names) * (bar_height + bar_spacing)
        total_height = header_height + action_section_height + panel_padding

        # Position close to ninja in opposite quadrant
        offset_distance = 80  # Distance from ninja
        screen_margin = 10  # Minimum margin from screen edge

        if ninja_in_left:
            # Place to the right of ninja
            panel_x = ninja_screen_x + offset_distance
            # Keep on screen
            if panel_x + panel_width > self.screen.get_width() - screen_margin:
                panel_x = self.screen.get_width() - panel_width - screen_margin
        else:
            # Place to the left of ninja
            panel_x = ninja_screen_x - offset_distance - panel_width
            # Keep on screen
            if panel_x < screen_margin:
                panel_x = screen_margin

        if ninja_in_top:
            # Place below ninja
            panel_y = ninja_screen_y + offset_distance
            # Keep on screen
            if panel_y + total_height > self.screen.get_height() - screen_margin:
                panel_y = self.screen.get_height() - total_height - screen_margin
        else:
            # Place above ninja
            panel_y = ninja_screen_y - offset_distance - total_height
            # Keep on screen
            if panel_y < screen_margin:
                panel_y = screen_margin

        # Draw background panel
        pygame.draw.rect(
            surface,
            BG_COLOR,
            (panel_x, panel_y, panel_width, total_height),
            border_radius=4,
        )
        pygame.draw.rect(
            surface,
            (255, 200, 50, 255),
            (panel_x, panel_y, panel_width, total_height),
            2,
            border_radius=4,
        )

        # Fonts
        try:
            title_font = pygame.font.Font(None, 18)
            value_font = pygame.font.Font(None, 16)
            action_font = pygame.font.Font(None, 14)
        except pygame.error:
            title_font = pygame.font.SysFont("arial", 16)
            value_font = pygame.font.SysFont("arial", 14)
            action_font = pygame.font.SysFont("arial", 12)

        y_offset = panel_y + panel_padding

        # Title
        title_text = f"Mine Death ({result.frames_simulated}f)"
        title_surf = title_font.render(title_text, True, TEXT_COLOR)
        surface.blit(title_surf, (panel_x + panel_padding, y_offset))
        y_offset += 18

        # Overall death probability bar
        overall_bar_height = 12
        overall_bar_width = panel_width - 2 * panel_padding

        # Determine overall color
        if overall_prob >= 1.0:
            overall_color = DEADLY_COLOR
        elif overall_prob >= 0.5:
            overall_color = DANGER_COLOR
        else:
            overall_color = SAFE_COLOR

        # Draw overall bar background
        pygame.draw.rect(
            surface,
            (40, 40, 40, 255),
            (panel_x + panel_padding, y_offset, overall_bar_width, overall_bar_height),
        )

        # Draw overall bar fill
        overall_fill_width = int(overall_bar_width * overall_prob)
        if overall_fill_width > 0:
            pygame.draw.rect(
                surface,
                overall_color,
                (
                    panel_x + panel_padding,
                    y_offset,
                    overall_fill_width,
                    overall_bar_height,
                ),
            )

        # Draw overall bar border
        pygame.draw.rect(
            surface,
            (180, 180, 180, 255),
            (panel_x + panel_padding, y_offset, overall_bar_width, overall_bar_height),
            1,
        )

        # Draw overall percentage text on bar
        overall_text = f"{overall_prob * 100:.0f}%"
        overall_surf = value_font.render(overall_text, True, TEXT_COLOR)
        text_x = (
            panel_x
            + panel_padding
            + (overall_bar_width - overall_surf.get_width()) // 2
        )
        surface.blit(overall_surf, (text_x, y_offset + 1))
        y_offset += overall_bar_height + 6

        # Draw action bars (compact)
        for i, (action_name, prob) in enumerate(
            zip(action_names, result.action_death_probs)
        ):
            # Determine color based on probability
            if i in result.masked_actions:
                bar_color = DEADLY_COLOR
            elif prob > 0.5:
                bar_color = DANGER_COLOR
            else:
                bar_color = SAFE_COLOR

            # Draw action label (compact)
            label_surf = action_font.render(action_name, True, TEXT_COLOR)
            surface.blit(label_surf, (panel_x + panel_padding, y_offset))

            # Draw probability bar
            bar_x = panel_x + panel_padding + 24
            bar_fill_width = int(bar_width * prob)

            # Background (empty bar)
            pygame.draw.rect(
                surface,
                (40, 40, 40, 255),
                (bar_x, y_offset, bar_width, bar_height),
            )

            # Foreground (filled bar)
            if bar_fill_width > 0:
                pygame.draw.rect(
                    surface,
                    bar_color,
                    (bar_x, y_offset, bar_fill_width, bar_height),
                )

            # Border
            pygame.draw.rect(
                surface,
                (150, 150, 150, 255),
                (bar_x, y_offset, bar_width, bar_height),
                1,
            )

            # Draw percentage text (compact)
            if i in result.masked_actions:
                status_text = "X"
                status_color = DEADLY_COLOR
            else:
                status_text = f"{prob * 100:.0f}"
                status_color = TEXT_COLOR

            status_surf = action_font.render(status_text, True, status_color)
            surface.blit(status_surf, (bar_x + bar_width + 4, y_offset))

            y_offset += bar_height + bar_spacing

        # Cache the surface for next frame
        self.cached_death_surface = surface
        return surface

    def _draw_terminal_velocity_probability(
        self, terminal_velocity_prob_info: dict
    ) -> Optional[pygame.Surface]:
        """Draw terminal velocity death probability visualization showing action risks.

        Args:
            terminal_velocity_prob_info: Dictionary containing:
                - result: DeathProbabilityResult object from terminal velocity predictor
                - ninja_position: tuple of (x, y) ninja position

        Returns:
            pygame.Surface with terminal velocity death probability visualization

        PERFORMANCE: Frame throttling reduces cost
        """
        # Frame throttling: Only redraw every N frames
        current_frame = self.sim.frame
        if (
            current_frame - self.last_terminal_velocity_prob_frame
            < self.frame_throttle_interval
            and self.cached_terminal_velocity_surface is not None
        ):
            return self.cached_terminal_velocity_surface

        self.last_terminal_velocity_prob_frame = current_frame

        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        result = terminal_velocity_prob_info.get("result")
        ninja_pos = terminal_velocity_prob_info.get("ninja_position", (0, 0))

        if not result:
            return surface

        # Action names for display (compact)
        action_names = ["NO", "LT", "RT", "JP", "JL", "JR"]

        # Colors (using blue/cyan theme for terminal velocity vs yellow for mines)
        SAFE_COLOR = (100, 200, 255, 255)  # Cyan for safe actions
        DANGER_COLOR = (255, 150, 50, 255)  # Orange for risky actions
        DEADLY_COLOR = (255, 50, 50, 255)  # Red for masked actions
        TEXT_COLOR = (255, 255, 255, 255)  # White text
        BG_COLOR = (0, 0, 0, 220)  # Semi-transparent black background
        BORDER_COLOR = (100, 200, 255, 255)  # Cyan border

        # Calculate overall death probability (minimum of non-masked actions)
        non_masked_probs = [
            prob
            for i, prob in enumerate(result.action_death_probs)
            if i not in result.masked_actions
        ]
        overall_prob = min(non_masked_probs) if non_masked_probs else 1.0

        # Determine which quadrant ninja is in and place panel in opposite quadrant
        ninja_screen_x = int(ninja_pos[0] * self.adjust) + self.tile_x_offset
        ninja_screen_y = int(ninja_pos[1] * self.adjust) + self.tile_y_offset

        screen_center_x = self.screen.get_width() / 2
        screen_center_y = self.screen.get_height() / 2

        # Determine ninja quadrant
        ninja_in_left = ninja_screen_x < screen_center_x
        ninja_in_top = ninja_screen_y < screen_center_y

        # Compact panel dimensions
        panel_padding = 8
        bar_width = 50
        bar_height = 10
        bar_spacing = 2
        panel_width = 140

        # Calculate panel height
        header_height = 55  # Title + overall bar + distance
        action_section_height = len(action_names) * (bar_height + bar_spacing)
        total_height = header_height + action_section_height + panel_padding

        # Position panel below mine death panel (offset by 170px down)
        vertical_offset = 170  # Offset from mine panel
        offset_distance = 80  # Distance from ninja
        screen_margin = 10  # Minimum margin from screen edge

        if ninja_in_left:
            # Place to the right of ninja
            panel_x = ninja_screen_x + offset_distance
            # Keep on screen
            if panel_x + panel_width > self.screen.get_width() - screen_margin:
                panel_x = self.screen.get_width() - panel_width - screen_margin
        else:
            # Place to the left of ninja
            panel_x = ninja_screen_x - offset_distance - panel_width
            # Keep on screen
            if panel_x < screen_margin:
                panel_x = screen_margin

        if ninja_in_top:
            # Place below ninja, below mine panel
            panel_y = ninja_screen_y + offset_distance + vertical_offset
            # Keep on screen
            if panel_y + total_height > self.screen.get_height() - screen_margin:
                panel_y = self.screen.get_height() - total_height - screen_margin
        else:
            # Place above ninja, above mine panel
            panel_y = ninja_screen_y - offset_distance - total_height - vertical_offset
            # Keep on screen
            if panel_y < screen_margin:
                panel_y = screen_margin

        # Draw background panel
        pygame.draw.rect(
            surface,
            BG_COLOR,
            (panel_x, panel_y, panel_width, total_height),
            border_radius=4,
        )
        pygame.draw.rect(
            surface,
            BORDER_COLOR,
            (panel_x, panel_y, panel_width, total_height),
            2,
            border_radius=4,
        )

        # Fonts
        try:
            title_font = pygame.font.Font(None, 18)
            value_font = pygame.font.Font(None, 16)
            action_font = pygame.font.Font(None, 14)
        except pygame.error:
            title_font = pygame.font.SysFont("arial", 16)
            value_font = pygame.font.SysFont("arial", 14)
            action_font = pygame.font.SysFont("arial", 12)

        y_offset = panel_y + panel_padding

        # Title
        title_text = f"Term. Velocity ({result.frames_simulated}f)"
        title_surf = title_font.render(title_text, True, TEXT_COLOR)
        surface.blit(title_surf, (panel_x + panel_padding, y_offset))
        y_offset += 18

        # Overall death probability bar
        overall_bar_height = 12
        overall_bar_width = panel_width - 2 * panel_padding

        # Determine overall color
        if overall_prob >= 1.0:
            overall_color = DEADLY_COLOR
        elif overall_prob >= 0.5:
            overall_color = DANGER_COLOR
        else:
            overall_color = SAFE_COLOR

        # Draw overall bar background
        pygame.draw.rect(
            surface,
            (40, 40, 40, 255),
            (panel_x + panel_padding, y_offset, overall_bar_width, overall_bar_height),
        )

        # Draw overall bar fill
        overall_fill_width = int(overall_bar_width * overall_prob)
        if overall_fill_width > 0:
            pygame.draw.rect(
                surface,
                overall_color,
                (
                    panel_x + panel_padding,
                    y_offset,
                    overall_fill_width,
                    overall_bar_height,
                ),
            )

        # Draw overall bar border
        pygame.draw.rect(
            surface,
            (180, 180, 180, 255),
            (panel_x + panel_padding, y_offset, overall_bar_width, overall_bar_height),
            1,
        )

        # Draw overall percentage text on bar
        overall_text = f"{overall_prob * 100:.0f}%"
        overall_surf = value_font.render(overall_text, True, TEXT_COLOR)
        text_x = (
            panel_x
            + panel_padding
            + (overall_bar_width - overall_surf.get_width()) // 2
        )
        surface.blit(overall_surf, (text_x, y_offset + 1))
        y_offset += overall_bar_height + 6

        # Draw action bars (compact)
        for i, (action_name, prob) in enumerate(
            zip(action_names, result.action_death_probs)
        ):
            # Determine color based on probability
            if i in result.masked_actions:
                bar_color = DEADLY_COLOR
            elif prob > 0.5:
                bar_color = DANGER_COLOR
            else:
                bar_color = SAFE_COLOR

            # Draw action label (compact)
            label_surf = action_font.render(action_name, True, TEXT_COLOR)
            surface.blit(label_surf, (panel_x + panel_padding, y_offset))

            # Draw probability bar
            bar_x = panel_x + panel_padding + 24
            bar_fill_width = int(bar_width * prob)

            # Background (empty bar)
            pygame.draw.rect(
                surface,
                (40, 40, 40, 255),
                (bar_x, y_offset, bar_width, bar_height),
            )

            # Foreground (filled bar)
            if bar_fill_width > 0:
                pygame.draw.rect(
                    surface,
                    bar_color,
                    (bar_x, y_offset, bar_fill_width, bar_height),
                )

            # Border
            pygame.draw.rect(
                surface,
                (150, 150, 150, 255),
                (bar_x, y_offset, bar_width, bar_height),
                1,
            )

            # Draw percentage text (compact)
            if i in result.masked_actions:
                status_text = "X"
                status_color = DEADLY_COLOR
            else:
                status_text = f"{prob * 100:.0f}"
                status_color = TEXT_COLOR

            status_surf = action_font.render(status_text, True, status_color)
            surface.blit(status_surf, (bar_x + bar_width + 4, y_offset))

            y_offset += bar_height + bar_spacing

        # Cache the surface for next frame
        self.cached_terminal_velocity_surface = surface
        return surface

    def _draw_reachable_walls(self, reachable_walls_info: dict) -> Optional[pygame.Surface]:
        """Draw reachable wall segments visualization for action masking debug.

        Args:
            reachable_walls_info: Dictionary containing:
                - wall_segments: list of ((x1, y1), (x2, y2)) wall segment tuples
                - ninja_position: tuple of (x, y) ninja position

        Returns:
            pygame.Surface with reachable wall segments visualization
        """
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        wall_segments = reachable_walls_info.get("wall_segments", [])
        ninja_pos = reachable_walls_info.get("ninja_position", (0, 0))

        if not wall_segments:
            return surface

        # Colors
        WALL_COLOR = (255, 100, 255, 255)  # Magenta for reachable walls
        WALL_THICK = 3

        # Draw each wall segment
        for (x1, y1), (x2, y2) in wall_segments:
            # Convert world coordinates to screen coordinates
            screen_x1 = int(x1 * self.adjust) + self.tile_x_offset
            screen_y1 = int(y1 * self.adjust) + self.tile_y_offset
            screen_x2 = int(x2 * self.adjust) + self.tile_x_offset
            screen_y2 = int(y2 * self.adjust) + self.tile_y_offset

            # Draw the wall segment
            pygame.draw.line(
                surface,
                WALL_COLOR,
                (screen_x1, screen_y1),
                (screen_x2, screen_y2),
                WALL_THICK,
            )

        return surface

    def _draw_action_mask(self, action_mask_info: dict) -> Optional[pygame.Surface]:
        """Draw action mask visualization showing allowed vs masked actions.

        Args:
            action_mask_info: Dictionary containing:
                - action_mask: list or array of 6 booleans (True=allowed, False=masked)
                - ninja_position: tuple of (x, y) ninja position

        Returns:
            pygame.Surface with action mask visualization

        Layout (keyboard-style grid):
        Row 1: [JL] [JP] [JR]  <- Jump+Left(4), Jump(3), Jump+Right(5)
        Row 2: [LT] [NO] [RT]  <- Left(1), NOOP(0), Right(2)
        """
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        action_mask = action_mask_info.get("action_mask")
        ninja_pos = action_mask_info.get("ninja_position", (0, 0))
        last_action = action_mask_info.get("last_action")

        if action_mask is None or len(action_mask) != 6:
            return surface

        # Colors
        ALLOWED_COLOR = (100, 255, 100, 255)  # Green for allowed
        MASKED_COLOR = (255, 100, 100, 255)   # Red for masked
        INVALID_ACTION_COLOR = (255, 255, 0, 255)  # Yellow for invalid action taken
        TEXT_COLOR = (255, 255, 255, 255)     # White text
        BG_COLOR = (0, 0, 0, 220)             # Semi-transparent black background
        BORDER_COLOR = (150, 150, 255, 255)   # Light blue border

        # Determine ninja quadrant for positioning
        ninja_screen_x = int(ninja_pos[0] * self.adjust) + self.tile_x_offset
        ninja_screen_y = int(ninja_pos[1] * self.adjust) + self.tile_y_offset

        screen_center_x = self.screen.get_width() / 2
        screen_center_y = self.screen.get_height() / 2

        ninja_in_left = ninja_screen_x < screen_center_x
        ninja_in_top = ninja_screen_y < screen_center_y

        # Compact keyboard layout dimensions
        key_size = 20         # Size of each key box
        key_spacing = 3       # Space between keys
        panel_padding = 8
        
        # Calculate panel dimensions
        # 3 keys wide, 2 keys tall
        panel_width = 3 * key_size + 2 * key_spacing + 2 * panel_padding
        panel_height = 2 * key_size + key_spacing + 2 * panel_padding + 20  # +20 for title

        # Position panel close to ninja
        offset_distance = 30  # Distance from ninja (horizontal and vertical)
        screen_margin = 10

        if ninja_in_left:
            panel_x = ninja_screen_x + offset_distance
            if panel_x + panel_width > self.screen.get_width() - screen_margin:
                panel_x = self.screen.get_width() - panel_width - screen_margin
        else:
            panel_x = ninja_screen_x - offset_distance - panel_width
            if panel_x < screen_margin:
                panel_x = screen_margin

        if ninja_in_top:
            # Ninja in top half - place panel below ninja
            panel_y = ninja_screen_y + offset_distance
            if panel_y + panel_height > self.screen.get_height() - screen_margin:
                panel_y = self.screen.get_height() - panel_height - screen_margin
        else:
            # Ninja in bottom half - place panel above ninja
            panel_y = ninja_screen_y - offset_distance - panel_height
            if panel_y < screen_margin:
                panel_y = screen_margin

        # Draw background panel
        pygame.draw.rect(
            surface,
            BG_COLOR,
            (panel_x, panel_y, panel_width, panel_height),
            border_radius=4,
        )
        pygame.draw.rect(
            surface,
            BORDER_COLOR,
            (panel_x, panel_y, panel_width, panel_height),
            2,
            border_radius=4,
        )

        # Fonts
        try:
            title_font = pygame.font.Font(None, 18)
            key_font = pygame.font.Font(None, 14)
        except pygame.error:
            title_font = pygame.font.SysFont("arial", 16)
            key_font = pygame.font.SysFont("arial", 12)

        # Draw title
        title_text = "Action Mask"
        title_surf = title_font.render(title_text, True, TEXT_COLOR)
        title_x = panel_x + (panel_width - title_surf.get_width()) // 2
        surface.blit(title_surf, (title_x, panel_y + 4))

        # Key layout mapping: (row, col) -> (action_idx, label)
        # Row 0 (top): Jump actions
        # Row 1 (bottom): Movement actions
        key_layout = [
            [(4, "JL"), (3, "JP"), (5, "JR")],  # Row 0: Jump+Left, Jump, Jump+Right
            [(1, "LT"), (0, "NO"), (2, "RT")],  # Row 1: Left, NOOP, Right
        ]

        # Starting Y position for keys (below title)
        keys_start_y = panel_y + 24

        # Draw keys in grid
        for row_idx, row in enumerate(key_layout):
            for col_idx, (action_idx, label) in enumerate(row):
                # Calculate key position
                key_x = panel_x + panel_padding + col_idx * (key_size + key_spacing)
                key_y = keys_start_y + row_idx * (key_size + key_spacing)

                # Determine color based on mask and whether this action was just taken
                is_allowed = action_mask[action_idx]
                is_last_action = (last_action is not None and last_action == action_idx)
                
                # Yellow if this masked action was just taken, otherwise green/red
                if is_last_action and not is_allowed:
                    key_color = INVALID_ACTION_COLOR
                    border_color = (200, 200, 0, 255)  # Dark yellow border
                elif is_allowed:
                    key_color = ALLOWED_COLOR
                    border_color = (255, 255, 255, 200)  # White border
                else:
                    key_color = MASKED_COLOR
                    border_color = (150, 0, 0, 255)  # Dark red border

                # Draw key background
                key_rect = pygame.Rect(key_x, key_y, key_size, key_size)
                pygame.draw.rect(surface, key_color, key_rect, border_radius=2)
                
                # Draw key border
                pygame.draw.rect(surface, border_color, key_rect, 1, border_radius=2)

                # Draw label
                label_surf = key_font.render(label, True, (0, 0, 0, 255))
                label_x = key_x + (key_size - label_surf.get_width()) // 2
                label_y = key_y + (key_size - label_surf.get_height()) // 2
                surface.blit(label_surf, (label_x, label_y))

        return surface

    def _draw_path_aware(self, path_aware_info: dict) -> Optional[pygame.Surface]:
        """Draw path-aware debugging information (adjacency graph, path distances, blocked entities).

        Args:
            path_aware_info: Dictionary containing:
                - show_distances: bool - whether to show path distances
                - show_adjacency: bool - whether to show adjacency graph
                - show_blocked: bool - whether to show blocked entities
                - graph_data: dict - graph adjacency data
                - entity_mask: EntityMask - entity blocking data
                - ninja_position: tuple - current ninja position
                - entities: list - level entities for switch/exit location

        Returns:
            pygame.Surface with path-aware visualization, or None if no graph data available
        """
        if not path_aware_info.get("graph_data"):
            return None

        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        graph_data = path_aware_info["graph_data"]
        adjacency = graph_data.get("adjacency", {})
        ninja_pos = path_aware_info.get("ninja_position", (0, 0))
        entities = path_aware_info.get("entities", [])

        # Import shared utilities from pathfinding modules
        from nclone.graph.reachability.pathfinding_utils import (
            find_closest_node_to_position,
            bfs_distance_from_start,
            find_shortest_path_with_parents,
            extract_spatial_lookups_from_graph_data,
        )

        # Extract spatial hash and subcell lookup from graph_data if available
        spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
            graph_data
        )

        show_adjacency = path_aware_info.get("show_adjacency", False)
        show_distances = path_aware_info.get("show_distances", False)
        show_blocked = path_aware_info.get("show_blocked", False)
        show_paths = path_aware_info.get("show_paths", False)

        # Colors for visualization
        NODE_COLOR = (100, 255, 100, 180)  # Green nodes
        EDGE_COLOR = (255, 255, 100, 100)  # Yellow edges
        BLOCKED_NODE_COLOR = (255, 50, 50, 180)  # Red for blocked
        NINJA_NODE_COLOR = (100, 150, 255, 255)  # Blue for ninja position
        SWITCH_NODE_COLOR = (100, 255, 100, 255)  # Bright green for switches
        EXIT_NODE_COLOR = (255, 200, 100, 255)  # Orange for exits
        SWITCH_PATH_COLOR = (50, 255, 50, 220)  # Bright green path to switch
        EXIT_PATH_COLOR = (255, 180, 50, 220)  # Bright orange path to exit
        TEXT_COLOR = (255, 255, 255, 255)  # White text

        # Helper to find ninja node using shared utility
        def find_ninja_node(ninja_pos, adjacency):
            """Find the node that represents the ninja's current position.
            Uses shared utility function for consistency."""
            if not ninja_pos or not adjacency:
                return None

            # Use shared utility with tight threshold for ninja node finding
            # First try exact match (within 5 pixels)
            for pos in adjacency.keys():
                x, y = pos
                if abs(x + 24 - ninja_pos[0]) < 5 and abs(y + 24 - ninja_pos[1]) < 5:
                    return pos

            # Fallback: use shared utility with relaxed threshold
            return find_closest_node_to_position(
                ninja_pos,
                adjacency,
                threshold=50.0,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
            )

        # Compute reachable set dynamically from current ninja position
        def compute_reachability_from_position(start_pos, adjacency):
            """Compute reachable nodes from a starting position via flood-fill."""
            from collections import deque

            # Use the same logic as blue node highlighting to find starting node
            start_node = find_ninja_node(start_pos, adjacency)

            if not start_node:
                return None

            # Flood-fill from ninja node
            reachable = set()
            visited = {start_node}
            queue = deque([start_node])
            reachable.add(start_node)

            while queue:
                current = queue.popleft()
                neighbors = adjacency.get(current, [])
                for neighbor_pos, _ in neighbors:
                    if neighbor_pos not in visited:
                        visited.add(neighbor_pos)
                        reachable.add(neighbor_pos)
                        queue.append(neighbor_pos)

            return reachable

        # Compute reachability from current ninja position
        reachable = (
            compute_reachability_from_position(ninja_pos, adjacency)
            if ninja_pos
            else None
        )

        # Get blocked positions if available
        blocked_positions = set()
        blocked_edges = set()
        if show_blocked and graph_data.get("blocked_positions"):
            blocked_positions = graph_data["blocked_positions"]
        if show_blocked and graph_data.get("blocked_edges"):
            blocked_edges = graph_data["blocked_edges"]

        # Draw adjacency graph edges first (so they appear behind nodes)
        if show_adjacency:
            for pos, neighbors in adjacency.items():
                # Skip unreachable nodes
                if reachable is not None and pos not in reachable:
                    continue
                if not neighbors:
                    continue
                x1, y1 = pos
                screen_x1 = int(x1 * self.adjust + self.tile_x_offset) + 24
                screen_y1 = int(y1 * self.adjust + self.tile_y_offset) + 24

                for neighbor_info in neighbors:
                    # neighbor_info is ((x, y), cost)
                    neighbor_pos, cost = neighbor_info
                    # Skip edges to unreachable nodes
                    if reachable is not None and neighbor_pos not in reachable:
                        continue
                    x2, y2 = neighbor_pos
                    screen_x2 = int(x2 * self.adjust + self.tile_x_offset) + 24
                    screen_y2 = int(y2 * self.adjust + self.tile_y_offset) + 24

                    # Check if edge is blocked
                    edge_blocked = (pos, neighbor_pos) in blocked_edges or (
                        neighbor_pos,
                        pos,
                    ) in blocked_edges
                    edge_color = (255, 0, 0, 80) if edge_blocked else EDGE_COLOR

                    # Draw line
                    pygame.draw.line(
                        surface,
                        edge_color,
                        (screen_x1, screen_y1),
                        (screen_x2, screen_y2),
                        1,
                    )

        # Draw nodes
        if show_adjacency or show_blocked:
            try:
                font = pygame.font.Font(None, 16)
            except pygame.error:
                font = pygame.font.SysFont("arial", 14)

            for pos in adjacency.keys():
                # Skip unreachable nodes
                if reachable is not None and pos not in reachable:
                    continue
                x, y = pos
                screen_x = int(x * self.adjust + self.tile_x_offset) + 24
                screen_y = int(y * self.adjust + self.tile_y_offset) + 24

                # Determine node color
                if pos in blocked_positions:
                    node_color = BLOCKED_NODE_COLOR
                elif abs(x + 24 - ninja_pos[0]) < 5 and abs(y + 24 - ninja_pos[1]) < 5:
                    node_color = NINJA_NODE_COLOR
                else:
                    node_color = NODE_COLOR

                # Draw node circle
                pygame.draw.circle(surface, node_color, (screen_x, screen_y), 3)

        # Draw path distances from ninja position
        if show_distances and ninja_pos:
            try:
                font = pygame.font.Font(None, 16)
            except pygame.error:
                font = pygame.font.SysFont("arial", 14)

            # Use the same logic as blue node highlighting to find starting node
            closest_node = find_ninja_node(ninja_pos, adjacency)

            # Only proceed if we found a valid ninja node and it's reachable
            if closest_node and (reachable is None or closest_node in reachable):
                # Use shared BFS utility to calculate distances
                distances, _ = bfs_distance_from_start(closest_node, None, adjacency)

                # Draw distances on screen (only for reachable nodes)
                for pos, dist in distances.items():
                    # Skip unreachable nodes
                    if reachable is not None and pos not in reachable:
                        continue
                    if dist > 1000:  # Don't show very far nodes
                        continue
                    x, y = pos
                    screen_x = int(x * self.adjust + self.tile_x_offset)
                    screen_y = int(y * self.adjust + self.tile_y_offset)

                    # Draw distance text
                    text = font.render(f"{int(dist)}", True, TEXT_COLOR)
                    surface.blit(text, (screen_x + 5, screen_y - 10))

        # Find and visualize switches and exits
        switch_positions = []
        exit_positions = []
        exit_switch_activated = False

        for entity in entities:
            entity_type = entity.get("type")
            if entity_type == EntityType.EXIT_SWITCH:
                switch_pos = (entity.get("x", 0), entity.get("y", 0))
                # Check if switch is still active (not collected)
                # In nclone: active=True means NOT collected, active=False means collected
                is_active = entity.get("active", True)
                if is_active:
                    # Switch not yet collected - add as goal
                    switch_positions.append(switch_pos)
                else:
                    # Switch has been collected
                    exit_switch_activated = True
            elif entity_type == EntityType.EXIT_DOOR:
                exit_positions.append((entity.get("x", 0), entity.get("y", 0)))

        # Draw switch and exit markers on graph
        if show_adjacency and (switch_positions or exit_positions):
            for switch_pos in switch_positions:
                x, y = switch_pos
                screen_x = int(x * self.adjust + self.tile_x_offset)
                screen_y = int(y * self.adjust + self.tile_y_offset)
                pygame.draw.circle(surface, SWITCH_NODE_COLOR, (screen_x, screen_y), 6)
                pygame.draw.circle(surface, (0, 0, 0, 255), (screen_x, screen_y), 6, 2)

            for exit_pos in exit_positions:
                x, y = exit_pos
                screen_x = int(x * self.adjust + self.tile_x_offset)
                screen_y = int(y * self.adjust + self.tile_y_offset)
                pygame.draw.circle(surface, EXIT_NODE_COLOR, (screen_x, screen_y), 6)
                pygame.draw.circle(surface, (0, 0, 0, 255), (screen_x, screen_y), 6, 2)

        # Helper function to get segment color based on direction
        def get_segment_color(segment_type, segment_sign):
            """Get color for a path segment based on its direction.
            
            Args:
                segment_type: 'horizontal' or 'vertical'
                segment_sign: +1 or -1
            
            Returns:
                RGB color tuple
            """
            if segment_type == 'horizontal':
                if segment_sign == 1:
                    return (0, 200, 255, 220)  # Cyan (right)
                else:
                    return (255, 50, 255, 220)  # Magenta (left)
            else:  # vertical
                if segment_sign == 1:
                    return (255, 200, 0, 220)  # Gold (down)
                else:
                    return (100, 255, 100, 220)  # Light green (up)
        
        # Helper function to draw an arrow
        def draw_arrow(surface, color, start_pos, end_pos, arrow_size=8):
            """Draw an arrow from start to end position.
            
            Args:
                surface: Pygame surface to draw on
                color: Arrow color
                start_pos: (x, y) start position
                end_pos: (x, y) end position
                arrow_size: Size of the arrowhead in pixels
            """
            import math
            
            # Calculate direction vector
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            length = math.sqrt(dx * dx + dy * dy)
            
            if length < 0.01:  # Avoid division by zero
                return
            
            # Normalize direction
            dir_x = dx / length
            dir_y = dy / length
            
            # Calculate arrow midpoint (where we'll draw the arrowhead)
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
            
            # Calculate arrowhead points
            # Perpendicular vector for arrowhead wings
            perp_x = -dir_y
            perp_y = dir_x
            
            # Arrow tip
            tip_x = mid_x + dir_x * arrow_size * 0.5
            tip_y = mid_y + dir_y * arrow_size * 0.5
            
            # Arrow wing points
            wing1_x = mid_x - dir_x * arrow_size * 0.5 + perp_x * arrow_size * 0.4
            wing1_y = mid_y - dir_y * arrow_size * 0.5 + perp_y * arrow_size * 0.4
            
            wing2_x = mid_x - dir_x * arrow_size * 0.5 - perp_x * arrow_size * 0.4
            wing2_y = mid_y - dir_y * arrow_size * 0.5 - perp_y * arrow_size * 0.4
            
            # Draw filled triangle for arrowhead
            arrow_points = [
                (int(tip_x), int(tip_y)),
                (int(wing1_x), int(wing1_y)),
                (int(wing2_x), int(wing2_y))
            ]
            pygame.draw.polygon(surface, color, arrow_points)
            
            # Draw outline for better visibility
            pygame.draw.polygon(surface, (0, 0, 0, 255), arrow_points, 1)

        # Use shared pathfinding function
        find_shortest_path = find_shortest_path_with_parents

        # Get segment visualization flag
        show_segments = path_aware_info.get("show_segments", False)
        segments = path_aware_info.get("segments")
        current_segment_index = path_aware_info.get("current_segment_index")

        # Draw paths to goals if enabled
        if show_paths and ninja_pos and adjacency and reachable:
            # Get level data for cache validation
            from nclone.graph.level_data import LevelData

            # Try to get level_data from path_aware_info, fallback to creating from entities
            level_data = path_aware_info.get("level_data")
            if level_data is None and entities:
                # Create minimal LevelData for cache validation if not provided
                import numpy as np

                level_data = LevelData(
                    tiles=np.zeros((1, 1), dtype=np.int32),
                    entities=entities,
                    start_position=(0, 0),
                )

            # Try to get cached paths
            cached_data = None
            if level_data and hasattr(self, "_path_visualization_cache"):
                cached_data = self._path_visualization_cache.get_cached_paths(
                    ninja_pos,
                    adjacency,
                    level_data,
                    switch_positions,
                    exit_positions,
                    exit_switch_activated,
                )

            if cached_data is None:
                # Need to recompute paths
                closest_node = find_ninja_node(ninja_pos, adjacency)
                switch_path = None
                exit_path = None
                switch_node = None
                exit_node = None

                # Only proceed if we found a valid ninja node and it's reachable
                if closest_node and closest_node in reachable:
                    # Draw path to nearest switch (if not yet activated)
                    if switch_positions:  # Only if there are uncollected switches
                        for switch_pos in switch_positions:
                            # Find closest REACHABLE node to switch using shared utility
                            # Filter adjacency to only reachable nodes first
                            reachable_adjacency = {
                                k: v for k, v in adjacency.items() if k in reachable
                            }
                            switch_node = find_closest_node_to_position(
                                switch_pos,
                                reachable_adjacency,
                                threshold=50.0,
                                spatial_hash=spatial_hash,
                                subcell_lookup=subcell_lookup,
                            )

                            if switch_node:
                                path, _ = find_shortest_path(
                                    closest_node, switch_node, adjacency
                                )
                                if path:
                                    switch_path = path
                                    break  # Only draw path to nearest switch

                    # Draw path to nearest exit ONLY if switch has been activated
                    if exit_switch_activated and exit_positions:
                        for exit_pos in exit_positions:
                            # Find closest REACHABLE node to exit using shared utility
                            # Filter adjacency to only reachable nodes first
                            reachable_adjacency = {
                                k: v for k, v in adjacency.items() if k in reachable
                            }
                            exit_node = find_closest_node_to_position(
                                exit_pos,
                                reachable_adjacency,
                                threshold=50.0,
                                spatial_hash=spatial_hash,
                                subcell_lookup=subcell_lookup,
                            )

                            if exit_node:
                                path, _ = find_shortest_path(
                                    closest_node, exit_node, adjacency
                                )
                                if path:
                                    exit_path = path
                                    break  # Only draw path to nearest exit

                # Cache the computed paths
                if level_data and hasattr(self, "_path_visualization_cache"):
                    self._path_visualization_cache.cache_paths(
                        ninja_pos,
                        adjacency,
                        level_data,
                        switch_positions,
                        exit_positions,
                        exit_switch_activated,
                        closest_node,
                        switch_path,
                        exit_path,
                        switch_node,
                        exit_node,
                    )

                cached_data = {
                    "closest_node": closest_node,
                    "switch_path": switch_path,
                    "exit_path": exit_path,
                    "switch_node": switch_node,
                    "exit_node": exit_node,
                }

            # Extract cached data
            closest_node = cached_data.get("closest_node")
            switch_path = cached_data.get("switch_path")
            exit_path = cached_data.get("exit_path")

            # Draw paths (from cache or newly computed)
            if closest_node and cached_data:
                # Draw switch path
                if switch_path:
                    # Draw path with single color
                    for i in range(len(switch_path) - 1):
                        node1 = switch_path[i]
                        node2 = switch_path[i + 1]

                        # Verify nodes are adjacent in the graph
                        neighbors = adjacency.get(node1, [])
                        is_adjacent = any(n[0] == node2 for n in neighbors)

                        if is_adjacent:
                            x1, y1 = node1
                            x2, y2 = node2
                            # Add +24 offset to match node visualization (accounts for tile padding)
                            screen_x1 = int(x1 * self.adjust + self.tile_x_offset) + 24
                            screen_y1 = int(y1 * self.adjust + self.tile_y_offset) + 24
                            screen_x2 = int(x2 * self.adjust + self.tile_x_offset) + 24
                            screen_y2 = int(y2 * self.adjust + self.tile_y_offset) + 24
                            pygame.draw.line(
                                surface,
                                    SWITCH_PATH_COLOR,
                                    (screen_x1, screen_y1),
                                    (screen_x2, screen_y2),
                                    3,
                                )

                # Draw exit path
                if exit_path:
                    # Draw path with single color
                    for i in range(len(exit_path) - 1):
                        node1 = exit_path[i]
                        node2 = exit_path[i + 1]

                        # Verify nodes are adjacent in the graph
                        neighbors = adjacency.get(node1, [])
                        is_adjacent = any(n[0] == node2 for n in neighbors)

                        if is_adjacent:
                            x1, y1 = node1
                            x2, y2 = node2
                            # Add +24 offset to match node visualization (accounts for tile padding)
                            screen_x1 = int(x1 * self.adjust + self.tile_x_offset) + 24
                            screen_y1 = int(y1 * self.adjust + self.tile_y_offset) + 24
                            screen_x2 = int(x2 * self.adjust + self.tile_x_offset) + 24
                            screen_y2 = int(y2 * self.adjust + self.tile_y_offset) + 24
                            pygame.draw.line(
                                surface,
                                EXIT_PATH_COLOR,
                                (screen_x1, screen_y1),
                                (screen_x2, screen_y2),
                                3,
                            )

        # Calculate and display switch/exit distances
        if show_distances and ninja_pos and adjacency:
            # Extract ninja position for screen coordinates
            ninja_x, ninja_y = ninja_pos

            # Use the same logic as blue node highlighting to find starting node
            closest_node = find_ninja_node(ninja_pos, adjacency)

            if closest_node:
                # Calculate distance to nearest switch using shared utilities
                switch_dist = float("inf")
                for switch_pos in switch_positions:
                    # Find closest node to switch using shared utility
                    switch_node = find_closest_node_to_position(
                        switch_pos,
                        adjacency,
                        threshold=50.0,
                        spatial_hash=spatial_hash,
                        subcell_lookup=subcell_lookup,
                    )

                    if switch_node is not None:
                        # BFS from ninja to switch using shared utility
                        distances, target_dist = bfs_distance_from_start(
                            closest_node, switch_node, adjacency
                        )

                        if target_dist is not None:
                            switch_dist = min(switch_dist, target_dist)

                # Calculate distance to nearest exit using shared utilities
                exit_dist = float("inf")
                for exit_pos in exit_positions:
                    # Find closest node to exit using shared utility
                    exit_node = find_closest_node_to_position(
                        exit_pos,
                        adjacency,
                        threshold=50.0,
                        spatial_hash=spatial_hash,
                        subcell_lookup=subcell_lookup,
                    )

                    if exit_node is not None:
                        # BFS from ninja to exit using shared utility
                        distances, target_dist = bfs_distance_from_start(
                            closest_node, exit_node, adjacency
                        )

                        if target_dist is not None:
                            exit_dist = min(exit_dist, target_dist)

                # Draw info box with switch/exit distances
                try:
                    box_font = pygame.font.Font(None, 20)
                except pygame.error:
                    box_font = pygame.font.SysFont("monospace", 16, bold=True)

                ninja_screen_x = int(ninja_x * self.adjust + self.tile_x_offset)
                ninja_screen_y = int(ninja_y * self.adjust + self.tile_y_offset)

                # Draw background box
                box_width = 180
                box_height = 60
                box_x = ninja_screen_x + 20
                box_y = ninja_screen_y - 40

                # Keep box on screen
                if box_x + box_width > self.screen.get_width():
                    box_x = ninja_screen_x - box_width - 20
                if box_y < 0:
                    box_y = ninja_screen_y + 20

                pygame.draw.rect(
                    surface,
                    (0, 0, 0, 200),
                    (box_x, box_y, box_width, box_height),
                    border_radius=5,
                )
                pygame.draw.rect(
                    surface,
                    (100, 200, 255, 255),
                    (box_x, box_y, box_width, box_height),
                    2,
                    border_radius=5,
                )

                # Draw text
                switch_text = f"Switch: {int(switch_dist) if switch_dist != float('inf') else '∞'}"
                exit_text = (
                    f"Exit: {int(exit_dist) if exit_dist != float('inf') else '∞'}"
                )

                switch_surf = box_font.render(switch_text, True, SWITCH_NODE_COLOR)
                exit_surf = box_font.render(exit_text, True, EXIT_NODE_COLOR)

                surface.blit(switch_surf, (box_x + 10, box_y + 10))
                surface.blit(exit_surf, (box_x + 10, box_y + 35))

        # Draw legend
        if show_adjacency:
            try:
                legend_font = pygame.font.Font(None, 18)
            except pygame.error:
                legend_font = pygame.font.SysFont("arial", 14)

            legend_x = 20
            legend_y = 20
            legend_width = 180
            legend_height = 110

            # Background
            pygame.draw.rect(
                surface,
                (0, 0, 0, 200),
                (legend_x, legend_y, legend_width, legend_height),
                border_radius=5,
            )
            pygame.draw.rect(
                surface,
                (100, 200, 255, 255),
                (legend_x, legend_y, legend_width, legend_height),
                2,
                border_radius=5,
            )

            # Title
            title_surf = legend_font.render("Adjacency Graph:", True, TEXT_COLOR)
            surface.blit(title_surf, (legend_x + 10, legend_y + 10))

            # Legend items
            legend_items = [
                ("● Ninja", NINJA_NODE_COLOR),
                ("● Switch", SWITCH_NODE_COLOR),
                ("● Exit", EXIT_NODE_COLOR),
                ("● Tile", NODE_COLOR),
            ]

            y_offset = legend_y + 35
            for text, color in legend_items:
                text_surf = legend_font.render(text, True, color)
                surface.blit(text_surf, (legend_x + 10, y_offset))
                y_offset += 20

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

        # Draw tile types if provided
        if debug_info and "tile_types" in debug_info:
            tile_types_surface = self._draw_tile_types()
            surface.blit(tile_types_surface, (0, 0))

        # Draw path-aware visualization if provided
        if debug_info and "path_aware" in debug_info:
            path_aware_surface = self._draw_path_aware(debug_info["path_aware"])
            if path_aware_surface:
                surface.blit(path_aware_surface, (0, 0))

        # Draw mine predictor visualization if provided
        if debug_info and "mine_predictor" in debug_info:
            mine_predictor_surface = self._draw_mine_predictor(
                debug_info["mine_predictor"]
            )
            if mine_predictor_surface:
                surface.blit(mine_predictor_surface, (0, 0))

        # Draw mine death probability visualization if provided
        # Always show to account for terminal velocity deaths even without mines
        if debug_info and "death_probability" in debug_info:
            death_prob_surface = self._draw_death_probability(
                debug_info["death_probability"]
            )
            if death_prob_surface:
                surface.blit(death_prob_surface, (0, 0))

        # Draw terminal velocity death probability visualization if provided
        if debug_info and "terminal_velocity_probability" in debug_info:
            terminal_velocity_prob_surface = self._draw_terminal_velocity_probability(
                debug_info["terminal_velocity_probability"]
            )
            if terminal_velocity_prob_surface:
                surface.blit(terminal_velocity_prob_surface, (0, 0))

        # Draw reachable wall segments visualization if provided
        if debug_info and "reachable_walls" in debug_info:
            reachable_walls_surface = self._draw_reachable_walls(debug_info["reachable_walls"])
            if reachable_walls_surface:
                surface.blit(reachable_walls_surface, (0, 0))

        # Draw action mask visualization if provided
        if debug_info and "action_mask" in debug_info:
            action_mask_surface = self._draw_action_mask(debug_info["action_mask"])
            if action_mask_surface:
                surface.blit(action_mask_surface, (0, 0))

        # Draw subgoal visualization if enabled
        if self.subgoal_debug_enabled:
            ninja_pos = self._get_ninja_position()

            # Render even if no subgoals to show reachability areas
            surface = self.subgoal_visualizer.render_subgoals_overlay(
                surface,
                self.current_subgoals if self.current_subgoals else [],
                ninja_pos,
                self.current_reachable_positions
                if self.current_reachable_positions
                else set(),
                self.current_subgoal_plan,
                self.tile_x_offset,
                self.tile_y_offset,
                self.adjust,
            )

        return surface

    def _get_ninja_position(self) -> Tuple[float, float]:
        """Get current ninja position from simulation."""
        try:
            if hasattr(self.sim, "ninja"):
                return (self.sim.ninja.x, self.sim.ninja.y)
            else:
                return (100.0, 100.0)  # Fallback position
        except Exception:
            return (100.0, 100.0)  # Fallback position

    def set_subgoal_debug_enabled(self, enabled: bool):
        """Enable or disable subgoal visualization."""
        self.subgoal_debug_enabled = enabled

    def set_subgoal_data(
        self,
        subgoals: List[Subgoal],
        plan: Optional[SubgoalPlan] = None,
        reachable_positions: Optional[Set[Tuple[int, int]]] = None,
    ):
        """Set subgoal data for visualization."""
        self.current_subgoals = subgoals
        self.current_subgoal_plan = plan
        self.current_reachable_positions = reachable_positions

    def export_subgoal_visualization(
        self, filename: str = "subgoal_export.png"
    ) -> bool:
        """Export current subgoal visualization to image file."""
        ninja_pos = self._get_ninja_position()
        level_dimensions = (FULL_MAP_WIDTH, FULL_MAP_HEIGHT)

        # Allow export even with empty subgoals to show basic visualization
        subgoals = self.current_subgoals if self.current_subgoals else []

        # Get level data and entities from sim
        level_data = None
        entities = None

        # Try to get tile data from simulator
        if hasattr(self.sim, "level_data"):
            level_data = self.sim.level_data
        elif hasattr(self.sim, "tiles"):
            level_data = self.sim.tiles
        elif hasattr(self.sim, "tile_dic"):
            # Pass tile_dic directly for proper tile rendering
            level_data = self.sim.tile_dic

        # Try to get entities from simulator
        if hasattr(self.sim, "entities"):
            entities = self.sim.entities
        elif hasattr(self.sim, "entity_list"):
            entities = self.sim.entity_list
        elif hasattr(self.sim, "entity_dic"):
            # Flatten entity_dic to list of entities
            entities = []
            for entity_list in self.sim.entity_dic.values():
                entities.extend(entity_list)

        return self.subgoal_visualizer.export_subgoal_visualization(
            subgoals,
            ninja_pos,
            level_dimensions,
            self.current_reachable_positions,
            self.current_subgoal_plan,
            filename,
            level_data=level_data,
            entities=entities,
        )

    def _convert_tile_dic_to_array(self):
        """Convert simulator's tile_dic to numpy array for visualization."""
        try:
            import numpy as np

            # The simulator uses a 44x25 tile grid (including borders)
            width_tiles = 44
            height_tiles = 25

            tiles = np.zeros((height_tiles, width_tiles), dtype=int)

            # Fill array from tile_dic
            for (x, y), tile_value in self.sim.tile_dic.items():
                if 0 <= x < width_tiles and 0 <= y < height_tiles:
                    tiles[y, x] = tile_value
            return tiles
        except Exception as e:
            print(f"Error converting tile_dic to array: {e}")
            import traceback

            traceback.print_exc()
            return None

    def update_subgoal_visualization_config(self, **kwargs):
        """Update subgoal visualization configuration."""
        self.subgoal_visualizer.update_config(**kwargs)

    def set_subgoal_visualization_mode(self, mode_name: str):
        """Set subgoal visualization mode by name."""
        from .graph.subgoal_visualizer import SubgoalVisualizationMode

        mode_map = {
            "basic": SubgoalVisualizationMode.BASIC,
            "detailed": SubgoalVisualizationMode.DETAILED,
            "reachability": SubgoalVisualizationMode.REACHABILITY,
        }

        if mode_name in mode_map:
            self.subgoal_visualizer.set_mode(mode_map[mode_name])

    def draw_adjacency_graph(self, graph_data: dict) -> pygame.Surface:
        """
        Draw adjacency graph overlay showing tile connectivity.

        Args:
            graph_data: Dict with 'nodes' and 'edges' keys
            ninja_pos: Current ninja position (x, y)

        Returns:
            Surface with adjacency graph visualization
        """
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        if not graph_data or "nodes" not in graph_data or "edges" not in graph_data:
            return surface

        nodes = graph_data["nodes"]
        edges = graph_data["edges"]

        # Draw edges first (so nodes are on top)
        for edge in edges:
            pos1 = edge.get("pos1")
            pos2 = edge.get("pos2")
            if pos1 and pos2:
                screen_x1 = int(pos1[0] * self.adjust) + self.tile_x_offset
                screen_y1 = int(pos1[1] * self.adjust) + self.tile_y_offset
                screen_x2 = int(pos2[0] * self.adjust) + self.tile_x_offset
                screen_y2 = int(pos2[1] * self.adjust) + self.tile_y_offset

                # Draw edge line
                pygame.draw.line(
                    surface,
                    (150, 150, 255, 100),
                    (screen_x1, screen_y1),
                    (screen_x2, screen_y2),
                    2,
                )

        # Draw nodes
        for node in nodes:
            pos = node.get("pos")
            node_type = node.get("type", "normal")

            if pos:
                screen_x = int(pos[0] * self.adjust) + self.tile_x_offset
                screen_y = int(pos[1] * self.adjust) + self.tile_y_offset

                # Choose color based on node type
                if node_type == "ninja":
                    color = (60, 220, 255, 255)
                    radius = 8
                elif node_type == "switch":
                    color = (100, 255, 100, 255)
                    radius = 6
                elif node_type == "exit":
                    color = (255, 200, 100, 255)
                    radius = 6
                else:
                    color = (200, 200, 255, 180)
                    radius = 4

                # Draw node
                pygame.draw.circle(surface, color, (screen_x, screen_y), radius)
                pygame.draw.circle(
                    surface, (255, 255, 255, 255), (screen_x, screen_y), radius, 1
                )

        # Draw legend
        font = pygame.font.SysFont("monospace", 14)
        legend_x = 10
        legend_y = self.screen.get_height() - 120

        pygame.draw.rect(
            surface, (0, 0, 0, 200), (legend_x, legend_y, 180, 110), border_radius=5
        )
        pygame.draw.rect(
            surface,
            (100, 200, 255, 255),
            (legend_x, legend_y, 180, 110),
            2,
            border_radius=5,
        )

        title = font.render("Adjacency Graph:", True, (255, 255, 255))
        surface.blit(title, (legend_x + 10, legend_y + 5))

        # Legend items
        legend_items = [
            ((60, 220, 255), "Ninja"),
            ((100, 255, 100), "Switch"),
            ((255, 200, 100), "Exit"),
            ((200, 200, 255), "Tile"),
        ]

        for i, (color, label) in enumerate(legend_items):
            y_pos = legend_y + 25 + i * 20
            pygame.draw.circle(surface, color, (legend_x + 20, y_pos + 7), 5)
            text = font.render(label, True, (255, 255, 255))
            surface.blit(text, (legend_x + 35, y_pos))

        return surface
