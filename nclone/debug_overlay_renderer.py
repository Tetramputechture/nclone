import pygame
from typing import Optional, Tuple
from .constants.physics_constants import (
    TILE_PIXEL_SIZE,
    FULL_MAP_WIDTH,
    FULL_MAP_HEIGHT,
)
from .constants.entity_types import EntityType


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

        # Pathfinding cache: use shared visualization cache
        from nclone.graph.reachability.path_visualization_cache import (
            PathVisualizationCache,
        )
        from nclone.graph.reachability.mine_proximity_cache import (
            MineProximityCostCache,
        )

        self._path_visualization_cache = PathVisualizationCache()
        self._mine_proximity_cache = MineProximityCostCache()

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

        # Text rendering cache with size limit to prevent unbounded growth
        self.text_cache = {}  # (text, font_size, color) -> surface
        self._text_cache_max_size = 1000  # Prevent unbounded growth across episodes

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
        DEFENSIVE: Prunes cache when it exceeds max size to prevent unbounded growth
        """
        # Convert color to hashable tuple
        if isinstance(color, list):
            color = tuple(color)

        # Create cache key
        font_size = font.get_height()
        cache_key = (text, font_size, color)

        # Check cache
        if cache_key not in self.text_cache:
            # Defensive: prune cache if too large (prevents unbounded growth)
            if len(self.text_cache) >= self._text_cache_max_size:
                # Remove oldest 20% of entries (simple FIFO)
                keys_to_remove = list(self.text_cache.keys())[:200]
                for key in keys_to_remove:
                    del self.text_cache[key]

            # Cache miss - render and store
            self.text_cache[cache_key] = font.render(text, True, color)

        return self.text_cache[cache_key]

    def _draw_arrow(
        self,
        surface: pygame.Surface,
        start: Tuple[float, float],
        direction: Tuple[float, float],
        length: float,
        color: Tuple[int, int, int, int],
        width: int = 3,
        head_size: float = 8,
    ):
        """Draw an arrow from start position in the given direction.

        Args:
            surface: Surface to draw on
            start: Starting position (screen coordinates)
            direction: Normalized direction vector (dx, dy)
            length: Length of the arrow in pixels
            color: RGBA color tuple
            width: Line width
            head_size: Size of the arrowhead
        """
        import math

        dx, dy = direction
        if abs(dx) < 0.001 and abs(dy) < 0.001:
            return  # Zero direction, nothing to draw

        # Calculate end point
        end_x = start[0] + dx * length
        end_y = start[1] + dy * length

        # Draw main line
        pygame.draw.line(
            surface,
            color,
            (int(start[0]), int(start[1])),
            (int(end_x), int(end_y)),
            width,
        )

        # Draw arrowhead
        # Calculate perpendicular vector for arrowhead wings
        angle = math.atan2(dy, dx)
        head_angle = math.pi / 6  # 30 degrees

        # Left wing
        left_x = end_x - head_size * math.cos(angle - head_angle)
        left_y = end_y - head_size * math.sin(angle - head_angle)

        # Right wing
        right_x = end_x - head_size * math.cos(angle + head_angle)
        right_y = end_y - head_size * math.sin(angle + head_angle)

        # Draw arrowhead as filled triangle
        pygame.draw.polygon(
            surface,
            color,
            [
                (int(end_x), int(end_y)),
                (int(left_x), int(left_y)),
                (int(right_x), int(right_y)),
            ],
        )

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

    def _draw_mine_sdf(self, mine_sdf_info: dict) -> Optional[pygame.Surface]:
        """Draw mine SDF (Signed Distance Field) heatmap visualization.

        Shows a heatmap of distance to nearest deadly mine:
        - Dark red = at mine location (most dangerous)
        - Orange/Yellow = danger zone boundary
        - Green = safe zone (far from mines)
        - White arrows = direction TO nearest mine (danger indicator)
        - Dark red circles = exact mine positions

        Args:
            mine_sdf_info: Dictionary containing:
                - sdf_grid: 2D numpy array [rows, cols] with normalized distances (12px resolution)
                - gradient_grid: 3D numpy array [rows, cols, 2] with directions away from mine
                - ninja_position: Current ninja position (x, y)
                - deadly_mine_positions: List of (x, y) tuples for deadly mines

        Returns:
            pygame.Surface with SDF heatmap overlay
        """

        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        sdf_grid = mine_sdf_info.get("sdf_grid")
        gradient_grid = mine_sdf_info.get("gradient_grid")
        ninja_pos = mine_sdf_info.get("ninja_position", (0, 0))
        deadly_mine_positions = mine_sdf_info.get("deadly_mine_positions", [])

        if sdf_grid is None:
            return surface

        rows, cols = sdf_grid.shape
        # SDF uses 12px cells (sub-node resolution) for higher accuracy
        sdf_cell_size = 12 * self.adjust

        # Draw SDF heatmap at 12px resolution
        for row in range(rows):
            for col in range(cols):
                sdf_value = sdf_grid[row, col]

                # Map SDF to color (dark red=at mine, orange=danger, green=safe)
                if sdf_value < -0.7:
                    # Very close to mine: dark red (saturated)
                    r, g, b = 180, 0, 0
                    alpha = 200
                elif sdf_value < 0:
                    # Danger zone: dark red to orange
                    # sdf_value goes from -0.7 to 0 (boundary)
                    t = (-sdf_value) / 0.7  # 1 at -0.7, 0 at boundary
                    r = 180 + int(75 * (1 - t))  # 180 -> 255
                    g = int(165 * (1 - t))  # 0 -> 165 (orange)
                    b = 0
                    alpha = int(120 + 80 * t)  # More opaque closer to mine
                elif sdf_value < 0.3:
                    # Transition zone: orange to yellow-green
                    t = sdf_value / 0.3  # 0 to 1
                    r = int(255 * (1 - t))
                    g = 165 + int(90 * t)  # 165 -> 255
                    b = 0
                    alpha = int(100 * (1 - t) + 40)
                else:
                    # Safe zone: light green, mostly transparent
                    r, g, b, alpha = 50, 200, 50, 15

                screen_x = col * sdf_cell_size + self.tile_x_offset
                screen_y = row * sdf_cell_size + self.tile_y_offset
                rect = pygame.Rect(screen_x, screen_y, sdf_cell_size, sdf_cell_size)
                pygame.draw.rect(surface, (r, g, b, alpha), rect)

        # # Draw exact mine position markers
        # for mine_x, mine_y in deadly_mine_positions:
        #     screen_x = int(mine_x * self.adjust) + self.tile_x_offset
        #     screen_y = int(mine_y * self.adjust) + self.tile_y_offset
        #     # Dark red filled circle with white border
        #     pygame.draw.circle(surface, (140, 0, 0, 255), (screen_x, screen_y), 6)
        #     pygame.draw.circle(surface, (255, 255, 255, 200), (screen_x, screen_y), 6, 2)

        # Draw danger approach arrows at regular 24px intervals (2 cells × 12px)
        # Static grid showing direction TO nearest mine - doesn't change with player position
        # Dense grid with shorter arrows for clearer visualization
        if gradient_grid is not None:
            arrow_spacing = 2  # Every 2 cells = 24px intervals (denser)
            for row in range(arrow_spacing // 2, rows, arrow_spacing):
                for col in range(arrow_spacing // 2, cols, arrow_spacing):
                    gx, gy = gradient_grid[row, col]
                    # Only draw if there's a meaningful gradient (not at mine center or no mines)
                    if abs(gx) > 0.01 or abs(gy) > 0.01:
                        cx = (
                            col * sdf_cell_size + self.tile_x_offset + sdf_cell_size / 2
                        )
                        cy = (
                            row * sdf_cell_size + self.tile_y_offset + sdf_cell_size / 2
                        )

                        # Arrow opacity/color based on distance to mine
                        sdf_val = sdf_grid[row, col]
                        if sdf_val < 0:
                            # In danger zone: bright white arrows
                            arrow_color = (255, 255, 255, 220)
                            arrow_len = sdf_cell_size * 0.8
                        elif sdf_val < 0.5:
                            # Transition zone: semi-transparent
                            arrow_color = (255, 255, 255, 140)
                            arrow_len = sdf_cell_size * 0.7
                        else:
                            # Safe zone: faint arrows
                            arrow_color = (255, 255, 255, 60)
                            arrow_len = sdf_cell_size * 0.6

                        # Negate to point TOWARD mine (gradient points away)
                        self._draw_arrow(
                            surface,
                            (cx, cy),
                            (-gx, -gy),
                            arrow_len,
                            arrow_color,
                            width=1,
                            head_size=3,
                        )

        # Draw ninja indicator
        # nx = int(ninja_pos[0] * self.adjust) + self.tile_x_offset
        # ny = int(ninja_pos[1] * self.adjust) + self.tile_y_offset
        # pygame.draw.circle(surface, (0, 200, 255, 255), (nx, ny), 8)
        # pygame.draw.circle(surface, (255, 255, 255, 255), (nx, ny), 8, 2)

        # # Draw legend
        # try:
        #     font = pygame.font.Font(None, 18)
        # except pygame.error:
        #     font = pygame.font.SysFont("arial", 14)

        # lx, ly, lw, lh = self.screen.get_width() - 170, 20, 150, 110
        # pygame.draw.rect(surface, (0, 0, 0, 200), (lx, ly, lw, lh), border_radius=5)
        # pygame.draw.rect(surface, (100, 200, 255), (lx, ly, lw, lh), 2, border_radius=5)

        # surface.blit(
        #     font.render("Mine SDF (12px)", True, (255, 255, 255)), (lx + 10, ly + 8)
        # )
        # for i, (color, label) in enumerate(
        #     [
        #         ((140, 0, 0), "Mine Position"),
        #         ((180, 0, 0), "At Mine"),
        #         ((255, 100, 0), "Danger Zone"),
        #         ((255, 200, 0), "Transition"),
        #         ((50, 200, 50), "Safe"),
        #     ]
        # ):
        #     y = ly + 28 + i * 16
        #     pygame.draw.rect(surface, color + (200,), (lx + 10, y, 12, 12))
        #     surface.blit(font.render(label, True, (255, 255, 255)), (lx + 28, y - 1))

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
        MASKED_COLOR = (255, 100, 100, 255)  # Red for masked
        INVALID_ACTION_COLOR = (255, 255, 0, 255)  # Yellow for invalid action taken
        TEXT_COLOR = (255, 255, 255, 255)  # White text
        BG_COLOR = (0, 0, 0, 220)  # Semi-transparent black background
        BORDER_COLOR = (150, 150, 255, 255)  # Light blue border

        # Determine ninja quadrant for positioning
        ninja_screen_x = int(ninja_pos[0] * self.adjust) + self.tile_x_offset
        ninja_screen_y = int(ninja_pos[1] * self.adjust) + self.tile_y_offset

        screen_center_x = self.screen.get_width() / 2
        screen_center_y = self.screen.get_height() / 2

        ninja_in_left = ninja_screen_x < screen_center_x
        ninja_in_top = ninja_screen_y < screen_center_y

        # Compact keyboard layout dimensions
        key_size = 20  # Size of each key box
        key_spacing = 3  # Space between keys
        panel_padding = 8

        # Calculate panel dimensions
        # 3 keys wide, 2 keys tall
        panel_width = 3 * key_size + 2 * key_spacing + 2 * panel_padding
        panel_height = (
            2 * key_size + key_spacing + 2 * panel_padding + 20
        )  # +20 for title

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
                is_last_action = last_action is not None and last_action == action_idx

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
        base_adjacency = graph_data.get("base_adjacency", adjacency)
        ninja_pos = path_aware_info.get("ninja_position", (0, 0))
        entities = path_aware_info.get("entities", [])
        level_data = path_aware_info.get("level_data", None)

        # Import shared utilities from pathfinding modules
        from nclone.graph.reachability.pathfinding_utils import (
            bfs_distance_from_start,
            find_shortest_path,
            extract_spatial_lookups_from_graph_data,
            find_goal_node_closest_to_start,
            find_ninja_node,
            classify_edge_type,
            get_edge_type_color,
            get_edge_type_label,
            get_edge_type_cost_multiplier,
        )

        # Extract spatial hash and subcell lookup from graph_data if available
        spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
            graph_data
        )

        # PERFORMANCE OPTIMIZATION: Extract pre-computed physics cache from graph_data
        physics_cache = graph_data.get("node_physics")

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
        TEXT_COLOR = (255, 255, 255, 255)  # White text
        # Note: Path edge colors are now determined dynamically by edge type classification

        # Compute reachable set dynamically from current ninja position
        def compute_reachability_from_position(start_pos, adjacency):
            """Compute reachable nodes from a starting position via flood-fill."""
            from collections import deque

            # Use the same logic as blue node highlighting to find starting node
            start_node = find_ninja_node(
                start_pos,
                adjacency,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
                ninja_radius=10.0,
            )

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

        used_edge_types = set()  # Track edge types used in paths for legend

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

            # Find ninja node once using canonical function
            ninja_node = (
                find_ninja_node(
                    ninja_pos,
                    adjacency,
                    spatial_hash=spatial_hash,
                    subcell_lookup=subcell_lookup,
                    ninja_radius=10.0,
                )
                if ninja_pos
                else None
            )

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
                elif ninja_node is not None and pos == ninja_node:
                    node_color = NINJA_NODE_COLOR
                else:
                    node_color = NODE_COLOR

                # Draw node circle
                pygame.draw.circle(surface, node_color, (screen_x, screen_y), 3)

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

            # Build mine proximity cache for hazard avoidance in pathfinding
            if level_data and hasattr(self, "_mine_proximity_cache"):
                self._mine_proximity_cache.build_cache(level_data, adjacency)

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
                closest_node = find_ninja_node(
                    ninja_pos,
                    adjacency,
                    spatial_hash=spatial_hash,
                    subcell_lookup=subcell_lookup,
                    ninja_radius=10.0,
                )
                switch_path = None
                exit_path = None
                switch_node = None
                exit_node = None

                # Only proceed if we found a valid ninja node and it's reachable
                if closest_node and closest_node in reachable:
                    # Draw path to nearest switch (if not yet activated)
                    if switch_positions:  # Only if there are uncollected switches
                        for switch_pos in switch_positions:
                            # Find goal node using same logic as pathfinding
                            # Filter adjacency to only reachable nodes first
                            reachable_adjacency = {
                                k: v for k, v in adjacency.items() if k in reachable
                            }
                            switch_node = find_goal_node_closest_to_start(
                                switch_pos,
                                closest_node,  # ninja start node
                                reachable_adjacency,
                                entity_radius=6.0,  # Exit switch radius
                                ninja_radius=10.0,
                                spatial_hash=spatial_hash,
                                subcell_lookup=subcell_lookup,
                            )

                            if switch_node:
                                path, _ = find_shortest_path(
                                    closest_node,
                                    switch_node,
                                    adjacency,
                                    base_adjacency,
                                    physics_cache,
                                    level_data,
                                    self._mine_proximity_cache,
                                )
                                if path:
                                    switch_path = path
                                    break  # Only draw path to nearest switch

                    # Draw path to nearest exit ONLY if switch has been activated
                    if exit_switch_activated and exit_positions:
                        for exit_pos in exit_positions:
                            # Find goal node using same logic as pathfinding
                            # Filter adjacency to only reachable nodes first
                            reachable_adjacency = {
                                k: v for k, v in adjacency.items() if k in reachable
                            }
                            exit_node = find_goal_node_closest_to_start(
                                exit_pos,
                                closest_node,  # ninja start node
                                reachable_adjacency,
                                entity_radius=12.0,  # Exit door radius
                                ninja_radius=10.0,
                                spatial_hash=spatial_hash,
                                subcell_lookup=subcell_lookup,
                            )

                            if exit_node:
                                path, _ = find_shortest_path(
                                    closest_node,
                                    exit_node,
                                    adjacency,
                                    base_adjacency,
                                    physics_cache,
                                    level_data,
                                    self._mine_proximity_cache,
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
                    # Draw path with color-coded edges
                    for i in range(len(switch_path) - 1):
                        node1 = switch_path[i]
                        node2 = switch_path[i + 1]

                        # Verify nodes are adjacent in the graph
                        neighbors = adjacency.get(node1, [])
                        is_adjacent = any(n[0] == node2 for n in neighbors)

                        if is_adjacent:
                            # Classify edge type and get color (uses base_adjacency for physics)
                            edge_type = classify_edge_type(node1, node2, base_adjacency)
                            edge_color = get_edge_type_color(edge_type)
                            used_edge_types.add(edge_type)

                            x1, y1 = node1
                            x2, y2 = node2
                            # Add +24 offset to match node visualization (accounts for tile padding)
                            screen_x1 = int(x1 * self.adjust + self.tile_x_offset) + 24
                            screen_y1 = int(y1 * self.adjust + self.tile_y_offset) + 24
                            screen_x2 = int(x2 * self.adjust + self.tile_x_offset) + 24
                            screen_y2 = int(y2 * self.adjust + self.tile_y_offset) + 24
                            pygame.draw.line(
                                surface,
                                edge_color,
                                (screen_x1, screen_y1),
                                (screen_x2, screen_y2),
                                4,  # Thicker for better visibility
                            )

                # Draw exit path
                if exit_path:
                    # Draw path with color-coded edges
                    for i in range(len(exit_path) - 1):
                        node1 = exit_path[i]
                        node2 = exit_path[i + 1]

                        # Verify nodes are adjacent in the graph
                        neighbors = adjacency.get(node1, [])
                        is_adjacent = any(n[0] == node2 for n in neighbors)

                        if is_adjacent:
                            # Classify edge type and get color (uses base_adjacency for physics)
                            edge_type = classify_edge_type(node1, node2, base_adjacency)
                            edge_color = get_edge_type_color(edge_type)
                            used_edge_types.add(edge_type)

                            x1, y1 = node1
                            x2, y2 = node2
                            # Add +24 offset to match node visualization (accounts for tile padding)
                            screen_x1 = int(x1 * self.adjust + self.tile_x_offset) + 24
                            screen_y1 = int(y1 * self.adjust + self.tile_y_offset) + 24
                            screen_x2 = int(x2 * self.adjust + self.tile_x_offset) + 24
                            screen_y2 = int(y2 * self.adjust + self.tile_y_offset) + 24
                            pygame.draw.line(
                                surface,
                                edge_color,
                                (screen_x1, screen_y1),
                                (screen_x2, screen_y2),
                                4,  # Thicker for better visibility
                            )

            # === MOVEMENT VECTOR VISUALIZATION ===
            # Draw optimal direction (next_hop) and actual velocity vectors
            if ninja_pos and closest_node:
                ninja_x, ninja_y = ninja_pos
                ninja_screen_x = int(ninja_x * self.adjust + self.tile_x_offset)
                ninja_screen_y = int(ninja_y * self.adjust + self.tile_y_offset)

                # Determine current goal and get next_hop from cached path
                # The next hop is the second node in the path (first is current position)
                next_hop = None
                if cached_data:
                    if exit_switch_activated:
                        # Use exit path
                        path = cached_data.get("exit_path")
                    else:
                        # Use switch path
                        path = cached_data.get("switch_path")

                    if path and len(path) >= 2:
                        # Path starts at closest_node, so path[1] is the next hop
                        next_hop = path[1]

                # Draw optimal direction arrow (green) from next_hop
                if next_hop is not None:
                    # Compute direction from current node to next_hop
                    dx = next_hop[0] - closest_node[0]
                    dy = next_hop[1] - closest_node[1]
                    length = (dx * dx + dy * dy) ** 0.5
                    if length > 0.001:
                        # Normalize direction
                        opt_dir = (dx / length, dy / length)
                        # Draw arrow (length 40 pixels, bright green)
                        self._draw_arrow(
                            surface,
                            (ninja_screen_x, ninja_screen_y),
                            opt_dir,
                            40,  # Arrow length
                            (50, 255, 50, 255),  # Bright green
                            width=3,
                            head_size=10,
                        )

                # Draw actual velocity arrow (cyan) from ninja velocity
                ninja_velocity = path_aware_info.get("ninja_velocity")
                if ninja_velocity is not None:
                    vx, vy = ninja_velocity
                    speed = (vx * vx + vy * vy) ** 0.5
                    if speed > 0.5:  # Only draw if moving with some velocity
                        # Normalize velocity direction
                        vel_dir = (vx / speed, vy / speed)
                        # Scale arrow length by speed (min 20, max 60 pixels)
                        arrow_len = min(60, max(20, speed * 10))
                        # Draw arrow (cyan/light blue)
                        self._draw_arrow(
                            surface,
                            (ninja_screen_x, ninja_screen_y),
                            vel_dir,
                            arrow_len,
                            (100, 220, 255, 255),  # Cyan
                            width=2,
                            head_size=8,
                        )

                # Draw legend for movement vectors
                try:
                    vector_font = pygame.font.Font(None, 16)
                except pygame.error:
                    vector_font = pygame.font.SysFont("arial", 12)

                legend_x = ninja_screen_x + 50
                legend_y = ninja_screen_y - 30

                # Keep legend on screen
                if legend_x + 100 > self.screen.get_width():
                    legend_x = ninja_screen_x - 150
                if legend_y < 20:
                    legend_y = ninja_screen_y + 50

                # Draw mini legend
                pygame.draw.rect(
                    surface,
                    (0, 0, 0, 180),
                    (legend_x, legend_y, 95, 36),
                    border_radius=3,
                )

                # Optimal direction indicator
                pygame.draw.line(
                    surface,
                    (50, 255, 50, 255),
                    (legend_x + 5, legend_y + 10),
                    (legend_x + 20, legend_y + 10),
                    2,
                )
                opt_text = vector_font.render("Optimal", True, (200, 255, 200))
                surface.blit(opt_text, (legend_x + 25, legend_y + 4))

                # Velocity indicator
                pygame.draw.line(
                    surface,
                    (100, 220, 255, 255),
                    (legend_x + 5, legend_y + 24),
                    (legend_x + 20, legend_y + 24),
                    2,
                )
                vel_text = vector_font.render("Velocity", True, (180, 230, 255))
                surface.blit(vel_text, (legend_x + 25, legend_y + 18))

        # Calculate and display switch/exit distances
        if show_distances and ninja_pos and adjacency:
            # Extract ninja position for screen coordinates
            ninja_x, ninja_y = ninja_pos

            # Use the same logic as blue node highlighting to find starting node
            closest_node = find_ninja_node(
                ninja_pos,
                adjacency,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
                ninja_radius=10.0,
            )

            if closest_node:
                # Calculate distance to nearest switch using shared utilities
                switch_dist = float("inf")
                for switch_pos in switch_positions:
                    # Find goal node using same logic as pathfinding
                    switch_node = find_goal_node_closest_to_start(
                        switch_pos,
                        closest_node,
                        adjacency,
                        entity_radius=6.0,  # Exit switch radius
                        ninja_radius=10.0,
                        spatial_hash=spatial_hash,
                        subcell_lookup=subcell_lookup,
                    )

                    if switch_node is not None:
                        # BFS from ninja to switch using shared utility
                        # Use mine proximity cache for consistent path distances with training
                        distances, target_dist, _, _ = bfs_distance_from_start(
                            closest_node,
                            switch_node,
                            adjacency,
                            base_adjacency,
                            None,  # max_distance
                            physics_cache,
                            level_data,
                            self._mine_proximity_cache,
                        )

                        if target_dist is not None:
                            switch_dist = min(switch_dist, target_dist)

                # Calculate distance to nearest exit using shared utilities
                exit_dist = float("inf")
                for exit_pos in exit_positions:
                    # Find goal node using same logic as pathfinding
                    exit_node = find_goal_node_closest_to_start(
                        exit_pos,
                        closest_node,
                        adjacency,
                        entity_radius=12.0,  # Exit door radius
                        ninja_radius=10.0,
                        spatial_hash=spatial_hash,
                        subcell_lookup=subcell_lookup,
                    )

                    if exit_node is not None:
                        # BFS from ninja to exit using shared utility
                        # Use mine proximity cache for consistent path distances with training
                        distances, target_dist, _, _ = bfs_distance_from_start(
                            closest_node,
                            exit_node,
                            adjacency,
                            base_adjacency,
                            None,  # max_distance
                            physics_cache,
                            level_data,
                            self._mine_proximity_cache,
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

        # Draw path edge types legend (only when paths are shown)
        if show_paths and used_edge_types:
            try:
                legend_font = pygame.font.Font(None, 18)
            except pygame.error:
                legend_font = pygame.font.SysFont("arial", 14)

            # Sort edge types by cost (cheapest first) for consistent display
            sorted_edge_types = sorted(
                used_edge_types, key=lambda et: get_edge_type_cost_multiplier(et)
            )

            # Calculate legend dimensions based on number of edge types
            edge_legend_width = 220
            edge_legend_height = 40 + len(sorted_edge_types) * 22
            edge_legend_x = 20
            edge_legend_y = self.screen.get_height() - edge_legend_height - 20

            # Background
            pygame.draw.rect(
                surface,
                (0, 0, 0, 200),
                (edge_legend_x, edge_legend_y, edge_legend_width, edge_legend_height),
                border_radius=5,
            )
            pygame.draw.rect(
                surface,
                (100, 200, 255, 255),
                (edge_legend_x, edge_legend_y, edge_legend_width, edge_legend_height),
                2,
                border_radius=5,
            )

            # Title
            title_surf = legend_font.render("Path Edge Types:", True, TEXT_COLOR)
            surface.blit(title_surf, (edge_legend_x + 10, edge_legend_y + 10))

            # Edge type items
            y_offset = edge_legend_y + 35
            for edge_type in sorted_edge_types:
                # Get edge type info
                edge_color = get_edge_type_color(edge_type)
                edge_label = get_edge_type_label(edge_type)
                edge_cost = get_edge_type_cost_multiplier(edge_type)

                # Draw color indicator line
                line_x1 = edge_legend_x + 15
                line_y = y_offset + 8
                line_x2 = edge_legend_x + 35
                pygame.draw.line(
                    surface, edge_color, (line_x1, line_y), (line_x2, line_y), 4
                )

                # Draw label and cost
                text = f"{edge_label} ({edge_cost:.2f})"
                text_surf = legend_font.render(text, True, TEXT_COLOR)
                surface.blit(text_surf, (edge_legend_x + 45, y_offset))
                y_offset += 22

        return surface

    def _draw_demo_checkpoints_heatmap(
        self, demo_checkpoint_info: dict
    ) -> Optional[pygame.Surface]:
        """Draw demo checkpoint heatmap visualization showing expert trajectories.

        Renders checkpoints as a heatmap with colors indicating cumulative reward:
        Blue (low reward) → Cyan → Green → Yellow → Red (high reward)

        Args:
            demo_checkpoint_info: Dictionary containing:
                - checkpoints: List of checkpoint dicts with:
                    - cell: (x, y) discretized position
                    - cumulative_reward: Reward value (frame progression proxy)
                    - position: (x, y) actual position in pixels
                - ninja_position: Current ninja position (x, y)
                - grid_size: Cell size in pixels (12px)

        Returns:
            pygame.Surface with demo checkpoint heatmap overlay
        """
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        checkpoints = demo_checkpoint_info.get("checkpoints", [])
        ninja_pos = demo_checkpoint_info.get("ninja_position", (0, 0))
        grid_size = demo_checkpoint_info.get("grid_size", 12)

        if not checkpoints:
            return surface

        # Debug: Log checkpoint data structure
        import logging
        _logger = logging.getLogger(__name__)
        _logger.info(
            f"[DEMO_HEATMAP] Rendering {len(checkpoints)} checkpoints, "
            f"grid_size={grid_size}, ninja_pos={ninja_pos}, "
            f"tile_offset=({self.tile_x_offset}, {self.tile_y_offset}), "
            f"adjust={self.adjust}"
        )
        if checkpoints:
            sample = checkpoints[0]
            _logger.info(
                f"[DEMO_HEATMAP] Sample checkpoint keys: {list(sample.keys())}"
            )

        # Calculate screen cell size
        cell_size = grid_size * self.adjust

        # Normalize cumulative rewards to [0, 1] range for color mapping
        rewards = [cp.get("cumulative_reward", 0.0) for cp in checkpoints]
        min_reward = min(rewards) if rewards else 0.0
        max_reward = max(rewards) if rewards else 1.0
        reward_range = max_reward - min_reward if max_reward > min_reward else 1.0

        # Track highest reward checkpoint for marker
        highest_checkpoint = None
        highest_reward = float("-inf")

        # Draw checkpoint cells as heatmap
        for i, checkpoint in enumerate(checkpoints):
            cell = checkpoint.get("cell")
            cumulative_reward = checkpoint.get("cumulative_reward", 0.0)
            position = checkpoint.get("position")

            # Debug first few checkpoints
            if i < 3:
                import logging
                _logger = logging.getLogger(__name__)
                _logger.info(
                    f"[DEMO_CHECKPOINT_DEBUG] checkpoint {i}: position={position} (type={type(position)}), "
                    f"cell={cell}, reward={cumulative_reward:.3f}"
                )

            # Skip if no position data
            if position is None:
                continue

            # Handle position being a tuple or list
            try:
                if isinstance(position, (tuple, list)) and len(position) >= 2:
                    pos_x, pos_y = position[0], position[1]
                else:
                    # Skip malformed position
                    if i < 3:
                        _logger.warning(f"Malformed position: {position}")
                    continue
            except Exception as e:
                if i < 3:
                    _logger.warning(f"Error unpacking position {position}: {e}")
                continue

            # Debug calculated coordinates
            if i < 3:
                grid_cell_x = int(pos_x) // grid_size
                grid_cell_y = int(pos_y) // grid_size
                screen_x = grid_cell_x * cell_size + self.tile_x_offset
                screen_y = grid_cell_y * cell_size + self.tile_y_offset
                _logger.info(
                    f"  -> grid_cell=({grid_cell_x}, {grid_cell_y}), "
                    f"cell_size={cell_size:.2f}, screen=({screen_x:.1f}, {screen_y:.1f})"
                )

            # Normalize reward to [0, 1]
            normalized_reward = (cumulative_reward - min_reward) / reward_range

            # Track highest reward checkpoint
            if cumulative_reward > highest_reward:
                highest_reward = cumulative_reward
                highest_checkpoint = position

            # Map normalized reward to color (blue → cyan → green → yellow → red)
            # This creates a "hot" colormap similar to thermal imaging
            if normalized_reward < 0.25:
                # Blue to Cyan (0.0 - 0.25)
                t = normalized_reward / 0.25
                r = 0
                g = int(100 + 155 * t)  # 100 → 255
                b = 255
                alpha = 120
            elif normalized_reward < 0.5:
                # Cyan to Green (0.25 - 0.5)
                t = (normalized_reward - 0.25) / 0.25
                r = 0
                g = 255
                b = int(255 * (1 - t))  # 255 → 0
                alpha = 130
            elif normalized_reward < 0.75:
                # Green to Yellow (0.5 - 0.75)
                t = (normalized_reward - 0.5) / 0.25
                r = int(255 * t)  # 0 → 255
                g = 255
                b = 0
                alpha = 140
            else:
                # Yellow to Red (0.75 - 1.0)
                t = (normalized_reward - 0.75) / 0.25
                r = 255
                g = int(255 * (1 - t))  # 255 → 0
                b = 0
                alpha = 150

            # Convert world coordinates (pixels) to screen coordinates
            # Standard pattern: screen = int(world * adjust) + tile_offset
            # For grid-based rendering, snap to grid first
            
            # Snap position to grid boundaries (12px cells in world space)
            grid_world_x = (int(pos_x) // grid_size) * grid_size
            grid_world_y = (int(pos_y) // grid_size) * grid_size
            
            # Convert snapped world coordinates to screen coordinates
            screen_x = int(grid_world_x * self.adjust) + self.tile_x_offset
            screen_y = int(grid_world_y * self.adjust) + self.tile_y_offset

            # Debug calculated coordinates for first few
            if i < 3:
                _logger.info(
                    f"  -> world=({pos_x}, {pos_y}), grid_world=({grid_world_x}, {grid_world_y}), "
                    f"screen=({screen_x}, {screen_y}), cell_size={cell_size:.2f}"
                )

            # Draw filled rectangle for this checkpoint cell
            rect = pygame.Rect(screen_x, screen_y, cell_size, cell_size)
            pygame.draw.rect(surface, (r, g, b, alpha), rect)

        # Draw marker for highest reward checkpoint (goal position indicator)
        if highest_checkpoint is not None:
            hc_x, hc_y = highest_checkpoint
            screen_x = int(hc_x * self.adjust) + self.tile_x_offset
            screen_y = int(hc_y * self.adjust) + self.tile_y_offset

            # Draw bright red circle with yellow border
            pygame.draw.circle(surface, (255, 50, 50, 255), (screen_x, screen_y), 8)
            pygame.draw.circle(surface, (255, 255, 100, 255), (screen_x, screen_y), 8, 2)

        # Draw legend
        try:
            legend_font = pygame.font.Font(None, 18)
            title_font = pygame.font.Font(None, 20)
        except pygame.error:
            legend_font = pygame.font.SysFont("arial", 14)
            title_font = pygame.font.SysFont("arial", 16)

        # Legend dimensions
        legend_width = 180
        legend_height = 140
        legend_x = self.screen.get_width() - legend_width - 20
        legend_y = 20

        # Draw legend background
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
        title_text = "Demo Checkpoints"
        title_surf = title_font.render(title_text, True, (255, 255, 255))
        surface.blit(title_surf, (legend_x + 10, legend_y + 8))

        # Checkpoint count
        count_text = f"Count: {len(checkpoints)}"
        count_surf = legend_font.render(count_text, True, (255, 255, 255))
        surface.blit(count_surf, (legend_x + 10, legend_y + 32))

        # Reward range
        range_text = f"Reward: {min_reward:.2f}-{max_reward:.2f}"
        range_surf = legend_font.render(range_text, True, (255, 255, 255))
        surface.blit(range_surf, (legend_x + 10, legend_y + 52))

        # Color scale
        scale_y = legend_y + 75
        scale_height = 12
        scale_colors = [
            (0, 100, 255),  # Blue (low)
            (0, 255, 255),  # Cyan
            (0, 255, 0),  # Green
            (255, 255, 0),  # Yellow
            (255, 0, 0),  # Red (high)
        ]

        # Draw color gradient bar
        bar_width = legend_width - 20
        segment_width = bar_width / len(scale_colors)
        for i, color in enumerate(scale_colors):
            segment_x = legend_x + 10 + i * segment_width
            pygame.draw.rect(
                surface,
                color + (200,),
                (segment_x, scale_y, segment_width, scale_height),
            )

        # Scale labels
        low_label = legend_font.render("Low", True, (200, 200, 255))
        high_label = legend_font.render("High", True, (255, 200, 200))
        surface.blit(low_label, (legend_x + 10, scale_y + scale_height + 5))
        high_x = legend_x + legend_width - high_label.get_width() - 10
        surface.blit(high_label, (high_x, scale_y + scale_height + 5))

        # Highest checkpoint indicator
        marker_text = legend_font.render("● = Goal", True, (255, 50, 50))
        surface.blit(marker_text, (legend_x + 10, legend_y + 110))

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

        # Draw action mask visualization if provided
        if debug_info and "action_mask" in debug_info:
            action_mask_surface = self._draw_action_mask(debug_info["action_mask"])
            if action_mask_surface:
                surface.blit(action_mask_surface, (0, 0))

        # Draw mine SDF visualization if provided
        if debug_info and "mine_sdf" in debug_info:
            mine_sdf_surface = self._draw_mine_sdf(debug_info["mine_sdf"])
            if mine_sdf_surface:
                surface.blit(mine_sdf_surface, (0, 0))

        # Draw demo checkpoint heatmap if provided
        if debug_info and "demo_checkpoints" in debug_info:
            demo_checkpoint_surface = self._draw_demo_checkpoints_heatmap(
                debug_info["demo_checkpoints"]
            )
            if demo_checkpoint_surface:
                surface.blit(demo_checkpoint_surface, (0, 0))

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
