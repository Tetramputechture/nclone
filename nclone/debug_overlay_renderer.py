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
        from nclone.graph.reachability.path_distance_calculator import (
            CachedPathDistanceCalculator,
        )

        self._path_visualization_cache = PathVisualizationCache()
        self._mine_proximity_cache = MineProximityCostCache()
        self._path_calculator = CachedPathDistanceCalculator(max_cache_size=200)

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

        if sdf_grid is None:
            return surface

        rows, cols = sdf_grid.shape
        # Match MineSignedDistanceField grid (see mine_proximity_cache.SDF_CELL_SIZE)
        cell_px = float(mine_sdf_info.get("sdf_cell_size", 6))
        sdf_cell_size = cell_px * self.adjust

        # Draw SDF heatmap at SDF_CELL_SIZE resolution
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

        # Draw danger approach arrows at regular intervals in SDF cells
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

    def _jump_label_font(self):
        try:
            return pygame.font.Font(None, int(18 * self.adjust))
        except pygame.error:
            return pygame.font.SysFont("arial", int(14 * self.adjust))

    def _draw_jump_cost_to_go_heatmap(
        self, info: dict
    ) -> Optional[pygame.Surface]:
        """Return a blit-ready jump cost-to-go overlay surface (non-blocking).

        The heatmap is ~17k cells, so painting it every frame would lag the GUI
        continuously. Instead the painted surface is rendered ONCE and cached,
        keyed by (map_hash, phase, render params); every subsequent frame just
        returns the cached Surface (the caller blits it). The cache is rebuilt
        only when the level (map_hash), objective phase, or view params change.

        While the field is still building off-thread (info["computing"]), this
        returns a tiny per-frame surface with only a "computing… (Xs)" label.

        Args:
            info: from ``_get_jump_cost_to_go_data`` -- either a "computing"
                payload, or one carrying ``field`` (shape ``(W, H)`` x-major,
                cost-to-go in frames, ``inf`` = unreachable), ``phase``,
                ``map_hash`` and ``cell_px``.

        Returns:
            pygame.Surface (screen-sized, SRCALPHA) ready to blit.
        """
        # Building: cheap per-frame indicator only, no heatmap, never blocks.
        if info.get("computing"):
            surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            elapsed = info.get("elapsed", 0.0)
            label = f"Jump cost-to-go: computing… ({elapsed:.0f}s)"
            surface.blit(
                self._jump_label_font().render(label, True, (255, 255, 255)),
                (self.tile_x_offset + 8, self.tile_y_offset + 8),
            )
            return surface

        field = info.get("field")
        if field is None:
            return pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        phase = info.get("phase", "switch")
        # Cache key: invalidate on level change, phase switch<->exit, or any view
        # param change (size/zoom/pan) that would move/resize the painted cells.
        cache_key = (
            info.get("map_hash"),
            phase,
            float(info.get("cell_px", 6)),
            round(float(self.adjust), 4),
            int(self.tile_x_offset),
            int(self.tile_y_offset),
            self.screen.get_size(),
        )
        cache = getattr(self, "_jump_heatmap_cache", None)
        if cache is None:
            cache = self._jump_heatmap_cache = {}
        cached = cache.get(cache_key)
        if cached is not None:
            return cached  # instant: just blit the pre-painted surface

        surface = self._render_jump_cost_to_go_surface(field, phase)
        # Bound the cache (level x phase x view params); keep it small.
        if len(cache) > 8:
            cache.clear()
        cache[cache_key] = surface
        return surface

    def _render_jump_cost_to_go_surface(self, field, phase: str) -> pygame.Surface:
        """Paint the ~17k-cell heatmap once (called only on a cache miss)."""
        import numpy as np

        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        # The JumpDistanceSource field is (W, H) x-major (field[x, y]) whereas the
        # mine sdf_grid is [rows, cols] = [H, W]. We iterate rows=H, cols=W and index
        # field[col, row] (col=x, row=y) so cells map straight to screen space.
        W, H = field.shape
        cell = 6.0 * self.adjust

        finite = np.isfinite(field)
        if not finite.any():
            return surface
        fmax = float(field[finite].max())
        if fmax <= 0.0:
            fmax = 1.0

        for row in range(H):  # row = y
            for col in range(W):  # col = x
                c = field[col, row]  # (col=x, row=y) -> screen-aligned
                if not np.isfinite(c):
                    continue  # unreachable: transparent
                t = c / fmax  # 0 near goal, 1 far
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                # Green (50,200,50) -> yellow/orange (255,200,40) -> red (200,40,40).
                if t <= 0.5:
                    u = t / 0.5
                    r = int(50 + (255 - 50) * u)
                    g = 200
                    b = int(50 + (40 - 50) * u)
                else:
                    u = (t - 0.5) / 0.5
                    r = int(255 + (200 - 255) * u)
                    g = int(200 + (40 - 200) * u)
                    b = 40
                alpha = 110

                screen_x = col * cell + self.tile_x_offset
                screen_y = row * cell + self.tile_y_offset
                rect = pygame.Rect(screen_x, screen_y, cell, cell)
                pygame.draw.rect(surface, (r, g, b, alpha), rect)

        # Text label naming the current objective phase (matches other overlays).
        label = f"Jump cost-to-go: {phase}"
        surface.blit(
            self._jump_label_font().render(label, True, (255, 255, 255)),
            (self.tile_x_offset + 8, self.tile_y_offset + 8),
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
            extract_spatial_lookups_from_graph_data,
            find_goal_node_closest_to_start,
            find_ninja_node,
        )

        # Extract spatial hash and subcell lookup from graph_data if available
        spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
            graph_data
        )

        show_adjacency = path_aware_info.get("show_adjacency", False)
        show_distances = path_aware_info.get("show_distances", False)
        show_blocked = path_aware_info.get("show_blocked", False)
        show_paths = path_aware_info.get("show_paths", False)
        mine_proximity_by_node = path_aware_info.get("mine_proximity_by_node")
        hazard_combo = path_aware_info.get("hazard_combo", False)

        # Colors for visualization
        NODE_COLOR = (100, 255, 100, 180)  # Green nodes
        EDGE_COLOR = (255, 255, 100, 100)  # Yellow edges
        BLOCKED_NODE_COLOR = (255, 50, 50, 180)  # Red for blocked
        NINJA_NODE_COLOR = (100, 150, 255, 255)  # Blue for ninja position
        SWITCH_NODE_COLOR = (100, 255, 100, 255)  # Bright green for switches
        EXIT_NODE_COLOR = (255, 200, 100, 255)  # Orange for exits
        TEXT_COLOR = (255, 255, 255, 255)  # White text
        PATH_EDGE_COLOR = (0, 255, 255, 255)  # Bright cyan for geometric path

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
                elif mine_proximity_by_node:
                    mult = mine_proximity_by_node.get(pos, 1.0)
                    if mult > 1.02:
                        from nclone.gym_environment.reward_calculation.reward_constants import (
                            MINE_HAZARD_COST_MULTIPLIER,
                        )

                        denom = max(
                            MINE_HAZARD_COST_MULTIPLIER - 1.0, 1e-6
                        )
                        t = min(1.0, (mult - 1.0) / denom)
                        r = int(90 + 165 * t)
                        g = int(255 - 200 * t)
                        b = int(60 + 40 * (1.0 - t))
                        node_color = (r, g, b, 240)
                    else:
                        node_color = NODE_COLOR
                else:
                    node_color = NODE_COLOR

                # Draw node circle (slightly larger when showing mine path costs)
                radius = 4 if mine_proximity_by_node else 3
                pygame.draw.circle(surface, node_color, (screen_x, screen_y), radius)

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
            # CRITICAL: This MUST be done before checking path cache to ensure
            # cached paths were computed with correct mine avoidance costs

            # STRICT VALIDATION: Require mine proximity cache for physics-aware pathfinding
            if not hasattr(self, "_mine_proximity_cache"):
                raise RuntimeError(
                    "DebugOverlayRenderer is missing _mine_proximity_cache. "
                    "This is required for mine avoidance in physics-based pathfinding. "
                    "Ensure initialization creates MineProximityCostCache."
                )

            if level_data is None:
                raise RuntimeError(
                    "level_data is None when building mine cache for path visualization. "
                    "Level data is required for mine proximity calculations."
                )

            mine_cache_rebuilt = self._mine_proximity_cache.build_cache(
                level_data, adjacency, graph_data
            )

            # DIAGNOSTIC: Log mine cache state for path cost validation
            import logging

            logger = logging.getLogger(__name__)
            if mine_cache_rebuilt:
                # logger.warning(
                #     f"[PATH_VIZ] Mine proximity cache rebuilt: {len(self._mine_proximity_cache.cache)} nodes with mine costs"
                # )
                pass
            elif len(self._mine_proximity_cache.cache) > 0:
                # logger.warning(
                #     f"[PATH_VIZ] Mine proximity cache valid: {len(self._mine_proximity_cache.cache)} nodes cached"
                # )
                pass
            # else:
            #     logger.warning(
            #         "[PATH_VIZ] Mine proximity cache empty (no deadly mines or avoidance disabled)"
            #     )

            # Sync mine cache with path calculator for unified pathfinding
            self._path_calculator.mine_proximity_cache = self._mine_proximity_cache
            # Build level cache if needed (for path cost computation)
            if level_data is not None:
                self._path_calculator.build_level_cache(
                    level_data, adjacency, base_adjacency, graph_data
                )

            # Try to get cached paths
            # IMPORTANT: If mine cache was rebuilt, invalidate path cache to ensure
            # paths are recomputed with new mine avoidance costs
            cached_data = None
            mine_cache_size = len(self._mine_proximity_cache.cache)

            if level_data and hasattr(self, "_path_visualization_cache"):
                if mine_cache_rebuilt:
                    # Mine costs changed - invalidate path cache to force recomputation
                    self._path_visualization_cache.clear_cache()
                    # logger.warning(
                    #     "[PATH_VIZ] Path cache invalidated due to mine cache rebuild"
                    # )
                else:
                    cached_data = self._path_visualization_cache.get_cached_paths(
                        ninja_pos,
                        adjacency,
                        level_data,
                        switch_positions,
                        exit_positions,
                        exit_switch_activated,
                        mine_cache_size=mine_cache_size,
                    )

            if cached_data is None:
                # NOTE: Reward calculator paths are computed from spawn and only update on reset
                # For visualization, we need paths from CURRENT ninja position that update every 6px
                # So we always recompute paths from current position (not using cached_reward_paths)
                # print("[PATH_VIZ] Computing paths from current ninja position...")

                # STRICT VALIDATION: Require base_adjacency for pathfinding
                if base_adjacency is None:
                    raise RuntimeError(
                        "base_adjacency is None when computing paths for visualization. "
                        "Base adjacency is REQUIRED for physics edge classification. "
                        "Ensure graph building includes base_adjacency in graph_data."
                    )

                # logger.warning(
                #     f"[PATH_VIZ] Computing fresh paths (cache miss). "
                #     f"mine_cache_size={mine_cache_size}, "
                #     f"physics_cache_size={len(physics_cache) if physics_cache else 0}"
                # )

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
                    if switch_positions:
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
                                # Compute path using unified get_path_cost (same as training PBRS)
                                path_cost, path = self._path_calculator.get_path_cost(
                                    ninja_pos,
                                    switch_pos,
                                    adjacency,
                                    base_adjacency,
                                    level_data=level_data,
                                    graph_data=graph_data,
                                    entity_radius=6.0,
                                    ninja_radius=10.0,
                                    return_path=True,
                                )
                                if path:
                                    switch_path = path
                                    break  # Only draw path to nearest switch
                                else:
                                    logger.warning(
                                        f"[PATH_COMPUTE] No path to switch from {ninja_pos}"
                                    )

                    # Draw path to nearest exit ONLY if switch has been activated
                    if exit_switch_activated and exit_positions:
                        reachable_adjacency = {
                            k: v for k, v in adjacency.items() if k in reachable
                        }
                        for exit_pos in exit_positions:
                            # Find goal node using same logic as pathfinding
                            # Filter adjacency to only reachable nodes first

                            exit_node = find_goal_node_closest_to_start(
                                exit_pos,
                                closest_node,  # ninja start node
                                reachable_adjacency,
                                entity_radius=12.0,  # Exit door radius
                                ninja_radius=10.0,
                                subcell_lookup=subcell_lookup,
                            )

                            if exit_node:
                                # Compute path using unified get_path_cost (same as training PBRS)
                                path_cost, path = self._path_calculator.get_path_cost(
                                    ninja_pos,
                                    exit_pos,
                                    adjacency,
                                    base_adjacency,
                                    level_data=level_data,
                                    graph_data=graph_data,
                                    entity_radius=12.0,
                                    ninja_radius=10.0,
                                    return_path=True,
                                )
                                if path:
                                    exit_path = path
                                    break  # Only draw path to nearest exit
                                else:
                                    logger.warning(
                                        f"[PATH_COMPUTE] No path to exit from {ninja_pos}"
                                    )

                # Cache the computed paths
                if level_data and hasattr(self, "_path_visualization_cache"):
                    mine_cache_size = (
                        len(self._mine_proximity_cache.cache)
                        if hasattr(self, "_mine_proximity_cache")
                        else 0
                    )
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
                        mine_cache_size=mine_cache_size,
                    )

                # Create cached_data dict from newly computed paths
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

            # DIAGNOSTIC: Log when using cached paths
            if cached_data and (switch_path or exit_path):
                mine_cache_size = len(self._mine_proximity_cache.cache)
                # logger.warning(
                #     f"[PATH_VIZ] Using cached paths: switch={len(switch_path) if switch_path else 0} nodes, "
                #     f"exit={len(exit_path) if exit_path else 0} nodes, "
                #     f"current_mine_cache_size={mine_cache_size}"
                # )

            # Draw paths (from cache or newly computed)
            # IMPORTANT: Only draw the currently relevant path based on switch activation state
            if closest_node and cached_data:
                # Select which path to draw based on switch activation state
                if exit_switch_activated:
                    # Switch collected - show path to exit
                    current_path = exit_path
                else:
                    # Switch not collected - show path to switch
                    current_path = switch_path

                # Draw the currently relevant path with color-coded edges
                if current_path:
                    for i in range(len(current_path) - 1):
                        node1 = current_path[i]
                        node2 = current_path[i + 1]

                        # Verify nodes are adjacent in the graph
                        neighbors = adjacency.get(node1, [])
                        is_adjacent = any(n[0] == node2 for n in neighbors)

                        if is_adjacent:
                            # Single-color geometric path (physics edge types deprecated)
                            x1, y1 = node1
                            x2, y2 = node2
                            # Add +24 offset to match node visualization (accounts for tile padding)
                            screen_x1 = int(x1 * self.adjust + self.tile_x_offset) + 24
                            screen_y1 = int(y1 * self.adjust + self.tile_y_offset) + 24
                            screen_x2 = int(x2 * self.adjust + self.tile_x_offset) + 24
                            screen_y2 = int(y2 * self.adjust + self.tile_y_offset) + 24
                            pygame.draw.line(
                                surface,
                                PATH_EDGE_COLOR,
                                (screen_x1, screen_y1),
                                (screen_x2, screen_y2),
                                4,  # Thicker for better visibility
                            )

            # === MOVEMENT VECTOR VISUALIZATION ===
            # Draw optimal direction (multi-hop lookahead) and actual velocity vectors
            if ninja_pos and closest_node:
                ninja_x, ninja_y = ninja_pos
                ninja_screen_x = int(ninja_x * self.adjust + self.tile_x_offset)
                ninja_screen_y = int(ninja_y * self.adjust + self.tile_y_offset)

                # Determine current goal and get path for multi-hop visualization
                path = None
                if cached_data:
                    if exit_switch_activated:
                        # Use exit path
                        path = cached_data.get("exit_path")
                    else:
                        # Use switch path
                        path = cached_data.get("switch_path")

                # Draw next 4 hops with decreasing opacity/size
                if path and len(path) >= 2:
                    # Multi-hop visualization: show next 4 hops
                    num_hops_to_show = min(4, len(path) - 1)

                    for hop_idx in range(1, num_hops_to_show + 1):
                        next_hop = path[hop_idx]
                        prev_node = path[hop_idx - 1]

                        # Compute direction from previous node to this hop
                        dx = next_hop[0] - prev_node[0]
                        dy = next_hop[1] - prev_node[1]
                        length = (dx * dx + dy * dy) ** 0.5

                        if length > 0.001:
                            # Normalize direction
                            opt_dir = (dx / length, dy / length)

                            # Compute start position for this arrow
                            # All node positions need +24 offset to match path line rendering
                            if hop_idx == 1:
                                # First hop starts from ninja position (no +24 offset needed, already in screen space)
                                start_x = ninja_screen_x
                                start_y = ninja_screen_y
                            else:
                                # Subsequent hops start from previous node WITH +24 offset to match path lines
                                start_x = (
                                    int(prev_node[0] * self.adjust + self.tile_x_offset)
                                    + 24
                                )
                                start_y = (
                                    int(prev_node[1] * self.adjust + self.tile_y_offset)
                                    + 24
                                )

                            # Visual properties fade with distance
                            # Hop 1: bright green, full opacity, largest
                            # Hop 2-4: progressively dimmer, smaller, more transparent
                            opacity = max(
                                100, 255 - (hop_idx - 1) * 50
                            )  # 255, 205, 155, 105
                            arrow_width = max(1, 4 - hop_idx)  # 3, 2, 1, 1
                            arrow_length = max(
                                25, 50 - (hop_idx - 1) * 8
                            )  # 50, 42, 34, 26
                            head_size = max(4, 12 - (hop_idx - 1) * 2)  # 12, 10, 8, 6

                            # Color: bright green for hop 1, fade to yellow-green for later hops
                            green_val = max(
                                180, 255 - (hop_idx - 1) * 25
                            )  # Keep high green
                            red_val = min(
                                150, 50 + (hop_idx - 1) * 33
                            )  # Add yellow tint
                            color = (red_val, green_val, 50, opacity)

                            # Draw arrow for this hop
                            self._draw_arrow(
                                surface,
                                (start_x, start_y),
                                opt_dir,
                                arrow_length,
                                color,
                                width=arrow_width,
                                head_size=head_size,
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

            # === STRAIGHTNESS MASKING DEBUG VISUALIZATION ===
            # Show the exact physics-optimal path direction used for straightness masking
            straightness_debug = path_aware_info.get("straightness_masking_debug")
            if straightness_debug:
                masking_ninja_node = straightness_debug.get("ninja_node")
                short_horizon_dir = straightness_debug.get("short_horizon_direction")
                multi_hop_geometric = straightness_debug.get(
                    "multi_hop_direction_geometric"
                )
                path_segment = straightness_debug.get("path_segment", [])

                if masking_ninja_node and short_horizon_dir:
                    # Draw the ninja node used for straightness masking (bright magenta circle)
                    node_x, node_y = masking_ninja_node
                    screen_node_x = int(node_x * self.adjust + self.tile_x_offset) + 24
                    screen_node_y = int(node_y * self.adjust + self.tile_y_offset) + 24

                    # Bright magenta with pulsing effect
                    pygame.draw.circle(
                        surface, (255, 0, 255, 255), (screen_node_x, screen_node_y), 8
                    )
                    pygame.draw.circle(
                        surface,
                        (255, 255, 255, 255),
                        (screen_node_x, screen_node_y),
                        8,
                        2,
                    )

                    # Draw PHYSICS-OPTIMAL direction arrow from ninja node (bright magenta)
                    # This is the direction actually used for straightness masking
                    # Computed from the actual physics-optimal path (with mine avoidance, momentum, etc.)
                    dx_physics, dy_physics = short_horizon_dir
                    arrow_length = 60
                    self._draw_arrow(
                        surface,
                        (screen_node_x, screen_node_y),
                        (dx_physics, dy_physics),
                        arrow_length,
                        (255, 0, 255, 255),  # Bright magenta
                        width=5,  # Very thick for visibility
                        head_size=14,
                    )

                    # Draw GEOMETRIC direction arrow for comparison (semi-transparent cyan)
                    # This shows the geometric (straight-line) path from the level cache
                    # which is NOT used for masking (only physics-optimal is used)
                    if multi_hop_geometric:
                        dx_geom, dy_geom = multi_hop_geometric
                        self._draw_arrow(
                            surface,
                            (screen_node_x, screen_node_y),
                            (dx_geom, dy_geom),
                            arrow_length,
                            (0, 255, 255, 140),  # Semi-transparent cyan
                            width=2,
                            head_size=8,
                        )

                    # Draw path segment nodes (the exact hops used for straightness check)
                    for i, segment_node in enumerate(path_segment):
                        seg_x, seg_y = segment_node
                        screen_seg_x = (
                            int(seg_x * self.adjust + self.tile_x_offset) + 24
                        )
                        screen_seg_y = (
                            int(seg_y * self.adjust + self.tile_y_offset) + 24
                        )

                        if i == 0:
                            # First node - bright magenta (start)
                            color = (255, 0, 255, 255)
                            radius = 8
                        elif i == len(path_segment) - 1:
                            # Last node - bright orange (end of segment)
                            color = (255, 128, 0, 255)
                            radius = 7
                        else:
                            # Middle nodes - yellow
                            color = (255, 255, 0, 255)
                            radius = 5

                        pygame.draw.circle(
                            surface, color, (screen_seg_x, screen_seg_y), radius
                        )
                        pygame.draw.circle(
                            surface,
                            (0, 0, 0, 255),
                            (screen_seg_x, screen_seg_y),
                            radius,
                            2,
                        )

                        # Draw line between consecutive segment nodes
                        if i > 0:
                            prev_node = path_segment[i - 1]
                            prev_x, prev_y = prev_node
                            screen_prev_x = (
                                int(prev_x * self.adjust + self.tile_x_offset) + 24
                            )
                            screen_prev_y = (
                                int(prev_y * self.adjust + self.tile_y_offset) + 24
                            )
                            pygame.draw.line(
                                surface,
                                (255, 128, 0, 200),
                                (screen_prev_x, screen_prev_y),
                                (screen_seg_x, screen_seg_y),
                                3,
                            )

                    # Draw text label showing the physics-optimal direction
                    try:
                        label_font = pygame.font.Font(None, 16)
                    except pygame.error:
                        label_font = pygame.font.SysFont("arial", 14)

                    label_text = f"Physics: ({dx_physics:.2f}, {dy_physics:.2f})"
                    label_surf = label_font.render(label_text, True, (255, 0, 255, 255))
                    label_bg = pygame.Surface(
                        (label_surf.get_width() + 8, label_surf.get_height() + 4),
                        pygame.SRCALPHA,
                    )
                    label_bg.fill((0, 0, 0, 200))
                    label_bg.blit(label_surf, (4, 2))

                    # Position label near the arrow
                    label_x = screen_node_x + int(dx_physics * 35)
                    label_y = screen_node_y + int(dy_physics * 35) - 30
                    surface.blit(label_bg, (label_x, label_y))

                    # Draw label for geometric direction if available
                    if multi_hop_geometric:
                        dx_geom, dy_geom = multi_hop_geometric
                        label_text_geom = f"Geometric: ({dx_geom:.2f}, {dy_geom:.2f})"
                        label_surf_geom = label_font.render(
                            label_text_geom, True, (0, 255, 255, 255)
                        )
                        label_bg_geom = pygame.Surface(
                            (
                                label_surf_geom.get_width() + 8,
                                label_surf_geom.get_height() + 4,
                            ),
                            pygame.SRCALPHA,
                        )
                        label_bg_geom.fill((0, 0, 0, 200))
                        label_bg_geom.blit(label_surf_geom, (4, 2))

                        label_x_geom = screen_node_x + int(dx_geom * 35)
                        label_y_geom = screen_node_y + int(dy_geom * 35) + 5
                        surface.blit(label_bg_geom, (label_x_geom, label_y_geom))

                    # Remove the old next hop visualization code
                    masking_next_hop = None  # Not needed anymore

                    if False and masking_next_hop:  # Disabled
                        hop_x, hop_y = masking_next_hop
                        screen_hop_x = (
                            int(hop_x * self.adjust + self.tile_x_offset) + 24
                        )
                        screen_hop_y = (
                            int(hop_y * self.adjust + self.tile_y_offset) + 24
                        )

                        # Draw dashed line
                        num_dashes = 8
                        for i in range(num_dashes):
                            if i % 2 == 0:  # Only draw even dashes
                                t1 = i / num_dashes
                                t2 = (i + 1) / num_dashes
                                x1 = screen_node_x + int(
                                    (screen_hop_x - screen_node_x) * t1
                                )
                                y1 = screen_node_y + int(
                                    (screen_hop_y - screen_node_y) * t1
                                )
                                x2 = screen_node_x + int(
                                    (screen_hop_x - screen_node_x) * t2
                                )
                                y2 = screen_node_y + int(
                                    (screen_hop_y - screen_node_y) * t2
                                )
                                pygame.draw.line(
                                    surface, (255, 255, 0, 255), (x1, y1), (x2, y2), 2
                                )

                        # Draw next hop node (yellow circle)
                        pygame.draw.circle(
                            surface, (255, 255, 0, 255), (screen_hop_x, screen_hop_y), 6
                        )
                        pygame.draw.circle(
                            surface, (0, 0, 0, 255), (screen_hop_x, screen_hop_y), 6, 2
                        )

        # Calculate and display switch/exit distances
        if ninja_pos and adjacency:
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
                # Calculate distances using unified get_path_cost (same as training PBRS)
                switch_dist = float("inf")
                for switch_pos in switch_positions:
                    dist = self._path_calculator.get_path_cost(
                        ninja_pos,
                        switch_pos,
                        adjacency,
                        base_adjacency,
                        level_data=level_data,
                        graph_data=graph_data,
                        entity_radius=6.0,
                        ninja_radius=10.0,
                    )
                    if dist < switch_dist:
                        switch_dist = dist

                exit_dist = float("inf")
                for exit_pos in exit_positions:
                    dist = self._path_calculator.get_path_cost(
                        ninja_pos,
                        exit_pos,
                        adjacency,
                        base_adjacency,
                        level_data=level_data,
                        graph_data=graph_data,
                        entity_radius=12.0,
                        ninja_radius=10.0,
                    )
                    if dist < exit_dist:
                        exit_dist = dist

                # Draw info box with switch/exit distances
                try:
                    title_font = pygame.font.Font(None, 18)
                    box_font = pygame.font.Font(None, 20)
                except pygame.error:
                    title_font = pygame.font.SysFont("monospace", 14, bold=True)
                    box_font = pygame.font.SysFont("monospace", 16, bold=True)

                ninja_screen_x = int(ninja_x * self.adjust + self.tile_x_offset)
                ninja_screen_y = int(ninja_y * self.adjust + self.tile_y_offset)

                # Draw background box
                box_width = 200
                box_height = 75
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
                    (50, 50, box_width, box_height),
                    border_radius=5,
                )
                pygame.draw.rect(
                    surface,
                    (100, 200, 255, 255),
                    (50, 50, box_width, box_height),
                    2,
                    border_radius=5,
                )

                # Draw title at top of box
                title_text = "Physics Cost (PBRS)"
                title_surf = title_font.render(title_text, True, (200, 200, 255, 255))
                title_x = 50 + (box_width - title_surf.get_width()) // 2
                surface.blit(title_surf, (title_x, 50 + 5))

                # Draw text with clearer labels and decimal precision
                switch_text = (
                    f"Switch Cost: {switch_dist:.1f}"
                    if switch_dist != float("inf")
                    else "Switch Cost: ∞"
                )
                exit_text = (
                    f"Exit Cost: {exit_dist:.1f}"
                    if exit_dist != float("inf")
                    else "Exit Cost: ∞"
                )

                switch_surf = box_font.render(switch_text, True, SWITCH_NODE_COLOR)
                exit_surf = box_font.render(exit_text, True, EXIT_NODE_COLOR)

                surface.blit(switch_surf, (50 + 10, 50 + 28))
                surface.blit(exit_surf, (50 + 10, 50 + 50))

        if hazard_combo:
            try:
                legend_font = pygame.font.Font(None, 18)
            except pygame.error:
                legend_font = pygame.font.SysFont("arial", 14)
            lines = [
                "Hazard (H): SDF heatmap under graph",
                "Yellow-red nodes: path mine cost >1x",
            ]
            pad = 6
            y = self.screen.get_height() - 8
            for line in reversed(lines):
                text = legend_font.render(line, True, (255, 255, 255))
                bg = pygame.Surface(
                    (text.get_width() + pad * 2, text.get_height() + 4), pygame.SRCALPHA
                )
                bg.fill((0, 0, 0, 200))
                bg.blit(text, (pad, 2))
                y -= bg.get_height() + 4
                surface.blit(bg, (8, y))

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
            pygame.draw.circle(
                surface, (255, 255, 100, 255), (screen_x, screen_y), 8, 2
            )

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

    def _draw_reachability_overlay(
        self, reachability_info: dict
    ) -> Optional[pygame.Surface]:
        """Draw hull-based reachability overlay as a shaded area.

        Two layers:
        - Dim blue: full physics hull (everywhere the ninja CAN go)
        - Bright green: level-masked reachable subset (geometry + mines)

        Args:
            reachability_info: Dictionary containing:
                - hull_cells: list of (abs_x, abs_y, min_frames)
                - masked_cells: list of (abs_x, abs_y, min_frames)
                - ninja_position: (x, y)
                - ninja_velocity: (vx, vy)
                - state: int
                - contact_type: str
                - surface_type: int

        Returns:
            pygame.Surface with hull area overlay
        """
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        hull_cells = reachability_info.get("hull_cells", [])
        masked_cells = reachability_info.get("masked_cells", [])
        ninja_vel = reachability_info.get("ninja_velocity", (0, 0))
        contact_type = reachability_info.get("contact_type", "floor")
        surface_type = reachability_info.get("surface_type", 0)
        walled = reachability_info.get("walled", False)
        wall_normal = reachability_info.get("wall_normal")
        jump_type = reachability_info.get("jump_type", "floor")
        airborn = reachability_info.get("airborn", False)

        # Cell size in screen pixels (12 world px * scale factor)
        cell_screen = max(1, int(12 * self.adjust))
        half_cell = cell_screen // 2

        # 1. Draw full physics hull (dim blue rectangles)
        for abs_x, abs_y, min_frames in hull_cells:
            sx = int(abs_x * self.adjust) + self.tile_x_offset
            sy = int(abs_y * self.adjust) + self.tile_y_offset
            rect = pygame.Rect(
                sx - half_cell,
                sy - half_cell,
                cell_screen,
                cell_screen,
            )
            pygame.draw.rect(surface, (60, 80, 180, 40), rect)

        # 2. Draw level-masked reachable subset (bright green rectangles)
        for abs_x, abs_y, min_frames in masked_cells:
            sx = int(abs_x * self.adjust) + self.tile_x_offset
            sy = int(abs_y * self.adjust) + self.tile_y_offset
            rect = pygame.Rect(
                sx - half_cell,
                sy - half_cell,
                cell_screen,
                cell_screen,
            )
            # Alpha varies by min_frames: closer cells are brighter
            alpha = max(40, 120 - min_frames)
            pygame.draw.rect(surface, (50, 220, 80, alpha), rect)

        # 3. Draw info panel
        try:
            title_font = pygame.font.Font(None, 18)
            info_font = pygame.font.Font(None, 16)
        except pygame.error:
            title_font = pygame.font.SysFont("monospace", 14, bold=True)
            info_font = pygame.font.SysFont("monospace", 12)

        panel_width = 210
        panel_height = 145
        panel_x = 10
        panel_y = self.screen.get_height() - panel_height - 10

        pygame.draw.rect(
            surface,
            (0, 0, 0, 210),
            (panel_x, panel_y, panel_width, panel_height),
            border_radius=5,
        )
        pygame.draw.rect(
            surface,
            (100, 200, 255, 255),
            (panel_x, panel_y, panel_width, panel_height),
            2,
            border_radius=5,
        )

        title_surf = title_font.render("Hull Reachability", True, (200, 220, 255))
        surface.blit(title_surf, (panel_x + 10, panel_y + 6))

        vx, vy = ninja_vel
        surface_names = {
            0: "flat",
            1: "mild-L",
            2: "mild-R",
            3: "45-L",
            4: "45-R",
            5: "steep-L",
            6: "steep-R",
        }
        if surface_type >= 0:
            sname = surface_names.get(surface_type, "?")
            contact_label = f"Contact: {contact_type}  ({sname})"
        else:
            contact_label = f"Contact: {contact_type}"
        jump_state_lines = [
            (f"walled={walled}  jump={jump_type}  airborn={airborn}", (255, 220, 150)),
        ]
        if walled and wall_normal is not None:
            jump_state_lines.append(
                (f"wall_normal={wall_normal:.2f}", (255, 220, 150))
            )
        lines = [
            (contact_label, (200, 200, 255)),
            *jump_state_lines,
            (f"vx={vx:.2f}  vy={vy:.2f}", (180, 255, 180)),
            (f"Hull cells: {len(hull_cells)}", (100, 150, 255)),
            (f"Masked (reachable): {len(masked_cells)}", (100, 255, 100)),
        ]

        y_off = 26
        for text, color in lines:
            surf = info_font.render(text, True, color)
            surface.blit(surf, (panel_x + 10, panel_y + y_off))
            y_off += 16

        return surface

    def _draw_value_function(
        self, value_function_info: dict
    ) -> Optional[pygame.Surface]:
        """Draw value function (PBRS cost-to-go) overlay.

        Shows real-time cost-to-go and potential to help understand how
        PBRS shapes rewards during training. Displays:
        - Current potential (Phi) and cost-to-go (frames)
        - Phase (switch vs exit goal)
        - Contact state (floor/wall/airborne)
        - Optional: gradient arrow toward lower cost

        Args:
            value_function_info: Dictionary containing:
                - potential: float - current PBRS potential
                - cost_frames: float - estimated frames to goal
                - phase: str - "switch" or "exit"
                - contact_state: int - 0=floor, 1=wall, 2=airborne
                - ninja_position: (x, y)
                - show_heatmap: bool - if True, show surface point heatmap

        Returns:
            pygame.Surface with value function overlay
        """
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        potential = value_function_info.get("potential")
        if potential is None:
            return surface

        cost_frames = value_function_info.get("cost_frames")
        phase = value_function_info.get("phase", "switch")
        contact_state = value_function_info.get("contact_state", 2)
        ninja_pos = value_function_info.get("ninja_position", (0, 0))
        show_heatmap = value_function_info.get("show_heatmap", False)

        # Build info panel
        try:
            title_font = pygame.font.Font(None, 18)
            box_font = pygame.font.Font(None, 16)
        except pygame.error:
            title_font = pygame.font.SysFont("monospace", 14, bold=True)
            box_font = pygame.font.SysFont("monospace", 12)

        ninja_x, ninja_y = ninja_pos
        ninja_screen_x = int(ninja_x * self.adjust) + self.tile_x_offset
        ninja_screen_y = int(ninja_y * self.adjust) + self.tile_y_offset

        box_width = 220
        box_height = 110
        box_x = ninja_screen_x + 25
        box_y = ninja_screen_y - 55

        if box_x + box_width > self.screen.get_width():
            box_x = ninja_screen_x - box_width - 25
        if box_y < 0:
            box_y = ninja_screen_y + 25
        if box_y + box_height > self.screen.get_height():
            box_y = self.screen.get_height() - box_height - 10

        pygame.draw.rect(
            surface,
            (0, 0, 0, 220),
            (box_x, box_y, box_width, box_height),
            border_radius=5,
        )
        pygame.draw.rect(
            surface,
            (80, 180, 255, 255),
            (box_x, box_y, box_width, box_height),
            2,
            border_radius=5,
        )

        title_text = "Value Function (PBRS)"
        title_surf = title_font.render(title_text, True, (200, 220, 255))
        surface.blit(title_surf, (box_x + 10, box_y + 6))

        potential_str = f"Φ = {potential:.3f}"
        cost_str = (
            f"Cost: {cost_frames:.0f} frames"
            if cost_frames is not None and cost_frames < 1e6
            else "Cost: ∞"
        )
        phase_str = f"Phase: {phase}"
        contact_names = ("floor", "wall", "airborne")
        contact_str = f"Contact: {contact_names[min(2, max(0, contact_state))]}"

        y_off = 28
        for text, color in [
            (potential_str, (150, 255, 150)),
            (cost_str, (255, 220, 100)),
            (phase_str, (255, 180, 100)),
            (contact_str, (200, 200, 255)),
        ]:
            surf = box_font.render(text, True, color)
            surface.blit(surf, (box_x + 10, box_y + y_off))
            y_off += 18

        # Draw gradient arrow if available (points toward higher potential / lower cost)
        gradient = value_function_info.get("gradient")
        if gradient is not None:
            dx, dy = gradient
            length = (dx * dx + dy * dy) ** 0.5
            if length > 0.01:
                norm = 1.0 / length
                self._draw_arrow(
                    surface,
                    (ninja_screen_x, ninja_screen_y),
                    (dx * norm, dy * norm),
                    50,
                    (100, 255, 150, 200),
                    width=2,
                    head_size=8,
                )

        # Optional: surface point heatmap (sparse, for performance)
        if show_heatmap and "surface_points" in value_function_info:
            self._draw_value_function_heatmap(
                surface, value_function_info["surface_points"]
            )

        return surface

    def _draw_value_function_heatmap(
        self, surface: pygame.Surface, points_data: dict
    ) -> None:
        """Draw sparse heatmap of surface points colored by cost-to-go."""
        xs = points_data.get("x")
        ys = points_data.get("y")
        costs = points_data.get("cost")
        if xs is None or ys is None or costs is None or len(xs) == 0:
            return

        cost_max = max(c for c in costs if c < 1e6) if costs else 1.0
        if cost_max <= 0:
            cost_max = 1.0

        cell_size = max(6, 12 * self.adjust)
        for i in range(min(len(xs), 2000)):  # Limit for performance
            x, y = float(xs[i]), float(ys[i])
            c = float(costs[i])
            if c >= cost_max * 0.99:
                continue
            t = c / cost_max
            r = int(255 * (1 - t))
            g = int(100 + 155 * (1 - t))
            b = 50
            screen_x = int(x * self.adjust) + self.tile_x_offset
            screen_y = int(y * self.adjust) + self.tile_y_offset
            rect = pygame.Rect(
                screen_x - cell_size // 2,
                screen_y - cell_size // 2,
                cell_size,
                cell_size,
            )
            pygame.draw.rect(surface, (r, g, b, 80), rect)

    def _blit_benchmark_hud_text(self, surface: pygame.Surface, lines) -> None:
        """Top-left mechanic-benchmark / tool HUD (keeps parity with test_environment styling)."""
        if not lines:
            return
        font = pygame.font.SysFont("monospace", 14)
        x0, y = 8, 6
        pad = 4
        for line in lines:
            s = str(line)
            if not s:
                continue
            text = font.render(s, True, (255, 255, 255))
            bg = pygame.Surface(
                (text.get_width() + pad * 2, text.get_height() + 4)
            )
            bg.set_alpha(210)
            bg.fill((20, 22, 32))
            surface.blit(bg, (x0, y))
            surface.blit(text, (x0 + pad, y + 2))
            y += bg.get_height() + 4

    # Ghost-arc overlay (Audit II P0-1): per-u polyline colors and per-class
    # contact-marker colors. u=-1 orange / u=0 cyan / u=+1 green; contact
    # class 0=floor 1=wall 2=ceiling 3=deadly-mine 4=none.
    GHOST_ARC_COLORS = {
        -1: (255, 160, 60, 220),
        0: (80, 220, 255, 230),
        1: (120, 255, 120, 220),
    }
    # Grounded takeoff rows (v13 jump-now full-hold / v14 tap): the same
    # per-u hue family, but the tap row is drawn dimmer + thinner so both
    # counterfactual rows read at a glance without swamping the scene.
    TAP_ARC_COLORS = {
        -1: (200, 120, 40, 170),
        0: (60, 160, 190, 180),
        1: (90, 190, 90, 170),
    }
    GHOST_CONTACT_COLORS = {
        0: (90, 230, 90, 255),    # floor: green
        1: (90, 140, 255, 255),   # wall: blue
        2: (200, 110, 255, 255),  # ceiling: purple
        3: (255, 70, 70, 255),    # deadly mine: red
        4: (160, 160, 160, 200),  # none within horizon: grey
    }

    def _draw_ghost_arcs(self, info: dict) -> Optional[pygame.Surface]:
        """Draw the ballistic ghost arcs the obs bundles are computed from.

        Airborne: the three committed steering arcs (u = -1/0/+1). Grounded:
        BOTH counterfactual takeoff rows the obs exposes — the v13 jump-now
        full-hold arcs (solid, bright) and the v14 tap-jump arcs (thin, dim;
        jump released immediately, the low-ceiling hop). Per arc: the exact
        native polyline to first contact, a contact marker colored by contact
        class (mine = X), the surface normal tick, and a compact annotation
        "τ / impact-ratio / clear-width". When the native projector is
        unavailable a small status label is shown instead.
        """
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        font = pygame.font.Font(None, 18)

        def to_screen(p):
            return (
                int(p[0] * self.adjust) + self.tile_x_offset,
                int(p[1] * self.adjust) + self.tile_y_offset,
            )

        ninja = info.get("ninja", (0.0, 0.0))
        if info.get("unavailable"):
            label = self._render_text_cached(
                font, "ghost arcs: native projector unavailable", (255, 120, 120)
            )
            surface.blit(label, to_screen((ninja[0] + 14, ninja[1] - 14)))
            return surface

        def draw_row(arcs, colors, width, tag, note_row):
            for arc in arcs:
                u = arc.get("u", 0)
                color = colors.get(u, (255, 255, 255, 220))
                pts = [to_screen(p) for p in arc.get("path", [])]
                if len(pts) >= 2:
                    pygame.draw.lines(surface, color, False, pts, width)

                cls = int(arc.get("class", 4))
                ccolor = self.GHOST_CONTACT_COLORS.get(cls, (255, 255, 255, 255))
                cx, cy = to_screen(arc.get("contact", ninja))
                if cls == 3:  # deadly mine: X marker
                    r = 7
                    pygame.draw.line(
                        surface, ccolor, (cx - r, cy - r), (cx + r, cy + r), 3
                    )
                    pygame.draw.line(
                        surface, ccolor, (cx - r, cy + r), (cx + r, cy - r), 3
                    )
                elif cls == 4:  # no contact within horizon: hollow circle
                    pygame.draw.circle(surface, ccolor, (cx, cy), 6, 2)
                else:
                    pygame.draw.circle(surface, ccolor, (cx, cy), 6, 0 if width > 1 else 1)
                    # Surface normal tick (points away from the surface).
                    nx, ny = arc.get("normal", (0.0, 0.0))
                    if abs(nx) + abs(ny) > 1e-6:
                        tick_len = 14
                        pygame.draw.line(
                            surface, ccolor, (cx, cy),
                            (int(cx + nx * tick_len), int(cy + ny * tick_len)), 2,
                        )

                # v15 contact-velocity vector (the landing velocity the obs
                # exposes at [9]/[10]): a yellow arrow from the contact along
                # (vx, vy), length scaled for visibility. Drawn for real surface
                # contacts (floor/wall/ceiling); mine/none arcs skip it.
                if cls in (0, 1, 2):
                    cvx, cvy = arc.get("contact_vel", (0.0, 0.0))
                    vmag = (cvx * cvx + cvy * cvy) ** 0.5
                    if vmag > 0.05:
                        vscale = 7.0  # screen px per (px/frame)
                        ex = int(cx + cvx * vscale)
                        ey = int(cy + cvy * vscale)
                        vcol = (250, 230, 60, 255)
                        pygame.draw.line(surface, vcol, (cx, cy), (ex, ey), 2)
                        # Arrowhead: two short barbs against the velocity dir.
                        ux, uy = cvx / vmag, cvy / vmag
                        for s in (0.5, -0.5):
                            hx = int(ex - 6 * ux + 4 * (-uy) * s)
                            hy = int(ey - 6 * uy + 4 * ux * s)
                            pygame.draw.line(surface, vcol, (ex, ey), (hx, hy), 2)

                # v15 mine-threading warning ring: when the arc's closest
                # approach to a deadly mine ([12] mine_proximity) is inside the
                # danger band, ring the contact orange->red as clearance shrinks
                # (a SURVIVING arc that still threads dangerously close).
                mclr = arc.get("mine_clearance", float("inf"))
                if mclr != float("inf") and mclr < 48.0:
                    t = max(0.0, min(1.0, (48.0 - mclr) / 34.0))  # 0 far -> 1 near
                    warn = (255, int(170 * (1.0 - t)), 40, 255)
                    pygame.draw.circle(surface, warn, (cx, cy), 11, 2)

                # Per-arc annotation: τ frames / impact ratio / clear width,
                # plus v15 contact velocity + surface class. Impact ratio >= 1
                # means the no-brake landing is LETHAL — red.
                imp = float(arc.get("impact_ratio", 0.0))
                cvx, cvy = arc.get("contact_vel", (0.0, 0.0))
                sc = arc.get("surface_class", 0)
                sc_tag = {1: "flr", -1: "cei"}.get(sc, "wal")
                mclr = arc.get("mine_clearance", float("inf"))
                mtag = "" if mclr == float("inf") else f" m{mclr:.0f}"
                # v16 (feature C): arc->goal closest approach. Gold ring at the
                # contact when this arc reaches the active objective (<=12px);
                # "G<px>" tag otherwise while within the graded range.
                gclr = arc.get("goal_clearance", float("inf"))
                gtag = ""
                if gclr != float("inf"):
                    if gclr <= 12.0:
                        gtag = " G-HIT"
                        pygame.draw.circle(surface, (255, 215, 0, 255), (cx, cy), 13, 3)
                    elif gclr < 108.0:
                        gtag = f" G{gclr:.0f}"
                note = (
                    f"{tag}u{u:+d}  τ{arc.get('tau', 0):.0f}f  "
                    f"imp{imp:.2f}  cw{arc.get('clear_width', 0)}  "
                    f"v({cvx:+.1f},{cvy:+.1f}) {sc_tag}{mtag}{gtag}"
                )
                note_color = (255, 90, 90) if imp >= 1.0 else color[:3]
                label = self._render_text_cached(font, note, note_color)
                surface.blit(
                    label, (cx + 10, cy + 4 + 14 * (u + 1) + note_row * 46)
                )

        def draw_passive(pa):
            """v16 batch A passive-release arc: the 'let go of everything now'
            trajectory (u=0, jf=0). Drawn as a white dashed polyline so it is
            distinct from the committed steering arcs, with a large ring at the
            landing point (green=safe, red=FATAL) and a landing-velocity arrow.
            This is exactly the arc the game_state passive block encodes."""
            path = pa.get("path", [])
            pts = [to_screen(p) for p in path]
            fatal = bool(pa.get("fatal", False))
            is_none = bool(pa.get("is_none", False))
            base = (255, 80, 80, 255) if fatal else (240, 240, 240, 230)
            # Dashed: draw every other segment.
            for i in range(0, len(pts) - 1, 2):
                pygame.draw.line(surface, base, pts[i], pts[i + 1], 2)
            cx, cy = to_screen(pa.get("contact", ninja))
            if is_none:
                pygame.draw.circle(surface, (200, 200, 200, 200), (cx, cy), 9, 2)
            else:
                ring = (255, 60, 60, 255) if fatal else (90, 230, 90, 255)
                pygame.draw.circle(surface, ring, (cx, cy), 10, 3)
                cvx, cvy = pa.get("contact_vel", (0.0, 0.0))
                vmag = (cvx * cvx + cvy * cvy) ** 0.5
                if vmag > 0.05:
                    ex, ey = int(cx + cvx * 7.0), int(cy + cvy * 7.0)
                    pygame.draw.line(surface, (255, 255, 255, 255), (cx, cy), (ex, ey), 2)
            imp = float(pa.get("impact_ratio", 0.0))
            tag = "PASSIVE FATAL" if fatal else ("passive none" if is_none else "passive")
            note = f"{tag}  τ{pa.get('tau', 0):.0f}f  imp{imp:.2f}"
            label = self._render_text_cached(
                font, note, (255, 90, 90) if fatal else (230, 230, 230)
            )
            surface.blit(label, (cx + 10, cy - 16))

        if info.get("airborne", False):
            draw_row(info.get("arcs", []), self.GHOST_ARC_COLORS, 2, "", 0)
            passive = info.get("passive_arc")
            if passive:
                draw_passive(passive)
            # Wall-jump-now row (v14): present only while walled — dim/thin,
            # same palette family as the tap row.
            wall_arcs = info.get("wall_arcs", [])
            if wall_arcs:
                draw_row(wall_arcs, self.TAP_ARC_COLORS, 1, "wj ", 1)
                key = self._render_text_cached(
                    font, "bright=committed  dim=wall-jump-now", (200, 200, 200)
                )
                surface.blit(key, to_screen((ninja[0] + 14, ninja[1] - 34)))
        else:
            jump_arcs = info.get("jump_arcs", [])
            tap_arcs = info.get("tap_arcs", [])
            if not jump_arcs and not tap_arcs:
                label = self._render_text_cached(
                    font, "ghost arcs: grounded (fast-path zeros)", (170, 170, 170)
                )
                surface.blit(label, to_screen((ninja[0] + 14, ninja[1] - 20)))
                return surface
            # v13 jump-now full-hold row (bright/solid) + v14 tap row (dim/thin).
            draw_row(jump_arcs, self.GHOST_ARC_COLORS, 2, "hold ", 0)
            draw_row(tap_arcs, self.TAP_ARC_COLORS, 1, "tap ", 1)
            key = self._render_text_cached(
                font, "takeoff arcs: bright=full-hold  dim=tap", (200, 200, 200)
            )
            surface.blit(key, to_screen((ninja[0] + 14, ninja[1] - 34)))

        return surface

    def _draw_raycasts(self, info: dict) -> Optional[pygame.Surface]:
        """Draw the 8 cardinal surface-distance rays the game_state RAYCAST block
        exposes ([E, NE, N, NW, W, SW, S, SE]).

        Each ray is drawn from the ninja to the EXACT normalized obs distance
        (0 = adjacent surface, 1.0 = nothing within 500 px), so the picture is
        literally the feature vector. A ray that found a surface is solid with a
        filled endpoint colored near(red) -> far(green); a ray that saturated
        (no surface within range) fades out to the horizon with a hollow
        endpoint. Each ray is labelled with its compass point + normalized value.
        """
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        font = pygame.font.Font(None, 16)

        def to_screen(p):
            return (
                int(p[0] * self.adjust) + self.tile_x_offset,
                int(p[1] * self.adjust) + self.tile_y_offset,
            )

        for ray in info.get("rays", []):
            sx, sy = to_screen(ray.get("start", (0.0, 0.0)))
            ex, ey = to_screen(ray.get("end", (0.0, 0.0)))
            d = float(ray.get("dist_norm", 1.0))
            if ray.get("hit", False):
                # near surface (d small) = red; open (d large) = green.
                col = (
                    min(255, int(230 * (1.0 - d) + 40)),
                    min(255, int(200 * d + 40)),
                    50,
                    255,
                )
                pygame.draw.line(surface, col, (sx, sy), (ex, ey), 2)
                pygame.draw.circle(surface, col, (ex, ey), 5, 0)
            else:
                col = (90, 150, 90, 130)  # no surface within range: faded
                pygame.draw.line(surface, col, (sx, sy), (ex, ey), 1)
                pygame.draw.circle(surface, col, (ex, ey), 5, 1)
            lbl = self._render_text_cached(
                font, f"{ray.get('label', '')}:{d:.2f}", col[:3]
            )
            surface.blit(lbl, (ex + 4, ey - 6))
        return surface

    def draw_debug_overlay(self, debug_info: dict = None) -> pygame.Surface:
        """Helper method to draw debug overlay with nested dictionary support.

        Args:
            debug_info: Optional dictionary containing debug information to display

        Returns:
            pygame.Surface: Surface containing the rendered debug text and exploration grid.
        """
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        if debug_info is None:
            return surface

        # Draw grid outline if provided
        if debug_info and "grid_outline" in debug_info:
            grid_surface = self._draw_grid_outline()
            surface.blit(grid_surface, (0, 0))

        # Draw tile types if provided
        if debug_info and "tile_types" in debug_info:
            tile_types_surface = self._draw_tile_types()
            surface.blit(tile_types_surface, (0, 0))

        # Path-aware + mine SDF: when hazard combo, draw SDF first so graph/nodes stay readable
        path_aware_info = debug_info.get("path_aware") if debug_info else None
        mine_sdf_info = debug_info.get("mine_sdf") if debug_info else None
        hazard_combo = bool(
            path_aware_info and path_aware_info.get("hazard_combo") and mine_sdf_info
        )

        if hazard_combo:
            mine_sdf_surface = self._draw_mine_sdf(mine_sdf_info)
            if mine_sdf_surface:
                surface.blit(mine_sdf_surface, (0, 0))
            path_aware_surface = self._draw_path_aware(path_aware_info)
            if path_aware_surface:
                surface.blit(path_aware_surface, (0, 0))
        else:
            if path_aware_info is not None:
                path_aware_surface = self._draw_path_aware(path_aware_info)
                if path_aware_surface:
                    surface.blit(path_aware_surface, (0, 0))

            if mine_sdf_info is not None:
                mine_sdf_surface = self._draw_mine_sdf(mine_sdf_info)
                if mine_sdf_surface:
                    surface.blit(mine_sdf_surface, (0, 0))

        # Draw ghost-arc ballistic projections if provided (Audit II P0-1)
        if debug_info and "ghost_arcs" in debug_info:
            ghost_surface = self._draw_ghost_arcs(debug_info["ghost_arcs"])
            if ghost_surface:
                surface.blit(ghost_surface, (0, 0))

        # Draw cardinal surface-distance raycasts if provided (game_state RAYCAST)
        if debug_info and "raycasts" in debug_info:
            raycast_surface = self._draw_raycasts(debug_info["raycasts"])
            if raycast_surface:
                surface.blit(raycast_surface, (0, 0))

        # Draw jump-aware cost-to-go heatmap if provided
        jump_cost_info = debug_info.get("jump_cost_to_go") if debug_info else None
        if jump_cost_info is not None:
            jump_cost_surface = self._draw_jump_cost_to_go_heatmap(jump_cost_info)
            if jump_cost_surface:
                surface.blit(jump_cost_surface, (0, 0))

        # Draw action mask visualization if provided
        if debug_info and "action_mask" in debug_info:
            action_mask_surface = self._draw_action_mask(debug_info["action_mask"])
            if action_mask_surface:
                surface.blit(action_mask_surface, (0, 0))

        # Draw demo checkpoint heatmap if provided
        if debug_info and "demo_checkpoints" in debug_info:
            demo_checkpoint_surface = self._draw_demo_checkpoints_heatmap(
                debug_info["demo_checkpoints"]
            )
            if demo_checkpoint_surface:
                surface.blit(demo_checkpoint_surface, (0, 0))

        # Draw analytical reachability overlay if provided
        if debug_info and "reachability_overlay" in debug_info:
            reachability_surface = self._draw_reachability_overlay(
                debug_info["reachability_overlay"]
            )
            if reachability_surface:
                surface.blit(reachability_surface, (0, 0))

        # Draw value function (PBRS cost-to-go) overlay if provided
        if debug_info and "value_function" in debug_info:
            value_fn_surface = self._draw_value_function(debug_info["value_function"])
            if value_fn_surface:
                surface.blit(value_fn_surface, (0, 0))

        # Draw PBRS path overlay if provided
        if debug_info and "pbrs_path_overlay" in debug_info:
            pbrs_path_surface = self._draw_pbrs_path_overlay(
                debug_info["pbrs_path_overlay"]
            )
            if pbrs_path_surface:
                surface.blit(pbrs_path_surface, (0, 0))

        # Draw route PBRS overlay if provided (surface heatmap + arcs + route)
        if debug_info and "route_overlay" in debug_info:
            from nclone.value_function.route_overlay_draw import draw_route_overlay_pygame
            route_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            data = dict(debug_info["route_overlay"])
            data["screen_w"] = self.screen.get_width()
            data["screen_h"] = self.screen.get_height()
            draw_route_overlay_pygame(
                route_surface, data, self.adjust, self.tile_x_offset, self.tile_y_offset
            )
            surface.blit(route_surface, (0, 0))

        # Draw step-on-input sequence overlay if provided
        if debug_info and "input_sequence" in debug_info:
            input_seq_surface = self._draw_input_sequence_overlay(
                debug_info["input_sequence"]
            )
            if input_seq_surface:
                surface.blit(input_seq_surface, (0, 0))

        # TAS editor overlay: trajectory polyline + piano-roll timeline + HUD
        if debug_info and "tas" in debug_info:
            self._draw_tas_overlay(surface, debug_info["tas"])

        # Mechanic-benchmark / tooling HUD: top-left text (single composited pass with the game)
        if debug_info and "benchmark_hud" in debug_info:
            self._blit_benchmark_hud_text(surface, debug_info["benchmark_hud"])

        return surface

    def _draw_tas_overlay(self, surface: pygame.Surface, tas: dict) -> None:
        """Draw the TAS editor: ninja trajectory + piano-roll timeline + HUD.

        ``tas`` is the payload from
        :func:`nclone.tas_tool.overlay.build_tas_payload`. Trajectory points are
        in sim pixel coordinates and transformed to screen space here using the
        renderer's current ``adjust``/offset (kept in sync each draw).
        """
        from .tas_tool.layout import (
            ACTION_LABEL,
            CELL_W,
            strip_rect,
            visible_window,
        )

        def to_screen(x, y):
            return (
                int(x * self.adjust + self.tile_x_offset),
                int(y * self.adjust + self.tile_y_offset),
            )

        # ---- trajectory polyline (grounded green, airborne cyan) ----------
        traj = tas.get("trajectory") or []
        if len(traj) >= 2:
            for i in range(1, len(traj)):
                x0, y0, _ = traj[i - 1]
                x1, y1, air1 = traj[i]
                color = (90, 200, 255) if air1 else (120, 230, 140)
                pygame.draw.line(
                    surface, color, to_screen(x0, y0), to_screen(x1, y1), 2
                )

        # ---- NOOP fall-prediction ghost (dashed orange, from the cursor) --
        pred = tas.get("prediction") or []
        if len(pred) >= 1:
            cpx0, cpy0 = tas.get("cursor_point", (0, 0))
            prev = to_screen(cpx0, cpy0)
            for i, (px, py, _air) in enumerate(pred):
                cur_pt = to_screen(px, py)
                if i % 2 == 0:  # dashed: draw every other segment
                    pygame.draw.line(surface, (255, 160, 40), prev, cur_pt, 1)
                prev = cur_pt
            # Mark the predicted landing/endpoint.
            ex, ey = to_screen(pred[-1][0], pred[-1][1])
            pygame.draw.circle(surface, (255, 160, 40), (ex, ey), 3, 1)

        # Cursor marker (where the live sim currently is) + faint full-path dots
        cpx, cpy = tas.get("cursor_point", (0, 0))
        csx, csy = to_screen(cpx, cpy)
        pygame.draw.circle(surface, (255, 230, 40), (csx, csy), 6, 2)
        pygame.draw.circle(surface, (255, 230, 40), (csx, csy), 2)

        # Mark win/death at the terminal frame if any.
        for fx, fy, _air in traj[-1:]:
            if tas.get("status") == "WON":
                pygame.draw.circle(surface, (80, 255, 120), to_screen(fx, fy), 8, 2)
            elif tas.get("status") == "DEAD":
                ex, ey = to_screen(fx, fy)
                pygame.draw.line(surface, (255, 70, 70), (ex - 6, ey - 6), (ex + 6, ey + 6), 2)
                pygame.draw.line(surface, (255, 70, 70), (ex - 6, ey + 6), (ex + 6, ey - 6), 2)

        screen_w, screen_h = self.screen.get_size()

        # ---- HUD text row -------------------------------------------------
        font = pygame.font.Font(None, 22)
        small = pygame.font.Font(None, 18)
        cursor = int(tas.get("cursor", 0))
        length = int(tas.get("length", 0))
        status = tas.get("status", "EDIT")
        status_color = {
            "WON": (80, 255, 120),
            "DEAD": (255, 90, 90),
            "EDIT": (230, 230, 230),
        }.get(status, (230, 230, 230))
        vx, vy = tas.get("cursor_vel", (0.0, 0.0))
        play = "▶PLAY" if tas.get("playing") else "❚❚PAUSE"
        batch = tas.get("batch_typed") or 0
        batch_str = f"  x{batch}" if batch else ""
        hud_line = (
            f"TAS  f {cursor}/{length}  [{status}]  pend:{tas.get('pending_name', '?')}"
            f"{batch_str}  {play}   vel({vx:+.2f},{vy:+.2f})"
        )
        hud_surf = font.render(hud_line, True, status_color)
        hud_y = screen_h - 8 - CELL_W - 24 - hud_surf.get_height()
        bg = pygame.Surface((hud_surf.get_width() + 10, hud_surf.get_height() + 4), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 150))
        surface.blit(bg, (8, hud_y - 2))
        surface.blit(hud_surf, (12, hud_y))

        # Controls legend (top-right, compact).
        legend = [
            "←/→ pend dir   ↑ pend jump   C clear",
            "SPACE step(xN)   ,/. scrub   BKSP undo",
            "O overwrite  I insert  DEL del  T cut tail",
            "P play  G fall-predict  digits=batch",
            "S save  L load  R reset  Q quit",
        ]
        ly = 10
        for line in legend:
            ls = small.render(line, True, (210, 210, 210))
            lb = pygame.Surface((ls.get_width() + 8, ls.get_height() + 2), pygame.SRCALPHA)
            lb.fill((0, 0, 0, 140))
            surface.blit(lb, (screen_w - ls.get_width() - 14, ly))
            surface.blit(ls, (screen_w - ls.get_width() - 10, ly + 1))
            ly += ls.get_height() + 3

        msg = tas.get("message")
        if msg:
            ms = small.render(str(msg), True, (255, 220, 120))
            surface.blit(ms, (12, hud_y - ms.get_height() - 4))

        # ---- piano-roll strip --------------------------------------------
        sx, sy, sw, sh = strip_rect(screen_w, screen_h)
        strip_bg = pygame.Surface((sw, sh), pygame.SRCALPHA)
        strip_bg.fill((0, 0, 0, 160))
        surface.blit(strip_bg, (sx, sy))

        actions = tas.get("actions") or []
        start, visible = visible_window(screen_w, cursor, length)
        cell_font = pygame.font.Font(None, 18)
        idx_font = pygame.font.Font(None, 13)
        for col in range(visible):
            frame = start + col
            if frame > length:
                break
            cx = sx + col * CELL_W
            is_cursor = frame == cursor
            is_tip = frame == length  # one-past-end "append here" slot
            # Cell background
            if is_cursor:
                cell_color = (250, 210, 40, 220)
            elif is_tip:
                cell_color = (40, 40, 40, 120)
            elif (frame // 10) % 2 == 0:
                cell_color = (60, 60, 70, 120)
            else:
                cell_color = (45, 45, 55, 120)
            cell = pygame.Surface((CELL_W - 1, sh - 1), pygame.SRCALPHA)
            cell.fill(cell_color)
            surface.blit(cell, (cx, sy))

            # Frame index every 10 frames (and at the cursor).
            if frame % 10 == 0 or is_cursor:
                istxt = idx_font.render(str(frame), True, (180, 180, 180))
                surface.blit(istxt, (cx + 1, sy + 1))

            if frame < len(actions):
                label = ACTION_LABEL.get(actions[frame], "-")
                tcol = (20, 20, 20) if is_cursor else (235, 235, 235)
                gs = cell_font.render(label, True, tcol)
                surface.blit(
                    gs,
                    (cx + (CELL_W - gs.get_width()) // 2, sy + sh - gs.get_height() - 2),
                )

    def _draw_input_sequence_overlay(
        self, input_seq_info: dict
    ) -> Optional[pygame.Surface]:
        """Draw step-on-input sequence overlay.

        Shows the frame-perfect input sequence as human-readable labels
        (LT, RT, JP, JL, JR, NO) in a compact panel. Displays last N
        actions with most recent on the right, plus step count. Panel
        is positioned in the quadrant opposite the player.

        Args:
            input_seq_info: Dictionary with "sequence" (list of action indices),
                "step_count" (int), and "ninja_position" (x, y in game coords).

        Returns:
            pygame.Surface with the overlay.
        """
        ACTION_LABELS = {
            0: "NO",
            1: "LT",
            2: "RT",
            3: "JP",
            4: "JL",
            5: "JR",
        }
        # Key hints for overlay (Z/X = Jump+Left/Right, . = NOOP)
        KEY_HINTS = "←→↑ | Z=JL X=JR .=NO"
        MAX_DISPLAY = 50  # Show last N actions (wrapped, so we can show more)
        MARGIN = 10
        CONTENT_PADDING = 10
        LINE_HEIGHT = 16
        MAX_PANEL_HEIGHT = 150  # Cap height so panel stays on screen

        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        sequence = input_seq_info.get("sequence", [])
        step_count = input_seq_info.get("step_count", len(sequence))

        try:
            title_font = pygame.font.Font(None, 18)
            label_font = pygame.font.Font(None, 16)
        except pygame.error:
            title_font = pygame.font.SysFont("monospace", 14, bold=True)
            label_font = pygame.font.SysFont("monospace", 12)

        panel_width = 280
        content_width = panel_width - 2 * CONTENT_PADDING
        screen_w = self.screen.get_width()
        screen_h = self.screen.get_height()

        # Wrap sequence labels to fit within panel width
        display_seq = (
            sequence[-MAX_DISPLAY:] if len(sequence) > MAX_DISPLAY else sequence
        )
        labels = [ACTION_LABELS.get(a, "?") for a in display_seq]
        label_strs = labels if labels else ["(waiting for input)"]

        wrapped_lines = []
        current_line = []
        current_width = 0
        space_width = label_font.size(" ")[0]
        for i, lbl in enumerate(label_strs):
            lbl_width = label_font.size(lbl)[0]
            need_space = len(current_line) > 0
            add_width = (space_width + lbl_width) if need_space else lbl_width
            if current_width + add_width <= content_width or not current_line:
                current_line.append(lbl)
                current_width += add_width
            else:
                wrapped_lines.append(" ".join(current_line))
                current_line = [lbl]
                current_width = lbl_width
        if current_line:
            wrapped_lines.append(" ".join(current_line))

        # Panel height: title + hints + wrapped lines + padding (capped)
        HINTS_ROW_HEIGHT = 16
        n_lines = len(wrapped_lines)
        header_height = 24 + HINTS_ROW_HEIGHT
        max_lines = (MAX_PANEL_HEIGHT - header_height - CONTENT_PADDING) // LINE_HEIGHT
        if n_lines > max_lines:
            wrapped_lines = wrapped_lines[-max_lines:]
            n_lines = max_lines
        panel_height = header_height + n_lines * LINE_HEIGHT + CONTENT_PADDING

        # Position panel opposite from player quadrant
        ninja_pos = input_seq_info.get("ninja_position")
        if ninja_pos:
            # Convert game coords to screen coords
            px = int(ninja_pos[0] * self.adjust) + self.tile_x_offset
            py = int(ninja_pos[1] * self.adjust) + self.tile_y_offset
            mid_w, mid_h = screen_w / 2, screen_h / 2
            # Player in top-left -> panel bottom-right; top-right -> bottom-left; etc.
            if px < mid_w and py < mid_h:
                panel_x, panel_y = (
                    screen_w - panel_width - MARGIN,
                    screen_h - panel_height - MARGIN,
                )
            elif px >= mid_w and py < mid_h:
                panel_x, panel_y = MARGIN, screen_h - panel_height - MARGIN
            elif px < mid_w and py >= mid_h:
                panel_x, panel_y = screen_w - panel_width - MARGIN, MARGIN
            else:
                panel_x, panel_y = MARGIN, MARGIN
        else:
            panel_x, panel_y = MARGIN, MARGIN

        pygame.draw.rect(
            surface,
            (0, 0, 0, 210),
            (panel_x, panel_y, panel_width, panel_height),
            border_radius=5,
        )
        pygame.draw.rect(
            surface,
            (100, 180, 255, 255),
            (panel_x, panel_y, panel_width, panel_height),
            2,
            border_radius=5,
        )

        title_surf = title_font.render("Step-on-Input", True, (150, 200, 255))
        surface.blit(title_surf, (panel_x + CONTENT_PADDING, panel_y + 6))

        steps_text = f"Steps: {step_count}"
        steps_surf = label_font.render(steps_text, True, (200, 220, 255))
        surface.blit(
            steps_surf,
            (
                panel_x + panel_width - steps_surf.get_width() - CONTENT_PADDING,
                panel_y + 6,
            ),
        )

        hints_surf = label_font.render(KEY_HINTS, True, (160, 180, 200))
        surface.blit(hints_surf, (panel_x + CONTENT_PADDING, panel_y + 24))

        # Draw wrapped sequence lines
        seq_start_y = 24 + HINTS_ROW_HEIGHT
        for i, line_text in enumerate(wrapped_lines):
            seq_surf = label_font.render(line_text, True, (255, 255, 255))
            surface.blit(
                seq_surf,
                (panel_x + CONTENT_PADDING, panel_y + seq_start_y + i * LINE_HEIGHT),
            )

        return surface

    def _draw_pbrs_path_overlay(self, pbrs_info: dict) -> Optional[pygame.Surface]:
        """Draw PBRS shortest-path overlay.

        Visualizes:
        1. Shortest path edges: cyan lines for walk, orange arcs for jumps
        2. Goal marker
        3. Info panel with potential, phase, and graph stats

        Args:
            pbrs_info: Dictionary from _get_pbrs_path_overlay_data().

        Returns:
            pygame.Surface with the overlay.
        """
        import math as _math

        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        path_edges = pbrs_info.get("path_edges", [])
        goal_pos = pbrs_info.get("goal_position", (0, 0))
        phase = pbrs_info.get("phase", "switch")
        potential = pbrs_info.get("potential", 0.0)
        d_max = pbrs_info.get("d_max", 1.0)
        n_surface_points = pbrs_info.get("n_surface_points", 0)
        n_edges = pbrs_info.get("n_edges", 0)

        # Graph node positions are in inner-grid pixel space (excludes 1-tile
        # border padding).  Add TILE_PIXEL_SIZE (24 px) to align with the
        # full-map rendering coordinate system used by the entity/tile renderers.
        border_offset = TILE_PIXEL_SIZE

        # 1. Draw shortest path edges
        for edge in path_edges:
            src_x, src_y = edge["src_point"]
            dst_x, dst_y = edge["dst_point"]
            edge_type = edge["edge_type"]

            sx1 = int(src_x * self.adjust + self.tile_x_offset) + border_offset
            sy1 = int(src_y * self.adjust + self.tile_y_offset) + border_offset
            sx2 = int(dst_x * self.adjust + self.tile_x_offset) + border_offset
            sy2 = int(dst_y * self.adjust + self.tile_y_offset) + border_offset

            if edge_type == "walk":
                # Walk edges: bright cyan, thin
                color = (60, 220, 255, 220)
                width = max(1, int(2 * self.adjust))
                pygame.draw.line(surface, color, (sx1, sy1), (sx2, sy2), width)
            else:
                # Jump edges: bright orange, thicker, with arrowhead
                color = (255, 160, 40, 240)
                width = max(2, int(3 * self.adjust))
                pygame.draw.line(surface, color, (sx1, sy1), (sx2, sy2), width)

                # Arrowhead at destination
                dx = sx2 - sx1
                dy = sy2 - sy1
                length = _math.sqrt(dx * dx + dy * dy)
                if length > 0:
                    ux, uy = dx / length, dy / length
                    arrow_size = max(4, int(6 * self.adjust))
                    # Two points for the arrowhead
                    ax1 = int(sx2 - arrow_size * (ux + uy * 0.5))
                    ay1 = int(sy2 - arrow_size * (uy - ux * 0.5))
                    ax2 = int(sx2 - arrow_size * (ux - uy * 0.5))
                    ay2 = int(sy2 - arrow_size * (uy + ux * 0.5))
                    pygame.draw.polygon(
                        surface, color, [(sx2, sy2), (ax1, ay1), (ax2, ay2)]
                    )

            # Draw small circles at edge endpoints
            pygame.draw.circle(surface, (255, 255, 255, 200), (sx1, sy1), 2)
            pygame.draw.circle(surface, (255, 255, 255, 200), (sx2, sy2), 2)

        # 2. Draw goal marker (goal_pos is in full-map coordinates, no border offset needed)
        if goal_pos:
            gx = int(goal_pos[0] * self.adjust) + self.tile_x_offset
            gy = int(goal_pos[1] * self.adjust) + self.tile_y_offset
            goal_color = (
                (255, 255, 50, 220) if phase == "switch" else (50, 255, 50, 220)
            )
            pygame.draw.circle(
                surface, goal_color, (gx, gy), max(6, int(8 * self.adjust)), 2
            )
            pygame.draw.circle(
                surface, goal_color, (gx, gy), max(3, int(4 * self.adjust))
            )

        # 3. Draw info panel
        try:
            title_font = pygame.font.Font(None, 18)
            info_font = pygame.font.Font(None, 16)
        except pygame.error:
            title_font = pygame.font.SysFont("monospace", 14, bold=True)
            info_font = pygame.font.SysFont("monospace", 12)

        panel_width = 230
        panel_height = 130
        panel_x = self.screen.get_width() - panel_width - 10
        panel_y = self.screen.get_height() - panel_height - 10

        pygame.draw.rect(
            surface,
            (0, 0, 0, 210),
            (panel_x, panel_y, panel_width, panel_height),
            border_radius=5,
        )
        pygame.draw.rect(
            surface,
            (255, 180, 50, 255),
            (panel_x, panel_y, panel_width, panel_height),
            2,
            border_radius=5,
        )

        title_surf = title_font.render("PBRS Path", True, (255, 200, 100))
        surface.blit(title_surf, (panel_x + 10, panel_y + 6))

        walk_count = sum(1 for e in path_edges if e["edge_type"] == "walk")
        jump_count = sum(1 for e in path_edges if e["edge_type"] == "jump")

        lines = [
            (f"Phase: {phase}", (200, 200, 255)),
            (f"Potential: {potential:.4f}", (180, 255, 180)),
            (f"Path: {walk_count} walk + {jump_count} jump", (255, 220, 100)),
            (f"d_max: {d_max:.0f} frames", (200, 200, 200)),
            (f"Surface pts: {n_surface_points}", (200, 200, 200)),
            (f"Graph edges: {n_edges}", (200, 200, 200)),
        ]

        y_off = 26
        for text, color in lines:
            text_surf = info_font.render(text, True, color)
            surface.blit(text_surf, (panel_x + 10, panel_y + y_off))
            y_off += 16

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
