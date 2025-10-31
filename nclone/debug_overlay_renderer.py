import pygame
import numpy as np
from . import render_utils
from typing import Optional, List, Set, Tuple
from .constants.physics_constants import (
    TILE_PIXEL_SIZE,
    FULL_MAP_WIDTH,
    FULL_MAP_HEIGHT,
)
from .graph.hierarchical_builder import HierarchicalGraphBuilder
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
        self._graph_builder_for_dims = HierarchicalGraphBuilder()

        # Initialize subgoal visualizer
        self.subgoal_visualizer = SubgoalVisualizer()
        self.subgoal_debug_enabled = False
        self.current_subgoals = []
        self.current_subgoal_plan = None
        self.current_reachable_positions = None

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
        if not path_aware_info.get('graph_data'):
            return None
        
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        graph_data = path_aware_info['graph_data']
        adjacency = graph_data.get('adjacency', {})
        ninja_pos = path_aware_info.get('ninja_position', (0, 0))
        entities = path_aware_info.get('entities', [])
        
        show_adjacency = path_aware_info.get('show_adjacency', False)
        show_distances = path_aware_info.get('show_distances', False)
        show_blocked = path_aware_info.get('show_blocked', False)
        show_paths = path_aware_info.get('show_paths', False)
        
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
        
        # Get blocked positions if available
        blocked_positions = set()
        blocked_edges = set()
        if show_blocked and graph_data.get('blocked_positions'):
            blocked_positions = graph_data['blocked_positions']
        if show_blocked and graph_data.get('blocked_edges'):
            blocked_edges = graph_data['blocked_edges']
        
        # Draw adjacency graph edges first (so they appear behind nodes)
        if show_adjacency:
            for pos, neighbors in adjacency.items():
                if not neighbors:
                    continue
                x1, y1 = pos
                screen_x1 = int(x1 * self.adjust + self.tile_x_offset)
                screen_y1 = int(y1 * self.adjust + self.tile_y_offset)
                
                for neighbor_info in neighbors:
                    # neighbor_info is ((x, y), cost)
                    neighbor_pos, cost = neighbor_info
                    x2, y2 = neighbor_pos
                    screen_x2 = int(x2 * self.adjust + self.tile_x_offset)
                    screen_y2 = int(y2 * self.adjust + self.tile_y_offset)
                    
                    # Check if edge is blocked
                    edge_blocked = (pos, neighbor_pos) in blocked_edges or (neighbor_pos, pos) in blocked_edges
                    edge_color = (255, 0, 0, 80) if edge_blocked else EDGE_COLOR
                    
                    # Draw line
                    pygame.draw.line(surface, edge_color, (screen_x1, screen_y1), (screen_x2, screen_y2), 1)
        
        # Draw nodes
        if show_adjacency or show_blocked:
            try:
                font = pygame.font.Font(None, 16)
            except pygame.error:
                font = pygame.font.SysFont("arial", 14)
            
            for pos in adjacency.keys():
                x, y = pos
                screen_x = int(x * self.adjust + self.tile_x_offset)
                screen_y = int(y * self.adjust + self.tile_y_offset)
                
                # Determine node color
                if pos in blocked_positions:
                    node_color = BLOCKED_NODE_COLOR
                elif abs(x - ninja_pos[0]) < 5 and abs(y - ninja_pos[1]) < 5:
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
            
            # Find closest node to ninja position
            ninja_x, ninja_y = ninja_pos
            closest_node = None
            min_dist = float('inf')
            for pos in adjacency.keys():
                x, y = pos
                dist = ((x - ninja_x)**2 + (y - ninja_y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_node = pos
            
            if closest_node and min_dist < 50:  # Only if ninja is close to a node
                # Use simple BFS to calculate distances
                from collections import deque
                distances = {closest_node: 0}
                queue = deque([closest_node])
                
                while queue:
                    current = queue.popleft()
                    current_dist = distances[current]
                    
                    neighbors = adjacency.get(current, [])
                    for neighbor_info in neighbors:
                        neighbor_pos, cost = neighbor_info
                        if neighbor_pos not in distances:
                            distances[neighbor_pos] = current_dist + cost
                            queue.append(neighbor_pos)
                
                # Draw distances on screen
                for pos, dist in distances.items():
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
        for entity in entities:
            entity_type = entity.get("type", "")
            if entity_type in ["switch", "exit_switch"]:
                switch_positions.append((entity.get("x", 0), entity.get("y", 0)))
            elif entity_type in ["exit", "exit_door"]:
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
        
        # Helper function to find shortest path using BFS
        def find_shortest_path(start_node, end_node, adjacency):
            """Find shortest path from start to end node using BFS.
            Returns (path, distance) where path is list of nodes."""
            from collections import deque
            
            if start_node == end_node:
                return [start_node], 0
            
            distances = {start_node: 0}
            parents = {start_node: None}
            queue = deque([start_node])
            
            while queue:
                current = queue.popleft()
                
                if current == end_node:
                    # Reconstruct path
                    path = []
                    node = end_node
                    while node is not None:
                        path.append(node)
                        node = parents.get(node)
                    path.reverse()
                    return path, distances[end_node]
                
                current_dist = distances[current]
                neighbors = adjacency.get(current, [])
                for neighbor_info in neighbors:
                    neighbor_pos, cost = neighbor_info
                    if neighbor_pos not in distances:
                        distances[neighbor_pos] = current_dist + cost
                        parents[neighbor_pos] = current
                        queue.append(neighbor_pos)
            
            return None, float('inf')
        
        # Draw paths to goals if enabled
        if show_paths and ninja_pos and adjacency:
            from collections import deque
            
            # Find closest node to ninja
            ninja_x, ninja_y = ninja_pos
            closest_node = None
            min_dist = float('inf')
            for pos in adjacency.keys():
                x, y = pos
                dist = ((x - ninja_x)**2 + (y - ninja_y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_node = pos
            
            if closest_node and min_dist < 50:
                # Draw path to nearest switch
                for switch_pos in switch_positions:
                    switch_x, switch_y = switch_pos
                    switch_node = None
                    min_switch_dist = float('inf')
                    for pos in adjacency.keys():
                        x, y = pos
                        dist = ((x - switch_x)**2 + (y - switch_y)**2)**0.5
                        if dist < min_switch_dist:
                            min_switch_dist = dist
                            switch_node = pos
                    
                    if switch_node and min_switch_dist < 50:
                        path, _ = find_shortest_path(closest_node, switch_node, adjacency)
                        if path:
                            # Draw path as thick line connecting nodes
                            for i in range(len(path) - 1):
                                x1, y1 = path[i]
                                x2, y2 = path[i + 1]
                                screen_x1 = int(x1 * self.adjust + self.tile_x_offset)
                                screen_y1 = int(y1 * self.adjust + self.tile_y_offset)
                                screen_x2 = int(x2 * self.adjust + self.tile_x_offset)
                                screen_y2 = int(y2 * self.adjust + self.tile_y_offset)
                                pygame.draw.line(surface, SWITCH_PATH_COLOR, (screen_x1, screen_y1), (screen_x2, screen_y2), 3)
                            break  # Only draw path to nearest switch
                
                # Draw path to nearest exit
                for exit_pos in exit_positions:
                    exit_x, exit_y = exit_pos
                    exit_node = None
                    min_exit_dist = float('inf')
                    for pos in adjacency.keys():
                        x, y = pos
                        dist = ((x - exit_x)**2 + (y - exit_y)**2)**0.5
                        if dist < min_exit_dist:
                            min_exit_dist = dist
                            exit_node = pos
                    
                    if exit_node and min_exit_dist < 50:
                        path, _ = find_shortest_path(closest_node, exit_node, adjacency)
                        if path:
                            # Draw path as thick line connecting nodes
                            for i in range(len(path) - 1):
                                x1, y1 = path[i]
                                x2, y2 = path[i + 1]
                                screen_x1 = int(x1 * self.adjust + self.tile_x_offset)
                                screen_y1 = int(y1 * self.adjust + self.tile_y_offset)
                                screen_x2 = int(x2 * self.adjust + self.tile_x_offset)
                                screen_y2 = int(y2 * self.adjust + self.tile_y_offset)
                                pygame.draw.line(surface, EXIT_PATH_COLOR, (screen_x1, screen_y1), (screen_x2, screen_y2), 3)
                            break  # Only draw path to nearest exit
        
        # Calculate and display switch/exit distances
        if show_distances and ninja_pos and adjacency:
            from collections import deque
            
            # Find closest node to ninja
            ninja_x, ninja_y = ninja_pos
            closest_node = None
            min_dist = float('inf')
            for pos in adjacency.keys():
                x, y = pos
                dist = ((x - ninja_x)**2 + (y - ninja_y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_node = pos
            
            if closest_node and min_dist < 50:
                # Calculate distance to nearest switch
                switch_dist = float('inf')
                for switch_pos in switch_positions:
                    # Find closest node to switch
                    switch_x, switch_y = switch_pos
                    switch_node = None
                    min_switch_dist = float('inf')
                    for pos in adjacency.keys():
                        x, y = pos
                        dist = ((x - switch_x)**2 + (y - switch_y)**2)**0.5
                        if dist < min_switch_dist:
                            min_switch_dist = dist
                            switch_node = pos
                    
                    if switch_node and min_switch_dist < 50:
                        # BFS from ninja to switch
                        distances = {closest_node: 0}
                        queue = deque([closest_node])
                        
                        while queue and switch_node not in distances:
                            current = queue.popleft()
                            current_dist = distances[current]
                            
                            neighbors = adjacency.get(current, [])
                            for neighbor_info in neighbors:
                                neighbor_pos, cost = neighbor_info
                                if neighbor_pos not in distances:
                                    distances[neighbor_pos] = current_dist + cost
                                    queue.append(neighbor_pos)
                        
                        if switch_node in distances:
                            switch_dist = min(switch_dist, distances[switch_node])
                
                # Calculate distance to nearest exit
                exit_dist = float('inf')
                for exit_pos in exit_positions:
                    # Find closest node to exit
                    exit_x, exit_y = exit_pos
                    exit_node = None
                    min_exit_dist = float('inf')
                    for pos in adjacency.keys():
                        x, y = pos
                        dist = ((x - exit_x)**2 + (y - exit_y)**2)**0.5
                        if dist < min_exit_dist:
                            min_exit_dist = dist
                            exit_node = pos
                    
                    if exit_node and min_exit_dist < 50:
                        # BFS from ninja to exit
                        distances = {closest_node: 0}
                        queue = deque([closest_node])
                        
                        while queue and exit_node not in distances:
                            current = queue.popleft()
                            current_dist = distances[current]
                            
                            neighbors = adjacency.get(current, [])
                            for neighbor_info in neighbors:
                                neighbor_pos, cost = neighbor_info
                                if neighbor_pos not in distances:
                                    distances[neighbor_pos] = current_dist + cost
                                    queue.append(neighbor_pos)
                        
                        if exit_node in distances:
                            exit_dist = min(exit_dist, distances[exit_node])
                
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
                
                pygame.draw.rect(surface, (0, 0, 0, 200), (box_x, box_y, box_width, box_height), border_radius=5)
                pygame.draw.rect(surface, (100, 200, 255, 255), (box_x, box_y, box_width, box_height), 2, border_radius=5)
                
                # Draw text
                switch_text = f"Switch: {int(switch_dist) if switch_dist != float('inf') else '∞'}"
                exit_text = f"Exit: {int(exit_dist) if exit_dist != float('inf') else '∞'}"
                
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
            pygame.draw.rect(surface, (0, 0, 0, 200), (legend_x, legend_y, legend_width, legend_height), border_radius=5)
            pygame.draw.rect(surface, (100, 200, 255, 255), (legend_x, legend_y, legend_width, legend_height), 2, border_radius=5)
            
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
                if (
                    key == "tile_types"
                ):  # Don't count tile types dict for text height, it's visual
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
                if key == "tile_types":  # Skip rendering tile types data as text
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
    
    def draw_path_distances(self, path_distances: dict, ninja_pos: Tuple[float, float]) -> pygame.Surface:
        """
        Draw path distance overlay showing distances to objectives.
        
        Args:
            path_distances: Dict with 'switch_distance' and 'exit_distance' keys
            ninja_pos: Current ninja position (x, y)
        
        Returns:
            Surface with path distance visualization
        """
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        font = pygame.font.SysFont('monospace', 16, bold=True)
        
        # Draw distances at ninja position
        if path_distances:
            switch_dist = path_distances.get('switch_distance', float('inf'))
            exit_dist = path_distances.get('exit_distance', float('inf'))
            
            # Convert ninja position to screen coordinates
            screen_x = int(ninja_pos[0] * self.adjust) + self.tile_x_offset
            screen_y = int(ninja_pos[1] * self.adjust) + self.tile_y_offset
            
            # Draw background box
            box_width = 200
            box_height = 60
            box_x = screen_x + 20
            box_y = screen_y - 40
            
            # Keep box on screen
            if box_x + box_width > self.screen.get_width():
                box_x = screen_x - box_width - 20
            if box_y < 0:
                box_y = screen_y + 20
            
            pygame.draw.rect(surface, (0, 0, 0, 200), (box_x, box_y, box_width, box_height), border_radius=5)
            pygame.draw.rect(surface, (100, 200, 255, 255), (box_x, box_y, box_width, box_height), 2, border_radius=5)
            
            # Draw text
            switch_text = f"Switch: {switch_dist if switch_dist != float('inf') else '∞'}"
            exit_text = f"Exit: {exit_dist if exit_dist != float('inf') else '∞'}"
            
            switch_surf = font.render(switch_text, True, (100, 255, 100))
            exit_surf = font.render(exit_text, True, (255, 200, 100))
            
            surface.blit(switch_surf, (box_x + 10, box_y + 10))
            surface.blit(exit_surf, (box_x + 10, box_y + 35))
        
        return surface
    
    def draw_adjacency_graph(self, graph_data: dict, ninja_pos: Tuple[float, float]) -> pygame.Surface:
        """
        Draw adjacency graph overlay showing tile connectivity.
        
        Args:
            graph_data: Dict with 'nodes' and 'edges' keys
            ninja_pos: Current ninja position (x, y)
        
        Returns:
            Surface with adjacency graph visualization
        """
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        
        if not graph_data or 'nodes' not in graph_data or 'edges' not in graph_data:
            return surface
        
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        
        # Draw edges first (so nodes are on top)
        for edge in edges:
            pos1 = edge.get('pos1')
            pos2 = edge.get('pos2')
            if pos1 and pos2:
                screen_x1 = int(pos1[0] * self.adjust) + self.tile_x_offset
                screen_y1 = int(pos1[1] * self.adjust) + self.tile_y_offset
                screen_x2 = int(pos2[0] * self.adjust) + self.tile_x_offset
                screen_y2 = int(pos2[1] * self.adjust) + self.tile_y_offset
                
                # Draw edge line
                pygame.draw.line(surface, (150, 150, 255, 100), 
                               (screen_x1, screen_y1), (screen_x2, screen_y2), 2)
        
        # Draw nodes
        for node in nodes:
            pos = node.get('pos')
            node_type = node.get('type', 'normal')
            
            if pos:
                screen_x = int(pos[0] * self.adjust) + self.tile_x_offset
                screen_y = int(pos[1] * self.adjust) + self.tile_y_offset
                
                # Choose color based on node type
                if node_type == 'ninja':
                    color = (60, 220, 255, 255)
                    radius = 8
                elif node_type == 'switch':
                    color = (100, 255, 100, 255)
                    radius = 6
                elif node_type == 'exit':
                    color = (255, 200, 100, 255)
                    radius = 6
                else:
                    color = (200, 200, 255, 180)
                    radius = 4
                
                # Draw node
                pygame.draw.circle(surface, color, (screen_x, screen_y), radius)
                pygame.draw.circle(surface, (255, 255, 255, 255), (screen_x, screen_y), radius, 1)
        
        # Draw legend
        font = pygame.font.SysFont('monospace', 14)
        legend_x = 10
        legend_y = self.screen.get_height() - 120
        
        pygame.draw.rect(surface, (0, 0, 0, 200), (legend_x, legend_y, 180, 110), border_radius=5)
        pygame.draw.rect(surface, (100, 200, 255, 255), (legend_x, legend_y, 180, 110), 2, border_radius=5)
        
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
