"""
Enhanced debug overlay renderer with comprehensive graph visualization.

This module extends the existing debug overlay system with advanced graph
visualization capabilities, including navigation, trajectory display,
and interactive graph exploration.
"""

import pygame
from typing import Optional, Dict, Any, List, Tuple
from enum import IntEnum

from .visualization import GraphVisualizer, VisualizationConfig
from .navigation import PathfindingEngine, PathfindingAlgorithm
from .common import NodeType, EdgeType
from .hierarchical_builder import HierarchicalGraphBuilder


class OverlayMode(IntEnum):
    """Debug overlay display modes."""

    DISABLED = 0
    BASIC_GRAPH = 1
    PATHFINDING = 2
    FULL_ANALYSIS = 3


class EnhancedDebugOverlay:
    """
    Enhanced debug overlay with comprehensive graph visualization.

    Integrates with the existing debug overlay system while providing
    advanced graph analysis and navigation visualization capabilities.
    """

    def __init__(self, sim, screen, adjust, tile_x_offset, tile_y_offset):
        """Initialize enhanced debug overlay."""
        self.sim = sim
        self.screen = screen
        self.adjust = adjust
        self.tile_x_offset = tile_x_offset
        self.tile_y_offset = tile_y_offset

        # Initialize pygame font if needed
        if not pygame.font.get_init():
            pygame.font.init()

        # Visualization components
        self.config = VisualizationConfig()
        self.visualizer = GraphVisualizer(self.config)
        self.navigation_engine = None  # Will be initialized with level data
        self.graph_builder = HierarchicalGraphBuilder()

        # State
        self.overlay_mode = OverlayMode.DISABLED
        self.current_graph_data = None
        self.current_path_result = None
        self.goal_position = None
        self.show_hierarchical = False

        # Fonts
        self.small_font = pygame.font.Font(None, 16)
        self.medium_font = pygame.font.Font(None, 20)
        self.large_font = pygame.font.Font(None, 24)

        # Colors
        self.info_panel_color = (0, 0, 0, 180)
        self.text_color = (255, 255, 255)
        self.highlight_color = (255, 255, 0, 200)

        # Performance tracking
        self.last_graph_build_time = 0
        self.last_navigation_time = 0

    def update_params(self, adjust, tile_x_offset, tile_y_offset):
        """Update rendering parameters."""
        self.adjust = adjust
        self.tile_x_offset = tile_x_offset
        self.tile_y_offset = tile_y_offset

    def set_overlay_mode(self, mode: OverlayMode):
        """Set the current overlay mode."""
        self.overlay_mode = mode

        # Update visualization config based on mode
        if mode == OverlayMode.BASIC_GRAPH:
            self.config.show_nodes = True
            self.config.show_edges = True
            self.config.show_shortest_path = False
            self.config.alpha = 0.6
        elif mode == OverlayMode.PATHFINDING:
            self.config.show_nodes = True
            self.config.show_edges = True
            self.config.show_shortest_path = True
            self.config.alpha = 0.8
        elif mode == OverlayMode.FULL_ANALYSIS:
            self.config.show_nodes = True
            self.config.show_edges = True
            self.config.show_shortest_path = True
            self.config.show_node_labels = True
            self.config.alpha = 0.9
        else:
            self.config.show_nodes = False
            self.config.show_edges = False
            self.config.show_shortest_path = False

    def set_goal_position(self, position: Tuple[float, float]):
        """Set goal position for navigation."""
        self.goal_position = position
        self._update_navigation()

    def toggle_hierarchical_view(self):
        """Toggle between single-resolution and hierarchical graph view."""
        self.show_hierarchical = not self.show_hierarchical

    def draw_overlay(
        self, debug_info: Optional[Dict[str, Any]] = None
    ) -> pygame.Surface:
        """
        Draw enhanced debug overlay.

        Args:
            debug_info: Debug information from simulator

        Returns:
            Overlay surface to be blitted onto main screen
        """
        if self.overlay_mode == OverlayMode.DISABLED:
            return pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        # Create overlay surface
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        # Build or update graph data
        self._update_graph_data()

        if self.current_graph_data is None:
            return overlay

        # Draw graph visualization based on mode
        if self.overlay_mode == OverlayMode.BASIC_GRAPH:
            self._draw_basic_graph_overlay(overlay)
        elif self.overlay_mode == OverlayMode.PATHFINDING:
            self._draw_navigation_overlay(overlay)
        elif self.overlay_mode == OverlayMode.FULL_ANALYSIS:
            self._draw_full_analysis_overlay(overlay, debug_info)

        # Draw info panels
        self._draw_info_panels(overlay, debug_info)

        return overlay

    def _update_graph_data(self):
        """Update graph data from current simulation state."""
        try:
            # Get current simulation state
            ninja_position = (self.sim.ninja.x, self.sim.ninja.y)
            ninja_velocity = (self.sim.ninja.vx, self.sim.ninja.vy)
            ninja_state = getattr(self.sim.ninja, "movement_state", 0)

            # Get level data
            level_data = self.sim.level_data if hasattr(self.sim, "level_data") else {}
            entities = self._extract_entities_from_sim()

            # Build graph
            if self.show_hierarchical:
                # Build hierarchical graph
                hierarchical_data = self.graph_builder.build_graph(
                    level_data,
                    ninja_position,
                    entities,
                    ninja_velocity,
                    ninja_state,
                    node_feature_dim=16,
                    edge_feature_dim=8,
                )
                self.current_graph_data = hierarchical_data.sub_cell_graph
            else:
                # Build single-resolution graph
                from .graph_construction import GraphConstructor
                from .feature_extraction import FeatureExtractor
                from .edge_building import EdgeBuilder

                feature_extractor = FeatureExtractor()
                edge_builder = EdgeBuilder()
                graph_constructor = GraphConstructor(feature_extractor, edge_builder)

                self.current_graph_data = graph_constructor.build_sub_cell_graph(
                    level_data,
                    ninja_position,
                    ninja_velocity,
                    ninja_state,
                    node_feature_dim=16,
                    edge_feature_dim=8,
                )

            # Update navigation if goal is set
            if self.goal_position:
                self._update_navigation()

        except Exception as e:
            print(f"Error updating graph data: {e}")
            self.current_graph_data = None

    def _update_navigation(self):
        """Update navigation results."""
        if not self.current_graph_data or not self.goal_position:
            self.current_path_result = None
            return

        try:
            # Get ninja position as start
            ninja_position = (self.sim.ninja.x, self.sim.ninja.y)
            ninja_velocity = (self.sim.ninja.vx, self.sim.ninja.vy)
            ninja_state = getattr(self.sim.ninja, "movement_state", 0)

            # Initialize navigation engine with current level data
            level_data = self.sim.level_data if hasattr(self.sim, "level_data") else {}
            entities = self._extract_entities_from_sim()

            if self.navigation_engine is None:
                self.navigation_engine = PathfindingEngine(
                    level_data=level_data, entities=entities
                )

            # Create ninja state for accurate navigation
            ninja_state_dict = {
                "movement_state": ninja_state,
                "velocity": ninja_velocity,
                "position": ninja_position,
                "ground_contact": True,  # Would need to get from sim
                "wall_contact": False,  # Would need to get from sim
            }

            # Find shortest path with accurate physics
            start_time = pygame.time.get_ticks()
            self.current_path_result = self.navigation_engine.find_shortest_path(
                self.current_graph_data,
                self._find_closest_node(ninja_position),
                self._find_closest_node(self.goal_position),
                PathfindingAlgorithm.A_STAR,
                ninja_state=ninja_state_dict,
            )
            self.last_navigation_time = pygame.time.get_ticks() - start_time

        except Exception as e:
            print(f"Error in navigation: {e}")
            self.current_path_result = None

    def _draw_basic_graph_overlay(self, overlay: pygame.Surface):
        """Draw basic graph structure overlay."""
        if not self.current_graph_data:
            return

        # Use visualizer to create overlay
        graph_overlay = self.visualizer.create_overlay_visualization(
            self.current_graph_data,
            overlay,
            self.goal_position,
            ninja_position=(self.sim.ninja.x, self.sim.ninja.y),
        )

        # Copy the overlay content
        overlay.blit(graph_overlay, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)

    def _draw_navigation_overlay(self, overlay: pygame.Surface):
        """Draw navigation visualization overlay."""
        if not self.current_graph_data:
            return

        # Draw basic graph first
        self._draw_basic_graph_overlay(overlay)

        # Draw navigation-specific elements
        if self.current_path_result and self.current_path_result.success:
            self._draw_path_details(overlay)

    def _draw_full_analysis_overlay(
        self, overlay: pygame.Surface, debug_info: Optional[Dict[str, Any]]
    ):
        """Draw comprehensive graph analysis overlay."""
        if not self.current_graph_data:
            return

        # Draw navigation overlay first
        self._draw_navigation_overlay(overlay)

        # Add analysis-specific visualizations
        self._draw_node_analysis(overlay)
        self._draw_edge_analysis(overlay)
        self._draw_traversability_analysis(overlay)

    def _draw_path_details(self, overlay: pygame.Surface):
        """Draw detailed path information."""
        if not self.current_path_result or not self.current_path_result.success:
            return

        # Draw path waypoints with numbers
        for i, (x, y) in enumerate(self.current_path_result.path_coordinates):
            screen_x = int(x * self.adjust + self.tile_x_offset)
            screen_y = int(y * self.adjust + self.tile_y_offset)

            # Draw waypoint circle
            pygame.draw.circle(overlay, (255, 255, 0, 200), (screen_x, screen_y), 8)
            pygame.draw.circle(
                overlay, (255, 255, 255, 255), (screen_x, screen_y), 8, 2
            )

            # Draw waypoint number
            if i % 5 == 0 or i == len(self.current_path_result.path_coordinates) - 1:
                text = self.small_font.render(str(i), True, (255, 255, 255))
                overlay.blit(text, (screen_x + 10, screen_y - 8))

        # Draw movement type indicators along path
        for i, edge_type in enumerate(self.current_path_result.edge_types):
            if i < len(self.current_path_result.path_coordinates) - 1:
                start_pos = self.current_path_result.path_coordinates[i]
                end_pos = self.current_path_result.path_coordinates[i + 1]

                # Calculate midpoint
                mid_x = (start_pos[0] + end_pos[0]) / 2
                mid_y = (start_pos[1] + end_pos[1]) / 2

                screen_x = int(mid_x * self.adjust + self.tile_x_offset)
                screen_y = int(mid_y * self.adjust + self.tile_y_offset)

                # Draw edge type indicator
                color = self._get_edge_type_color(edge_type)
                pygame.draw.circle(overlay, color, (screen_x, screen_y), 4)

    def _draw_node_analysis(self, overlay: pygame.Surface):
        """Draw node-level analysis information."""
        if not self.current_graph_data:
            return

        # Highlight special nodes (entities, high-connectivity nodes, etc.)
        for node_idx in range(self.current_graph_data.num_nodes):
            if self.current_graph_data.node_mask[node_idx] == 0:
                continue

            node_type = NodeType(self.current_graph_data.node_types[node_idx])
            if node_type == NodeType.ENTITY:
                # Highlight entity nodes
                x, y = self._get_node_position(node_idx)
                screen_x = int(x * self.adjust + self.tile_x_offset)
                screen_y = int(y * self.adjust + self.tile_y_offset)

                pygame.draw.circle(
                    overlay, (255, 100, 255, 150), (screen_x, screen_y), 12
                )
                pygame.draw.circle(
                    overlay, (255, 255, 255, 255), (screen_x, screen_y), 12, 2
                )

    def _draw_edge_analysis(self, overlay: pygame.Surface):
        """Draw edge-level analysis information."""
        if not self.current_graph_data:
            return

        # Count edge types for statistics
        edge_type_counts = {}
        for edge_idx in range(self.current_graph_data.num_edges):
            if self.current_graph_data.edge_mask[edge_idx] == 0:
                continue

            edge_type = EdgeType(self.current_graph_data.edge_types[edge_idx])
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

        # This information will be displayed in the info panel

    def _draw_traversability_analysis(self, overlay: pygame.Surface):
        """Draw traversability analysis visualization."""
        # This could show areas that are difficult to traverse,
        # dead ends, optimal paths, etc.
        pass

    def _draw_info_panels(
        self, overlay: pygame.Surface, debug_info: Optional[Dict[str, Any]]
    ):
        """Draw information panels with graph and navigation data."""
        panel_width = 300
        panel_height = 200
        panel_x = overlay.get_width() - panel_width - 10
        panel_y = 10

        # Create semi-transparent panel
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill(self.info_panel_color)
        overlay.blit(panel_surface, (panel_x, panel_y))

        # Draw panel content
        y_offset = 20
        line_height = 16

        # Title
        title = self.medium_font.render("Graph Analysis", True, self.text_color)
        overlay.blit(title, (panel_x + 10, panel_y + 5))

        # Graph statistics
        if self.current_graph_data:
            stats = [
                f"Nodes: {self.current_graph_data.num_nodes}",
                f"Edges: {self.current_graph_data.num_edges}",
                f"Mode: {'Hierarchical' if self.show_hierarchical else 'Single-res'}",
            ]

            for stat in stats:
                text = self.small_font.render(stat, True, self.text_color)
                overlay.blit(text, (panel_x + 10, panel_y + y_offset))
                y_offset += line_height

        # Pathfinding statistics
        if self.current_path_result:
            y_offset += 5
            path_title = self.small_font.render(
                "Pathfinding:", True, self.highlight_color
            )
            overlay.blit(path_title, (panel_x + 10, panel_y + y_offset))
            y_offset += line_height

            path_stats = [
                f"Success: {'Yes' if self.current_path_result.success else 'No'}",
                f"Path Length: {len(self.current_path_result.path)}",
                f"Total Cost: {self.current_path_result.total_cost:.2f}",
                f"Nodes Explored: {self.current_path_result.nodes_explored}",
                f"Time: {self.last_navigation_time}ms",
            ]

            for stat in path_stats:
                text = self.small_font.render(stat, True, self.text_color)
                overlay.blit(text, (panel_x + 10, panel_y + y_offset))
                y_offset += line_height

        # Controls
        y_offset += 10
        controls_title = self.small_font.render("Controls:", True, self.highlight_color)
        overlay.blit(controls_title, (panel_x + 10, panel_y + y_offset))
        y_offset += line_height

        controls = [
            "G: Toggle graph mode",
            "H: Toggle hierarchical",
            "P: Set navigation goal",
            "R: Reset visualization",
        ]

        for control in controls:
            text = self.small_font.render(control, True, self.text_color)
            overlay.blit(text, (panel_x + 10, panel_y + y_offset))
            y_offset += line_height

    def _extract_entities_from_sim(self) -> List[Dict[str, Any]]:
        """Extract entity data from simulator."""
        entities = []

        try:
            # Extract entities from simulator
            if hasattr(self.sim, "entities"):
                for entity in self.sim.entities:
                    entity_data = {
                        "type": getattr(entity, "type", 0),
                        "x": getattr(entity, "x", 0.0),
                        "y": getattr(entity, "y", 0.0),
                        "state": getattr(entity, "state", 0),
                    }
                    entities.append(entity_data)
        except Exception as e:
            print(f"Error extracting entities: {e}")

        return entities

    def _find_closest_node(self, position: Tuple[float, float]) -> Optional[int]:
        """Find the node closest to a given position."""
        if not self.current_graph_data:
            return None

        target_x, target_y = position
        best_node = None
        best_distance = float("inf")

        for node_idx in range(self.current_graph_data.num_nodes):
            if self.current_graph_data.node_mask[node_idx] == 0:
                continue

            node_x, node_y = self._get_node_position(node_idx)
            distance = ((node_x - target_x) ** 2 + (node_y - target_y) ** 2) ** 0.5

            if distance < best_distance:
                best_distance = distance
                best_node = node_idx

        return best_node

    def _get_node_position(self, node_idx: int) -> Tuple[float, float]:
        """Get world position of a node."""
        if not self.current_graph_data or node_idx >= self.current_graph_data.num_nodes:
            return (0.0, 0.0)

        from .common import SUB_GRID_WIDTH, SUB_GRID_HEIGHT, SUB_CELL_SIZE
        from ..constants.physics_constants import (
            TILE_PIXEL_SIZE,
            FULL_MAP_WIDTH_PX,
            FULL_MAP_HEIGHT_PX,
        )

        # Calculate sub-grid nodes count
        sub_grid_nodes_count = SUB_GRID_WIDTH * SUB_GRID_HEIGHT

        if node_idx < sub_grid_nodes_count:
            # Sub-grid node: calculate position from index
            sub_row = node_idx // SUB_GRID_WIDTH
            sub_col = node_idx % SUB_GRID_WIDTH
            # Center in sub-cell, add 1-tile offset for simulator border
            x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5 + TILE_PIXEL_SIZE
            y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5 + TILE_PIXEL_SIZE
            return (float(x), float(y))
        else:
            # Entity node: extract normalized position from features
            node_features = self.current_graph_data.node_features[node_idx]
            # Feature layout: tile_type + 4 + entity_type + state_features
            tile_type_dim = 38  # From hierarchical_builder.py
            entity_type_dim = 30  # From hierarchical_builder.py
            state_offset = tile_type_dim + 4 + entity_type_dim

            if len(node_features) > state_offset + 2:
                norm_x = float(node_features[state_offset + 1])
                norm_y = float(node_features[state_offset + 2])
                # Denormalize from [0,1] to pixel coordinates
                x = norm_x * float(FULL_MAP_WIDTH_PX)  # 1056 pixels
                y = norm_y * float(FULL_MAP_HEIGHT_PX)  # 600 pixels
                return (float(x), float(y))
            else:
                return (0.0, 0.0)

    def _get_edge_type_color(self, edge_type: EdgeType) -> Tuple[int, int, int, int]:
        """Get color for edge type visualization."""
        colors = {
            EdgeType.WALK: (100, 255, 100, 200),
            EdgeType.JUMP: (255, 200, 100, 200),
            EdgeType.FALL: (100, 150, 255, 200),
            EdgeType.WALL_SLIDE: (200, 100, 255, 200),
            EdgeType.ONE_WAY: (200, 200, 200, 200),
            EdgeType.FUNCTIONAL: (255, 255, 100, 200),
        }
        return colors.get(edge_type, (255, 255, 255, 200))

    def handle_key_press(self, key: int) -> bool:
        """
        Handle key press for interactive controls.

        Args:
            key: Pygame key constant

        Returns:
            True if key was handled, False otherwise
        """
        if key == pygame.K_g:
            # Cycle through overlay modes
            current_mode = int(self.overlay_mode)
            next_mode = (current_mode + 1) % len(OverlayMode)
            self.set_overlay_mode(OverlayMode(next_mode))
            return True
        elif key == pygame.K_h:
            # Toggle hierarchical view
            self.toggle_hierarchical_view()
            return True
        elif key == pygame.K_p:
            # Set navigation goal to current mouse position
            mouse_pos = pygame.mouse.get_pos()
            world_pos = self._screen_to_world_coords(mouse_pos)
            self.set_goal_position(world_pos)
            return True
        elif key == pygame.K_r:
            # Reset visualization
            self.goal_position = None
            self.current_path_result = None
            return True

        return False

    def _screen_to_world_coords(
        self, screen_pos: Tuple[int, int]
    ) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        screen_x, screen_y = screen_pos
        world_x = (screen_x - self.tile_x_offset) / self.adjust
        world_y = (screen_y - self.tile_y_offset) / self.adjust
        return (world_x, world_y)
