"""
Comprehensive graph visualization system for N++ level analysis.

This module provides both standalone graph rendering and simulator overlay
capabilities for visualizing graph structure, pathfinding results, and
traversability analysis.
"""

import pygame
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import IntEnum

from .common import GraphData, NodeType, EdgeType
from .pathfinding import PathfindingEngine, PathResult, PathfindingAlgorithm
from .hierarchical_builder import HierarchicalGraphData
from ..constants import TILE_PIXEL_SIZE, FULL_MAP_WIDTH, FULL_MAP_HEIGHT


class VisualizationMode(IntEnum):
    """Visualization display modes."""
    STANDALONE = 0      # Independent graph visualization
    OVERLAY = 1         # Overlay on simulator
    SIDE_BY_SIDE = 2    # Both standalone and overlay


@dataclass
class VisualizationConfig:
    """Configuration for graph visualization."""
    # Display options
    show_nodes: bool = True
    show_edges: bool = True
    show_node_labels: bool = False
    show_edge_labels: bool = False
    show_grid: bool = True
    
    # Path visualization
    show_shortest_path: bool = True
    highlight_path_nodes: bool = True
    highlight_path_edges: bool = True
    
    # Node filtering
    show_grid_nodes: bool = True
    show_entity_nodes: bool = True
    show_ninja_node: bool = True
    
    # Edge filtering
    show_walk_edges: bool = True
    show_jump_edges: bool = True
    show_fall_edges: bool = True
    show_wall_slide_edges: bool = True
    show_one_way_edges: bool = True
    show_functional_edges: bool = True
    
    # Visual settings
    node_size: float = 3.0
    edge_width: float = 1.0
    path_width: float = 3.0
    alpha: float = 0.8
    
    # Colors
    background_color: Tuple[int, int, int] = (20, 20, 30)
    grid_color: Tuple[int, int, int] = (60, 60, 70)
    text_color: Tuple[int, int, int] = (255, 255, 255)


class GraphVisualizer:
    """
    Comprehensive graph visualization system.
    
    Provides both standalone rendering and simulator overlay capabilities
    with full control over what graph components are displayed.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize graph visualizer."""
        self.config = config or VisualizationConfig()
        self.pathfinding_engine = PathfindingEngine()
        
        # Initialize pygame if not already done
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        
        # Fonts for text rendering
        self.small_font = pygame.font.Font(None, 16)
        self.medium_font = pygame.font.Font(None, 20)
        self.large_font = pygame.font.Font(None, 24)
        
        # Color schemes
        self.node_colors = {
            NodeType.GRID_CELL: (120, 120, 140, 180),
            NodeType.ENTITY: (255, 100, 100, 220),
            NodeType.NINJA: (100, 255, 255, 255),
        }
        
        self.edge_colors = {
            EdgeType.WALK: (100, 255, 100, 150),
            EdgeType.JUMP: (255, 200, 100, 180),
            EdgeType.FALL: (100, 150, 255, 120),
            EdgeType.WALL_SLIDE: (200, 100, 255, 160),
            EdgeType.ONE_WAY: (200, 200, 200, 140),
            EdgeType.FUNCTIONAL: (255, 255, 100, 200),
        }
        
        self.path_color = (255, 50, 50, 255)  # Bright red for shortest path
        self.path_node_color = (255, 100, 100, 255)
        
        # Cached surfaces for performance
        self._cached_surfaces = {}
        self._cache_valid = False
    
    def create_standalone_visualization(
        self,
        graph_data: GraphData,
        width: int = 1200,
        height: int = 800,
        goal_position: Optional[Tuple[float, float]] = None,
        start_position: Optional[Tuple[float, float]] = None
    ) -> pygame.Surface:
        """
        Create standalone graph visualization.
        
        Args:
            graph_data: Graph data to visualize
            width: Surface width in pixels
            height: Surface height in pixels
            goal_position: Goal position for pathfinding (x, y)
            start_position: Start position for pathfinding (x, y)
            
        Returns:
            Pygame surface with graph visualization
        """
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill(self.config.background_color)
        
        # Calculate scaling to fit graph in surface
        scale_x, scale_y, offset_x, offset_y = self._calculate_scaling(
            graph_data, width, height
        )
        
        # Find shortest path if start and goal are provided
        path_result = None
        if start_position and goal_position:
            start_node = self._find_closest_node(graph_data, start_position)
            goal_node = self._find_closest_node(graph_data, goal_position)
            
            if start_node is not None and goal_node is not None:
                path_result = self.pathfinding_engine.find_shortest_path(
                    graph_data, start_node, goal_node, PathfindingAlgorithm.A_STAR
                )
        
        # Draw grid background if enabled
        if self.config.show_grid:
            self._draw_grid_background(surface, width, height, scale_x, scale_y, offset_x, offset_y)
        
        # Draw edges first (so they appear behind nodes)
        if self.config.show_edges:
            self._draw_edges(surface, graph_data, scale_x, scale_y, offset_x, offset_y, path_result)
        
        # Draw nodes
        if self.config.show_nodes:
            self._draw_nodes(surface, graph_data, scale_x, scale_y, offset_x, offset_y, path_result)
        
        # Draw path information
        if path_result and path_result.success:
            self._draw_path_info(surface, path_result, width, height)
        
        # Draw legend
        self._draw_legend(surface, width, height)
        
        return surface
    
    def create_overlay_visualization(
        self,
        graph_data: GraphData,
        simulator_surface: pygame.Surface,
        goal_position: Optional[Tuple[float, float]] = None,
        start_position: Optional[Tuple[float, float]] = None,
        ninja_position: Optional[Tuple[float, float]] = None
    ) -> pygame.Surface:
        """
        Create overlay visualization for simulator.
        
        Args:
            graph_data: Graph data to visualize
            simulator_surface: Base simulator surface
            goal_position: Goal position for pathfinding (x, y)
            start_position: Start position for pathfinding (x, y)
            ninja_position: Current ninja position (x, y)
            
        Returns:
            Simulator surface with graph overlay
        """
        # Create overlay surface
        overlay = pygame.Surface(simulator_surface.get_size(), pygame.SRCALPHA)
        
        # Use simulator dimensions and scaling
        width, height = simulator_surface.get_size()
        
        # Calculate scaling based on simulator coordinate system
        scale_x = width / (FULL_MAP_WIDTH * TILE_PIXEL_SIZE)
        scale_y = height / (FULL_MAP_HEIGHT * TILE_PIXEL_SIZE)
        offset_x = 0
        offset_y = 0
        
        # Find shortest path if positions are provided
        path_result = None
        if ninja_position and goal_position:
            start_node = self._find_closest_node(graph_data, ninja_position)
            goal_node = self._find_closest_node(graph_data, goal_position)
            
            if start_node is not None and goal_node is not None:
                path_result = self.pathfinding_engine.find_shortest_path(
                    graph_data, start_node, goal_node, PathfindingAlgorithm.A_STAR
                )
        
        # Draw only the shortest path and key nodes for overlay (less visual noise)
        if path_result and path_result.success and self.config.show_shortest_path:
            self._draw_path_overlay(overlay, graph_data, path_result, scale_x, scale_y, offset_x, offset_y)
        
        # Draw goal and start markers
        if goal_position:
            self._draw_position_marker(overlay, goal_position, scale_x, scale_y, offset_x, offset_y, 
                                     (255, 255, 0, 255), "GOAL")
        if ninja_position:
            self._draw_position_marker(overlay, ninja_position, scale_x, scale_y, offset_x, offset_y,
                                     (0, 255, 255, 255), "NINJA")
        
        # Composite overlay onto simulator surface
        result_surface = simulator_surface.copy()
        result_surface.blit(overlay, (0, 0))
        
        return result_surface
    
    def create_side_by_side_visualization(
        self,
        graph_data: GraphData,
        simulator_surface: pygame.Surface,
        goal_position: Optional[Tuple[float, float]] = None,
        start_position: Optional[Tuple[float, float]] = None,
        ninja_position: Optional[Tuple[float, float]] = None
    ) -> pygame.Surface:
        """
        Create side-by-side visualization showing both simulator and standalone graph.
        
        Args:
            graph_data: Graph data to visualize
            simulator_surface: Base simulator surface
            goal_position: Goal position for pathfinding (x, y)
            start_position: Start position for pathfinding (x, y)
            ninja_position: Current ninja position (x, y)
            
        Returns:
            Combined surface with simulator and graph side by side
        """
        sim_width, sim_height = simulator_surface.get_size()
        
        # Create combined surface (double width)
        combined_surface = pygame.Surface((sim_width * 2, sim_height))
        combined_surface.fill(self.config.background_color)
        
        # Draw simulator on left side
        overlay_surface = self.create_overlay_visualization(
            graph_data, simulator_surface, goal_position, start_position, ninja_position
        )
        combined_surface.blit(overlay_surface, (0, 0))
        
        # Draw standalone graph on right side
        standalone_surface = self.create_standalone_visualization(
            graph_data, sim_width, sim_height, goal_position, start_position or ninja_position
        )
        combined_surface.blit(standalone_surface, (sim_width, 0))
        
        # Draw separator line
        pygame.draw.line(combined_surface, (100, 100, 100), 
                        (sim_width, 0), (sim_width, sim_height), 2)
        
        # Add labels
        label_sim = self.medium_font.render("Simulator with Path Overlay", True, self.config.text_color)
        label_graph = self.medium_font.render("Standalone Graph Visualization", True, self.config.text_color)
        
        combined_surface.blit(label_sim, (10, 10))
        combined_surface.blit(label_graph, (sim_width + 10, 10))
        
        return combined_surface
    
    def _calculate_scaling(
        self,
        graph_data: GraphData,
        width: int,
        height: int,
        margin: float = 0.1
    ) -> Tuple[float, float, float, float]:
        """Calculate scaling and offset to fit graph in surface."""
        # Find bounds of all nodes
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        for node_idx in range(graph_data.num_nodes):
            if graph_data.node_mask[node_idx] == 0:
                continue
            
            x, y = self._get_node_position(graph_data, node_idx)
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        
        # Handle empty graph
        if min_x == float('inf'):
            return 1.0, 1.0, 0.0, 0.0
        
        # Calculate scaling with margin
        graph_width = max_x - min_x
        graph_height = max_y - min_y
        
        if graph_width == 0:
            graph_width = 1.0
        if graph_height == 0:
            graph_height = 1.0
        
        available_width = width * (1.0 - 2 * margin)
        available_height = height * (1.0 - 2 * margin)
        
        scale_x = available_width / graph_width
        scale_y = available_height / graph_height
        
        # Use uniform scaling to maintain aspect ratio
        scale = min(scale_x, scale_y)
        
        # Calculate offsets to center the graph
        offset_x = (width - graph_width * scale) / 2 - min_x * scale
        offset_y = (height - graph_height * scale) / 2 - min_y * scale
        
        return scale, scale, offset_x, offset_y
    
    def _draw_grid_background(
        self,
        surface: pygame.Surface,
        width: int,
        height: int,
        scale_x: float,
        scale_y: float,
        offset_x: float,
        offset_y: float
    ):
        """Draw grid background."""
        grid_spacing = TILE_PIXEL_SIZE * scale_x
        
        # Draw vertical lines
        x = offset_x % grid_spacing
        while x < width:
            pygame.draw.line(surface, self.config.grid_color, (int(x), 0), (int(x), height), 1)
            x += grid_spacing
        
        # Draw horizontal lines
        y = offset_y % grid_spacing
        while y < height:
            pygame.draw.line(surface, self.config.grid_color, (0, int(y)), (width, int(y)), 1)
            y += grid_spacing
    
    def _draw_nodes(
        self,
        surface: pygame.Surface,
        graph_data: GraphData,
        scale_x: float,
        scale_y: float,
        offset_x: float,
        offset_y: float,
        path_result: Optional[PathResult] = None
    ):
        """Draw graph nodes."""
        path_nodes = set(path_result.path) if path_result and path_result.success else set()
        
        for node_idx in range(graph_data.num_nodes):
            if graph_data.node_mask[node_idx] == 0:
                continue
            
            node_type = NodeType(graph_data.node_types[node_idx])
            
            # Check if this node type should be shown
            if not self._should_show_node_type(node_type):
                continue
            
            x, y = self._get_node_position(graph_data, node_idx)
            screen_x = int(x * scale_x + offset_x)
            screen_y = int(y * scale_y + offset_y)
            
            # Choose color based on whether node is in path
            if node_idx in path_nodes and self.config.highlight_path_nodes:
                color = self.path_node_color
                size = int(self.config.node_size * 1.5)
            else:
                color = self.node_colors.get(node_type, (255, 255, 255, 255))
                size = int(self.config.node_size)
            
            # Draw node
            if len(color) == 4:  # RGBA
                node_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(node_surface, color, (size, size), size)
                surface.blit(node_surface, (screen_x - size, screen_y - size))
            else:  # RGB
                pygame.draw.circle(surface, color, (screen_x, screen_y), size)
            
            # Draw node label if enabled
            if self.config.show_node_labels:
                label = f"{node_idx}"
                text_surface = self.small_font.render(label, True, self.config.text_color)
                surface.blit(text_surface, (screen_x + size + 2, screen_y - size))
    
    def _draw_edges(
        self,
        surface: pygame.Surface,
        graph_data: GraphData,
        scale_x: float,
        scale_y: float,
        offset_x: float,
        offset_y: float,
        path_result: Optional[PathResult] = None
    ):
        """Draw graph edges."""
        path_edges = set()
        if path_result and path_result.success:
            for i in range(len(path_result.path) - 1):
                path_edges.add((path_result.path[i], path_result.path[i + 1]))
        
        for edge_idx in range(graph_data.num_edges):
            if graph_data.edge_mask[edge_idx] == 0:
                continue
            
            src_node = graph_data.edge_index[0, edge_idx]
            dst_node = graph_data.edge_index[1, edge_idx]
            edge_type = EdgeType(graph_data.edge_types[edge_idx])
            
            # Check if this edge type should be shown
            if not self._should_show_edge_type(edge_type):
                continue
            
            # Get node positions
            src_x, src_y = self._get_node_position(graph_data, src_node)
            dst_x, dst_y = self._get_node_position(graph_data, dst_node)
            
            src_screen_x = int(src_x * scale_x + offset_x)
            src_screen_y = int(src_y * scale_y + offset_y)
            dst_screen_x = int(dst_x * scale_x + offset_x)
            dst_screen_y = int(dst_y * scale_y + offset_y)
            
            # Choose color and width based on whether edge is in path
            if (src_node, dst_node) in path_edges and self.config.highlight_path_edges:
                color = self.path_color
                width = int(self.config.path_width)
            else:
                color = self.edge_colors.get(edge_type, (255, 255, 255, 255))
                width = int(self.config.edge_width)
            
            # Draw edge
            if len(color) == 4:  # RGBA
                self._draw_line_with_alpha(surface, color, 
                                         (src_screen_x, src_screen_y), 
                                         (dst_screen_x, dst_screen_y), width)
            else:  # RGB
                pygame.draw.line(surface, color, 
                               (src_screen_x, src_screen_y), 
                               (dst_screen_x, dst_screen_y), width)
    
    def _draw_path_overlay(
        self,
        surface: pygame.Surface,
        graph_data: GraphData,
        path_result: PathResult,
        scale_x: float,
        scale_y: float,
        offset_x: float,
        offset_y: float
    ):
        """Draw path overlay for simulator."""
        if not path_result.success or len(path_result.path) < 2:
            return
        
        # Draw path as thick line
        path_points = []
        for node_idx in path_result.path:
            x, y = self._get_node_position(graph_data, node_idx)
            screen_x = int(x * scale_x + offset_x)
            screen_y = int(y * scale_y + offset_y)
            path_points.append((screen_x, screen_y))
        
        if len(path_points) >= 2:
            pygame.draw.lines(surface, self.path_color, False, path_points, 
                            int(self.config.path_width))
        
        # Draw start and end markers
        if path_points:
            # Start marker (green circle)
            pygame.draw.circle(surface, (0, 255, 0, 255), path_points[0], 8)
            pygame.draw.circle(surface, (255, 255, 255, 255), path_points[0], 8, 2)
            
            # End marker (red circle)
            pygame.draw.circle(surface, (255, 0, 0, 255), path_points[-1], 8)
            pygame.draw.circle(surface, (255, 255, 255, 255), path_points[-1], 8, 2)
    
    def _draw_position_marker(
        self,
        surface: pygame.Surface,
        position: Tuple[float, float],
        scale_x: float,
        scale_y: float,
        offset_x: float,
        offset_y: float,
        color: Tuple[int, int, int, int],
        label: str
    ):
        """Draw position marker with label."""
        x, y = position
        screen_x = int(x * scale_x + offset_x)
        screen_y = int(y * scale_y + offset_y)
        
        # Draw marker
        pygame.draw.circle(surface, color, (screen_x, screen_y), 10)
        pygame.draw.circle(surface, (255, 255, 255, 255), (screen_x, screen_y), 10, 2)
        
        # Draw label
        text_surface = self.small_font.render(label, True, self.config.text_color)
        surface.blit(text_surface, (screen_x + 12, screen_y - 8))
    
    def _draw_path_info(self, surface: pygame.Surface, path_result: PathResult, width: int, height: int):
        """Draw path information panel."""
        info_lines = [
            f"Path Length: {len(path_result.path)} nodes",
            f"Total Cost: {path_result.total_cost:.2f}",
            f"Nodes Explored: {path_result.nodes_explored}",
            f"Success: {'Yes' if path_result.success else 'No'}"
        ]
        
        # Draw background panel
        panel_height = len(info_lines) * 20 + 20
        panel_rect = pygame.Rect(width - 250, 10, 240, panel_height)
        panel_surface = pygame.Surface((240, panel_height), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 180))
        surface.blit(panel_surface, panel_rect.topleft)
        
        # Draw info text
        for i, line in enumerate(info_lines):
            text_surface = self.small_font.render(line, True, self.config.text_color)
            surface.blit(text_surface, (width - 240, 20 + i * 20))
    
    def _draw_legend(self, surface: pygame.Surface, width: int, height: int):
        """Draw legend showing node and edge types."""
        legend_items = []
        
        # Node types
        if self.config.show_nodes:
            legend_items.append(("Nodes:", None))
            if self.config.show_grid_nodes:
                legend_items.append(("  Grid Cell", self.node_colors[NodeType.GRID_CELL]))
            if self.config.show_entity_nodes:
                legend_items.append(("  Entity", self.node_colors[NodeType.ENTITY]))
            if self.config.show_ninja_node:
                legend_items.append(("  Ninja", self.node_colors[NodeType.NINJA]))
        
        # Edge types
        if self.config.show_edges:
            legend_items.append(("", None))  # Spacer
            legend_items.append(("Edges:", None))
            if self.config.show_walk_edges:
                legend_items.append(("  Walk", self.edge_colors[EdgeType.WALK]))
            if self.config.show_jump_edges:
                legend_items.append(("  Jump", self.edge_colors[EdgeType.JUMP]))
            if self.config.show_fall_edges:
                legend_items.append(("  Fall", self.edge_colors[EdgeType.FALL]))
            if self.config.show_wall_slide_edges:
                legend_items.append(("  Wall Slide", self.edge_colors[EdgeType.WALL_SLIDE]))
            if self.config.show_one_way_edges:
                legend_items.append(("  One Way", self.edge_colors[EdgeType.ONE_WAY]))
            if self.config.show_functional_edges:
                legend_items.append(("  Functional", self.edge_colors[EdgeType.FUNCTIONAL]))
        
        if not legend_items:
            return
        
        # Draw legend background
        legend_height = len(legend_items) * 16 + 20
        legend_rect = pygame.Rect(10, height - legend_height - 10, 200, legend_height)
        legend_surface = pygame.Surface((200, legend_height), pygame.SRCALPHA)
        legend_surface.fill((0, 0, 0, 180))
        surface.blit(legend_surface, legend_rect.topleft)
        
        # Draw legend items
        for i, (label, color) in enumerate(legend_items):
            y_pos = height - legend_height + 10 + i * 16
            
            if color is not None:
                # Draw color indicator
                if len(color) == 4:  # RGBA
                    color_surface = pygame.Surface((12, 12), pygame.SRCALPHA)
                    color_surface.fill(color)
                    surface.blit(color_surface, (20, y_pos + 2))
                else:  # RGB
                    pygame.draw.rect(surface, color, (20, y_pos + 2, 12, 12))
                
                # Draw label
                text_surface = self.small_font.render(label, True, self.config.text_color)
                surface.blit(text_surface, (36, y_pos))
            else:
                # Header or spacer
                if label:
                    text_surface = self.small_font.render(label, True, self.config.text_color)
                    surface.blit(text_surface, (20, y_pos))
    
    def _draw_line_with_alpha(
        self,
        surface: pygame.Surface,
        color: Tuple[int, int, int, int],
        start_pos: Tuple[int, int],
        end_pos: Tuple[int, int],
        width: int
    ):
        """Draw line with alpha transparency."""
        # Create temporary surface for alpha blending
        line_surface = pygame.Surface((abs(end_pos[0] - start_pos[0]) + width * 2,
                                     abs(end_pos[1] - start_pos[1]) + width * 2), pygame.SRCALPHA)
        
        # Calculate relative positions
        rel_start = (width, width)
        rel_end = (end_pos[0] - start_pos[0] + width, end_pos[1] - start_pos[1] + width)
        
        pygame.draw.line(line_surface, color, rel_start, rel_end, width)
        
        # Blit to main surface
        surface.blit(line_surface, (min(start_pos[0], end_pos[0]) - width,
                                   min(start_pos[1], end_pos[1]) - width))
    
    def _get_node_position(self, graph_data: GraphData, node_idx: int) -> Tuple[float, float]:
        """Extract world position from node features."""
        if node_idx >= graph_data.num_nodes or graph_data.node_mask[node_idx] == 0:
            return (0.0, 0.0)
        
        # Assume first two features are x, y coordinates
        node_features = graph_data.node_features[node_idx]
        x = node_features[0] if len(node_features) > 0 else 0.0
        y = node_features[1] if len(node_features) > 1 else 0.0
        
        return (float(x), float(y))
    
    def _find_closest_node(self, graph_data: GraphData, position: Tuple[float, float]) -> Optional[int]:
        """Find the node closest to a given world position."""
        target_x, target_y = position
        best_node = None
        best_distance = float('inf')
        
        for node_idx in range(graph_data.num_nodes):
            if graph_data.node_mask[node_idx] == 0:
                continue
            
            node_x, node_y = self._get_node_position(graph_data, node_idx)
            distance = math.sqrt((node_x - target_x) ** 2 + (node_y - target_y) ** 2)
            
            if distance < best_distance:
                best_distance = distance
                best_node = node_idx
        
        return best_node
    
    def _should_show_node_type(self, node_type: NodeType) -> bool:
        """Check if a node type should be displayed."""
        if node_type == NodeType.GRID_CELL:
            return self.config.show_grid_nodes
        elif node_type == NodeType.ENTITY:
            return self.config.show_entity_nodes
        elif node_type == NodeType.NINJA:
            return self.config.show_ninja_node
        return True
    
    def _should_show_edge_type(self, edge_type: EdgeType) -> bool:
        """Check if an edge type should be displayed."""
        if edge_type == EdgeType.WALK:
            return self.config.show_walk_edges
        elif edge_type == EdgeType.JUMP:
            return self.config.show_jump_edges
        elif edge_type == EdgeType.FALL:
            return self.config.show_fall_edges
        elif edge_type == EdgeType.WALL_SLIDE:
            return self.config.show_wall_slide_edges
        elif edge_type == EdgeType.ONE_WAY:
            return self.config.show_one_way_edges
        elif edge_type == EdgeType.FUNCTIONAL:
            return self.config.show_functional_edges
        return True


class InteractiveGraphVisualizer:
    """
    Interactive graph visualizer with real-time controls.
    
    Provides a GUI for adjusting visualization parameters and
    exploring graph structure interactively.
    """
    
    def __init__(self, width: int = 1400, height: int = 900):
        """Initialize interactive visualizer."""
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("N++ Graph Visualizer")
        
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()
        
        # Control panel dimensions
        self.control_panel_width = 300
        self.graph_area_width = width - self.control_panel_width
        
        # Visualization components
        self.config = VisualizationConfig()
        self.visualizer = GraphVisualizer(self.config)
        
        # State
        self.running = True
        self.graph_data = None
        self.goal_position = None
        self.start_position = None
        
        # UI elements (simplified for this implementation)
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)
    
    def run(self, graph_data: GraphData):
        """Run interactive visualization."""
        self.graph_data = graph_data
        
        while self.running:
            self._handle_events()
            self._update()
            self._draw()
            self.clock.tick(60)
        
        pygame.quit()
    
    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_click(event.pos, event.button)
    
    def _handle_mouse_click(self, pos: Tuple[int, int], button: int):
        """Handle mouse clicks for setting start/goal positions."""
        x, y = pos
        
        # Only handle clicks in graph area
        if x >= self.control_panel_width:
            # Convert screen coordinates to world coordinates
            # This is a simplified conversion - would need proper scaling
            world_x = (x - self.control_panel_width) * 2  # Rough conversion
            world_y = y * 2
            
            if button == 1:  # Left click - set start
                self.start_position = (world_x, world_y)
            elif button == 3:  # Right click - set goal
                self.goal_position = (world_x, world_y)
    
    def _update(self):
        """Update visualization state."""
        pass  # Add any dynamic updates here
    
    def _draw(self):
        """Draw the complete interface."""
        self.screen.fill((30, 30, 40))
        
        # Draw control panel
        self._draw_control_panel()
        
        # Draw graph visualization
        if self.graph_data is not None:
            graph_surface = self.visualizer.create_standalone_visualization(
                self.graph_data,
                self.graph_area_width,
                self.height,
                self.goal_position,
                self.start_position
            )
            self.screen.blit(graph_surface, (self.control_panel_width, 0))
        
        pygame.display.flip()
    
    def _draw_control_panel(self):
        """Draw control panel with options."""
        panel_rect = pygame.Rect(0, 0, self.control_panel_width, self.height)
        pygame.draw.rect(self.screen, (40, 40, 50), panel_rect)
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (self.control_panel_width, 0), 
                        (self.control_panel_width, self.height), 2)
        
        # Title
        title = self.font.render("Graph Controls", True, (255, 255, 255))
        self.screen.blit(title, (10, 10))
        
        # Instructions
        instructions = [
            "Left Click: Set Start",
            "Right Click: Set Goal",
            "ESC: Exit",
            "",
            "Current Settings:",
            f"Show Nodes: {self.config.show_nodes}",
            f"Show Edges: {self.config.show_edges}",
            f"Show Path: {self.config.show_shortest_path}",
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, (200, 200, 200))
            self.screen.blit(text, (10, 50 + i * 20))