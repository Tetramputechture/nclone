"""
Enhanced pathfinding visualization system for N++ level analysis.

This module provides comprehensive pathfinding visualization capabilities including:
- Path highlighting with different colors for different path types
- Entity targeting (switches, doors, exits)
- Interactive pathfinding controls
- Debug information overlays
- Standalone and integrated visualization modes
"""

import pygame
import math
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import IntEnum

from .common import GraphData, NodeType, EdgeType
from .pathfinding import PathfindingEngine, PathResult, PathfindingAlgorithm
from .visualization import GraphVisualizer, VisualizationConfig
from .constants import ColorScheme
from ..constants.entity_types import EntityType

logger = logging.getLogger(__name__)


class PathfindingVisualizationMode(IntEnum):
    """Pathfinding visualization modes."""
    SIMPLE_PATH = 0      # Single path from A to B
    ENTITY_TARGETING = 1 # Path to specific entity types
    MULTI_PATH = 2       # Multiple paths with different colors
    HIERARCHICAL = 3     # Hierarchical pathfinding visualization


@dataclass
class PathfindingTarget:
    """Target for pathfinding visualization."""
    entity_type: EntityType
    position: Tuple[float, float]
    node_idx: int
    label: str
    color: Tuple[int, int, int, int]


@dataclass
class PathVisualizationResult:
    """Result of pathfinding visualization."""
    path_result: PathResult
    target: PathfindingTarget
    start_position: Tuple[float, float]
    start_node: int
    success: bool
    error_message: Optional[str] = None


class PathfindingVisualizer:
    """
    Enhanced pathfinding visualization system.
    
    Provides comprehensive pathfinding visualization with entity targeting,
    multiple path types, and interactive debugging capabilities.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize pathfinding visualizer."""
        self.config = config or VisualizationConfig()
        self.base_visualizer = GraphVisualizer(config)
        self.pathfinding_engine = PathfindingEngine()
        
        # Path visualization settings
        self.path_colors = {
            'default': (0, 255, 255, 255),      # Cyan
            'exit_switch': (0, 255, 0, 255),           # Green
            'exit_door': (255, 0, 255, 255),           # Magenta
            'locked_door': (255, 255, 0, 255),        # Yellow
            'trap_door': (255, 165, 0, 255),          # Orange  
            'regular_door': (0, 255, 255, 255),       # Cyan
            'gold': (255, 215, 0, 255),               # Gold
            'ninja': (255, 255, 255, 255),            # White
        }
        
        # Entity type mappings for targeting
        self.entity_type_labels = {
            EntityType.EXIT_SWITCH: "Exit Switch",
            EntityType.EXIT_DOOR: "Exit Door",
            EntityType.LOCKED_DOOR: "Locked Door",
            EntityType.TRAP_DOOR: "Trap Door",
            EntityType.REGULAR_DOOR: "Regular Door",
            EntityType.GOLD: "Gold",
            EntityType.NINJA: "Ninja",
        }
        
        # Current visualization state
        self.current_paths: List[PathVisualizationResult] = []
        self.show_path_info = True
        self.show_entity_labels = True
        self.show_distance_info = True
        
    def find_entities_by_type(self, graph_data: GraphData, entity_type: int) -> List[PathfindingTarget]:
        """Find all entities of a specific type in the graph."""
        targets = []
        
        for node_idx in range(graph_data.num_nodes):
            if graph_data.node_mask[node_idx] == 0:
                continue
                
            node_type = NodeType(graph_data.node_types[node_idx])
            if node_type != NodeType.ENTITY:
                continue
                
            # Check entity features to determine type
            node_features = graph_data.node_features[node_idx]
            
            # Calculate the correct offset for entity type one-hot encoding
            tile_type_dim = 38
            entity_type_offset = 2 + tile_type_dim + 4  # position(2) + tile_type(38) + solidity(4)
            
            # Check entity type one-hot encoding
            node_entity_type = None
            for et in range(30):  # entity_type_dim = 30
                if entity_type_offset + et < len(node_features):
                    if node_features[entity_type_offset + et] > 0.5:  # One-hot encoded
                        node_entity_type = et
                        break
            
            if node_entity_type is not None and node_entity_type == entity_type:
                    position = self._get_node_position(graph_data, node_idx)
                    label = self.entity_type_labels.get(entity_type, f"Entity {entity_type}")
                    color = self.path_colors.get('default', self.path_colors['default'])
                    
                    target = PathfindingTarget(
                        entity_type=entity_type,
                        position=position,
                        node_idx=node_idx,
                        label=label,
                        color=color
                    )
                    targets.append(target)
                    
        return targets
    
    def find_path_to_entity_type(
        self, 
        graph_data: GraphData, 
        start_position: Tuple[float, float],
        entity_type: int,
        algorithm: PathfindingAlgorithm = PathfindingAlgorithm.A_STAR
    ) -> Optional[PathVisualizationResult]:
        """Find path from start position to nearest entity of specified type."""
        
        # Find start node
        start_node = self._find_closest_node(graph_data, start_position)
        if start_node is None:
            return PathVisualizationResult(
                path_result=PathResult([], float('inf'), False, 0, [], []),
                target=None,
                start_position=start_position,
                start_node=-1,
                success=False,
                error_message="Could not find start node"
            )
        
        # Find all entities of the target type
        targets = self.find_entities_by_type(graph_data, entity_type)
        if not targets:
            return PathVisualizationResult(
                path_result=PathResult([], float('inf'), False, 0, [], []),
                target=None,
                start_position=start_position,
                start_node=start_node,
                success=False,
                error_message=f"No entities of type {entity_type.name} found"
            )
        
        # Find path to closest target
        best_result = None
        best_distance = float('inf')
        
        for target in targets:
            path_result = self.pathfinding_engine.find_shortest_path(
                graph_data, start_node, target.node_idx, algorithm
            )
            
            if path_result.success and path_result.total_cost < best_distance:
                best_distance = path_result.total_cost
                best_result = PathVisualizationResult(
                    path_result=path_result,
                    target=target,
                    start_position=start_position,
                    start_node=start_node,
                    success=True
                )
        
        if best_result is None:
            return PathVisualizationResult(
                path_result=PathResult([], float('inf'), False, 0, [], []),
                target=targets[0] if targets else None,
                start_position=start_position,
                start_node=start_node,
                success=False,
                error_message=f"No path found to entity type {entity_type}"
            )
            
        return best_result
    
    def visualize_path_to_entity(
        self,
        surface: pygame.Surface,
        graph_data: GraphData,
        start_position: Tuple[float, float],
        entity_type: int,
        algorithm: PathfindingAlgorithm = PathfindingAlgorithm.A_STAR
    ) -> Optional[PathVisualizationResult]:
        """Visualize pathfinding to a specific entity type."""
        
        # Find and compute path
        result = self.find_path_to_entity_type(graph_data, start_position, entity_type, algorithm)
        
        if not result.success:
            self._draw_error_message(surface, result.error_message)
            return result
        
        # Draw the base graph
        self.base_visualizer.render_graph(surface, graph_data)
        
        # Draw the path with entity-specific color
        self._draw_enhanced_path(surface, graph_data, result)
        
        # Draw path information
        if self.show_path_info:
            self._draw_path_info_panel(surface, result)
        
        # Store result for interactive use
        self.current_paths = [result]
        
        return result
    
    def visualize_multiple_entity_paths(
        self,
        surface: pygame.Surface,
        graph_data: GraphData,
        start_position: Tuple[float, float],
        entity_types: List[EntityType],
        algorithm: PathfindingAlgorithm = PathfindingAlgorithm.A_STAR
    ) -> List[PathVisualizationResult]:
        """Visualize paths to multiple entity types with different colors."""
        
        results = []
        
        # Find paths to all entity types
        for entity_type in entity_types:
            result = self.find_path_to_entity_type(graph_data, start_position, entity_type, algorithm)
            if result.success:
                results.append(result)
        
        if not results:
            self._draw_error_message(surface, "No paths found to any target entities")
            return []
        
        # Draw the base graph
        self.base_visualizer.render_graph(surface, graph_data)
        
        # Draw all paths with different colors
        for result in results:
            self._draw_enhanced_path(surface, graph_data, result)
        
        # Draw combined path information
        if self.show_path_info:
            self._draw_multi_path_info_panel(surface, results)
        
        # Store results for interactive use
        self.current_paths = results
        
        return results
    
    def _draw_enhanced_path(
        self, 
        surface: pygame.Surface, 
        graph_data: GraphData, 
        result: PathVisualizationResult
    ):
        """Draw enhanced path visualization with entity-specific styling."""
        if not result.success or not result.path_result.path:
            return
        
        path = result.path_result.path
        target = result.target
        
        # Get path color based on entity type
        color = self.path_colors['default']
        if target and target.entity_type:
            if target.entity_type == EntityType.EXIT_SWITCH:
                color = self.path_colors['exit_switch']
            elif target.entity_type == EntityType.EXIT_DOOR:
                color = self.path_colors['exit_door']
            elif target.entity_type == EntityType.LOCKED_DOOR:
                color = self.path_colors['locked_door']
            elif target.entity_type == EntityType.TRAP_DOOR:
                color = self.path_colors['trap_door']
            elif target.entity_type == EntityType.REGULAR_DOOR:
                color = self.path_colors['regular_door']
            elif target.entity_type == EntityType.GOLD:
                color = self.path_colors['gold']
            elif target.entity_type == EntityType.NINJA:
                color = self.path_colors['ninja']
        
        # Draw path line with enhanced styling
        path_points = []
        for node_idx in path:
            position = self._get_node_position(graph_data, node_idx)
            path_points.append(position)
        
        if len(path_points) >= 2:
            # Draw thick path line
            pygame.draw.lines(surface, color, False, path_points, int(self.config.path_width * 2))
            
            # Draw thinner inner line for contrast
            inner_color = (min(255, color[0] + 50), min(255, color[1] + 50), min(255, color[2] + 50), color[3])
            pygame.draw.lines(surface, inner_color, False, path_points, max(1, int(self.config.path_width)))
        
        # Draw start marker (ninja position)
        if path_points:
            start_pos = path_points[0]
            pygame.draw.circle(surface, (0, 255, 0, 255), (int(start_pos[0]), int(start_pos[1])), 12)
            pygame.draw.circle(surface, (255, 255, 255, 255), (int(start_pos[0]), int(start_pos[1])), 12, 3)
            
            # Draw "NINJA" label
            if self.show_entity_labels:
                font = pygame.font.Font(None, 24)
                text = font.render("NINJA", True, (255, 255, 255))
                surface.blit(text, (int(start_pos[0]) - 25, int(start_pos[1]) - 35))
        
        # Draw target marker
        if target and path_points:
            end_pos = path_points[-1]
            pygame.draw.circle(surface, target.color, (int(end_pos[0]), int(end_pos[1])), 12)
            pygame.draw.circle(surface, (255, 255, 255, 255), (int(end_pos[0]), int(end_pos[1])), 12, 3)
            
            # Draw target label
            if self.show_entity_labels:
                font = pygame.font.Font(None, 20)
                text = font.render(target.label, True, (255, 255, 255))
                surface.blit(text, (int(end_pos[0]) - len(target.label) * 5, int(end_pos[1]) - 35))
        
        # Draw distance markers along path
        if self.show_distance_info and len(path_points) > 2:
            self._draw_distance_markers(surface, path_points, color)
    
    def _draw_distance_markers(self, surface: pygame.Surface, path_points: List[Tuple[float, float]], color: Tuple[int, int, int, int]):
        """Draw distance markers along the path."""
        if len(path_points) < 3:
            return
            
        # Draw markers at 25%, 50%, 75% of path
        markers = [0.25, 0.5, 0.75]
        
        for marker in markers:
            idx = int(len(path_points) * marker)
            if 0 <= idx < len(path_points):
                pos = path_points[idx]
                
                # Draw small marker circle
                pygame.draw.circle(surface, color, (int(pos[0]), int(pos[1])), 6)
                pygame.draw.circle(surface, (255, 255, 255, 255), (int(pos[0]), int(pos[1])), 6, 2)
    
    def _draw_path_info_panel(self, surface: pygame.Surface, result: PathVisualizationResult):
        """Draw detailed path information panel."""
        if not result.success:
            return
            
        # Panel background
        panel_width = 300
        panel_height = 120
        panel_x = surface.get_width() - panel_width - 10
        panel_y = 10
        
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 180))
        pygame.draw.rect(panel_surface, (255, 255, 255, 255), (0, 0, panel_width, panel_height), 2)
        
        # Path information
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 20)
        
        y_offset = 10
        
        # Title
        if result.target:
            title = f"Path to {result.target.label}"
            title_surface = font.render(title, True, (255, 255, 255))
            panel_surface.blit(title_surface, (10, y_offset))
            y_offset += 25
        
        # Path stats
        path_result = result.path_result
        stats = [
            f"Length: {len(path_result.path)} nodes",
            f"Cost: {path_result.total_cost:.1f}",
            f"Explored: {path_result.nodes_explored} nodes",
            f"Algorithm: {self.pathfinding_engine.algorithm.name}"
        ]
        
        for stat in stats:
            stat_surface = small_font.render(stat, True, (200, 200, 200))
            panel_surface.blit(stat_surface, (10, y_offset))
            y_offset += 18
        
        surface.blit(panel_surface, (panel_x, panel_y))
    
    def _draw_multi_path_info_panel(self, surface: pygame.Surface, results: List[PathVisualizationResult]):
        """Draw information panel for multiple paths."""
        if not results:
            return
            
        # Panel background
        panel_width = 350
        panel_height = min(200, 50 + len(results) * 25)
        panel_x = surface.get_width() - panel_width - 10
        panel_y = 10
        
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 180))
        pygame.draw.rect(panel_surface, (255, 255, 255, 255), (0, 0, panel_width, panel_height), 2)
        
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 18)
        
        y_offset = 10
        
        # Title
        title_surface = font.render("Multiple Entity Paths", True, (255, 255, 255))
        panel_surface.blit(title_surface, (10, y_offset))
        y_offset += 30
        
        # Path information for each result
        for result in results:
            if result.success and result.target:
                # Color indicator
                color_rect = pygame.Rect(10, y_offset, 15, 15)
                pygame.draw.rect(panel_surface, result.target.color, color_rect)
                pygame.draw.rect(panel_surface, (255, 255, 255, 255), color_rect, 1)
                
                # Path info
                info = f"{result.target.label}: {len(result.path_result.path)} nodes, cost {result.path_result.total_cost:.1f}"
                info_surface = small_font.render(info, True, (200, 200, 200))
                panel_surface.blit(info_surface, (30, y_offset))
                y_offset += 20
        
        surface.blit(panel_surface, (panel_x, panel_y))
    
    def _draw_error_message(self, surface: pygame.Surface, message: str):
        """Draw error message on surface."""
        if not message:
            return
            
        font = pygame.font.Font(None, 36)
        text_surface = font.render(f"Error: {message}", True, (255, 0, 0))
        
        # Center the message
        text_rect = text_surface.get_rect()
        text_rect.center = (surface.get_width() // 2, surface.get_height() // 2)
        
        # Draw background
        bg_rect = text_rect.inflate(20, 10)
        pygame.draw.rect(surface, (0, 0, 0, 180), bg_rect)
        pygame.draw.rect(surface, (255, 0, 0, 255), bg_rect, 2)
        
        surface.blit(text_surface, text_rect)
    
    def _get_node_position(self, graph_data: GraphData, node_idx: int) -> Tuple[float, float]:
        """Extract world position from node index and features."""
        return self.base_visualizer._get_node_position(graph_data, node_idx)
    
    def _find_closest_node(self, graph_data: GraphData, position: Tuple[float, float]) -> Optional[int]:
        """Find closest valid node to given position."""
        return self.base_visualizer._find_closest_node(graph_data, position)
    
    def get_available_entity_types(self, graph_data: GraphData) -> List[EntityType]:
        """Get list of entity types available in the current graph."""
        available_types = set()
        
        for node_idx in range(graph_data.num_nodes):
            if graph_data.node_mask[node_idx] == 0:
                continue
                
            node_type = NodeType(graph_data.node_types[node_idx])
            if node_type != NodeType.ENTITY:
                continue
                
            # Check entity features to determine type
            node_features = graph_data.node_features[node_idx]
            if len(node_features) >= 7:  # position(2) + tile_type + 4 + entity_type
                entity_type_value = int(node_features[6])
                try:
                    entity_type = EntityType(entity_type_value)
                    available_types.add(entity_type)
                except ValueError:
                    continue  # Unknown entity type
        
        return sorted(list(available_types), key=lambda x: x.value)
    
    def clear_paths(self):
        """Clear all current path visualizations."""
        self.current_paths = []
    
    def toggle_path_info(self):
        """Toggle path information display."""
        self.show_path_info = not self.show_path_info
    
    def toggle_entity_labels(self):
        """Toggle entity label display."""
        self.show_entity_labels = not self.show_entity_labels
    
    def toggle_distance_info(self):
        """Toggle distance marker display."""
        self.show_distance_info = not self.show_distance_info