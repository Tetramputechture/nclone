"""
Unified API for N++ graph visualization system.

This module provides a high-level interface for controlling all aspects
of graph visualization, including standalone rendering, simulator overlays,
pathfinding, and interactive exploration.
"""

import pygame
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, asdict
from enum import IntEnum
import json

from .visualization import GraphVisualizer, VisualizationConfig, VisualizationMode, InteractiveGraphVisualizer
from .pathfinding import PathfindingEngine, PathfindingAlgorithm, PathResult
from .enhanced_debug_overlay import EnhancedDebugOverlay, OverlayMode
from .hierarchical_builder import HierarchicalGraphBuilder
from .graph_construction import GraphConstructor
from .feature_extraction import FeatureExtractor
from .edge_building import EdgeBuilder
from .common import GraphData, NodeType, EdgeType


class RenderTarget(IntEnum):
    """Target for rendering output."""
    SCREEN = 0          # Render to pygame screen
    SURFACE = 1         # Render to pygame surface
    FILE = 2            # Save to image file
    MEMORY = 3          # Return as numpy array


@dataclass
class VisualizationRequest:
    """Request for graph visualization."""
    # Data sources
    level_data: Optional[Dict[str, Any]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    ninja_position: Optional[Tuple[float, float]] = None
    ninja_velocity: Optional[Tuple[float, float]] = None
    ninja_state: int = 0
    
    # Pathfinding
    goal_position: Optional[Tuple[float, float]] = None
    start_position: Optional[Tuple[float, float]] = None
    pathfinding_algorithm: PathfindingAlgorithm = PathfindingAlgorithm.A_STAR
    
    # Visualization settings
    mode: VisualizationMode = VisualizationMode.STANDALONE
    config: Optional[VisualizationConfig] = None
    use_hierarchical: bool = False
    
    # Output settings
    render_target: RenderTarget = RenderTarget.SURFACE
    output_size: Tuple[int, int] = (1200, 800)
    output_path: Optional[str] = None


@dataclass
class VisualizationResult:
    """Result of visualization operation."""
    success: bool
    surface: Optional[pygame.Surface] = None
    path_result: Optional[PathResult] = None
    graph_stats: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    render_time_ms: int = 0


class GraphVisualizationAPI:
    """
    Unified API for N++ graph visualization.
    
    Provides a high-level interface for all graph visualization operations,
    including standalone rendering, simulator integration, and interactive
    exploration.
    """
    
    def __init__(self):
        """Initialize visualization API."""
        # Core components
        self.visualizer = GraphVisualizer()
        self.pathfinding_engine = None  # Will be initialized with level data
        self.graph_builder = HierarchicalGraphBuilder()
        
        # Graph construction components
        self.feature_extractor = FeatureExtractor()
        self.edge_builder = EdgeBuilder()
        self.graph_constructor = GraphConstructor(self.feature_extractor, self.edge_builder)
        
        # State
        self.current_graph_data = None
        self.current_hierarchical_data = None
        self.cached_level_id = None
        
        # Performance tracking
        self.stats = {
            'total_visualizations': 0,
            'total_pathfinding_operations': 0,
            'average_render_time': 0.0,
            'cache_hits': 0,
        }
    
    def visualize_graph(self, request: VisualizationRequest) -> VisualizationResult:
        """
        Create graph visualization based on request parameters.
        
        Args:
            request: Visualization request with all parameters
            
        Returns:
            VisualizationResult with rendered output and metadata
        """
        start_time = pygame.time.get_ticks()
        
        try:
            # Build or retrieve graph data
            graph_data = self._get_graph_data(request)
            if graph_data is None:
                return VisualizationResult(
                    success=False,
                    error_message="Failed to build graph data"
                )
            
            # Perform pathfinding if requested
            path_result = None
            if request.goal_position and (request.start_position or request.ninja_position):
                start_pos = request.start_position or request.ninja_position
                
                # Initialize pathfinding engine with level data for accurate physics
                if self.pathfinding_engine is None or self.pathfinding_engine.level_data != request.level_data:
                    self.pathfinding_engine = PathfindingEngine(
                        level_data=request.level_data,
                        entities=request.entities or []
                    )
                
                # Create ninja state for accurate pathfinding
                ninja_state = {
                    'movement_state': request.ninja_state,
                    'velocity': request.ninja_velocity or (0.0, 0.0),
                    'position': start_pos,
                    'ground_contact': True,  # Default assumption
                    'wall_contact': False   # Default assumption
                }
                
                path_result = self.pathfinding_engine.find_shortest_path(
                    graph_data,
                    self._find_closest_node(graph_data, start_pos),
                    self._find_closest_node(graph_data, request.goal_position),
                    request.pathfinding_algorithm,
                    ninja_state=ninja_state
                )
            
            # Create visualization
            surface = self._create_visualization_surface(request, graph_data, path_result)
            
            # Handle output
            if request.render_target == RenderTarget.FILE and request.output_path:
                pygame.image.save(surface, request.output_path)
            
            # Calculate statistics
            graph_stats = self._calculate_graph_stats(graph_data, path_result)
            render_time = pygame.time.get_ticks() - start_time
            
            # Update performance stats
            self.stats['total_visualizations'] += 1
            self.stats['average_render_time'] = (
                (self.stats['average_render_time'] * (self.stats['total_visualizations'] - 1) + render_time) /
                self.stats['total_visualizations']
            )
            
            return VisualizationResult(
                success=True,
                surface=surface,
                path_result=path_result,
                graph_stats=graph_stats,
                render_time_ms=render_time
            )
            
        except Exception as e:
            return VisualizationResult(
                success=False,
                error_message=f"Visualization error: {str(e)}",
                render_time_ms=pygame.time.get_ticks() - start_time
            )
    
    def create_interactive_session(
        self,
        level_data: Dict[str, Any],
        entities: List[Dict[str, Any]],
        width: int = 1400,
        height: int = 900
    ) -> InteractiveGraphVisualizer:
        """
        Create interactive visualization session.
        
        Args:
            level_data: Level geometry data
            entities: List of entities
            width: Window width
            height: Window height
            
        Returns:
            Interactive visualizer instance
        """
        # Build graph data
        ninja_position = (0.0, 0.0)  # Default position
        ninja_velocity = (0.0, 0.0)
        ninja_state = 0
        
        graph_data = self.graph_constructor.build_sub_cell_graph(
            level_data, ninja_position, entities, ninja_velocity, ninja_state,
            node_feature_dim=16, edge_feature_dim=8
        )
        
        # Create interactive visualizer
        interactive_viz = InteractiveGraphVisualizer(width, height)
        return interactive_viz
    
    def create_simulator_overlay(
        self,
        sim,
        screen: pygame.Surface,
        adjust: float,
        tile_x_offset: float,
        tile_y_offset: float
    ) -> EnhancedDebugOverlay:
        """
        Create enhanced debug overlay for simulator integration.
        
        Args:
            sim: Simulator instance
            screen: Pygame screen surface
            adjust: Scaling adjustment factor
            tile_x_offset: X offset for tile rendering
            tile_y_offset: Y offset for tile rendering
            
        Returns:
            Enhanced debug overlay instance
        """
        return EnhancedDebugOverlay(sim, screen, adjust, tile_x_offset, tile_y_offset)
    
    def find_shortest_path(
        self,
        level_data: Dict[str, Any],
        entities: List[Dict[str, Any]],
        start_position: Tuple[float, float],
        goal_position: Tuple[float, float],
        ninja_velocity: Tuple[float, float] = (0.0, 0.0),
        ninja_state: int = 0,
        algorithm: PathfindingAlgorithm = PathfindingAlgorithm.A_STAR
    ) -> PathResult:
        """
        Find shortest path between two positions.
        
        Args:
            level_data: Level geometry data
            entities: List of entities
            start_position: Starting position (x, y)
            goal_position: Goal position (x, y)
            ninja_velocity: Current ninja velocity
            ninja_state: Current ninja state
            algorithm: Pathfinding algorithm to use
            
        Returns:
            PathResult with path information
        """
        # Build graph data
        graph_data = self.graph_constructor.build_sub_cell_graph(
            level_data, start_position, entities, ninja_velocity, ninja_state,
            node_feature_dim=16, edge_feature_dim=8
        )
        
        # Find path
        start_node = self._find_closest_node(graph_data, start_position)
        goal_node = self._find_closest_node(graph_data, goal_position)
        
        if start_node is None or goal_node is None:
            return PathResult(
                path=[], total_cost=float('inf'), success=False,
                nodes_explored=0, path_coordinates=[], edge_types=[]
            )
        
        self.stats['total_pathfinding_operations'] += 1
        return self.pathfinding_engine.find_shortest_path(
            graph_data, start_node, goal_node, algorithm
        )
    
    def export_visualization_config(self, config: VisualizationConfig, filepath: str):
        """Export visualization configuration to JSON file."""
        config_dict = asdict(config)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def import_visualization_config(self, filepath: str) -> VisualizationConfig:
        """Import visualization configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return VisualizationConfig(**config_dict)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear cached graph data."""
        self.current_graph_data = None
        self.current_hierarchical_data = None
        self.cached_level_id = None
        self.stats['cache_hits'] = 0
    
    def _get_graph_data(self, request: VisualizationRequest) -> Optional[GraphData]:
        """Get or build graph data based on request."""
        if not request.level_data:
            return None
        
        # Check cache
        level_id = request.level_data.get('level_id', id(request.level_data))
        if level_id == self.cached_level_id and self.current_graph_data is not None:
            self.stats['cache_hits'] += 1
            return self.current_graph_data
        
        # Build new graph data
        ninja_position = request.ninja_position or (0.0, 0.0)
        ninja_velocity = request.ninja_velocity or (0.0, 0.0)
        entities = request.entities or []
        
        if request.use_hierarchical:
            hierarchical_data = self.graph_builder.build_hierarchical_graph(
                request.level_data, ninja_position, entities,
                ninja_velocity, request.ninja_state,
                node_feature_dim=16, edge_feature_dim=8
            )
            self.current_hierarchical_data = hierarchical_data
            graph_data = hierarchical_data.sub_cell_graph
        else:
            graph_data = self.graph_constructor.build_sub_cell_graph(
                request.level_data, ninja_position, entities,
                ninja_velocity, request.ninja_state,
                node_feature_dim=16, edge_feature_dim=8
            )
        
        # Cache the result
        self.current_graph_data = graph_data
        self.cached_level_id = level_id
        
        return graph_data
    
    def _create_visualization_surface(
        self,
        request: VisualizationRequest,
        graph_data: GraphData,
        path_result: Optional[PathResult]
    ) -> pygame.Surface:
        """Create visualization surface based on request parameters."""
        # Update visualizer config
        if request.config:
            self.visualizer.config = request.config
        
        # Create visualization based on mode
        if request.mode == VisualizationMode.STANDALONE:
            return self.visualizer.create_standalone_visualization(
                graph_data,
                request.output_size[0],
                request.output_size[1],
                request.goal_position,
                request.start_position or request.ninja_position
            )
        elif request.mode == VisualizationMode.OVERLAY:
            # Create a dummy simulator surface for overlay
            sim_surface = pygame.Surface(request.output_size)
            sim_surface.fill((50, 50, 50))  # Dark background
            
            return self.visualizer.create_overlay_visualization(
                graph_data,
                sim_surface,
                request.goal_position,
                request.start_position or request.ninja_position,
                request.ninja_position
            )
        elif request.mode == VisualizationMode.SIDE_BY_SIDE:
            # Create a dummy simulator surface
            sim_surface = pygame.Surface((request.output_size[0] // 2, request.output_size[1]))
            sim_surface.fill((50, 50, 50))  # Dark background
            
            return self.visualizer.create_side_by_side_visualization(
                graph_data,
                sim_surface,
                request.goal_position,
                request.start_position or request.ninja_position,
                request.ninja_position
            )
        else:
            # Default to standalone
            return self.visualizer.create_standalone_visualization(
                graph_data,
                request.output_size[0],
                request.output_size[1],
                request.goal_position,
                request.start_position or request.ninja_position
            )
    
    def _find_closest_node(self, graph_data: GraphData, position: Tuple[float, float]) -> Optional[int]:
        """Find the node closest to a given position."""
        target_x, target_y = position
        best_node = None
        best_distance = float('inf')
        
        for node_idx in range(graph_data.num_nodes):
            if graph_data.node_mask[node_idx] == 0:
                continue
            
            # Get node position from features
            node_features = graph_data.node_features[node_idx]
            node_x = node_features[0] if len(node_features) > 0 else 0.0
            node_y = node_features[1] if len(node_features) > 1 else 0.0
            
            distance = ((node_x - target_x) ** 2 + (node_y - target_y) ** 2) ** 0.5
            
            if distance < best_distance:
                best_distance = distance
                best_node = node_idx
        
        return best_node
    
    def _calculate_graph_stats(
        self,
        graph_data: GraphData,
        path_result: Optional[PathResult]
    ) -> Dict[str, Any]:
        """Calculate graph statistics."""
        stats = {
            'total_nodes': graph_data.num_nodes,
            'total_edges': graph_data.num_edges,
            'active_nodes': int(graph_data.node_mask.sum()),
            'active_edges': int(graph_data.edge_mask.sum()),
        }
        
        # Node type distribution
        node_type_counts = {}
        for node_idx in range(graph_data.num_nodes):
            if graph_data.node_mask[node_idx] > 0:
                node_type = NodeType(graph_data.node_types[node_idx])
                node_type_counts[node_type.name] = node_type_counts.get(node_type.name, 0) + 1
        stats['node_types'] = node_type_counts
        
        # Edge type distribution
        edge_type_counts = {}
        for edge_idx in range(graph_data.num_edges):
            if graph_data.edge_mask[edge_idx] > 0:
                edge_type = EdgeType(graph_data.edge_types[edge_idx])
                edge_type_counts[edge_type.name] = edge_type_counts.get(edge_type.name, 0) + 1
        stats['edge_types'] = edge_type_counts
        
        # Path statistics
        if path_result:
            stats['path_found'] = path_result.success
            stats['path_length'] = len(path_result.path)
            stats['path_cost'] = path_result.total_cost
            stats['nodes_explored'] = path_result.nodes_explored
        
        return stats


# Convenience functions for common operations
def visualize_level_graph(
    level_data: Dict[str, Any],
    entities: List[Dict[str, Any]],
    output_path: str,
    goal_position: Optional[Tuple[float, float]] = None,
    ninja_position: Optional[Tuple[float, float]] = None,
    size: Tuple[int, int] = (1200, 800)
) -> bool:
    """
    Convenience function to visualize a level's graph and save to file.
    
    Args:
        level_data: Level geometry data
        entities: List of entities
        output_path: Path to save visualization image
        goal_position: Optional goal position for pathfinding
        ninja_position: Optional ninja position
        size: Output image size
        
    Returns:
        True if successful, False otherwise
    """
    api = GraphVisualizationAPI()
    
    request = VisualizationRequest(
        level_data=level_data,
        entities=entities,
        ninja_position=ninja_position,
        goal_position=goal_position,
        render_target=RenderTarget.FILE,
        output_size=size,
        output_path=output_path
    )
    
    result = api.visualize_graph(request)
    return result.success


def find_path_and_visualize(
    level_data: Dict[str, Any],
    entities: List[Dict[str, Any]],
    start_position: Tuple[float, float],
    goal_position: Tuple[float, float],
    output_path: str,
    size: Tuple[int, int] = (1200, 800)
) -> Tuple[bool, Optional[PathResult]]:
    """
    Convenience function to find path and create visualization.
    
    Args:
        level_data: Level geometry data
        entities: List of entities
        start_position: Starting position
        goal_position: Goal position
        output_path: Path to save visualization
        size: Output image size
        
    Returns:
        Tuple of (success, path_result)
    """
    api = GraphVisualizationAPI()
    
    request = VisualizationRequest(
        level_data=level_data,
        entities=entities,
        start_position=start_position,
        goal_position=goal_position,
        render_target=RenderTarget.FILE,
        output_size=size,
        output_path=output_path
    )
    
    result = api.visualize_graph(request)
    return result.success, result.path_result