"""
Edge building utilities for graph construction.

This module provides specialized edge building methods with comprehensive
bounce block awareness and traversability checking.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from ..constants.physics_constants import FULL_MAP_WIDTH, FULL_MAP_HEIGHT, TILE_PIXEL_SIZE
from .common import SUB_CELL_SIZE, EdgeType, E_MAX_EDGES
from .feature_extraction import FeatureExtractor


class EdgeBuilder:
    """
    Handles edge construction with bounce block awareness and traversability checking.
    """
    
    def __init__(self, feature_extractor: FeatureExtractor):
        """
        Initialize edge builder.
        
        Args:
            feature_extractor: Feature extraction utility instance
        """
        self.feature_extractor = feature_extractor
    
    def build_edges(
        self,
        sub_grid_node_map: Dict[Tuple[int, int], int],
        entity_nodes: List[Tuple[int, Dict[str, Any]]],
        entities: List[Dict[str, Any]],
        level_data: Dict[str, Any],
        ninja_position: Tuple[float, float],
        ninja_velocity: Optional[Tuple[float, float]],
        ninja_state: Optional[int],
        edge_index: np.ndarray,
        edge_features: np.ndarray,
        edge_mask: np.ndarray,
        edge_types: np.ndarray,
        edge_count: int,
        edge_feature_dim: int
    ) -> int:
        """
        Build edges with bounce block awareness.
        
        Args:
            sub_grid_node_map: Mapping from (row, col) to node indices
            entity_nodes: List of (node_idx, entity) tuples
            entities: List of entity dictionaries
            level_data: Level tile data and structure
            ninja_position: Current ninja position (x, y)
            ninja_velocity: Current ninja velocity (vx, vy)
            ninja_state: Current ninja movement state (0-8)
            edge_index: Edge connectivity array
            edge_features: Edge feature array
            edge_mask: Edge mask array
            edge_types: Edge type array
            edge_count: Current edge count
            edge_feature_dim: Edge feature dimension
            
        Returns:
            Updated edge count
        """
        # Build sub-grid traversability edges
        for (sub_row, sub_col), src_idx in sub_grid_node_map.items():
            if edge_count >= E_MAX_EDGES - 8:  # Leave room for 8 directions
                break
                
            # Check 8-connected neighbors
            directions = [
                (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal
                (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
            ]
            
            for dr, dc in directions:
                tgt_row = sub_row + dr
                tgt_col = sub_col + dc
                
                if (tgt_row, tgt_col) in sub_grid_node_map:
                    tgt_idx = sub_grid_node_map[(tgt_row, tgt_col)]
                    
                    # Check traversability with bounce block awareness
                    if self.is_traversable_with_bounce_blocks(
                        sub_row, sub_col, tgt_row, tgt_col, entities, level_data
                    ):
                        edge_index[0, edge_count] = src_idx
                        edge_index[1, edge_count] = tgt_idx
                        
                        # Extract edge features
                        edge_features[edge_count] = self.feature_extractor.extract_edge_features(
                            sub_row, sub_col, tgt_row, tgt_col, entities, ninja_position, ninja_velocity, edge_feature_dim
                        )
                        
                        edge_mask[edge_count] = 1.0
                        edge_types[edge_count] = EdgeType.WALK
                        edge_count += 1
        
        # Build entity interaction edges
        edge_count = self.build_entity_edges(
            entity_nodes, entities, ninja_position, ninja_velocity,
            edge_index, edge_features, edge_mask, edge_types, edge_count, edge_feature_dim
        )
        
        return edge_count
    
    def build_entity_edges(
        self,
        entity_nodes: List[Tuple[int, Dict[str, Any]]],
        entities: List[Dict[str, Any]],
        ninja_position: Tuple[float, float],
        ninja_velocity: Optional[Tuple[float, float]],
        edge_index: np.ndarray,
        edge_features: np.ndarray,
        edge_mask: np.ndarray,
        edge_types: np.ndarray,
        edge_count: int,
        edge_feature_dim: int
    ) -> int:
        """
        Build entity interaction edges with bounce block chaining support.
        
        Args:
            entity_nodes: List of (node_idx, entity) tuples
            entities: List of entity dictionaries
            ninja_position: Current ninja position (x, y)
            ninja_velocity: Current ninja velocity (vx, vy)
            edge_index: Edge connectivity array
            edge_features: Edge feature array
            edge_mask: Edge mask array
            edge_types: Edge type array
            edge_count: Current edge count
            edge_feature_dim: Edge feature dimension
            
        Returns:
            Updated edge count
        """
        # Build bounce block interaction edges
        for i, (node_idx, entity) in enumerate(entity_nodes):
            if edge_count >= E_MAX_EDGES - 10:  # Leave room for more edges
                break
                
            if entity.get('type') == 17:  # Bounce block
                # Create edges to nearby bounce blocks for chaining
                for j, (other_idx, other_entity) in enumerate(entity_nodes):
                    if i != j and other_entity.get('type') == 17:
                        entity_x = entity.get('x', 0.0)
                        entity_y = entity.get('y', 0.0)
                        other_x = other_entity.get('x', 0.0)
                        other_y = other_entity.get('y', 0.0)
                        
                        distance = math.sqrt((entity_x - other_x)**2 + (entity_y - other_y)**2)
                        
                        # Connect nearby bounce blocks for chaining mechanics
                        if distance < 50.0:  # Within chaining range
                            edge_index[0, edge_count] = node_idx
                            edge_index[1, edge_count] = other_idx
                            
                            # Create bounce chain edge features
                            chain_features = np.zeros(edge_feature_dim, dtype=np.float32)
                            chain_features[EdgeType.FUNCTIONAL] = 1.0  # Functional edge type
                            
                            # Direction
                            if distance > 0:
                                chain_features[len(EdgeType)] = (other_x - entity_x) / distance
                                chain_features[len(EdgeType) + 1] = (other_y - entity_y) / distance
                            
                            # Chain interaction parameters
                            chain_features[len(EdgeType) + 2] = 0.5  # Low traversability cost
                            chain_features[len(EdgeType) + 3] = 0.3  # Fast time of flight
                            chain_features[len(EdgeType) + 4] = 0.6  # Moderate energy cost
                            chain_features[len(EdgeType) + 5] = 0.85  # High success probability
                            chain_features[len(EdgeType) + 8] = 1.0  # Bounce boost available
                            
                            edge_features[edge_count] = chain_features
                            edge_mask[edge_count] = 1.0
                            edge_types[edge_count] = EdgeType.FUNCTIONAL
                            edge_count += 1
        
        return edge_count
    
    def is_traversable_with_bounce_blocks(
        self,
        src_row: int,
        src_col: int,
        tgt_row: int,
        tgt_col: int,
        entities: List[Dict[str, Any]],
        level_data: Dict[str, Any]
    ) -> bool:
        """
        Check if movement between sub-cells is traversable considering bounce blocks.
        
        Args:
            src_row: Source sub-cell row
            src_col: Source sub-cell column
            tgt_row: Target sub-cell row
            tgt_col: Target sub-cell column
            entities: List of entity dictionaries
            level_data: Level tile data and structure
            
        Returns:
            True if movement is traversable
        """
        # Convert to pixel coordinates
        src_x = src_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        src_y = src_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        tgt_x = tgt_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        tgt_y = tgt_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        
        # Check for bounce blocks in the path
        for entity in entities:
            if entity.get('type') == 17:  # Bounce block
                entity_x = entity.get('x', 0.0)
                entity_y = entity.get('y', 0.0)
                
                # Check if path intersects with bounce block
                if self.feature_extractor._path_intersects_bounce_block(src_x, src_y, tgt_x, tgt_y, entity_x, entity_y):
                    # Bounce blocks are traversable but may affect movement
                    return True
        
        # Check basic tile traversability
        return self.is_basic_traversable(src_x, src_y, tgt_x, tgt_y, level_data)
    
    def is_basic_traversable(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        level_data: Dict[str, Any]
    ) -> bool:
        """
        Check basic tile traversability.
        
        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            level_data: Level tile data and structure
            
        Returns:
            True if tiles are traversable
        """
        # Convert to tile coordinates
        src_tile_x = int(src_x // TILE_PIXEL_SIZE)
        src_tile_y = int(src_y // TILE_PIXEL_SIZE)
        tgt_tile_x = int(tgt_x // TILE_PIXEL_SIZE)
        tgt_tile_y = int(tgt_y // TILE_PIXEL_SIZE)
        
        # Check bounds
        if not (0 <= src_tile_x < FULL_MAP_WIDTH and 0 <= src_tile_y < FULL_MAP_HEIGHT):
            return False
        if not (0 <= tgt_tile_x < FULL_MAP_WIDTH and 0 <= tgt_tile_y < FULL_MAP_HEIGHT):
            return False
        
        # Check if tiles are solid
        tiles = level_data.get('tiles', np.zeros((FULL_MAP_HEIGHT, FULL_MAP_WIDTH), dtype=np.int32))
        src_tile = tiles[src_tile_y, src_tile_x]
        tgt_tile = tiles[tgt_tile_y, tgt_tile_x]
        
        # Basic traversability: both tiles should be empty (0) or non-solid
        return src_tile == 0 and tgt_tile == 0