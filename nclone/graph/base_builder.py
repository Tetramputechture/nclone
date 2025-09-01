"""
Base graph builder for creating structural representations of N++ levels.

This module provides the core GraphBuilder class that was moved from archive
to support the hierarchical graph builder.
"""

import numpy as np
import logging
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum

# Use shared constants from the simulator
from ..constants.physics_constants import FULL_MAP_WIDTH, FULL_MAP_HEIGHT, TILE_PIXEL_SIZE
from .common import GraphData, NodeType, EdgeType, SUB_CELL_SIZE, SUB_GRID_WIDTH, SUB_GRID_HEIGHT, N_MAX_NODES, E_MAX_EDGES


class GraphBuilder:
    """
    Builds graph representations of N++ levels.
    
    The graph construction follows a canonical ordering:
    1. Grid cells: iterate rows then columns (top-left to bottom-right)
    2. Entities: sorted by type then by position
    3. Edges: sorted by (source, target, edge_type)
    """
    
    def __init__(self):
        """Initialize graph builder."""
        # Node feature dimensions
        self.tile_type_dim = 38  # Number of tile types in N++
        self.entity_type_dim = 30  # Extended for bounce block states
        self.node_feature_dim = (
            self.tile_type_dim +  # One-hot tile type (38)
            4 +  # Solidity flags (solid, half, slope, hazard)
            self.entity_type_dim +  # One-hot entity type (30)
            8 +  # Entity state (active, position_x, position_y, custom_state, bounce_state, compression, velocity_x, velocity_y)
            1 +  # Ninja position flag
            # Physics state features (18 additional)
            2 +  # Ninja velocity (vx, vy) normalized by MAX_HOR_SPEED
            1 +  # Velocity magnitude
            1 +  # Movement state (0-9 from sim_mechanics_doc.md)
            3 +  # Contact flags (ground_contact, wall_contact, airborne)
            2 +  # Momentum direction (normalized)
            1 +  # Kinetic energy (0.5 * m * v²)
            1 +  # Potential energy (relative to level bottom)
            5 +  # Input buffers (jump_buffer, floor_buffer, wall_buffer, launch_pad_buffer, input_state)
            2    # Physics capabilities (can_jump, can_wall_jump)
        )
        
        # Edge feature dimensions  
        self.edge_feature_dim = (
            len(EdgeType) +  # One-hot edge type (6)
            2 +  # Direction (dx, dy normalized) 
            1 +  # Traversability cost
            3 +  # Trajectory parameters (time_of_flight, energy_cost, success_probability)
            2 +  # Physics constraints (min_velocity, max_velocity)
            4    # Movement requirements (requires_jump, requires_wall_contact, bounce_boost_available, compression_state)
        )
        
        # Initialize trajectory calculator (lazy loading to avoid import issues)
        self.trajectory_calc = None
        self.movement_classifier = None
        self.physics_extractor = None
    
    def build_graph(
        self,
        level_data: Dict[str, Any],
        ninja_position: Tuple[float, float],
        entities: List[Dict[str, Any]],
        ninja_velocity: Optional[Tuple[float, float]] = None,
        ninja_state: Optional[int] = None
    ) -> GraphData:
        """
        Build graph representation of a level.
        
        Args:
            level_data: Level tile data and structure
            ninja_position: Current ninja position (x, y)
            entities: List of entity dictionaries with type, position, state
            ninja_velocity: Current ninja velocity (vx, vy) for physics calculations
            ninja_state: Current ninja movement state (0-8)
            
        Returns:
            GraphData with padded arrays for Gym compatibility
        """
        # Initialize arrays
        node_features = np.zeros((N_MAX_NODES, self.node_feature_dim), dtype=np.float32)
        edge_index = np.zeros((2, E_MAX_EDGES), dtype=np.int32)
        edge_features = np.zeros((E_MAX_EDGES, self.edge_feature_dim), dtype=np.float32)
        node_mask = np.zeros(N_MAX_NODES, dtype=np.float32)
        edge_mask = np.zeros(E_MAX_EDGES, dtype=np.float32)
        node_types = np.zeros(N_MAX_NODES, dtype=np.int32)
        edge_types = np.zeros(E_MAX_EDGES, dtype=np.int32)
        
        node_count = 0
        edge_count = 0
        
        # Build sub-grid nodes (canonical ordering: sub-row by sub-row)
        sub_grid_node_map = {}  # (sub_row, sub_col) -> node_index
        
        for sub_row in range(SUB_GRID_HEIGHT):
            for sub_col in range(SUB_GRID_WIDTH):
                if node_count >= N_MAX_NODES:
                    break
                    
                node_idx = node_count
                sub_grid_node_map[(sub_row, sub_col)] = node_idx
                
                # Extract sub-cell features
                sub_cell_features = self._extract_sub_cell_features(
                    level_data, sub_row, sub_col, ninja_position, ninja_velocity, ninja_state, entities
                )
                
                node_features[node_idx] = sub_cell_features
                node_mask[node_idx] = 1.0
                node_types[node_idx] = NodeType.GRID_CELL
                node_count += 1
        
        # Build entity nodes (sorted by type then position for determinism)
        entity_nodes = []
        sorted_entities = sorted(entities, key=self._entity_sort_key)
        
        for entity in sorted_entities:
            if node_count >= N_MAX_NODES:
                break
                
            node_idx = node_count
            entity_nodes.append((node_idx, entity))
            
            # Extract entity features
            entity_features = self._extract_entity_features(entity, ninja_position, ninja_velocity)
            
            node_features[node_idx] = entity_features
            node_mask[node_idx] = 1.0
            node_types[node_idx] = NodeType.ENTITY
            node_count += 1
        
        # Build edges with bounce block awareness
        edge_count = self._build_edges(
            sub_grid_node_map, entity_nodes, entities, level_data,
            ninja_position, ninja_velocity, ninja_state,
            edge_index, edge_features, edge_mask, edge_types, edge_count
        )
        
        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_mask=node_mask,
            edge_mask=edge_mask,
            node_types=node_types,
            edge_types=edge_types,
            num_nodes=node_count,
            num_edges=edge_count
        )
    
    def _extract_sub_cell_features(
        self,
        level_data: Dict[str, Any],
        sub_row: int,
        sub_col: int,
        ninja_position: Tuple[float, float],
        ninja_velocity: Optional[Tuple[float, float]],
        ninja_state: Optional[int],
        entities: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Extract features for a sub-cell node."""
        features = np.zeros(self.node_feature_dim, dtype=np.float32)
        
        # Convert sub-cell to pixel coordinates
        pixel_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        pixel_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        
        # Convert to tile coordinates
        tile_x = pixel_x // TILE_PIXEL_SIZE
        tile_y = pixel_y // TILE_PIXEL_SIZE
        
        # Extract tile features if within bounds
        if 0 <= tile_x < FULL_MAP_WIDTH and 0 <= tile_y < FULL_MAP_HEIGHT:
            tile_type = level_data.get('tiles', np.zeros((FULL_MAP_HEIGHT, FULL_MAP_WIDTH), dtype=np.int32))[tile_y, tile_x]
            if 0 <= tile_type < self.tile_type_dim:
                features[tile_type] = 1.0
        
        # Add ninja position flag
        ninja_x, ninja_y = ninja_position
        ninja_distance = math.sqrt((pixel_x - ninja_x)**2 + (pixel_y - ninja_y)**2)
        features[self.tile_type_dim + 4 + self.entity_type_dim + 8] = 1.0 if ninja_distance < SUB_CELL_SIZE else 0.0
        
        # Add physics state features
        if ninja_velocity is not None:
            vx, vy = ninja_velocity
            # Normalize velocity features
            max_speed = 200.0  # Approximate max speed
            features[self.tile_type_dim + 4 + self.entity_type_dim + 8 + 1] = vx / max_speed
            features[self.tile_type_dim + 4 + self.entity_type_dim + 8 + 2] = vy / max_speed
            features[self.tile_type_dim + 4 + self.entity_type_dim + 8 + 3] = math.sqrt(vx*vx + vy*vy) / max_speed
        
        if ninja_state is not None:
            features[self.tile_type_dim + 4 + self.entity_type_dim + 8 + 4] = ninja_state / 8.0
        
        return features
    
    def _extract_entity_features(
        self,
        entity: Dict[str, Any],
        ninja_position: Tuple[float, float],
        ninja_velocity: Optional[Tuple[float, float]]
    ) -> np.ndarray:
        """Extract features for an entity node."""
        features = np.zeros(self.node_feature_dim, dtype=np.float32)
        
        # Entity type one-hot encoding
        entity_type = entity.get('type', 0)
        if 0 <= entity_type < self.entity_type_dim:
            features[self.tile_type_dim + 4 + entity_type] = 1.0
        
        # Entity state features
        state_offset = self.tile_type_dim + 4 + self.entity_type_dim
        features[state_offset] = 1.0  # Active flag
        features[state_offset + 1] = entity.get('x', 0.0) / 1000.0  # Normalized position
        features[state_offset + 2] = entity.get('y', 0.0) / 600.0
        features[state_offset + 3] = entity.get('state', 0.0) / 10.0  # Normalized state
        
        # Bounce block specific features
        if entity_type == 17:  # Bounce block
            features[state_offset + 4] = entity.get('bounce_state', 0.0) / 3.0  # Bounce state (0-3)
            features[state_offset + 5] = entity.get('compression', 0.0)  # Compression ratio (0-1)
            features[state_offset + 6] = entity.get('velocity_x', 0.0) / 100.0  # Normalized velocity
            features[state_offset + 7] = entity.get('velocity_y', 0.0) / 100.0
        
        return features
    
    def _build_edges(
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
        edge_count: int
    ) -> int:
        """Build edges with bounce block awareness."""
        
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
                    if self._is_traversable_with_bounce_blocks(
                        sub_row, sub_col, tgt_row, tgt_col, entities, level_data
                    ):
                        edge_index[0, edge_count] = src_idx
                        edge_index[1, edge_count] = tgt_idx
                        
                        # Extract edge features
                        edge_features[edge_count] = self._extract_edge_features(
                            sub_row, sub_col, tgt_row, tgt_col, entities, ninja_position, ninja_velocity
                        )
                        
                        edge_mask[edge_count] = 1.0
                        edge_types[edge_count] = EdgeType.WALK
                        edge_count += 1
        
        # Build entity interaction edges
        edge_count = self._build_entity_edges(
            entity_nodes, entities, ninja_position, ninja_velocity,
            edge_index, edge_features, edge_mask, edge_types, edge_count
        )
        
        return edge_count
    
    def _is_traversable_with_bounce_blocks(
        self,
        src_row: int,
        src_col: int,
        tgt_row: int,
        tgt_col: int,
        entities: List[Dict[str, Any]],
        level_data: Dict[str, Any]
    ) -> bool:
        """Check if movement between sub-cells is traversable considering bounce blocks."""
        
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
                if self._path_intersects_bounce_block(src_x, src_y, tgt_x, tgt_y, entity_x, entity_y):
                    # Bounce blocks are traversable but may affect movement
                    return True
        
        # Check basic tile traversability
        return self._is_basic_traversable(src_x, src_y, tgt_x, tgt_y, level_data)
    
    def _path_intersects_bounce_block(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        block_x: float,
        block_y: float
    ) -> bool:
        """Check if path intersects with a bounce block."""
        # Simple distance check for now
        block_radius = 9.0  # Bounce block semi-side
        
        # Check if either endpoint is within bounce block
        src_dist = math.sqrt((src_x - block_x)**2 + (src_y - block_y)**2)
        tgt_dist = math.sqrt((tgt_x - block_x)**2 + (tgt_y - block_y)**2)
        
        return src_dist <= block_radius or tgt_dist <= block_radius
    
    def _is_basic_traversable(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        level_data: Dict[str, Any]
    ) -> bool:
        """Check basic tile traversability."""
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
    
    def _extract_edge_features(
        self,
        src_row: int,
        src_col: int,
        tgt_row: int,
        tgt_col: int,
        entities: List[Dict[str, Any]],
        ninja_position: Tuple[float, float],
        ninja_velocity: Optional[Tuple[float, float]]
    ) -> np.ndarray:
        """Extract features for an edge."""
        features = np.zeros(self.edge_feature_dim, dtype=np.float32)
        
        # Direction vector
        dx = tgt_col - src_col
        dy = tgt_row - src_row
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 0:
            features[len(EdgeType)] = dx / distance  # Normalized dx
            features[len(EdgeType) + 1] = dy / distance  # Normalized dy
        
        # Traversability cost (base cost)
        features[len(EdgeType) + 2] = 1.0
        
        # Check for bounce block interactions
        src_x = src_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        src_y = src_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        tgt_x = tgt_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        tgt_y = tgt_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        
        bounce_boost_available = False
        compression_state = 0.0
        
        for entity in entities:
            if entity.get('type') == 17:  # Bounce block
                entity_x = entity.get('x', 0.0)
                entity_y = entity.get('y', 0.0)
                
                if self._path_intersects_bounce_block(src_x, src_y, tgt_x, tgt_y, entity_x, entity_y):
                    bounce_boost_available = True
                    compression_state = entity.get('compression', 0.0)
                    
                    # Modify trajectory parameters for bounce interaction
                    features[len(EdgeType) + 3] = 0.5  # Reduced time of flight
                    features[len(EdgeType) + 4] = 0.8  # Reduced energy cost
                    features[len(EdgeType) + 5] = 0.9  # High success probability
                    break
        
        # Movement requirements
        features[len(EdgeType) + 8] = 1.0 if bounce_boost_available else 0.0
        features[len(EdgeType) + 9] = compression_state
        
        return features
    
    def _build_entity_edges(
        self,
        entity_nodes: List[Tuple[int, Dict[str, Any]]],
        entities: List[Dict[str, Any]],
        ninja_position: Tuple[float, float],
        ninja_velocity: Optional[Tuple[float, float]],
        edge_index: np.ndarray,
        edge_features: np.ndarray,
        edge_mask: np.ndarray,
        edge_types: np.ndarray,
        edge_count: int
    ) -> int:
        """Build entity interaction edges."""
        
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
                            chain_features = np.zeros(self.edge_feature_dim, dtype=np.float32)
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
    
    def _entity_sort_key(self, entity: Dict[str, Any]) -> Tuple[int, float, float]:
        """Generate sort key for entity ordering."""
        entity_type = entity.get('type', 0)
        x = entity.get('x', 0.0)
        y = entity.get('y', 0.0)
        return (entity_type, x, y)