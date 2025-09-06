"""
Feature extraction utilities for graph nodes and edges.

This module provides specialized feature extraction methods for different
types of graph nodes and edges, with comprehensive bounce block support.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from ..constants.physics_constants import (
    MAP_TILE_WIDTH,
    MAP_TILE_HEIGHT,
    TILE_PIXEL_SIZE,
    FULL_MAP_WIDTH_PX,
    FULL_MAP_HEIGHT_PX,
)
from .level_data import LevelData
from .common import SUB_CELL_SIZE, EdgeType


class FeatureExtractor:
    """
    Handles feature extraction for graph nodes and edges with bounce block awareness.
    """

    def __init__(self, tile_type_dim: int = 64, entity_type_dim: int = 32):
        """
        Initialize feature extractor.

        Args:
            tile_type_dim: Number of tile types for one-hot encoding
            entity_type_dim: Number of entity types for one-hot encoding
        """
        self.tile_type_dim = tile_type_dim
        self.entity_type_dim = entity_type_dim

    def extract_sub_cell_features(
        self,
        level_data: LevelData,
        sub_row: int,
        sub_col: int,
        ninja_position: Tuple[float, float],
        ninja_velocity: Optional[Tuple[float, float]],
        ninja_state: Optional[int],
        feature_dim: int,
    ) -> np.ndarray:
        """
        Extract features for a sub-cell node.

        Args:
            level_data: Level data containing tiles and entities
            sub_row: Sub-cell row index
            sub_col: Sub-cell column index
            ninja_position: Current ninja position (x, y)
            ninja_velocity: Current ninja velocity (vx, vy)
            ninja_state: Current ninja movement state (0-8)
            feature_dim: Total feature dimension

        Returns:
            Feature vector for the sub-cell node
        """
        features = np.zeros(feature_dim, dtype=np.float32)

        # Convert sub-cell to pixel coordinates
        pixel_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        pixel_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2

        # Convert to tile coordinates
        tile_x = pixel_x // TILE_PIXEL_SIZE
        tile_y = pixel_y // TILE_PIXEL_SIZE

        # Use tiles array directly and get its dimensions for bounds checking
        tiles_array = (
            level_data.tiles
            if level_data.tiles is not None
            else np.zeros((MAP_TILE_HEIGHT, MAP_TILE_WIDTH), dtype=np.int32)
        )
        map_height, map_width = tiles_array.shape

        # Extract tile features if within bounds
        if 0 <= tile_x < map_width and 0 <= tile_y < map_height:
            tile_type = tiles_array[tile_y, tile_x]
            if 0 <= tile_type < self.tile_type_dim:
                features[tile_type] = 1.0

        # Add ninja position flag
        ninja_x, ninja_y = ninja_position
        ninja_distance = math.sqrt((pixel_x - ninja_x) ** 2 + (pixel_y - ninja_y) ** 2)
        features[self.tile_type_dim + 4 + self.entity_type_dim + 8] = (
            1.0 if ninja_distance < SUB_CELL_SIZE else 0.0
        )

        # Add physics state features
        if ninja_velocity is not None:
            vx, vy = ninja_velocity
            # Normalize velocity features
            max_speed = 200.0  # Approximate max speed
            features[self.tile_type_dim + 4 + self.entity_type_dim + 8 + 1] = (
                vx / max_speed
            )
            features[self.tile_type_dim + 4 + self.entity_type_dim + 8 + 2] = (
                vy / max_speed
            )
            features[self.tile_type_dim + 4 + self.entity_type_dim + 8 + 3] = (
                math.sqrt(vx * vx + vy * vy) / max_speed
            )

        if ninja_state is not None:
            features[self.tile_type_dim + 4 + self.entity_type_dim + 8 + 4] = (
                ninja_state / 8.0
            )

        return features

    def extract_entity_features(
        self,
        entity: Dict[str, Any],
        ninja_position: Tuple[float, float],
        ninja_velocity: Optional[Tuple[float, float]],
        feature_dim: int,
    ) -> np.ndarray:
        """
        Extract features for an entity node.

        Args:
            entity: Entity dictionary with type, position, state
            ninja_position: Current ninja position (x, y)
            ninja_velocity: Current ninja velocity (vx, vy)
            feature_dim: Total feature dimension

        Returns:
            Feature vector for the entity node
        """
        features = np.zeros(feature_dim, dtype=np.float32)

        # Entity type one-hot encoding
        entity_type = entity.get("type", 0)
        if 0 <= entity_type < self.entity_type_dim:
            features[self.tile_type_dim + 4 + entity_type] = 1.0

        # Entity state features
        state_offset = self.tile_type_dim + 4 + self.entity_type_dim
        features[state_offset] = 1.0  # Active flag
        # Entity positions from simulator already include the 1-tile border offset
        # They are in the full 44x25 map coordinate system (0-1056, 0-600)
        entity_x = entity.get("x", 0.0)  # Already in full map coords
        entity_y = entity.get("y", 0.0)  # Already in full map coords
        features[state_offset + 1] = entity_x / float(
            FULL_MAP_WIDTH_PX
        )  # Normalized position (1056)
        features[state_offset + 2] = entity_y / float(
            FULL_MAP_HEIGHT_PX
        )  # Normalized position (600)
        features[state_offset + 3] = entity.get("state", 0.0) / 10.0  # Normalized state

        # Bounce block specific features
        if entity_type == 17:  # Bounce block
            features[state_offset + 4] = (
                entity.get("bounce_state", 0.0) / 3.0
            )  # Bounce state (0-3)
            features[state_offset + 5] = entity.get(
                "compression", 0.0
            )  # Compression ratio (0-1)
            features[state_offset + 6] = (
                entity.get("velocity_x", 0.0) / 100.0
            )  # Normalized velocity
            features[state_offset + 7] = entity.get("velocity_y", 0.0) / 100.0

        return features

    def extract_edge_features(
        self,
        src_row: int,
        src_col: int,
        tgt_row: int,
        tgt_col: int,
        entities: List[Dict[str, Any]],
        ninja_position: Tuple[float, float],
        ninja_velocity: Optional[Tuple[float, float]],
        feature_dim: int,
    ) -> np.ndarray:
        """
        Extract features for an edge with bounce block awareness.

        Args:
            src_row: Source sub-cell row
            src_col: Source sub-cell column
            tgt_row: Target sub-cell row
            tgt_col: Target sub-cell column
            entities: List of entity dictionaries
            ninja_position: Current ninja position (x, y)
            ninja_velocity: Current ninja velocity (vx, vy)
            feature_dim: Total feature dimension

        Returns:
            Feature vector for the edge
        """
        features = np.zeros(feature_dim, dtype=np.float32)

        # Direction vector
        dx = tgt_col - src_col
        dy = tgt_row - src_row
        distance = math.sqrt(dx * dx + dy * dy)

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
            if entity.get("type") == 17:  # Bounce block
                entity_x = entity.get("x", 0.0)
                entity_y = entity.get("y", 0.0)

                if self._path_intersects_bounce_block(
                    src_x, src_y, tgt_x, tgt_y, entity_x, entity_y
                ):
                    bounce_boost_available = True
                    compression_state = entity.get("compression", 0.0)

                    # Modify trajectory parameters for bounce interaction
                    features[len(EdgeType) + 3] = 0.5  # Reduced time of flight
                    features[len(EdgeType) + 4] = 0.8  # Reduced energy cost
                    features[len(EdgeType) + 5] = 0.9  # High success probability
                    break

        # Movement requirements
        features[len(EdgeType) + 8] = 1.0 if bounce_boost_available else 0.0
        features[len(EdgeType) + 9] = compression_state

        return features

    def _path_intersects_bounce_block(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        block_x: float,
        block_y: float,
    ) -> bool:
        """
        Check if path intersects with a bounce block.

        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            block_x: Bounce block x coordinate
            block_y: Bounce block y coordinate

        Returns:
            True if path intersects with bounce block
        """
        # Simple distance check for now
        block_radius = 9.0  # Bounce block semi-side

        # Check if either endpoint is within bounce block
        src_dist = math.sqrt((src_x - block_x) ** 2 + (src_y - block_y) ** 2)
        tgt_dist = math.sqrt((tgt_x - block_x) ** 2 + (tgt_y - block_y) ** 2)

        return src_dist <= block_radius or tgt_dist <= block_radius
