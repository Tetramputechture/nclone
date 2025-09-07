"""
Edge building utilities for graph construction.

This module provides specialized edge building methods with comprehensive
bounce block awareness and traversability checking.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from ..constants.entity_types import EntityType
from ..constants.physics_constants import TILE_PIXEL_SIZE
from ..graph.level_data import LevelData
from .common import SUB_CELL_SIZE, EdgeType, E_MAX_EDGES
from .feature_extraction import FeatureExtractor
from .precise_collision import PreciseTileCollision
from .hazard_system import HazardClassificationSystem


class EdgeBuilder:
    """
    Handles edge construction with precise collision detection and hazard awareness.

    This enhanced edge builder uses segment-based tile collision detection and
    comprehensive hazard classification to create accurate traversability graphs.
    """

    def __init__(self, feature_extractor: FeatureExtractor):
        """
        Initialize edge builder with precise collision and hazard systems.

        Args:
            feature_extractor: Feature extraction utility instance
        """
        self.feature_extractor = feature_extractor
        self.precise_collision = PreciseTileCollision()
        self.hazard_system = HazardClassificationSystem(self.precise_collision)

        # Cache for static hazards (rebuilt when level changes)
        self._static_hazard_cache = {}
        self._current_level_id = None

    def build_edges(
        self,
        sub_grid_node_map: Dict[Tuple[int, int], int],
        entity_nodes: List[Tuple[int, Dict[str, Any]]],
        level_data: LevelData,
        ninja_position: Tuple[float, float],
        ninja_velocity: Optional[Tuple[float, float]],
        ninja_state: Optional[int],
        edge_index: np.ndarray,
        edge_features: np.ndarray,
        edge_mask: np.ndarray,
        edge_types: np.ndarray,
        edge_count: int,
        edge_feature_dim: int,
    ) -> int:
        """
        Build edges with bounce block awareness.

        Args:
            sub_grid_node_map: Mapping from (row, col) to node indices
            entity_nodes: List of (node_idx, entity) tuples
            level_data: Level tile data and structure, including entities
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
        # Set tile data for hazard system collision checking
        self.hazard_system.set_tile_data(level_data.tiles)

        # Build sub-grid traversability edges
        for (sub_row, sub_col), src_idx in sub_grid_node_map.items():
            if edge_count >= E_MAX_EDGES - 8:  # Leave room for 8 directions
                break

            # Check 8-connected neighbors
            directions = [
                (0, 1),
                (1, 0),
                (0, -1),
                (-1, 0),  # Cardinal
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),  # Diagonal
            ]

            for dr, dc in directions:
                tgt_row = sub_row + dr
                tgt_col = sub_col + dc

                if (tgt_row, tgt_col) in sub_grid_node_map:
                    tgt_idx = sub_grid_node_map[(tgt_row, tgt_col)]

                    # Check traversability with precise collision and hazard awareness
                    if self.is_traversable_with_hazards(
                        sub_row,
                        sub_col,
                        tgt_row,
                        tgt_col,
                        level_data,
                        ninja_position,
                    ):
                        edge_index[0, edge_count] = src_idx
                        edge_index[1, edge_count] = tgt_idx

                        # Extract edge features
                        edge_features[edge_count] = (
                            self.feature_extractor.extract_edge_features(
                                sub_row,
                                sub_col,
                                tgt_row,
                                tgt_col,
                                level_data.entities,
                                ninja_position,
                                ninja_velocity,
                                edge_feature_dim,
                            )
                        )

                        edge_mask[edge_count] = 1.0
                        edge_types[edge_count] = EdgeType.WALK
                        edge_count += 1

        # No special ninja escape connections - all edges must follow proper collision detection

        # Build edges connecting sub-grid nodes to entity nodes
        for entity_node_idx, entity in entity_nodes:
            if edge_count >= E_MAX_EDGES - 10:
                break

            # Entity positions are already in full map coordinates (including border)
            entity_x = entity.get("x", 0.0)  # Already in full map coords
            entity_y = entity.get("y", 0.0)  # Already in full map coords

            # Check if entity is in a solid tile - if so, don't create walkable edges
            # Exception: Always create edges for ninja (type 0) even if in solid tile
            entity_type = entity.get("type", -1)
            entity_tile_x = int(entity_x // TILE_PIXEL_SIZE)
            entity_tile_y = int(entity_y // TILE_PIXEL_SIZE)
            
            entity_in_solid_tile = False
            if (0 <= entity_tile_y < level_data.height and 0 <= entity_tile_x < level_data.width):
                tile_value = level_data.get_tile(entity_tile_y, entity_tile_x)
                if tile_value == 1:  # Solid tile
                    entity_in_solid_tile = True

            # Skip creating walkable edges for entities in solid tiles, except for ninja
            if entity_in_solid_tile and entity_type != EntityType.NINJA:
                continue

            # Find nearby sub-grid nodes
            entity_sub_col = int(entity_x // SUB_CELL_SIZE)
            entity_sub_row = int(entity_y // SUB_CELL_SIZE)

            # Connect to nearby sub-grid nodes (3x3 area around entity)
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nearby_row = entity_sub_row + dr
                    nearby_col = entity_sub_col + dc

                    if (nearby_row, nearby_col) in sub_grid_node_map:
                        grid_node_idx = sub_grid_node_map[(nearby_row, nearby_col)]

                        # Calculate distance
                        grid_x = nearby_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
                        grid_y = nearby_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
                        distance = math.sqrt(
                            (entity_x - grid_x) ** 2 + (entity_y - grid_y) ** 2
                        )

                        # Only connect if close enough (within 1.5 sub-cells)
                        if distance <= SUB_CELL_SIZE * 1.5:
                            # Create bidirectional edges
                            for src, dst in [
                                (grid_node_idx, entity_node_idx),
                                (entity_node_idx, grid_node_idx),
                            ]:
                                if edge_count >= E_MAX_EDGES:
                                    break

                                edge_index[0, edge_count] = src
                                edge_index[1, edge_count] = dst

                                # Simple walk edge for grid-entity connections
                                connect_features = np.zeros(
                                    edge_feature_dim, dtype=np.float32
                                )
                                connect_features[EdgeType.WALK] = 1.0
                                connect_features[len(EdgeType) + 2] = 0.5  # Medium cost

                                edge_features[edge_count] = connect_features
                                edge_mask[edge_count] = 1.0
                                edge_types[edge_count] = EdgeType.WALK
                                edge_count += 1

        # Build entity interaction edges
        edge_count = self.build_entity_edges(
            entity_nodes,
            level_data.entities,
            ninja_position,
            ninja_velocity,
            edge_index,
            edge_features,
            edge_mask,
            edge_types,
            edge_count,
            edge_feature_dim,
        )

        # Build long-distance corridor connections between empty tile clusters
        edge_count = self.build_corridor_connections(
            sub_grid_node_map,
            level_data,
            ninja_position,
            edge_index,
            edge_features,
            edge_mask,
            edge_types,
            edge_count,
            edge_feature_dim,
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
        edge_feature_dim: int,
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
        # Build entity interaction edges (bounce blocks, switches/doors, etc.)
        for i, (node_idx, entity) in enumerate(entity_nodes):
            if edge_count >= E_MAX_EDGES - 10:  # Leave room for more edges
                break

            entity_type = entity.get("type")

            # Handle switch-door functional edges
            if entity_type in [
                EntityType.EXIT_SWITCH,
                EntityType.LOCKED_DOOR,
                EntityType.TRAP_DOOR,
            ]:
                # Look for corresponding door
                for j, (other_idx, other_entity) in enumerate(entity_nodes):
                    if i == j:
                        continue

                    # Check if this is a matching door
                    is_matching = False
                    if (
                        entity_type == EntityType.EXIT_SWITCH
                        and other_entity.get("type") == EntityType.EXIT_DOOR
                    ):
                        # Match exit switches to exit doors by entity_id
                        if entity.get("entity_id") == other_entity.get(
                            "switch_entity_id"
                        ):
                            is_matching = True
                    elif entity_type == EntityType.LOCKED_DOOR:
                        # For LOCKED_DOOR, match switch node to door node of same entity
                        if (entity.get("entity_id") == other_entity.get("entity_id") and
                            not entity.get("is_door_part", False) and
                            other_entity.get("is_door_part", False)):
                            is_matching = True
                    elif entity_type == EntityType.TRAP_DOOR:
                        # For TRAP_DOOR, match switch node to door node of same entity
                        if (entity.get("entity_id") == other_entity.get("entity_id") and
                            not entity.get("is_door_part", False) and
                            other_entity.get("is_door_part", False)):
                            is_matching = True

                    if is_matching:
                        # Create functional edge from switch to door
                        edge_index[0, edge_count] = node_idx
                        edge_index[1, edge_count] = other_idx

                        func_features = np.zeros(edge_feature_dim, dtype=np.float32)
                        func_features[EdgeType.FUNCTIONAL] = 1.0  # Functional edge type

                        # Calculate direction
                        entity_x = entity.get("x", 0.0)
                        entity_y = entity.get("y", 0.0)
                        other_x = other_entity.get("x", 0.0)
                        other_y = other_entity.get("y", 0.0)
                        distance = math.sqrt(
                            (other_x - entity_x) ** 2 + (other_y - entity_y) ** 2
                        )

                        if distance > 0:
                            func_features[len(EdgeType)] = (
                                other_x - entity_x
                            ) / distance
                            func_features[len(EdgeType) + 1] = (
                                other_y - entity_y
                            ) / distance

                        # Set functional edge parameters
                        func_features[len(EdgeType) + 2] = (
                            0.1  # Very low traversability cost
                        )
                        func_features[len(EdgeType) + 3] = 0.0  # Instant activation
                        func_features[len(EdgeType) + 4] = 0.1  # Low energy cost
                        func_features[len(EdgeType) + 5] = 1.0  # Always succeeds

                        edge_features[edge_count] = func_features
                        edge_mask[edge_count] = 1.0
                        edge_types[edge_count] = EdgeType.FUNCTIONAL
                        edge_count += 1

                        if edge_count >= E_MAX_EDGES - 10:
                            break

            elif entity.get("type") == 17:  # Bounce block
                # Create edges to nearby bounce blocks for chaining
                for j, (other_idx, other_entity) in enumerate(entity_nodes):
                    if i != j and other_entity.get("type") == 17:
                        entity_x = entity.get("x", 0.0)
                        entity_y = entity.get("y", 0.0)
                        other_x = other_entity.get("x", 0.0)
                        other_y = other_entity.get("y", 0.0)

                        distance = math.sqrt(
                            (entity_x - other_x) ** 2 + (entity_y - other_y) ** 2
                        )

                        # Connect nearby bounce blocks for chaining mechanics
                        if distance < 50.0:  # Within chaining range
                            edge_index[0, edge_count] = node_idx
                            edge_index[1, edge_count] = other_idx

                            # Create bounce chain edge features
                            chain_features = np.zeros(
                                edge_feature_dim, dtype=np.float32
                            )
                            chain_features[EdgeType.FUNCTIONAL] = (
                                1.0  # Functional edge type
                            )

                            # Direction
                            if distance > 0:
                                chain_features[len(EdgeType)] = (
                                    other_x - entity_x
                                ) / distance
                                chain_features[len(EdgeType) + 1] = (
                                    other_y - entity_y
                                ) / distance

                            # Chain interaction parameters
                            chain_features[len(EdgeType) + 2] = (
                                0.5  # Low traversability cost
                            )
                            chain_features[len(EdgeType) + 3] = (
                                0.3  # Fast time of flight
                            )
                            chain_features[len(EdgeType) + 4] = (
                                0.6  # Moderate energy cost
                            )
                            chain_features[len(EdgeType) + 5] = (
                                0.85  # High success probability
                            )
                            chain_features[len(EdgeType) + 8] = (
                                1.0  # Bounce boost available
                            )

                            edge_features[edge_count] = chain_features
                            edge_mask[edge_count] = 1.0
                            edge_types[edge_count] = EdgeType.FUNCTIONAL
                            edge_count += 1

        return edge_count

    def is_traversable_with_hazards(
        self,
        src_row: int,
        src_col: int,
        tgt_row: int,
        tgt_col: int,
        level_data: LevelData,
        ninja_position: Tuple[float, float],
    ) -> bool:
        """
        Check if movement between sub-cells is traversable with hazard awareness.

        This method uses precise tile collision detection and comprehensive hazard
        classification to determine path traversability.

        Args:
            src_row: Source sub-cell row
            src_col: Source sub-cell column
            tgt_row: Target sub-cell row
            tgt_col: Target sub-cell column
            level_data: Level tile data and structure, including entities
            ninja_position: Current ninja position for dynamic hazard range

        Returns:
            True if movement is traversable
        """
        # Convert to pixel coordinates
        src_x = src_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        src_y = src_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        tgt_x = tgt_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        tgt_y = tgt_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2

        # No special ninja escape logic - all movement must follow proper collision detection
        # The ninja should only be able to move between traversable positions

        # Check precise tile collision first
        if not self.is_precise_traversable(
            src_x, src_y, tgt_x, tgt_y, level_data.tiles
        ):
            return False

        # Check static hazards
        if not self._check_static_hazards(src_x, src_y, tgt_x, tgt_y, level_data):
            return False

        # Check dynamic hazards within range
        if not self._check_dynamic_hazards(
            src_x, src_y, tgt_x, tgt_y, level_data.entities, ninja_position
        ):
            return False

        # Check for bounce blocks that may block traversal in narrow passages
        if not self._check_bounce_block_traversal(
            src_x, src_y, tgt_x, tgt_y, level_data.entities
        ):
            return False

        return True

    def is_precise_traversable(
        self, src_x: float, src_y: float, tgt_x: float, tgt_y: float, tiles: np.ndarray
    ) -> bool:
        """
        Check precise tile traversability using segment-based collision detection.

        This method replaces the simplified boolean tile checking with accurate
        physics-based collision testing against tile geometry segments.

        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            tiles: Level tile data as NumPy array

        Returns:
            True if path is traversable through tile geometry
        """
        return self.precise_collision.is_path_traversable(
            src_x, src_y, tgt_x, tgt_y, tiles
        )

    def _check_static_hazards(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        level_data: LevelData,
    ) -> bool:
        """
        Check if path is blocked by static hazards.

        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            level_data: Level data and structure, including tiles and entities

        Returns:
            True if path is safe from static hazards
        """
        # Build/update static hazard cache
        level_id = id(level_data)
        if self._current_level_id != level_id:
            self._static_hazard_cache = self.hazard_system.build_static_hazard_cache(
                level_data
            )
            self._current_level_id = level_id

        # Check path against cached static hazards
        src_sub_x = int(src_x // 12)  # Sub-cell coordinates
        src_sub_y = int(src_y // 12)
        tgt_sub_x = int(tgt_x // 12)
        tgt_sub_y = int(tgt_y // 12)

        # Check all sub-cells along the path
        steps = max(abs(tgt_sub_x - src_sub_x), abs(tgt_sub_y - src_sub_y))
        if steps == 0:
            steps = 1

        for i in range(steps + 1):
            t = i / steps
            check_sub_x = int(src_sub_x + t * (tgt_sub_x - src_sub_x))
            check_sub_y = int(src_sub_y + t * (tgt_sub_y - src_sub_y))

            if (check_sub_x, check_sub_y) in self._static_hazard_cache:
                hazard_info = self._static_hazard_cache[(check_sub_x, check_sub_y)]
                if self.hazard_system.check_path_hazard_intersection(
                    src_x, src_y, tgt_x, tgt_y, hazard_info
                ):
                    return False  # Path blocked by static hazard

        return True  # Path is safe from static hazards

    def _check_dynamic_hazards(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        entities: List[Dict[str, Any]],
        ninja_position: Tuple[float, float],
    ) -> bool:
        """
        Check if path is blocked by dynamic hazards within range.

        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            entities: List of entity dictionaries
            ninja_position: Current ninja position for range calculation

        Returns:
            True if path is safe from dynamic hazards
        """
        # Get dynamic hazards within range
        dynamic_hazards = self.hazard_system.get_dynamic_hazards_in_range(
            entities, ninja_position
        )

        # Check path against each dynamic hazard
        for hazard_info in dynamic_hazards:
            if self.hazard_system.check_path_hazard_intersection(
                src_x, src_y, tgt_x, tgt_y, hazard_info
            ):
                return False  # Path blocked by dynamic hazard

        return True  # Path is safe from dynamic hazards

    def _check_bounce_block_traversal(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        entities: List[Dict[str, Any]],
    ) -> bool:
        """
        Check if bounce blocks block traversal in narrow passages.

        Bounce blocks can block traversal if they're positioned in the center
        of a one-tile (24px) path and the ninja cannot displace them enough
        horizontally or vertically to get clearance.

        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            entities: List of entity dictionaries

        Returns:
            True if path is traversable (not blocked by bounce blocks)
        """
        for entity in entities:
            if entity.get("type") == EntityType.BOUNCE_BLOCK:
                # Use hazard system's bounce block traversal analysis
                if self.hazard_system.analyze_bounce_block_traversal_blocking(
                    entity, entities, (src_x, src_y), (tgt_x, tgt_y)
                ):
                    return False  # Bounce block blocks this path

        return True  # No bounce blocks block the path

    def _check_bounce_block_interactions(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        entities: List[Dict[str, Any]],
    ) -> bool:
        """
        Check for bounce block interactions along the path for feature extraction.

        This method identifies potential bounce interactions that affect movement
        but don't necessarily block traversal.

        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            entities: List of entity dictionaries

        Returns:
            True if bounce blocks detected (for feature extraction)
        """
        bounce_detected = False

        for entity in entities:
            if entity.get("type") == EntityType.BOUNCE_BLOCK:
                entity_x = entity.get("x", 0.0)
                entity_y = entity.get("y", 0.0)

                # Check if path intersects with bounce block
                if self.feature_extractor._path_intersects_bounce_block(
                    src_x, src_y, tgt_x, tgt_y, entity_x, entity_y
                ):
                    bounce_detected = True
                    # Note: This is for feature extraction, not traversal blocking

        return bounce_detected

    def build_corridor_connections(
        self,
        sub_grid_node_map: Dict[Tuple[int, int], int],
        level_data: LevelData,
        ninja_position: Tuple[float, float],
        edge_index: np.ndarray,
        edge_features: np.ndarray,
        edge_mask: np.ndarray,
        edge_types: np.ndarray,
        edge_count: int,
        edge_feature_dim: int,
    ) -> int:
        """
        Build long-distance corridor connections between empty tile clusters.
        
        This method identifies separate empty tile clusters and creates bridge
        connections between them when they are separated by narrow gaps or
        single solid tiles, enabling long-distance pathfinding.
        
        Args:
            sub_grid_node_map: Mapping from (row, col) to node indices
            level_data: Level tile data and structure
            ninja_position: Current ninja position (x, y)
            edge_index: Edge connectivity array
            edge_features: Edge feature array
            edge_mask: Edge mask array
            edge_types: Edge type array
            edge_count: Current edge count
            edge_feature_dim: Edge feature dimension
            
        Returns:
            Updated edge count
        """
        # Find empty tile clusters
        visited_tiles = set()
        empty_clusters = []
        
        for y in range(level_data.height):
            for x in range(level_data.width):
                if (x, y) in visited_tiles or level_data.get_tile(y, x) != 0:
                    continue
                
                # Found new empty tile cluster
                cluster = []
                stack = [(x, y)]
                
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) in visited_tiles:
                        continue
                    
                    visited_tiles.add((cx, cy))
                    cluster.append((cx, cy))
                    
                    # Check 4-connected neighbors
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < level_data.width and 0 <= ny < level_data.height and
                            (nx, ny) not in visited_tiles and level_data.get_tile(ny, nx) == 0):
                            stack.append((nx, ny))
                
                empty_clusters.append(cluster)
        
        # Sort clusters by size (largest first)
        empty_clusters.sort(key=len, reverse=True)
        
        # Only process if we have multiple clusters
        if len(empty_clusters) < 2:
            return edge_count
        
        # Create corridor connections between nearby clusters
        max_corridor_distance = 120.0  # Maximum distance for corridor connections (5 tiles)
        
        for i in range(len(empty_clusters)):
            for j in range(i + 1, len(empty_clusters)):
                if edge_count >= E_MAX_EDGES - 10:
                    break
                
                cluster1 = empty_clusters[i]
                cluster2 = empty_clusters[j]
                
                # Find closest points between clusters
                min_distance = float('inf')
                best_connection = None
                
                for tile1_x, tile1_y in cluster1:
                    for tile2_x, tile2_y in cluster2:
                        # Calculate distance between tile centers
                        center1_x = tile1_x * 24 + 12
                        center1_y = tile1_y * 24 + 12
                        center2_x = tile2_x * 24 + 12
                        center2_y = tile2_y * 24 + 12
                        
                        distance = math.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
                        
                        if distance < min_distance and distance <= max_corridor_distance:
                            # Check if there's a potential corridor (not too many blocking tiles)
                            blocking_tiles = self._count_blocking_tiles(
                                center1_x, center1_y, center2_x, center2_y, level_data
                            )
                            
                            # Allow corridors with up to 2 blocking tiles (narrow passages)
                            if blocking_tiles <= 2:
                                min_distance = distance
                                best_connection = (
                                    (tile1_x, tile1_y, center1_x, center1_y),
                                    (tile2_x, tile2_y, center2_x, center2_y),
                                    distance
                                )
                
                # Create corridor connection if found
                if best_connection:
                    (tile1_x, tile1_y, center1_x, center1_y), (tile2_x, tile2_y, center2_x, center2_y), distance = best_connection
                    
                    # Find nodes in these tiles
                    nodes1 = self._find_nodes_in_tile(tile1_x, tile1_y, sub_grid_node_map)
                    nodes2 = self._find_nodes_in_tile(tile2_x, tile2_y, sub_grid_node_map)
                    
                    # Create connections between closest nodes
                    for node1_idx, (node1_row, node1_col) in nodes1:
                        for node2_idx, (node2_row, node2_col) in nodes2:
                            if edge_count >= E_MAX_EDGES:
                                break
                            
                            # Create bidirectional corridor edge
                            for src_idx, dst_idx in [(node1_idx, node2_idx), (node2_idx, node1_idx)]:
                                if edge_count >= E_MAX_EDGES:
                                    break
                                
                                edge_index[0, edge_count] = src_idx
                                edge_index[1, edge_count] = dst_idx
                                
                                # Create corridor edge features
                                corridor_features = np.zeros(edge_feature_dim, dtype=np.float32)
                                corridor_features[EdgeType.WALK] = 1.0  # Walk edge type
                                
                                # Higher cost for corridor connections (they're longer/harder)
                                corridor_features[len(EdgeType) + 2] = min(2.0, distance / 24.0)  # Cost based on distance
                                
                                edge_features[edge_count] = corridor_features
                                edge_mask[edge_count] = 1.0
                                edge_types[edge_count] = EdgeType.WALK
                                edge_count += 1
                            
                            # Only connect the closest pair to avoid too many edges
                            break
                        if edge_count >= E_MAX_EDGES:
                            break
        
        return edge_count
    
    def _count_blocking_tiles(
        self, 
        x1: float, 
        y1: float, 
        x2: float, 
        y2: float, 
        level_data: LevelData
    ) -> int:
        """Count solid tiles that block the path between two points."""
        # Sample points along the line
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        num_samples = max(3, int(distance / 12))  # Sample every 12 pixels
        
        blocking_tiles = set()
        
        for i in range(1, num_samples):  # Skip endpoints
            t = i / num_samples
            sample_x = x1 + t * (x2 - x1)
            sample_y = y1 + t * (y2 - y1)
            
            # Check tile at this position
            tile_x = int(sample_x // 24)
            tile_y = int(sample_y // 24)
            
            if (0 <= tile_x < level_data.width and 0 <= tile_y < level_data.height):
                tile_value = level_data.get_tile(tile_y, tile_x)
                if tile_value == 1:  # Solid tile
                    blocking_tiles.add((tile_x, tile_y))
        
        return len(blocking_tiles)
    
    def _find_nodes_in_tile(
        self, 
        tile_x: int, 
        tile_y: int, 
        sub_grid_node_map: Dict[Tuple[int, int], int]
    ) -> List[Tuple[int, Tuple[int, int]]]:
        """Find all sub-grid nodes within a specific tile."""
        nodes_in_tile = []
        
        # Calculate sub-grid range for this tile
        tile_left = tile_x * 24
        tile_right = (tile_x + 1) * 24
        tile_top = tile_y * 24
        tile_bottom = (tile_y + 1) * 24
        
        for (sub_row, sub_col), node_idx in sub_grid_node_map.items():
            # Calculate node position
            node_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            node_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            
            # Check if node is in this tile
            if tile_left <= node_x < tile_right and tile_top <= node_y < tile_bottom:
                nodes_in_tile.append((node_idx, (sub_row, sub_col)))
        
        return nodes_in_tile
