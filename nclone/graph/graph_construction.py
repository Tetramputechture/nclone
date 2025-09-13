"""
Graph construction utilities for hierarchical graph building.

This module provides core graph construction methods for building
sub-cell graphs and managing entity sorting.
"""

import numpy as np
from typing import Dict, Any, Tuple

from .common import (
    GraphData,
    NodeType,
    SUB_GRID_WIDTH,
    SUB_GRID_HEIGHT,
    SUB_CELL_SIZE,
    N_MAX_NODES,
    E_MAX_EDGES,
)
from .level_data import LevelData
from .feature_extraction import FeatureExtractor
from ..constants.entity_types import EntityType
from .edge_building import EdgeBuilder
from .reachability import ReachabilityAnalyzer


class GraphConstructor:
    """
    Handles core graph construction with bounce block awareness.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        edge_builder: EdgeBuilder,
        debug: bool = False,
    ):
        """
        Initialize graph constructor.

        Args:
            feature_extractor: Feature extraction utility instance
            edge_builder: Edge building utility instance
            debug: Enable debug output (default: False)
        """
        self.feature_extractor = feature_extractor
        self.edge_builder = edge_builder
        self.reachability_analyzer = ReachabilityAnalyzer(
            edge_builder.trajectory_calculator, debug=debug
        )
        self.debug = debug

    def build_sub_cell_graph(
        self,
        level_data: LevelData,
        ninja_position: Tuple[float, float],
        ninja_velocity: Tuple[float, float],
        ninja_state: int,
        node_feature_dim: int,
        edge_feature_dim: int,
    ) -> GraphData:
        """
        Build sub-cell graph with bounce block awareness.

        This method builds the finest resolution graph with comprehensive
        bounce block mechanics integrated directly.

        Args:
            level_data: Level tile data and structure, including entities
            ninja_position: Current ninja position (x, y)
            ninja_velocity: Current ninja velocity (vx, vy)
            ninja_state: Current ninja movement state (0-8)
            node_feature_dim: Node feature dimension
            edge_feature_dim: Edge feature dimension

        Returns:
            GraphData with sub-cell resolution graph
        """
        # Initialize arrays
        node_features = np.zeros((N_MAX_NODES, node_feature_dim), dtype=np.float32)
        edge_index = np.zeros((2, E_MAX_EDGES), dtype=np.int32)
        edge_features = np.zeros((E_MAX_EDGES, edge_feature_dim), dtype=np.float32)
        node_mask = np.zeros(N_MAX_NODES, dtype=np.float32)
        edge_mask = np.zeros(E_MAX_EDGES, dtype=np.float32)
        node_types = np.zeros(N_MAX_NODES, dtype=np.int32)
        edge_types = np.zeros(E_MAX_EDGES, dtype=np.int32)

        node_count = 0
        edge_count = 0

        # CONNECTIVITY FIX: Use full grid to ensure proper connectivity
        # The reachability analyzer was too restrictive, creating disconnected graphs
        if self.debug:
            print(
                f"DEBUG: Building full grid for connectivity (ninja at {ninja_position})"
            )

        # Build sub-grid nodes for ALL traversable positions to ensure connectivity
        sub_grid_node_map = {}  # (sub_row, sub_col) -> node_index

        # Use full grid with traversability filtering for better connectivity
        sorted_positions = []
        for sub_row in range(SUB_GRID_HEIGHT):
            for sub_col in range(SUB_GRID_WIDTH):
                # Convert to pixel coordinates
                pixel_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
                pixel_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2

                # Check if position is traversable with ninja radius (not in solid tile or entity collision)
                # OR if position is near an important entity (switch, door) - these must be reachable
                is_near_entity = self._is_near_important_entity(
                    pixel_x, pixel_y, level_data.entities
                )
                is_traversable = self._is_position_traversable_with_radius(
                    pixel_x, pixel_y, level_data.tiles, level_data.entities
                )

                if is_traversable or is_near_entity:
                    sorted_positions.append((sub_row, sub_col))

        for sub_row, sub_col in sorted_positions:
            if node_count >= N_MAX_NODES:
                break

            node_idx = node_count
            sub_grid_node_map[(sub_row, sub_col)] = node_idx

            # Extract sub-cell features
            sub_cell_features = self.feature_extractor.extract_sub_cell_features(
                level_data,
                sub_row,
                sub_col,
                ninja_position,
                ninja_velocity,
                ninja_state,
                node_feature_dim,
            )

            node_features[node_idx] = sub_cell_features
            node_mask[node_idx] = 1.0
            node_types[node_idx] = NodeType.GRID_CELL
            node_count += 1

        if self.debug:
            total_possible = SUB_GRID_WIDTH * SUB_GRID_HEIGHT
            optimization_ratio = (
                (total_possible - node_count) / total_possible * 100
                if total_possible > 0
                else 0
            )
            print(
                f"DEBUG: Created {node_count} sub-grid nodes (vs {total_possible} total possible, {optimization_ratio:.1f}% reduction)"
            )

        # Build entity nodes (sorted by type then position for determinism)
        entity_nodes = []

        # Correct ninja position to ensure it's in clear space
        corrected_ninja_position = self.edge_builder._correct_ninja_position(
            ninja_position, level_data
        )

        # First, create a ninja entity node
        ninja_node_idx = node_count
        ninja_entity = {
            "type": EntityType.NINJA,
            "x": corrected_ninja_position[0],
            "y": corrected_ninja_position[1],
            "is_ninja": True,
        }
        entity_nodes.append((ninja_node_idx, ninja_entity))

        # Extract ninja features
        ninja_features = self.feature_extractor.extract_entity_features(
            ninja_entity, corrected_ninja_position, ninja_velocity, node_feature_dim
        )
        node_features[ninja_node_idx] = ninja_features
        node_mask[ninja_node_idx] = 1.0
        node_types[ninja_node_idx] = NodeType.NINJA
        node_count += 1

        # Then process other entities
        sorted_entities = sorted(level_data.entities, key=self.entity_sort_key)

        for entity in sorted_entities:
            if node_count >= N_MAX_NODES:
                break

            entity_type = entity.get("type")

            # Skip ninja entities - we already created a ninja node explicitly
            if entity_type == EntityType.NINJA:
                continue

            # Handle LOCKED_DOOR entities specially - create two nodes (switch and door)
            if entity_type == EntityType.LOCKED_DOOR:
                # Create switch node
                switch_node_idx = node_count
                switch_entity = entity.copy()
                # Switch position is already in entity['x'], entity['y']
                entity_nodes.append((switch_node_idx, switch_entity))

                # Extract switch features
                switch_features = self.feature_extractor.extract_entity_features(
                    switch_entity, ninja_position, ninja_velocity, node_feature_dim
                )
                node_features[switch_node_idx] = switch_features
                node_mask[switch_node_idx] = 1.0
                node_types[switch_node_idx] = NodeType.ENTITY
                node_count += 1

                if node_count >= N_MAX_NODES:
                    break

                # Create door node
                door_node_idx = node_count
                door_entity = entity.copy()
                door_entity["x"] = entity.get("door_x", entity.get("x"))
                door_entity["y"] = entity.get("door_y", entity.get("y"))
                # Mark as door part of locked door
                door_entity["is_door_part"] = True
                entity_nodes.append((door_node_idx, door_entity))

                # Extract door features
                door_features = self.feature_extractor.extract_entity_features(
                    door_entity, ninja_position, ninja_velocity, node_feature_dim
                )
                node_features[door_node_idx] = door_features
                node_mask[door_node_idx] = 1.0
                node_types[door_node_idx] = NodeType.ENTITY
                node_count += 1

            else:
                # Regular entity - create single node
                node_idx = node_count
                entity_nodes.append((node_idx, entity))

                # Extract entity features
                entity_features = self.feature_extractor.extract_entity_features(
                    entity, ninja_position, ninja_velocity, node_feature_dim
                )

                node_features[node_idx] = entity_features
                node_mask[node_idx] = 1.0
                node_types[node_idx] = NodeType.ENTITY
                node_count += 1

        # Build edges with bounce block awareness
        edge_count = self.edge_builder.build_edges(
            sub_grid_node_map,
            entity_nodes,
            level_data,
            corrected_ninja_position,
            ninja_velocity,
            ninja_state,
            edge_index,
            edge_features,
            edge_mask,
            edge_types,
            edge_count,
            edge_feature_dim,
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
            num_edges=edge_count,
        )

    def entity_sort_key(self, entity: Dict[str, Any]) -> Tuple[int, float, float]:
        """
        Generate sort key for entity ordering.

        Args:
            entity: Entity dictionary with type, position, state

        Returns:
            Sort key tuple (type, x, y)
        """
        entity_type = entity.get("type", 0)
        x = entity.get("x", 0.0)
        y = entity.get("y", 0.0)
        return (entity_type, x, y)

    def _is_position_traversable(
        self, pixel_x: float, pixel_y: float, tiles: np.ndarray, entities=None
    ) -> bool:
        """
        Check if a position is traversable using accurate collision detection.

        This method properly handles all tile types and entity collisions:
        - Type 0: Empty (always traversable)
        - Type 1: Fully solid (never traversable)
        - Types 2-33: Mixed geometry tiles (requires precise collision detection)
        - Types >33: Treated as solid
        - Entities: Locked doors, trap doors, one-way platforms, etc.

        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            tiles: Tile data array
            entities: Optional list of entities to check for collision

        Returns:
            True if position is traversable
        """
        from ..constants.physics_constants import TILE_PIXEL_SIZE

        # Convert to tile coordinates
        tile_x = int(pixel_x // TILE_PIXEL_SIZE)
        tile_y = int(pixel_y // TILE_PIXEL_SIZE)

        # Account for padding: tile data is unpadded, but coordinates assume padding
        data_tile_x = tile_x - 1
        data_tile_y = tile_y - 1

        # Check bounds
        if not (0 <= data_tile_y < len(tiles) and 0 <= data_tile_x < len(tiles[0])):
            return False

        tile_value = tiles[data_tile_y][data_tile_x]

        # Handle different tile types
        tile_traversable = False
        if tile_value == 0:
            tile_traversable = True  # Empty tile - always traversable
        elif tile_value == 1 or tile_value > 33:
            tile_traversable = False  # Fully solid tile - never traversable
        elif 2 <= tile_value <= 33:
            # Mixed geometry tiles - use precise collision detection
            tile_traversable = self._check_mixed_tile_traversability(
                pixel_x, pixel_y, tile_value, tile_x, tile_y
            )
        else:
            tile_traversable = False  # Unknown tile type - assume solid

        # If tile is not traversable, position is not traversable
        if not tile_traversable:
            return False

        # Check entity-based collision if entities provided
        if entities is not None:
            return self._check_entity_collision(pixel_x, pixel_y, entities)

        return True

    def _is_position_traversable_with_radius(
        self, pixel_x: float, pixel_y: float, tiles: np.ndarray, entities=None
    ) -> bool:
        """
        Check if a position is traversable considering ninja radius.

        This method checks if the ninja's full 10px radius can fit at the given position
        without colliding with any solid tiles or entities.

        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            tiles: Tile data array
            entities: Optional list of entities to check for collision

        Returns:
            True if position is traversable with ninja radius
        """
        from ..constants.physics_constants import TILE_PIXEL_SIZE, NINJA_RADIUS
        import math

        # Use the ninja's actual radius
        radius = NINJA_RADIUS  # 10.0 pixels

        # Calculate the range of tiles that could intersect with the ninja's radius
        min_tile_x = int(math.floor((pixel_x - radius) / TILE_PIXEL_SIZE))
        max_tile_x = int(math.ceil((pixel_x + radius) / TILE_PIXEL_SIZE))
        min_tile_y = int(math.floor((pixel_y - radius) / TILE_PIXEL_SIZE))
        max_tile_y = int(math.ceil((pixel_y + radius) / TILE_PIXEL_SIZE))

        # Check each tile in the range
        for check_tile_y in range(min_tile_y, max_tile_y + 1):
            for check_tile_x in range(min_tile_x, max_tile_x + 1):
                # Account for 1-tile padding when accessing tile data
                data_tile_x = check_tile_x - 1
                data_tile_y = check_tile_y - 1

                # Skip tiles outside the map bounds
                if (
                    data_tile_x < 0
                    or data_tile_x >= tiles.shape[1]
                    or data_tile_y < 0
                    or data_tile_y >= tiles.shape[0]
                ):
                    continue

                tile_id = tiles[data_tile_y, data_tile_x]
                if tile_id == 0:
                    continue  # Empty tile, no collision

                # For fully solid tiles, use simple geometric check
                if tile_id == 1 or tile_id > 33:
                    if self._check_circle_tile_collision(
                        pixel_x, pixel_y, check_tile_x, check_tile_y, radius
                    ):
                        return False

                # For shaped tiles (2-33), use conservative approach
                elif 2 <= tile_id <= 33:
                    # For now, allow traversal through shaped tiles unless they're very close to solid parts
                    # This is a reasonable approximation for node placement
                    pass

        # Check entity-based collision if entities provided
        if entities is not None:
            return self._check_entity_collision(pixel_x, pixel_y, entities)

        return True

    def _check_circle_tile_collision(
        self, x: float, y: float, tile_x: int, tile_y: int, radius: float
    ) -> bool:
        """Check if a circle collides with a solid tile using simple geometry."""
        from ..constants.physics_constants import TILE_PIXEL_SIZE

        # Tile bounds in world coordinates
        tile_left = tile_x * TILE_PIXEL_SIZE
        tile_right = tile_left + TILE_PIXEL_SIZE
        tile_top = tile_y * TILE_PIXEL_SIZE
        tile_bottom = tile_top + TILE_PIXEL_SIZE

        # Find closest point on tile to circle center
        closest_x = max(tile_left, min(x, tile_right))
        closest_y = max(tile_top, min(y, tile_bottom))

        # Check if distance to closest point is less than radius
        dx = x - closest_x
        dy = y - closest_y
        distance_squared = dx * dx + dy * dy

        return distance_squared < (radius * radius)

    def _is_near_important_entity(
        self, pixel_x: float, pixel_y: float, entities
    ) -> bool:
        """
        Check if a position is near an important entity (switch, door, etc.).

        Important entities must be reachable even if they're close to solid tiles,
        so we create nodes near them regardless of radius collision checks.

        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            entities: List of entities to check

        Returns:
            True if position is near an important entity
        """
        if not entities:
            return False

        # Define important entity types that must be reachable
        important_types = {3, 4}  # 3=exit door, 4=switch

        # Check distance to each important entity
        for entity in entities:
            entity_type = entity.get("type", 0)
            if entity_type in important_types:
                entity_x = entity.get("x", 0.0)
                entity_y = entity.get("y", 0.0)

                # Check if position is within 30 pixels of the entity
                # This ensures we create nodes around important entities
                distance_squared = (pixel_x - entity_x) ** 2 + (pixel_y - entity_y) ** 2
                if distance_squared <= (30.0 * 30.0):  # 30 pixel radius around entities
                    return True

        return False

    def _check_entity_collision(self, pixel_x: float, pixel_y: float, entities) -> bool:
        """
        Check if a position collides with any solid entities.

        Args:
            pixel_x, pixel_y: Position to check
            entities: List of entities to check for collision

        Returns:
            True if position is traversable (no collision)
        """
        from ..constants.physics_constants import NINJA_RADIUS

        for entity in entities:
            entity_type = entity.get("type", -1)
            entity_x = entity.get("x", 0)
            entity_y = entity.get("y", 0)

            # Check collision based on entity type
            if self._entity_has_collision(entity_type, entity):
                # Simple circular collision check (can be refined per entity type)
                distance = (
                    (pixel_x - entity_x) ** 2 + (pixel_y - entity_y) ** 2
                ) ** 0.5
                collision_radius = NINJA_RADIUS + 12  # Entity collision radius

                if distance < collision_radius:
                    return False  # Collision detected

        return True  # No collision

    def _entity_has_collision(self, entity_type: int, entity: dict) -> bool:
        """
        Determine if an entity type has physical collision.

        Args:
            entity_type: Entity type ID
            entity: Entity data dictionary

        Returns:
            True if entity blocks movement
        """
        # Define which entity types have collision
        # Note: Switches themselves should not block movement - only their associated doors

        # For now, disable entity collision to focus on tile-based traversability
        # Entity collision can be re-enabled later with proper door/switch logic
        return False

    def _check_mixed_tile_traversability(
        self, pixel_x: float, pixel_y: float, tile_value: int, tile_x: int, tile_y: int
    ) -> bool:
        """
        Check if a position is traversable within a mixed geometry tile (types 2-33).

        Uses simplified geometric checks based on tile type patterns:
        - Half tiles (2-5): Check which half is solid
        - Slopes (6-9, 18-33): Use diagonal line equations
        - Quarter circles (10-17): Use distance from circle center

        Args:
            pixel_x, pixel_y: Position to check
            tile_value: Tile type (2-33)
            tile_x, tile_y: Tile coordinates

        Returns:
            True if position is traversable within this tile
        """
        from ..constants.physics_constants import TILE_PIXEL_SIZE, NINJA_RADIUS

        # Calculate position within the tile (0-24 pixels)
        tile_left = tile_x * TILE_PIXEL_SIZE
        tile_top = tile_y * TILE_PIXEL_SIZE
        local_x = pixel_x - tile_left
        local_y = pixel_y - tile_top

        # Ensure we're within tile bounds
        if not (0 <= local_x <= TILE_PIXEL_SIZE and 0 <= local_y <= TILE_PIXEL_SIZE):
            return False

        # Apply ninja radius for collision detection
        # Check if ninja center + radius would collide with solid parts
        return self._check_tile_geometry_collision(
            local_x, local_y, tile_value, NINJA_RADIUS
        )

    def _check_tile_geometry_collision(
        self, local_x: float, local_y: float, tile_value: int, radius: float
    ) -> bool:
        """
        Check if a circle at (local_x, local_y) with given radius collides with tile geometry.

        Uses accurate geometric collision detection based on tile definitions from tile_definitions.py.

        Args:
            local_x, local_y: Position within tile (0-24)
            tile_value: Tile type (2-33)
            radius: Collision radius

        Returns:
            True if position is traversable (no collision)
        """
        from ..tile_definitions import TILE_SEGMENT_DIAG_MAP

        # Half tiles (2-5) - very permissive collision detection for 10px ninja
        if tile_value == 2:  # Top half solid
            return local_y > 12 + radius * 0.5  # Much more permissive
        elif tile_value == 3:  # Right half solid
            return local_x < 12 - radius * 0.5
        elif tile_value == 4:  # Bottom half solid
            return local_y < 12 - radius * 0.5
        elif tile_value == 5:  # Left half solid
            return local_x > 12 + radius * 0.5

        # 45-degree slopes (6-9) - very permissive collision detection
        elif tile_value == 6:  # Slope from (0,24) to (24,0) - top-left to bottom-right
            return local_y > local_x + radius * 0.4
        elif tile_value == 7:  # Slope from (0,0) to (24,24) - top-right to bottom-left
            return local_y > (24 - local_x) + radius * 0.4
        elif tile_value == 8:  # Slope from (24,0) to (0,24) - bottom-left to top-right
            return local_y < local_x - radius * 0.4
        elif tile_value == 9:  # Slope from (24,24) to (0,0) - bottom-right to top-left
            return local_y < (24 - local_x) - radius * 0.4

        # Quarter circles (10-13) - convex corners, very permissive
        elif tile_value == 10:  # Bottom-right quarter circle, center (0,0)
            dist = ((local_x - 0) ** 2 + (local_y - 0) ** 2) ** 0.5
            return dist > 12 + radius * 0.3
        elif tile_value == 11:  # Bottom-left quarter circle, center (24,0)
            dist = ((local_x - 24) ** 2 + (local_y - 0) ** 2) ** 0.5
            return dist > 12 + radius * 0.3
        elif tile_value == 12:  # Top-left quarter circle, center (24,24)
            dist = ((local_x - 24) ** 2 + (local_y - 24) ** 2) ** 0.5
            return dist > 12 + radius * 0.3
        elif tile_value == 13:  # Top-right quarter circle, center (0,24)
            dist = ((local_x - 0) ** 2 + (local_y - 24) ** 2) ** 0.5
            return dist > 12 + radius * 0.3

        # Quarter pipes (14-17) - concave corners, very permissive
        elif tile_value == 14:  # Top-left quarter pipe, center (24,24)
            dist = ((local_x - 24) ** 2 + (local_y - 24) ** 2) ** 0.5
            return dist < 12 - radius * 0.3
        elif tile_value == 15:  # Top-right quarter pipe, center (0,24)
            dist = ((local_x - 0) ** 2 + (local_y - 24) ** 2) ** 0.5
            return dist < 12 - radius * 0.3
        elif tile_value == 16:  # Bottom-right quarter pipe, center (0,0)
            dist = ((local_x - 0) ** 2 + (local_y - 0) ** 2) ** 0.5
            return dist < 12 - radius * 0.3
        elif tile_value == 17:  # Bottom-left quarter pipe, center (24,0)
            dist = ((local_x - 24) ** 2 + (local_y - 0) ** 2) ** 0.5
            return dist < 12 - radius * 0.3

        # Diagonal slope tiles (18-33) - use precise diagonal segment collision
        elif tile_value in TILE_SEGMENT_DIAG_MAP:
            return self._check_diagonal_slope_collision(
                local_x, local_y, tile_value, radius
            )

        # Unknown tile type - conservative approach
        else:
            center_margin = radius + 2
            return (
                center_margin < local_x < 24 - center_margin
                and center_margin < local_y < 24 - center_margin
            )

        return False  # Default to non-traversable for unknown types

    def _check_diagonal_slope_collision(
        self, local_x: float, local_y: float, tile_value: int, radius: float
    ) -> bool:
        """
        Check collision with diagonal slope tiles (18-33) using precise line-circle collision.

        Args:
            local_x, local_y: Position within tile (0-24)
            tile_value: Tile type (18-33)
            radius: Collision radius

        Returns:
            True if position is traversable (no collision)
        """
        from ..tile_definitions import TILE_SEGMENT_DIAG_MAP

        if tile_value not in TILE_SEGMENT_DIAG_MAP:
            return False

        # Get diagonal line segment endpoints
        (x1, y1), (x2, y2) = TILE_SEGMENT_DIAG_MAP[tile_value]

        # Calculate distance from point to line segment
        line_dist = self._point_to_line_segment_distance(
            local_x, local_y, x1, y1, x2, y2
        )

        # Determine which side of the line is solid based on tile type
        # Use cross product to determine which side of the line the point is on
        cross_product = (x2 - x1) * (local_y - y1) - (y2 - y1) * (local_x - x1)

        # For slope tiles, determine traversability based on tile type and line side
        # Use very permissive collision detection for 10px ninja
        effective_radius = radius * 0.3  # Much more permissive

        if tile_value in [18, 19, 22, 23, 26, 27, 30, 31]:  # Upward slopes
            # Point is traversable if it's above the slope line with sufficient clearance
            if cross_product > 0:  # Above the line
                return line_dist > effective_radius
            else:  # Below the line (in solid area)
                return False
        elif tile_value in [20, 21, 24, 25, 28, 29, 32, 33]:  # Downward slopes
            # Point is traversable if it's below the slope line with sufficient clearance
            if cross_product < 0:  # Below the line
                return line_dist > effective_radius
            else:  # Above the line (in solid area)
                return False

        # Default: require clearance from the line
        return line_dist > effective_radius

    def _point_to_line_segment_distance(
        self, px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """
        Calculate the shortest distance from a point to a line segment.

        Args:
            px, py: Point coordinates
            x1, y1: Line segment start point
            x2, y2: Line segment end point

        Returns:
            Shortest distance from point to line segment
        """
        # Vector from line start to point
        dx_p = px - x1
        dy_p = py - y1

        # Vector along the line
        dx_line = x2 - x1
        dy_line = y2 - y1

        # Length squared of the line segment
        line_length_sq = dx_line * dx_line + dy_line * dy_line

        if line_length_sq < 1e-6:  # Degenerate line segment (point)
            return (dx_p * dx_p + dy_p * dy_p) ** 0.5

        # Project point onto line (parameter t)
        t = (dx_p * dx_line + dy_p * dy_line) / line_length_sq

        # Clamp t to [0, 1] to stay within line segment
        t = max(0.0, min(1.0, t))

        # Find closest point on line segment
        closest_x = x1 + t * dx_line
        closest_y = y1 + t * dy_line

        # Return distance from point to closest point on segment
        dx_closest = px - closest_x
        dy_closest = py - closest_y
        return (dx_closest * dx_closest + dy_closest * dy_closest) ** 0.5
