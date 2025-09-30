"""
Reachability functionality mixin for N++ environment.

This module contains all reachability analysis functionality that was previously
integrated into the main NppEnvironment class.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, Set

from ...graph.reachability.reachability_system import ReachabilitySystem
from ...graph.reachability.feature_extractor import ReachabilityFeatureExtractor
from ...constants.entity_types import EntityType

# Constants for reachability analysis
TILE_PIXEL_SIZE = 24  # Standard N++ tile size
FULL_MAP_WIDTH_PX = 1056  # Standard N++ level width
FULL_MAP_HEIGHT_PX = 600  # Standard N++ level height


class ReachabilityMixin:
    """
    Mixin class providing reachability analysis functionality for N++ environment.

    This mixin handles:
    - Simplified reachability feature extraction
    - Flood-fill based reachability analysis
    - Performance tracking and caching
    - Entity type classification for reachability
    """

    def _init_reachability_system(
        self, enable_reachability: bool = True, debug: bool = False
    ):
        """Initialize the reachability system components."""
        self.enable_reachability = enable_reachability
        self.debug = debug

        # Initialize reachability system
        self._reachability_system = None
        self._reachability_extractor = None
        self._reachability_cache = {}
        self._reachability_cache_ttl = 0.1  # 100ms cache TTL
        self._last_reachability_time = 0

        # Reachability performance tracking
        self.reachability_times = []
        self.max_time_samples = 100

        # Simple cache for reachability performance
        self._last_ninja_pos = None
        self._cached_reachability = None
        self._cache_valid = False

        if enable_reachability:
            self._reachability_system = ReachabilitySystem()
            self._reachability_extractor = ReachabilityFeatureExtractor()

        # Initialize logging if debug is enabled
        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)

    def _reset_reachability_state(self):
        """Reset reachability state during environment reset."""
        if self.enable_reachability:
            self._clear_reachability_cache()

    def _get_reachability_features(self) -> np.ndarray:
        """Get 8-dimensional reachability features."""
        reachability_features = self._compute_reachability(
            self.nplay_headless.ninja_position()
        )

        return reachability_features

    def _compute_reachability(self, ninja_pos: Tuple[int, int]) -> np.ndarray:
        """
        Compute 8-dimensional reachability features.

        Features:
        1. Reachable area ratio (0-1)
        2. Distance to nearest switch (normalized)
        3. Distance to exit (normalized)
        4. Reachable switches count (normalized)
        5. Reachable hazards count (normalized)
        6. Connectivity score (0-1)
        7. Exit reachable flag (0-1)
        8. Switch-to-exit path exists (0-1)
        """
        # Check cache validity
        if (
            self._cache_valid
            and self._last_ninja_pos == ninja_pos
            and self._cached_reachability is not None
        ):
            return self._cached_reachability

        level_data = self._extract_level_data_for_reachability()
        if level_data is None:
            raise ValueError("Level data is None")

        # Perform simple flood fill reachability analysis
        reachable_positions = self._flood_fill_reachability(ninja_pos, level_data)

        # Compute features
        features = self._compute_reachability_features(
            ninja_pos, reachable_positions, level_data
        )

        # Update cache
        self._last_ninja_pos = ninja_pos
        self._cached_reachability = features
        self._cache_valid = True

        return features

    def _extract_level_data_for_reachability(self) -> Optional[Dict[str, Any]]:
        """Extract level data for reachability analysis."""
        try:
            # Use existing level data extraction
            level_data_obj = self.level_data

            # Convert to dictionary format expected by reachability functions
            level_data = {
                "tiles": level_data_obj.tiles,
                "entities": level_data_obj.entities,
            }

            return level_data

        except Exception as e:
            if self.debug:
                self.logger.warning(
                    f"Could not extract level data for reachability: {e}"
                )
            return None

    def _flood_fill_reachability(
        self, start_pos: Tuple[int, int], level_data: Dict[str, Any]
    ) -> Set[Tuple[int, int]]:
        """
        Perform simple flood fill reachability analysis.

        This is the core simplification - uses basic 4-connectivity
        instead of complex physics-based reachability.
        """
        # Get traversable positions from tiles
        traversable = self._get_traversable_positions(level_data)

        if not traversable:
            return set()

        # Snap start position to grid
        start_grid = self._snap_to_grid(start_pos)
        if start_grid not in traversable:
            # Find nearest traversable position
            start_grid = self._find_nearest_traversable(start_pos, traversable)

        # Simple flood fill with 4-connectivity
        visited = set()
        queue = [start_grid]
        visited.add(start_grid)

        directions = [
            (0, TILE_PIXEL_SIZE),  # Down
            (0, -TILE_PIXEL_SIZE),  # Up
            (TILE_PIXEL_SIZE, 0),  # Right
            (-TILE_PIXEL_SIZE, 0),  # Left
        ]

        while queue:
            current = queue.pop(0)
            x, y = current

            for dx, dy in directions:
                neighbor = (x + dx, y + dy)
                if neighbor in traversable and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return visited

    def _get_traversable_positions(
        self, level_data: Dict[str, Any]
    ) -> Set[Tuple[int, int]]:
        """Get traversable positions from level data."""
        traversable = set()

        tiles = level_data.get("tiles")
        if tiles is None:
            raise ValueError("Tiles are None")

        if isinstance(tiles, np.ndarray):
            height, width = tiles.shape
        else:
            # Assume standard N++ dimensions
            width, height = 42, 23

        for row in range(height):
            for col in range(width):
                if isinstance(tiles, np.ndarray):
                    tile_id = tiles[row, col]
                else:
                    tile_id = 0  # Default to traversable

                # Tile type 0 is empty space (traversable)
                if tile_id == 0:
                    pixel_x = col * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                    pixel_y = row * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                    traversable.add((pixel_x, pixel_y))

        return traversable

    def _snap_to_grid(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """Snap position to tile grid."""
        x, y = position
        grid_x = int(x // TILE_PIXEL_SIZE) * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
        grid_y = int(y // TILE_PIXEL_SIZE) * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
        return (grid_x, grid_y)

    def _find_nearest_traversable(
        self, pos: Tuple[int, int], traversable: Set[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """Find nearest traversable position."""
        if not traversable:
            return pos  # Return original position if no traversable found

        min_dist = float("inf")
        nearest = pos

        for t_pos in traversable:
            dist = (pos[0] - t_pos[0]) ** 2 + (pos[1] - t_pos[1]) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest = t_pos

        return nearest

    def _compute_reachability_features(
        self,
        ninja_pos: Tuple[int, int],
        reachable_positions: Set[Tuple[int, int]],
        level_data: Dict[str, Any],
    ) -> np.ndarray:
        """Compute 8-dimensional reachability feature vector."""
        # Total level area (approximate)
        total_area = (FULL_MAP_WIDTH_PX // TILE_PIXEL_SIZE) * (
            FULL_MAP_HEIGHT_PX // TILE_PIXEL_SIZE
        )

        # 1. Reachable area ratio
        reachable_ratio = len(reachable_positions) / max(total_area, 1)
        reachable_ratio = np.clip(reachable_ratio, 0.0, 1.0)

        # Get entities for remaining features
        entities = level_data.get("entities", [])

        # 2. Distance to nearest switch (normalized)
        switch_distance = self._get_distance_to_entity_type(
            ninja_pos, entities, "switch"
        )
        switch_distance_norm = np.clip(
            switch_distance / 1000.0, 0.0, 1.0
        )  # Normalize by max level width

        # 3. Distance to exit (normalized)
        exit_distance = self._get_distance_to_entity_type(ninja_pos, entities, "exit")
        exit_distance_norm = np.clip(exit_distance / 1000.0, 0.0, 1.0)

        # 4. Reachable switches count (normalized)
        reachable_switches = self._count_reachable_entities(
            reachable_positions, entities, "switch"
        )
        switches_norm = np.clip(
            reachable_switches / 5.0, 0.0, 1.0
        )  # Assume max 5 switches

        # 5. Reachable hazards count (normalized)
        reachable_hazards = self._count_reachable_entities(
            reachable_positions, entities, "hazard"
        )
        hazards_norm = np.clip(
            reachable_hazards / 10.0, 0.0, 1.0
        )  # Assume max 10 hazards

        # 6. Connectivity score (simple metric)
        connectivity = min(
            len(reachable_positions) / 100.0, 1.0
        )  # Simple connectivity metric

        # 7. Exit reachable flag
        exit_reachable = float(
            self._is_entity_reachable(reachable_positions, entities, "exit")
        )

        # 8. Switch-to-exit path exists
        switch_to_exit_path = float(reachable_switches > 0 and exit_reachable > 0)

        return np.array(
            [
                reachable_ratio,
                switch_distance_norm,
                exit_distance_norm,
                switches_norm,
                hazards_norm,
                connectivity,
                exit_reachable,
                switch_to_exit_path,
            ],
            dtype=np.float32,
        )

    def _get_distance_to_entity_type(
        self, pos: Tuple[int, int], entities: list, entity_type: str
    ) -> float:
        """Get distance to nearest entity of specified type."""
        min_distance = 1000.0  # Default large distance

        for entity in entities:
            if self._get_entity_type_for_reachability(entity) == entity_type:
                entity_pos = self._get_entity_position_for_reachability(entity)
                if entity_pos:
                    distance = np.sqrt(
                        (pos[0] - entity_pos[0]) ** 2 + (pos[1] - entity_pos[1]) ** 2
                    )
                    min_distance = min(min_distance, distance)

        return min_distance

    def _count_reachable_entities(
        self,
        reachable_positions: Set[Tuple[int, int]],
        entities: list,
        entity_type: str,
    ) -> int:
        """Count reachable entities of specified type."""
        count = 0

        for entity in entities:
            if self._get_entity_type_for_reachability(entity) == entity_type:
                entity_pos = self._get_entity_position_for_reachability(entity)
                if entity_pos:
                    entity_grid = self._snap_to_grid(entity_pos)
                    if entity_grid in reachable_positions:
                        count += 1

        return count

    def _is_entity_reachable(
        self,
        reachable_positions: Set[Tuple[int, int]],
        entities: list,
        entity_type: str,
    ) -> bool:
        """Check if any entity of specified type is reachable."""
        return (
            self._count_reachable_entities(reachable_positions, entities, entity_type)
            > 0
        )

    def _get_entity_type_for_reachability(self, entity) -> str:
        """Get simplified entity type string for reachability analysis."""
        if isinstance(entity, dict):
            entity_type = entity.get("type", 0)
        else:
            entity_type = getattr(entity, "type", 0)

        # Map entity types to simplified types for reachability
        if entity_type == EntityType.EXIT_DOOR or entity_type == EntityType.EXIT_SWITCH:
            return "exit"
        elif entity_type in [EntityType.LOCKED_DOOR, EntityType.TRAP_DOOR]:
            return "switch"
        elif entity_type in [1, 14, 20, 25, 26]:  # Various hazards
            return "hazard"
        else:
            return "other"

    def _get_entity_position_for_reachability(
        self, entity
    ) -> Optional[Tuple[int, int]]:
        """Get entity position for reachability analysis."""
        if isinstance(entity, dict):
            x = entity.get("x", 0)
            y = entity.get("y", 0)
        else:
            x = getattr(entity, "x", 0)
            y = getattr(entity, "y", 0)

        return (int(x), int(y))

    def _clear_reachability_cache(self):
        """Clear reachability cache."""
        self._last_ninja_pos = None
        self._cached_reachability = None
        self._cache_valid = False

    def get_reachability_performance_stats(self) -> Dict[str, float]:
        """Get reachability performance statistics."""
        if not self.reachability_times:
            return {}

        times_ms = [t * 1000 for t in self.reachability_times]
        return {
            "avg_time_ms": np.mean(times_ms),
            "max_time_ms": np.max(times_ms),
            "min_time_ms": np.min(times_ms),
            "std_time_ms": np.std(times_ms),
        }

    def _reinit_reachability_system_after_unpickling(self):
        """Reinitialize reachability system components after unpickling."""
        if self.enable_reachability:
            if not hasattr(self, "_reachability_system"):
                self._reachability_system = ReachabilitySystem()
            if not hasattr(self, "_reachability_extractor"):
                self._reachability_extractor = ReachabilityFeatureExtractor()
