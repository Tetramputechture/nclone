"""
Precomputed door feature cache for aggressive performance optimization.

This module provides grid-based precomputation of all path distances from every
reachable grid cell to all doors/switches at level load, eliminating expensive
runtime path distance calculations.

Performance:
- Build time: ~50-200ms per level (one-time cost)
- Lookup time: <0.01ms (O(1) dict access)
- Memory: ~1-5MB per level
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from ..graph.level_data import LevelData
from ..constants.physics_constants import (
    LEVEL_WIDTH_PX,
    LEVEL_HEIGHT_PX,
    LOCKED_DOOR_SWITCH_RADIUS,
)

logger = logging.getLogger(__name__)

# Grid cell size for caching (24px matches ninja grid cell quantization)
GRID_CELL_SIZE = 24

# Feature dimensions
MAX_LOCKED_DOORS_ATTENTION = 16
LOCKED_DOOR_FEATURES_DIM = 8


class PrecomputedDoorFeatureCache:
    """
    Precomputes and caches door features for all reachable grid cells.

    This cache eliminates the need for runtime path distance calculations by
    precomputing all possible path distances from every reachable grid cell
    to all doors and switches at level load.

    Cache structure:
    - grid_cache: {(grid_x, grid_y) -> np.ndarray[n_doors, 8]}
    - door_positions: List[(door_x, door_y, switch_x, switch_y)]

    Features per door (8 values):
    [switch_rel_x, switch_rel_y, switch_collected_placeholder, switch_path_dist,
     door_rel_x, door_rel_y, door_open_placeholder, door_path_dist]
    """

    def __init__(self):
        """Initialize empty cache."""
        self.grid_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self.door_positions: List[Tuple[float, float, float, float]] = []
        self.n_doors: int = 0
        self.area_scale: float = 1.0
        self.cache_built: bool = False
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def clear(self):
        """Clear all cached data."""
        self.grid_cache.clear()
        self.door_positions.clear()
        self.n_doors = 0
        self.area_scale = 1.0
        self.cache_built = False
        self.cache_hits = 0
        self.cache_misses = 0

    def build_cache(
        self,
        locked_doors: List[Any],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        path_calculator: Any,
        level_data: LevelData,
        graph_data: Dict[str, Any],
        area_scale: float,
        verbose: bool = False,
    ):
        """
        Precompute features for all reachable grid cells.

        Args:
            locked_doors: List of locked door objects from environment
            adjacency: Graph adjacency structure for pathfinding
            path_calculator: PathDistanceCalculator instance
            level_data: LevelData object with switch states
            graph_data: Graph data dict with adjacency and reachable info
            area_scale: Normalization scale for path distances
            verbose: Print progress information
        """
        if not locked_doors or not adjacency:
            self.cache_built = True
            return

        import time

        start_time = time.perf_counter()

        self.area_scale = area_scale
        self.n_doors = min(len(locked_doors), MAX_LOCKED_DOORS_ATTENTION)

        # Extract static door/switch positions
        self._extract_door_positions(locked_doors)

        if verbose:
            logger.info(f"Building door feature cache for {self.n_doors} doors...")

        # Get all reachable positions from graph
        reachable_positions = graph_data.get("reachable", set())
        if not reachable_positions:
            # Fallback: extract from adjacency
            reachable_positions = set(adjacency.keys())

        # Convert positions to grid cells
        grid_cells = set()
        for pos in reachable_positions:
            grid_x = int(pos[0] // GRID_CELL_SIZE)
            grid_y = int(pos[1] // GRID_CELL_SIZE)
            grid_cells.add((grid_x, grid_y))

        if verbose:
            logger.info(f"Precomputing for {len(grid_cells)} grid cells...")

        # Precompute features for each grid cell
        for grid_x, grid_y in grid_cells:
            # Convert grid cell to world position (center of cell)
            world_x = int(grid_x * GRID_CELL_SIZE + GRID_CELL_SIZE // 2)
            world_y = int(grid_y * GRID_CELL_SIZE + GRID_CELL_SIZE // 2)

            features = self._compute_features_for_position(
                (world_x, world_y), adjacency, path_calculator, level_data, graph_data
            )

            self.grid_cache[(grid_x, grid_y)] = features

        build_time = (time.perf_counter() - start_time) * 1000
        self.cache_built = True

        if verbose:
            memory_mb = (
                len(self.grid_cache) * self.n_doors * LOCKED_DOOR_FEATURES_DIM * 4
            ) / (1024 * 1024)
            logger.info(
                f"Door feature cache built in {build_time:.1f}ms:\n"
                f"  - Grid cells: {len(self.grid_cache)}\n"
                f"  - Doors: {self.n_doors}\n"
                f"  - Memory: ~{memory_mb:.2f} MB"
            )

    def _extract_door_positions(self, locked_doors: List[Any]):
        """Extract static door and switch positions."""
        self.door_positions.clear()

        for door in locked_doors[:MAX_LOCKED_DOORS_ATTENTION]:
            # Extract switch position
            switch_x = float(getattr(door, "switch_x", 0.0))
            switch_y = float(getattr(door, "switch_y", 0.0))

            # Extract door position (center of door segments)
            door_x = float(getattr(door, "door_x", 0.0))
            door_y = float(getattr(door, "door_y", 0.0))

            self.door_positions.append((door_x, door_y, switch_x, switch_y))

    def _compute_features_for_position(
        self,
        ninja_pos: Tuple[int, int],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        path_calculator: Any,
        level_data: LevelData,
        graph_data: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute features for a specific position.

        Returns array of shape (n_doors, 8) with precomputed static features.
        Dynamic features (switch collected, door open) are set to 0 and updated at runtime.
        """
        features = np.zeros(
            (MAX_LOCKED_DOORS_ATTENTION, LOCKED_DOOR_FEATURES_DIM), dtype=np.float32
        )

        for idx in range(self.n_doors):
            door_x, door_y, switch_x, switch_y = self.door_positions[idx]

            # Switch features
            rel_switch_x = (switch_x - ninja_pos[0]) / (LEVEL_WIDTH_PX / 2)
            rel_switch_y = (switch_y - ninja_pos[1]) / (LEVEL_HEIGHT_PX / 2)
            features[idx, 0] = np.clip(rel_switch_x, -1.0, 1.0)
            features[idx, 1] = np.clip(rel_switch_y, -1.0, 1.0)
            # features[idx, 2] = switch_collected (set at runtime)

            # Extract base_adjacency for physics checks
            base_adjacency = (
                graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
            )

            # Switch path distance
            switch_pos = (int(switch_x), int(switch_y))
            switch_path_dist = path_calculator.get_distance(
                ninja_pos,
                switch_pos,
                adjacency,
                base_adjacency,
                level_data=level_data,
                graph_data=graph_data,
                entity_radius=LOCKED_DOOR_SWITCH_RADIUS,
            )

            if switch_path_dist == float("inf"):
                # Fallback to Euclidean
                switch_path_dist = np.sqrt(
                    (switch_x - ninja_pos[0]) ** 2 + (switch_y - ninja_pos[1]) ** 2
                )

            features[idx, 3] = np.clip(switch_path_dist / self.area_scale, 0.0, 1.0)

            # Door features
            rel_door_x = (door_x - ninja_pos[0]) / (LEVEL_WIDTH_PX / 2)
            rel_door_y = (door_y - ninja_pos[1]) / (LEVEL_HEIGHT_PX / 2)
            features[idx, 4] = np.clip(rel_door_x, -1.0, 1.0)
            features[idx, 5] = np.clip(rel_door_y, -1.0, 1.0)
            # features[idx, 6] = door_open (set at runtime)

            # Door path distance
            door_pos = (int(door_x), int(door_y))
            door_path_dist = path_calculator.get_distance(
                ninja_pos,
                door_pos,
                adjacency,
                base_adjacency,
                level_data=level_data,
                graph_data=graph_data,
                entity_radius=0,
            )

            if door_path_dist == float("inf"):
                # Fallback to Euclidean
                door_path_dist = np.sqrt(
                    (door_x - ninja_pos[0]) ** 2 + (door_y - ninja_pos[1]) ** 2
                )

            features[idx, 7] = np.clip(door_path_dist / self.area_scale, 0.0, 1.0)

        return features

    def get_features(
        self, ninja_x: float, ninja_y: float, switch_states: List[bool]
    ) -> np.ndarray:
        """
        Get cached features for current ninja position with updated switch states.

        Args:
            ninja_x: Ninja x position in pixels
            ninja_y: Ninja y position in pixels
            switch_states: List of switch collected states (one per door)

        Returns:
            Array of shape (16, 8) with features for all doors
        """
        if not self.cache_built:
            # Return zeros if cache not built
            return np.zeros(
                (MAX_LOCKED_DOORS_ATTENTION, LOCKED_DOOR_FEATURES_DIM), dtype=np.float32
            )

        # Convert to grid cell
        grid_x = int(ninja_x // GRID_CELL_SIZE)
        grid_y = int(ninja_y // GRID_CELL_SIZE)

        # Lookup cached features
        if (grid_x, grid_y) in self.grid_cache:
            self.cache_hits += 1
            features = self.grid_cache[(grid_x, grid_y)].copy()
        else:
            self.cache_misses += 1
            # Cache miss - return zeros (should be rare)
            features = np.zeros(
                (MAX_LOCKED_DOORS_ATTENTION, LOCKED_DOOR_FEATURES_DIM), dtype=np.float32
            )

        # Update dynamic switch states
        for idx in range(min(len(switch_states), self.n_doors)):
            switch_collected = switch_states[idx]
            features[idx, 2] = float(switch_collected)  # switch_collected
            features[idx, 6] = float(switch_collected)  # door_open (same as collected)

        return features

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics for debugging."""
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_queries if total_queries > 0 else 0.0

        return {
            "cache_built": self.cache_built,
            "grid_cells": len(self.grid_cache),
            "n_doors": self.n_doors,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "memory_mb": (
                len(self.grid_cache) * self.n_doors * LOCKED_DOOR_FEATURES_DIM * 4
            )
            / (1024 * 1024),
        }
