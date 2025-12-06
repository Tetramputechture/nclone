"""
Reachability functionality mixin for N++ environment.

This module contains all reachability analysis functionality that was previously
integrated into the main NppEnvironment class.
"""

import logging
import numpy as np
from typing import Tuple

from ...graph.reachability.reachability_system import ReachabilitySystem
from ...graph.reachability.feature_computation import (
    compute_reachability_features_from_graph,
)
from ...graph.reachability.path_distance_calculator import (
    CachedPathDistanceCalculator,
)


class ReachabilityMixin:
    """
    Mixin class providing reachability analysis functionality for N++ environment.

    This mixin handles:
    - Simplified reachability feature extraction
    - Flood-fill based reachability analysis
    - Performance tracking and caching
    - Entity type classification for reachability
    """

    def _init_reachability_system(self, debug: bool = False):
        """Initialize the reachability system components."""
        self.debug = debug

        # Initialize reachability system
        self._reachability_system = None
        self._reachability_cache = {}
        self._reachability_cache_ttl = 0.1  # 100ms cache TTL
        self._last_reachability_time = 0

        # Grid-based cache for reachability performance (Phase 6 optimization)
        # PERFORMANCE: Use grid-based caching to avoid recomputing when ninja in same cell
        self._last_ninja_pos = None
        self._cached_reachability = None
        self._cache_valid = False
        self._reachability_grid_size = (
            24  # Pixels - invalidate when ninja moves to new grid cell
        )

        self._reachability_system = ReachabilitySystem()
        self._path_calculator = CachedPathDistanceCalculator(
            max_cache_size=200, use_astar=True
        )

        # Initialize logging if debug is enabled
        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)

    def _reset_reachability_state(self):
        """Reset reachability state during environment reset."""
        self._clear_reachability_cache()
        if hasattr(self, "_path_calculator"):
            self._path_calculator.clear_cache()

    def _get_reachability_features(self) -> np.ndarray:
        """Get 13-dimensional reachability features.

        Features include:
        - Base (4): reachable_area_ratio, dist_to_switch_inv, dist_to_exit_inv, exit_reachable
        - Path distances (2): path_dist_to_switch_norm, path_dist_to_exit_norm
        - Direction vectors (4): dir_to_switch_x/y, dir_to_exit_x/y
        - Mine context (2): total_mines_norm, deadly_mine_ratio
        - Phase indicator (1): switch_activated
        """
        reachability_features = self._compute_reachability(
            self.nplay_headless.ninja_position()
        )

        return reachability_features

    def _compute_reachability(self, ninja_pos: Tuple[int, int]) -> np.ndarray:
        """
        Compute 13-dimensional reachability features using adjacency graph.

        Base features (4 dims):
        0. Reachable area ratio (0-1)
        1. Distance to nearest switch (normalized, inverted)
        2. Distance to exit (normalized, inverted)
        3. Exit reachable flag (0-1)

        Path distances (2 dims):
        4. Path distance to switch (normalized 0-1)
        5. Path distance to exit (normalized 0-1)

        Direction vectors (4 dims):
        6. Direction to switch X (-1 to 1)
        7. Direction to switch Y (-1 to 1)
        8. Direction to exit X (-1 to 1)
        9. Direction to exit Y (-1 to 1)

        Mine context features (2 dims):
        10. Total mines normalized (0-1)
        11. Deadly mine ratio (0-1)

        Phase indicator (1 dim):
        12. Switch activated flag (0-1) - CRITICAL for Markov property

        PERFORMANCE: Uses grid-based caching to avoid recomputing when ninja
        hasn't moved significantly (>24px grid cell).
        """
        # Check cache validity using grid-based position matching
        if (
            self._cache_valid
            and self._last_ninja_pos is not None
            and self._cached_reachability is not None
        ):
            last_x, last_y = self._last_ninja_pos
            ninja_x, ninja_y = ninja_pos

            # Check if ninja moved to a new grid cell
            ninja_grid_x = int(ninja_x // self._reachability_grid_size)
            ninja_grid_y = int(ninja_y // self._reachability_grid_size)
            last_grid_x = int(last_x // self._reachability_grid_size)
            last_grid_y = int(last_y // self._reachability_grid_size)

            if ninja_grid_x == last_grid_x and ninja_grid_y == last_grid_y:
                # Cache hit: ninja in same grid cell, return cached features
                return self._cached_reachability

        # Get graph data from GraphMixin (required)
        if not hasattr(self, "current_graph_data") or self.current_graph_data is None:
            raise RuntimeError(
                "current_graph_data not available. Graph must be built before computing reachability features."
            )

        # Get level data
        level_data = self.level_data

        # Extract adjacency from graph data
        adjacency = self.current_graph_data.get("adjacency")
        if adjacency is None:
            raise RuntimeError("adjacency not found in current_graph_data")

        # Ensure level cache is built
        base_adjacency = self.current_graph_data.get("base_adjacency", adjacency)
        self._path_calculator.build_level_cache(
            level_data, adjacency, base_adjacency, self.current_graph_data
        )

        # Compute all reachability features (7 dims)
        # Includes: 4 base + 2 mine context + 1 phase indicator (switch_activated)
        features = compute_reachability_features_from_graph(
            adjacency,
            self.current_graph_data,
            level_data,
            ninja_pos,
            self._path_calculator,
        )

        # Update cache
        self._last_ninja_pos = ninja_pos
        self._cached_reachability = features
        self._cache_valid = True

        return features

    def _clear_reachability_cache(self):
        """Clear reachability cache."""
        self._last_ninja_pos = None
        self._cached_reachability = None
        self._cache_valid = False

    def _reinit_reachability_system_after_unpickling(self):
        """Reinitialize reachability system components after unpickling."""
        if not hasattr(self, "_reachability_system"):
            self._reachability_system = ReachabilitySystem()
