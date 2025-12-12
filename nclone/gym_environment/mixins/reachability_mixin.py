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

        self._reachability_system = ReachabilitySystem()

        # Get shared_level_cache from parent environment if available
        shared_cache = getattr(self, "shared_level_cache", None)

        self._path_calculator = CachedPathDistanceCalculator(
            max_cache_size=200,
            use_astar=True,
            shared_level_cache=shared_cache,
        )

        # Initialize logging if debug is enabled
        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)

    def _reset_reachability_state(self):
        """Reset reachability state during environment reset."""
        if hasattr(self, "_path_calculator"):
            self._path_calculator.clear_cache()

    def _get_reachability_features(self) -> np.ndarray:
        """Get 22-dimensional reachability features (expanded in Phases 1-3).

        Features include:
        - Base (4): reachable_area_ratio, dist_to_switch_inv, dist_to_exit_inv, exit_reachable
        - Path distances (2): path_dist_to_switch_norm, path_dist_to_exit_norm
        - Direction vectors (4): dir_to_switch_x/y, dir_to_exit_x/y
        - Mine context (2): total_mines_norm, deadly_mine_ratio
        - Phase indicator (1): switch_activated
        - Path direction (8): next_hop_dir_x/y, waypoint_dir_x/y, waypoint_dist, path_detour_flag, mine_clearance_dir_x/y
        - Path difficulty (1): path_difficulty_ratio (physics_cost/geometric_distance)
        """
        reachability_features = self._compute_reachability(
            self.nplay_headless.ninja_position()
        )

        return reachability_features

    def _compute_reachability(self, ninja_pos: Tuple[int, int]) -> np.ndarray:
        """
        Compute 22-dimensional reachability features using adjacency graph (expanded in Phases 1-3).

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

        Path direction features (8 dims) - NEW Phase 1.1:
        13. Next hop direction X (-1 to 1) - optimal path direction
        14. Next hop direction Y (-1 to 1)
        15. Waypoint direction X (-1 to 1) - toward discovered waypoint
        16. Waypoint direction Y (-1 to 1)
        17. Waypoint distance (0-1) - normalized
        18. Path requires detour (0-1) - flag if must go away from goal
        19. Mine clearance direction X (-1 to 1) - safe direction
        20. Mine clearance direction Y (-1 to 1)

        Path difficulty (1 dim) - NEW Phase 3.3:
        21. Path difficulty ratio (0-1) - physics_cost/geometric_distance

        """
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

        # Ensure level cache is built (defensive, but now cheap with fast rejection)
        # Normally built by _update_graph_from_env_state, but this handles edge cases
        # like checkpoint resets where the normal flow may be bypassed.
        # With the fast rejection optimization in build_level_cache, redundant calls
        # are essentially free (single string comparison).
        base_adjacency = self.current_graph_data.get("base_adjacency", adjacency)
        self._path_calculator.build_level_cache(
            level_data, adjacency, base_adjacency, self.current_graph_data
        )

        # Compute all reachability features (22 dims)
        # Includes: 4 base + 2 path distances + 4 direction vectors + 2 mine context + 1 phase indicator + 8 path direction + 1 difficulty
        features = compute_reachability_features_from_graph(
            adjacency,
            self.current_graph_data,
            level_data,
            ninja_pos,
            self._path_calculator,
        )

        return features

    def _reinit_reachability_system_after_unpickling(self):
        """Reinitialize reachability system components after unpickling."""
        if not hasattr(self, "_reachability_system"):
            self._reachability_system = ReachabilitySystem()
