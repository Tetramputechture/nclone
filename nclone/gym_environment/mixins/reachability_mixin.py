"""
Reachability functionality mixin for N++ environment.

This module contains all reachability analysis functionality that was previously
integrated into the main NppEnvironment class.
"""

import logging
import numpy as np
from typing import Dict, Tuple

from ...graph.reachability.reachability_system import ReachabilitySystem
from ...graph.reachability.feature_extractor import ReachabilityFeatureExtractor
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

        self._reachability_system = ReachabilitySystem()
        self._reachability_extractor = ReachabilityFeatureExtractor()
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
        """Get 8-dimensional reachability features."""
        reachability_features = self._compute_reachability(
            self.nplay_headless.ninja_position()
        )

        return reachability_features

    def _compute_reachability(self, ninja_pos: Tuple[int, int]) -> np.ndarray:
        """
        Compute 8-dimensional reachability features using adjacency graph.

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
        self._path_calculator.build_level_cache(
            level_data, adjacency, self.current_graph_data
        )

        # Compute features using shared function
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
        if not hasattr(self, "_reachability_system"):
            self._reachability_system = ReachabilitySystem()
        if not hasattr(self, "_reachability_extractor"):
            self._reachability_extractor = ReachabilityFeatureExtractor()
