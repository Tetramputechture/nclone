"""
Path distance calculator with caching support.

Provides CachedPathDistanceCalculator which combines pathfinding algorithms
with both per-query caching and level-based precomputed caching.
"""

from typing import Dict, Tuple, Optional, Any, List
from ..level_data import LevelData
from .pathfinding_algorithms import PathDistanceCalculator
from .pathfinding_utils import (
    find_closest_node_to_position,
    extract_spatial_lookups_from_graph_data,
)
from .path_distance_cache import LevelBasedPathDistanceCache
from .performance_timer import PerformanceTimer

# Hardcoded cell size as per N++ constants
CELL_SIZE = 24

# Re-export PathDistanceCalculator for backward compatibility
__all__ = ["PathDistanceCalculator", "CachedPathDistanceCalculator"]


class CachedPathDistanceCalculator:
    """Path distance calculator with caching for static goals."""

    def __init__(
        self,
        max_cache_size: int = 200,
        use_astar: bool = True,
        enable_timing: bool = False,
    ):
        """
        Initialize cached path distance calculator.

        Args:
            max_cache_size: Maximum number of cached distance queries
            use_astar: Use A* (True) or BFS (False) for pathfinding
            enable_timing: Enable performance timing instrumentation
        """
        self.calculator = PathDistanceCalculator(use_astar=use_astar)
        self.cache: Dict[Tuple, float] = {}
        self.max_cache_size = max_cache_size
        self.hits = 0
        self.misses = 0

        # Level-based cache for precomputed distances
        self.level_cache: Optional[LevelBasedPathDistanceCache] = (
            LevelBasedPathDistanceCache()
        )
        self.level_data: Optional[LevelData] = None

        # Track current level to avoid redundant cache rebuilds
        self._current_level_id: Optional[str] = None

        # Spatial hash for fast node lookup (updated per level)
        self._spatial_hash: Optional[any] = None

        # Performance timing
        self.timer = PerformanceTimer(enabled=enable_timing)
        self.enable_timing = enable_timing

    def build_level_cache(
        self,
        level_data: LevelData,
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        graph_data: Optional[Dict] = None,
    ) -> bool:
        """
        Build or rebuild level cache if needed.

        The adjacency graph should only contain nodes reachable from the initial
        player position (from LevelData.start_position). This ensures all cached
        distances are for reachable areas only, preventing caching of unreachable
        or isolated regions.

        Args:
            level_data: Current level data
            adjacency: Graph adjacency structure (should be filtered to only reachable nodes)
            graph_data: Optional graph data dict with spatial_hash for fast lookup

        Returns:
            True if cache was rebuilt, False if cache was valid
        """
        if self.level_cache is None:
            self.level_cache = LevelBasedPathDistanceCache()

        rebuilt = self.level_cache.build_cache(level_data, adjacency, graph_data)
        if rebuilt:
            self.level_data = level_data

        return rebuilt

    def get_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        adjacency: Dict,
        cache_key: Optional[str] = None,
        level_data: Optional[LevelData] = None,
        graph_data: Optional[Dict] = None,
    ) -> float:
        """
        Get path distance with caching and spatial indexing.

        The adjacency graph should only contain nodes reachable from the initial
        player position. This ensures all cached distances are for reachable areas only.

        Args:
            start: Start position (world/full map space)
            goal: Goal position (world/full map space)
            adjacency: Graph adjacency (filtered to only reachable nodes)
            cache_key: Optional key for cache invalidation (e.g., entity type)
            level_data: Optional level data for level-based caching
            graph_data: Optional graph data dict with spatial_hash for fast lookup

        Returns:
            Shortest path distance in pixels
        """
        # Extract spatial hash and subcell lookup from graph_data if available
        spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
            graph_data
        )

        # Store spatial_hash for later use if available
        if spatial_hash is not None:
            self._spatial_hash = spatial_hash

        # Try level cache first if level_data is provided
        if level_data is not None and self.level_cache is not None:
            # Only rebuild cache if level actually changed (Fix for Issue #2)
            level_id = getattr(level_data, "level_id", None)
            if level_id != self._current_level_id:
                with self.timer.measure("build_level_cache"):
                    self.build_level_cache(level_data, adjacency, graph_data)
                self._current_level_id = level_id

            # Find goal node using spatial hash for O(1) lookup
            with self.timer.measure("find_goal_node"):
                goal_node = find_closest_node_to_position(
                    goal,
                    adjacency,
                    threshold=50.0,
                    spatial_hash=self._spatial_hash,
                    subcell_lookup=subcell_lookup,
                )

            # Only proceed if we found a valid goal node within threshold
            if goal_node is not None:
                # Look up goal_id from goal_node using the cache's mapping
                # This ensures we use the same goal_node that was used during cache building
                goal_id = self.level_cache.get_goal_id_from_node(goal_node)

                if goal_id is not None:
                    # Get goal_pos for validation
                    goal_pos = self.level_cache.get_goal_pos_from_id(goal_id)

                    if goal_pos is not None:
                        # Validate that cached goal_pos matches input goal (within tolerance)
                        # This prevents returning wrong cached distance if multiple goals map to same node
                        dx = abs(goal_pos[0] - goal[0])
                        dy = abs(goal_pos[1] - goal[1])
                        if (
                            dx <= 12 and dy <= 12
                        ):  # Within half tile (matching old validation logic)
                            # Snap start position to nearest node using spatial hash
                            # Use same threshold as visualization for ninja node finding
                            with self.timer.measure("find_start_node"):
                                start_node = find_closest_node_to_position(
                                    start,
                                    adjacency,
                                    threshold=24.0,
                                    spatial_hash=self._spatial_hash,
                                    subcell_lookup=subcell_lookup,
                                )

                            # Fallback: if start is already a node, use it directly
                            if start_node is None and start in adjacency:
                                start_node = start

                            # Try level cache with snapped start position
                            if start_node is not None and start_node in adjacency:
                                cached_dist = self.level_cache.get_distance(
                                    start_node, goal_pos, goal_id
                                )
                                if cached_dist != float("inf"):
                                    return cached_dist

        # Fall back to per-query cache
        # Snap to grid for cache consistency (24 pixel tiles)
        start_grid = (start[0] // 24, start[1] // 24)
        goal_grid = (goal[0] // 24, goal[1] // 24)
        key = (start_grid, goal_grid, cache_key)

        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        # Cache miss - compute
        self.misses += 1
        with self.timer.measure("pathfinding_compute"):
            distance = self.calculator.calculate_distance(start, goal, adjacency)

        # Validate that positions are in adjacency before caching
        # Since adjacency is filtered to only reachable nodes, this ensures
        # we only cache distances for reachable areas
        start_node = find_closest_node_to_position(
            start,
            adjacency,
            threshold=24.0,
            spatial_hash=self._spatial_hash,
            subcell_lookup=subcell_lookup,
        )
        goal_node = find_closest_node_to_position(
            goal,
            adjacency,
            threshold=50.0,
            spatial_hash=self._spatial_hash,
            subcell_lookup=subcell_lookup,
        )

        # Only cache if both positions are reachable (in adjacency graph)
        if start_node is not None and goal_node is not None:
            # Store in cache (FIFO eviction)
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest = next(iter(self.cache))
                del self.cache[oldest]

            self.cache[key] = distance

        return distance

    def clear_cache(self):
        """Clear cache (call on level change)."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

        # Clear level cache
        if self.level_cache is not None:
            self.level_cache.clear_cache()
        self.level_data = None

        # Clear level tracking and spatial hash
        self._current_level_id = None
        self._spatial_hash = None

    def get_statistics(self) -> Dict[str, float]:
        """Get cache performance statistics and timing data."""
        total_queries = self.hits + self.misses
        hit_rate = self.hits / total_queries if total_queries > 0 else 0

        stats = {
            "hits": self.hits,
            "misses": self.misses,
            "total_queries": total_queries,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
        }

        # Add level cache statistics if available
        if self.level_cache is not None:
            level_stats = self.level_cache.get_statistics()
            stats["level_cache"] = level_stats

        # Add timing statistics if enabled
        if self.enable_timing:
            stats["timing"] = self.timer.get_stats()

        return stats

    def get_timing_summary(self):
        """Print timing summary for debugging."""
        if self.enable_timing:
            self.timer.print_summary(recent_only=True)
        else:
            print("Timing not enabled for this calculator")

    def get_level_cache_statistics(self) -> Optional[Dict[str, Any]]:
        """Get level cache statistics."""
        if self.level_cache is not None:
            return self.level_cache.get_statistics()
        return None
