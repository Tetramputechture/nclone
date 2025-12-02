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
    find_ninja_node,
)
from .path_distance_cache import LevelBasedPathDistanceCache
from .mine_proximity_cache import MineProximityCostCache
from .performance_timer import PerformanceTimer

# Hardcoded cell size as per N++ constants
CELL_SIZE = 24

# Re-export PathDistanceCalculator for backward compatibility
__all__ = ["PathDistanceCalculator", "CachedPathDistanceCalculator"]


class CachedPathDistanceCalculator:
    """Path distance calculator with caching for static goals."""

    def __init__(
        self,
        max_cache_size: int = 50,
        use_astar: bool = True,
        enable_timing: bool = False,
    ):
        """
        Initialize cached path distance calculator.

        Args:
            max_cache_size: Maximum number of cached distance queries
                          (reduced from 200 to 50 for single-level training;
                          same level means same paths, so smaller cache is sufficient)
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

        # Mine proximity cost cache for hazard avoidance
        self.mine_proximity_cache: Optional[MineProximityCostCache] = (
            MineProximityCostCache()
        )

        # Track current level to avoid redundant cache rebuilds
        self._current_level_id: Optional[str] = None

        # Spatial hash for fast node lookup (updated per level)
        self._spatial_hash: Optional[any] = None

        # Cached adjacency bounds (min_x, max_x, min_y, max_y) for fast bounds checking
        self._adjacency_bounds: Optional[Tuple[int, int, int, int]] = None

        # Performance timing
        self.timer = PerformanceTimer(enabled=enable_timing)
        self.enable_timing = enable_timing

    def build_level_cache(
        self,
        level_data: LevelData,
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
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
            adjacency: Masked graph adjacency structure (should be filtered to only reachable nodes)
            base_adjacency: Base graph adjacency structure (pre-entity-mask, for physics checks)
            graph_data: Optional graph data dict with spatial_hash for fast lookup

        Returns:
            True if cache was rebuilt, False if cache was valid
        """
        if self.level_cache is None:
            self.level_cache = LevelBasedPathDistanceCache()

        # Build mine proximity cache FIRST (needed for level cache building)
        mine_cache_rebuilt = False
        if self.mine_proximity_cache is not None:
            mine_cache_rebuilt = self.mine_proximity_cache.build_cache(
                level_data, adjacency
            )

        # Then build level cache (uses mine proximity cache during BFS)
        rebuilt = self.level_cache.build_cache(
            level_data, adjacency, base_adjacency, graph_data, self.mine_proximity_cache
        )

        if rebuilt or mine_cache_rebuilt:
            self.level_data = level_data
            # Update level ID tracking so get_distance() knows cache is ready
            # LevelData.__post_init__ always sets level_id, so we can trust it exists
            self._current_level_id = level_data.level_id

            # Cache adjacency bounds for fast bounds checking
            if adjacency:
                min_x = min(n[0] for n in adjacency.keys())
                max_x = max(n[0] for n in adjacency.keys())
                min_y = min(n[1] for n in adjacency.keys())
                max_y = max(n[1] for n in adjacency.keys())
                self._adjacency_bounds = (min_x, max_x, min_y, max_y)
            else:
                self._adjacency_bounds = None

        return rebuilt or mine_cache_rebuilt

    def get_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        adjacency: Dict,
        base_adjacency: Dict,
        cache_key: Optional[str] = None,
        level_data: Optional[LevelData] = None,
        graph_data: Optional[Dict] = None,
        entity_radius: float = 0.0,
        ninja_radius: float = 10.0,
    ) -> float:
        """
        Get path distance with caching and spatial indexing.

        The adjacency graph should only contain nodes reachable from the initial
        player position. This ensures all cached distances are for reachable areas only.

        Args:
            start: Start position (world/full map space)
            goal: Goal position (world/full map space)
            adjacency: Masked graph adjacency (filtered to only reachable nodes, for pathfinding)
            base_adjacency: Base graph adjacency (pre-entity-mask, for physics checks)
            cache_key: Optional key for cache invalidation (e.g., entity type)
            level_data: Optional level data for level-based caching
            graph_data: Optional graph data dict with spatial_hash for fast lookup
            entity_radius: Collision radius of the goal entity (default 0.0)
            ninja_radius: Collision radius of the ninja (default 10.0)

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

        # OPTIMIZATION: Early per-query cache check before expensive operations
        # Snap to grid for cache consistency (24 pixel tiles)
        start_grid = (start[0] // 24, start[1] // 24)
        goal_grid = (goal[0] // 24, goal[1] // 24)
        key = (start_grid, goal_grid, cache_key)

        if key in self.cache:
            self.hits += 1
            cached_distance = self.cache[key]
            # Adjust for collision radii
            adjusted_dist = max(0.0, cached_distance - (ninja_radius + entity_radius))
            return adjusted_dist

        # Try level cache first if level_data is provided
        if level_data is not None and self.level_cache is not None:
            # Only rebuild cache if level actually changed
            # LevelData.__post_init__ always sets level_id, so we can trust it exists
            level_id = level_data.level_id

            if level_id != self._current_level_id:
                with self.timer.measure("build_level_cache"):
                    self.build_level_cache(
                        level_data, adjacency, base_adjacency, graph_data
                    )
                # build_level_cache() now sets _current_level_id, but ensure it's set here too
                self._current_level_id = level_id

            # Find goal node using spatial hash for O(1) lookup
            with self.timer.measure("find_goal_node"):
                goal_node = find_closest_node_to_position(
                    goal,
                    adjacency,
                    threshold=None,  # Will be calculated from radii
                    entity_radius=entity_radius,
                    ninja_radius=ninja_radius,
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
                            # Find ninja node using canonical ninja node selection
                            # FIX: Pass level_cache to use PATH distance instead of Euclidean
                            with self.timer.measure("find_start_node"):
                                start_node = find_ninja_node(
                                    start,
                                    adjacency,
                                    spatial_hash=self._spatial_hash,
                                    subcell_lookup=subcell_lookup,
                                    ninja_radius=ninja_radius,
                                    goal_node=goal_node,
                                    level_cache=self.level_cache,
                                    goal_id=goal_id,
                                )

                            # Try level cache with snapped start position
                            if start_node is not None and start_node in adjacency:
                                cached_dist = self.level_cache.get_distance(
                                    start_node, goal_pos, goal_id
                                )
                                if cached_dist != float("inf"):
                                    # SUB-NODE PBRS RESOLUTION: Dense rewards for slow movement
                                    # (0.66-3px per action vs 12px node spacing)
                                    #
                                    # APPROACH: Next-hop directional projection
                                    #
                                    # The next_hop is the neighbor node that's one step closer
                                    # to the goal on the optimal A* path. Direction from player
                                    # to next_hop IS the optimal path direction, respecting
                                    # the adjacency graph (walls, corridors, etc.).
                                    #
                                    # Example: Goal is upper-right, but path goes LEFT first
                                    # - start_node's next_hop points LEFT (toward optimal path)
                                    # - Moving LEFT = positive projection = distance decreases
                                    # - Moving RIGHT = negative projection = distance increases
                                    #
                                    # This is simpler and faster than searching nearby nodes,
                                    # and correctly handles cases where path != Euclidean.
                                    from .pathfinding_utils import (
                                        NODE_WORLD_COORD_OFFSET,
                                    )

                                    # Get next hop toward goal
                                    next_hop = self.level_cache.get_next_hop(
                                        start_node, goal_id
                                    )

                                    if next_hop is not None:
                                        # Convert positions to world coordinates
                                        start_node_world_x = (
                                            start_node[0] + NODE_WORLD_COORD_OFFSET
                                        )
                                        start_node_world_y = (
                                            start_node[1] + NODE_WORLD_COORD_OFFSET
                                        )
                                        next_hop_world_x = (
                                            next_hop[0] + NODE_WORLD_COORD_OFFSET
                                        )
                                        next_hop_world_y = (
                                            next_hop[1] + NODE_WORLD_COORD_OFFSET
                                        )

                                        # Vector from start_node to next_hop (optimal direction)
                                        path_dx = next_hop_world_x - start_node_world_x
                                        path_dy = next_hop_world_y - start_node_world_y
                                        path_len = (
                                            path_dx * path_dx + path_dy * path_dy
                                        ) ** 0.5

                                        if path_len > 0.001:
                                            # Normalize path direction
                                            path_dir_x = path_dx / path_len
                                            path_dir_y = path_dy / path_len

                                            # Vector from start_node to player
                                            player_dx = start[0] - start_node_world_x
                                            player_dy = start[1] - start_node_world_y

                                            # Project player offset onto path direction
                                            # Positive = player ahead (toward next_hop) = closer
                                            # Negative = player behind = further
                                            projection = (
                                                player_dx * path_dir_x
                                                + player_dy * path_dir_y
                                            )

                                            # Subtract projection from cached distance
                                            # (positive projection = closer to goal)
                                            total_dist = max(
                                                0.0,
                                                cached_dist
                                                - projection
                                                - (ninja_radius + entity_radius),
                                            )
                                            return total_dist

                                    # Fallback: no next_hop (at goal) or path too short
                                    return max(
                                        0.0,
                                        cached_dist - (ninja_radius + entity_radius),
                                    )
            else:
                # Check if goal is even within reachable area using cached bounds
                if self._adjacency_bounds is not None:
                    min_x, max_x, min_y, max_y = self._adjacency_bounds

                    # Convert goal to tile data space for comparison
                    from .pathfinding_utils import NODE_WORLD_COORD_OFFSET

                    goal_tile_x = goal[0] - NODE_WORLD_COORD_OFFSET
                    goal_tile_y = goal[1] - NODE_WORLD_COORD_OFFSET

                    # If goal is way outside bounds, this is likely a stale position from previous level
                    # Return a safe max distance instead of inf to avoid warnings
                    if (
                        goal_tile_x < min_x - 100
                        or goal_tile_x > max_x + 100
                        or goal_tile_y < min_y - 100
                        or goal_tile_y > max_y + 100
                    ):
                        # Return a large but finite distance instead of inf
                        # Use a safe max value (level diagonal * 2) to avoid inf warnings
                        # This matches the behavior in _safe_path_distance
                        return 2000.0  # Safe max distance for unreachable goals

        # Cache miss - compute
        # First snap positions to nodes before calculating distance using smart selection
        from .pathfinding_utils import (
            find_goal_node_closest_to_start,
        )

        # Find goal node first (needed for optimal start node selection)
        # Use a temporary start node for goal selection
        temp_start_node = find_ninja_node(
            start,
            adjacency,
            spatial_hash=self._spatial_hash,
            subcell_lookup=subcell_lookup,
            ninja_radius=ninja_radius,
        )
        goal_node = find_goal_node_closest_to_start(
            goal,
            temp_start_node,
            adjacency,
            entity_radius=entity_radius,
            ninja_radius=ninja_radius,
            spatial_hash=self._spatial_hash,
            subcell_lookup=subcell_lookup,
        )

        # Now find the optimal start node with goal_node known
        start_node = find_ninja_node(
            start,
            adjacency,
            spatial_hash=self._spatial_hash,
            subcell_lookup=subcell_lookup,
            ninja_radius=ninja_radius,
            goal_node=goal_node,
        )

        # If we can't find nodes for start or goal, path is unreachable
        if start_node is None or goal_node is None:
            return float("inf")

        # PERFORMANCE OPTIMIZATION: Extract pre-computed physics cache from graph_data
        # Ensure we raise clear error if graph_data or physics cache is missing
        if graph_data is None:
            raise ValueError(
                "graph_data is required for physics cache extraction. "
                "Ensure graph building includes physics cache precomputation."
            )

        physics_cache = graph_data.get("node_physics")
        if physics_cache is None:
            raise ValueError(
                "Physics cache (node_physics) not found in graph_data. "
                "Ensure graph building includes physics cache precomputation."
            )

        self.misses += 1
        with self.timer.measure("pathfinding_compute"):
            distance = self.calculator.calculate_distance(
                start_node,
                goal_node,
                adjacency,
                base_adjacency,
                physics_cache,
                level_data,
                self.mine_proximity_cache,
            )

        # Cache raw node-to-node distance (before adjustment)
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest = next(iter(self.cache))
            del self.cache[oldest]

        self.cache[key] = distance

        # SUB-NODE PBRS RESOLUTION (cache miss path)
        # For cache misses, we don't have precomputed distances for nearby nodes,
        # so we use a simpler approach: just the computed distance without sub-node
        # interpolation. This is acceptable because cache misses are rare after warmup.
        #
        # The primary path (level cache hit) handles sub-node resolution properly
        # by checking minimum distance among nearby nodes.
        total_dist = max(0.0, distance - (ninja_radius + entity_radius))
        return total_dist

    def get_geometric_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        adjacency: Dict,
        base_adjacency: Dict,
        level_data: Optional[LevelData] = None,
        graph_data: Optional[Dict] = None,
        entity_radius: float = 0.0,
        ninja_radius: float = 10.0,
    ) -> float:
        """
        Get GEOMETRIC path distance (actual pixels) with caching.

        Unlike get_distance() which returns physics-weighted costs, this returns
        the actual path length in pixels. Use this for PBRS normalization where
        you need the true geometric distance, not the physics-optimal cost.

        For example, an 800px horizontal path returns ~800px (not ~36 with physics costs).

        Args:
            start: Start position (world/full map space)
            goal: Goal position (world/full map space)
            adjacency: Masked graph adjacency (filtered to only reachable nodes, for pathfinding)
            base_adjacency: Base graph adjacency (pre-entity-mask, for physics checks)
            level_data: Optional level data for level-based caching
            graph_data: Optional graph data dict with spatial_hash for fast lookup
            entity_radius: Collision radius of the goal entity (default 0.0)
            ninja_radius: Collision radius of the ninja (default 10.0)

        Returns:
            Geometric path distance in pixels, or float('inf') if unreachable
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
            # Only rebuild cache if level actually changed
            level_id = level_data.level_id

            if level_id != self._current_level_id:
                self.build_level_cache(
                    level_data, adjacency, base_adjacency, graph_data
                )
                self._current_level_id = level_id

            # Find goal node using spatial hash for O(1) lookup
            goal_node = find_closest_node_to_position(
                goal,
                adjacency,
                threshold=None,  # Will be calculated from radii
                entity_radius=entity_radius,
                ninja_radius=ninja_radius,
                spatial_hash=self._spatial_hash,
                subcell_lookup=subcell_lookup,
            )

            # Only proceed if we found a valid goal node within threshold
            if goal_node is not None:
                # Look up goal_id from goal_node using the cache's mapping
                goal_id = self.level_cache.get_goal_id_from_node(goal_node)

                if goal_id is not None:
                    # Get goal_pos for validation
                    goal_pos = self.level_cache.get_goal_pos_from_id(goal_id)

                    if goal_pos is not None:
                        # Validate that cached goal_pos matches input goal (within tolerance)
                        dx = abs(goal_pos[0] - goal[0])
                        dy = abs(goal_pos[1] - goal[1])
                        if dx <= 12 and dy <= 12:
                            # Find ninja node
                            # FIX: Pass level_cache to use PATH distance instead of Euclidean
                            start_node = find_ninja_node(
                                start,
                                adjacency,
                                spatial_hash=self._spatial_hash,
                                subcell_lookup=subcell_lookup,
                                ninja_radius=ninja_radius,
                                goal_node=goal_node,
                                level_cache=self.level_cache,
                                goal_id=goal_id,
                            )

                            # Try level cache with snapped start position
                            if start_node is not None and start_node in adjacency:
                                # Get GEOMETRIC distance from cache
                                cached_dist = self.level_cache.get_geometric_distance(
                                    start_node, goal_pos, goal_id
                                )
                                if cached_dist != float("inf"):
                                    # Adjust for collision radii
                                    return max(
                                        0.0,
                                        cached_dist - (ninja_radius + entity_radius),
                                    )

        # Cache miss - compute directly using BFS with geometric costs
        from .pathfinding_utils import calculate_geometric_path_distance

        physics_cache = graph_data.get("node_physics") if graph_data else None

        return calculate_geometric_path_distance(
            start,
            goal,
            adjacency,
            base_adjacency,
            physics_cache=physics_cache,
            spatial_hash=self._spatial_hash,
            subcell_lookup=subcell_lookup,
            entity_radius=entity_radius,
            ninja_radius=ninja_radius,
        )

    def clear_cache(self):
        """Clear cache (call on level change)."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

        # Clear level cache
        if self.level_cache is not None:
            self.level_cache.clear_cache()
        self.level_data = None

        # Clear mine proximity cache
        if self.mine_proximity_cache is not None:
            self.mine_proximity_cache.clear_cache()

        # Clear level tracking, spatial hash, and adjacency bounds
        self._current_level_id = None
        self._spatial_hash = None
        self._adjacency_bounds = None

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

        # Add mine proximity cache statistics if available
        if self.mine_proximity_cache is not None:
            mine_stats = self.mine_proximity_cache.get_statistics()
            stats["mine_proximity_cache"] = mine_stats

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
