"""
Path distance calculator with caching support.

Provides CachedPathDistanceCalculator which consolidates:
- BFS and A* pathfinding algorithms for shortest path calculation
- Per-query caching for frequently accessed paths
- Level-based precomputed caching for goal distances
- Step-level caching for within-step duplicate calls
- Integration with shared memory caching for multi-worker training
"""

import logging
from typing import Dict, Tuple, Optional, Any, List
from collections import deque
import heapq

from ..level_data import LevelData
from .pathfinding_utils import (
    find_closest_node_to_position,
    extract_spatial_lookups_from_graph_data,
    find_ninja_node,
    _violates_horizontal_rule,
    _calculate_physics_aware_cost,
)
from .path_distance_cache import LevelBasedPathDistanceCache
from .mine_proximity_cache import MineProximityCostCache, MineSignedDistanceField
from .performance_timer import PerformanceTimer

logger = logging.getLogger(__name__)

# Hardcoded cell size as per N++ constants
CELL_SIZE = 24


class CachedPathDistanceCalculator:
    """
    Path distance calculator with caching for static goals.

    ARCHITECTURE: Physics costs are directional and cached per (start, goal) pair.

    Caching layers:
    - Per-query cache: Physics-optimal paths (start_node, goal_node) → cost
      * Directionally correct (computed FROM start TO goal)
      * Deterministic (same start+goal+mines always gives same path/cost)
      * Aggressive caching (5000+ entries) for performance

    - Level cache: GEOMETRIC distances only (goal → all nodes)
      * Non-directional (pixels, not physics costs)
      * Used for PBRS normalization and next_hop navigation
      * Cannot store physics costs (would be backwards from flood-fill)

    - Step cache: Dedupe within-step calls (cleared each step)

    For visualization: Use find_shortest_path FROM start TO goal (per-query cached).
    """

    def __init__(
        self,
        max_cache_size: int = 50,
        use_astar: bool = True,
        enable_timing: bool = False,
        shared_level_cache: Optional[Any] = None,
    ):
        """
        Initialize cached path distance calculator.

        Args:
            max_cache_size: Maximum number of cached distance queries
                          (increased to 5000 for physics-optimal path caching;
                          physics paths are deterministic for (start, goal, mine_state) pairs)
            use_astar: Use A* (True) or BFS (False) for pathfinding
            enable_timing: Enable performance timing instrumentation
            shared_level_cache: Optional SharedLevelCache for zero-copy multi-worker training
        """
        self.use_astar = use_astar
        # Per-query cache for physics-optimal paths (start_node, goal_node) → distance
        # Physics costs are deterministic, so aggressive caching is beneficial
        self.cache: Dict[Tuple, float] = {}
        self.max_cache_size = max(
            max_cache_size, 5000
        )  # At least 5000 for physics paths
        self.hits = 0
        self.misses = 0

        # Store shared cache reference (used instead of local caches if provided)
        self._shared_level_cache = shared_level_cache

        # Level-based cache for precomputed distances
        # If shared cache provided, this will be set to a view wrapper
        self.level_cache: Optional[LevelBasedPathDistanceCache] = (
            None if shared_level_cache is not None else LevelBasedPathDistanceCache()
        )
        self.level_data: Optional[LevelData] = None

        # Mine proximity cost cache for hazard avoidance
        # If shared cache provided, this will be set to a view wrapper
        self.mine_proximity_cache: Optional[MineProximityCostCache] = (
            None if shared_level_cache is not None else MineProximityCostCache()
        )

        # Signed Distance Field to mines for O(1) observation space features
        # If shared cache provided, this will be set to a view wrapper
        self.mine_sdf: MineSignedDistanceField = (
            None if shared_level_cache is not None else MineSignedDistanceField()
        )

        # Track current level to avoid redundant cache rebuilds
        self._current_level_id: Optional[str] = None

        # Spatial hash for fast node lookup (updated per level)
        self._spatial_hash: Optional[any] = None

        # OPTIMIZATION: Cache entity positions per level for goal_id inference
        # This avoids repeated entity lookups in hot path
        self._cached_switch_positions: List[Tuple[int, int]] = []
        self._cached_exit_positions: List[Tuple[int, int]] = []

        # OPTIMIZATION: Cache waypoint positions per level for waypoint_id inference
        # Similar to goal caching, avoids repeated waypoint lookups in hot path
        self._cached_waypoint_positions: List[Tuple[int, int]] = []

        # Cache graph data for waypoint-triggered cache rebuilds
        # Allows set_waypoints() to rebuild cache when waypoints change mid-episode
        self._last_adjacency: Optional[Dict] = None
        self._last_base_adjacency: Optional[Dict] = None
        self._last_graph_data: Optional[Dict] = None

        # OPTIMIZATION: Step-level cache for within-step duplicate calls
        # Key: (start, goal, entity_radius) -> distance
        # Cleared at each new step to avoid stale data
        self._step_cache: Dict[
            Tuple[Tuple[int, int], Tuple[int, int], float], float
        ] = {}
        self._step_cache_geo: Dict[
            Tuple[Tuple[int, int], Tuple[int, int], float], float
        ] = {}
        self._step_cache_hits = 0
        self._step_cache_misses = 0

        # Cached adjacency bounds (min_x, max_x, min_y, max_y) for fast bounds checking
        self._adjacency_bounds: Optional[Tuple[int, int, int, int]] = None

        # Performance timing
        self.timer = PerformanceTimer(enabled=enable_timing)
        self.enable_timing = enable_timing

        # OPTIMIZATION: Cache last computed node info for PBRS reuse
        # This avoids redundant find_ninja_node calls when computing potentials
        self._last_start_node: Optional[Tuple[int, int]] = None
        self._last_goal_id: Optional[str] = None
        self._last_next_hop: Optional[Tuple[int, int]] = None
        self._last_geometric_distance: Optional[float] = None

        # OPTIMIZATION: Aggressive geometric distance cache with LRU eviction
        # Key: (start_node, goal_node) -> geometric distance
        # These don't change during an episode, so we can cache aggressively
        # Use OrderedDict for O(1) LRU operations
        from collections import OrderedDict

        self._geometric_distance_cache: OrderedDict[
            Tuple[Tuple[int, int], Tuple[int, int]], float
        ] = OrderedDict()
        # Aggressive cache size: 5000 node pairs per level (~200KB)
        # With generous memory budget, we can cache extensively
        self._geometric_cache_max_size = 5000

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
        # Fast rejection: skip if same level (avoid expensive LevelData comparison)
        # This makes redundant calls essentially free (single string comparison)
        if (
            self._current_level_id is not None
            and level_data.level_id == self._current_level_id
        ):
            return False  # Cache already valid for this level

        # OPTIMIZATION: If shared cache is provided, use view wrappers instead of building locally
        if self._shared_level_cache is not None:
            if self.level_cache is None:
                # Initialize view wrappers on first call
                self.level_cache = self._shared_level_cache.get_path_cache_view()
                self.mine_proximity_cache = (
                    self._shared_level_cache.get_mine_proximity_view()
                )
                self.mine_sdf = self._shared_level_cache.get_sdf_view()

                # Store level data for consistency checks
                self.level_data = level_data
                self._current_level_id = level_data.level_id

                logger.debug(
                    f"Using shared level cache: {self._shared_level_cache.memory_usage_kb:.1f}KB, "
                    f"{self._shared_level_cache.num_nodes} nodes, {self._shared_level_cache.num_goals} goals"
                )

                return True  # "Rebuilt" by using shared cache

            return False  # Already using shared cache, no rebuild needed

        # Standard path: build caches locally
        if self.level_cache is None:
            self.level_cache = LevelBasedPathDistanceCache()

        # Build mine proximity cache FIRST (needed for level cache building)
        # CRITICAL: This MUST be done before level cache building to ensure
        # BFS pathfinding uses correct mine avoidance costs
        mine_cache_rebuilt = False
        if self.mine_proximity_cache is not None:
            mine_cache_rebuilt = self.mine_proximity_cache.build_cache(
                level_data, adjacency
            )

            # DIAGNOSTIC: Validate mine cache state after build
            mine_cache_size = (
                len(self.mine_proximity_cache.cache)
                if hasattr(self.mine_proximity_cache, "cache")
                else 0
            )

            from nclone.constants.entity_types import EntityType
            from nclone.gym_environment.reward_calculation.reward_constants import (
                MINE_HAZARD_COST_MULTIPLIER,
                MINE_PENALIZE_DEADLY_ONLY,
            )

            mines = level_data.get_entities_by_type(
                EntityType.TOGGLE_MINE
            ) + level_data.get_entities_by_type(EntityType.TOGGLE_MINE_TOGGLED)

            # Count deadly mines (state 0) only if MINE_PENALIZE_DEADLY_ONLY is True
            # This matches the logic in mine_proximity_cache._precompute_mine_proximity_costs
            if MINE_PENALIZE_DEADLY_ONLY:
                deadly_mines = [m for m in mines if m.get("state", 0) == 0]
            else:
                deadly_mines = mines

            # logger.warning(
            #     f"[BUILD_LEVEL_CACHE] Mine proximity cache after build: "
            #     f"cache_size={mine_cache_size}, level_has_{len(mines)}_mines, deadly={len(deadly_mines)}, "
            #     f"rebuilt={mine_cache_rebuilt}"
            # )

            # STRICT: If level has deadly mines but cache is empty, something is wrong
            # Only check deadly mines since cache only stores proximity costs for deadly mines
            # NOTE: Cache can be empty if adjacency graph is empty or has no nodes within
            # MINE_HAZARD_RADIUS (75px) of mines - this is a valid case, not an error
            if len(deadly_mines) > 0 and mine_cache_size == 0:
                if MINE_HAZARD_COST_MULTIPLIER > 1.0:
                    # Check if adjacency is empty - if so, this is expected
                    if not adjacency or len(adjacency) == 0:
                        logger.warning(
                            f"[BUILD_LEVEL_CACHE] Level has {len(deadly_mines)} deadly mines but adjacency graph is empty. "
                            f"Mine cache is empty (no nodes to cache). This is expected if graph building failed."
                        )
                    else:
                        # Adjacency exists but cache is empty
                        # This is VALID if mines are >MINE_HAZARD_RADIUS from all reachable nodes
                        # (e.g., mines in unreachable areas). Just log at debug level.
                        from nclone.gym_environment.reward_calculation.reward_constants import (
                            MINE_HAZARD_RADIUS,
                        )

                        logger.debug(
                            f"[BUILD_LEVEL_CACHE] Level has {len(deadly_mines)} deadly mines but mine cache is empty. "
                            f"Adjacency has {len(adjacency)} nodes, but no nodes are within {MINE_HAZARD_RADIUS}px of mines. "
                            f"This is expected if mines are in unreachable areas or far from the navigable graph."
                        )
        else:
            raise RuntimeError(
                "mine_proximity_cache is None during level cache build. "
                "Cannot build physics-optimal paths without mine avoidance data."
            )

        # Build mine SDF for O(1) observation space features
        # This is precomputed at tile resolution (44x25) for zero runtime cost
        sdf_rebuilt = (
            self.mine_sdf.build_sdf(level_data) if self.mine_sdf is not None else False
        )

        # Cache graph data for potential waypoint-triggered rebuilds
        # This allows set_waypoints() to rebuild cache if waypoints change mid-episode
        self._last_adjacency = adjacency
        self._last_base_adjacency = base_adjacency
        self._last_graph_data = graph_data

        # DIAGNOSTIC: Log waypoint state before passing to level cache
        # logger.info(
        #     f"[BUILD_LEVEL_CACHE] About to build with {len(self._cached_waypoint_positions)} waypoints: "
        #     f"{self._cached_waypoint_positions[:3] if len(self._cached_waypoint_positions) > 3 else self._cached_waypoint_positions}"
        # )

        # Then build level cache (uses mine proximity cache and SDF during BFS)
        rebuilt = self.level_cache.build_cache(
            level_data,
            adjacency,
            base_adjacency,
            graph_data,
            self.mine_proximity_cache,
            self.mine_sdf,
            waypoints=self._cached_waypoint_positions,
        )

        if rebuilt or mine_cache_rebuilt or sdf_rebuilt:
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

            # OPTIMIZATION: Cache entity positions for fast goal_id inference
            from ...constants.entity_types import EntityType

            self._cached_switch_positions = [
                (entity.get("x", 0), entity.get("y", 0))
                for entity in level_data.get_entities_by_type(EntityType.EXIT_SWITCH)
            ]
            self._cached_exit_positions = [
                (entity.get("x", 0), entity.get("y", 0))
                for entity in level_data.get_entities_by_type(EntityType.EXIT_DOOR)
            ]

            # NOTE: _cached_waypoint_positions is NOT repopulated here
            # It's only set via set_waypoints() and should persist across cache rebuilds
            # If it gets cleared, that's a bug

            # # DIAGNOSTIC: Log cache build details
            # if rebuilt and self.level_cache is not None:
            #     cache_stats = self.level_cache.get_statistics()
            #     print(
            #         f"[CACHE_BUILD] Level cache built for level_id={self._current_level_id}: "
            #         f"geometric_entries={cache_stats.get('geometric_cache_size', 0)}, "
            #         f"switch_positions={self._cached_switch_positions}, "
            #         f"exit_positions={self._cached_exit_positions}"
            #     )

        return rebuilt or mine_cache_rebuilt or sdf_rebuilt

    def set_waypoints(self, waypoints: Optional[List[Any]]) -> None:
        """Set waypoints for precomputed caching.

        Should be called before build_level_cache() to include waypoints
        in the BFS flood-fill precomputation.

        If called after cache is built, automatically triggers cache rebuild
        to include the new waypoints.

        Args:
            waypoints: List of waypoint objects with .position attribute
        """
        # Convert waypoints to positions
        new_waypoint_positions = []
        if waypoints:
            new_waypoint_positions = [
                (int(wp.position[0]), int(wp.position[1])) for wp in waypoints
            ]

        # Check if waypoints changed
        waypoints_changed = new_waypoint_positions != self._cached_waypoint_positions

        # DIAGNOSTIC: Log BEFORE setting to see if we're clearing accidentally
        logger.info(
            f"[WAYPOINT_CACHE] BEFORE set: {len(self._cached_waypoint_positions)} waypoints, "
            f"NEW: {len(new_waypoint_positions)} waypoints, "
            f"changed={waypoints_changed}"
        )

        self._cached_waypoint_positions = new_waypoint_positions

        logger.info(
            f"[WAYPOINT_CACHE] AFTER set: {len(self._cached_waypoint_positions)} waypoints for caching: "
            f"{self._cached_waypoint_positions}"
        )

        # DIAGNOSTIC: Log waypoint list id to track if list object changes
        logger.info(
            f"[WAYPOINT_CACHE] Waypoint list id: {id(self._cached_waypoint_positions)}"
        )

        # CRITICAL: If waypoints changed and cache already exists, rebuild it
        # This handles mid-episode waypoint discovery by adaptive waypoint system
        if (
            waypoints_changed
            and self.level_cache is not None
            and self.level_data is not None
        ):
            logger.info(
                f"[WAYPOINT_CACHE] Waypoints changed, rebuilding level cache to include "
                f"{len(self._cached_waypoint_positions)} new waypoints"
            )
            # Rebuild cache with new waypoints
            # Need to get adjacency from somewhere - check if we have cached graph data
            if (
                hasattr(self, "_last_adjacency")
                and hasattr(self, "_last_base_adjacency")
                and hasattr(self, "_last_graph_data")
            ):
                # Clear level ID to bypass fast rejection check and force rebuild
                old_level_id = self._current_level_id
                self._current_level_id = None
                logger.info(
                    f"[WAYPOINT_CACHE] Cleared level_id (was {old_level_id}) to force cache rebuild"
                )
                self.build_level_cache(
                    self.level_data,
                    self._last_adjacency,
                    self._last_base_adjacency,
                    self._last_graph_data,
                )
                logger.info(
                    f"[WAYPOINT_CACHE] Cache rebuild complete, level_id now={self._current_level_id}"
                )
            else:
                logger.warning(
                    "[WAYPOINT_CACHE] Cannot rebuild cache - no cached graph data. "
                    "Waypoints will use BFS fallback until next cache build."
                )

    def _calculate_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
        level_data: Optional[Any] = None,
        mine_proximity_cache: Optional[Any] = None,
        mine_sdf: Optional[Any] = None,
        hazard_cost_multiplier: Optional[float] = None,
    ) -> float:
        """
        Calculate shortest navigable path distance using BFS or A*.

        Args:
            start: Starting position (x, y) in pixels
            goal: Goal position (x, y) in pixels
            adjacency: Masked graph adjacency structure (for pathfinding)
            base_adjacency: Base graph adjacency structure (pre-entity-mask, for physics checks)
            physics_cache: Optional pre-computed physics properties for O(1) lookups
            level_data: Optional LevelData for mine proximity checks (fallback)
            mine_proximity_cache: Optional MineProximityCostCache for O(1) mine cost lookup
            mine_sdf: Optional MineSignedDistanceField for velocity-aware hazard costs
            hazard_cost_multiplier: Optional curriculum-adaptive mine hazard cost multiplier

        Returns:
            Shortest path distance in pixels, or float('inf') if unreachable
        """
        # Quick checks
        if start not in adjacency or goal not in adjacency:
            return float("inf")
        if start == goal:
            return 0.0

        if physics_cache is None or len(physics_cache.keys()) == 0:
            raise ValueError(
                "Physics cache is required for physics-aware cost calculation"
            )
        if level_data is None:
            raise ValueError(
                "Level data is required for mine proximity cost calculation"
            )
        if mine_proximity_cache is None:
            raise ValueError(
                "Mine proximity cache is required for mine proximity cost calculation"
            )

        # Choose algorithm
        if self.use_astar:
            return self._astar_distance(
                start,
                goal,
                adjacency,
                base_adjacency,
                physics_cache,
                level_data,
                mine_proximity_cache,
                mine_sdf,
                hazard_cost_multiplier,
            )
        else:
            return self._bfs_distance(
                start,
                goal,
                adjacency,
                base_adjacency,
                physics_cache,
                level_data,
                mine_proximity_cache,
                mine_sdf,
                hazard_cost_multiplier,
            )

    def _bfs_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        adjacency: Dict,
        base_adjacency: Dict,
        physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
        level_data: Optional[Any] = None,
        mine_proximity_cache: Optional[Any] = None,
        mine_sdf: Optional[Any] = None,
        hazard_cost_multiplier: Optional[float] = None,
    ) -> float:
        """BFS pathfinding - guaranteed shortest path with physics and momentum validation."""
        queue = deque([(start, 0.0)])
        visited = {start}
        parents = {start: None}  # Track parents for momentum inference
        grandparents = {start: None}  # Track grandparents for momentum inference
        aerial_chains = {start: 0}  # Track consecutive aerial upward moves

        while queue:
            current, dist = queue.popleft()

            if current == goal:
                return dist

            # Get current node's physics properties for chain tracking
            current_physics = physics_cache[current]
            current_grounded = current_physics["grounded"]
            current_walled = current_physics["walled"]
            current_chain = aerial_chains.get(current, 0)

            # OPTIMIZATION: Cache parent/grandparent lookups to avoid repeated dict.get()
            current_parent = parents.get(current)
            current_grandparent = grandparents.get(current)

            # Explore neighbors from adjacency graph (masked for pathfinding)
            for neighbor, _ in adjacency.get(current, []):
                if neighbor not in visited:
                    # Check horizontal rule before accepting edge (uses base_adjacency + physics_cache)
                    # OPTIMIZATION: Pass physics_cache for O(1) grounding checks
                    if _violates_horizontal_rule(
                        current, neighbor, parents, base_adjacency, physics_cache
                    ):
                        continue

                    # Determine if this edge is aerial upward (for chain tracking)
                    dy = neighbor[1] - current[1]
                    is_aerial_upward = (
                        not current_grounded and not current_walled and dy < 0
                    )

                    # Calculate new chain count: increment for aerial upward, reset otherwise
                    new_chain = current_chain + 1 if is_aerial_upward else 0

                    # Calculate physics-aware edge cost with momentum tracking
                    # OPTIMIZATION: Pass cached parent/grandparent
                    edge_cost = _calculate_physics_aware_cost(
                        current,
                        neighbor,
                        base_adjacency,
                        current_parent,
                        physics_cache,
                        level_data,
                        mine_proximity_cache,
                        current_chain,  # Pass current chain count for cost calculation
                        current_grandparent,  # Pass cached grandparent
                        mine_sdf,  # Pass SDF for velocity-aware mine costs
                        hazard_cost_multiplier,  # Curriculum-adaptive mine costs
                    )

                    visited.add(neighbor)
                    parents[neighbor] = current  # Track parent
                    grandparents[neighbor] = current_parent  # Track cached grandparent
                    aerial_chains[neighbor] = new_chain  # Track chain count
                    queue.append((neighbor, dist + edge_cost))

        return float("inf")

    def _astar_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        adjacency: Dict,
        base_adjacency: Dict,
        physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
        level_data: Optional[Any] = None,
        mine_proximity_cache: Optional[Any] = None,
        mine_sdf: Optional[Any] = None,
        hazard_cost_multiplier: Optional[float] = None,
    ) -> float:
        """A* pathfinding - faster than BFS with heuristic, physics, and momentum validation."""

        def manhattan_heuristic(pos: Tuple[int, int]) -> float:
            """Manhattan distance heuristic (admissible)."""
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # Priority queue: (f_score, g_score, position)
        open_set = [(manhattan_heuristic(start), 0.0, start)]
        g_score = {start: 0.0}
        visited = set()
        parents = {start: None}  # Track parents for momentum inference
        grandparents = {start: None}  # Track grandparents for momentum inference
        aerial_chains = {start: 0}  # Track consecutive aerial upward moves

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return current_g

            # Get current node's physics properties for chain tracking
            current_physics = physics_cache[current]
            current_grounded = current_physics["grounded"]
            current_walled = current_physics["walled"]
            current_chain = aerial_chains.get(current, 0)

            # OPTIMIZATION: Cache parent/grandparent lookups to avoid repeated dict.get()
            current_parent = parents.get(current)
            current_grandparent = grandparents.get(current)

            # Explore neighbors from adjacency graph (masked for pathfinding)
            for neighbor, _ in adjacency.get(current, []):
                if neighbor in visited:
                    continue

                # Check horizontal rule before accepting edge (uses base_adjacency + physics_cache)
                # OPTIMIZATION: Pass physics_cache for O(1) grounding checks
                if _violates_horizontal_rule(
                    current, neighbor, parents, base_adjacency, physics_cache
                ):
                    continue

                # Determine if this edge is aerial upward (for chain tracking)
                dy = neighbor[1] - current[1]
                is_aerial_upward = (
                    not current_grounded and not current_walled and dy < 0
                )

                # Calculate new chain count: increment for aerial upward, reset otherwise
                new_chain = current_chain + 1 if is_aerial_upward else 0

                # Calculate physics-aware edge cost with momentum tracking
                # OPTIMIZATION: Pass cached parent/grandparent
                edge_cost = _calculate_physics_aware_cost(
                    current,
                    neighbor,
                    base_adjacency,
                    current_parent,
                    physics_cache,
                    level_data,
                    mine_proximity_cache,
                    current_chain,  # Pass current chain count for cost calculation
                    current_grandparent,  # Pass cached grandparent
                    mine_sdf,  # Pass SDF for velocity-aware mine costs
                    hazard_cost_multiplier,  # Curriculum-adaptive mine costs
                )

                tentative_g = current_g + edge_cost

                # OPTIMIZATION: Direct comparison instead of .get() with default
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    parents[neighbor] = current  # Track parent
                    grandparents[neighbor] = current_parent  # Track cached grandparent
                    aerial_chains[neighbor] = new_chain  # Track chain count
                    f_score = tentative_g + manhattan_heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return float("inf")

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
        hazard_cost_multiplier: Optional[float] = None,
        goal_id: Optional[str] = None,
    ) -> float:
        """
        Get physics-weighted path distance with caching and spatial indexing.

        The adjacency graph should only contain nodes reachable from the initial
        player position. This ensures all cached distances are for reachable areas only.

        STRICT VALIDATION: graph_data MUST contain physics_cache for PBRS pathfinding.

        Args:
            start: Start position (world/full map space)
            goal: Goal position (world/full map space)
            adjacency: Masked graph adjacency (filtered to only reachable nodes, for pathfinding)
            base_adjacency: Base graph adjacency (pre-entity-mask, for physics checks)
            cache_key: Optional key for cache invalidation (e.g., entity type)
            level_data: Optional level data for level-based caching
            graph_data: Graph data dict with physics_cache (REQUIRED for physics-aware pathfinding)
            entity_radius: Collision radius of the goal entity (default 0.0)
            ninja_radius: Collision radius of the ninja (default 10.0)
            goal_id: Optional goal identifier for level_cache lookup (e.g., "switch" or "exit")
        Returns:
            Shortest physics-weighted path distance
        """
        # STRICT VALIDATION: Ensure graph_data and physics_cache are available
        if graph_data is None:
            raise ValueError(
                "graph_data is REQUIRED for get_distance(). "
                "PBRS pathfinding requires physics_cache from graph_data. "
                "Ensure graph building provides graph_data with physics_cache."
            )

        physics_cache = graph_data.get("node_physics")
        if physics_cache is None or len(physics_cache.keys()) == 0:
            raise ValueError(
                "Physics cache (node_physics) not found in graph_data. "
                "PBRS pathfinding requires physics_cache for accurate path costs. "
                "Ensure graph building includes physics cache precomputation."
            )
        # OPTIMIZATION: Step-level cache check (eliminates duplicate calls within same step)
        step_key = (start, goal, entity_radius)
        if step_key in self._step_cache:
            self._step_cache_hits += 1
            cached_value = self._step_cache[step_key]

            # # CRITICAL DEBUG: Log cache hits to diagnose stuck distance
            # if not hasattr(self, "_cache_hit_count"):
            #     self._cache_hit_count = 0
            # self._cache_hit_count += 1

            # if self._cache_hit_count % 100 == 0:
            #     import logging

            #     logger = logging.getLogger(__name__)
            #     logger.error(
            #         f"[STEP_CACHE_HIT #{self._cache_hit_count}] "
            #         f"start={start}, cached_distance={cached_value:.2f}, "
            #         f"cache_size={len(self._step_cache)}"
            #     )

            return cached_value
        self._step_cache_misses += 1

        # CRITICAL: Early exit for invalid goal positions
        # Goals at (0, 0) indicate entity loading failure - return inf to avoid cache pollution
        if goal[0] == 0 and goal[1] == 0:
            return float("inf")

        # Extract spatial hash and subcell lookup from graph_data if available
        spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
            graph_data
        )

        # Store spatial_hash for later use if available
        if spatial_hash is not None:
            self._spatial_hash = spatial_hash

        # EARLY EXIT: If start and goal are within combined collision radii,
        # the ninja is already touching/overlapping the goal - distance is 0.
        # This handles the edge case where player spawns in the same cell as a goal
        # (common in curriculum-generated levels).
        combined_radius = ninja_radius + entity_radius
        dx = start[0] - goal[0]
        dy = start[1] - goal[1]
        dist_sq = dx * dx + dy * dy
        if dist_sq <= combined_radius * combined_radius:
            return 0.0

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
                    threshold=None,  # Will be calculated from radii (16px default)
                    entity_radius=entity_radius,
                    ninja_radius=ninja_radius,
                    spatial_hash=self._spatial_hash,
                    subcell_lookup=subcell_lookup,
                )

                # Fallback: If 16px threshold fails, try with larger threshold (32px)
                # This handles edge cases where the nearest node is slightly further
                # than the combined collision radii (e.g., entities at cell boundaries)
                if goal_node is None:
                    goal_node = find_closest_node_to_position(
                        goal,
                        adjacency,
                        threshold=32.0,  # Increased from default 16px
                        entity_radius=entity_radius,
                        ninja_radius=ninja_radius,
                        spatial_hash=self._spatial_hash,
                        subcell_lookup=subcell_lookup,
                    )
                    if goal_node is not None:
                        logger.debug(
                            f"Goal node found with extended threshold (32px): "
                            f"goal={goal}, node={goal_node}"
                        )

            # Only proceed if we found a valid goal node within threshold
            if goal_node is not None:
                if goal_id is None:
                    # ROBUST GOAL_ID INFERENCE: Use cached entity positions (O(1) per level)
                    # Avoids repeated entity queries in hot path (2000+ calls per profile)

                    # Check if goal matches a cached switch position (within 24px tolerance)
                    for switch_pos in self._cached_switch_positions:
                        if (
                            abs(switch_pos[0] - goal[0]) < 24
                            and abs(switch_pos[1] - goal[1]) < 24
                        ):
                            goal_id = "switch"  # Use generic alias
                            break

                    # Check if goal matches a cached exit position (within 24px tolerance)
                    if goal_id is None:
                        for exit_pos in self._cached_exit_positions:
                            if (
                                abs(exit_pos[0] - goal[0]) < 24
                                and abs(exit_pos[1] - goal[1]) < 24
                            ):
                                goal_id = "exit"  # Use generic alias
                                break

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
                                    # === SUB-NODE PBRS RESOLUTION: Dense Rewards for Micro-Movements ===
                                    #
                                    # PROBLEM: Agent moves 0.05-3.33px per step, but graph nodes are
                                    # spaced 12px apart. Without sub-node resolution, PBRS only changes
                                    # when crossing node boundaries (every 3-240 steps), making rewards
                                    # too sparse for effective learning.
                                    #
                                    # SOLUTION: Project agent's actual position onto optimal path direction
                                    # to get continuous distance estimates between grid nodes.
                                    #
                                    # APPROACH: Next-Hop Directional Projection
                                    #
                                    # 1. Find agent's closest grid node (start_node)
                                    # 2. Get next_hop = next node along optimal path from level cache
                                    # 3. Compute direction vector: start_node → next_hop
                                    # 4. Project agent's position offset onto this direction
                                    # 5. Adjust distance: node_distance - projection
                                    #
                                    # MATHEMATICS:
                                    # - path_dir = normalize(next_hop - start_node)  [unit vector]
                                    # - player_offset = player_pos - start_node       [offset vector]
                                    # - projection = dot(player_offset, path_dir)     [scalar, in pixels]
                                    # - adjusted_distance = node_distance - projection
                                    #
                                    # INTUITION:
                                    # - projection > 0: Agent ahead of node (toward next_hop) → closer
                                    # - projection < 0: Agent behind node (away from next_hop) → further
                                    # - projection = ±0.1px: Distance changes by ±0.1px (DENSE!)
                                    #
                                    # PATH-AWARE PROPERTY (Critical for Complex Levels):
                                    # Uses next_hop direction (from level cache optimal path), NOT
                                    # Euclidean direction to goal. This correctly handles levels where
                                    # the optimal path goes AWAY from goal first (obstacles, momentum).
                                    #
                                    # Example: Goal at (800, 400), agent must go LEFT first for momentum
                                    # - Euclidean says: move RIGHT toward goal (WRONG)
                                    # - Next-hop says: move LEFT toward next_hop (CORRECT)
                                    # - Agent moves LEFT 0.1px: projection = +0.1, distance -= 0.1px ✓
                                    # - Agent moves RIGHT 0.1px: projection = -0.1, distance += 0.1px ✓
                                    #
                                    # PERFORMANCE: O(1) after level cache warmup (no pathfinding needed)
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

                                            # DIAGNOSTIC: Log sub-node projection values (debug level)
                                            # This helps verify that small movements produce proportional
                                            # distance changes for dense PBRS feedback
                                            logger.debug(
                                                f"[SUB_NODE_FAST] projection={projection:.3f}px, "
                                                f"cached_dist={cached_dist:.1f}px, "
                                                f"adjusted_dist={total_dist:.1f}px, "
                                                f"player_offset=({player_dx:.2f}, {player_dy:.2f}), "
                                                f"path_dir=({path_dir_x:.3f}, {path_dir_y:.3f}), "
                                                f"goal_id={goal_id}"
                                            )

                                            # Store in step cache
                                            self._step_cache[step_key] = total_dist
                                            return total_dist

                                    # Fallback: no next_hop (at goal) or path too short
                                    result = max(
                                        0.0,
                                        cached_dist - (ninja_radius + entity_radius),
                                    )
                                    # Store in step cache
                                    self._step_cache[step_key] = result
                                    return result
            else:
                raise RuntimeError("Goal node not found in level cache path")

        # Cache miss - compute
        # First snap positions to nodes before calculating distance using smart selection
        from .pathfinding_utils import (
            find_goal_node_closest_to_start,
        )

        # Find start node (temp_start_node) for goal selection
        # Use standard radius first, then fallback to extended radius
        temp_start_node = find_ninja_node(
            start,
            adjacency,
            spatial_hash=self._spatial_hash,
            subcell_lookup=subcell_lookup,
            ninja_radius=ninja_radius,
        )

        # Fallback: If default radius (10px) fails for start, try with progressively larger radii
        # This handles cases where the start position (e.g., switch) is far from any graph node
        if temp_start_node is None:
            temp_start_node = find_ninja_node(
                start,
                adjacency,
                spatial_hash=self._spatial_hash,
                subcell_lookup=subcell_lookup,
                ninja_radius=ninja_radius,
                search_radius_override=48.0,
            )
            if temp_start_node is not None:
                logger.debug(
                    f"Start node found with extended search radius (48px): "
                    f"start={start}, node={temp_start_node}"
                )

        # Last resort: Try with very large radius (150px)
        if temp_start_node is None:
            temp_start_node = find_ninja_node(
                start,
                adjacency,
                spatial_hash=self._spatial_hash,
                subcell_lookup=subcell_lookup,
                ninja_radius=ninja_radius,
                search_radius_override=150.0,
            )
            if temp_start_node is not None:
                logger.debug(
                    f"Start node found with large search radius (150px): "
                    f"start={start}, node={temp_start_node}"
                )

        # Ultimate fallback: Find ANY closest node in the entire adjacency graph
        if temp_start_node is None and adjacency:
            from .pathfinding_utils import NODE_WORLD_COORD_OFFSET

            start_tile_x = start[0] - NODE_WORLD_COORD_OFFSET
            start_tile_y = start[1] - NODE_WORLD_COORD_OFFSET

            min_dist_sq = float("inf")
            closest_node = None
            for node_pos in adjacency.keys():
                nx, ny = node_pos
                dist_sq = (nx - start_tile_x) ** 2 + (ny - start_tile_y) ** 2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_node = node_pos

            if closest_node is not None:
                temp_start_node = closest_node
                logger.warning(
                    f"Start node found via unlimited search (dist={min_dist_sq**0.5:.1f}px): "
                    f"start={start}, node={temp_start_node}"
                )

        # Find goal node (needed for optimal start node selection)
        goal_node = find_goal_node_closest_to_start(
            goal,
            temp_start_node,
            adjacency,
            entity_radius=entity_radius,
            ninja_radius=ninja_radius,
            spatial_hash=self._spatial_hash,
            subcell_lookup=subcell_lookup,
        )

        # Fallback: If default radius (16px) fails for goal, try with progressively larger radii
        # This handles edge cases where nodes near the goal are further than collision radii
        if goal_node is None and temp_start_node is not None:
            # Try 48px first
            goal_node = find_goal_node_closest_to_start(
                goal,
                temp_start_node,
                adjacency,
                entity_radius=entity_radius,
                ninja_radius=ninja_radius,
                spatial_hash=self._spatial_hash,
                subcell_lookup=subcell_lookup,
                search_radius_override=48.0,
            )
            if goal_node is not None:
                logger.debug(
                    f"Goal node found with extended search radius (48px): "
                    f"goal={goal}, node={goal_node}"
                )

        # Last resort: Try with very large radius (150px - ~6 tiles)
        # This handles extreme cases where entity is positioned far from any graph node
        if goal_node is None and temp_start_node is not None:
            goal_node = find_goal_node_closest_to_start(
                goal,
                temp_start_node,
                adjacency,
                entity_radius=entity_radius,
                ninja_radius=ninja_radius,
                spatial_hash=self._spatial_hash,
                subcell_lookup=subcell_lookup,
                search_radius_override=150.0,
            )
            if goal_node is not None:
                logger.debug(
                    f"Goal node found with large search radius (150px): "
                    f"goal={goal}, node={goal_node}"
                )

        # Ultimate fallback: Find ANY closest node in the entire adjacency graph
        # This handles cases where the goal is extremely far from any graph node
        if goal_node is None and temp_start_node is not None and adjacency:
            from .pathfinding_utils import NODE_WORLD_COORD_OFFSET

            goal_tile_x = goal[0] - NODE_WORLD_COORD_OFFSET
            goal_tile_y = goal[1] - NODE_WORLD_COORD_OFFSET

            min_dist_sq = float("inf")
            closest_node = None
            for node_pos in adjacency.keys():
                nx, ny = node_pos
                dist_sq = (nx - goal_tile_x) ** 2 + (ny - goal_tile_y) ** 2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_node = node_pos

            if closest_node is not None:
                goal_node = closest_node
                # Only log warning if goal is near origin (likely a bug)
                # Valid goals at (0, 0) are extremely rare in actual levels
                if goal[0] < 50 and goal[1] < 50:
                    logger.warning(
                        f"Goal node found via unlimited search (dist={min_dist_sq**0.5:.1f}px): "
                        f"goal={goal}, node={goal_node}. "
                        f"Goal near origin likely indicates cache pollution or entity loading issue. "
                        f"Check curriculum cache clearing and entity extraction."
                    )
                else:
                    logger.debug(
                        f"Goal node found via unlimited search (dist={min_dist_sq**0.5:.1f}px): "
                        f"goal={goal}, node={goal_node}"
                    )

        # Now find the optimal start node with goal_node known
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

        # Fallback: If default radius fails for final start_node, try with progressively larger radii
        if start_node is None:
            start_node = find_ninja_node(
                start,
                adjacency,
                spatial_hash=self._spatial_hash,
                subcell_lookup=subcell_lookup,
                ninja_radius=ninja_radius,
                goal_node=goal_node,
                search_radius_override=48.0,
            )

        if start_node is None:
            start_node = find_ninja_node(
                start,
                adjacency,
                spatial_hash=self._spatial_hash,
                subcell_lookup=subcell_lookup,
                ninja_radius=ninja_radius,
                goal_node=goal_node,
                search_radius_override=150.0,
            )

        # Ultimate fallback for final start_node
        if start_node is None and adjacency:
            from .pathfinding_utils import NODE_WORLD_COORD_OFFSET

            start_tile_x = start[0] - NODE_WORLD_COORD_OFFSET
            start_tile_y = start[1] - NODE_WORLD_COORD_OFFSET

            min_dist_sq = float("inf")
            closest_node = None
            for node_pos in adjacency.keys():
                nx, ny = node_pos
                dist_sq = (nx - start_tile_x) ** 2 + (ny - start_tile_y) ** 2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_node = node_pos

            if closest_node is not None:
                start_node = closest_node

        # If we can't find nodes for start or goal, path is unreachable
        if start_node is None or goal_node is None:
            # Log diagnostic info to help debug false "unreachable" errors
            logger.warning(
                f"Path unreachable - node finding failed in cache miss path. "
                f"start={start}, goal={goal}, "
                f"temp_start_node={temp_start_node}, start_node={start_node}, "
                f"goal_node={goal_node}, adjacency_size={len(adjacency) if adjacency else 0}"
            )
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
            distance = self._calculate_distance(
                start_node,
                goal_node,
                adjacency,
                base_adjacency,
                physics_cache,
                level_data,
                self.mine_proximity_cache,
                self.mine_sdf,
                hazard_cost_multiplier,
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
        goal_id: Optional[str] = None,
    ) -> float:
        """
        Get GEOMETRIC path distance (actual pixels) along physics-optimal path.

        OPTIMIZATION: Uses pre-computed node-to-node distances from level cache,
        then applies sub-node position offset. This eliminates expensive BFS calls.

        This returns the actual path length in pixels along the physics-optimal path.
        Physics costs are used during pathfinding to find the optimal route, but the
        returned value is the geometric distance (pixels) of that route.

        For example, an 800px horizontal path returns ~800px (not ~36 with physics costs).

        STRICT VALIDATION: graph_data MUST contain physics_cache for physics-optimal pathfinding.

        Args:
            start: Start position (world/full map space)
            goal: Goal position (world/full map space)
            adjacency: Masked graph adjacency (filtered to only reachable nodes, for pathfinding)
            base_adjacency: Base graph adjacency (pre-entity-mask, for physics checks)
            level_data: Optional level data for level-based caching
            graph_data: Graph data dict with physics_cache (REQUIRED for physics-aware pathfinding)
            entity_radius: Collision radius of the goal entity (default 0.0)
            ninja_radius: Collision radius of the ninja (default 10.0)
            goal_id: Optional goal identifier for level_cache lookup (e.g., "switch" or "exit")
        Returns:
            Geometric path distance in pixels, or float('inf') if unreachable
        """
        # STRICT VALIDATION: Ensure graph_data and physics_cache are available
        if graph_data is None:
            raise ValueError(
                "graph_data is REQUIRED for get_geometric_distance(). "
                "This function uses physics-optimal pathfinding to measure pixel distances. "
                "Ensure graph building provides graph_data with physics_cache."
            )

        # Validate physics_cache is in graph_data (will be checked again in pathfinding)
        if graph_data.get("node_physics") is None:
            raise ValueError(
                "Physics cache (node_physics) not found in graph_data. "
                "get_geometric_distance() requires physics_cache for optimal route finding. "
                "Ensure graph building includes physics cache precomputation."
            )
        # Import needed for ninja node finding
        from .pathfinding_utils import find_ninja_node
        import logging

        logger = logging.getLogger(__name__)
        # OPTIMIZATION: Step-level cache check (eliminates duplicate calls within same step)
        step_key = (start, goal, entity_radius)
        if step_key in self._step_cache_geo:
            self._step_cache_hits += 1
            return self._step_cache_geo[step_key]
        self._step_cache_misses += 1

        # CRITICAL: Early exit for invalid goal positions
        # Goals at (0, 0) indicate entity loading failure - return inf to avoid cache pollution
        if goal[0] == 0 and goal[1] == 0:
            return float("inf")

        # Extract spatial hash and subcell lookup from graph_data if available
        spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
            graph_data
        )

        # Store spatial_hash for later use if available
        if spatial_hash is not None:
            self._spatial_hash = spatial_hash

        # EARLY EXIT: If start and goal are within combined collision radii,
        # the ninja is already touching/overlapping the goal - distance is 0.
        combined_radius = ninja_radius + entity_radius
        dx = start[0] - goal[0]
        dy = start[1] - goal[1]
        dist_sq = dx * dx + dy * dy
        if dist_sq <= combined_radius * combined_radius:
            return 0.0

        # FAST PATH: Use level cache with position offset (eliminates BFS calls)
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
                if goal_id is None:
                    # ROBUST GOAL_ID INFERENCE: Use cached entity positions (O(1) per level)
                    # Avoids repeated entity queries in hot path (2000+ calls per profile)

                    # DIAGNOSTIC: Log waypoint cache state
                    logger.debug(
                        f"[GOAL_ID_INFERENCE] Checking goal {goal} against "
                        f"{len(self._cached_waypoint_positions)} cached waypoints"
                    )

                    # Check if goal matches a cached switch position (within 24px tolerance)
                    for switch_pos in self._cached_switch_positions:
                        if (
                            abs(switch_pos[0] - goal[0]) < 24
                            and abs(switch_pos[1] - goal[1]) < 24
                        ):
                            goal_id = "switch"  # Use generic alias
                            break

                    # Check if goal matches a cached exit position (within 24px tolerance)
                    if goal_id is None:
                        for exit_pos in self._cached_exit_positions:
                            if (
                                abs(exit_pos[0] - goal[0]) < 24
                                and abs(exit_pos[1] - goal[1]) < 24
                            ):
                                goal_id = "exit"  # Use generic alias
                                break

                    # Check if goal matches a cached waypoint position (within 24px tolerance)
                    if goal_id is None:
                        for i, waypoint_pos in enumerate(
                            self._cached_waypoint_positions
                        ):
                            if (
                                abs(waypoint_pos[0] - goal[0]) < 24
                                and abs(waypoint_pos[1] - goal[1]) < 24
                            ):
                                goal_id = f"waypoint_{i}"
                                break

                if goal_id is None:
                    # logger.warning(
                    #     f"[GOAL_ID_INFERENCE] Goal ID not found for goal: {goal}. "
                    #     f"Cached waypoints: {len(self._cached_waypoint_positions)} "
                    #     f"(list id: {id(self._cached_waypoint_positions)})"
                    # )
                    # Reduced verbosity - remove duplicate prints
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"cached_exit_positions: {self._cached_exit_positions}"
                        )
                        logger.debug(
                            f"cached_switch_positions: {self._cached_switch_positions}"
                        )
                        logger.debug(
                            f"cached_waypoint_positions: {self._cached_waypoint_positions}"
                        )

                if goal_id is not None:
                    # Get goal_pos for validation
                    goal_pos = self.level_cache.get_goal_pos_from_id(goal_id)

                    if goal_pos is not None:
                        # Validate that cached goal_pos matches input goal (within tolerance)
                        dx_check = abs(goal_pos[0] - goal[0])
                        dy_check = abs(goal_pos[1] - goal[1])

                        # DIAGNOSTIC: Log validation failures
                        if dx_check > 12 or dy_check > 12:
                            if not hasattr(self, "_validation_fail_count"):
                                self._validation_fail_count = 0
                            self._validation_fail_count += 1
                            if self._validation_fail_count <= 5:
                                logger.warning(
                                    f"[VALIDATION_FAIL #{self._validation_fail_count}] goal_id={goal_id}, "
                                    f"goal={goal}, goal_pos={goal_pos}, "
                                    f"dx={dx_check:.1f}, dy={dy_check:.1f}, "
                                    f"threshold=12"
                                )

                        if dx_check <= 12 and dy_check <= 12:
                            # Find ninja node using canonical selection
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

                            # FAST PATH: Use cached node-to-node distance + position offset
                            if start_node is not None and start_node in adjacency:
                                # OPTIMIZATION: Check per-query cache first (node pair cache)
                                cache_key = (start_node, goal_node)
                                if cache_key in self._geometric_distance_cache:
                                    cached_dist = self._geometric_distance_cache[
                                        cache_key
                                    ]
                                    # LRU: Move to end to mark as recently used
                                    self._geometric_distance_cache.move_to_end(
                                        cache_key
                                    )
                                    # Apply sub-node position offset using next_hop projection
                                    # (already implemented below, reuse that logic)
                                else:
                                    # Get GEOMETRIC distance from level cache (node-to-node)
                                    cached_dist = (
                                        self.level_cache.get_geometric_distance(
                                            start_node, goal_pos, goal_id
                                        )
                                    )
                                    if cached_dist != float("inf"):
                                        # Store in per-query cache with LRU eviction
                                        self._geometric_distance_cache[cache_key] = (
                                            cached_dist
                                        )
                                        self._geometric_distance_cache.move_to_end(
                                            cache_key
                                        )

                                        # LRU eviction: Remove oldest if exceeds max size
                                        if (
                                            len(self._geometric_distance_cache)
                                            > self._geometric_cache_max_size
                                        ):
                                            self._geometric_distance_cache.popitem(
                                                last=False
                                            )

                                if cached_dist != float("inf"):
                                    # === SUB-NODE PBRS RESOLUTION: Dense Rewards for Micro-Movements ===
                                    #
                                    # PROBLEM: Agent moves 0.05-3.33px per step, but graph nodes are
                                    # spaced 12px apart. Without sub-node resolution, PBRS would only
                                    # change when crossing node boundaries (~3-240 steps), causing:
                                    # - Sparse reward signal (agent blind to small progress)
                                    # - Oscillation within node (no gradient to guide micro-corrections)
                                    # - Poor sample efficiency (can't learn from incremental progress)
                                    #
                                    # SOLUTION: Project agent's actual world position onto the optimal
                                    # path direction to interpolate distance between grid nodes. This
                                    # provides continuous feedback for every frame of movement.
                                    #
                                    # APPROACH: Next-Hop Directional Projection
                                    #
                                    # The next_hop is the neighbor node that's one step closer to the
                                    # goal along the optimal physics-aware path (from level cache BFS).
                                    # The vector from start_node → next_hop represents the locally
                                    # optimal movement direction, respecting:
                                    # - Level geometry (walls, corridors, platforms)
                                    # - Physics costs (grounded routes preferred over aerial)
                                    # - Obstacles requiring detours away from goal
                                    #
                                    # KEY INSIGHT: This is PATH-AWARE, not Euclidean
                                    # Example level with vertical wall blocking direct path:
                                    #   Goal at (800, 400) [upper-right]
                                    #   Agent at (400, 400) [same height, left of goal]
                                    #   Wall blocks direct rightward movement
                                    #   Optimal path: go DOWN to (400, 500), then RIGHT
                                    #
                                    #   At (400, 400): next_hop points DOWN (toward detour), not RIGHT
                                    #   - Moving DOWN 0.1px: projection = +0.1, distance -= 0.1 ✓ REWARD
                                    #   - Moving RIGHT 0.1px: projection = -0.1, distance += 0.1 ✓ PENALTY
                                    #
                                    #   This guides agent along optimal path, not toward unreachable goal!
                                    #
                                    # MATHEMATICAL PROPERTIES:
                                    # - Linear interpolation: 1px movement = 1px distance change
                                    # - Continuous: No discontinuities between nodes
                                    # - Consistent: Same projection for all movements in same direction
                                    # - Bounded: Projection capped by node spacing (±12px typical)
                                    #
                                    # PERFORMANCE: O(1) lookup after level cache warmup
                                    # - Level cache precomputes next_hop for all nodes (one-time cost)
                                    # - Each query: 2 cache lookups + vector math (no pathfinding!)
                                    #
                                    # Get next hop toward goal for optimal path direction
                                    from .pathfinding_utils import (
                                        NODE_WORLD_COORD_OFFSET,
                                    )

                                    next_hop = self.level_cache.get_next_hop(
                                        start_node, goal_id
                                    )

                                    if next_hop is not None:
                                        # Convert positions to world coordinates for projection
                                        # (Nodes are in tile data space, need +24px offset)
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

                                        # Compute optimal path direction (start_node → next_hop)
                                        path_dx = next_hop_world_x - start_node_world_x
                                        path_dy = next_hop_world_y - start_node_world_y
                                        path_len = (
                                            path_dx * path_dx + path_dy * path_dy
                                        ) ** 0.5

                                        if path_len > 0.001:
                                            # Normalize to unit vector (length = 1.0)
                                            path_dir_x = path_dx / path_len
                                            path_dir_y = path_dy / path_len

                                            # Compute agent's offset from start_node
                                            player_dx = start[0] - start_node_world_x
                                            player_dy = start[1] - start_node_world_y

                                            # Project offset onto path direction (dot product)
                                            # Result: Scalar distance along path direction
                                            # - Positive: Agent ahead of node (toward next_hop)
                                            # - Negative: Agent behind node (away from next_hop)
                                            # - Zero: Agent exactly at start_node center
                                            projection = (
                                                player_dx * path_dir_x
                                                + player_dy * path_dir_y
                                            )

                                            # Adjust cached distance by projection
                                            # cached_dist = distance from start_node to goal
                                            # projection = how far agent is ahead (+) or behind (-)
                                            # total_dist = cached_dist - projection
                                            #
                                            # Example: cached_dist = 100px, projection = +5px
                                            # → total_dist = 95px (agent is 5px closer than node)
                                            total_dist = max(
                                                0.0,
                                                cached_dist
                                                - projection
                                                - (ninja_radius + entity_radius),
                                            )

                                            # DIAGNOSTIC: Log sub-node projection for geometric distance
                                            # This tracks the PBRS potential calculation path specifically
                                            logger.debug(
                                                f"[SUB_NODE_GEO_FAST] projection={projection:.3f}px, "
                                                f"cached_dist={cached_dist:.1f}px, "
                                                f"adjusted_dist={total_dist:.1f}px, "
                                                f"player_offset=({player_dx:.2f}, {player_dy:.2f}), "
                                                f"path_dir=({path_dir_x:.3f}, {path_dir_y:.3f}), "
                                                f"goal_id={goal_id}"
                                            )

                                            # OPTIMIZATION: Store for PBRS reuse
                                            self._last_start_node = start_node
                                            self._last_goal_id = goal_id
                                            self._last_geometric_distance = cached_dist
                                            self._last_next_hop = next_hop

                                            # Store in step cache
                                            self._step_cache_geo[step_key] = total_dist

                                            return total_dist

                                    # Fallback: no next_hop (at goal) or path too short
                                    result = max(
                                        0.0,
                                        cached_dist - (ninja_radius + entity_radius),
                                    )
                                    # Store in step cache
                                    self._step_cache_geo[step_key] = result
                                    return result

        # SLOW PATH: Cache miss - compute directly using BFS with geometric costs
        # This should be rare after level cache warm-up
        # LOG cache miss for diagnostic purposes (helps identify cache issues)
        if self.level_cache is not None and level_data is not None:
            # Detailed diagnostic: why did cache miss?
            start_node = None
            goal_node = None
            goal_id = None

            # Try to find nodes for diagnostic
            subcell_lookup = graph_data.get("subcell_lookup") if graph_data else None
            if adjacency:
                start_node = find_ninja_node(
                    start,
                    adjacency,
                    spatial_hash=self._spatial_hash,
                    subcell_lookup=subcell_lookup,
                    ninja_radius=ninja_radius,
                )

            # Check goal inference
            if start[0] == 0 and start[1] == 0:
                goal_id = "invalid_zero_goal"
            else:
                # Check switch positions
                for switch_pos in self._cached_switch_positions:
                    if (
                        abs(switch_pos[0] - goal[0]) < 24
                        and abs(switch_pos[1] - goal[1]) < 24
                    ):
                        goal_id = "switch"
                        break
                # Check exit positions
                if goal_id is None:
                    for exit_pos in self._cached_exit_positions:
                        if (
                            abs(exit_pos[0] - goal[0]) < 24
                            and abs(exit_pos[1] - goal[1]) < 24
                        ):
                            goal_id = "exit"
                            break

            # cache_key_would_be = (
            #     (start_node, goal_id) if start_node and goal_id else None
            # )
            # in_cache = (
            #     cache_key_would_be in self.level_cache.geometric_cache
            #     if cache_key_would_be
            #     else False
            # )

            # logger.warning(
            #     f"[CACHE_MISS] get_geometric_distance fallback to BFS: "
            #     f"start={start}, goal={goal}, level={self._current_level_id}, "
            #     f"start_node={start_node}, goal_id={goal_id}, "
            #     f"would_be_in_cache={in_cache}, "
            #     f"cache_size={len(self.level_cache.geometric_cache) if self.level_cache else 0}"
            # )

        from .pathfinding_utils import calculate_geometric_path_distance

        physics_cache = graph_data.get("node_physics") if graph_data else None

        # CRITICAL: Pass mine cache and level data for mine avoidance in cache miss path
        # This fallback BFS must use same costs as cached paths
        result = calculate_geometric_path_distance(
            start,
            goal,
            adjacency,
            base_adjacency,
            physics_cache=physics_cache,
            spatial_hash=self._spatial_hash,
            subcell_lookup=subcell_lookup,
            entity_radius=entity_radius,
            ninja_radius=ninja_radius,
            level_data=level_data,  # Pass for mine proximity calculations
            mine_proximity_cache=self.mine_proximity_cache,  # Pass for mine avoidance
            mine_sdf=self.mine_sdf,  # Pass for velocity-aware mine costs
        )

        # Store in step cache even for BFS fallback (avoid duplicate BFS within same step)
        self._step_cache_geo[step_key] = result
        return result

    def clear_step_cache(self):
        """Clear step-level cache (call between environment steps)."""
        self._step_cache.clear()
        self._step_cache_geo.clear()

    def clear_cache(self):
        """Clear cache (call on level change)."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

        # Clear level cache (but not if it's a shared cache view)
        if self.level_cache is not None and hasattr(self.level_cache, "clear_cache"):
            self.level_cache.clear_cache()
        self.level_data = None

        # Clear mine proximity cache (but not if it's a shared cache view)
        if self.mine_proximity_cache is not None and hasattr(
            self.mine_proximity_cache, "clear_cache"
        ):
            self.mine_proximity_cache.clear_cache()

        # Clear mine SDF (but not if it's a shared SDF view)
        if self.mine_sdf is not None and hasattr(self.mine_sdf, "clear"):
            self.mine_sdf.clear()

        # Clear level tracking, spatial hash, and adjacency bounds
        self._current_level_id = None
        self._spatial_hash = None
        self._adjacency_bounds = None

        # Clear cached entity positions
        self._cached_switch_positions.clear()
        self._cached_exit_positions.clear()

        # Clear cached waypoint positions
        self._cached_waypoint_positions.clear()

        # Clear cached graph data
        self._last_adjacency = None
        self._last_base_adjacency = None
        self._last_graph_data = None

        # Clear aggressive geometric distance cache (OrderedDict)
        self._geometric_distance_cache.clear()

        # Clear step-level cache
        self.clear_step_cache()
        self._step_cache_hits = 0
        self._step_cache_misses = 0

    def get_statistics(self) -> Dict[str, float]:
        """Get cache performance statistics and timing data."""
        total_queries = self.hits + self.misses
        hit_rate = self.hits / total_queries if total_queries > 0 else 0

        step_total = self._step_cache_hits + self._step_cache_misses
        step_hit_rate = self._step_cache_hits / step_total if step_total > 0 else 0

        stats = {
            "hits": self.hits,
            "misses": self.misses,
            "total_queries": total_queries,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "step_cache_hits": self._step_cache_hits,
            "step_cache_misses": self._step_cache_misses,
            "step_cache_hit_rate": step_hit_rate,
            "step_cache_size": len(self._step_cache) + len(self._step_cache_geo),
        }

        # Add level cache statistics if available
        if self.level_cache is not None:
            level_stats = self.level_cache.get_statistics()
            stats["level_cache"] = level_stats

            # DIAGNOSTIC: Calculate sub-node resolution usage rate
            # Fast path (level cache hit) uses sub-node projection
            # Slow path (BFS fallback) also uses sub-node projection now
            # Log to verify both paths are being used appropriately
            level_hits = level_stats.get("geometric_hits", 0)
            level_misses = level_stats.get("geometric_misses", 0)
            level_total = level_hits + level_misses
            if level_total > 0:
                fast_path_rate = level_hits / level_total
                stats["sub_node_fast_path_rate"] = fast_path_rate
                stats["sub_node_slow_path_rate"] = 1.0 - fast_path_rate

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
