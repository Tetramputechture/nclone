"""
Path distance calculator with caching support.

Provides CachedPathDistanceCalculator which consolidates:
- A* pathfinding algorithm for shortest path calculation
- Per-query caching for frequently accessed paths
- Level-based precomputed caching for goal distances
- Step-level caching for within-step duplicate calls
- Integration with shared memory caching for multi-worker training
"""

import logging
from typing import Dict, Tuple, Optional, Any, List
import heapq

from ..level_data import LevelData
from .level_data_helpers import get_mine_state_signature
from .node_lookup import (
    find_closest_node_to_position,
    extract_spatial_lookups_from_graph_data,
    find_ninja_node,
)
from .path_distance_cache import LevelBasedPathDistanceCache
from .mine_proximity_cache import MineProximityCostCache, create_mine_sdf
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
        enable_timing: bool = False,
        shared_level_cache: Optional[Any] = None,
        shared_level_caches_by_stage: Optional[Dict[int, Any]] = None,
        goal_curriculum_manager: Optional[Any] = None,
    ):
        """
        Initialize cached path distance calculator.

        Args:
            max_cache_size: Maximum number of cached distance queries
                          (increased to 5000 for physics-optimal path caching;
                          physics paths are deterministic for (start, goal, mine_state) pairs)
            enable_timing: Enable performance timing instrumentation
            shared_level_cache: Optional SharedLevelCache for zero-copy multi-worker training
            shared_level_caches_by_stage: Optional dict of stage -> SharedLevelCache for curriculum
            goal_curriculum_manager: Optional IntermediateGoalManager for stage selection
        """
        # Per-query cache for physics-optimal paths (start_node, goal_node) → distance
        # Physics costs are deterministic, so aggressive caching is beneficial
        self.cache: Dict[Tuple, float] = {}
        self.max_cache_size = max(
            max_cache_size, 5000
        )  # At least 5000 for physics paths
        self.hits = 0
        self.misses = 0

        # Store shared cache references (used instead of local caches if provided)
        self._shared_level_cache = shared_level_cache
        self._shared_level_caches_by_stage = shared_level_caches_by_stage
        self.goal_curriculum_manager = goal_curriculum_manager

        # Determine which shared cache to use
        has_shared_cache = (
            shared_level_cache is not None or shared_level_caches_by_stage is not None
        )

        # Level-based cache for precomputed distances
        # If shared cache provided, this will be set to a view wrapper
        self.level_cache: Optional[LevelBasedPathDistanceCache] = (
            None if has_shared_cache else LevelBasedPathDistanceCache()
        )
        self.level_data: Optional[LevelData] = None

        # Mine proximity cost cache for hazard avoidance
        # If shared cache provided, this will be set to a view wrapper
        self.mine_proximity_cache: Optional[MineProximityCostCache] = (
            None if shared_level_cache is not None else MineProximityCostCache()
        )

        # Signed Distance Field to mines for O(1) observation space features
        # If shared cache provided, this will be set to a view wrapper
        self.mine_sdf = None if shared_level_cache is not None else create_mine_sdf()

        # Track current level to avoid redundant cache rebuilds
        self._current_level_id: Optional[str] = None

        # Track current curriculum stage to detect stage changes (for multi-stage caches)
        self._current_curriculum_stage: Optional[int] = None

        # Mine / locked-door signature mirrors (L1.5): invalidate cache on toggle-mine updates
        self._cached_mine_signature: Optional[Tuple] = None
        self._cached_switch_signature: Optional[Tuple] = None

        # Spatial hash for fast node lookup (updated per level)
        self._spatial_hash: Optional[any] = None

        # OPTIMIZATION: Cache entity positions per level for goal_id inference
        # This avoids repeated entity lookups in hot path
        self._cached_switch_positions: List[Tuple[int, int]] = []
        self._cached_exit_positions: List[Tuple[int, int]] = []

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

        # OPTIMIZATION: Step-scoped ninja node cache to avoid redundant find_ninja_node
        # Key: (rounded_pos, goal_id) -> ninja_node. Cleared each step.
        # Same ninja position queried 5x/step for observation + reward; cache to 1-2 lookups.
        self._step_ninja_node_cache: Dict[
            Tuple[int, int, Optional[str]], Tuple[int, int]
        ] = {}

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

    def _switch_to_local_caches(self) -> None:
        """Detach shared-memory views; build fresh local caches for dynamic mine/switch updates."""
        self.level_cache = LevelBasedPathDistanceCache()
        self.mine_proximity_cache = MineProximityCostCache()
        self.mine_sdf = create_mine_sdf()

    def _select_curriculum_cache(self) -> Optional[Any]:
        """Select appropriate SharedLevelCache based on current curriculum stage.

        When goal curriculum is active, selects the precomputed cache for the current
        stage from multi-stage caches. This enables SharedLevelCache to work with
        curriculum without rebuilding caches locally.

        Returns:
            SharedLevelCache for current stage, or None if no caches available
        """
        # Goal curriculum active - select stage-specific cache
        if self.goal_curriculum_manager is not None:
            stage = self.goal_curriculum_manager.state.unified_stage

            # Use multi-stage caches if available (precomputed for each stage)
            if self._shared_level_caches_by_stage is not None:
                selected_cache = self._shared_level_caches_by_stage.get(stage)
                if selected_cache is not None:
                    logger.debug(
                        f"[CURRICULUM_CACHE] Using precomputed cache for stage {stage} "
                        f"({selected_cache.num_nodes} nodes, {selected_cache.num_goals} goals)"
                    )
                    return selected_cache
                else:
                    logger.warning(
                        f"[CURRICULUM_CACHE] No cache for stage {stage}, falling back to local build. "
                        f"Available stages: {list(self._shared_level_caches_by_stage.keys())}"
                    )
                    return None

            # No multi-stage caches - fall back to local cache building
            logger.debug(
                f"[CURRICULUM_CACHE] Goal curriculum active (stage {stage}) but no "
                f"multi-stage caches available. Using local cache with curriculum positions."
            )
            return None

        # No curriculum active - use shared caches if available
        if self._shared_level_caches_by_stage is not None:
            # Multi-stage mode (shouldn't happen without curriculum, but handle it)
            selected_cache = self._shared_level_caches_by_stage.get(0)
            if selected_cache is not None:
                logger.debug(
                    f"[SHARED_CACHE] Using multi-stage cache stage 0 "
                    f"({selected_cache.num_nodes} nodes, {selected_cache.num_goals} goals)"
                )
            return selected_cache

        # Standard mode: return single shared cache
        return self._shared_level_cache

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

        # Get current curriculum stage (for stage change detection)
        current_stage = (
            self.goal_curriculum_manager.state.unified_stage
            if self.goal_curriculum_manager
            else None
        )

        # Detect level or stage changes
        level_changed = (
            self._current_level_id is None
            or level_data.level_id != self._current_level_id
        )
        stage_changed = self._current_curriculum_stage != current_stage

        new_mine_sig = get_mine_state_signature(level_data)
        new_switch_sig = tuple(sorted(level_data.switch_states.items()))

        # Fast rejection: same level, stage, and mine/switch world state
        if not level_changed and not stage_changed:
            if (
                self._cached_mine_signature is not None
                and new_mine_sig == self._cached_mine_signature
                and new_switch_sig == self._cached_switch_signature
            ):
                return False

        # OPTIMIZATION: If shared cache is provided, use view wrappers instead of building locally
        # Select appropriate cache based on curriculum stage
        active_shared_cache = self._select_curriculum_cache()

        if active_shared_cache is not None:
            # Check if we need to switch to a different stage's cache
            # Triggers on: no cache, stage change, or cache object mismatch
            needs_cache_switch = (
                self.level_cache is None
                or stage_changed
                or (
                    hasattr(self.level_cache, "shared_cache")
                    and self.level_cache.shared_cache != active_shared_cache
                )
            )

            if needs_cache_switch:
                # Initialize or switch to view wrappers for shared cache
                # NOTE: This path only executes when curriculum is NOT active
                # (curriculum active returns None from _select_curriculum_cache)
                self.level_cache = active_shared_cache.get_path_cache_view()
                self.mine_proximity_cache = (
                    active_shared_cache.get_mine_proximity_view()
                )
                self.mine_sdf = active_shared_cache.get_sdf_view()

                # Store level data for consistency checks
                self.level_data = level_data
                self._current_level_id = level_data.level_id
                self._current_curriculum_stage = current_stage

                # Update cached entity positions from level_data for goal_id inference
                from ...constants.entity_types import EntityType

                self._cached_switch_positions = [
                    (entity.get("x", 0), entity.get("y", 0))
                    for entity in level_data.get_entities_by_type(
                        EntityType.EXIT_SWITCH
                    )
                ]
                self._cached_exit_positions = [
                    (entity.get("x", 0), entity.get("y", 0))
                    for entity in level_data.get_entities_by_type(EntityType.EXIT_DOOR)
                ]

                self._cached_mine_signature = new_mine_sig
                self._cached_switch_signature = new_switch_sig

                stage_info = (
                    f", stage={current_stage}" if current_stage is not None else ""
                )
                logger.debug(
                    f"[SHARED_CACHE] Switched to shared cache: "
                    f"{active_shared_cache.num_nodes} nodes, {active_shared_cache.num_goals} goals"
                    f"{stage_info}"
                )

                return True  # "Rebuilt" by using/switching shared cache

            if (
                self._cached_mine_signature is not None
                and new_mine_sig == self._cached_mine_signature
                and new_switch_sig == self._cached_switch_signature
            ):
                return False

            # Static shared views cannot reflect toggled mines — rebuild locally.
            logger.debug(
                "[SHARED_CACHE] Mine/switch signature changed; switching to local caches"
            )
            self._switch_to_local_caches()

        # Standard path: build caches locally
        if self.level_cache is None:
            self.level_cache = LevelBasedPathDistanceCache()

        # Build mine proximity cache FIRST (needed for level cache building)
        # CRITICAL: This MUST be done before level cache building to ensure
        # BFS pathfinding uses correct mine avoidance costs
        mine_cache_rebuilt = False
        if self.mine_proximity_cache is not None:
            mine_cache_rebuilt = self.mine_proximity_cache.build_cache(
                level_data, adjacency, graph_data
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

            mines = level_data.get_all_toggle_mines()

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
            self.mine_sdf.build_sdf(level_data, graph_data)
            if self.mine_sdf is not None
            else False
        )

        # Then build level cache (uses mine proximity cache and SDF during BFS)
        rebuilt = self.level_cache.build_cache(
            level_data,
            adjacency,
            base_adjacency,
            graph_data,
            self.mine_proximity_cache,
        )

        if rebuilt or mine_cache_rebuilt or sdf_rebuilt:
            self.level_data = level_data
            # Update level ID tracking so get_distance() knows cache is ready
            # LevelData.__post_init__ always sets level_id, so we can trust it exists
            self._current_level_id = level_data.level_id
            self._current_curriculum_stage = current_stage

            # Cache adjacency bounds for fast bounds checking
            if adjacency:
                # Single-pass calculation of all bounds (was 4 separate iterations)
                min_x = max_x = min_y = max_y = None
                for nx, ny in adjacency:
                    if min_x is None:
                        min_x = max_x = nx
                        min_y = max_y = ny
                    else:
                        if nx < min_x:
                            min_x = nx
                        elif nx > max_x:
                            max_x = nx
                        if ny < min_y:
                            min_y = ny
                        elif ny > max_y:
                            max_y = ny
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

        # Record signatures after any local build attempt so fast-reject matches current world.
        self._cached_mine_signature = new_mine_sig
        self._cached_switch_signature = new_switch_sig

        return rebuilt or mine_cache_rebuilt or sdf_rebuilt

    def _calculate_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        return_path: bool = False,
    ):
        """Calculate shortest path distance using geometric A*."""
        if start not in adjacency or goal not in adjacency:
            if return_path:
                return float("inf"), []
            return float("inf")
        if start == goal:
            if return_path:
                return 0.0, [start]
            return 0.0

        return self._astar_distance(start, goal, adjacency, base_adjacency, return_path)

    def _astar_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        adjacency: Dict,
        base_adjacency: Dict,
        return_path: bool = False,
    ):
        """A* pathfinding with geometric (euclidean) edge costs."""

        def manhattan_heuristic(pos: Tuple[int, int]) -> float:
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        open_set = [(manhattan_heuristic(start), 0.0, start)]
        g_score = {start: 0.0}
        visited = set()
        parents = {start: None}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                if return_path:
                    path = []
                    node = current
                    while node is not None:
                        path.append(node)
                        node = parents.get(node)
                    path.reverse()
                    return current_g, path
                return current_g

            for neighbor, _ in adjacency.get(current, []):
                if neighbor in visited:
                    continue

                dx = neighbor[0] - current[0]
                dy = neighbor[1] - current[1]
                edge_cost = (dx * dx + dy * dy) ** 0.5
                tentative_g = current_g + edge_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    parents[neighbor] = current
                    f_score = tentative_g + manhattan_heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        if return_path:
            return float("inf"), []
        return float("inf")

    def _reconstruct_path_from_next_hop(
        self,
        start_node: Tuple[int, int],
        goal_node: Tuple[int, int],
        goal_id: str,
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from start_node to goal_node using level cache next_hop chain."""
        if self.level_cache is None:
            return []
        path: List[Tuple[int, int]] = []
        current = start_node
        seen = set()
        while current is not None:
            if current in seen:
                break
            seen.add(current)
            path.append(current)
            if current == goal_node:
                break
            next_hop = self.level_cache.get_next_hop(current, goal_id)
            current = next_hop
        return path

    def get_path_cost(
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
        goal_id: Optional[str] = None,
        return_path: bool = False,
    ):
        """
        Get geometric path cost with caching and spatial indexing.

        This is the single unified method for all path cost queries, used by
        both training (PBRS) and visualization. Uses geometric (euclidean)
        distances only.

        Args:
            start: Start position (world/full map space)
            goal: Goal position (world/full map space)
            adjacency: Masked graph adjacency
            base_adjacency: Base graph adjacency (pre-entity-mask)
            cache_key: Optional key for cache invalidation
            level_data: Optional level data for level-based caching
            graph_data: Optional graph data for spatial_hash extraction
            entity_radius: Collision radius of the goal entity (default 0.0)
            ninja_radius: Collision radius of the ninja (default 10.0)
            goal_id: Optional goal identifier for level_cache lookup (e.g., "switch" or "exit")
            return_path: If True, returns (cost, path_list) for visualization.

        Returns:
            float: Path cost (when return_path=False)
            Tuple[float, List]: (cost, path_node_list) (when return_path=True)
        """
        # OPTIMIZATION: Step-level cache check (eliminates duplicate calls within same step)
        # Skip cache when return_path=True (paths are not cached, only costs)
        step_key = (start, goal, entity_radius)
        if not return_path and step_key in self._step_cache:
            self._step_cache_hits += 1
            return self._step_cache[step_key]
        if not return_path:
            self._step_cache_misses += 1

        # CRITICAL: Early exit for invalid goal positions
        # Goals at (0, 0) indicate entity loading failure - return inf to avoid cache pollution
        if goal[0] == 0 and goal[1] == 0:
            if return_path:
                return float("inf"), []
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
            if return_path:
                return 0.0, []
            return 0.0

        # OPTIMIZATION: Early per-query cache check before expensive operations
        # Skip caches when return_path=True (caches only store costs, not paths)
        start_grid = (start[0] // 24, start[1] // 24)
        goal_grid = (goal[0] // 24, goal[1] // 24)
        key = (start_grid, goal_grid, cache_key)

        if not return_path and key in self.cache:
            self.hits += 1
            cached_distance = self.cache[key]
            # Adjust for collision radii
            adjusted_dist = max(0.0, cached_distance - (ninja_radius + entity_radius))
            return adjusted_dist

        # Try level cache first if level_data is provided
        # When return_path=True, reconstruct path from level cache parents (next_hop chain)
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

                    # Log goal_id inference for curriculum debugging
                    if goal_id is not None:
                        current_stage = (
                            self.goal_curriculum_manager.state.unified_stage
                            if self.goal_curriculum_manager
                            else "N/A"
                        )
                        logger.debug(
                            f"[GOAL_ID_INFER] goal={goal}, inferred goal_id={goal_id}, "
                            f"cached_switch_positions={self._cached_switch_positions[:2] if len(self._cached_switch_positions) > 0 else []}, "
                            f"cached_exit_positions={self._cached_exit_positions[:2] if len(self._cached_exit_positions) > 0 else []}, "
                            f"stage={current_stage}"
                        )

                if goal_id is not None:
                    # Get goal_pos for validation
                    goal_pos = self.level_cache.get_goal_pos_from_id(goal_id)

                    if goal_pos is not None:
                        # Validate that cached goal_pos matches input goal (within tolerance)
                        # This prevents returning wrong cached distance if multiple goals map to same node
                        dx = abs(goal_pos[0] - goal[0])
                        dy = abs(goal_pos[1] - goal[1])

                        # Log validation failures with curriculum context
                        if dx > 6 or dy > 6:
                            current_stage = (
                                self.goal_curriculum_manager.state.unified_stage
                                if self.goal_curriculum_manager
                                else "N/A"
                            )
                            logger.warning(
                                f"[VALIDATION_FAIL] get_distance: goal_id={goal_id}, "
                                f"goal={goal} (from caller), goal_pos={goal_pos} (from cache), "
                                f"dx={dx:.1f}, dy={dy:.1f}, threshold=6, stage={current_stage}. "
                                f"Cache/entity position mismatch. Skipping level cache."
                            )

                        if (
                            dx <= 6 and dy <= 6
                        ):  # Within one sub-node (matching graph resolution)
                            # Find ninja node using canonical ninja node selection
                            # FIX: Pass level_cache to use PATH distance instead of Euclidean
                            with self.timer.measure("find_start_node"):
                                start_node = self._get_cached_or_find_ninja_node(
                                    start,
                                    adjacency,
                                    goal_id=goal_id,
                                    goal_node=goal_node,
                                    subcell_lookup=subcell_lookup,
                                    ninja_radius=ninja_radius,
                                )

                            # Try level cache with snapped start position
                            if start_node is not None and start_node in adjacency:
                                cached_dist = None
                                path_dir = None  # (dx, dy) normalized for sub-node

                                # Try Python level cache
                                if cached_dist is None:
                                    cached_dist = self.level_cache.get_distance(
                                        start_node, goal_pos, goal_id
                                    )
                                    if cached_dist != float("inf"):
                                        next_hop = self.level_cache.get_next_hop(
                                            start_node, goal_id
                                        )
                                        if next_hop is not None:
                                            from .pathfinding_utils import (
                                                NODE_WORLD_COORD_OFFSET,
                                            )

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
                                            path_dx = (
                                                next_hop_world_x - start_node_world_x
                                            )
                                            path_dy = (
                                                next_hop_world_y - start_node_world_y
                                            )
                                            path_len = (
                                                path_dx * path_dx + path_dy * path_dy
                                            ) ** 0.5
                                            if path_len > 0.001:
                                                path_dir = (
                                                    path_dx / path_len,
                                                    path_dy / path_len,
                                                )

                                if cached_dist != float("inf"):
                                    # === SUB-NODE PBRS RESOLUTION ===
                                    from .pathfinding_utils import (
                                        NODE_WORLD_COORD_OFFSET,
                                    )

                                    start_node_world_x = (
                                        start_node[0] + NODE_WORLD_COORD_OFFSET
                                    )
                                    start_node_world_y = (
                                        start_node[1] + NODE_WORLD_COORD_OFFSET
                                    )
                                    if path_dir is not None:
                                        path_dir_x, path_dir_y = path_dir
                                        player_dx = start[0] - start_node_world_x
                                        player_dy = start[1] - start_node_world_y
                                        projection = (
                                            player_dx * path_dir_x
                                            + player_dy * path_dir_y
                                        )
                                        total_dist = max(
                                            0.0,
                                            cached_dist
                                            - projection
                                            - (ninja_radius + entity_radius),
                                        )
                                        self._step_cache[step_key] = total_dist
                                        if return_path:
                                            path = self._reconstruct_path_from_next_hop(
                                                start_node, goal_node, goal_id
                                            )
                                            return total_dist, path
                                        return total_dist

                                    result = max(
                                        0.0,
                                        cached_dist - (ninja_radius + entity_radius),
                                    )
                                    self._step_cache[step_key] = result
                                    if return_path:
                                        path = self._reconstruct_path_from_next_hop(
                                            start_node, goal_node, goal_id
                                        )
                                        return result, path
                                    return result
                            # Level cache couldn't be used (start_node None or not in
                            # adjacency). Fall through to cache miss - compute path.
            # Goal node not found or level cache unusable - fall through to compute

        # Cache miss - compute
        # First snap positions to nodes before calculating distance using smart selection
        from .pathfinding_utils import (
            find_goal_node_closest_to_start,
        )

        # Find start node (temp_start_node) for goal selection
        # Use standard radius first, then fallback to extended radius
        temp_start_node = self._get_cached_or_find_ninja_node(
            start,
            adjacency,
            subcell_lookup=subcell_lookup,
            ninja_radius=ninja_radius,
        )

        # Fallback: If default radius (10px) fails for start, try with progressively larger radii
        # This handles cases where the start position (e.g., switch) is far from any graph node
        if temp_start_node is None:
            temp_start_node = self._get_cached_or_find_ninja_node(
                start,
                adjacency,
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
            temp_start_node = self._get_cached_or_find_ninja_node(
                start,
                adjacency,
                subcell_lookup=subcell_lookup,
                ninja_radius=ninja_radius,
                search_radius_override=150.0,
            )
            if temp_start_node is not None:
                logger.debug(
                    f"Start node found with large search radius (150px): "
                    f"start={start}, node={temp_start_node}"
                )

        # Ultimate fallback: Find ANY closest node in the entire adjacency graph.
        # Guard: if the start position is far outside the adjacency bounding box
        # (e.g. ninja fell below the map due to OOB spawn), skip the O(N) scan
        # and return inf immediately so we don't compute meaningless distances.
        if temp_start_node is None and adjacency:
            from .pathfinding_utils import NODE_WORLD_COORD_OFFSET

            start_tile_x = start[0] - NODE_WORLD_COORD_OFFSET
            start_tile_y = start[1] - NODE_WORLD_COORD_OFFSET

            _OOB_MARGIN = 200  # px beyond graph bounding box
            if self._adjacency_bounds is not None:
                _bmin_x, _bmax_x, _bmin_y, _bmax_y = self._adjacency_bounds
                if (
                    start_tile_x < _bmin_x - _OOB_MARGIN
                    or start_tile_x > _bmax_x + _OOB_MARGIN
                    or start_tile_y < _bmin_y - _OOB_MARGIN
                    or start_tile_y > _bmax_y + _OOB_MARGIN
                ):
                    logger.warning(
                        f"Start position {start} is far outside graph bounds "
                        f"(tile_xy=({start_tile_x:.0f},{start_tile_y:.0f}), "
                        f"bounds=[{_bmin_x},{_bmax_x}]x[{_bmin_y},{_bmax_y}]); "
                        f"returning inf to avoid meaningless distance computation."
                    )
                    if return_path:
                        return float("inf"), []
                    return float("inf")

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
        start_node = self._get_cached_or_find_ninja_node(
            start,
            adjacency,
            goal_id=goal_id,
            goal_node=goal_node,
            subcell_lookup=subcell_lookup,
            ninja_radius=ninja_radius,
        )

        # Fallback: If default radius fails for final start_node, try with progressively larger radii
        if start_node is None:
            start_node = self._get_cached_or_find_ninja_node(
                start,
                adjacency,
                goal_id=goal_id,
                goal_node=goal_node,
                subcell_lookup=subcell_lookup,
                ninja_radius=ninja_radius,
                search_radius_override=48.0,
            )

        if start_node is None:
            start_node = self._get_cached_or_find_ninja_node(
                start,
                adjacency,
                goal_id=goal_id,
                goal_node=goal_node,
                subcell_lookup=subcell_lookup,
                ninja_radius=ninja_radius,
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
            if return_path:
                return float("inf"), []
            return float("inf")

        self.misses += 1
        with self.timer.measure("pathfinding_compute"):
            result = self._calculate_distance(
                start_node,
                goal_node,
                adjacency,
                base_adjacency,
                return_path=return_path,
            )

        if return_path:
            distance, path = result
            total_dist = max(0.0, distance - (ninja_radius + entity_radius))
            return total_dist, path

        distance = result

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

    def compute_baseline_physics_costs(
        self,
        spawn_pos: Tuple[int, int],
        switch_pos: Tuple[int, int],
        exit_pos: Tuple[int, int],
        adjacency: Dict,
        base_adjacency: Dict,
        level_data: Any,
        graph_data: Optional[Dict],
    ) -> Tuple[float, float, float]:
        """
        Compute baseline physics costs for PBRS normalization.

        Uses zero velocity and full jump duration to get "neutral" baseline costs
        representing the minimum physics cost to traverse each path segment without
        velocity state dependency.

        Args:
            spawn_pos: Spawn position (x, y)
            switch_pos: Switch position (x, y)
            exit_pos: Exit position (x, y)
            adjacency: Masked graph adjacency structure
            base_adjacency: Base graph adjacency (pre-entity-mask)
            level_data: LevelData instance
            graph_data: Graph data dict with node_physics, spatial_hash

        Returns:
            (spawn_to_switch_cost, switch_to_exit_cost, combined_cost)
        """
        from ...constants.physics_constants import (
            EXIT_SWITCH_RADIUS,
            EXIT_DOOR_RADIUS,
            NINJA_RADIUS,
        )

        # Compute spawn→switch geometric cost
        spawn_to_switch = self.get_path_cost(
            spawn_pos,
            switch_pos,
            adjacency,
            base_adjacency,
            level_data=level_data,
            graph_data=graph_data,
            entity_radius=EXIT_SWITCH_RADIUS,
            ninja_radius=NINJA_RADIUS,
        )

        # Compute switch→exit geometric cost
        switch_to_exit = self.get_path_cost(
            switch_pos,
            exit_pos,
            adjacency,
            base_adjacency,
            level_data=level_data,
            graph_data=graph_data,
            entity_radius=EXIT_DOOR_RADIUS,
            ninja_radius=NINJA_RADIUS,
        )

        # Handle infinite costs (unreachable)
        if spawn_to_switch == float("inf"):
            logger.warning(
                f"Baseline physics cost spawn→switch is infinite. "
                f"Positions: spawn={spawn_pos}, switch={switch_pos}"
            )
            spawn_to_switch = 2000.0  # Fallback

        if switch_to_exit == float("inf"):
            logger.warning(
                f"Baseline physics cost switch→exit is infinite. "
                f"Positions: switch={switch_pos}, exit={exit_pos}"
            )
            switch_to_exit = 2000.0  # Fallback

        combined = spawn_to_switch + switch_to_exit

        logger.debug(
            f"[PBRS] Baseline physics costs: "
            f"spawn→switch={spawn_to_switch:.1f}, "
            f"switch→exit={switch_to_exit:.1f}, "
            f"combined={combined:.1f}"
        )

        return spawn_to_switch, switch_to_exit, combined

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
        ninja_velocity: Optional[Tuple[float, float]] = None,
    ) -> float:
        """DEPRECATED: Use get_path_cost instead (all distances are geometric). Kept for backward compatibility."""
        return self.get_path_cost(
            start,
            goal,
            adjacency,
            base_adjacency,
            level_data=level_data,
            graph_data=graph_data,
            entity_radius=entity_radius,
            ninja_radius=ninja_radius,
            goal_id=goal_id,
        )

    def _get_cached_or_find_ninja_node(
        self,
        start: Tuple[float, float],
        adjacency: Dict,
        goal_id: Optional[str] = None,
        goal_node: Optional[Tuple[int, int]] = None,
        subcell_lookup: Optional[Any] = None,
        ninja_radius: float = 10.0,
        search_radius_override: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[Tuple[int, int]]:
        """Get ninja node with step-scoped caching to avoid redundant find_ninja_node.

        Same ninja position is typically queried 5+ times per step across
        observation and reward. Caching reduces to 1 lookup per step.
        Also checks position-only key (x, y, None) so callers with different
        goal_ids share the same result (ninja node is independent of goal).
        """
        pos = (int(start[0]), int(start[1]))
        cache_key = (pos[0], pos[1], goal_id)
        if cache_key in self._step_ninja_node_cache:
            return self._step_ninja_node_cache[cache_key]
        # Position-only fallback: ninja node is same regardless of goal_id
        pos_only_key = (pos[0], pos[1], None)
        if pos_only_key in self._step_ninja_node_cache:
            node = self._step_ninja_node_cache[pos_only_key]
            self._step_ninja_node_cache[cache_key] = node
            return node
        node = find_ninja_node(
            start,
            adjacency,
            spatial_hash=self._spatial_hash,
            subcell_lookup=subcell_lookup,
            ninja_radius=ninja_radius,
            goal_node=goal_node,
            level_cache=self.level_cache,
            goal_id=goal_id,
            search_radius_override=search_radius_override,
            **kwargs,
        )
        if node is not None:
            self._step_ninja_node_cache[cache_key] = node
            self._step_ninja_node_cache[pos_only_key] = node
        return node

    def clear_step_cache(self):
        """Clear step-level cache (call between environment steps)."""
        self._step_cache.clear()
        self._step_cache_geo.clear()
        self._step_ninja_node_cache.clear()

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
        self._current_curriculum_stage = None
        self._spatial_hash = None
        self._adjacency_bounds = None

        # Clear cached entity positions
        self._cached_switch_positions.clear()
        self._cached_exit_positions.clear()

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
