"""
Level-based path distance cache for precomputed shortest path distances.

Uses flood fill (BFS) from goal positions to precompute all node-to-goal distances.
Cache invalidates when level or mine states change.
"""

import logging
from typing import Dict, Tuple, List, Optional, Any
from ..level_data import LevelData
from .pathfinding_utils import (
    find_closest_node_to_position,
    bfs_distance_from_start,
    extract_spatial_lookups_from_graph_data,
)
from .level_data_helpers import extract_goal_positions, get_mine_state_signature

logger = logging.getLogger(__name__)


class LevelBasedPathDistanceCache:
    """
    Precomputed cache for GEOMETRIC distances from nodes to goals.

    ARCHITECTURE: Caches only GEOMETRIC distances (pixels), not physics costs.
    Physics costs are directional (momentum, gravity) and cannot be cached
    via flood-fill. They must be computed on-demand via find_shortest_path
    from the actual start position to goal.

    Uses flood fill (BFS) from each goal position to precompute GEOMETRIC
    distances to all reachable nodes. Also stores next_hop navigation.

    Cache invalidates when level or mine states change.

    For physics-optimal paths: Use find_shortest_path with physics costs FROM
    current position TO goal (not cached, computed fresh each time).
    """

    def __init__(self):
        """Initialize level-based path distance cache."""
        # Cache mapping (node_position, goal_id) -> physics-weighted distance
        self.cache: Dict[Tuple[Tuple[int, int], str], float] = {}
        # Cache mapping (node_position, goal_id) -> geometric distance (pixels)
        # Used for PBRS normalization where actual path length is needed
        self.geometric_cache: Dict[Tuple[Tuple[int, int], str], float] = {}
        # Cache mapping (node_position, goal_id) -> next_hop_node (toward goal)
        # Used for sub-node PBRS resolution: direction to next_hop = optimal path direction
        self.next_hop_cache: Dict[
            Tuple[Tuple[int, int], str], Optional[Tuple[int, int]]
        ] = {}
        # Cache mapping (node_position, goal_id) -> multi_hop_direction (weighted lookahead)
        # Used for anticipatory path guidance: shows direction accounting for upcoming curvature
        # Stores normalized direction vector (dx, dy) weighted across next 3-5 hops
        self.multi_hop_direction_cache: Dict[
            Tuple[Tuple[int, int], str], Optional[Tuple[float, float]]
        ] = {}
        # Mapping of goal_node -> goal_id for quick lookup
        self._goal_node_to_goal_id: Dict[Tuple[int, int], str] = {}
        # Mapping of goal_id -> goal_pos for validation
        self._goal_id_to_goal_pos: Dict[str, Tuple[int, int]] = {}
        self._cached_level_data: Optional[LevelData] = None
        self._cached_mine_signature: Optional[Tuple] = None
        self._cached_switch_signature: Optional[Tuple] = None
        self._cached_waypoint_signature: Optional[Tuple] = None
        self.level_cache_hits = 0
        self.level_cache_misses = 0
        self.geometric_cache_hits = 0
        self.geometric_cache_misses = 0

    def get_distance(
        self, node_pos: Tuple[int, int], goal_pos: Tuple[int, int], goal_id: str
    ) -> float:
        """
        Get cached GEOMETRIC distance from node to goal.

        IMPORTANT: This returns GEOMETRIC distance (pixels), NOT physics-weighted cost!
        Physics costs are directional and cannot be cached via flood-fill.
        For physics-optimal paths, use find_shortest_path from start to goal.

        Args:
            node_pos: Node position (x, y) in pixels
            goal_pos: Goal position (x, y) in pixels (for validation)
            goal_id: Unique identifier for the goal

        Returns:
            Cached geometric distance in pixels, or float('inf') if not cached
        """
        key = (node_pos, goal_id)
        if key in self.cache:
            self.level_cache_hits += 1
            return self.cache[key]

        self.level_cache_misses += 1
        return float("inf")

    def get_geometric_distance(
        self, node_pos: Tuple[int, int], goal_pos: Tuple[int, int], goal_id: str
    ) -> float:
        """
        Get cached GEOMETRIC distance from node to goal (actual pixels).

        Unlike get_distance() which returns physics-weighted costs, this returns
        the actual path length in pixels. Use this for PBRS normalization.

        Args:
            node_pos: Node position (x, y) in pixels
            goal_pos: Goal position (x, y) in pixels (for validation)
            goal_id: Unique identifier for the goal

        Returns:
            Cached geometric distance in pixels, or float('inf') if not cached
        """
        key = (node_pos, goal_id)
        if key in self.geometric_cache:
            self.geometric_cache_hits += 1
            return self.geometric_cache[key]

        self.geometric_cache_misses += 1

        # # DIAGNOSTIC: Log cache miss details (only first few to avoid spam)
        # if self.geometric_cache_misses <= 10:
        #     available_goals = set(
        #         k[1] for k in self.geometric_cache.keys() if k[0] == node_pos
        #     )
        #     # Check if node exists in cache at all
        #     node_exists = any(k[0] == node_pos for k in self.geometric_cache.keys())
        #     # Sample a few cached nodes near this position
        #     nearby_nodes = [
        #         k
        #         for k in self.geometric_cache.keys()
        #         if abs(k[0][0] - node_pos[0]) < 50 and abs(k[0][1] - node_pos[1]) < 50
        #     ]
        #     logger.warning(
        #         f"[LEVEL_CACHE_MISS #{self.geometric_cache_misses}] node={node_pos}, goal_id={goal_id}, "
        #         f"node_in_cache={node_exists}, available_goals_for_node={available_goals}, "
        #         f"all_cached_goals={set(self._goal_id_to_goal_pos.keys())}, "
        #         f"nearby_cached_entries={len(nearby_nodes)}, sample={nearby_nodes[:3]}"
        #     )

        return float("inf")

    def get_next_hop(
        self, node_pos: Tuple[int, int], goal_id: str
    ) -> Optional[Tuple[int, int]]:
        """
        Get the next hop node toward the goal from the given node.

        The next hop is the neighbor that's one step closer to the goal
        along the optimal path. This is used for sub-node PBRS resolution:
        the direction from player to next_hop is the optimal path direction,
        respecting the adjacency graph (walls, corridors, etc.).

        Args:
            node_pos: Current node position
            goal_id: Goal identifier

        Returns:
            Next hop node position, or None if not cached or at goal
        """
        return self.next_hop_cache.get((node_pos, goal_id))

    def get_multi_hop_direction(
        self, node_pos: Tuple[int, int], goal_id: str
    ) -> Optional[Tuple[float, float]]:
        """
        Get multi-hop lookahead direction toward goal from given node.

        This provides a weighted direction vector that accounts for upcoming
        path curvature, not just the immediate next step. The direction is
        computed as a weighted sum of the next 8 hops along the optimal path,
        with exponentially decaying weights (0.45, 0.25, 0.15, 0.08, ...).

        UPDATED 2025-12-20: Increased from 5 to 8 hops for better anticipation
        of inflection points. This solves the "sharp turn near hazard" problem
        by giving the agent visibility into upcoming turns, allowing it to adjust
        trajectory well before the critical inflection point.

        Args:
            node_pos: Current node position
            goal_id: Goal identifier

        Returns:
            Normalized direction vector (dx, dy), or None if not cached or at goal
        """
        return self.multi_hop_direction_cache.get((node_pos, goal_id))

    def _compute_multi_hop_direction(
        self,
        node_pos: Tuple[int, int],
        parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        max_hops: int = 8,
    ) -> Optional[Tuple[float, float]]:
        """
        Compute weighted multi-hop lookahead direction from node toward goal.

        Follows the parent chain (which points toward goal since BFS was from goal)
        up to max_hops steps, computing a weighted average direction vector.
        Weights decay exponentially: [0.45, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01] for 8 hops.

        UPDATED 2025-12-20: Increased from 5 to 8 hops for better inflection point anticipation.
        This allows the agent to "see" sharp turns further ahead, especially critical for
        turns near hazards where early trajectory adjustment is necessary.

        Args:
            node_pos: Starting node position
            parents: Parent map from BFS (parent is closer to goal)
            max_hops: Maximum number of hops to look ahead (default 8, increased from 5)

        Returns:
            Normalized direction vector (dx, dy), or None if at goal or no path
        """
        # Exponentially decaying weights for hops 1-8
        # Sum to ~1.0: [0.45, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01]
        weights = [0.45, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005][:max_hops]

        # Accumulate weighted direction vectors
        total_dx = 0.0
        total_dy = 0.0
        current_node = node_pos

        for i, weight in enumerate(weights):
            # Get next hop (parent is closer to goal)
            next_node = parents.get(current_node)

            if next_node is None:
                # Reached goal or dead end - stop accumulating
                break

            # Direction from current to next (toward goal)
            dx = float(next_node[0] - current_node[0])
            dy = float(next_node[1] - current_node[1])

            # Accumulate weighted direction
            total_dx += weight * dx
            total_dy += weight * dy

            # Move to next hop for next iteration
            current_node = next_node

        # Normalize the accumulated direction
        magnitude = (total_dx * total_dx + total_dy * total_dy) ** 0.5

        if magnitude < 0.001:
            # No meaningful direction (at goal or path too short)
            return None

        return (total_dx / magnitude, total_dy / magnitude)

    def _precompute_distances_from_goals(
        self,
        goals: List[Tuple[Tuple[int, int], str]],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        graph_data: Optional[Dict[str, Any]] = None,
        level_data: Optional[Any] = None,
        mine_proximity_cache: Optional[Any] = None,
        mine_sdf: Optional[Any] = None,
    ):
        """
        Precompute distances from all reachable nodes to each goal using flood fill.

        Computes TWO types of distances for each node in a SINGLE BFS pass:
        1. Physics-weighted costs (for pathfinding optimization)
        2. Geometric distances along the physics-optimal path (for PBRS normalization)

        IMPORTANT: The geometric distances are computed along the PHYSICS-OPTIMAL path,
        not a separate geometrically-shortest path. This ensures PBRS uses the correct
        path length when the physics-optimal path differs from the geometric-shortest.

        The adjacency graph should only contain nodes reachable from the initial
        player position. BFS traversal will only reach nodes in the adjacency graph,
        ensuring all cached distances are for reachable areas.

        Args:
            goals: List of (goal_position, goal_id) tuples
            adjacency: Masked graph adjacency structure (for pathfinding)
            base_adjacency: Base graph adjacency structure (pre-entity-mask, for physics checks)
            graph_data: Optional graph data dict with spatial_hash for fast lookup
            level_data: Optional LevelData for mine proximity checks (fallback)
            mine_proximity_cache: Optional MineProximityCostCache for mine cost lookup
            mine_sdf: Optional MineSignedDistanceField for velocity-aware hazard costs
        """
        # Clear existing cache and mappings
        self.cache.clear()
        self.geometric_cache.clear()
        self._goal_node_to_goal_id.clear()
        self._goal_id_to_goal_pos.clear()

        # Extract spatial hash and subcell lookup from graph_data if available
        spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
            graph_data
        )

        # PERFORMANCE OPTIMIZATION: Extract pre-computed physics cache from graph_data
        physics_cache = (
            graph_data.get("node_physics") if graph_data is not None else None
        )

        # Validate physics cache is available (required for physics-aware pathfinding)
        if physics_cache is None:
            raise ValueError(
                "Physics cache (node_physics) not found in graph_data. "
                "Level cache building requires physics cache for accurate path distances. "
                "Ensure graph building includes physics cache precomputation."
            )

        # DIAGNOSTIC: Log physics cache validation success
        # logger.warning(
        #     f"[LEVEL_CACHE_BUILD] Physics cache validated: {len(physics_cache)} nodes with physics properties"
        # )
        # Sample first node to verify structure
        # if physics_cache:
        #     sample_node = next(iter(physics_cache))
        #     sample_props = physics_cache[sample_node]
        #     # logger.warning(
        #     #     f"[LEVEL_CACHE_BUILD] Sample node {sample_node} physics: {sample_props}"
        #     # )

        # STRICT VALIDATION: Mine proximity cache must be provided for mine avoidance
        if mine_proximity_cache is None:
            raise ValueError(
                "mine_proximity_cache is None in _precompute_distances_from_goals. "
                "Cannot build level cache without mine avoidance costs. "
                "Paths will ignore mines, leading to dangerous/incorrect behavior."
            )

        # CRITICAL DIAGNOSTIC: Check mine cache population
        # mine_cache_size = (
        #     len(mine_proximity_cache.cache)
        #     if hasattr(mine_proximity_cache, "cache")
        #     else 0
        # )
        # from nclone.constants.entity_types import EntityType
        # from nclone.gym_environment.reward_calculation.reward_constants import (
        #     MINE_HAZARD_COST_MULTIPLIER,
        #     MINE_PENALIZE_DEADLY_ONLY,
        # )

        # mines = level_data.get_entities_by_type(
        #     EntityType.TOGGLE_MINE
        # ) + level_data.get_entities_by_type(EntityType.TOGGLE_MINE_TOGGLED)

        # # Count deadly mines (state 0) only if MINE_PENALIZE_DEADLY_ONLY is True
        # # This matches the logic in mine_proximity_cache._precompute_mine_proximity_costs
        # if MINE_PENALIZE_DEADLY_ONLY:
        #     deadly_mines = [m for m in mines if m.get("state", 0) == 0]
        # else:
        #     pass

        # logger.warning(
        #     f"[LEVEL_CACHE_BUILD] Mine avoidance state: "
        #     f"level_has_{len(mines)}_mines, deadly={len(deadly_mines)}, mine_cache_size={mine_cache_size}, "
        #     f"avoidance={'ACTIVE' if mine_cache_size > 0 else 'INACTIVE'}"
        # )

        # # STRICT: If level has deadly mines but cache is empty, mine cache wasn't built properly
        # # Only check deadly mines since cache only stores proximity costs for deadly mines
        # # NOTE: Cache can be empty if adjacency graph is empty or has no nodes within
        # # MINE_HAZARD_RADIUS (75px) of mines - check adjacency before raising error
        # if len(deadly_mines) > 0 and mine_cache_size == 0:
        #     if MINE_HAZARD_COST_MULTIPLIER > 1.0:
        #         # Check if adjacency is empty - if so, this is expected and not an error
        #         if not adjacency or len(adjacency) == 0:
        #             # Adjacency is empty - mine cache will be empty, this is expected
        #             # Don't raise error, just log warning
        #             import logging

        #             logger = logging.getLogger(__name__)
        #             logger.warning(
        #                 f"[LEVEL_CACHE_BUILD] Level has {len(deadly_mines)} deadly mines but adjacency graph is empty. "
        #                 f"Mine cache is empty (no nodes to cache). This is expected if graph building failed."
        #             )
        #         else:
        #             # Adjacency exists but cache is empty
        #             # This is VALID if mines are >MINE_HAZARD_RADIUS from all reachable nodes
        #             # (e.g., mines in unreachable areas). Just log a warning.
        #             from nclone.gym_environment.reward_calculation.reward_constants import (
        #                 MINE_HAZARD_RADIUS,
        #             )
        #             import logging

        #             logger = logging.getLogger(__name__)
        #             logger.warning(
        #                 f"[LEVEL_CACHE_BUILD] Level has {len(deadly_mines)} deadly mines but mine cache is empty. "
        #                 f"Adjacency has {len(adjacency)} nodes, but no nodes are within {MINE_HAZARD_RADIUS}px of mines. "
        #                 f"This is expected if mines are in unreachable areas or far from the navigable graph."
        #             )

        # For each goal, run BFS to compute distances to all reachable nodes
        # Use exact same logic as visualization in debug_overlay_renderer.py
        for goal_pos, goal_id in goals:
            # Find closest node to goal position using shared utility
            goal_node = find_closest_node_to_position(
                goal_pos,
                adjacency,
                threshold=50.0,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
            )

            # Only proceed if we found a valid goal node within threshold
            if goal_node is None:
                continue

            # Store mapping of goal_node -> goal_id for quick lookup
            self._goal_node_to_goal_id[goal_node] = goal_id
            # Store mapping of goal_id -> goal_pos for validation
            # CRITICAL: Store the ENTITY position (goal_pos), not the NODE position (goal_node)
            # The fast path validation compares this against the input goal entity position
            self._goal_id_to_goal_pos[goal_id] = goal_pos

            # CRITICAL FIX: Add generic aliases for backward compatibility
            # Code uses generic "switch" and "exit" but goals are stored as "exit_switch_0", etc.
            # Determine if we need to create alias cache entries BEFORE updating mappings
            needs_switch_alias = (
                goal_id.startswith("exit_switch_")
                and "switch" not in self._goal_id_to_goal_pos
            )
            needs_exit_alias = (
                goal_id.startswith("exit_door_")
                and "exit" not in self._goal_id_to_goal_pos
            )

            # Update goal position mappings for aliases (use entity position, not node)
            if needs_switch_alias:
                self._goal_id_to_goal_pos["switch"] = goal_pos
            elif needs_exit_alias:
                self._goal_id_to_goal_pos["exit"] = goal_pos

            # DIAGNOSTIC: Log before BFS to confirm mine cache is available
            # mine_cache_size = (
            #     len(mine_proximity_cache.cache)
            #     if mine_proximity_cache and hasattr(mine_proximity_cache, "cache")
            #     else 0
            # )
            # logger.warning(
            #     f"[LEVEL_CACHE_BFS] Computing distances from goal '{goal_id}' at {goal_pos}: "
            #     f"goal_node={goal_node}, mine_cache_size={mine_cache_size}, "
            #     f"adjacency_size={len(adjacency)}"
            # )

            # ARCHITECTURE: Cache only GEOMETRIC distances, compute physics on-demand
            #
            # Physics costs are DIRECTIONAL (momentum, gravity, grounded state):
            # - Cannot be cached from flood-fill (costs depend on approach direction)
            # - MUST be computed via find_shortest_path from actual start position
            #
            # Level cache stores:
            # ✓ Geometric distances (pixels, direction-independent) - FROM GOAL for efficiency
            # ✓ Next-hop navigation (topology, not physics costs)
            #
            # For physics-optimal paths (visualization, PBRS):
            # ✓ Use find_shortest_path FROM current position TO goal (correct directionality)

            # logger.warning(
            #     f"[LEVEL_CACHE_BFS] Computing GEOMETRIC distances from goal '{goal_id}' at {goal_pos}: "
            #     f"goal_node={goal_node}, adjacency_size={len(adjacency)}. "
            #     f"(Physics costs computed on-demand for correct directionality)"
            # )

            # Run BFS with GEOMETRIC costs only (direction-independent, cache-safe)
            geometric_distances, _, parents, _ = bfs_distance_from_start(
                goal_node,  # From goal (OK for geometric, efficient for caching)
                None,
                adjacency,
                base_adjacency,
                None,
                physics_cache,  # Still needed for horizontal rule validation
                level_data,
                None,  # NO mine_proximity_cache (not used with geometric costs)
                return_parents=True,
                use_geometric_costs=True,  # GEOMETRIC ONLY (direction-independent)
                track_geometric_distances=False,  # Not needed
                mine_sdf=None,  # Not used with geometric costs
            )

            # # DIAGNOSTIC: Log BFS results
            # logger.warning(
            #     f"[LEVEL_CACHE_BFS] Computed {len(geometric_distances)} geometric distances for goal '{goal_id}'"
            # )

            # Store geometric distances and next_hop for all computed nodes
            # NOTE: We do NOT store physics costs - those are direction-dependent
            # and must be computed via find_shortest_path from actual start position
            for node_pos, distance in geometric_distances.items():
                # Store geometric distance in BOTH caches for compatibility
                self.cache[(node_pos, goal_id)] = (
                    distance  # Legacy physics cache (now geometric)
                )
                self.geometric_cache[(node_pos, goal_id)] = distance

                # Parent in BFS from goal = next hop toward goal
                # (since we flooded outward from goal, parent is closer to goal)
                next_hop = parents.get(node_pos) if parents else None
                self.next_hop_cache[(node_pos, goal_id)] = next_hop

                # Compute multi-hop lookahead direction (weighted average of next 8 hops)
                # This provides anticipatory guidance for sharp turns near hazards
                # UPDATED: Increased from 5 to 8 hops for better inflection point visibility
                multi_hop_direction = self._compute_multi_hop_direction(
                    node_pos, parents, max_hops=4
                )
                self.multi_hop_direction_cache[(node_pos, goal_id)] = (
                    multi_hop_direction
                )

            # CRITICAL FIX: Also populate cache entries for generic aliases ("switch", "exit")
            # This ensures lookups using generic goal_ids hit the cache
            # Use the flags we determined earlier to avoid race condition with mapping updates
            if needs_switch_alias:
                # Populate cache entries for "switch" alias (same geometric distances)
                for node_pos, distance in geometric_distances.items():
                    self.cache[(node_pos, "switch")] = (
                        distance  # Geometric (not physics)
                    )
                    self.geometric_cache[(node_pos, "switch")] = distance
                    next_hop = parents.get(node_pos) if parents else None
                    self.next_hop_cache[(node_pos, "switch")] = next_hop
                    # Also compute multi-hop direction for alias (8 hops for inflection points)
                    multi_hop_direction = self._compute_multi_hop_direction(
                        node_pos, parents, max_hops=4
                    )
                    self.multi_hop_direction_cache[(node_pos, "switch")] = (
                        multi_hop_direction
                    )
            elif needs_exit_alias:
                # Populate cache entries for "exit" alias (same geometric distances)
                for node_pos, distance in geometric_distances.items():
                    self.cache[(node_pos, "exit")] = distance  # Geometric (not physics)
                    self.geometric_cache[(node_pos, "exit")] = distance
                    next_hop = parents.get(node_pos) if parents else None
                    self.next_hop_cache[(node_pos, "exit")] = next_hop
                    # Also compute multi-hop direction for alias (8 hops for inflection points)
                    multi_hop_direction = self._compute_multi_hop_direction(
                        node_pos, parents, max_hops=4
                    )
                    self.multi_hop_direction_cache[(node_pos, "exit")] = (
                        multi_hop_direction
                    )

    def build_cache(
        self,
        level_data: LevelData,
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        graph_data: Optional[Dict[str, Any]] = None,
        mine_proximity_cache: Optional[Any] = None,
        mine_sdf: Optional[Any] = None,
        waypoints: Optional[List[Tuple[int, int]]] = None,
    ) -> bool:
        """
        Build or rebuild cache if needed.

        The adjacency graph should only contain nodes reachable from the initial
        player position. This ensures all cached distances are for reachable areas only.

        Args:
            level_data: Current level data
            adjacency: Masked graph adjacency structure (should be filtered to only reachable nodes)
            base_adjacency: Base graph adjacency structure (pre-entity-mask, for physics checks)
            graph_data: Optional graph data dict with spatial_hash for fast lookup
            mine_proximity_cache: Optional MineProximityCostCache for mine cost lookup
            mine_sdf: Optional MineSignedDistanceField for velocity-aware hazard costs

        Returns:
            True if cache was rebuilt, False if cache was valid
        """
        # Extract mine state signature
        mine_signature = get_mine_state_signature(level_data)

        # CRITICAL FIX: Extract switch state signature for cache invalidation
        # Switch activation changes goal positions (switch removed from goals)
        # Without this check, cache won't rebuild when switches activate
        switch_signature = tuple(sorted(level_data.switch_states.items()))

        # CRITICAL FIX: Extract waypoint signature for cache invalidation
        # Waypoints can change mid-episode (adaptive discovery)
        # Without this check, cache won't rebuild when waypoints change
        waypoint_signature = tuple(waypoints) if waypoints else None

        # Check if cache needs invalidation
        needs_rebuild = (
            self._cached_level_data is None
            or self._cached_level_data != level_data
            or self._cached_mine_signature != mine_signature
            or self._cached_switch_signature != switch_signature
            or self._cached_waypoint_signature != waypoint_signature
        )

        if needs_rebuild:
            # Extract goals and precompute distances
            # Note: BFS traversal will only reach nodes in adjacency, so if adjacency
            # is filtered to reachable nodes, all cached distances will be for reachable areas
            goals = extract_goal_positions(level_data)

            # Add waypoints to goals list for precomputation
            if waypoints:
                for i, waypoint_pos in enumerate(waypoints):
                    waypoint_id = f"waypoint_{i}"
                    goals.append((waypoint_pos, waypoint_id))
                # logger.info(
                #     f"[LEVEL_CACHE_BUILD] Added {len(waypoints)} waypoints for BFS precomputation: "
                #     f"{[(wp_pos, f'waypoint_{i}') for i, wp_pos in enumerate(waypoints)]}"
                # )
            else:
                logger.info(
                    f"[LEVEL_CACHE_BUILD] No waypoints provided (waypoints={waypoints})"
                )

            self._precompute_distances_from_goals(
                goals,
                adjacency,
                base_adjacency,
                graph_data,
                level_data,
                mine_proximity_cache,
                mine_sdf,
            )

            # Update cached state
            self._cached_level_data = level_data
            self._cached_mine_signature = mine_signature
            self._cached_switch_signature = switch_signature
            self._cached_waypoint_signature = waypoint_signature

            return True

        return False

    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        self.geometric_cache.clear()
        self.next_hop_cache.clear()
        self.multi_hop_direction_cache.clear()
        self._goal_node_to_goal_id.clear()
        self._goal_id_to_goal_pos.clear()
        self._cached_level_data = None
        self._cached_mine_signature = None
        self._cached_switch_signature = None
        self._cached_waypoint_signature = None
        self.level_cache_hits = 0
        self.level_cache_misses = 0
        self.geometric_cache_hits = 0
        self.geometric_cache_misses = 0

    def get_goal_id_from_node(self, goal_node: Tuple[int, int]) -> Optional[str]:
        """
        Get goal_id for a given goal_node.

        Args:
            goal_node: Node position that corresponds to a goal

        Returns:
            goal_id if found, None otherwise
        """
        return self._goal_node_to_goal_id.get(goal_node)

    def get_goal_pos_from_id(self, goal_id: str) -> Optional[Tuple[int, int]]:
        """
        Get goal position for a given goal_id.

        Args:
            goal_id: Unique identifier for the goal

        Returns:
            goal_pos if found, None otherwise
        """
        return self._goal_id_to_goal_pos.get(goal_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_queries = self.level_cache_hits + self.level_cache_misses
        hit_rate = self.level_cache_hits / total_queries if total_queries > 0 else 0.0

        geo_total = self.geometric_cache_hits + self.geometric_cache_misses
        geo_hit_rate = self.geometric_cache_hits / geo_total if geo_total > 0 else 0.0

        return {
            "hits": self.level_cache_hits,
            "misses": self.level_cache_misses,
            "total_queries": total_queries,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "geometric_cache_size": len(self.geometric_cache),
            "geometric_hits": self.geometric_cache_hits,
            "geometric_misses": self.geometric_cache_misses,
            "geometric_hit_rate": geo_hit_rate,
        }
