"""
Level-based path distance cache for precomputed shortest path distances.

Uses flood fill (BFS) from goal positions to precompute all node-to-goal distances.
Cache invalidates when level or mine states change.
"""

import logging
from typing import Dict, Tuple, List, Optional, Any
from ..level_data import LevelData

logger = logging.getLogger(__name__)
from .pathfinding_utils import (
    find_closest_node_to_position,
    bfs_distance_from_start,
    extract_spatial_lookups_from_graph_data,
)
from .level_data_helpers import extract_goal_positions, get_mine_state_signature


class LevelBasedPathDistanceCache:
    """
    Precomputed cache for shortest path distances from all nodes to goals.

    Uses flood fill (BFS) from each goal position to precompute distances
    to all reachable nodes. The adjacency graph passed to build_cache() should
    be filtered to only include nodes reachable from the initial player position,
    ensuring all cached distances are for reachable areas only.

    Cache invalidates when level or mine states change.

    Stores TWO types of distances:
    1. Physics-weighted costs (for pathfinding optimization)
    2. Geometric distances (for PBRS normalization - actual pixels)
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
        # Mapping of goal_node -> goal_id for quick lookup
        self._goal_node_to_goal_id: Dict[Tuple[int, int], str] = {}
        # Mapping of goal_id -> goal_pos for validation
        self._goal_id_to_goal_pos: Dict[str, Tuple[int, int]] = {}
        self._cached_level_data: Optional[LevelData] = None
        self._cached_mine_signature: Optional[Tuple] = None
        self._cached_switch_signature: Optional[Tuple] = None
        self.level_cache_hits = 0
        self.level_cache_misses = 0
        self.geometric_cache_hits = 0
        self.geometric_cache_misses = 0

    def get_distance(
        self, node_pos: Tuple[int, int], goal_pos: Tuple[int, int], goal_id: str
    ) -> float:
        """
        Get cached physics-weighted distance from node to goal.

        Args:
            node_pos: Node position (x, y) in pixels
            goal_pos: Goal position (x, y) in pixels (for validation)
            goal_id: Unique identifier for the goal

        Returns:
            Cached physics-weighted distance, or float('inf') if not cached
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

        # DIAGNOSTIC: Log cache miss details (only first few to avoid spam)
        if self.geometric_cache_misses <= 10:
            available_goals = set(
                k[1] for k in self.geometric_cache.keys() if k[0] == node_pos
            )
            # Check if node exists in cache at all
            node_exists = any(k[0] == node_pos for k in self.geometric_cache.keys())
            # Sample a few cached nodes near this position
            nearby_nodes = [
                k
                for k in self.geometric_cache.keys()
                if abs(k[0][0] - node_pos[0]) < 50 and abs(k[0][1] - node_pos[1]) < 50
            ]
            logger.warning(
                f"[LEVEL_CACHE_MISS #{self.geometric_cache_misses}] node={node_pos}, goal_id={goal_id}, "
                f"node_in_cache={node_exists}, available_goals_for_node={available_goals}, "
                f"all_cached_goals={set(self._goal_id_to_goal_pos.keys())}, "
                f"nearby_cached_entries={len(nearby_nodes)}, sample={nearby_nodes[:3]}"
            )

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

            # Run SINGLE BFS with physics costs, tracking geometric distances along the path
            # This computes both physics costs (for pathfinding) and pixel distances
            # along the physics-optimal path (for PBRS normalization) in one pass.
            # Request parents dict to compute next_hop for each node.
            # Since we flood from goal, parent[node] = neighbor closer to goal = next_hop
            physics_distances, _, parents, geometric_distances = (
                bfs_distance_from_start(
                    goal_node,
                    None,
                    adjacency,
                    base_adjacency,
                    None,
                    physics_cache,
                    level_data,
                    mine_proximity_cache,
                    return_parents=True,
                    use_geometric_costs=False,  # Physics-weighted costs for priority
                    track_geometric_distances=True,  # Also track pixel distances along physics path
                    mine_sdf=mine_sdf,  # Pass SDF for velocity-aware mine costs
                )
            )

            # Store physics-weighted distances and next_hop for all computed nodes
            for node_pos, distance in physics_distances.items():
                self.cache[(node_pos, goal_id)] = distance

                # Parent in BFS from goal = next hop toward goal
                # (since we flooded outward from goal, parent is closer to goal)
                next_hop = parents.get(node_pos) if parents else None
                self.next_hop_cache[(node_pos, goal_id)] = next_hop

            # Store geometric distances along physics-optimal path for PBRS normalization
            # These are the pixel lengths of the physics-optimal paths, NOT separate
            # geometrically-shortest paths
            if geometric_distances is not None:
                for node_pos, distance in geometric_distances.items():
                    self.geometric_cache[(node_pos, goal_id)] = distance

            # CRITICAL FIX: Also populate cache entries for generic aliases ("switch", "exit")
            # This ensures lookups using generic goal_ids hit the cache
            # Use the flags we determined earlier to avoid race condition with mapping updates
            if needs_switch_alias:
                # Populate cache entries for "switch" alias
                for node_pos, distance in physics_distances.items():
                    self.cache[(node_pos, "switch")] = distance
                    self.next_hop_cache[(node_pos, "switch")] = (
                        parents.get(node_pos) if parents else None
                    )
                if geometric_distances is not None:
                    for node_pos, distance in geometric_distances.items():
                        self.geometric_cache[(node_pos, "switch")] = distance
            elif needs_exit_alias:
                # Populate cache entries for "exit" alias
                for node_pos, distance in physics_distances.items():
                    self.cache[(node_pos, "exit")] = distance
                    self.next_hop_cache[(node_pos, "exit")] = (
                        parents.get(node_pos) if parents else None
                    )
                if geometric_distances is not None:
                    for node_pos, distance in geometric_distances.items():
                        self.geometric_cache[(node_pos, "exit")] = distance

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

        # Check if cache needs invalidation
        needs_rebuild = (
            self._cached_level_data is None
            or self._cached_level_data != level_data
            or self._cached_mine_signature != mine_signature
            or self._cached_switch_signature != switch_signature
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
                logger.info(
                    f"[WAYPOINT_CACHE] Added {len(waypoints)} waypoints for precomputation: "
                    f"{[(wp_pos, f'waypoint_{i}') for i, wp_pos in enumerate(waypoints)]}"
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

            # DIAGNOSTIC: Log what was cached
            # Sample some cached node positions
            sample_nodes = list(set(k[0] for k in self.geometric_cache.keys()))[:10]
            sample_goals = list(set(k[1] for k in self.geometric_cache.keys()))
            logger.info(
                f"[CACHE_POPULATED] Level cache built: "
                f"goals={list(self._goal_id_to_goal_pos.keys())}, "
                f"physics_cache={len(self.cache)} entries, "
                f"geometric_cache={len(self.geometric_cache)} entries, "
                f"next_hop_cache={len(self.next_hop_cache)} entries, "
                f"sample_nodes={sample_nodes}, "
                f"cached_goal_ids={sample_goals}"
            )

            return True

        return False

    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        self.geometric_cache.clear()
        self.next_hop_cache.clear()
        self._goal_node_to_goal_id.clear()
        self._goal_id_to_goal_pos.clear()
        self._cached_level_data = None
        self._cached_mine_signature = None
        self._cached_switch_signature = None
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
