"""
Level-based path distance cache for precomputed shortest path distances.

Uses flood fill (BFS) from goal positions to precompute all node-to-goal distances.
Cache invalidates when level or mine states change.

Includes precomputed Phi grids for O(1) PBRS potential lookup at runtime.
Supports both BFS-based (6px graph resolution) and simulation-based (6px) phi grids.
When simulation phi grids are available, they are used for higher-quality
physics-aware potentials; otherwise falls back to BFS-based grids.
"""

import logging
import os
from typing import Dict, Tuple, List, Optional, Any

import numpy as np

from ..level_data import LevelData
from .pathfinding_utils import (
    find_closest_node_to_position,
    bfs_distance_from_start,
    extract_spatial_lookups_from_graph_data,
)
from .level_data_helpers import extract_goal_positions, get_mine_state_signature

logger = logging.getLogger(__name__)

# --- Phi grid resolution constants ---
# BFS-based (graph resolution): 6px resolution, 176x100 grid
BFS_PHI_GRID_RESOLUTION = 6
BFS_PHI_GRID_WIDTH = 176  # 1056 / 6
BFS_PHI_GRID_HEIGHT = 100  # 600 / 6

# Simulation-based: 6px resolution, 176x100 grid
SIM_PHI_GRID_RESOLUTION = 6
SIM_PHI_GRID_WIDTH = 176  # 1056 / 6
SIM_PHI_GRID_HEIGHT = 100  # 600 / 6

# Active resolution (set dynamically based on available phi source)
# Both BFS and simulation grids now use 6px resolution
PHI_GRID_RESOLUTION = BFS_PHI_GRID_RESOLUTION
PHI_GRID_WIDTH = BFS_PHI_GRID_WIDTH
PHI_GRID_HEIGHT = BFS_PHI_GRID_HEIGHT

NODE_WORLD_COORD_OFFSET = 24  # tile data space -> world coordinate offset


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

        # Max distances for denormalization (stored for diagnostics)
        self.max_dist_switch: float = 1.0
        self.max_dist_exit: float = 1.0

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

            distances, _, parents, _ = bfs_distance_from_start(
                goal_node,
                None,
                adjacency,
                base_adjacency,
                max_distance=None,
                return_parents=True,
                use_geometric_costs=True,
            )
            geometric_distances = distances

            # Store distances and next_hop for all computed nodes
            # cache: physics-weighted cost (for Phi potential when use_physics) or geometric
            # geometric_cache: geometric distance in pixels (for normalization, diagnostics)
            for node_pos in geometric_distances.keys():
                phys_dist = distances.get(node_pos, float("inf"))
                geo_dist = geometric_distances.get(node_pos, phys_dist)
                self.cache[(node_pos, goal_id)] = phys_dist
                self.geometric_cache[(node_pos, goal_id)] = geo_dist

                # Parent in BFS from goal = next hop toward goal
                # (since we flooded outward from goal, parent is closer to goal)
                next_hop = parents.get(node_pos) if parents else None
                self.next_hop_cache[(node_pos, goal_id)] = next_hop

            # CRITICAL FIX: Also populate cache entries for generic aliases ("switch", "exit")
            # This ensures lookups using generic goal_ids hit the cache
            if needs_switch_alias:
                for node_pos in geometric_distances.keys():
                    phys_dist = distances.get(node_pos, float("inf"))
                    geo_dist = geometric_distances.get(node_pos, phys_dist)
                    self.cache[(node_pos, "switch")] = phys_dist
                    self.geometric_cache[(node_pos, "switch")] = geo_dist
                    next_hop = parents.get(node_pos) if parents else None
                    self.next_hop_cache[(node_pos, "switch")] = next_hop
            elif needs_exit_alias:
                for node_pos in geometric_distances.keys():
                    phys_dist = distances.get(node_pos, float("inf"))
                    geo_dist = geometric_distances.get(node_pos, phys_dist)
                    self.cache[(node_pos, "exit")] = phys_dist
                    self.geometric_cache[(node_pos, "exit")] = geo_dist
                    next_hop = parents.get(node_pos) if parents else None
                    self.next_hop_cache[(node_pos, "exit")] = next_hop

    def build_cache(
        self,
        level_data: LevelData,
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        graph_data: Optional[Dict[str, Any]] = None,
        mine_proximity_cache: Optional[Any] = None,
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

            self._precompute_distances_from_goals(
                goals,
                adjacency,
                base_adjacency,
                graph_data,
                level_data,
                mine_proximity_cache,
            )

            # Update cached state
            self._cached_level_data = level_data
            self._cached_mine_signature = mine_signature
            self._cached_switch_signature = switch_signature

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
        self.max_dist_switch = 1.0
        self.max_dist_exit = 1.0

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
