"""
Mine proximity cost cache for efficient hazard avoidance in pathfinding.

Precomputes mine proximity cost multipliers for all nodes to avoid repeated
distance calculations during A* pathfinding. Cache invalidates when mine
states change (toggle mines switching between safe/deadly states).
"""

from typing import Dict, Tuple, Optional, Any
from ..level_data import LevelData
from .level_data_helpers import get_mine_state_signature


class MineProximityCostCache:
    """
    Precomputed cache for mine proximity cost multipliers at each node.

    Builds a mapping of node_position -> cost_multiplier based on proximity
    to deadly toggle mines. Cache invalidates when mine states change.

    Performance benefit: O(1) lookup during pathfinding vs O(M) per edge
    where M is the number of mines in the level.
    """

    def __init__(self):
        """Initialize mine proximity cost cache."""
        # Cache mapping node_position -> cost_multiplier
        self.cache: Dict[Tuple[int, int], float] = {}
        self._cached_level_data: Optional[LevelData] = None
        self._cached_mine_signature: Optional[Tuple] = None
        self.cache_hits = 0
        self.cache_misses = 0

    def get_cost_multiplier(self, node_pos: Tuple[int, int]) -> float:
        """
        Get cached mine proximity cost multiplier for a node.

        Args:
            node_pos: Node position (x, y) in pixels

        Returns:
            Cost multiplier in range [1.0, MINE_HAZARD_COST_MULTIPLIER]
            1.0 if not in cache (no penalty)
        """
        if node_pos in self.cache:
            self.cache_hits += 1
            return self.cache[node_pos]

        self.cache_misses += 1
        return 1.0  # No penalty if not cached

    def _precompute_mine_proximity_costs(
        self,
        adjacency: Dict[Tuple[int, int], Any],
        level_data: LevelData,
    ):
        """
        Precompute mine proximity cost multipliers for all nodes.

        For each node in the adjacency graph, calculate the cost multiplier
        based on proximity to deadly toggle mines. This replaces the per-edge
        calculation with a one-time precomputation.

        Args:
            adjacency: Graph adjacency structure (keys are node positions)
            level_data: Level data containing mine entities
        """
        from ...gym_environment.reward_calculation.reward_constants import (
            MINE_HAZARD_RADIUS,
            MINE_HAZARD_COST_MULTIPLIER,
            MINE_PENALIZE_DEADLY_ONLY,
        )
        from ...constants.entity_types import EntityType

        # Clear existing cache
        self.cache.clear()

        # Early exit if hazard avoidance is disabled
        if MINE_HAZARD_COST_MULTIPLIER <= 1.0:
            return

        # Get all toggle mines from level_data
        # Type 1 (TOGGLE_MINE) starts untoggled (state 1)
        # Type 21 (TOGGLE_MINE_TOGGLED) starts toggled/deadly (state 0)
        mines = level_data.get_entities_by_type(
            EntityType.TOGGLE_MINE
        ) + level_data.get_entities_by_type(EntityType.TOGGLE_MINE_TOGGLED)

        if not mines:
            return  # No mines, all nodes get 1.0 multiplier (default)

        # Extract deadly mine positions for efficient lookup
        deadly_mine_positions = []
        for mine in mines:
            mine_state = mine.get("state", 0)
            # Only include deadly mines (state 0 = toggled/deadly)
            if MINE_PENALIZE_DEADLY_ONLY and mine_state != 0:
                continue  # Skip safe mines (state 1) and toggling mines (state 2)

            mine_x = mine.get("x", 0)
            mine_y = mine.get("y", 0)
            deadly_mine_positions.append((mine_x, mine_y))

        if not deadly_mine_positions:
            return  # No deadly mines, all nodes get 1.0 multiplier

        # Precompute cost multiplier for each node in adjacency graph
        for node_pos in adjacency.keys():
            # Find closest deadly mine to this node
            min_distance = float("inf")
            for mine_x, mine_y in deadly_mine_positions:
                dx = node_pos[0] - mine_x
                dy = node_pos[1] - mine_y
                distance = (dx * dx + dy * dy) ** 0.5
                min_distance = min(min_distance, distance)

            # Calculate cost multiplier based on proximity
            if min_distance < MINE_HAZARD_RADIUS:
                # Linear interpolation:
                # - At mine center (distance=0): full multiplier
                # - At radius edge (distance=RADIUS): 1.0 (no penalty)
                proximity_factor = 1.0 - (min_distance / MINE_HAZARD_RADIUS)
                multiplier = 1.0 + proximity_factor * (
                    MINE_HAZARD_COST_MULTIPLIER - 1.0
                )
                self.cache[node_pos] = multiplier
            # If min_distance >= MINE_HAZARD_RADIUS, no penalty (1.0), don't store

    def build_cache(
        self,
        level_data: LevelData,
        adjacency: Dict[Tuple[int, int], Any],
    ) -> bool:
        """
        Build or rebuild cache if needed.

        Cache is rebuilt when:
        1. Level changes (new level loaded)
        2. Mine states change (toggle mines switched)

        Args:
            level_data: Current level data
            adjacency: Graph adjacency structure (keys are node positions)

        Returns:
            True if cache was rebuilt, False if cache was valid
        """
        # Extract mine state signature
        mine_signature = get_mine_state_signature(level_data)

        # Check if cache needs invalidation
        needs_rebuild = (
            self._cached_level_data is None
            or self._cached_level_data != level_data
            or self._cached_mine_signature != mine_signature
        )

        if needs_rebuild:
            # Precompute mine proximity costs for all nodes
            self._precompute_mine_proximity_costs(adjacency, level_data)

            # Update cached state
            self._cached_level_data = level_data
            self._cached_mine_signature = mine_signature

            return True

        return False

    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        self._cached_level_data = None
        self._cached_mine_signature = None
        self.cache_hits = 0
        self.cache_misses = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_queries if total_queries > 0 else 0.0

        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total_queries": total_queries,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
        }
