"""
Cache for path visualization data (debug/visualization purposes only).

Caches computed paths to switches and exits for efficient visualization.
Invalidates when level changes, mine states change, or ninja moves significantly.
"""

from typing import Dict, Tuple, List, Optional, Any
from ..level_data import LevelData
from .level_data_helpers import get_mine_state_signature


class PathVisualizationCache:
    """
    Cache for path visualization data (debug/visualization purposes only).

    Caches computed paths to switches and exits for efficient visualization.
    Invalidates when level changes, mine states change, or ninja moves significantly.
    """

    def __init__(self):
        """Initialize path visualization cache."""
        self._cached_path_data: Optional[Dict[str, Any]] = None
        self._cached_ninja_pos: Optional[Tuple[float, float]] = None
        self._cached_adjacency_signature: Optional[int] = None
        self._cached_level_data: Optional[LevelData] = None
        self._cached_mine_signature: Optional[Tuple] = None
        self._cached_mine_cache_size: Optional[int] = None  # Track mine cache state
        self._ninja_move_threshold = 6.0  # pixels - update paths every 6px (half a tile) for accurate next-hop visualization

    def get_cached_paths(
        self,
        ninja_pos: Tuple[float, float],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: LevelData,
        switch_positions: List[Tuple[int, int]],
        exit_positions: List[Tuple[int, int]],
        exit_switch_activated: bool,
        mine_cache_size: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached path data if still valid, otherwise return None.

        Args:
            ninja_pos: Current ninja position
            adjacency: Graph adjacency structure
            level_data: Current level data
            switch_positions: List of switch positions
            exit_positions: List of exit positions
            exit_switch_activated: Whether exit switch is activated

        Returns:
            Cached path data dict if valid, None if cache needs rebuild
        """
        # Check if cache is empty
        if self._cached_path_data is None or self._cached_ninja_pos is None:
            return None

        # Check if level changed
        if self._cached_level_data != level_data:
            return None

        # Check if mine states changed
        mine_signature = get_mine_state_signature(level_data)
        if self._cached_mine_signature != mine_signature:
            return None

        # CRITICAL: Check if mine proximity cache size changed
        # This catches cases where paths were cached before mine cache was populated
        # Even if mine_signature is same, cache size can differ if cache wasn't built yet
        if mine_cache_size is not None and self._cached_mine_cache_size != mine_cache_size:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"[PATH_CACHE_INVALIDATE] Mine cache size changed: "
                f"{self._cached_mine_cache_size} -> {mine_cache_size}. "
                f"Invalidating path cache to recompute with correct mine costs."
            )
            return None

        # Check if adjacency graph changed
        adjacency_signature = hash(tuple(sorted(adjacency.keys())))
        if self._cached_adjacency_signature != adjacency_signature:
            return None

        # Check if ninja moved significantly
        dx = ninja_pos[0] - self._cached_ninja_pos[0]
        dy = ninja_pos[1] - self._cached_ninja_pos[1]
        dist_moved = (dx**2 + dy**2) ** 0.5
        if dist_moved >= self._ninja_move_threshold:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(
                f"[PATH_CACHE_INVALIDATE] Ninja moved {dist_moved:.1f}px "
                f"(threshold: {self._ninja_move_threshold}px). Invalidating path cache."
            )
            return None

        # Check if entities changed
        entity_signature = (
            tuple(switch_positions),
            tuple(exit_positions),
            exit_switch_activated,
        )
        cached_entity_signature = self._cached_path_data.get("entity_signature")
        if entity_signature != cached_entity_signature:
            return None

        # Cache is valid
        return self._cached_path_data

    def cache_paths(
        self,
        ninja_pos: Tuple[float, float],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: LevelData,
        switch_positions: List[Tuple[int, int]],
        exit_positions: List[Tuple[int, int]],
        exit_switch_activated: bool,
        closest_node: Optional[Tuple[int, int]],
        switch_path: Optional[List[Tuple[int, int]]],
        exit_path: Optional[List[Tuple[int, int]]],
        switch_node: Optional[Tuple[int, int]],
        exit_node: Optional[Tuple[int, int]],
        mine_cache_size: Optional[int] = None,
    ):
        """
        Cache computed path data.

        Args:
            ninja_pos: Current ninja position
            adjacency: Graph adjacency structure
            level_data: Current level data
            switch_positions: List of switch positions
            exit_positions: List of exit positions
            exit_switch_activated: Whether exit switch is activated
            closest_node: Closest node to ninja position
            switch_path: Path to switch (list of nodes)
            exit_path: Path to exit (list of nodes)
            switch_node: Node closest to switch
            exit_node: Node closest to exit
        """
        adjacency_signature = hash(tuple(sorted(adjacency.keys())))
        mine_signature = get_mine_state_signature(level_data)
        entity_signature = (
            tuple(switch_positions),
            tuple(exit_positions),
            exit_switch_activated,
        )

        self._cached_ninja_pos = ninja_pos
        self._cached_adjacency_signature = adjacency_signature
        self._cached_level_data = level_data
        self._cached_mine_signature = mine_signature
        self._cached_mine_cache_size = mine_cache_size  # Track mine cache state
        self._cached_path_data = {
            "closest_node": closest_node,
            "entity_signature": entity_signature,
            "switch_path": switch_path,
            "exit_path": exit_path,
            "switch_node": switch_node,
            "exit_node": exit_node,
        }

    def clear_cache(self):
        """Clear the visualization cache."""
        self._cached_path_data = None
        self._cached_ninja_pos = None
        self._cached_adjacency_signature = None
        self._cached_level_data = None
        self._cached_mine_signature = None
        self._cached_mine_cache_size = None
