"""
Mine proximity cost cache for efficient hazard avoidance in pathfinding.

Precomputes mine proximity cost multipliers for all nodes to avoid repeated
distance calculations during A* pathfinding. Cache invalidates when mine
states change (toggle mines switching between safe/deadly states).

Also includes a Signed Distance Field (SDF) to mines for O(1) hazard lookups
in the observation space, and gradient vectors for escape direction features.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
from ..level_data import LevelData
from .level_data_helpers import get_mine_state_signature

# Level dimensions in tiles (N++ standard level size)
LEVEL_WIDTH_TILES = 44
LEVEL_HEIGHT_TILES = 25
TILE_SIZE = 24  # pixels per tile

# SDF grid resolution (12px = sub-node resolution for higher accuracy)
# Using 12px instead of 24px reduces max error from 12px to 6px from mine center
SDF_CELL_SIZE = 12  # pixels per SDF cell
SDF_WIDTH = 88  # 44 tiles * 2 = 88 cells at 12px resolution
SDF_HEIGHT = 50  # 25 tiles * 2 = 50 cells at 12px resolution


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

            # Calculate cost multiplier based on proximity using QUADRATIC falloff
            # Quadratic is gentler at radius edge, steeper near mines
            # This provides better hazard avoidance without excessive path detours
            if min_distance < MINE_HAZARD_RADIUS:
                # Quadratic interpolation (changed from linear per plan Phase 6.1):
                # - At mine center (distance=0): full multiplier
                # - At radius edge (distance=RADIUS): 1.0 (no penalty)
                # - Quadratic curve: gentler near edge, steeper near center
                proximity_factor = 1.0 - (min_distance / MINE_HAZARD_RADIUS)
                # Apply quadratic scaling for more aggressive close-range avoidance
                multiplier = 1.0 + (proximity_factor**2) * (
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


class MineSignedDistanceField:
    """
    Precomputed Signed Distance Field (SDF) to mines for O(1) hazard lookups.

    Computes at 12px (sub-node) resolution (88x50 grid) for higher accuracy.
    Provides:
    - signed_distance: Distance to nearest deadly mine (negative inside danger zone)
    - gradient: Normalized direction vector away from nearest mine

    This is computed ONCE per level load (mines are static per level) and
    used for observation space features without any per-step computation cost.

    Using 12px instead of 24px resolution reduces max position error from 12px to 6px,
    ensuring mines on tile edges show correctly as dangerous.

    Markov Property: Fully compliant - lookups depend only on current position.
    """

    def __init__(self):
        """Initialize the SDF cache."""
        # SDF grid: signed distance to nearest mine at each 12px cell
        # Shape: [SDF_HEIGHT, SDF_WIDTH] = [50, 88]
        self.sdf_grid: Optional[np.ndarray] = None

        # Gradient grid: escape direction at each 12px cell (unit vector away from mine)
        # Shape: [SDF_HEIGHT, SDF_WIDTH, 2] = [50, 88, 2]
        self.gradient_grid: Optional[np.ndarray] = None

        # Danger radius (NINJA_RADIUS + max mine radius + safety margin)
        self.danger_radius: float = 20.0  # 10 (ninja) + 4.5 (mine) + 5.5 (margin)

        # Cache invalidation tracking
        self._cached_level_data: Optional[LevelData] = None
        self._cached_mine_signature: Optional[Tuple] = None

    def build_sdf(self, level_data: LevelData) -> bool:
        """
        Build or rebuild the SDF if needed.

        Args:
            level_data: Current level data with mine entities

        Returns:
            True if SDF was rebuilt, False if cache was valid
        """
        from ...constants.entity_types import EntityType

        # Check cache validity
        mine_signature = get_mine_state_signature(level_data)
        if (
            self._cached_level_data == level_data
            and self._cached_mine_signature == mine_signature
            and self.sdf_grid is not None
        ):
            return False  # Cache still valid

        # Initialize grids at 12px resolution for higher accuracy
        self.sdf_grid = np.full(
            (SDF_HEIGHT, SDF_WIDTH),
            fill_value=1.0,  # Max normalized distance (safe)
            dtype=np.float32,
        )
        self.gradient_grid = np.zeros(
            (SDF_HEIGHT, SDF_WIDTH, 2),
            dtype=np.float32,
        )

        # Get all toggle mines from level_data
        mines = level_data.get_entities_by_type(
            EntityType.TOGGLE_MINE
        ) + level_data.get_entities_by_type(EntityType.TOGGLE_MINE_TOGGLED)

        # Extract deadly mine positions (state 0 = deadly)
        deadly_mines = []
        for mine in mines:
            mine_state = mine.get("state", 0)
            if mine_state == 0:  # Only deadly mines
                mine_x = mine.get("x", 0)
                mine_y = mine.get("y", 0)
                deadly_mines.append((mine_x, mine_y))

        if not deadly_mines:
            # No deadly mines - all tiles are safe (max distance)
            self._cached_level_data = level_data
            self._cached_mine_signature = mine_signature
            return True

        # Compute SDF for each 12px cell (higher resolution than tiles)
        for row in range(SDF_HEIGHT):
            for col in range(SDF_WIDTH):
                # Cell center in pixel coordinates (12px resolution)
                cell_center_x = (col + 0.5) * SDF_CELL_SIZE
                cell_center_y = (row + 0.5) * SDF_CELL_SIZE

                # Find nearest mine and direction
                min_dist = float("inf")
                nearest_dx, nearest_dy = 0.0, 0.0

                for mine_x, mine_y in deadly_mines:
                    dx = cell_center_x - mine_x
                    dy = cell_center_y - mine_y
                    dist = np.sqrt(dx * dx + dy * dy)

                    if dist < min_dist:
                        min_dist = dist
                        nearest_dx = dx
                        nearest_dy = dy

                # Normalize distance to [-1, 1] range
                # -1 = at mine center, 0 = at danger_radius edge, 1 = far from mines
                if min_dist <= self.danger_radius:
                    # Inside danger zone: negative SDF
                    normalized_dist = (min_dist / self.danger_radius) - 1.0
                else:
                    # Outside danger zone: positive SDF, capped at 1.0
                    # Map [danger_radius, 3*danger_radius] to [0, 1]
                    normalized_dist = min(
                        1.0, (min_dist - self.danger_radius) / (2 * self.danger_radius)
                    )

                self.sdf_grid[row, col] = normalized_dist

                # Compute normalized gradient (escape direction)
                if min_dist > 1e-6:  # Avoid division by zero
                    self.gradient_grid[row, col, 0] = nearest_dx / min_dist
                    self.gradient_grid[row, col, 1] = nearest_dy / min_dist
                # If at mine center, gradient stays (0, 0)

        # Update cache state
        self._cached_level_data = level_data
        self._cached_mine_signature = mine_signature
        return True

    def get_sdf_at_position(self, x: float, y: float) -> float:
        """
        Get signed distance field value at a pixel position.

        O(1) lookup - fully Markovian (depends only on current position).

        Args:
            x: X position in pixels
            y: Y position in pixels

        Returns:
            Normalized SDF value in [-1, 1] range.
            Negative = inside danger zone, positive = safe.
        """
        if self.sdf_grid is None:
            return 1.0  # No SDF computed, assume safe

        # Convert to 12px cell indices
        col = int(x / SDF_CELL_SIZE)
        row = int(y / SDF_CELL_SIZE)

        # Clamp to valid range
        col = max(0, min(SDF_WIDTH - 1, col))
        row = max(0, min(SDF_HEIGHT - 1, row))

        return float(self.sdf_grid[row, col])

    def get_gradient_at_position(self, x: float, y: float) -> Tuple[float, float]:
        """
        Get escape direction (gradient) at a pixel position.

        O(1) lookup - fully Markovian (depends only on current position).

        Args:
            x: X position in pixels
            y: Y position in pixels

        Returns:
            Unit vector (dx, dy) pointing away from nearest mine.
            (0, 0) if at mine center or no mines.
        """
        if self.gradient_grid is None:
            return (0.0, 0.0)  # No gradient computed

        # Convert to 12px cell indices
        col = int(x / SDF_CELL_SIZE)
        row = int(y / SDF_CELL_SIZE)

        # Clamp to valid range
        col = max(0, min(SDF_WIDTH - 1, col))
        row = max(0, min(SDF_HEIGHT - 1, row))

        return (
            float(self.gradient_grid[row, col, 0]),
            float(self.gradient_grid[row, col, 1]),
        )

    def get_features_at_position(
        self, x: float, y: float
    ) -> Tuple[float, float, float]:
        """
        Get all SDF features at a position for observation space.

        O(1) lookup - fully Markovian.

        Args:
            x: X position in pixels
            y: Y position in pixels

        Returns:
            Tuple of (sdf_value, gradient_x, gradient_y)
        """
        sdf = self.get_sdf_at_position(x, y)
        grad_x, grad_y = self.get_gradient_at_position(x, y)
        return (sdf, grad_x, grad_y)

    def clear(self):
        """Clear the SDF cache."""
        self.sdf_grid = None
        self.gradient_grid = None
        self._cached_level_data = None
        self._cached_mine_signature = None
