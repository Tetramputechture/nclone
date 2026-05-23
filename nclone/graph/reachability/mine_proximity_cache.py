"""
Mine proximity cost cache for efficient hazard avoidance in pathfinding.

Precomputes mine proximity cost multipliers for all nodes to avoid repeated
distance calculations during A* pathfinding. Cache invalidates when mine
states change (toggle mines switching between safe/deadly states).

Also includes a Signed Distance Field (SDF) to mines for O(1) hazard lookups
in the observation space, and gradient vectors for escape direction features.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from ..level_data import LevelData
from .level_data_helpers import get_mine_state_signature
from .cost_functions import NODE_WORLD_COORD_OFFSET
from .spatial_hash import SpatialHash

# Level dimensions in tiles (N++ standard level size)
LEVEL_WIDTH_TILES = 44
LEVEL_HEIGHT_TILES = 25
TILE_SIZE = 24  # pixels per tile

# SDF grid resolution (6px = sub-node resolution for higher accuracy)
# Using 6px provides max error of 3px from mine center
SDF_CELL_SIZE = 6  # pixels per SDF cell
SDF_WIDTH = 176  # 44 tiles * 4 = 176 cells at 6px resolution
SDF_HEIGHT = 100  # 25 tiles * 4 = 100 cells at 6px resolution


def get_playable_nodes_for_mine_filter(
    level_data: LevelData,
    adjacency: Dict[Tuple[int, int], Any],
    graph_data: Optional[Dict[str, Any]],
) -> Set[Tuple[int, int]]:
    """
    Floodable spawn-reachable nodes (tile data space) for filtering mines.

    When graph_data contains spawn_reachable_nodes from GraphBuilder, uses that
    to avoid a duplicate flood fill. Otherwise runs flood_fill_reachable_nodes
    from level_data.start_position. If the flood returns empty, falls back to
    all adjacency keys (matches graph_builder). Returns empty if adjacency empty.
    """
    if not adjacency:
        return set()
    if graph_data is not None and graph_data.get("spawn_reachable_nodes") is not None:
        return set(graph_data["spawn_reachable_nodes"])

    from .surface_area import flood_fill_reachable_nodes
    from .pathfinding_utils import extract_spatial_lookups_from_graph_data

    spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(graph_data)
    reachable = flood_fill_reachable_nodes(
        level_data.start_position,
        adjacency,
        spatial_hash=spatial_hash,
        subcell_lookup=subcell_lookup,
        player_radius=10.0,
    )
    if not reachable:
        return set(adjacency.keys())
    return reachable


def filter_mine_world_positions_to_playable_area(
    mine_world_positions: List[Tuple[float, float]],
    playable_nodes: Set[Tuple[int, int]],
) -> List[Tuple[float, float]]:
    """
    Keep mines that lie within R pixels (tile data) of the playable component.

    Uses spatial hash on playable node positions. Mines are in world space;
    queries use the same node/tile space convention as find_closest_node_to_position.
    """
    if not mine_world_positions:
        return []
    if not playable_nodes:
        return []

    from ...gym_environment.reward_calculation.reward_constants import MINE_HAZARD_RADIUS

    r_assoc = max(96.0, 2.0 * MINE_HAZARD_RADIUS)
    sh = SpatialHash(cell_size=24.0)
    sh.build(list(playable_nodes))
    out: List[Tuple[float, float]] = []
    for mx, my in mine_world_positions:
        qx = float(mx) - NODE_WORLD_COORD_OFFSET
        qy = float(my) - NODE_WORLD_COORD_OFFSET
        if sh.find_closest(qx, qy, max_distance=r_assoc) is not None:
            out.append((float(mx), float(my)))
    return out


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
            1.0 if not in cache (no penalty - either no mines nearby or cache not built)
        """
        if node_pos in self.cache:
            self.cache_hits += 1
            return self.cache[node_pos]

        self.cache_misses += 1
        # Return 1.0 (no penalty) if node not in cache
        # This is correct for: (a) nodes far from mines, (b) no deadly mines, (c) cache not built
        # Caller should ensure cache is built before pathfinding to get mine avoidance
        return 1.0

    def _precompute_mine_proximity_costs(
        self,
        adjacency: Dict[Tuple[int, int], Any],
        level_data: LevelData,
        graph_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Precompute mine proximity cost multipliers for all nodes.

        For each node in the adjacency graph, calculate the cost multiplier
        based on proximity to deadly toggle mines. This replaces the per-edge
        calculation with a one-time precomputation.

        Args:
            adjacency: Graph adjacency structure (keys are node positions)
            level_data: Level data containing mine entities
            graph_data: Optional graph build output; if provided, only mines
                associated with the spawn flood-fill component are used.
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
        mines = level_data.get_all_toggle_mines()

        if not mines:
            return  # No mines, all nodes get 1.0 multiplier (default)

        # Extract deadly mine positions for efficient lookup
        deadly_mine_positions = []
        for mine in mines:
            mine_state = mine.get("state", 0)
            mine_type = mine.get("type")
            # Only include deadly mines (state 0 = toggled/deadly)
            if MINE_PENALIZE_DEADLY_ONLY and mine_state != 0:
                continue  # Skip safe mines (state 1) and toggling mines (state 2)

            mine_x = mine.get("x", 0)
            mine_y = mine.get("y", 0)
            # Validate mine position is not (0, 0) which might indicate extraction failure
            if mine_x == 0.0 and mine_y == 0.0:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"[MINE_CACHE] Warning: Mine at (0, 0) - type={mine_type}, state={mine_state}. "
                    f"This might indicate extraction failure."
                )
            deadly_mine_positions.append((mine_x, mine_y))

        if not deadly_mine_positions:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"[MINE_CACHE] No deadly mines found (total mines={len(mines)}, "
                f"PENALIZE_DEADLY_ONLY={MINE_PENALIZE_DEADLY_ONLY}). "
                f"All paths will have 1.0x mine cost (no avoidance)."
            )
            return  # No deadly mines, all nodes get 1.0 multiplier

        if graph_data is not None and adjacency:
            playable = get_playable_nodes_for_mine_filter(
                level_data, adjacency, graph_data
            )
            deadly_mine_positions = filter_mine_world_positions_to_playable_area(
                deadly_mine_positions, playable
            )
            if not deadly_mine_positions:
                return

        # Log deadly mine count for diagnostics
        import logging

        logger = logging.getLogger(__name__)
        # logger.warning(
        #     f"[MINE_CACHE] Found {len(deadly_mine_positions)} deadly mines, "
        #     f"computing proximity costs for {len(adjacency)} nodes "
        #     f"(radius={MINE_HAZARD_RADIUS}px, multiplier={MINE_HAZARD_COST_MULTIPLIER}x)"
        # )

        # Precompute cost multiplier for each node in adjacency graph
        nodes_with_costs = 0
        if not adjacency or len(adjacency) == 0:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"[MINE_CACHE] Adjacency graph is empty - cannot build mine proximity cache. "
                f"Found {len(deadly_mine_positions)} deadly mines but no nodes to cache."
            )
            # Don't return early - just skip caching (cache will remain empty)

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
                nodes_with_costs += 1
            # If min_distance >= MINE_HAZARD_RADIUS, no penalty (1.0), don't store

        # Valid when mines are >radius from all walkable nodes (e.g. unreachable mine fields).
        # Match path_distance_calculator: avoid WARNING spam on every cache rebuild / worker.
        if len(deadly_mine_positions) > 0 and nodes_with_costs == 0:
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(
                f"[MINE_CACHE] No nodes within {MINE_HAZARD_RADIUS}px of {len(deadly_mine_positions)} deadly mines. "
                f"Cache is empty. Mines: {deadly_mine_positions[:3]}, "
                f"Adjacency nodes: {len(adjacency)}"
            )

        # Log final cache statistics
        # logger.warning(
        #     f"[MINE_CACHE] Precomputed {nodes_with_costs} nodes with proximity costs "
        #     f"(out of {len(adjacency)} total nodes). "
        #     f"Nodes within {MINE_HAZARD_RADIUS}px of mines will have increased path costs."
        # )

    def build_cache(
        self,
        level_data: LevelData,
        adjacency: Dict[Tuple[int, int], Any],
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Build or rebuild cache if needed.

        Cache is rebuilt when:
        1. Level changes (new level loaded)
        2. Mine states change (toggle mines switched)

        Args:
            level_data: Current level data
            adjacency: Graph adjacency structure (keys are node positions)
            graph_data: Optional graph dict from GraphBuilder; used to restrict
                mines to the spawn-reachable (playable) component when set.

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
            self._precompute_mine_proximity_costs(adjacency, level_data, graph_data)

            # Update cached state
            self._cached_level_data = level_data
            self._cached_mine_signature = mine_signature

            # DIAGNOSTIC: Log cache build results
            # import logging

            # logger = logging.getLogger(__name__)
            # logger.warning(
            #     f"[MINE_CACHE_BUILD] Built mine proximity cache: "
            #     f"{len(self.cache)} nodes with proximity costs, "
            #     f"mine_signature={mine_signature[:3] if mine_signature else 'none'}. "
            #     f"Paths will now avoid mines with cost multiplier."
            # )

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

    Computes at 6px (sub-node) resolution (176x100 grid) for higher accuracy.
    Provides:
    - signed_distance: Distance to nearest deadly mine (negative inside danger zone)
    - gradient: Normalized direction vector away from nearest mine

    This is computed ONCE per level load (mines are static per level) and
    used for observation space features without any per-step computation cost.

    Using 6px resolution reduces max position error to 3px from mine center,
    ensuring mines on tile edges show correctly as dangerous.

    Markov Property: Fully compliant - lookups depend only on current position.
    """

    def __init__(self):
        """Initialize the SDF cache."""
        # SDF grid: signed distance to nearest mine at each 6px cell
        # Shape: [SDF_HEIGHT, SDF_WIDTH] = [100, 176]
        self.sdf_grid: Optional[np.ndarray] = None

        # Gradient grid: escape direction at each 6px cell (unit vector away from mine)
        # Shape: [SDF_HEIGHT, SDF_WIDTH, 2] = [100, 176, 2]
        self.gradient_grid: Optional[np.ndarray] = None

        # Danger radius (NINJA_RADIUS + max mine radius + safety margin)
        self.danger_radius: float = 20.0  # 10 (ninja) + 4.5 (mine) + 5.5 (margin)

        # Cache invalidation: (level_key, mine_signature); level_key prefers level_id
        self._cached_sdf_key: Optional[Tuple[Any, Tuple]] = None

    def build_sdf(
        self,
        level_data: LevelData,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Build or rebuild the SDF if needed.

        Args:
            level_data: Current level data with mine entities
            graph_data: Optional graph dict; when provided, only mines in the
                spawn-reachable (playable) component affect the SDF.

        Returns:
            True if SDF was rebuilt, False if cache was valid
        """
        mine_signature = get_mine_state_signature(level_data)
        level_key = getattr(level_data, "level_id", None) or f"ld_{id(level_data)}"
        cache_key = (level_key, mine_signature)
        if self._cached_sdf_key == cache_key and self.sdf_grid is not None:
            return False  # Cache still valid

        # Initialize grids at 6px resolution for higher accuracy
        self.sdf_grid = np.full(
            (SDF_HEIGHT, SDF_WIDTH),
            fill_value=1.0,  # Max normalized distance (safe)
            dtype=np.float32,
        )
        self.gradient_grid = np.zeros(
            (SDF_HEIGHT, SDF_WIDTH, 2),
            dtype=np.float32,
        )

        # Import configuration flag for consistency with proximity costs
        from ...gym_environment.reward_calculation.reward_constants import (
            MINE_PENALIZE_DEADLY_ONLY,
        )

        # Get all toggle mines from level_data
        mines = level_data.get_all_toggle_mines()

        # Extract mine positions (respecting MINE_PENALIZE_DEADLY_ONLY for consistency)
        sdf_mine_positions = []
        for mine in mines:
            mine_state = mine.get("state", 0)
            # Use same filter logic as proximity costs for consistency
            if MINE_PENALIZE_DEADLY_ONLY and mine_state != 0:
                continue  # Skip safe mines if MINE_PENALIZE_DEADLY_ONLY is True
            mine_x = mine.get("x", 0)
            mine_y = mine.get("y", 0)
            sdf_mine_positions.append((mine_x, mine_y))

        if graph_data is not None and graph_data.get("adjacency"):
            playable = get_playable_nodes_for_mine_filter(
                level_data, graph_data["adjacency"], graph_data
            )
            world_list = [(float(x), float(y)) for x, y in sdf_mine_positions]
            filtered = filter_mine_world_positions_to_playable_area(
                world_list, playable
            )
            sdf_mine_positions = [(x, y) for x, y in filtered]

        if not sdf_mine_positions:
            # No mines to penalize - all tiles are safe (max distance)
            self._cached_sdf_key = cache_key
            return True

        cols = (np.arange(SDF_WIDTH, dtype=np.float64) + 0.5) * SDF_CELL_SIZE
        rows = (np.arange(SDF_HEIGHT, dtype=np.float64) + 0.5) * SDF_CELL_SIZE
        mx = np.array([p[0] for p in sdf_mine_positions], dtype=np.float64)
        my = np.array([p[1] for p in sdf_mine_positions], dtype=np.float64)
        # Distances from each cell center to every mine: (H, W, M)
        ccx = cols[np.newaxis, :, np.newaxis]  # (1, W, 1)
        ccy = rows[:, np.newaxis, np.newaxis]  # (H, 1, 1)
        mxa = mx.reshape(1, 1, -1)
        mya = my.reshape(1, 1, -1)
        d2 = (ccx - mxa) ** 2 + (ccy - mya) ** 2
        min_d = np.sqrt(np.min(d2, axis=2))
        nearest_idx = np.argmin(d2, axis=2)
        mx_s = mx[nearest_idx]
        my_s = my[nearest_idx]
        cell_x = cols[np.newaxis, :]  # (1, W)
        cell_y = rows[:, np.newaxis]  # (H, 1)
        nearest_dx = cell_x - mx_s
        nearest_dy = cell_y - my_s

        dr = self.danger_radius
        norm = np.empty((SDF_HEIGHT, SDF_WIDTH), dtype=np.float32)
        inside = min_d <= dr
        norm[inside] = (min_d[inside] / dr - 1.0).astype(np.float32)
        outside = ~inside
        norm[outside] = np.minimum(1.0, (min_d[outside] - dr) / (2.0 * dr)).astype(
            np.float32
        )
        self.sdf_grid = norm

        g = np.zeros((SDF_HEIGHT, SDF_WIDTH, 2), dtype=np.float32)
        safe = min_d > 1e-6
        inv = np.zeros_like(min_d)
        np.divide(1.0, min_d, out=inv, where=safe)
        g[:, :, 0] = (nearest_dx * inv).astype(np.float32)
        g[:, :, 1] = (nearest_dy * inv).astype(np.float32)
        g[~safe, 0] = 0.0
        g[~safe, 1] = 0.0
        self.gradient_grid = g

        self._cached_sdf_key = cache_key
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

        # Convert to 6px cell indices
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

        # Convert to 6px cell indices
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
        self._cached_sdf_key = None


def create_mine_sdf() -> MineSignedDistanceField:
    """
    Create MineSignedDistanceField for O(1) mine SDF lookups.

    Returns:
        MineSignedDistanceField-compatible instance.
    """
    return MineSignedDistanceField()
