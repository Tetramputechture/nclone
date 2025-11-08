"""Potential-Based Reward Shaping (PBRS) potential functions.

This module implements reusable potential functions Φ(s) for reward shaping
following the theory of Ng, Harada, and Russell (1999): "Policy Invariance
Under Reward Transformations: Theory and Application to Reward Shaping".

PBRS provides dense reward signals without changing the optimal policy by using
the formula: F(s,s') = γ * Φ(s') - Φ(s)

Key properties:
- Policy invariance: Optimal policy unchanged under PBRS
- Dense rewards: Provides gradient at every step
- Normalization: All potentials normalized to [0, 1] range
- Composability: Multiple potentials can be combined with weights

All constants defined in reward_constants.py with full documentation.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import deque
import heapq
from ..util.util import calculate_distance
from .reward_constants import (
    PBRS_EXPLORATION_VISIT_THRESHOLD,
    PBRS_EXPLORATION_RADIUS,
    PBRS_SWITCH_DISTANCE_SCALE,
    PBRS_EXIT_DISTANCE_SCALE,
    PBRS_OBJECTIVE_WEIGHT,
    PBRS_HAZARD_WEIGHT,  # Applied in calculate_combined_potential
    PBRS_IMPACT_WEIGHT,  # Applied in calculate_combined_potential
    PBRS_EXPLORATION_WEIGHT,
)
from ...constants.physics_constants import MAX_SURVIVABLE_IMPACT
from ...constants.entity_types import EntityType
from ...graph.reachability.pathfinding_utils import (
    find_closest_node_to_position,
    extract_spatial_lookups_from_graph_data,
)

# Sub-node size for surface area calculations (from graph builder)
# The graph builder creates a 2x2 grid of sub-nodes per 24px tile
SUB_NODE_SIZE = 12  # pixels per sub-node
PLAYER_RADIUS = 10  # Player collision radius in pixels (from graph_builder)


def _flood_fill_reachable_nodes(
    start_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    graph_data: Optional[Dict[str, Any]] = None,
) -> Set[Tuple[int, int]]:
    """
    Perform flood fill on adjacency graph to find reachable positions from start.

    Reuses logic from graph_builder._flood_fill_from_graph to ensure consistency.
    Uses lookup-based node finding (spatial_hash/subcell_lookup) for O(1) performance.
    Finds all nodes reachable from the starting position using BFS.

    Args:
        start_pos: Starting position (pixel coordinates, in world space)
        adjacency: Graph adjacency structure (keys in tile data space)
        graph_data: Optional graph data dict with spatial_hash for O(1) lookup

    Returns:
        Set of reachable node positions from the starting position
    """
    if not adjacency:
        return set()

    # Extract spatial lookup mechanisms from graph_data
    spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(graph_data)

    # Use lookup-based method to find closest node(s) within player radius
    # This uses O(1) spatial_hash or subcell_lookup when available
    closest_node = find_closest_node_to_position(
        start_pos,
        adjacency,
        threshold=PLAYER_RADIUS,
        spatial_hash=spatial_hash,
        subcell_lookup=subcell_lookup,
    )

    # If no node found within threshold, try with larger threshold for fallback
    if closest_node is None:
        closest_node = find_closest_node_to_position(
            start_pos,
            adjacency,
            threshold=50.0,  # Relaxed threshold for fallback
            spatial_hash=spatial_hash,
            subcell_lookup=subcell_lookup,
        )

    if closest_node is None or closest_node not in adjacency:
        return set()

    start_nodes = [closest_node]

    # Flood fill from all starting nodes
    reachable = set()
    queue = deque(start_nodes)
    visited = set(start_nodes)

    while queue:
        current = queue.popleft()
        reachable.add(current)

        # Get neighbors from adjacency
        neighbors = adjacency.get(current, [])
        for neighbor_pos, _ in neighbors:
            if neighbor_pos not in visited:
                visited.add(neighbor_pos)
                queue.append(neighbor_pos)

    return reachable


class PBRSPotentials:
    """Collection of potential functions for reward shaping.

    Each potential function Φ(s) represents a heuristic estimate of state value.
    Potentials are normalized to [0, 1] range for consistent scaling.

    Available potentials:
    - objective_distance_potential: Distance to current objective (switch/exit)
    - hazard_proximity_potential: Proximity to dangerous hazards
    - impact_risk_potential: Risk of high-velocity collisions
    - exploration_potential: Novelty of current state
    """

    @staticmethod
    def objective_distance_potential(
        state: Dict[str, Any],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        path_calculator: Any,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Potential based on shortest path distance to nearest objective.

        STRICT REQUIREMENTS - NO FALLBACKS:
        - adjacency, level_data, and path_calculator are REQUIRED
        - No fallback to Euclidean distance
        - Throws descriptive RuntimeErrors if any required data is missing

        Returns higher potential when closer to the current objective:
        - Switch when inactive
        - Exit when switch is active

        Args:
            state: Game state dictionary (must contain _pbrs_surface_area)
            adjacency: Graph adjacency structure (REQUIRED)
            level_data: Level data object (REQUIRED)
            path_calculator: CachedPathDistanceCalculator instance (REQUIRED)
            graph_data: Graph data dict with spatial_hash for O(1) node lookup (optional)

        Returns:
            float: Potential in range [0.0, 1.0], higher when closer to objective

        Raises:
            RuntimeError: If required data is missing or invalid
        """
        # STRICT: Validate all required parameters
        if adjacency is None:
            raise RuntimeError(
                "PBRS objective_distance_potential requires adjacency graph.\n"
                "Adjacency is None, which means graph building failed or is disabled.\n"
            )

        if level_data is None:
            raise RuntimeError(
                "PBRS objective_distance_potential requires level_data.\n"
                "Level data is None, which should never happen in normal operation.\n"
                "This indicates a serious configuration or initialization error."
            )

        if path_calculator is None:
            raise RuntimeError(
                "PBRS objective_distance_potential requires path_calculator.\n"
                "Path calculator is None, which means PBRS was not properly initialized.\n"
                "This should never happen - PBRS is always enabled and should auto-initialize."
            )

        # Determine goal position
        if not state["switch_activated"]:
            goal_pos = (int(state["switch_x"]), int(state["switch_y"]))
            cache_key = "switch"
        else:
            goal_pos = (int(state["exit_door_x"]), int(state["exit_door_y"]))
            cache_key = "exit"

        player_pos = (int(state["player_x"]), int(state["player_y"]))

        # Calculate shortest path distance
        try:
            distance = path_calculator.get_distance(
                player_pos,
                goal_pos,
                adjacency,
                cache_key=cache_key,
                level_data=level_data,
                graph_data=graph_data,
            )
        except Exception as e:
            raise RuntimeError(
                f"PBRS path distance calculation failed: {e}\n"
                "This indicates a problem with:\n"
                "  1. Graph adjacency structure (corrupted or invalid)\n"
                "  2. Player or goal position (outside graph bounds)\n"
                "  3. Path calculator internal error\n"
                "Check that player and goal positions are within level bounds."
            ) from e

        # Get surface area for normalization (REQUIRED)
        surface_area = state.get("_pbrs_surface_area")
        if surface_area is None:
            raise RuntimeError(
                "PBRS potential calculation requires '_pbrs_surface_area' in state.\n"
                "This should be set by PBRSCalculator.calculate_combined_potential().\n"
                "If you see this error, there's a bug in the PBRS calculation flow."
            )

        # Handle unreachable goals
        if distance == float("inf"):
            # Goal is unreachable - return minimum potential
            # This is NOT an error - some goals may be legitimately unreachable
            normalized_distance = 1.0
        else:
            # Surface area scaling: sqrt converts 2D area to 1D distance scale
            # Multiply by SUB_NODE_SIZE (12px) to get pixel distance equivalent
            # Result: scale grows with sqrt(area), keeping gradients consistent
            area_scale = np.sqrt(surface_area) * SUB_NODE_SIZE
            normalized_distance = min(1.0, distance / area_scale)

        potential = 1.0 - normalized_distance
        return max(0.0, min(1.0, potential))

    @staticmethod
    def hazard_proximity_potential(
        state: Dict[str, Any],
        adjacency: Optional[
            Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]
        ] = None,
        graph_data: Optional[Dict[str, Any]] = None,
        reachable_mines_lookup: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> float:
        """Potential penalty based on proximity to dangerous toggle mines.

        Only considers toggled mines (state 0) as hazards. Untoggled mines (state 1)
        and toggling mines (state 2) are safe and ignored.

        Optimized for performance with many mines:
        - Only evaluates nearest 16 REACHABLE mines (using max-heap)
        - Uses squared distances for faster comparison
        - Early exit for very close mines (within collision distance)
        - Filters unreachable mines using adjacency graph (behind walls, etc.)
        - Dynamic search radius based on PBRS surface area (level-appropriate scaling)

        Returns lower potential when close to dangerous mines, encouraging
        the agent to maintain safe distance from hazards.

        Args:
            state: Game state dictionary containing entities and player position
                (should contain '_pbrs_surface_area' for optimal performance)
            adjacency: Graph adjacency structure for reachability checking (optional)
            graph_data: Graph data dict with spatial_hash for O(1) lookup (optional)
            reachable_mines_lookup: Pre-computed dict of entity id -> (xpos, ypos) for
                reachable mines only. If provided, skips expensive reachability checks.
                Significant performance optimization when provided (optional).

        Returns:
            float: Potential in range [0.0, 1.0], lower when close to hazards
        """
        # Extract entity objects from state
        entities = state.get("entities", [])
        if not entities:
            return 1.0  # No entities, assume safe

        player_x = state.get("player_x", 0.0)
        player_y = state.get("player_y", 0.0)

        # Constants for optimization
        MAX_NEAREST_MINES = 16  # Only consider nearest 16 mines
        NINJA_RADIUS = 10.0  # Player collision radius
        MINE_RADIUS = 4.0  # Toggled mine radius (worst case)
        COLLISION_DISTANCE = NINJA_RADIUS + MINE_RADIUS  # ~14 pixels
        # Reachability check threshold: mines must be within reasonable distance of graph node
        REACHABILITY_THRESHOLD = 50.0  # pixels

        # Calculate dynamic max search radius based on PBRS surface area
        # Uses same area_scale formula as objective_distance_potential for consistency
        # Multiplier of 2.0 ensures we capture threats across the entire level
        surface_area = state.get("_pbrs_surface_area")
        if surface_area is None or surface_area <= 0:
            raise RuntimeError(
                "PBRS hazard_proximity_potential requires '_pbrs_surface_area' in state.\n"
                "This should be set by PBRSCalculator.calculate_combined_potential().\n"
                "If you see this error, there's a bug in the PBRS calculation flow."
            )
        # Surface area scaling: sqrt converts 2D area to 1D distance scale
        # Multiply by SUB_NODE_SIZE (12px) to get pixel distance equivalent
        area_scale = np.sqrt(surface_area) * SUB_NODE_SIZE
        # Use 2.0x multiplier to ensure we capture threats across the level
        max_search_radius = area_scale * 2.0
        MAX_SEARCH_RADIUS_SQ = max_search_radius**2  # Squared to avoid sqrt in loop

        # Use max-heap to keep track of nearest mines (negative distance for max-heap)
        # Format: (-distance_sq, id(e)) - we want smallest distances (largest negatives)
        nearest_mines_heap = []

        # Performance optimization: use pre-computed reachable mines lookup if available
        # This eliminates ~29 calls to find_closest_node_to_position per invocation
        if reachable_mines_lookup is not None:
            # Fast path: iterate pre-filtered reachable mines from lookup table
            for mine_id, (mine_x, mine_y) in reachable_mines_lookup.items():
                # Calculate squared distance (faster than sqrt)
                dx = player_x - mine_x
                dy = player_y - mine_y
                dist_sq = dx * dx + dy * dy

                # Skip mines beyond max search radius (threat is negligible)
                if dist_sq > MAX_SEARCH_RADIUS_SQ:
                    continue

                # Early exit: if mine is within collision distance, threat is maximum
                if dist_sq <= COLLISION_DISTANCE * COLLISION_DISTANCE:
                    # Direct collision threat - return minimum potential
                    hazard_threat = 1.0
                    return 1.0 - hazard_threat

                # Maintain heap of nearest MAX_NEAREST_MINES mines
                # Use mine_id as tie-breaker to avoid comparison errors when distances are equal
                if len(nearest_mines_heap) < MAX_NEAREST_MINES:
                    # Heap not full, add mine
                    heapq.heappush(nearest_mines_heap, (-dist_sq, mine_id))
                else:
                    # Heap full, compare with farthest mine in heap
                    if dist_sq < -nearest_mines_heap[0][0]:
                        # This mine is closer than farthest in heap, replace it
                        heapq.heapreplace(nearest_mines_heap, (-dist_sq, mine_id))
        else:
            # Slow path: iterate entities and check reachability (backward compatibility)
            # Extract spatial lookups for O(1) reachability checking
            spatial_hash = None
            subcell_lookup = None
            if graph_data is not None:
                spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
                    graph_data
                )

            # Single pass: filter dangerous mines, check reachability, and compute distances
            # Early exit if we find a mine within collision distance
            for e in entities:
                # Quick filter: check if it's a toggle mine first (fast attribute check)
                if not (
                    hasattr(e, "type")
                    and hasattr(e, "state")
                    and e.type
                    in (EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED)
                    and e.state == 0  # Only toggled mines are dangerous
                ):
                    continue

                if not (hasattr(e, "xpos") and hasattr(e, "ypos")):
                    continue

                # Calculate squared distance (faster than sqrt)
                dx = player_x - e.xpos
                dy = player_y - e.ypos
                dist_sq = dx * dx + dy * dy

                # Skip mines beyond max search radius (threat is negligible)
                if dist_sq > MAX_SEARCH_RADIUS_SQ:
                    continue

                # Early exit: if mine is within collision distance, threat is maximum
                if dist_sq <= COLLISION_DISTANCE * COLLISION_DISTANCE:
                    # Direct collision threat - return minimum potential
                    hazard_threat = 1.0
                    return 1.0 - hazard_threat

                # Optimize reachability checking: only check if mine could enter heap
                # If heap is full and mine is farther than farthest in heap, skip expensive check
                should_check_reachability = True
                if len(nearest_mines_heap) >= MAX_NEAREST_MINES:
                    # Heap is full - only check reachability if this mine could replace farthest
                    farthest_dist_sq = -nearest_mines_heap[0][0]
                    if dist_sq >= farthest_dist_sq:
                        # Mine is farther than farthest in heap, skip reachability check
                        should_check_reachability = False

                # Check reachability if adjacency graph is available and mine could enter heap
                if should_check_reachability and adjacency is not None:
                    mine_pos = (int(e.xpos), int(e.ypos))
                    closest_node = find_closest_node_to_position(
                        mine_pos,
                        adjacency,
                        threshold=REACHABILITY_THRESHOLD,
                        spatial_hash=spatial_hash,
                        subcell_lookup=subcell_lookup,
                    )
                    # Skip unreachable mines (behind walls, in unreachable areas)
                    if closest_node is None:
                        continue

                # Maintain heap of nearest MAX_NEAREST_MINES mines
                # Use id(e) as tie-breaker to avoid comparison errors when distances are equal
                if len(nearest_mines_heap) < MAX_NEAREST_MINES:
                    # Heap not full, add mine
                    heapq.heappush(nearest_mines_heap, (-dist_sq, id(e)))
                else:
                    # Heap full, compare with farthest mine in heap
                    if dist_sq < -nearest_mines_heap[0][0]:
                        # This mine is closer than farthest in heap, replace it
                        heapq.heapreplace(nearest_mines_heap, (-dist_sq, id(e)))

        if not nearest_mines_heap:
            return 1.0  # No dangerous mines found, maximum potential

        # Get nearest mine from heap (largest negative = smallest distance)
        nearest_dist_sq = -nearest_mines_heap[0][0]
        nearest_dist = np.sqrt(nearest_dist_sq)

        # Calculate threat using exponential decay
        # Decay factor of 100 pixels means threat drops to ~0.37 at 100px
        decay_factor = 100.0
        hazard_threat = np.exp(-nearest_dist / decay_factor)

        # Return potential based on hazard proximity
        # Raw potential in [0.0, 1.0] range (weight applied by caller):
        # When far from hazards: high potential (1.0)
        # When close to hazards: low potential (approaches 0.0)
        return 1.0 - hazard_threat

    @staticmethod
    def impact_risk_potential(state: Dict[str, Any]) -> float:
        """Potential penalty based on terminal impact risk.

        Matches the exact terminal impact calculation from ninja.py (lines 366-407).
        Calculates impact velocity using previous frame velocities and normalized
        surface normals, comparing against threshold to determine risk.

        Floor impact only considered if ninja was airborne before landing.
        Ceiling impact considered whenever ceiling contact exists.

        Args:
            state: Game state dictionary containing ninja properties

        Returns:
            float: Potential in range [0.0, 1.0], lower when high impact risk
        """
        # Extract required ninja properties from state
        xspeed_old = state.get("player_xspeed_old", 0.0)
        yspeed_old = state.get("player_yspeed_old", 0.0)
        airborn_old = state.get("player_airborn_old", False)
        floor_normalized_x = state.get("floor_normal_x", 0.0)
        floor_normalized_y = state.get("floor_normal_y", -1.0)
        ceiling_normalized_x = state.get("ceiling_normal_x", 0.0)
        ceiling_normalized_y = state.get("ceiling_normal_y", 1.0)

        # Extract floor_count and ceiling_count from game_state array
        # Indices 14 (floor), 15 (wall), 16 (ceiling) are normalized to [-1, 1]
        # Normalization: count=0 -> -1, count>=1 -> 1
        game_state = state.get("game_state", [])
        floor_count_norm = game_state[14] if len(game_state) > 14 else -1.0
        ceiling_count_norm = game_state[16] if len(game_state) > 16 else -1.0

        # Denormalize: (-1, 1] -> (0, 1] (check if > 0 means count > 0)
        floor_count = (floor_count_norm + 1.0) / 2.0
        ceiling_count = (ceiling_count_norm + 1.0) / 2.0
        has_floor_contact = floor_count > 0.5
        has_ceiling_contact = ceiling_count > 0.5

        max_risk = 0.0

        # Calculate floor impact risk (only if was airborne before landing)
        # Matches ninja.py line 366-381
        if has_floor_contact and airborn_old:
            # Calculate impact velocity using exact formula from ninja.py line 369-371
            impact_vel_floor = -(
                floor_normalized_x * xspeed_old + floor_normalized_y * yspeed_old
            )

            # Calculate threshold using exact formula from ninja.py line 373-374
            threshold_floor = MAX_SURVIVABLE_IMPACT - (4.0 / 3.0) * abs(
                floor_normalized_y
            )

            # Calculate risk as normalized distance above threshold
            if threshold_floor > 1e-6:  # Avoid division by zero
                if impact_vel_floor > threshold_floor:
                    # Impact velocity exceeds threshold - calculate risk
                    # Risk increases as impact_vel exceeds threshold
                    excess = impact_vel_floor - threshold_floor
                    risk_floor = min(1.0, excess / threshold_floor)
                    max_risk = max(max_risk, risk_floor)
                # If impact_vel <= threshold, no risk from floor

        # Calculate ceiling impact risk (always checked if ceiling contact exists)
        # Matches ninja.py line 383-407 (no airborn_old check for ceiling)
        if has_ceiling_contact:
            # Calculate impact velocity using exact formula from ninja.py line 395-397
            impact_vel_ceiling = -(
                ceiling_normalized_x * xspeed_old + ceiling_normalized_y * yspeed_old
            )

            # Calculate threshold using exact formula from ninja.py line 399-400
            threshold_ceiling = MAX_SURVIVABLE_IMPACT - (4.0 / 3.0) * abs(
                ceiling_normalized_y
            )

            # Calculate risk as normalized distance above threshold
            if threshold_ceiling > 1e-6:  # Avoid division by zero
                if impact_vel_ceiling > threshold_ceiling:
                    # Impact velocity exceeds threshold - calculate risk
                    excess = impact_vel_ceiling - threshold_ceiling
                    risk_ceiling = min(1.0, excess / threshold_ceiling)
                    max_risk = max(max_risk, risk_ceiling)
                # If impact_vel <= threshold, no risk from ceiling

        # Return potential: lower risk = higher potential
        # Raw potential in [0.0, 1.0] range (weight applied by caller)
        potential = 1.0 - max_risk
        return max(0.0, min(1.0, potential))

    @staticmethod
    def exploration_potential(
        state: Dict[str, Any], visited_positions: List[Tuple[float, float]]
    ) -> float:
        """Potential based on exploration of new areas.

        Encourages visiting unexplored regions of the level.

        Args:
            state: Game state dictionary
            visited_positions: List of previously visited (x, y) positions

        Returns:
            float: Potential in range [0.0, 1.0], higher in unexplored areas
        """
        player_x, player_y = state["player_x"], state["player_y"]

        if not visited_positions:
            return 1.0  # First position, maximum exploration potential

        # Find minimum distance to any previously visited position
        min_distance = float("inf")
        for prev_x, prev_y in visited_positions:
            distance = calculate_distance(player_x, player_y, prev_x, prev_y)
            min_distance = min(min_distance, distance)

        # Exploration radius - positions within this distance are considered "visited"
        exploration_radius = PBRS_EXPLORATION_RADIUS

        # Return potential based on distance from visited areas
        if min_distance > exploration_radius:
            return 1.0  # Far from visited areas, high exploration potential
        else:
            # Linear decay within exploration radius
            return 1.0 - (min_distance / exploration_radius * PBRS_EXPLORATION_WEIGHT)


class PBRSCalculator:
    """Calculator for completion-focused potential functions.

    Combines multiple PBRS potentials with configurable weights to create
    a composite potential function. The calculator maintains state across
    steps and computes the shaping reward: F(s,s') = γ * Φ(s') - Φ(s)

    Uses shortest path distances for objective distance calculations, respecting
    level geometry and obstacles. REQUIRES adjacency graph and level_data when
    calculating potentials.

    All constants imported from reward_constants.py for consistency.
    """

    def __init__(
        self,
        path_calculator: Optional[Any] = None,
    ):
        """Initialize PBRS calculator for completion-focused training.

        Args:
            objective_weight: Weight for objective distance potential (switch/exit)
            hazard_weight: Weight for hazard proximity potential (0.0 = disabled)
            impact_weight: Weight for impact risk potential (0.0 = disabled)
            exploration_weight: Weight for exploration potential (0.0 = disabled)
            path_calculator: CachedPathDistanceCalculator instance for path-aware distances
        """
        # Initialize path distance calculator for path-aware reward shaping
        self.path_calculator = path_calculator

        # Cache for surface area per level (invalidated when level or switch states change)
        self._cached_surface_area: Optional[float] = None
        self._cached_level_id: Optional[str] = None
        self._cached_switch_states: Optional[Tuple[Tuple[str, bool], ...]] = None

        # Cache for reachable mine positions per level (performance optimization)
        # Key: level_id, Value: Dict mapping entity id -> (xpos, ypos)
        # Only includes mines that are reachable from player start position
        self._cached_reachable_mines: Optional[Dict[int, Tuple[float, float]]] = None
        self._cached_mines_level_id: Optional[str] = None

        # Track visited positions for exploration potential (minimal usage)
        self.visited_positions: List[Tuple[float, float]] = []
        self.visit_threshold = PBRS_EXPLORATION_VISIT_THRESHOLD

    def _compute_reachable_surface_area(
        self,
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute total reachable surface area as number of sub-nodes.

        STRICT - NO FALLBACKS: Throws error if adjacency is empty or None.

        Uses flood-fill from player start position to count only nodes actually
        reachable from spawn, matching the behavior shown in debug overlay visualization.
        This ensures surface area calculation respects actual navigable space,
        not just all nodes in the adjacency graph.

        Surface area scaling rationale:
        - Small confined levels: fewer nodes → less normalization → stronger gradients
        - Large open levels: more nodes → more normalization → consistent gradients
        - Scale-invariant: gradient strength proportional to level complexity

        Args:
            adjacency: Graph adjacency structure (may include unreachable nodes)
            level_data: Level data object (must have start_position attribute)
            graph_data: Optional graph data dict with spatial_hash for O(1) node lookup

        Returns:
            Total number of reachable sub-nodes from start position (surface area metric)

        Raises:
            RuntimeError: If adjacency is None or empty, or start_position is missing
        """
        if not adjacency:
            raise RuntimeError(
                "PBRS surface area calculation failed: adjacency graph is empty or None.\n"
                "PBRS requires valid graph data to compute surface-area-based normalization.\n"
                "This typically means:\n"
                "  1. Graph building is not enabled in environment config\n"
                "  2. Graph builder failed to create reachable nodes\n"
                "  3. Level has no traversable space\n"
            )

        # Get player start position from level_data
        start_position = getattr(level_data, "start_position", None)
        if start_position is None:
            raise RuntimeError(
                "PBRS surface area calculation failed: level_data missing start_position.\n"
                "Surface area calculation requires player start position to compute reachability.\n"
                "Verify that level_data.start_position is set correctly."
            )

        # Convert start position to integer tuple
        # start_position is in tile data space, but _flood_fill_reachable_nodes
        # expects world space coordinates (it uses find_closest_node_to_position which expects world space)
        # Convert from tile data space to world space by adding NODE_WORLD_COORD_OFFSET (+24)
        from ...graph.reachability.pathfinding_utils import NODE_WORLD_COORD_OFFSET

        start_pos = (
            int(start_position[0]) + NODE_WORLD_COORD_OFFSET,
            int(start_position[1]) + NODE_WORLD_COORD_OFFSET,
        )

        # Use flood-fill to find all nodes reachable from start position
        # Pass graph_data for O(1) lookup-based node finding
        reachable_nodes = _flood_fill_reachable_nodes(start_pos, adjacency, graph_data)
        total_reachable_nodes = len(reachable_nodes)

        if total_reachable_nodes == 0:
            raise RuntimeError(
                "PBRS surface area calculation failed: no nodes reachable from start position.\n"
                "This means the level has no reachable space from player spawn.\n"
                "Verify that:\n"
                "  1. Level geometry allows player movement\n"
                "  2. Player spawn position is valid\n"
                "  3. Graph builder successfully found start position"
            )

        return float(total_reachable_nodes)

    def _precompute_reachable_mines(
        self,
        state: Dict[str, Any],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, Tuple[float, float]]:
        """Pre-compute which mines are reachable and cache their positions.

        This method is called once per level to build a lookup table of reachable
        mine positions. Since mine positions don't change during a level, this
        eliminates thousands of repeated reachability checks per episode.

        Performance optimization rationale:
        - Profiling shows find_closest_node_to_position is called ~29 times per
          hazard_proximity_potential call (14,160 times for 496 calls)
        - Pre-computing reachability once per level reduces this to O(num_mines)
          one-time cost instead of O(num_mines * num_steps)

        Args:
            state: Game state containing entities
            adjacency: Graph adjacency structure for reachability checking
            graph_data: Graph data dict with spatial_hash for O(1) lookup

        Returns:
            Dict mapping entity id -> (xpos, ypos) for reachable mines only
        """
        reachable_mines = {}
        entities = state.get("entities", [])
        if not entities or adjacency is None:
            return reachable_mines

        # Extract spatial lookups for O(1) reachability checking
        spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
            graph_data
        )

        # Constants for reachability check
        REACHABILITY_THRESHOLD = 50.0  # pixels

        # Single pass: filter dangerous mines and check reachability once
        for e in entities:
            # Only check toggle mines (same filter as hazard_proximity_potential)
            if not (
                hasattr(e, "type")
                and hasattr(e, "state")
                and e.type in (EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED)
            ):
                continue

            if not (hasattr(e, "xpos") and hasattr(e, "ypos")):
                continue

            # Check reachability once per mine
            mine_pos = (int(e.xpos), int(e.ypos))
            closest_node = find_closest_node_to_position(
                mine_pos,
                adjacency,
                threshold=REACHABILITY_THRESHOLD,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
            )

            # Only cache reachable mines
            if closest_node is not None:
                reachable_mines[id(e)] = (e.xpos, e.ypos)

        return reachable_mines

    def calculate_combined_potential(
        self,
        state: Dict[str, Any],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate completion-focused potential using surface area scaling.

        STRICT REQUIREMENTS:
        - adjacency and level_data are REQUIRED (no Optional)
        - path_calculator MUST be initialized (always is)
        - Throws descriptive errors if any required data is missing

        Args:
            state: Game state dictionary
            adjacency: Graph adjacency structure (REQUIRED)
            level_data: Level data object (REQUIRED)
            graph_data: Graph data dict with spatial_hash for O(1) node lookup (optional)

        Returns:
            float: Combined potential value

        Raises:
            RuntimeError: If required data is missing or invalid
        """
        # Strict validation - no fallbacks
        if adjacency is None:
            raise RuntimeError(
                "PBRS calculate_combined_potential requires adjacency graph.\n"
                "Adjacency is None, which means graph building failed or is disabled.\n"
                "Fix: Ensure graph building is properly configured in environment."
            )

        if level_data is None:
            raise RuntimeError(
                "PBRS calculate_combined_potential requires level_data.\n"
                "Level data is None, which should never happen.\n"
                "This indicates a serious configuration error in the environment."
            )

        if self.path_calculator is None:
            raise RuntimeError(
                "PBRS calculator has no path_calculator initialized.\n"
                "This should never happen - path_calculator is required for PBRS.\n"
                "Check PBRSCalculator initialization in RewardCalculator."
            )

        # Get or compute level ID for caching
        level_id = getattr(level_data, "level_id", None)
        if level_id is None:
            level_id = str(getattr(level_data, "start_position", "unknown"))

        # Get switch states signature for cache invalidation
        switch_states = getattr(level_data, "switch_states", {})
        # Create hashable signature: sorted tuple of (switch_id, activated) pairs
        switch_states_signature = (
            tuple(sorted(switch_states.items())) if switch_states else ()
        )

        # Check if cache needs invalidation (level changed or switch states changed)
        cache_invalid = (
            self._cached_level_id != level_id
            or self._cached_switch_states != switch_states_signature
            or self._cached_surface_area is None
        )

        # Compute and cache surface area per level (recalculate when level or switches change)
        if cache_invalid:
            self._cached_surface_area = self._compute_reachable_surface_area(
                adjacency, level_data, graph_data
            )
            self._cached_level_id = level_id
            self._cached_switch_states = switch_states_signature

        surface_area = self._cached_surface_area

        # Compute and cache reachable mines per level (for performance optimization)
        # Only compute if level changed or not cached yet
        if (
            self._cached_mines_level_id != level_id
            or self._cached_reachable_mines is None
        ):
            self._cached_reachable_mines = self._precompute_reachable_mines(
                state, adjacency, graph_data
            )
            self._cached_mines_level_id = level_id

        # Add surface area to state for potential calculation
        state_with_metrics = dict(state)
        state_with_metrics["_pbrs_surface_area"] = surface_area

        # Calculate objective distance potential with surface area scaling
        # All parameters are required - will throw descriptive errors if missing
        objective_pot = PBRSPotentials.objective_distance_potential(
            state_with_metrics,
            adjacency=adjacency,
            level_data=level_data,
            path_calculator=self.path_calculator,
            graph_data=graph_data,
        )

        # Calculate safety potentials (hazard proximity and impact risk)
        # These provide safety signals without overwhelming completion objective
        # Pass pre-computed reachable mines lookup for performance
        hazard_pot = PBRSPotentials.hazard_proximity_potential(
            state_with_metrics,
            adjacency=adjacency,
            graph_data=graph_data,
            reachable_mines_lookup=self._cached_reachable_mines,
        )
        impact_pot = PBRSPotentials.impact_risk_potential(state)

        # Apply phase-specific scaling (switch vs exit) for objective potential
        if not state.get("switch_activated", False):
            objective_component = (
                PBRS_SWITCH_DISTANCE_SCALE * objective_pot * PBRS_OBJECTIVE_WEIGHT
            )
        else:
            objective_component = (
                PBRS_EXIT_DISTANCE_SCALE * objective_pot * PBRS_OBJECTIVE_WEIGHT
            )

        # Combine all potentials with their respective weights
        # Objective potential is the primary signal, safety potentials provide guidance
        combined_potential = (
            objective_component
            + hazard_pot * PBRS_HAZARD_WEIGHT
            + impact_pot * PBRS_IMPACT_WEIGHT
        )

        # Clamp to reasonable range (objective component is the dominant term)
        max_potential = max(PBRS_SWITCH_DISTANCE_SCALE, PBRS_EXIT_DISTANCE_SCALE)
        return max(0.0, min(max_potential, combined_potential))

    def _update_visited_positions(self, x: float, y: float) -> None:
        """Update the list of visited positions."""
        # Check if this position is significantly different from previous ones
        for prev_x, prev_y in self.visited_positions:
            if calculate_distance(x, y, prev_x, prev_y) < self.visit_threshold:
                return  # Too close to existing position

        # Add new position
        self.visited_positions.append((x, y))

        # Limit memory to prevent unbounded growth
        max_positions = 100
        if len(self.visited_positions) > max_positions:
            self.visited_positions = self.visited_positions[-max_positions:]

    def get_potential_components(
        self,
        state: Dict[str, Any],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Get individual potential components for debugging/logging.

        Args:
            state: Game state dictionary
            adjacency: Graph adjacency structure (REQUIRED)
            level_data: Level data object (REQUIRED)
            graph_data: Graph data dict with spatial_hash for O(1) node lookup (optional)

        Returns:
            dict: Dictionary of potential component values (completion-focused)
        """
        # Get or compute level ID for caching
        level_id = getattr(level_data, "level_id", None)
        if level_id is None:
            level_id = str(getattr(level_data, "start_position", "unknown"))

        # Get switch states signature for cache invalidation
        switch_states = getattr(level_data, "switch_states", {})
        # Create hashable signature: sorted tuple of (switch_id, activated) pairs
        switch_states_signature = (
            tuple(sorted(switch_states.items())) if switch_states else ()
        )

        # Check if cache needs invalidation (level changed or switch states changed)
        cache_invalid = (
            self._cached_level_id != level_id
            or self._cached_switch_states != switch_states_signature
            or self._cached_surface_area is None
        )

        # Compute and cache surface area per level (recalculate when level or switches change)
        if cache_invalid:
            self._cached_surface_area = self._compute_reachable_surface_area(
                adjacency, level_data, graph_data
            )
            self._cached_level_id = level_id
            self._cached_switch_states = switch_states_signature

        surface_area = self._cached_surface_area

        # Compute and cache reachable mines per level (for performance optimization)
        # Only compute if level changed or not cached yet
        if (
            self._cached_mines_level_id != level_id
            or self._cached_reachable_mines is None
        ):
            self._cached_reachable_mines = self._precompute_reachable_mines(
                state, adjacency, graph_data
            )
            self._cached_mines_level_id = level_id

        state_with_metrics = dict(state)
        state_with_metrics["_pbrs_surface_area"] = surface_area

        objective_pot = PBRSPotentials.objective_distance_potential(
            state_with_metrics,
            adjacency=adjacency,
            level_data=level_data,
            path_calculator=self.path_calculator,
            graph_data=graph_data,
        )

        # Calculate hazard and impact potentials for safety signals
        # Pass pre-computed reachable mines lookup for performance
        hazard_pot = PBRSPotentials.hazard_proximity_potential(
            state_with_metrics,
            adjacency=adjacency,
            graph_data=graph_data,
            reachable_mines_lookup=self._cached_reachable_mines,
        )
        impact_pot = PBRSPotentials.impact_risk_potential(state)

        return {
            "objective": objective_pot,
            "switch_distance_potential": PBRS_SWITCH_DISTANCE_SCALE * objective_pot
            if not state.get("switch_activated", False)
            else 0.0,
            "exit_distance_potential": PBRS_EXIT_DISTANCE_SCALE * objective_pot
            if state.get("switch_activated", False)
            else 0.0,
            "surface_area": surface_area,
            "sqrt_surface_area": np.sqrt(surface_area),
            "area_scale_px": np.sqrt(surface_area) * SUB_NODE_SIZE,
            "hazard": hazard_pot,
            "impact": impact_pot,
            "exploration": 0.0,  # Disabled - explicit exploration rewards handle this
        }

    def reset(self) -> None:
        """Reset calculator state for new episode."""
        self.visited_positions.clear()
        # Keep surface area cache - it's per level, not per episode
        # Will be invalidated when level_data or switch_states change
