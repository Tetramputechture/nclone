"""Potential-Based Reward Shaping (PBRS) potential functions.

This module implements reusable potential functions Φ(s) for reward shaping
following the theory of Ng, Harada, and Russell (1999): "Policy Invariance
Under Reward Transformations: Theory and Application to Reward Shaping".

PBRS provides dense reward signals without changing the optimal policy by using
the formula: F(s,s') = γ * Φ(s') - Φ(s)

Key properties:
- Policy invariance: Optimal policy unchanged under PBRS (holds for ANY γ)
- Dense rewards: Provides gradient at every step
- Normalization: All potentials normalized to [0, 1] range
- Composability: Multiple potentials can be combined with weights

Discount Factor (γ) Choice:
- We use γ=1.0 for heuristic potential functions (path distance)
- This eliminates negative bias: accumulated PBRS = Φ(goal) - Φ(start) exactly
- For heuristic potentials, γ should NOT match MDP's discount (that's only for learned V(s))
- Policy invariance holds for any γ, but γ=1.0 is standard for episodic tasks

Mathematical Justification:
With γ < 1.0, accumulated PBRS = γ*Φ(goal) - Φ(start) + (γ-1)*Σ Φ(intermediate)
The (γ-1)*Σ term is negative and grows with episode length, causing systematic negative bias.
With γ=1.0, this bias term vanishes, giving clean telescoping: Φ(goal) - Φ(start).

All constants defined in reward_constants.py with full documentation.
"""

import logging
import math
from typing import Dict, Any, List, Tuple, Optional
from .reward_constants import (
    PBRS_SWITCH_DISTANCE_SCALE,
    PBRS_EXIT_DISTANCE_SCALE,
    VELOCITY_ALIGNMENT_MIN_SPEED,
    VELOCITY_ALIGNMENT_WEIGHT,
)

logger = logging.getLogger(__name__)

# Sub-node size for surface area calculations (from graph builder)
# The graph builder creates a 2x2 grid of sub-nodes per 24px tile
SUB_NODE_SIZE = 12  # pixels per sub-node
PLAYER_RADIUS = 10  # Player collision radius in pixels (from graph_builder)

# Node coordinate offset for world coordinate conversion (same as pathfinding_utils)
NODE_WORLD_COORD_OFFSET = 24


def get_path_gradient_from_next_hop(
    player_pos: Tuple[float, float],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    level_cache: Any,
    goal_id: str,
    spatial_hash: Optional[Any] = None,
    subcell_lookup: Optional[Any] = None,
) -> Optional[Tuple[float, float]]:
    """Get path gradient using next_hop direction (respects winding paths).

    Instead of computing direction toward goal (Euclidean), this computes
    direction toward the next_hop node on the optimal path. This correctly
    handles levels where the path goes away from the goal first.

    Args:
        player_pos: Current player position (x, y) in world coordinates
        adjacency: Graph adjacency structure
        level_cache: LevelBasedPathDistanceCache with precomputed next_hop data
        goal_id: Goal identifier ("switch" or "exit")
        spatial_hash: Optional spatial hash for fast node lookup
        subcell_lookup: Optional subcell lookup for node snapping

    Returns:
        Normalized direction vector (dx, dy) pointing toward next_hop,
        or None if at goal or no path available.
    """
    from ...graph.reachability.pathfinding_utils import find_ninja_node

    if level_cache is None:
        return None

    # Find current node the player is on/near
    ninja_node = find_ninja_node(
        (int(player_pos[0]), int(player_pos[1])),
        adjacency,
        spatial_hash=spatial_hash,
        subcell_lookup=subcell_lookup,
        ninja_radius=PLAYER_RADIUS,
    )

    if ninja_node is None:
        return None

    # Get next hop toward goal from precomputed cache
    next_hop = level_cache.get_next_hop(ninja_node, goal_id)
    if next_hop is None:
        # At goal or no path - no gradient
        return None

    # Compute direction from current node to next_hop (in tile data space)
    # Both nodes are in tile data space, so we can directly compute direction
    dx = next_hop[0] - ninja_node[0]
    dy = next_hop[1] - ninja_node[1]
    length = (dx * dx + dy * dy) ** 0.5

    if length < 0.001:
        # next_hop is same as current node (shouldn't happen)
        return None

    # Return normalized direction vector
    return (dx / length, dy / length)


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
        scale_factor: float = 1.0,
    ) -> float:
        """Potential based on GEOMETRIC path distance to objective (in pixels).

        Uses geometric path normalization: Φ(s) = 1 - (distance / combined_path_distance)
        where both distance and combined_path_distance are actual pixel distances.

        IMPORTANT: Uses geometric distances (actual path length in pixels), NOT
        physics-weighted costs. This ensures PBRS gradients are proportional to
        actual movement and provides predictable, meaningful reward signals.

        Returns higher potential when closer to the current objective:
        - Switch when inactive
        - Exit when switch is active

        Args:
            state: Game state dictionary (must contain _pbrs_combined_path_distance)
            adjacency: Graph adjacency structure (always present)
            level_data: Level data object (always present)
            path_calculator: CachedPathDistanceCalculator instance (uses cached geometric distances)
            graph_data: Graph data dict with spatial_hash for O(1) node lookup
            scale_factor: DEPRECATED - no longer used, kept for backward compatibility

        Returns:
            float: Potential in range [0.0, 1.0], linearly proportional to progress.
                   0.0 at spawn, 1.0 at goal.
        """
        # Note: adjacency, level_data, and graph_data are assumed
        # to always be present. Validation removed for performance.

        # Determine goal position
        from ...constants.physics_constants import (
            EXIT_SWITCH_RADIUS,
            EXIT_DOOR_RADIUS,
            NINJA_RADIUS,
        )

        if not state["switch_activated"]:
            goal_pos = (int(state["switch_x"]), int(state["switch_y"]))
            entity_radius = EXIT_SWITCH_RADIUS
        else:
            goal_pos = (int(state["exit_door_x"]), int(state["exit_door_y"]))
            entity_radius = EXIT_DOOR_RADIUS

        player_pos = (int(state["player_x"]), int(state["player_y"]))

        # Extract base_adjacency for pathfinding
        base_adjacency = (
            graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
        )

        # Calculate GEOMETRIC path distance using cached values when available
        # This returns actual pixel distance, not physics-weighted cost
        try:
            distance = path_calculator.get_geometric_distance(
                player_pos,
                goal_pos,
                adjacency,
                base_adjacency,
                level_data=level_data,
                graph_data=graph_data,
                entity_radius=entity_radius,
                ninja_radius=NINJA_RADIUS,
            )
        except Exception as e:
            raise RuntimeError(
                f"PBRS geometric path distance calculation failed: {e}\n"
                "Check player/goal positions are within level bounds."
            ) from e

        # Get combined path distance for DIRECT path normalization (no scaling factors)
        combined_path_distance = state.get("_pbrs_combined_path_distance")
        if combined_path_distance is None:
            raise RuntimeError(
                "Missing '_pbrs_combined_path_distance' in state. "
                "Should be set by PBRSCalculator.calculate_combined_potential()."
            )

        # Handle unreachable goals
        if distance == float("inf") or combined_path_distance == float("inf"):
            # Unreachable: return zero potential (no gradient)
            state["_pbrs_normalized_distance"] = float("inf")
            return 0.0

        # ADAPTIVE NORMALIZATION: Full-path gradient coverage
        #
        # Problem with fixed 800px cap or ADAPTIVE_FACTOR < 1.0:
        #   - Long levels (>800px): spawn potential = 1 - 1200/800 = -0.5 → clamped to 0
        #   - This creates a "dead zone" where first portion produces zero PBRS reward
        #   - With factor=0.5 and 1300px path: dead zone = 500px (38% of path!)
        #
        # Solution: Adaptive cap = max(800, combined_path * 1.0)
        # - Short paths (400px): max(800, 400) = 800 → strong gradients
        # - Medium paths (1000px): max(800, 1000) = 1000 → no dead zone, spawn Φ=0.0
        # - Long paths (1300px): max(800, 1300) = 1300 → no dead zone, spawn Φ=0.0
        # - Very long paths (3000px): max(800, 3000) = 3000 → full gradient coverage
        #
        # Gradient strength examples (with effective_norm = max(800, path)):
        #   Short level (800px):   ΔΦ = 1px/800 = 0.00125 → strong signal
        #   Long level (1300px):   ΔΦ = 1px/1300 = 0.00077 → adequate signal
        #   Very long (3000px):    ΔΦ = 1px/3000 = 0.00033 → weaker but present
        MIN_NORMALIZATION_DISTANCE = (
            800.0  # Minimum cap for strong gradients on short levels
        )
        ADAPTIVE_FACTOR = 1.0  # Use full path for normalization to eliminate dead zone
        effective_normalization = max(
            MIN_NORMALIZATION_DISTANCE, combined_path_distance * ADAPTIVE_FACTOR
        )

        # ADAPTIVE PATH NORMALIZATION: Full gradient coverage, no dead zones
        # Φ(s) = 1 - (distance_to_goal / max(800, combined_path))
        #
        # Properties:
        # - Linear gradient throughout: dΦ/dd = -1/effective_normalization
        # - At spawn: potential = 0.0 exactly (normalized_distance = 1.0)
        # - At goal: potential = 1.0 (distance = 0)
        # - No dead zones: spawn potential is exactly 0, any progress gives positive reward
        # - PBRS-compatible: F(s,s') = γ * Φ(s') - Φ(s) maintains policy invariance
        #
        # Curriculum scaling (via objective_weight in calculate_combined_potential):
        #   - Discovery (weight=15): Strong signal for exploration
        #   - Early (weight=8-12): Moderate guidance
        #   - Late (weight=4): Light shaping for refinement

        normalized_distance = distance / effective_normalization
        potential = 1.0 - normalized_distance

        # Store normalized distance for diagnostic logging
        state["_pbrs_normalized_distance"] = normalized_distance

        return max(0.0, min(1.0, potential))

    @staticmethod
    def velocity_alignment_potential(
        state: Dict[str, Any],
        path_gradient: Optional[Tuple[float, float]] = None,
    ) -> float:
        """Potential based on velocity alignment with optimal path direction.

        Uses the pre-computed path gradient (direction of steepest potential increase)
        to determine if velocity is aligned with the optimal path, not just Euclidean
        direction to goal. This respects obstacles and level geometry.

        When path_gradient is not provided, falls back to Euclidean direction.

        This is policy-invariant as part of the potential function:
        F(s,s') = γ * Φ(s') - Φ(s) where Φ includes velocity alignment.

        Args:
            state: Game state dictionary with velocity and goal positions
            path_gradient: Optional (gx, gy) normalized direction toward goal via optimal path.
                          If None, falls back to Euclidean direction.

        Returns:
            float: Potential in range [-1.0, 1.0]
                   1.0 = moving directly along optimal path toward goal
                   0.0 = moving perpendicular or too slow
                   -1.0 = moving directly away from goal on optimal path
        """
        # Get velocity components
        vx = state.get("player_xspeed", 0.0)
        vy = state.get("player_yspeed", 0.0)

        # Calculate speed
        speed = math.sqrt(vx * vx + vy * vy)

        # Return 0 if speed is below threshold (direction too noisy)
        if speed < VELOCITY_ALIGNMENT_MIN_SPEED:
            return 0.0

        # Normalize velocity direction
        vel_dir_x = vx / speed
        vel_dir_y = vy / speed

        # Use path gradient if available, otherwise fall back to Euclidean direction
        if path_gradient is not None:
            grad_x, grad_y = path_gradient
            grad_mag = math.sqrt(grad_x * grad_x + grad_y * grad_y)
            if grad_mag > 0.001:
                # Normalize gradient direction
                path_dir_x = grad_x / grad_mag
                path_dir_y = grad_y / grad_mag
            else:
                # Gradient too small (at or very close to goal)
                return 1.0
        else:
            # Fallback: Euclidean direction to goal
            if not state.get("switch_activated", False):
                goal_x = state["switch_x"]
                goal_y = state["switch_y"]
            else:
                goal_x = state["exit_door_x"]
                goal_y = state["exit_door_y"]

            dx = goal_x - state["player_x"]
            dy = goal_y - state["player_y"]
            dist_to_goal = math.sqrt(dx * dx + dy * dy)

            if dist_to_goal < 1.0:
                return 1.0  # At goal

            path_dir_x = dx / dist_to_goal
            path_dir_y = dy / dist_to_goal

        # Calculate cosine of angle between velocity and path direction (dot product)
        # Range: [-1, 1] where 1 = same direction, -1 = opposite direction
        alignment = vel_dir_x * path_dir_x + vel_dir_y * path_dir_y

        return alignment


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
        """Initialize simplified PBRS calculator (objective potential only).

        Args:
            path_calculator: CachedPathDistanceCalculator instance for shortest path distances
        """
        # Initialize path distance calculator for path-aware reward shaping
        self.path_calculator = path_calculator

        # Cache for surface area per level (invalidated when level or switch states change)
        self._cached_surface_area: Optional[float] = None
        self._cached_level_id: Optional[str] = None
        self._cached_switch_states: Optional[Tuple[Tuple[str, bool], ...]] = None

        # Cache for combined path distance per level (spawn→switch + switch→exit)
        self._cached_combined_path_distance: Optional[float] = None
        self._path_distance_cache_key: Optional[str] = None

        # OPTIMIZATION: Position-based caching to avoid repeated pathfinding
        # Only recalculate path distance if ninja moves 3+ pixels
        self._last_player_pos: Optional[Tuple[int, int]] = None
        self._last_goal_pos: Optional[Tuple[int, int]] = None
        self._cached_objective_potential: Optional[float] = None

    def _compute_reachable_surface_area(
        self,
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute total reachable surface area as number of sub-nodes with caching.

        Uses flood-fill from player start position to count only nodes actually
        reachable from spawn, matching the behavior shown in debug overlay visualization.

        Args:
            adjacency: Graph adjacency structure (always present)
            level_data: Level data object with start_position attribute
            graph_data: Graph data dict with spatial_hash for O(1) node lookup

        Returns:
            Total number of reachable sub-nodes from start position (surface area metric)
        """
        from ...graph.reachability.pathfinding_utils import get_cached_surface_area

        # Get player start position from level_data
        start_position = getattr(level_data, "start_position", None)

        # Generate cache key using LevelData utility method for consistency
        # This ensures all components use the same cache key format
        cache_key = level_data.get_cache_key_for_reachability(
            include_switch_states=True
        )

        # start_position is already in world space (no conversion needed)
        # This matches the coordinate space used by ninja_position() and flood_fill_reachable_nodes()
        start_pos = (
            int(start_position[0]),
            int(start_position[1]),
        )

        # Use cached surface area calculation to avoid recomputation
        # Wrap in try-except for graceful fallback on degenerate maps
        try:
            return get_cached_surface_area(cache_key, start_pos, adjacency, graph_data)
        except RuntimeError as e:
            # Critical: Surface area computation failed
            logger.error(
                f"CRITICAL: Failed to compute surface area: {e}\n"
                f"Using fallback value. This indicates a map generation bug!"
            )
            # Return a reasonable fallback (diagonal of typical level ~1400px)
            # This prevents crashes while still allowing detection of the bug
            return 1000.0  # Fallback to avoid crash

    def _compute_combined_path_distance(
        self,
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute combined GEOMETRIC path distance from spawn→switch + switch→exit.

        IMPORTANT: This returns the actual geometric path length in PIXELS, not physics-
        weighted costs. The physics-aware pathfinder uses costs like 0.5 for grounded
        horizontal movement, which would return ~36 for an 800px path. For PBRS
        normalization, we need the actual path length (~800px).

        Uses BFS with geometric edge costs (12px for cardinal, ~17px for diagonal)
        to calculate the true path distance following the adjacency graph.

        Args:
            adjacency: Graph adjacency structure (always present)
            level_data: Level data object with start_position and entities
            graph_data: Graph data dict with spatial_hash for O(1) node lookup

        Returns:
            Combined geometric path distance in pixels (spawn→switch + switch→exit)
            Returns float('inf') if either path is unreachable
        """
        from ...constants.entity_types import EntityType
        from ...constants.physics_constants import (
            EXIT_SWITCH_RADIUS,
            EXIT_DOOR_RADIUS,
            NINJA_RADIUS,
        )
        from ...graph.reachability.pathfinding_utils import (
            calculate_geometric_path_distance,
            extract_spatial_lookups_from_graph_data,
        )

        # Get spawn position from level_data
        start_position = getattr(level_data, "start_position", None)
        spawn_pos = (int(start_position[0]), int(start_position[1]))

        # Get switch position from first exit switch
        exit_switches = level_data.get_entities_by_type(EntityType.EXIT_SWITCH)
        switch = exit_switches[0]
        switch_pos = (int(switch.get("x", 0)), int(switch.get("y", 0)))

        # Get exit door position from first exit door
        exit_doors = level_data.get_entities_by_type(EntityType.EXIT_DOOR)

        exit_door = exit_doors[0]
        exit_pos = (int(exit_door.get("x", 0)), int(exit_door.get("y", 0)))

        # Extract base_adjacency and spatial lookups for pathfinding
        base_adjacency = (
            graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
        )
        spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
            graph_data
        )
        physics_cache = graph_data.get("node_physics") if graph_data else None

        # Calculate spawn→switch GEOMETRIC distance (actual pixels, not physics costs)
        try:
            spawn_to_switch_dist = calculate_geometric_path_distance(
                spawn_pos,
                switch_pos,
                adjacency,
                base_adjacency,
                physics_cache=physics_cache,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
                entity_radius=EXIT_SWITCH_RADIUS,
                ninja_radius=NINJA_RADIUS,
            )
        except Exception as e:
            logger.warning(
                f"Failed to calculate spawn→switch distance: {e}. Using inf."
            )
            spawn_to_switch_dist = float("inf")

        # Calculate switch→exit GEOMETRIC distance (actual pixels, not physics costs)
        try:
            switch_to_exit_dist = calculate_geometric_path_distance(
                switch_pos,
                exit_pos,
                adjacency,
                base_adjacency,
                physics_cache=physics_cache,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
                entity_radius=EXIT_DOOR_RADIUS,
                ninja_radius=NINJA_RADIUS,
            )
        except Exception as e:
            logger.warning(f"Failed to calculate switch→exit distance: {e}. Using inf.")
            switch_to_exit_dist = float("inf")

        # Combine distances
        if spawn_to_switch_dist == float("inf") or switch_to_exit_dist == float("inf"):
            logger.warning(
                f"Combined path distance is infinite (unreachable goal). "
                f"spawn→switch: {spawn_to_switch_dist:.1f}, switch→exit: {switch_to_exit_dist:.1f}"
            )
            return float("inf")

        combined_distance = spawn_to_switch_dist + switch_to_exit_dist

        # DEBUG: Log combined path distance calculation details
        # Calculate Euclidean distances for comparison
        spawn_to_switch_euclidean = (
            (switch_pos[0] - spawn_pos[0]) ** 2 + (switch_pos[1] - spawn_pos[1]) ** 2
        ) ** 0.5
        switch_to_exit_euclidean = (
            (exit_pos[0] - switch_pos[0]) ** 2 + (exit_pos[1] - switch_pos[1]) ** 2
        ) ** 0.5
        euclidean_combined = spawn_to_switch_euclidean + switch_to_exit_euclidean

        logger.info(
            f"[PBRS DEBUG] Combined path distance calculation:\n"
            f"  Spawn position: {spawn_pos}\n"
            f"  Switch position: {switch_pos}\n"
            f"  Exit position: {exit_pos}\n"
            f"  Spawn→Switch path distance: {spawn_to_switch_dist:.1f}px\n"
            f"  Switch→Exit path distance: {switch_to_exit_dist:.1f}px\n"
            f"  Combined path distance: {combined_distance:.1f}px\n"
            f"  Spawn→Switch Euclidean: {spawn_to_switch_euclidean:.1f}px\n"
            f"  Switch→Exit Euclidean: {switch_to_exit_euclidean:.1f}px\n"
            f"  Euclidean combined: {euclidean_combined:.1f}px\n"
            f"  Path/Euclidean ratio: {combined_distance / euclidean_combined:.2f}x"
        )

        # CRITICAL: Warn if combined distance is suspiciously low
        if combined_distance < 100 and euclidean_combined > 200:
            logger.error(
                f"[PBRS BUG] Combined path distance ({combined_distance:.1f}px) is "
                f"much smaller than Euclidean distance ({euclidean_combined:.1f}px)! "
                f"This indicates a pathfinding bug."
            )

        return combined_distance

    def calculate_combined_potential(
        self,
        state: Dict[str, Any],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        graph_data: Optional[Dict[str, Any]] = None,
        objective_weight: float = 1.0,
        scale_factor: float = 1.0,
    ) -> float:
        """Calculate objective distance potential (simplified, curriculum-aware).

        Args:
            state: Game state dictionary
            adjacency: Graph adjacency structure (always present)
            level_data: Level data object (always present)
            graph_data: Graph data dict with spatial_hash for O(1) node lookup
            objective_weight: Curriculum-scaled weight from RewardConfig (default 1.0)
            scale_factor: Normalization adjustment from RewardConfig (default 1.0)

        Returns:
            float: Objective distance potential (curriculum-scaled)
        """
        # Note: adjacency, level_data, and path_calculator are assumed to always
        # be present. Validation removed for performance.

        # Get level ID for caching
        level_id = getattr(level_data, "level_id", None)
        if level_id is None:
            level_id = str(getattr(level_data, "start_position", "unknown"))

        # Get switch states for cache invalidation
        switch_states = getattr(level_data, "switch_states", {})
        switch_states_signature = (
            tuple(sorted(switch_states.items())) if switch_states else ()
        )

        # Check if cache needs invalidation
        cache_invalid = (
            self._cached_level_id != level_id
            or self._cached_switch_states != switch_states_signature
            or self._cached_surface_area is None
        )

        # Compute and cache surface area per level (kept for backward compatibility/fallback)
        if cache_invalid:
            self._cached_surface_area = self._compute_reachable_surface_area(
                adjacency, level_data, graph_data
            )
            self._cached_level_id = level_id
            self._cached_switch_states = switch_states_signature

            # Clear position-based cache when level/switch state changes
            self._last_player_pos = None
            self._last_goal_pos = None
            self._cached_objective_potential = None

        # Compute and cache combined path distance for path-based normalization
        # Cache key includes level_id to invalidate on level change
        path_cache_key = f"{level_id}_{switch_states_signature}"
        if (
            self._path_distance_cache_key != path_cache_key
            or self._cached_combined_path_distance is None
        ):
            self._cached_combined_path_distance = self._compute_combined_path_distance(
                adjacency, level_data, graph_data
            )
            self._path_distance_cache_key = path_cache_key
            logger.debug(
                f"Cached combined path distance: {self._cached_combined_path_distance:.1f}px for level {level_id}"
            )

        # Add metrics to state for potential calculation
        state_with_metrics = dict(state)
        state_with_metrics["_pbrs_surface_area"] = self._cached_surface_area
        state_with_metrics["_pbrs_combined_path_distance"] = (
            self._cached_combined_path_distance
        )

        # Always calculate fresh potential for accurate PBRS gradients
        objective_pot = PBRSPotentials.objective_distance_potential(
            state_with_metrics,
            adjacency=adjacency,
            level_data=level_data,
            path_calculator=self.path_calculator,
            graph_data=graph_data,
            scale_factor=scale_factor,  # Apply curriculum scaling
        )

        # Copy diagnostic values back to original state for logging
        # These are set by objective_distance_potential() in state_with_metrics
        state["_pbrs_normalized_distance"] = state_with_metrics.get(
            "_pbrs_normalized_distance", 0.0
        )
        state["_pbrs_combined_path_distance"] = state_with_metrics.get(
            "_pbrs_combined_path_distance"
        )
        state["_pbrs_surface_area"] = state_with_metrics.get("_pbrs_surface_area")

        # === VELOCITY ALIGNMENT WITH NEXT-HOP GRADIENT ===
        # Uses next_hop direction (respects winding paths) instead of Euclidean direction.
        # This provides continuous directional guidance even when optimal path goes away
        # from the goal first.
        velocity_alignment = 0.0
        path_gradient = None

        # Only compute if we have a level cache with next_hop data
        if (
            self.path_calculator is not None
            and self.path_calculator.level_cache is not None
            and VELOCITY_ALIGNMENT_WEIGHT > 0
        ):
            # Determine goal_id based on switch state
            goal_id = "switch" if not state.get("switch_activated", False) else "exit"

            # Extract spatial lookup from graph_data
            spatial_hash = graph_data.get("spatial_hash") if graph_data else None
            subcell_lookup = graph_data.get("subcell_lookup") if graph_data else None

            # Get path gradient from next_hop (respects winding paths)
            path_gradient = get_path_gradient_from_next_hop(
                player_pos=(state["player_x"], state["player_y"]),
                adjacency=adjacency,
                level_cache=self.path_calculator.level_cache,
                goal_id=goal_id,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
            )

            if path_gradient is not None:
                # Compute velocity alignment with path gradient
                velocity_alignment = PBRSPotentials.velocity_alignment_potential(
                    state, path_gradient=path_gradient
                )

        # Store for TensorBoard logging
        state["_pbrs_velocity_alignment"] = velocity_alignment
        state["_pbrs_velocity_weight"] = VELOCITY_ALIGNMENT_WEIGHT
        state["_pbrs_path_gradient"] = path_gradient

        # Apply phase-specific scaling (switch vs exit) and curriculum weight
        if not state.get("switch_activated", False):
            potential = PBRS_SWITCH_DISTANCE_SCALE * objective_pot * objective_weight
        else:
            potential = PBRS_EXIT_DISTANCE_SCALE * objective_pot * objective_weight

        # Add velocity alignment bonus (scaled by weight)
        # This is additive, not multiplicative, to preserve PBRS structure
        potential += VELOCITY_ALIGNMENT_WEIGHT * velocity_alignment

        return max(0.0, potential)

    def reset(self) -> None:
        """Reset calculator state for new episode.

        Clears position-based caching to ensure fresh potential calculations
        for the new episode. Surface area and combined path distance caches
        are kept as they are level-specific, not episode-specific.
        """
        # Clear position-based cache for fresh potential calculations
        # CRITICAL: Must reset when episode restarts (including same level reload)
        self._last_player_pos = None
        self._last_goal_pos = None
        self._cached_objective_potential = None

        # Keep surface area cache - it's per level, not per episode
        # Keep combined path distance cache - it's per level, not per episode
        # These will be invalidated when level_data or switch_states change
