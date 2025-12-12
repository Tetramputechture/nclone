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

Momentum-Aware Enhancements:
- Momentum waypoints: Intermediate goals for momentum-building strategies
- Multi-stage potentials: Route via waypoints when momentum is required
- Extracted from expert demonstrations to identify necessary "detours"

All constants defined in reward_constants.py with full documentation.
"""

import logging
import math
from typing import Dict, Any, List, Tuple, Optional
from .reward_constants import (
    PBRS_SWITCH_DISTANCE_SCALE,
    PBRS_EXIT_DISTANCE_SCALE,
    VELOCITY_ALIGNMENT_MIN_SPEED,
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
        raise ValueError("Level cache is None")

    # Find current node the player is on/near
    ninja_node = find_ninja_node(
        (int(player_pos[0]), int(player_pos[1])),
        adjacency,
        spatial_hash=spatial_hash,
        subcell_lookup=subcell_lookup,
        ninja_radius=PLAYER_RADIUS,
    )

    if ninja_node is None:
        raise ValueError(f"Ninja node is None for position: {player_pos}")

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
        raise ValueError(f"Next hop is same as current node: {next_hop}")

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
            path_calculator: PathDistanceCalculator instance (uses cached geometric distances)
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

        # CRITICAL: Validate goal position is not (0, 0) which indicates entity loading issue
        # Return 0 potential (neutral) if goal position is invalid to avoid cache pollution
        if goal_pos[0] == 0 and goal_pos[1] == 0:
            import logging

            _logger = logging.getLogger(__name__)
            _logger.warning(
                f"PBRS: Invalid goal position (0, 0) detected! "
                f"switch_activated={state.get('switch_activated', False)}, "
                f"Returning 0 potential to avoid cache pollution."
            )
            return 0.0

        # Extract base_adjacency for pathfinding
        base_adjacency = (
            graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
        )

        # === PHYSICS-WEIGHTED PBRS (Phase 2.1) ===
        # Use physics cost instead of geometric distance for normalization
        # This makes PBRS rewards proportional to physics difficulty reduction
        #
        # OLD: distance = get_geometric_distance() → Φ = 1 - (geo_dist / combined_geo)
        #      Problem: 12px grounded (easy) = same PBRS as 12px aerial (hard)
        #
        # NEW: distance = get_distance() → Φ = 1 - (physics_cost / combined_physics_cost)
        #      Benefit: Completing hard movement = more PBRS, completing easy = less PBRS
        #      Example: Grounded 12px (cost 0.15) vs aerial 12px (cost 40.0)
        #               Agent gets 267× more reward for completing the hard part!
        try:
            distance = path_calculator.get_distance(
                player_pos,
                goal_pos,
                adjacency,
                base_adjacency,
                level_data=level_data,
                graph_data=graph_data,
                entity_radius=entity_radius,
                ninja_radius=NINJA_RADIUS,
            )
            # OPTIMIZATION: Store distance in state for reuse in main_reward_calculator
            # This avoids redundant pathfinding calls for diagnostic tracking
            state["_pbrs_last_distance_to_goal"] = distance
        except Exception as e:
            raise RuntimeError(
                f"PBRS physics cost calculation failed: {e}\n"
                "Check player/goal positions are within level bounds."
            ) from e

        # Get combined physics cost for normalization (Phase 2.1)
        combined_physics_cost = state.get("_pbrs_combined_physics_cost")
        if combined_physics_cost is None:
            raise RuntimeError(
                "Missing '_pbrs_combined_physics_cost' in state. "
                "Should be set by PBRSCalculator.calculate_combined_potential()."
            )

        # Also get geometric distance for diagnostics
        combined_path_distance = state.get("_pbrs_combined_path_distance", 800.0)

        # Handle unreachable goals
        if distance == float("inf") or combined_physics_cost == float("inf"):
            # Unreachable: return zero potential (no gradient)
            state["_pbrs_normalized_distance"] = float("inf")
            state["_pbrs_normalized_physics_cost"] = float("inf")
            return 0.0

        # === PHYSICS-WEIGHTED NORMALIZATION (Phase 2.1) ===
        # Φ(s) = 1 - (remaining_physics_cost / total_physics_cost)
        #
        # This makes PBRS rewards proportional to physics difficulty reduction:
        # - Completing grounded horizontal (low cost): small PBRS reward
        # - Completing aerial horizontal (high cost): large PBRS reward
        # - Completing upward jumps (high cost): large PBRS reward
        #
        # Adaptive minimum for strong gradients on short/easy levels:
        # - Physics costs can be very low for pure grounded paths (~36 for 800px)
        # - Use min threshold to ensure adequate gradient strength
        # - For complex paths with jumps, physics cost >> geometric distance
        #
        # Estimate minimum cost: 800px pure grounded = 800 * 0.15 ≈ 120
        MIN_NORMALIZATION_COST = 120.0
        ADAPTIVE_FACTOR = 1.0
        effective_normalization = max(
            MIN_NORMALIZATION_COST, combined_physics_cost * ADAPTIVE_FACTOR
        )

        # PHYSICS-WEIGHTED POTENTIAL: Reflects movement difficulty
        # Φ(s) = 1 - (remaining_cost / total_cost)
        #
        # Properties:
        # - Gradient proportional to cost reduction: dΦ/dcost = -1/effective_normalization
        # - At spawn: potential = 0.0 (normalized_cost = 1.0)
        # - At goal: potential = 1.0 (cost = 0)
        # - Hard movements (high cost reduction) = higher PBRS reward
        # - Easy movements (low cost reduction) = lower PBRS reward
        # - PBRS-compatible: F(s,s') = γ * Φ(s') - Φ(s) maintains policy invariance
        #
        # Example with weight=20.0:
        #   - Complete 12px grounded (cost 1.8): PBRS = 20 × (1.8/1200) = 0.03
        #   - Complete 12px aerial (cost 480): PBRS = 20 × (480/1200) = 8.0
        #   - Agent learns: grounded movement is preferred (natural from reward structure)

        normalized_cost = distance / effective_normalization
        potential = 1.0 - normalized_cost

        # Store for diagnostic logging
        state["_pbrs_normalized_physics_cost"] = normalized_cost
        # Keep geometric for comparison
        state["_pbrs_normalized_distance"] = (
            path_calculator.get_geometric_distance(
                player_pos,
                goal_pos,
                adjacency,
                base_adjacency,
                level_data,
                graph_data,
                entity_radius,
                NINJA_RADIUS,
                goal_id="switch"
                if not state.get("switch_activated", False)
                else "exit",
            )
            / max(800.0, combined_path_distance)
            if combined_path_distance != float("inf")
            else float("inf")
        )

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

    @staticmethod
    def objective_distance_potential_with_waypoints(
        state: Dict[str, Any],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        path_calculator: Any,
        momentum_waypoints: Optional[List[Any]] = None,
        graph_data: Optional[Dict[str, Any]] = None,
        scale_factor: float = 1.0,
    ) -> float:
        """Potential based on path distance with momentum waypoint routing.

        When momentum waypoints are present, this potential function routes the agent
        through intermediate waypoints before heading to the goal. This prevents PBRS
        from penalizing necessary momentum-building "detours."

        The potential function becomes multi-stage:
        - Stage 1: current → nearest_waypoint (if waypoint needed)
        - Stage 2: waypoint → goal

        Waypoint is "needed" if:
        1. Agent hasn't reached it yet (distance > threshold)
        2. Waypoint is on the path toward goal (not behind agent)
        3. Agent doesn't have sufficient momentum to skip it

        Args:
            state: Game state dictionary
            adjacency: Graph adjacency structure
            level_data: Level data object
            path_calculator: PathDistanceCalculator instance
            momentum_waypoints: Optional list of MomentumWaypoint objects
            graph_data: Graph data dict with spatial_hash
            scale_factor: Normalization adjustment (deprecated, kept for compatibility)

        Returns:
            float: Potential in range [0.0, 1.0]
        """
        # If no waypoints provided, fall back to standard potential
        if not momentum_waypoints:
            return PBRSPotentials.objective_distance_potential(
                state, adjacency, level_data, path_calculator, graph_data, scale_factor
            )

        from ...constants.physics_constants import (
            EXIT_SWITCH_RADIUS,
            EXIT_DOOR_RADIUS,
            NINJA_RADIUS,
        )

        import logging

        logger = logging.getLogger(__name__)

        # Determine current goal
        if not state["switch_activated"]:
            goal_pos = (int(state["switch_x"]), int(state["switch_y"]))
            entity_radius = EXIT_SWITCH_RADIUS
        else:
            goal_pos = (int(state["exit_door_x"]), int(state["exit_door_y"]))
            entity_radius = EXIT_DOOR_RADIUS

        player_pos = (int(state["player_x"]), int(state["player_y"]))

        # Validate goal position
        if goal_pos[0] == 0 and goal_pos[1] == 0:
            logger.warning(
                "PBRS Waypoint: Invalid goal position (0, 0), using standard potential"
            )
            return PBRSPotentials.objective_distance_potential(
                state, adjacency, level_data, path_calculator, graph_data, scale_factor
            )

        # Extract base_adjacency for pathfinding
        base_adjacency = (
            graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
        )

        # Find relevant momentum waypoint (if any)
        active_waypoint = _find_active_momentum_waypoint(
            player_pos=player_pos,
            goal_pos=goal_pos,
            player_velocity=(
                state.get("player_xspeed", 0.0),
                state.get("player_yspeed", 0.0),
            ),
            waypoints=momentum_waypoints,
            path_calculator=path_calculator,
            adjacency=adjacency,
            base_adjacency=base_adjacency,
            level_data=level_data,
            graph_data=graph_data,
        )

        if active_waypoint is not None:
            # Route via waypoint: current → waypoint → goal
            waypoint_pos = (
                int(active_waypoint.position[0]),
                int(active_waypoint.position[1]),
            )

            try:
                # Distance to waypoint (momentum-aware pathfinding)
                dist_to_waypoint = path_calculator.get_geometric_distance(
                    player_pos,
                    waypoint_pos,
                    adjacency,
                    base_adjacency,
                    level_data=level_data,
                    graph_data=graph_data,
                    entity_radius=0.0,  # Waypoint is a position, not an entity
                    ninja_radius=NINJA_RADIUS,
                )

                # Distance from waypoint to goal (momentum-aware pathfinding)
                dist_waypoint_to_goal = path_calculator.get_geometric_distance(
                    waypoint_pos,
                    goal_pos,
                    adjacency,
                    base_adjacency,
                    level_data=level_data,
                    graph_data=graph_data,
                    entity_radius=entity_radius,
                    ninja_radius=NINJA_RADIUS,
                )

                # Check for unreachable paths
                if dist_to_waypoint == float("inf") or dist_waypoint_to_goal == float(
                    "inf"
                ):
                    # Waypoint routing failed, fall back to direct path
                    logger.debug(
                        "Waypoint routing failed (unreachable), using direct potential"
                    )
                    return PBRSPotentials.objective_distance_potential(
                        state,
                        adjacency,
                        level_data,
                        path_calculator,
                        graph_data,
                        scale_factor,
                    )

                # Normalize using same adaptive strategy as standard potential
                combined_path_distance = state.get(
                    "_pbrs_combined_path_distance", 800.0
                )
                MIN_NORMALIZATION_DISTANCE = 800.0
                ADAPTIVE_FACTOR = 1.0
                effective_normalization = max(
                    MIN_NORMALIZATION_DISTANCE, combined_path_distance * ADAPTIVE_FACTOR
                )

                # Calculate potential: 1.0 at goal, 0.0 at spawn, accounts for full path via waypoint
                # CRITICAL: Use FULL path distance (player→waypoint→goal) to maintain PBRS policy invariance
                # This prevents reward exploitation during backtracking/looping at waypoints
                total_distance = dist_to_waypoint + dist_waypoint_to_goal
                normalized_distance = total_distance / effective_normalization
                potential = 1.0 - normalized_distance

                # Store diagnostic info
                state["_pbrs_normalized_distance"] = normalized_distance
                state["_pbrs_using_waypoint"] = True
                state["_pbrs_waypoint_pos"] = waypoint_pos
                state["_pbrs_dist_to_waypoint"] = dist_to_waypoint
                state["_pbrs_dist_waypoint_to_goal"] = dist_waypoint_to_goal
                state["_pbrs_total_distance_via_waypoint"] = total_distance

                logger.debug(
                    f"Using waypoint routing: player→waypoint={dist_to_waypoint:.1f}px, "
                    f"waypoint→goal={dist_waypoint_to_goal:.1f}px, "
                    f"total={total_distance:.1f}px, potential={potential:.4f}"
                )

                return max(0.0, min(1.0, potential))

            except Exception as e:
                logger.warning(
                    "Waypoint potential calculation failed: %s, using standard potential",
                    e,
                )
                import traceback

                traceback.print_exc()
                return PBRSPotentials.objective_distance_potential(
                    state,
                    adjacency,
                    level_data,
                    path_calculator,
                    graph_data,
                    scale_factor,
                )
        else:
            # No active waypoint, use standard direct potential
            state["_pbrs_using_waypoint"] = False
            return PBRSPotentials.objective_distance_potential(
                state, adjacency, level_data, path_calculator, graph_data, scale_factor
            )


def _find_active_momentum_waypoint(
    player_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    player_velocity: Tuple[float, float],
    waypoints: List[Any],
    path_calculator: Any,
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    level_data: Any,
    graph_data: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Find the active momentum waypoint that agent should route through.

    A waypoint is "active" if:
    1. Agent hasn't reached it yet (distance > 20px threshold)
    2. Waypoint is between agent and goal (on the path)
    3. Agent doesn't have sufficient momentum to bypass it

    Args:
        player_pos: Current player position (x, y)
        goal_pos: Current goal position (x, y)
        player_velocity: Current player velocity (vx, vy)
        waypoints: List of MomentumWaypoint objects
        path_calculator: Path calculator for distance queries
        adjacency: Graph adjacency structure
        base_adjacency: Base adjacency for pathfinding
        level_data: Level data object
        graph_data: Optional graph data dict

    Returns:
        MomentumWaypoint if one is active, None otherwise
    """
    if not waypoints:
        return None

    WAYPOINT_REACHED_THRESHOLD = (
        20.0  # pixels - consider waypoint "reached" within this distance
    )
    MOMENTUM_SUFFICIENT_SPEED = (
        2.5  # pixels/frame - if moving this fast, can skip waypoint
    )

    # Calculate player speed
    vx, vy = player_velocity
    player_speed = math.sqrt(vx * vx + vy * vy)

    # Find waypoints that are:
    # 1. Not yet reached (distance > threshold)
    # 2. Between player and goal (routing makes sense)
    candidate_waypoints = []

    for waypoint in waypoints:
        waypoint_pos = (int(waypoint.position[0]), int(waypoint.position[1]))

        # Check if waypoint already reached
        dx = waypoint_pos[0] - player_pos[0]
        dy = waypoint_pos[1] - player_pos[1]
        distance_to_waypoint = math.sqrt(dx * dx + dy * dy)

        if distance_to_waypoint < WAYPOINT_REACHED_THRESHOLD:
            continue  # Already at waypoint, skip

        # Check if player has sufficient momentum to bypass waypoint
        # If moving fast in the right direction, don't force waypoint routing
        if player_speed >= MOMENTUM_SUFFICIENT_SPEED:
            # Check if velocity is aligned with waypoint's approach direction
            vel_dir_x = vx / player_speed
            vel_dir_y = vy / player_speed
            waypoint_dir_x, waypoint_dir_y = waypoint.approach_direction

            # Dot product: alignment check
            alignment = vel_dir_x * waypoint_dir_x + vel_dir_y * waypoint_dir_y

            if alignment > 0.7:  # Moving in similar direction with good speed
                continue  # Skip waypoint - agent already has momentum

        # Check if waypoint is "between" player and goal (heuristic)
        # Waypoint should be closer to goal than player is
        dist_player_to_goal = math.sqrt(
            (goal_pos[0] - player_pos[0]) ** 2 + (goal_pos[1] - player_pos[1]) ** 2
        )
        dist_waypoint_to_goal = math.sqrt(
            (goal_pos[0] - waypoint_pos[0]) ** 2 + (goal_pos[1] - waypoint_pos[1]) ** 2
        )

        if dist_waypoint_to_goal >= dist_player_to_goal:
            continue  # Waypoint is not between player and goal, skip

        candidate_waypoints.append((waypoint, distance_to_waypoint))

    if not candidate_waypoints:
        return None  # No active waypoints

    # Select closest waypoint as active (prioritize nearest momentum-building point)
    candidate_waypoints.sort(key=lambda x: x[1])
    active_waypoint = candidate_waypoints[0][0]

    return active_waypoint


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
        momentum_waypoints: Optional[List[Any]] = None,
        reward_config: Optional[Any] = None,
    ):
        """Initialize PBRS calculator with optional momentum waypoint support.

        Args:
            path_calculator: PathDistanceCalculator instance for shortest path distances
            momentum_waypoints: Optional list of MomentumWaypoint objects for momentum-aware routing
            reward_config: Optional RewardConfig instance for curriculum-adaptive weights
        """
        # Initialize path distance calculator for path-aware reward shaping
        self.path_calculator = path_calculator

        # Momentum waypoints for momentum-aware potential routing
        self.momentum_waypoints = momentum_waypoints or []

        # Reward configuration for curriculum-adaptive weights (Phase 1.2)
        self.reward_config = reward_config

        # Cache for surface area per level (invalidated when level or switch states change)
        self._cached_surface_area: Optional[float] = None
        self._cached_level_id: Optional[str] = None
        self._cached_switch_states: Optional[Tuple[Tuple[str, bool], ...]] = None

        # Cache for combined path distance per level (spawn→switch + switch→exit)
        self._cached_combined_path_distance: Optional[float] = None
        self._path_distance_cache_key: Optional[str] = None

        # Cache for combined physics cost per level (Phase 2.1 - physics-weighted PBRS)
        self._cached_combined_physics_cost: Optional[float] = None
        self._physics_cost_cache_key: Optional[str] = None

        # OPTIMIZATION: Position-based caching to avoid repeated pathfinding
        # Only recalculate path distance if ninja moves beyond threshold
        # 6px threshold: typical action moves 0.66-3px, so this allows 2-9 cached steps
        # Increased from 3px for better performance on large levels
        self._position_cache_threshold_sq = 6.0 * 6.0  # 6px squared (avoid sqrt)

        # PROFILING: Track cache hit/miss rates for optimization tuning
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_player_pos: Optional[Tuple[float, float]] = None
        self._last_goal_id: Optional[str] = (
            None  # Track goal changes (switch activation)
        )
        self._cached_distance: Optional[float] = None  # Cached geometric distance
        self._cached_start_node: Optional[Tuple[int, int]] = None  # Cached node lookup
        self._cached_next_hop: Optional[Tuple[int, int]] = (
            None  # Cached next_hop for interpolation
        )

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
    ) -> Tuple[float, float]:
        """Compute combined path metrics from spawn→switch + switch→exit.

        Returns BOTH geometric distance and physics cost for Phase 2.1 physics-weighted PBRS.

        GEOMETRIC distance: Actual path length in pixels (12px cardinal, ~17px diagonal)
        PHYSICS cost: Physics-weighted cost reflecting movement difficulty
            - Grounded horizontal: 0.15× (cheapest)
            - Aerial horizontal: 40× (very expensive)
            - Upward: varies (jump costs, aerial chain costs)
            - Downward: 0.5× (gravity assists)

        Physics-weighted PBRS uses physics cost for normalization to make rewards
        proportional to difficulty reduction, not just distance reduction.

        Args:
            adjacency: Graph adjacency structure (always present)
            level_data: Level data object with start_position and entities
            graph_data: Graph data dict with spatial_hash for O(1) node lookup

        Returns:
            Tuple of (combined_geometric_distance, combined_physics_cost) in pixels and cost units
            Returns (float('inf'), float('inf')) if either path is unreachable
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
        if not exit_switches:
            logger.warning(
                "calculate_level_surface_area: No exit switches found in level_data! "
                "Entity extraction may have failed."
            )
            return (float("inf"), float("inf"))
        switch = exit_switches[0]
        switch_x = switch.get("x", 0)
        switch_y = switch.get("y", 0)
        if switch_x == 0 and switch_y == 0:
            logger.warning(
                f"calculate_level_surface_area: Switch position is (0, 0), "
                f"entity data may be incomplete. Switch entity: {switch}"
            )
        switch_pos = (int(switch_x), int(switch_y))

        # Get exit door position from first exit door
        exit_doors = level_data.get_entities_by_type(EntityType.EXIT_DOOR)
        if not exit_doors:
            logger.warning(
                "calculate_level_surface_area: No exit doors found in level_data! "
                "Entity extraction may have failed."
            )
            return (float("inf"), float("inf"))
        exit_door = exit_doors[0]
        exit_x = exit_door.get("x", 0)
        exit_y = exit_door.get("y", 0)
        if exit_x == 0 and exit_y == 0:
            logger.warning(
                f"calculate_level_surface_area: Exit door position is (0, 0), "
                f"entity data may be incomplete. Exit door entity: {exit_door}"
            )
        exit_pos = (int(exit_x), int(exit_y))

        # Extract base_adjacency and spatial lookups for pathfinding
        # IMPORTANT: Use base_adjacency for surface area calculation to disregard locked doors
        # This gives the theoretical path distance as if all doors were open
        base_adjacency = (
            graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
        )
        spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
            graph_data
        )
        physics_cache = graph_data.get("node_physics") if graph_data else None

        # Calculate spawn→switch metrics (both geometric and physics cost)
        # Phase 2.1: Track physics cost for physics-weighted PBRS normalization
        try:
            # Geometric distance (pixels)
            spawn_to_switch_geo = calculate_geometric_path_distance(
                spawn_pos,
                switch_pos,
                base_adjacency,  # Use base_adjacency to ignore locked doors
                base_adjacency,
                physics_cache=physics_cache,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
                entity_radius=EXIT_SWITCH_RADIUS,
                ninja_radius=NINJA_RADIUS,
            )

            # Physics cost (difficulty-weighted)
            # Use path_calculator.get_distance() which handles cache building and fallback
            # Note: This call requires mine_proximity_cache and physics_cache to be built
            # If caches aren't ready yet, this may return inf - that's okay for initialization
            try:
                spawn_to_switch_physics = self.path_calculator.get_distance(
                    spawn_pos,
                    switch_pos,
                    base_adjacency,
                    base_adjacency,
                    level_data=level_data,
                    graph_data=graph_data,
                    entity_radius=EXIT_SWITCH_RADIUS,
                    ninja_radius=NINJA_RADIUS,
                )
            except (ValueError, RuntimeError) as e:
                # Caches not built yet (e.g., first call during initialization)
                # Fall back to inf - will be computed properly on next episode
                logger.error(
                    f"CRITICAL: Physics cost calculation failed for spawn→switch: {e}. "
                    f"This will cause all PBRS potentials to be 0! "
                    f"Caches may not be built properly."
                )
                spawn_to_switch_physics = float("inf")

        except Exception as e:
            logger.warning(
                f"Failed to calculate spawn→switch distances: {e}. Using inf."
            )
            spawn_to_switch_geo = float("inf")
            spawn_to_switch_physics = float("inf")

        # Calculate switch→exit metrics (both geometric and physics cost)
        try:
            # Geometric distance (pixels)
            switch_to_exit_geo = calculate_geometric_path_distance(
                switch_pos,
                exit_pos,
                base_adjacency,  # Use base_adjacency to ignore locked doors
                base_adjacency,
                physics_cache=physics_cache,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
                entity_radius=EXIT_DOOR_RADIUS,
                ninja_radius=NINJA_RADIUS,
            )

            # Physics cost (difficulty-weighted)
            # Use path_calculator.get_distance() which handles cache building and fallback
            try:
                switch_to_exit_physics = self.path_calculator.get_distance(
                    switch_pos,
                    exit_pos,
                    base_adjacency,
                    base_adjacency,
                    level_data=level_data,
                    graph_data=graph_data,
                    entity_radius=EXIT_DOOR_RADIUS,
                    ninja_radius=NINJA_RADIUS,
                )
            except (ValueError, RuntimeError) as e:
                # Caches not built yet (e.g., first call during initialization)
                logger.error(
                    f"CRITICAL: Physics cost calculation failed for switch→exit: {e}. "
                    f"This will cause all PBRS potentials to be 0! "
                    f"Caches may not be built properly."
                )
                switch_to_exit_physics = float("inf")
        except Exception as e:
            import traceback

            logger.error(
                f"Failed to calculate switch→exit distances: {e}\n{traceback.format_exc()}"
            )
            raise e

        # Combine distances and costs
        if spawn_to_switch_geo == float("inf") or switch_to_exit_geo == float("inf"):
            logger.warning(
                f"Combined path distance is infinite (unreachable goal). "
                f"spawn→switch: {spawn_to_switch_geo:.1f}, switch→exit: {switch_to_exit_geo:.1f}. "
                f"Positions: spawn={spawn_pos}, switch={switch_pos}, exit={exit_pos}"
            )
            return (float("inf"), float("inf"))

        if spawn_to_switch_physics == float("inf") or switch_to_exit_physics == float(
            "inf"
        ):
            logger.warning(
                f"Combined physics cost is infinite (unreachable goal). "
                f"spawn→switch: {spawn_to_switch_physics:.1f}, switch→exit: {switch_to_exit_physics:.1f}. "
                f"Positions: spawn={spawn_pos}, switch={switch_pos}, exit={exit_pos}"
            )
            return (float("inf"), float("inf"))

        combined_distance = spawn_to_switch_geo + switch_to_exit_geo
        combined_physics_cost = spawn_to_switch_physics + switch_to_exit_physics

        # DEBUG: Log combined path metrics calculation details
        # Calculate Euclidean distances for comparison
        spawn_to_switch_euclidean = (
            (switch_pos[0] - spawn_pos[0]) ** 2 + (switch_pos[1] - spawn_pos[1]) ** 2
        ) ** 0.5
        switch_to_exit_euclidean = (
            (exit_pos[0] - switch_pos[0]) ** 2 + (exit_pos[1] - switch_pos[1]) ** 2
        ) ** 0.5
        euclidean_combined = spawn_to_switch_euclidean + switch_to_exit_euclidean

        logger.info(
            f"[PBRS DEBUG] Combined path metrics calculation:\n"
            f"  Spawn position: {spawn_pos}\n"
            f"  Switch position: {switch_pos}\n"
            f"  Exit position: {exit_pos}\n"
            f"  Spawn→Switch geometric: {spawn_to_switch_geo:.1f}px, physics: {spawn_to_switch_physics:.1f}\n"
            f"  Switch→Exit geometric: {switch_to_exit_geo:.1f}px, physics: {switch_to_exit_physics:.1f}\n"
            f"  Combined geometric: {combined_distance:.1f}px\n"
            f"  Combined physics cost: {combined_physics_cost:.1f}\n"
            f"  Euclidean combined: {euclidean_combined:.1f}px\n"
            f"  Geo/Euclidean ratio: {combined_distance / euclidean_combined:.2f}x\n"
            f"  Physics/Geo ratio: {combined_physics_cost / max(combined_distance, 1.0):.2f}x"
        )

        # CRITICAL: Warn if combined distance is suspiciously low
        if combined_distance < 100 and euclidean_combined > 200:
            logger.error(
                f"[PBRS BUG] Combined path distance ({combined_distance:.1f}px) is "
                f"much smaller than Euclidean distance ({euclidean_combined:.1f}px)! "
                f"This indicates a pathfinding bug."
            )

        return (combined_distance, combined_physics_cost)

    def _get_cached_or_compute_potential(
        self,
        current_pos: Tuple[float, float],
        goal_id: str,
        state_with_metrics: Dict[str, Any],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        graph_data: Optional[Dict[str, Any]] = None,
        scale_factor: float = 1.0,
    ) -> float:
        """Get potential with position-based caching for small movements.

        OPTIMIZATION: When player moves < 6px from last cached position, use
        sub-node interpolation instead of full path distance recalculation.
        This provides dense per-step rewards without expensive pathfinding.

        The interpolation uses the cached next_hop direction to project
        small movements onto the optimal path, giving accurate PBRS gradients.

        Args:
            current_pos: Current player position (x, y)
            goal_id: "switch" or "exit"
            state_with_metrics: State dict with PBRS metrics
            adjacency: Graph adjacency structure
            level_data: Level data object
            graph_data: Graph data dict
            scale_factor: Normalization scale factor

        Returns:
            Objective distance potential [0, 1]
        """
        import time

        t_start = time.perf_counter()

        # Check if we can use cached value with interpolation
        use_cache = (
            self._last_player_pos is not None
            and self._last_goal_id == goal_id
            and self._cached_distance is not None
            and self._cached_start_node is not None
        )

        if use_cache:
            # Calculate squared distance moved (avoid sqrt for speed)
            dx = current_pos[0] - self._last_player_pos[0]
            dy = current_pos[1] - self._last_player_pos[1]
            dist_sq = dx * dx + dy * dy

            if dist_sq < self._position_cache_threshold_sq:
                # Small movement: use cached distance with sub-node interpolation
                # This provides dense per-step rewards without recalculating path

                if self._cached_next_hop is not None:
                    # Interpolate using next_hop direction
                    # Direction from cached node to next_hop is optimal path direction
                    start_node = self._cached_start_node
                    next_hop = self._cached_next_hop

                    # Convert to world coordinates
                    start_node_x = start_node[0] + NODE_WORLD_COORD_OFFSET
                    start_node_y = start_node[1] + NODE_WORLD_COORD_OFFSET
                    next_hop_x = next_hop[0] + NODE_WORLD_COORD_OFFSET
                    next_hop_y = next_hop[1] + NODE_WORLD_COORD_OFFSET

                    # Path direction (normalized)
                    path_dx = next_hop_x - start_node_x
                    path_dy = next_hop_y - start_node_y
                    path_len = (path_dx * path_dx + path_dy * path_dy) ** 0.5

                    if path_len > 0.001:
                        path_dir_x = path_dx / path_len
                        path_dir_y = path_dy / path_len

                        # Project current position onto path direction relative to cached node
                        player_offset_x = current_pos[0] - start_node_x
                        player_offset_y = current_pos[1] - start_node_y

                        # Projection: positive = moved toward next_hop = closer to goal
                        projection = (
                            player_offset_x * path_dir_x + player_offset_y * path_dir_y
                        )

                        # Calculate interpolated distance
                        # cached_distance is from cached_start_node to goal
                        # Subtract projection (positive projection = closer)
                        interpolated_distance = max(
                            0.0, self._cached_distance - projection
                        )

                        # Convert to potential using same normalization as full calculation
                        combined_path = state_with_metrics.get(
                            "_pbrs_combined_path_distance", 800.0
                        )
                        MIN_NORMALIZATION = 800.0
                        effective_norm = max(MIN_NORMALIZATION, combined_path)

                        normalized_distance = interpolated_distance / effective_norm
                        potential = max(0.0, min(1.0, 1.0 - normalized_distance))

                        # Store for diagnostics
                        state_with_metrics["_pbrs_normalized_distance"] = (
                            normalized_distance
                        )
                        state_with_metrics["_pbrs_cache_hit"] = True
                        self._cache_hits += 1
                        state_with_metrics["_pbrs_time_ms"] = (
                            time.perf_counter() - t_start
                        ) * 1000

                        return potential

                # No next_hop (at goal) - use cached potential directly
                # Small movements at goal don't need recalculation
                state_with_metrics["_pbrs_cache_hit"] = True
                self._cache_hits += 1
                state_with_metrics["_pbrs_time_ms"] = (
                    time.perf_counter() - t_start
                ) * 1000
                return (
                    self._cached_distance
                )  # This is actually cached potential in this case

        # Cache miss or large movement: compute full path distance
        self._cache_misses += 1
        t_potential = time.perf_counter()
        objective_pot = PBRSPotentials.objective_distance_potential(
            state_with_metrics,
            adjacency=adjacency,
            level_data=level_data,
            path_calculator=self.path_calculator,
            graph_data=graph_data,
            scale_factor=scale_factor,
        )
        t_after_potential = time.perf_counter()
        state_with_metrics["_pbrs_potential_calc_ms"] = (
            t_after_potential - t_potential
        ) * 1000

        # Update cache for next step
        self._last_player_pos = current_pos
        self._last_goal_id = goal_id

        # Cache the geometric distance and node info for interpolation
        # OPTIMIZATION: Reuse node info from path_calculator if available
        # This avoids redundant find_ninja_node calls (saves ~1.3ms per step)
        t_cache_update = time.perf_counter()
        if self.path_calculator is not None:
            # Check if path_calculator has cached the node from get_geometric_distance
            # Note: goal_id here is "switch" or "exit" but path_calculator uses
            # "exit_switch_0" or "exit_door_0" - check if types match
            cached_goal_id = self.path_calculator._last_goal_id or ""
            goal_type_matches = (
                goal_id == "switch" and "switch" in cached_goal_id
            ) or (goal_id == "exit" and "door" in cached_goal_id)
            if self.path_calculator._last_start_node is not None and goal_type_matches:
                # Reuse cached node info from path_calculator
                self._cached_start_node = self.path_calculator._last_start_node
                self._cached_distance = self.path_calculator._last_geometric_distance
                self._cached_next_hop = self.path_calculator._last_next_hop
                state_with_metrics["_pbrs_node_find_ms"] = 0.0  # No extra lookup needed
            elif self.path_calculator.level_cache is not None:
                # Fallback: find node manually (shouldn't normally happen)
                from ...graph.reachability.pathfinding_utils import (
                    find_ninja_node,
                    extract_spatial_lookups_from_graph_data,
                )

                spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
                    graph_data
                )

                # Find current node
                t_node_find = time.perf_counter()
                start_node = find_ninja_node(
                    (int(current_pos[0]), int(current_pos[1])),
                    adjacency,
                    spatial_hash=spatial_hash,
                    subcell_lookup=subcell_lookup,
                    ninja_radius=PLAYER_RADIUS,
                )
                state_with_metrics["_pbrs_node_find_ms"] = (
                    time.perf_counter() - t_node_find
                ) * 1000

                if start_node is not None:
                    self._cached_start_node = start_node

                    # Get cached distance from level cache
                    goal_pos = (
                        self.path_calculator.level_cache._goal_id_to_goal_pos.get(
                            goal_id
                        )
                    )
                    if goal_pos is not None:
                        cached_dist = (
                            self.path_calculator.level_cache.get_geometric_distance(
                                start_node, goal_pos, goal_id
                            )
                        )
                        if cached_dist != float("inf"):
                            self._cached_distance = cached_dist

                    # Cache next_hop for interpolation
                    self._cached_next_hop = (
                        self.path_calculator.level_cache.get_next_hop(
                            start_node, goal_id
                        )
                    )

        state_with_metrics["_pbrs_cache_update_ms"] = (
            time.perf_counter() - t_cache_update
        ) * 1000

        state_with_metrics["_pbrs_cache_hit"] = False
        state_with_metrics["_pbrs_time_ms"] = (time.perf_counter() - t_start) * 1000
        return objective_pot

    def _get_cached_or_compute_potential_with_waypoints(
        self,
        current_pos: Tuple[float, float],
        goal_id: str,
        state_with_metrics: Dict[str, Any],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        graph_data: Optional[Dict[str, Any]] = None,
        scale_factor: float = 1.0,
    ) -> float:
        """Get potential with waypoint routing and position-based caching.

        Similar to _get_cached_or_compute_potential but uses waypoint-aware potential
        function when momentum waypoints are available.

        Args:
            current_pos: Current player position (x, y)
            goal_id: "switch" or "exit"
            state_with_metrics: State dict with PBRS metrics
            adjacency: Graph adjacency structure
            level_data: Level data object
            graph_data: Graph data dict
            scale_factor: Normalization scale factor

        Returns:
            Objective distance potential [0, 1] with waypoint routing
        """
        import time

        t_start = time.perf_counter()

        # For waypoint routing, we skip position caching since waypoint selection
        # depends on velocity and may change even with small position changes
        # This is acceptable since waypoints are only used on specific levels

        t_potential = time.perf_counter()
        objective_pot = PBRSPotentials.objective_distance_potential_with_waypoints(
            state_with_metrics,
            adjacency=adjacency,
            level_data=level_data,
            path_calculator=self.path_calculator,
            momentum_waypoints=self.momentum_waypoints,
            graph_data=graph_data,
            scale_factor=scale_factor,
        )
        t_after_potential = time.perf_counter()
        state_with_metrics["_pbrs_potential_calc_ms"] = (
            t_after_potential - t_potential
        ) * 1000

        # Update cache for next step (simplified for waypoint case)
        self._last_player_pos = current_pos
        self._last_goal_id = goal_id

        state_with_metrics["_pbrs_cache_hit"] = False
        state_with_metrics["_pbrs_time_ms"] = (time.perf_counter() - t_start) * 1000
        return objective_pot

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
            self._last_goal_id = None
            self._cached_distance = None
            self._cached_start_node = None
            self._cached_next_hop = None

        # Compute and cache combined path metrics for path-based normalization
        # Cache key includes level_id to invalidate on level change
        # Phase 2.1: Cache both geometric distance and physics cost
        path_cache_key = f"{level_id}_{switch_states_signature}"
        if (
            self._path_distance_cache_key != path_cache_key
            or self._cached_combined_path_distance is None
            or self._cached_combined_physics_cost is None
        ):
            (
                self._cached_combined_path_distance,
                self._cached_combined_physics_cost,
            ) = self._compute_combined_path_distance(adjacency, level_data, graph_data)
            self._path_distance_cache_key = path_cache_key
            logger.debug(
                f"Cached combined path metrics for level {level_id}: "
                f"geometric={self._cached_combined_path_distance:.1f}px, "
                f"physics_cost={self._cached_combined_physics_cost:.1f}"
            )

        # Add metrics to state for potential calculation
        state_with_metrics = dict(state)
        state_with_metrics["_pbrs_surface_area"] = self._cached_surface_area
        state_with_metrics["_pbrs_combined_path_distance"] = (
            self._cached_combined_path_distance
        )
        state_with_metrics["_pbrs_combined_physics_cost"] = (
            self._cached_combined_physics_cost
        )

        # OPTIMIZATION: Skip full path recalculation if player hasn't moved much
        # This uses cached node + sub-node interpolation for dense rewards
        current_pos = (float(state["player_x"]), float(state["player_y"]))
        current_goal_id = (
            "switch" if not state.get("switch_activated", False) else "exit"
        )

        # Use waypoint-aware potential if waypoints available
        if self.momentum_waypoints:
            objective_pot = self._get_cached_or_compute_potential_with_waypoints(
                current_pos=current_pos,
                goal_id=current_goal_id,
                state_with_metrics=state_with_metrics,
                adjacency=adjacency,
                level_data=level_data,
                graph_data=graph_data,
                scale_factor=scale_factor,
            )
        # Fall back to momentum-aware geometric pathfinding
        else:
            objective_pot = self._get_cached_or_compute_potential(
                current_pos=current_pos,
                goal_id=current_goal_id,
                state_with_metrics=state_with_metrics,
                adjacency=adjacency,
                level_data=level_data,
                graph_data=graph_data,
                scale_factor=scale_factor,
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

        # Copy timing data for profiling
        state["_pbrs_cache_hit"] = state_with_metrics.get("_pbrs_cache_hit", False)
        state["_pbrs_time_ms"] = state_with_metrics.get("_pbrs_time_ms", 0.0)
        if "_pbrs_potential_calc_ms" in state_with_metrics:
            state["_pbrs_potential_calc_ms"] = state_with_metrics[
                "_pbrs_potential_calc_ms"
            ]
        if "_pbrs_node_find_ms" in state_with_metrics:
            state["_pbrs_node_find_ms"] = state_with_metrics["_pbrs_node_find_ms"]
        if "_pbrs_cache_update_ms" in state_with_metrics:
            state["_pbrs_cache_update_ms"] = state_with_metrics["_pbrs_cache_update_ms"]

        # === EMERGENCY PBRS DIAGNOSTIC LOGGING ===
        # Log potential calculation for debugging zero-gradient bug
        if level_id is not None:
            logger.debug(
                f"[PBRS_EMERGENCY] level={level_id}, "
                f"objective_pot={objective_pot:.4f}, "
                f"combined_physics={state_with_metrics.get('_pbrs_combined_physics_cost', 'MISSING')}, "
                f"combined_path={state_with_metrics.get('_pbrs_combined_path_distance', 'MISSING')}, "
                f"switch_activated={state.get('switch_activated', False)}"
            )

        # === VELOCITY ALIGNMENT WITH NEXT-HOP GRADIENT ===
        # Uses next_hop direction (respects winding paths) instead of Euclidean direction.
        # This provides continuous directional guidance even when optimal path goes away
        # from the goal first.
        velocity_alignment = 0.0
        path_gradient = None

        # Get curriculum-adaptive velocity alignment weight (Phase 1.2)
        # Uses reward_config if available, otherwise falls back to static constant
        if self.reward_config is not None:
            velocity_weight = self.reward_config.velocity_alignment_weight
        else:
            # Fallback to static constant for backward compatibility
            from .reward_constants import (
                VELOCITY_ALIGNMENT_WEIGHT as STATIC_VELOCITY_WEIGHT,
            )

            velocity_weight = STATIC_VELOCITY_WEIGHT

        # Only compute if we have a level cache with next_hop data and weight > 0
        if (
            self.path_calculator is not None
            and self.path_calculator.level_cache is not None
            and velocity_weight > 0
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
        state["_pbrs_velocity_weight"] = velocity_weight
        state["_pbrs_path_gradient"] = path_gradient

        # Apply phase-specific scaling (switch vs exit) and curriculum weight
        if not state.get("switch_activated", False):
            potential = PBRS_SWITCH_DISTANCE_SCALE * objective_pot * objective_weight
        else:
            potential = PBRS_EXIT_DISTANCE_SCALE * objective_pot * objective_weight

        # Add velocity alignment bonus (scaled by curriculum-adaptive weight)
        # This is additive, not multiplicative, to preserve PBRS structure
        potential += velocity_weight * velocity_alignment

        return max(0.0, potential)

    def set_momentum_waypoints(
        self, waypoints: Optional[List[Any]], waypoint_source: str = "unknown"
    ) -> None:
        """Set momentum waypoints for current level.

        Called when level changes or waypoints are loaded from cache.

        Args:
            waypoints: List of MomentumWaypoint or PathWaypoint objects, or None to disable
            waypoint_source: Source of waypoints ("path", "demo", "adaptive") for logging
        """
        self.momentum_waypoints = waypoints or []
        self.waypoint_source = waypoint_source

        # Pass waypoints to path calculator for precomputed caching
        if self.path_calculator is not None:
            logger.debug(
                f"[PBRS] Set {len(self.momentum_waypoints)} {waypoint_source} waypoints "
                f"for multi-stage routing"
            )
            self.path_calculator.set_waypoints(waypoints)
        else:
            logger.warning("[PBRS] path_calculator is None, cannot set waypoints")

        logger.debug(
            f"Set {len(self.momentum_waypoints)} momentum waypoints for PBRS "
            f"(source: {waypoint_source})"
        )

    def reset(self) -> None:
        """Reset calculator state for new episode.

        Clears position-based caching to ensure fresh potential calculations
        for the new episode. Surface area and combined path distance caches
        are kept as they are level-specific, not episode-specific.

        Momentum waypoints are NOT reset here - they are level-specific and
        should persist across episodes on the same level.
        """
        # Clear position-based cache for fresh potential calculations
        # CRITICAL: Must reset when episode restarts (including same level reload)
        self._last_player_pos = None
        self._last_goal_id = None
        self._cached_distance = None
        self._cached_start_node = None
        self._cached_next_hop = None

        # Keep surface area cache - it's per level, not per episode
        # Keep combined path distance cache - it's per level, not per episode
        # Keep momentum waypoints - they're per level, not per episode
        # These will be invalidated when level_data or switch_states change
