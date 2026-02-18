"""
Cost functions for physics-aware pathfinding.

Edge weight computation, aerial chain multipliers, mine proximity costs,
and grounding utilities used by graph building and pathfinding algorithms.
"""

import logging
from typing import Dict, Tuple, List, Optional, Any

# Lazy import to break circular dependency chain:
# cost_functions -> reward_constants -> gym_environment -> path_distance_calculator -> cost_functions
_MINE_HAZARD_COST_MULTIPLIER = None


def _get_mine_hazard_cost_multiplier():
    """Lazy loader for MINE_HAZARD_COST_MULTIPLIER to break circular imports."""
    global _MINE_HAZARD_COST_MULTIPLIER
    if _MINE_HAZARD_COST_MULTIPLIER is None:
        from ...gym_environment.reward_calculation.reward_constants import (
            MINE_HAZARD_COST_MULTIPLIER,
        )
        _MINE_HAZARD_COST_MULTIPLIER = MINE_HAZARD_COST_MULTIPLIER
    return _MINE_HAZARD_COST_MULTIPLIER


_logger = logging.getLogger(__name__)

# Node coordinate offset for world coordinate conversion
# Nodes are in tile data space, entity positions in world space differ by 24px
NODE_WORLD_COORD_OFFSET = 24

# Constants for horizontal edge validation and physics
SUB_NODE_SIZE = 12  # Sub-node spacing in pixels

# Aerial upward movement cost constants
# N++ physics: A jump from ground with held jump button can realistically cover
# ~5-6 sub-nodes (60-72px) of upward travel before gravity overcomes jump velocity.
AERIAL_UPWARD_CHAIN_THRESHOLD = (
    2  # Max chain before applying blocking cost (allows 6 moves: chain 0-5)
)
AERIAL_UPWARD_BLOCKING_COST = 500.0  # High cost for moves beyond threshold
AERIAL_UPWARD_BASE_MULTIPLIER = 3.0  # Base multiplier for aerial upward (not 1.0)

# Momentum-aware pathfinding cost constants
MOMENTUM_CONTINUE_MULTIPLIER = 0.7  # Cheaper to maintain momentum (30% discount)
MOMENTUM_REVERSE_MULTIPLIER = 2.5  # Expensive to reverse direction (2.5x penalty)
MOMENTUM_BUILDING_THRESHOLD = (
    12  # Min horizontal displacement (1 sub-node) to have momentum
)

# Velocity-aware first-edge cost constants
NINJA_RISING_FAST_THRESHOLD = (
    -2.0
)  # Below this yspeed = strong upward momentum (continuing jump)
NINJA_RISING_SLOW_THRESHOLD = (
    0.0  # Below this yspeed = some upward momentum (jump ending)
)
NINJA_HORIZONTAL_MOMENTUM_THRESHOLD = (
    2.0  # ~60% of MAX_HOR_SPEED (3.333) for significant momentum
)

# Cost multipliers for first edge based on ninja's actual velocity
VELOCITY_UPWARD_FAST_MULTIPLIER = 0.3  # Continuing strong jump (high upward velocity)
VELOCITY_UPWARD_SLOW_MULTIPLIER = 0.7  # Jump ending (low upward velocity)
VELOCITY_HORIZONTAL_CONTINUE_MULTIPLIER = 0.5  # Moving with momentum (same direction)
VELOCITY_HORIZONTAL_REVERSE_SLOW_MULTIPLIER = 1.5  # Slight direction change (low speed)
VELOCITY_HORIZONTAL_REVERSE_FAST_MULTIPLIER = 2.5  # Hard brake and reverse (high speed)

# Cost cap for impossible edges when allow_high_cost_edges is enabled
IMPOSSIBLE_EDGE_COST_CAP = 10000.0


def _get_aerial_chain_multiplier(chain_count: int) -> float:
    """Multiplicative cost scaling for consecutive aerial upward moves.

    Models N++ jump physics where upward movement in air is valid while
    continuing a jump trajectory (holding jump button), but becomes progressively
    expensive to reflect diminishing upward velocity as gravity overcomes jump force.

    Chain 0-5: Progressive cost scaling (within realistic jump physics)
        Chain 0: 3x, Chain 1: 9x, Chain 2: 27x, Chain 3: 81x, Chain 4: 243x, Chain 5: 729x
    Chain 6+: Beyond physics limits - apply blocking cost (100x base with continued scaling)
        Chain 6: 300x, Chain 7: 900x, etc.
    """
    if chain_count <= AERIAL_UPWARD_CHAIN_THRESHOLD:
        return AERIAL_UPWARD_BASE_MULTIPLIER ** (chain_count + 1)
    else:
        excess_chain = chain_count - AERIAL_UPWARD_CHAIN_THRESHOLD
        return AERIAL_UPWARD_BLOCKING_COST * (
            AERIAL_UPWARD_BASE_MULTIPLIER**excess_chain
        )


def _calculate_mine_proximity_cost(
    pos: Tuple[int, int],
    level_data: Optional[Any],
    mine_proximity_cache: Optional[Any] = None,
    hazard_cost_multiplier: Optional[float] = None,
) -> float:
    """Calculate cost multiplier based on proximity to deadly mines.

    Uses cached values when available for O(1) lookup. Returns 1.0 (no cost)
    if cache is not provided.
    """
    if mine_proximity_cache is None:
        return 1.0

    MINE_HAZARD_COST_MULTIPLIER = _get_mine_hazard_cost_multiplier()

    if hazard_cost_multiplier is not None and hazard_cost_multiplier <= 1.0:
        return 1.0
    elif hazard_cost_multiplier is None and MINE_HAZARD_COST_MULTIPLIER <= 1.0:
        return 1.0

    cached_multiplier = mine_proximity_cache.get_cost_multiplier(pos)

    if (
        hazard_cost_multiplier is None
        or hazard_cost_multiplier == MINE_HAZARD_COST_MULTIPLIER
    ):
        return cached_multiplier

    if MINE_HAZARD_COST_MULTIPLIER > 1.0:
        scale = (hazard_cost_multiplier - 1.0) / (MINE_HAZARD_COST_MULTIPLIER - 1.0)
        return 1.0 + (cached_multiplier - 1.0) * scale
    return cached_multiplier


def _calculate_physics_aware_cost(
    src_pos: Tuple[int, int],
    dst_pos: Tuple[int, int],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    parent_pos: Optional[Tuple[int, int]] = None,
    physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
    level_data: Optional[Any] = None,
    mine_proximity_cache: Optional[Any] = None,
    aerial_chain: int = 0,
    grandparent_pos: Optional[Tuple[int, int]] = None,
    mine_sdf: Optional[Any] = None,
    hazard_cost_multiplier: Optional[float] = None,
    ninja_velocity: Optional[Tuple[float, float]] = None,
    jump_duration_remaining: Optional[float] = None,
) -> float:
    """
    Calculate edge cost based on N++ movement physics, momentum, and mine hazard proximity.

    Mine-aware physics: When physics_cache includes mine_grounded/mine_walled (from
    _compute_mine_aware_physics in graph_builder), these are actively enforced:
    - mine_grounded: node appears grounded only due to mine blocking below -> treat as airborne
    - mine_walled: node appears walled only due to mine -> cannot walljump (treat as non-walled)
    """
    if physics_cache is None:
        raise ValueError(
            "physics_cache is REQUIRED for _calculate_physics_aware_cost(). "
            "This function is called by all PBRS pathfinding - physics cache must be built. "
            "Ensure graph building includes physics cache precomputation."
        )

    src_x, src_y = src_pos
    dst_x, dst_y = dst_pos
    dx = dst_x - src_x
    dy = dst_y - src_y

    src_physics = physics_cache.get(src_pos, {"grounded": True, "walled": False})
    dst_physics = physics_cache.get(dst_pos, {"grounded": True, "walled": False})
    src_grounded = src_physics["grounded"]
    dst_grounded = dst_physics["grounded"]
    src_walled = src_physics["walled"]

    # Mine-aware flags (set by _compute_mine_aware_physics if available)
    src_mine_grounded = src_physics.get("mine_grounded", False)
    src_mine_walled = src_physics.get("mine_walled", False)

    # ACTIVE ENFORCEMENT: mine_grounded -> treat as airborne (cannot walk on mines)
    if src_mine_grounded:
        src_grounded = False

    # ACTIVE ENFORCEMENT: mine_walled -> cannot walljump off mines
    if src_mine_walled:
        src_walled = False

    x_direction_change = False
    y_direction_change_to_rising = False
    from_horizontal_air_movement = False
    parent_dx = 0
    parent_dy = 0

    if parent_pos is not None:
        parent_x, parent_y = parent_pos
        parent_dx = src_x - parent_x
        parent_dy = src_y - parent_y

        if parent_dx != 0 and dx != 0:
            x_direction_change = (parent_dx > 0) != (dx > 0)

        if parent_dy >= 0 and dy < 0:
            y_direction_change_to_rising = True

        if parent_dy == 0:
            if not src_grounded:
                from_horizontal_air_movement = True

    # Base geometric cost (Euclidean distance)
    if dx != 0 and dy != 0:
        base_cost = 1.414  # sqrt(2) for diagonal
    else:
        base_cost = 1.0  # Unit cost for cardinal directions

    # Velocity-aware first-edge cost adjustment
    velocity_multiplier = 1.0
    if parent_pos is None and ninja_velocity is not None:
        ninja_xspeed, ninja_yspeed = ninja_velocity

        if dy < 0:  # Moving upward (negative y in screen coords)
            if not src_grounded and not src_walled:
                if ninja_yspeed >= NINJA_RISING_SLOW_THRESHOLD:
                    velocity_multiplier = float("inf")
                elif ninja_yspeed < NINJA_RISING_FAST_THRESHOLD:
                    velocity_multiplier = VELOCITY_UPWARD_FAST_MULTIPLIER
                elif (
                    jump_duration_remaining is not None
                    and jump_duration_remaining <= 0.0
                ):
                    velocity_multiplier = float("inf")
                else:
                    velocity_multiplier = VELOCITY_UPWARD_SLOW_MULTIPLIER
        elif dy == 0 and dx != 0:  # Pure horizontal movement
            ninja_has_momentum = abs(ninja_xspeed) > NINJA_HORIZONTAL_MOMENTUM_THRESHOLD
            same_direction = (ninja_xspeed * dx) > 0

            if ninja_has_momentum:
                if same_direction:
                    velocity_multiplier = VELOCITY_HORIZONTAL_CONTINUE_MULTIPLIER
                else:
                    velocity_multiplier = VELOCITY_HORIZONTAL_REVERSE_FAST_MULTIPLIER
            elif not same_direction and abs(ninja_xspeed) > 0.5:
                velocity_multiplier = VELOCITY_HORIZONTAL_REVERSE_SLOW_MULTIPLIER

    # Physics multipliers based on movement type (screen coords: y increases downward)
    if dx != 0 and dy > 0 and src_grounded:
        multiplier = 0.2
    elif dx != 0 and dy > 0 and not src_grounded and not src_walled:
        multiplier = 0.15
    elif dx != 0 and dy < 0 and not src_grounded and src_walled:
        if from_horizontal_air_movement or y_direction_change_to_rising:
            multiplier = float("inf")
        elif x_direction_change:
            multiplier = 1.5
        else:
            multiplier = 1.0
    elif dx != 0 and dy < 0 and src_grounded:
        multiplier = 0.2
    elif dx != 0 and dy < 0 and not src_grounded and not src_walled:
        if from_horizontal_air_movement:
            multiplier = float("inf")
        elif aerial_chain > 0 and parent_pos is not None and parent_dy >= 0:
            multiplier = float("inf")
        elif y_direction_change_to_rising:
            multiplier = float("inf")
        elif x_direction_change:
            # Cannot reverse horizontal direction while rising in air - N++ has limited air control
            multiplier = float("inf")
        else:
            multiplier = _get_aerial_chain_multiplier(aerial_chain)
    elif dy < 0:  # Moving up (against gravity) - vertical only
        if src_grounded:
            multiplier = 0.7
        elif src_walled:
            if from_horizontal_air_movement or y_direction_change_to_rising:
                multiplier = float("inf")
            else:
                multiplier = 1.2
        elif from_horizontal_air_movement:
            multiplier = float("inf")
        elif aerial_chain > 0 and parent_pos is not None and parent_dy >= 0:
            multiplier = float("inf")
        elif y_direction_change_to_rising:
            multiplier = float("inf")
        else:
            multiplier = _get_aerial_chain_multiplier(aerial_chain)
    elif dy > 0:  # Moving down (with gravity)
        multiplier = 0.5
    else:  # Horizontal (dy == 0)
        if src_grounded and dst_grounded:
            multiplier = 0.15
        elif x_direction_change and not src_grounded and not src_walled:
            # Flat penalty for direction change in air (N++ limited air control)
            multiplier = 1.5
        elif not src_grounded and not src_walled:
            # Flat cost for horizontal air drift
            multiplier = 0.5
        else:
            multiplier = 1.0

    # Momentum-aware cost adjustment for grounded horizontal movement
    momentum_multiplier = 1.0
    if (
        dy == 0
        and src_grounded
        and dst_grounded
        and grandparent_pos is not None
        and parent_pos is not None
    ):
        grandparent_x, grandparent_y = grandparent_pos
        recent_dx = src_x - parent_x
        prev_dx = parent_x - grandparent_x

        if recent_dx * prev_dx > 0 and abs(recent_dx) >= 12:
            momentum_direction = -1 if recent_dx < 0 else 1

            if dx != 0:
                continues_momentum = (momentum_direction * dx) > 0
                if continues_momentum:
                    momentum_multiplier = 0.7  # MOMENTUM_CONTINUE_MULTIPLIER
                else:
                    momentum_multiplier = 2.5  # MOMENTUM_REVERSE_MULTIPLIER

    # Mine hazard proximity cost multiplier
    mine_multiplier = _calculate_mine_proximity_cost(
        dst_pos, level_data, mine_proximity_cache, hazard_cost_multiplier
    )

    # Velocity-aware mine proximity cost using SDF
    if mine_sdf is not None and parent_pos is not None:
        velocity_mag_sq = dx * dx + dy * dy

        if velocity_mag_sq > 0.25:
            grad_x, grad_y = mine_sdf.get_gradient_at_position(dst_x, dst_y)
            grad_mag_sq = grad_x * grad_x + grad_y * grad_y
            if grad_mag_sq > 0.0001:
                velocity_mag = velocity_mag_sq**0.5
                vel_dir_x = dx / velocity_mag
                vel_dir_y = dy / velocity_mag
                toward_mine = -(vel_dir_x * grad_x + vel_dir_y * grad_y)

                if toward_mine > 0.3:
                    velocity_factor = 1.0 + toward_mine * velocity_mag * 0.16667
                    mine_multiplier *= velocity_factor

    return (
        base_cost
        * multiplier
        * momentum_multiplier
        * mine_multiplier
        * velocity_multiplier
    )


def _is_node_grounded_util(
    node_pos: Tuple[int, int],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
) -> bool:
    """
    Check if a node is grounded (has solid surface below).

    Uses base_adjacency (pre-entity-mask) to determine actual level geometry.
    """
    x, y = node_pos
    below_pos = (x, y + SUB_NODE_SIZE)

    if below_pos not in base_adjacency:
        return True

    if node_pos in base_adjacency:
        for neighbor_pos, _ in base_adjacency[node_pos]:
            if neighbor_pos == below_pos:
                return False

    return True
