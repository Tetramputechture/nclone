"""
Utility functions for pathfinding operations.

Shared utilities for node finding, BFS operations, and path reconstruction
used by both performance-critical pathfinding and visualization systems.
"""

import logging
import heapq
from typing import Dict, Tuple, List, Optional, Any
from collections import deque, OrderedDict

# Node coordinate offset for world coordinate conversion
# Nodes are in tile data space, entity positions in world space differ by 24px
NODE_WORLD_COORD_OFFSET = 24

# Logger for pathfinding utilities
_logger = logging.getLogger(__name__)

# Cache for subcell lookup loader singleton (lazy-loaded)
_subcell_lookup_loader_cache = None

# Module-level cache for surface area by level ID
# Uses OrderedDict for LRU eviction to prevent unbounded growth
_surface_area_cache: OrderedDict[str, float] = OrderedDict()
_SURFACE_AREA_CACHE_MAX_SIZE = 1000  # Limit to 1000 levels to prevent memory growth

# Constants for horizontal edge validation and physics
SUB_NODE_SIZE = 12  # Sub-node spacing in pixels

# Aerial upward movement cost constants
# N++ physics: A jump from ground can realistically cover ~2-3 sub-nodes (24-36px)
# of upward travel before gravity and momentum make further upward movement
# increasingly difficult. We use a tight threshold to prevent impossible paths.
AERIAL_UPWARD_CHAIN_THRESHOLD = 2  # Max chain before applying blocking cost
AERIAL_UPWARD_BLOCKING_COST = 100.0  # High cost for moves beyond threshold
AERIAL_UPWARD_BASE_MULTIPLIER = 3.0  # Base multiplier for aerial upward (not 1.0)

# Momentum-aware pathfinding cost constants
# These multipliers adjust edge costs based on whether movement continues or reverses momentum
MOMENTUM_CONTINUE_MULTIPLIER = 0.7  # Cheaper to maintain momentum (30% discount)
MOMENTUM_REVERSE_MULTIPLIER = 2.5  # Expensive to reverse direction (2.5x penalty)
MOMENTUM_BUILDING_THRESHOLD = (
    12  # Min horizontal displacement (1 sub-node) to have momentum
)


def _get_aerial_chain_multiplier(chain_count: int) -> float:
    """Multiplicative cost scaling for consecutive aerial upward moves.

    Models N++ jump physics where upward movement in air is valid while
    continuing a jump trajectory, but becomes quickly expensive to prevent
    impossible long-distance aerial paths.

    Chain 0: First aerial upward - 3x (immediate penalty for leaving ground/wall)
    Chain 1: Second aerial upward - 9x (3^2)
    Chain 2: Third aerial upward - 27x (3^3)
    Chain 3+: Beyond physics limits - apply blocking cost (100x base)

    NOTE: This is the canonical implementation shared across all pathfinding code.
    Imported by pathfinding_algorithms.py to avoid duplication.

    Args:
        chain_count: Number of consecutive aerial upward moves

    Returns:
        Cost multiplier reflecting physics feasibility
    """
    if chain_count <= AERIAL_UPWARD_CHAIN_THRESHOLD:
        # Within jump physics limits - exponential increase with base 3
        # Chain 0: 3, Chain 1: 9, Chain 2: 27
        return AERIAL_UPWARD_BASE_MULTIPLIER ** (chain_count + 1)
    else:
        # Beyond physics limits - apply blocking cost with continued scaling
        # Chain 3: 100 * 3 = 300, Chain 4: 100 * 9 = 900, etc.
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
    if cache is not provided (for geometric distance calculations that don't need hazard costs).

    NOTE: This is the canonical implementation shared across all pathfinding code.
    Imported by pathfinding_algorithms.py to avoid duplication.

    OPTIMIZATION: Heavily optimized hot path (called 4M times in profile).

    Args:
        pos: Node position (x, y) in pixels
        level_data: LevelData instance containing mine entities (optional)
        mine_proximity_cache: MineProximityCostCache instance (optional)
        hazard_cost_multiplier: Optional override for curriculum-adaptive cost multiplier
                                If None, uses static constant from reward_constants

    Returns:
        float: Cost multiplier in range [1.0, hazard_cost_multiplier]
               1.0 if far from mines, no cache, or no penalty applies
               Higher values when close to deadly mines
    """
    # If no cache provided, return neutral multiplier (1.0 = no cost adjustment)
    # This is acceptable for geometric distance calculations that don't need hazard costs
    if mine_proximity_cache is None:
        return 1.0

    # OPTIMIZATION: Import once at module level would be better, but keep here for safety
    from ...gym_environment.reward_calculation.reward_constants import (
        MINE_HAZARD_COST_MULTIPLIER,
    )

    # OPTIMIZATION: Early exit if hazard avoidance is disabled (check before cache lookup)
    # Most common case in early training phases
    if hazard_cost_multiplier is not None and hazard_cost_multiplier <= 1.0:
        return 1.0
    elif hazard_cost_multiplier is None and MINE_HAZARD_COST_MULTIPLIER <= 1.0:
        return 1.0

    # Use cache (O(1) lookup)
    cached_multiplier = mine_proximity_cache.get_cost_multiplier(pos)

    # OPTIMIZATION: Skip scaling if multiplier is default (common case - fast path)
    if (
        hazard_cost_multiplier is None
        or hazard_cost_multiplier == MINE_HAZARD_COST_MULTIPLIER
    ):
        return cached_multiplier

    # Adjust cached result for curriculum-adaptive multiplier
    # cached_multiplier = 1.0 + (proximity_factor^2) * (STATIC_MULTIPLIER - 1.0)
    # We want: 1.0 + (proximity_factor^2) * (effective_multiplier - 1.0)
    # So: scale = (effective_multiplier - 1.0) / (STATIC_MULTIPLIER - 1.0)
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
    aerial_upward_chain: int = 0,
    grandparent_pos: Optional[Tuple[int, int]] = None,
    mine_sdf: Optional[Any] = None,
    hazard_cost_multiplier: Optional[float] = None,
) -> float:
    """
    Calculate edge cost based on N++ movement physics, momentum, and mine hazard proximity.

    Movement costs reflect actual N++ physics (from sim_mechanics_doc.md):
    - Grounded horizontal: FAST (ground accel 0.0667, max speed 3.333)
    - Air horizontal: SLOW (air accel 0.0444, ~33% slower)
    - Vertical up: EXPENSIVE (requires jump, fights gravity 0.0667)
    - Vertical down: CHEAP (gravity assists 0.0667)
    - Diagonal movement: Combines horizontal and vertical physics
    - Momentum preservation: Continuing in same X direction is cheaper than changing

    Momentum-aware cost adjustments:
    - Tracks horizontal momentum from trajectory (requires grandparent_pos)
    - Paths that preserve momentum get 30% discount (0.7x multiplier)
    - Paths that reverse momentum get 2.5x penalty
    - Makes momentum-building "detours" cheaper than naive direct paths
    - Critical for levels requiring backtracking to build speed (e.g., long jumps)

    Mid-air jump prevention:
    - Upward movement while airborne (not grounded, not walled) is blocked with very high cost
    - Consecutive aerial upward moves accumulate multiplicative cost penalties
    - This reflects N++ physics where mid-air jumps are impossible

    Hazard avoidance:
    - Paths near deadly mines incur additional cost multiplier
    - Velocity-aware costs (NEW): Paths approaching mines at high speed get extra penalty
    - Uses SDF to detect movement toward mines (O(1) lookup)
    - Makes PBRS naturally guide agent along safer paths
    - Preserves policy invariance (optimal policy still reaches goal)

    Cost multipliers are tuned so A* naturally prefers physically efficient paths
    (e.g., running on ground > air movement, falling > jumping).

    NOTE: This is the canonical implementation shared across all pathfinding code.
    Imported by pathfinding_algorithms.py to avoid duplication.

    Args:
        src_pos: Source node position (x, y) in pixels
        dst_pos: Destination node position (x, y) in pixels
        base_adjacency: Base graph adjacency for grounding/wall checks (pre-entity-mask)
        parent_pos: Optional parent position for momentum/direction checking
        physics_cache: Optional pre-computed physics properties for O(1) lookups
        level_data: Optional LevelData for mine proximity checks (fallback)
        mine_proximity_cache: Optional MineProximityCostCache for O(1) mine cost lookup
        aerial_upward_chain: Number of consecutive aerial upward moves in current path
        grandparent_pos: Optional grandparent position for momentum inference
        mine_sdf: Optional MineSignedDistanceField for velocity-aware hazard costs
        hazard_cost_multiplier: Optional curriculum-adaptive mine hazard cost multiplier

    Returns:
        Edge cost (float) where 1.0 = baseline movement cost
    """
    # OPTIMIZATION: Extract coordinates once to avoid repeated tuple indexing (4M calls!)
    src_x, src_y = src_pos
    dst_x, dst_y = dst_pos
    dx = dst_x - src_x
    dy = dst_y - src_y

    # OPTIMIZATION: Direct dict access with local caching (avoid 19.7M dict.get() calls)
    # Extract physics properties once and cache in local variables
    src_physics = physics_cache[src_pos]
    dst_physics = physics_cache[dst_pos]
    src_grounded = src_physics["grounded"]
    dst_grounded = dst_physics["grounded"]
    src_walled = src_physics["walled"]

    # OPTIMIZATION: Cache parent coordinates to avoid repeated tuple indexing
    x_direction_change = False
    y_direction_change_to_rising = False
    parent_dx = 0
    parent_dy = 0

    if parent_pos is not None:
        # Extract parent coordinates once
        parent_x, parent_y = parent_pos
        parent_dx = src_x - parent_x
        parent_dy = src_y - parent_y

        # Check if X direction is changing (for momentum considerations)
        # X direction changed if signs differ and both are non-zero
        if parent_dx != 0 and dx != 0:
            x_direction_change = (parent_dx > 0) != (dx > 0)

        # Check if Y direction is changing from falling to rising (physically impossible in air)
        # Was falling or stationary (parent_dy >= 0) and now moving up (dy < 0)
        if parent_dy >= 0 and dy < 0:
            y_direction_change_to_rising = True

    # Base geometric cost (Euclidean distance)
    if dx != 0 and dy != 0:
        base_cost = 1.414  # sqrt(2) for diagonal
    else:
        base_cost = 1.0  # Unit cost for cardinal directions

    # Physics multipliers based on movement type
    # Screen coordinates: y increases downward

    # Special case: Diagonal falling from air (prefer over horizontal air movement)
    if dx != 0 and dy > 0 and not src_grounded and not src_walled:
        # Diagonal downward continuing same direction - efficient (gravity + momentum)
        multiplier = 0.4
    # Special case: Diagonal upward wall-assisted move (valid N++ mechanic)
    elif dx != 0 and dy < 0 and not src_grounded and src_walled:
        # Wall-assisted diagonal upward (wall-jump) - valid but more expensive than ground
        if x_direction_change:
            # Changing X direction while wall jumping is expensive
            # Cost: 1.414 × 1.5 = 2.12
            multiplier = 1.5
        else:
            # Continuing same X direction - more efficient wall jump
            # Cost: 1.414 × 1.0 = 1.414
            multiplier = 1.0
    # Special case: Diagonal upward from ground (efficient N++ movement)
    elif dx != 0 and dy < 0 and src_grounded:
        # Grounded diagonal jump - efficient upward movement
        # Cost: 1.414 × 0.6 = 0.85
        multiplier = 0.6
    # Special case: Diagonal upward from air without wall (aerial jump continuation)
    elif dx != 0 and dy < 0 and not src_grounded and not src_walled:
        if y_direction_change_to_rising:
            # Transitioning from falling/stationary to rising in mid-air is IMPOSSIBLE
            # Can't reverse vertical momentum without ground or wall contact
            # This prevents zigzag paths that go down then up in the air
            multiplier = 200.0
        elif x_direction_change:
            # Changing X direction while moving upward in mid-air is nearly impossible
            # You can't reverse horizontal momentum without wall or ground contact
            multiplier = 100.0
        else:
            # Aerial upward movement - valid as jump continuation, cost scales with chain
            # Chain 0-2: Reasonable cost (continuing jump trajectory)
            # Chain 3+: Very expensive (beyond physics limits, effectively blocked)
            multiplier = _get_aerial_chain_multiplier(aerial_upward_chain)
    elif dy < 0:  # Moving up (against gravity) - vertical only
        if src_grounded:
            # Vertical jump from ground - reasonable but not free
            # Cost: 1.0 × 0.7 = 0.7
            multiplier = 0.7
        elif src_walled:
            # Wall jump (vertical) - more expensive than ground jump
            # Cost: 1.0 × 1.2 = 1.2
            multiplier = 1.2
        elif y_direction_change_to_rising:
            # Transitioning from falling/stationary to rising in mid-air is IMPOSSIBLE
            # Can't reverse vertical momentum without ground or wall contact
            multiplier = 200.0
        else:
            # Vertical upward from air without wall (aerial jump continuation)
            # Same chain-based cost as diagonal aerial upward
            multiplier = _get_aerial_chain_multiplier(aerial_upward_chain)
    elif dy > 0:  # Moving down (with gravity) - vertical or grounded diagonal
        # Gravity assists falling (0.0667 pixels/frame²)
        # Cheap regardless of grounding state
        multiplier = 0.5
    else:  # Horizontal (dy == 0)
        if src_grounded and dst_grounded:
            # Grounded horizontal - FASTEST and CHEAPEST movement
            # Ground accel 0.0667, max speed 3.333 pixels/frame
            # This is the most efficient movement in N++
            # Base cost: 1.0 × 0.15 = 0.15
            multiplier = 0.15
        elif x_direction_change and not src_grounded and not src_walled:
            # Changing horizontal direction in mid-air is nearly impossible
            # Very limited air control in N++
            multiplier = 100.0
        else:
            # Air horizontal - more expensive than diagonal falling
            # Air accel 0.0444 (~33% slower than ground)
            # Higher cost to prefer diagonal falling over horizontal air movement
            multiplier = 40.0

    # Apply momentum-aware cost adjustment for grounded horizontal movement
    # This makes paths that build and preserve momentum cheaper, enabling
    # momentum-dependent strategies (e.g., running left to build speed before jumping right)
    # OPTIMIZATION: Inline momentum calculation to avoid function call overhead (4M calls!)
    momentum_multiplier = 1.0
    if (
        dy == 0
        and src_grounded
        and dst_grounded
        and grandparent_pos is not None
        and parent_pos is not None
    ):
        # Infer momentum direction inline (avoid function call)
        # Calculate horizontal displacement for last two moves
        grandparent_x, grandparent_y = grandparent_pos
        recent_dx = src_x - parent_x  # Already extracted above
        prev_dx = parent_x - grandparent_x

        # Check if moving consistently in same horizontal direction
        # Both moves must be in same direction and exceed threshold (12px)
        if recent_dx * prev_dx > 0 and abs(recent_dx) >= 12:
            # Consistent movement = momentum
            momentum_direction = -1 if recent_dx < 0 else 1

            # Calculate momentum multiplier inline (avoid function call)
            # Check if edge continues or reverses momentum
            if dx != 0:  # Only for horizontal edges
                continues_momentum = (momentum_direction * dx) > 0
                if continues_momentum:
                    momentum_multiplier = 0.7  # MOMENTUM_CONTINUE_MULTIPLIER
                else:
                    momentum_multiplier = 2.5  # MOMENTUM_REVERSE_MULTIPLIER

    # Apply mine hazard proximity cost multiplier
    # This makes paths near deadly mines more expensive, guiding PBRS toward safer routes
    # Uses cache for O(1) lookup when available
    # Pass hazard_cost_multiplier for curriculum-adaptive costs (Phase 2.2)
    mine_multiplier = _calculate_mine_proximity_cost(
        dst_pos, level_data, mine_proximity_cache, hazard_cost_multiplier
    )

    # NEW: Velocity-aware mine proximity cost using SDF
    # Penalizes building momentum toward hazards (addresses inflection point deaths)
    # OPTIMIZATION: Reuse dx, dy already computed above (velocity_mag uses dx, dy)
    if mine_sdf is not None and parent_pos is not None:
        velocity_mag_sq = dx * dx + dy * dy

        if velocity_mag_sq > 0.25:  # Only if moving significantly (0.5^2 = 0.25)
            # Get SDF gradient at destination (direction away from nearest mine)
            # O(1) lookup from pre-computed SDF
            grad_x, grad_y = mine_sdf.get_gradient_at_position(dst_x, dst_y)

            # Check if near a mine (gradient magnitude > 0.01)
            grad_mag_sq = grad_x * grad_x + grad_y * grad_y
            if grad_mag_sq > 0.0001:  # 0.01^2 = 0.0001
                # OPTIMIZATION: Delay sqrt until needed
                velocity_mag = velocity_mag_sq**0.5

                # Normalize velocity direction (reuse dx, dy)
                vel_dir_x = dx / velocity_mag
                vel_dir_y = dy / velocity_mag

                # Dot product: positive = moving toward mine, negative = moving away
                # We negate gradient because grad points away from mine
                toward_mine = -(vel_dir_x * grad_x + vel_dir_y * grad_y)

                if toward_mine > 0.3:  # Moving significantly toward mine (>17 degrees)
                    # Velocity multiplier: faster approach toward hazard = higher cost
                    # This makes paths that slow down/redirect before mines cheaper
                    velocity_factor = (
                        1.0 + toward_mine * velocity_mag * 0.16667
                    )  # / 12.0 * 2.0
                    mine_multiplier *= velocity_factor

    return base_cost * multiplier * momentum_multiplier * mine_multiplier


def _is_node_grounded_util(
    node_pos: Tuple[int, int],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
) -> bool:
    """
    Check if a node is grounded (has solid surface below).

    Uses base_adjacency (pre-entity-mask) to determine actual level geometry.

    NOTE: This is the canonical implementation shared across all pathfinding code.
    Imported by pathfinding_algorithms.py to avoid duplication.
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


def _is_horizontal_edge_util(
    from_pos: Tuple[int, int], to_pos: Tuple[int, int]
) -> bool:
    """
    Check if edge is horizontal (same y, different x).

    NOTE: This is the canonical implementation shared across all pathfinding code.
    Imported by pathfinding_algorithms.py to avoid duplication.
    """
    # OPTIMIZATION: Direct comparison is faster than tuple indexing
    # Checking y first is more efficient as it's more likely to differ
    return from_pos[1] == to_pos[1] and from_pos[0] != to_pos[0]


def _violates_horizontal_rule_util(
    current: Tuple[int, int],
    neighbor: Tuple[int, int],
    parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
) -> bool:
    """
    Check if edge violates consecutive horizontal non-grounded rule.

    Uses physics cache for O(1) grounding checks if available, otherwise falls back
    to base_adjacency traversal.

    NOTE: This is the canonical OPTIMIZED implementation shared across all pathfinding code.
    Imported by pathfinding_algorithms.py to avoid duplication.
    This version includes physics_cache optimization (not in old duplicate version).

    OPTIMIZATION: Heavily optimized hot path (called 3.7M times in profile).
    """
    # OPTIMIZATION: Quick check for horizontal edge using tuple unpacking
    # This avoids function call overhead for _is_horizontal_edge_util
    curr_x, curr_y = current
    neigh_x, neigh_y = neighbor

    # Not horizontal if y differs or x is same (most common case - early exit)
    if curr_y != neigh_y or curr_x == neigh_x:
        return False

    # OPTIMIZATION: Direct dict access with local caching (avoid repeated lookups)
    # Extract grounded status once and cache in local variables
    current_physics = physics_cache[current]
    neighbor_physics = physics_cache[neighbor]
    current_grounded = current_physics["grounded"]
    neighbor_grounded = neighbor_physics["grounded"]

    # Both grounded = no violation (common case - early exit)
    if current_grounded and neighbor_grounded:
        return False

    # Check parent (use .get() here as it's only called if not grounded)
    parent = parents.get(current)
    if parent is None:
        return False

    # Check if parent->current is also horizontal
    # OPTIMIZATION: Direct tuple unpacking and comparison
    par_x, par_y = parent
    # Horizontal if same y and different x
    return par_y == curr_y and par_x != curr_x


def classify_edge_type(
    src_pos: Tuple[int, int],
    dst_pos: Tuple[int, int],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
) -> str:
    """
    Classify edge type based on movement physics.

    Returns edge type as string matching the 8 cases in physics-aware cost calculation.

    Args:
        src_pos: Source node position (x, y) in pixels
        dst_pos: Destination node position (x, y) in pixels
        base_adjacency: Base graph adjacency for grounding checks (pre-entity-mask)

    Returns:
        Edge type string: one of "diagonal_jump_grounded", "diagonal_fall_air", "diagonal_up_air",
        "jump_grounded", "vertical_air_up", "falling", "grounded_horizontal", "air_horizontal"
    """
    dx = dst_pos[0] - src_pos[0]
    dy = dst_pos[1] - src_pos[1]

    # Check grounding for both nodes
    src_grounded = _is_node_grounded_util(src_pos, base_adjacency)
    dst_grounded = _is_node_grounded_util(dst_pos, base_adjacency)

    # Match the exact classification logic from _calculate_physics_aware_cost (pathfinding_algorithms.py)
    # Special case: Diagonal falling from air
    if dx != 0 and dy > 0 and not src_grounded:
        return "diagonal_fall_air"
    # Special case: Diagonal upward from air
    elif dx != 0 and dy < 0 and not src_grounded:
        return "diagonal_up_air"
    # Special case: Diagonal upward from ground (efficient)
    elif dx != 0 and dy < 0 and src_grounded:
        return "diagonal_jump_grounded"
    elif dy < 0:  # Moving up (against gravity) - vertical only
        if src_grounded:
            return "jump_grounded"
        else:
            return "vertical_air_up"
    elif dy > 0:  # Moving down (with gravity)
        return "falling"
    else:  # Horizontal (dy == 0)
        if src_grounded and dst_grounded:
            return "grounded_horizontal"
        else:
            return "air_horizontal"


def get_edge_type_color(edge_type: str) -> Tuple[int, int, int, int]:
    """
    Get RGBA color for edge type visualization.

    Colors are chosen to show cost gradient: green (cheap) -> red (expensive).

    Args:
        edge_type: Edge type string from classify_edge_type

    Returns:
        RGBA color tuple (red, green, blue, alpha)
    """
    color_map = {
        "diagonal_jump_grounded": (
            50,
            255,
            50,
            220,
        ),  # Very bright green - very cheap (0.2x)
        "grounded_horizontal": (100, 255, 100, 220),  # Bright green - cheap (0.5x)
        "jump_grounded": (150, 255, 100, 220),  # Yellow-green - moderate (0.5x)
        "diagonal_fall_air": (100, 220, 255, 220),  # Cyan - very cheap (0.6x)
        "falling": (150, 255, 150, 220),  # Light green - cheap (0.8x)
        "air_horizontal": (255, 150, 100, 220),  # Orange-red - expensive (2.0x)
        "vertical_air_up": (255, 100, 50, 220),  # Dark orange - very expensive (3.0x)
        "diagonal_up_air": (255, 50, 150, 220),  # Magenta - most expensive (10.0x)
    }
    return color_map.get(edge_type, (255, 255, 255, 220))  # Default white


def get_edge_type_label(edge_type: str) -> str:
    """
    Get human-readable label for edge type.

    Args:
        edge_type: Edge type string from classify_edge_type

    Returns:
        Human-readable label string
    """
    label_map = {
        "diagonal_jump_grounded": "Diagonal Jump (Ground)",
        "grounded_horizontal": "Grounded Horiz.",
        "diagonal_fall_air": "Diagonal Fall (Air)",
        "falling": "Falling",
        "jump_grounded": "Jump (Ground)",
        "air_horizontal": "Air Horizontal",
        "vertical_air_up": "Vertical Up (Air)",
        "diagonal_up_air": "Diagonal Up (Air)",
    }
    return label_map.get(edge_type, "Unknown")


def get_edge_type_cost_multiplier(edge_type: str) -> float:
    """
    Get actual edge cost for edge type (for display purposes).

    Computes edge cost = base_cost * multiplier, matching the calculation
    in _calculate_physics_aware_cost from pathfinding_algorithms.py.

    Args:
        edge_type: Edge type string from classify_edge_type

    Returns:
        Actual edge cost (float)
    """
    # Edge costs: base_cost (1.0 for cardinal, 1.414 for diagonal) * multiplier
    cost_map = {
        "grounded_horizontal": 1.0 * 0.5,  # 0.5 - horizontal movement on ground
        "diagonal_fall_air": 1.414 * 0.6,  # 0.848 - diagonal falling from air
        "falling": 1.0 * 0.8,  # 0.8 - vertical falling
        "jump_grounded": 1.0
        * 1.5,  # 1.5 - vertical jump from ground (also applies to diagonal)
        "air_horizontal": 1.0 * 2.0,  # 2.0 - horizontal movement in air
        "vertical_air_up": 1.0 * 3.0,  # 3.0 - vertical upward in air
        "diagonal_up_air": 1.414
        * 10.0,  # 14.14 - diagonal upward from air (very expensive)
    }
    return cost_map.get(edge_type, 1.0)


def _get_subcell_lookup_loader():
    """
    Get or create SubcellNodeLookupLoader singleton instance.

    Auto-generates the lookup file if it doesn't exist.

    Returns:
        SubcellNodeLookupLoader instance (always returns a valid instance)
    """
    global _subcell_lookup_loader_cache

    # Return cached instance if already loaded
    if _subcell_lookup_loader_cache is not None:
        return _subcell_lookup_loader_cache

    # Try to load the singleton
    try:
        from .subcell_node_lookup import (
            SubcellNodeLookupLoader,
            SubcellNodeLookupPrecomputer,
        )

        loader = SubcellNodeLookupLoader()
        # Verify it's actually loaded
        if loader._lookup_table is not None:
            _subcell_lookup_loader_cache = loader
            _logger.info(
                f"Subcell lookup loader initialized successfully: "
                f"shape={loader._lookup_table.shape}, "
                f"size={loader._lookup_table.nbytes / 1024:.2f} KB"
            )
            return loader
        else:
            _logger.error("Subcell lookup loader created but table is None")
            raise RuntimeError("Subcell lookup table is None")
    except FileNotFoundError as e:
        # Auto-generate the lookup file if it doesn't exist
        _logger.info(f"Subcell lookup file not found, auto-generating: {e}")
        try:
            import os
            from .subcell_node_lookup import SubcellNodeLookupPrecomputer

            # Get the data directory path (match path from subcell_node_lookup.py)
            data_path = os.path.join(
                os.path.dirname(__file__), "../../data/subcell_node_lookup.pkl.gz"
            )
            data_dir = os.path.dirname(data_path)

            # Ensure data directory exists
            os.makedirs(data_dir, exist_ok=True)

            # Generate the lookup table
            _logger.info("Precomputing subcell node lookup table...")
            precomputer = SubcellNodeLookupPrecomputer()
            lookup = precomputer.precompute_all(verbose=False)
            precomputer.save_to_file(lookup, data_path, verbose=False)
            _logger.info(f"Subcell lookup table generated successfully at {data_path}")

            # Now try loading again
            loader = SubcellNodeLookupLoader()
            if loader._lookup_table is not None:
                _subcell_lookup_loader_cache = loader
                _logger.info(
                    f"Subcell lookup loader initialized successfully: "
                    f"shape={loader._lookup_table.shape}, "
                    f"size={loader._lookup_table.nbytes / 1024:.2f} KB"
                )
                return loader
            else:
                raise RuntimeError("Failed to load generated lookup table")
        except Exception as gen_error:
            _logger.error(
                f"Failed to auto-generate subcell lookup file: {gen_error}. "
                f"Falling back to loader error."
            )
            raise
    except RuntimeError as e:
        _logger.error(f"Subcell lookup runtime error: {e}")
        raise
    except Exception as e:
        _logger.error(f"Subcell lookup unexpected error: {type(e).__name__}: {e}")
        raise


def extract_spatial_lookups_from_graph_data(
    graph_data: Optional[Dict[str, Any]],
) -> Tuple[Optional[any], Optional[any]]:
    """
    Extract spatial_hash and subcell_lookup from graph_data.

    Helper function to get both spatial lookup mechanisms from graph_data.
    This ensures consistent access across all call sites.

    Args:
        graph_data: Graph data dict (may contain spatial_hash)

    Returns:
        Tuple of (spatial_hash, subcell_lookup):
        - spatial_hash: SpatialHash instance from graph_data, or None
        - subcell_lookup: SubcellNodeLookupLoader singleton, or None if unavailable
    """
    spatial_hash = None
    if graph_data is not None:
        spatial_hash = graph_data.get("spatial_hash")
        _logger.debug(
            f"extract_spatial_lookups: graph_data provided, spatial_hash={spatial_hash is not None}"
        )
    else:
        _logger.debug("extract_spatial_lookups: graph_data is None")

    # Always try to load subcell lookup (singleton, loads once, auto-generates if needed)
    try:
        subcell_lookup = _get_subcell_lookup_loader()
        _logger.debug("extract_spatial_lookups: subcell_lookup available")
    except Exception as e:
        _logger.warning(f"Failed to load subcell lookup: {e}")
        subcell_lookup = None

    return spatial_hash, subcell_lookup


def find_closest_node_to_position(
    world_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    threshold: Optional[float] = None,
    entity_radius: float = 0.0,
    ninja_radius: float = 10.0,
    spatial_hash: Optional[any] = None,
    subcell_lookup: Optional[any] = None,
    prefer_grounded: bool = False,
) -> Optional[Tuple[int, int]]:
    """
    Find the closest node to a world position with optional spatial indexing.

    Priority order:
    1. Precomputed subcell lookup (fastest, O(1) direct array access)
    2. Spatial hash (O(1) grid-based lookup)
    3. Linear search (fallback, O(N))

    Coordinate Systems:
    - Node positions (adjacency keys): Tile data space (excludes 1-tile padding)
    - World positions (entities, ninja): Full map space (includes 1-tile padding)
    - Offset: Add +24 to node coords to convert to world coords for comparison

    Args:
        world_pos: World position (x, y) in pixels (full map space)
        adjacency: Graph adjacency structure (keys in tile data space)
        threshold: Maximum distance threshold (if None, calculated as ninja_radius + entity_radius)
        entity_radius: Collision radius of the entity at world_pos (default 0.0)
        ninja_radius: Collision radius of the ninja (default 10.0)
        spatial_hash: Optional SpatialHash instance for O(1) lookup
        subcell_lookup: Optional SubcellNodeLookupLoader instance for fastest lookup
        prefer_grounded: If True, prioritize grounded nodes over air nodes

    Returns:
        Closest node position (in tile data space), or None if no node within threshold
    """
    if not adjacency:
        return None

    # Calculate threshold from radii if not provided
    if threshold is None:
        threshold = ninja_radius + entity_radius

    world_x, world_y = world_pos

    # Convert query position from world space to tile data space
    query_x = world_x - NODE_WORLD_COORD_OFFSET
    query_y = world_y - NODE_WORLD_COORD_OFFSET

    # Get subcell_lookup if not provided (auto-loads and auto-generates if needed)
    if subcell_lookup is None:
        try:
            subcell_lookup = _get_subcell_lookup_loader()
        except Exception as e:
            _logger.warning(f"Failed to load subcell lookup: {e}")
            subcell_lookup = None

    # Fastest path: Use precomputed subcell lookup if available (O(1) direct access)
    if subcell_lookup is not None:
        try:
            # Use threshold as max_radius for entity radius handling (4-12px entities)
            closest_node = subcell_lookup.find_closest_node_position(
                query_x,
                query_y,
                adjacency,
                max_radius=threshold,
                prefer_grounded=prefer_grounded,
            )
            if closest_node is not None:
                return closest_node
        except Exception as e:
            # Other errors (e.g., lookup table not loaded)
            _logger.debug(
                f"Subcell lookup failed: {e}. Falling back to spatial hash or linear search."
            )

    # Fallback to spatial hash if available
    if spatial_hash is not None:
        try:
            # Spatial hash lookup
            candidates = spatial_hash.query(query_x, query_y, radius=threshold)
            if candidates:
                # Find closest candidate
                min_dist = float("inf")
                closest = None
                for candidate in candidates:
                    if candidate in adjacency:
                        dist_sq = (candidate[0] - query_x) ** 2 + (
                            candidate[1] - query_y
                        ) ** 2
                        if dist_sq < min_dist:
                            min_dist = dist_sq
                            closest = candidate
                if closest is not None and min_dist <= threshold * threshold:
                    return closest
        except Exception as e:
            _logger.debug(f"Spatial hash lookup failed: {e}")

    # Final fallback: Linear search (O(N))
    min_dist = float("inf")
    closest = None
    for node_pos in adjacency.keys():
        nx, ny = node_pos
        dist_sq = (nx - query_x) ** 2 + (ny - query_y) ** 2
        if dist_sq < min_dist:
            min_dist = dist_sq
            closest = node_pos

    if closest is not None and min_dist <= threshold * threshold:
        return closest

    return None


def _is_node_grounded(
    node_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
) -> bool:
    """
    Check if a node is grounded (has solid/blocked surface directly below).

    A node is grounded if there's a solid surface preventing downward movement.
    In screen coordinates: y=0 is top, y increases going DOWN.

    A node is grounded if the position 12px directly below (same x, y+12):
    - Does NOT exist in the adjacency graph (solid tile below), OR
    - Exists but there's NO direct vertical edge to it (blocked by geometry)

    Args:
        node_pos: Node position (x, y) in pixels
        adjacency: Adjacency graph structure

    Returns:
        True if node is grounded (on a surface), False otherwise (mid-air)
    """
    x, y = node_pos
    below_pos = (x, y + 12)  # 12px down (y increases downward)

    # If node directly below doesn't exist in graph, this node is on solid surface
    if below_pos not in adjacency:
        return True

    # Node below exists - check if we have a direct vertical edge to it
    # If we CAN fall to it (edge exists), we're NOT grounded
    # If we CAN'T fall to it (no edge), we ARE grounded
    if node_pos in adjacency:
        neighbors = adjacency[node_pos]
        for neighbor_info in neighbors:
            # Handle both tuple format (neighbor_pos, cost) and just neighbor_pos
            if isinstance(neighbor_info, tuple) and len(neighbor_info) >= 2:
                neighbor_pos = neighbor_info[0]
            else:
                neighbor_pos = neighbor_info

            if neighbor_pos == below_pos:
                # Found direct vertical edge downward - NOT grounded (can fall)
                return False

    # No direct downward edge found - node is grounded
    return True


def find_start_node_for_player(
    player_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    player_radius: float = 10.0,
    spatial_hash: Optional[any] = None,
    subcell_lookup: Optional[any] = None,
    goal_pos: Optional[Tuple[int, int]] = None,
    prefer_grounded: bool = True,
) -> Optional[Tuple[int, int]]:
    """
    Find best start node for player position, preferring grounded nodes.

    Priority:
    1. If prefer_grounded=True: Grounded nodes within player radius
       - Among grounded, prefer closest to goal or topmost
    2. If no grounded or prefer_grounded=False: Any overlapped nodes
       - If multiple overlapped and goal_pos provided: select closest to goal
       - Otherwise: select topmost (lowest y value)
    3. If no overlapped nodes, closest node within threshold

    Args:
        player_pos: Player world position (x, y) in pixels
        adjacency: Graph adjacency structure
        player_radius: Player collision radius (default 10.0)
        spatial_hash: Optional SpatialHash for fast lookup
        subcell_lookup: Optional SubcellNodeLookupLoader for fastest lookup
        goal_pos: Optional goal position to prefer nodes in goal direction
        prefer_grounded: If True, prioritize grounded nodes (default True)

    Returns:
        Best start node, or None if none found
    """
    if not adjacency:
        return None

    world_x, world_y = player_pos
    query_x = world_x - NODE_WORLD_COORD_OFFSET
    query_y = world_y - NODE_WORLD_COORD_OFFSET

    # Get candidates within larger search radius
    search_radius = player_radius * 2.0  # Search wider initially

    overlapped_grounded = []
    overlapped_air = []
    nearby_grounded = []
    nearby_air = []

    # Get subcell_lookup if not provided
    if subcell_lookup is None:
        try:
            subcell_lookup = _get_subcell_lookup_loader()
        except Exception:
            subcell_lookup = None

    # Use spatial indexing to get candidates
    candidates = []
    if subcell_lookup is not None:
        try:
            # Use subcell lookup to find nodes within search radius
            closest = subcell_lookup.find_closest_node_position(
                query_x, query_y, adjacency, max_radius=search_radius
            )
            if closest is not None:
                # Get all nearby candidates using spatial hash or linear search
                if spatial_hash is not None:
                    candidates = spatial_hash.query(
                        query_x, query_y, radius=search_radius
                    )
                    candidates = [c for c in candidates if c in adjacency]
                else:
                    # Fallback: check all nodes
                    for node in adjacency.keys():
                        nx, ny = node
                        dist_sq = (nx - query_x) ** 2 + (ny - query_y) ** 2
                        if dist_sq <= search_radius * search_radius:
                            candidates.append(node)
        except Exception:
            candidates = []

    if not candidates:
        if spatial_hash is not None:
            try:
                candidates = spatial_hash.query(query_x, query_y, radius=search_radius)
                candidates = [c for c in candidates if c in adjacency]
            except Exception:
                candidates = []

    # Final fallback: all nodes
    if not candidates:
        candidates = list(adjacency.keys())

    # Categorize candidates by overlap and grounding
    for node in candidates:
        nx, ny = node
        dist_sq = (nx - query_x) ** 2 + (ny - query_y) ** 2
        dist = dist_sq**0.5

        is_grounded = _is_node_grounded(node, adjacency) if prefer_grounded else False

        if dist <= player_radius:
            # Node center is within player radius (overlapped)
            if prefer_grounded:
                if is_grounded:
                    overlapped_grounded.append((node, dist))
                else:
                    overlapped_air.append((node, dist))
            else:
                overlapped_air.append(
                    (node, dist)
                )  # Use air list as "all" when not preferring
        elif dist <= search_radius:
            if prefer_grounded:
                if is_grounded:
                    nearby_grounded.append((node, dist))
                else:
                    nearby_air.append((node, dist))
            else:
                nearby_air.append(
                    (node, dist)
                )  # Use air list as "all" when not preferring

    # Selection priority: overlapped grounded > overlapped air > nearby grounded > nearby air
    selection_candidates = None
    if overlapped_grounded:
        selection_candidates = overlapped_grounded
    elif overlapped_air:
        selection_candidates = overlapped_air
    elif nearby_grounded:
        selection_candidates = nearby_grounded
    elif nearby_air:
        selection_candidates = nearby_air

    if selection_candidates:
        if len(selection_candidates) > 1:
            if goal_pos is not None:
                # Prefer node closest to goal (most relevant direction)
                goal_x = goal_pos[0] - NODE_WORLD_COORD_OFFSET
                goal_y = goal_pos[1] - NODE_WORLD_COORD_OFFSET
                selection_candidates.sort(
                    key=lambda x: (x[0][0] - goal_x) ** 2 + (x[0][1] - goal_y) ** 2
                )
                return selection_candidates[0][0]
            else:
                # Prefer topmost node (lowest y value) for consistency
                selection_candidates.sort(key=lambda x: x[0][1])
                return selection_candidates[0][0]
        else:
            return selection_candidates[0][0]

    return None


def find_goal_node_closest_to_start(
    goal_pos: Tuple[int, int],
    start_node: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    entity_radius: float = 0.0,
    ninja_radius: float = 10.0,
    spatial_hash: Optional[any] = None,
    subcell_lookup: Optional[any] = None,
    search_radius_override: Optional[float] = None,
) -> Optional[Tuple[int, int]]:
    """
    Find goal node closest to start among nodes overlapped by entity radius.

    Strategy:
    1. Find all nodes within entity_radius of goal position (overlapped nodes)
    2. Among those, select the one closest to start_node
    3. If no overlapped nodes, expand search to ninja_radius + entity_radius

    Args:
        goal_pos: Goal world position (x, y) in pixels
        start_node: Start node position (for proximity check)
        adjacency: Graph adjacency structure
        entity_radius: Entity collision radius
        ninja_radius: Ninja collision radius
        spatial_hash: Optional SpatialHash
        subcell_lookup: Optional SubcellNodeLookupLoader
        search_radius_override: Optional override for search radius (default: ninja_radius + entity_radius)

    Returns:
        Best goal node, or None if none found
    """
    if not adjacency or start_node is None:
        return None

    world_x, world_y = goal_pos
    query_x = world_x - NODE_WORLD_COORD_OFFSET
    query_y = world_y - NODE_WORLD_COORD_OFFSET

    # Get subcell_lookup if not provided
    if subcell_lookup is None:
        try:
            subcell_lookup = _get_subcell_lookup_loader()
        except Exception:
            subcell_lookup = None

    # First try to find nodes overlapped by entity radius
    overlapped_nodes = []
    nearby_nodes = []

    # Search radius for candidates (use override if provided)
    search_radius = (
        search_radius_override
        if search_radius_override is not None
        else (ninja_radius + entity_radius)
    )

    # Get candidates within search radius
    candidates = []
    if spatial_hash is not None:
        try:
            candidates = spatial_hash.query(query_x, query_y, radius=search_radius)
            candidates = [c for c in candidates if c in adjacency]
        except Exception:
            pass

    # Fallback: linear search if no spatial hash
    if not candidates:
        for node in adjacency.keys():
            nx, ny = node
            dist_sq = (nx - query_x) ** 2 + (ny - query_y) ** 2
            if dist_sq <= search_radius * search_radius:
                candidates.append(node)

    if not candidates:
        return None

    # Categorize candidates by distance to goal
    for candidate in candidates:
        cx, cy = candidate
        dist_to_goal = ((cx - query_x) ** 2 + (cy - query_y) ** 2) ** 0.5

        if dist_to_goal <= entity_radius:
            # Node overlapped by entity radius
            overlapped_nodes.append(candidate)
        else:
            nearby_nodes.append(candidate)

    # Prefer nodes overlapped by entity radius
    search_set = overlapped_nodes if overlapped_nodes else nearby_nodes

    if not search_set:
        return None

    # Among candidates, find closest to start_node
    sx, sy = start_node
    min_dist_to_start = float("inf")
    best_node = None

    for candidate in search_set:
        cx, cy = candidate
        dist_sq = (cx - sx) ** 2 + (cy - sy) ** 2
        if dist_sq < min_dist_to_start:
            min_dist_to_start = dist_sq
            best_node = candidate

    return best_node


def find_ninja_node(
    ninja_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    spatial_hash: Optional[any] = None,
    subcell_lookup: Optional[any] = None,
    ninja_radius: float = 10.0,
    goal_node: Optional[Tuple[int, int]] = None,
    level_cache: Optional[any] = None,
    goal_id: Optional[str] = None,
    search_radius_override: Optional[float] = None,
) -> Optional[Tuple[int, int]]:
    """
    Find the node representing the ninja's current position.

    This is the canonical function for finding the start node for the ninja,
    used by both pathfinding and debug visualization to ensure consistency.

    Strategy:
    1. Find all nodes within ninja_radius (10px by default - player collision radius)
    2. Among overlapping nodes:
       - If level_cache provided: use cached PATH distance to goal (correct behavior)
       - Else if goal_node provided: use Euclidean distance (legacy, may cause issues)
       - Otherwise: return the closest one to ninja center
    3. If no nodes overlap, fall back to visual matching (within 5px for blue highlight)

    Args:
        ninja_pos: Ninja world position (x, y) in pixels
        adjacency: Graph adjacency structure
        spatial_hash: Optional SpatialHash for fast lookup
        subcell_lookup: Optional SubcellNodeLookupLoader for fastest lookup
        ninja_radius: Ninja collision radius in pixels (default 10.0)
        goal_node: Optional goal node for path-aware start node selection
        level_cache: Optional LevelBasedPathDistanceCache for correct path distance selection
        goal_id: Optional goal identifier for level_cache lookup (e.g., "switch" or "exit")
        search_radius_override: Optional override for search radius (default: ninja_radius)

    Returns:
        Ninja node position, or None if no suitable node found
    """
    if not ninja_pos or not adjacency:
        return None

    # Convert ninja position to tile data space for comparison with nodes
    ninja_x = ninja_pos[0] - NODE_WORLD_COORD_OFFSET
    ninja_y = ninja_pos[1] - NODE_WORLD_COORD_OFFSET

    # Use search_radius_override if provided, otherwise use ninja_radius
    search_radius = (
        search_radius_override if search_radius_override is not None else ninja_radius
    )

    # OPTIMIZATION: Use spatial indexing if available (avoid O(N) linear search)
    overlapping_nodes = []

    if spatial_hash is not None:
        # Fast O(1) spatial hash lookup
        try:
            candidates = spatial_hash.query(ninja_x, ninja_y, radius=search_radius)
            for pos in candidates:
                if pos in adjacency:
                    x, y = pos
                    dist_sq = (x - ninja_x) ** 2 + (y - ninja_y) ** 2
                    if dist_sq <= search_radius * search_radius:
                        overlapping_nodes.append((pos, dist_sq))
        except Exception:
            # Fallback to linear search if spatial hash fails
            pass

    # Fallback: Linear search if no spatial hash or it failed
    if not overlapping_nodes:
        for pos in adjacency.keys():
            x, y = pos
            dist_sq = (x - ninja_x) ** 2 + (y - ninja_y) ** 2
            if dist_sq <= search_radius * search_radius:
                overlapping_nodes.append((pos, dist_sq))

    # If we found overlapping nodes, select the best one
    if overlapping_nodes:
        # If goal provided and multiple candidates, pick node closest to goal
        if goal_node is not None and len(overlapping_nodes) > 1:
            best_node = None
            best_distance = float("inf")
            gx, gy = goal_node
            candidate_info = []  # For diagnostic logging
            use_path_distance = level_cache is not None and goal_id is not None

            # Track if any fallbacks occurred for diagnostic logging
            euclidean_fallback_count = 0

            for node_pos, dist_sq_to_ninja in overlapping_nodes:
                nx, ny = node_pos

                if use_path_distance:
                    # FIX: Use cached PATH distance instead of Euclidean distance
                    # This correctly handles levels where the path goes away from the goal first
                    dist_to_goal = level_cache.get_geometric_distance(
                        node_pos, goal_node, goal_id
                    )
                    # Fallback to Euclidean if cache miss (can happen for nodes outside flood-fill)
                    if dist_to_goal == float("inf"):
                        dist_to_goal = ((gx - nx) ** 2 + (gy - ny) ** 2) ** 0.5
                        euclidean_fallback_count += 1
                        _logger.debug(
                            f"[PBRS] Cache miss for node {node_pos}, using Euclidean fallback."
                        )
                else:
                    # Legacy: Euclidean distance to goal
                    # WARNING: This can cause incorrect node selection when path requires
                    # going away from the goal first (e.g., going left when goal is right)
                    dist_to_goal = ((gx - nx) ** 2 + (gy - ny) ** 2) ** 0.5
                    euclidean_fallback_count += 1

                candidate_info.append((node_pos, dist_to_goal, dist_sq_to_ninja**0.5))
                if dist_to_goal < best_distance:
                    best_distance = dist_to_goal
                    best_node = node_pos

            # DIAGNOSTIC: Log when multiple nodes compete or Euclidean fallback was used
            if best_node is not None:
                # Check if nodes have significantly different distances
                distances = [info[1] for info in candidate_info]
                dist_spread = (
                    max(distances) - min(distances) if len(distances) > 1 else 0
                )

                # Log at debug level if Euclidean fallback was used (no level_cache)
                # This is expected in some code paths (e.g., visualization, cache miss)
                if not use_path_distance and len(overlapping_nodes) > 1:
                    _logger.debug(
                        f"[PBRS] Node selection without path cache. "
                        f"Using Euclidean distance. "
                        f"ninja_pos={ninja_pos}, candidates={len(overlapping_nodes)}"
                    )
                elif dist_spread > 10.0:  # Nodes differ by >10px
                    distance_type = "PATH" if use_path_distance else "EUCLIDEAN"
                    log_level = _logger.debug if use_path_distance else _logger.warning
                    log_level(
                        f"[PBRS_DIAG] Node selection using {distance_type} distance to goal. "
                        f"ninja_pos={ninja_pos}, selected={best_node}, "
                        f"dist_to_goal={best_distance:.1f}px, "
                        f"candidates (node, dist_to_goal, dist_to_ninja): {candidate_info}, "
                        f"spread={dist_spread:.1f}px."
                        + (
                            ""
                            if use_path_distance
                            else " WARNING: May select wrong node if path goes away from goal first!"
                        )
                    )
                return best_node

        # Otherwise, return closest to ninja center
        overlapping_nodes.sort(key=lambda node: node[1])  # Sort by distance
        return overlapping_nodes[0][0]

    # Fallback: visual matching within 5 pixels (for blue node coloring consistency)
    # This handles edge cases where ninja is very close but not quite overlapping
    for pos in adjacency.keys():
        x, y = pos
        if (
            abs(x + NODE_WORLD_COORD_OFFSET - ninja_pos[0]) < 5
            and abs(y + NODE_WORLD_COORD_OFFSET - ninja_pos[1]) < 5
        ):
            return pos

    return None


def bfs_distance_from_start(
    start_node: Tuple[int, int],
    target_node: Optional[Tuple[int, int]],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    max_distance: Optional[float] = None,
    physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
    level_data: Optional[Any] = None,
    mine_proximity_cache: Optional[Any] = None,
    return_parents: bool = False,
    use_geometric_costs: bool = False,
    track_geometric_distances: bool = False,
    mine_sdf: Optional[Any] = None,
) -> Tuple[
    Dict[Tuple[int, int], float],
    Optional[float],
    Optional[Dict],
    Optional[Dict[Tuple[int, int], float]],
]:
    """
    Calculate distances from start node using Dijkstra's algorithm with physics validation.

    Applies same physics-aware costs and validation as find_shortest_path:
    - Horizontal edges: Checks consecutive non-grounded rule
    - Diagonal upward edges: Requires source to be grounded
    - Edge costs: Physics-based (grounded horizontal fastest, upward expensive, etc.)
    - Velocity-aware mine costs: Uses SDF to penalize approaching hazards at high speed

    Uses Dijkstra's algorithm (priority queue) to find lowest-cost distances according to physics costs.

    Args:
        start_node: Starting node position
        target_node: Optional target node to find distance to (early termination)
        adjacency: Masked graph adjacency structure (for pathfinding)
        base_adjacency: Base graph adjacency structure (pre-entity-mask, for physics checks)
        max_distance: Optional maximum distance to compute (for early termination)
        physics_cache: Optional pre-computed physics properties for O(1) lookups
        level_data: Optional LevelData for mine proximity checks (fallback)
        mine_proximity_cache: Optional MineProximityCostCache for O(1) mine cost lookup
        return_parents: If True, return the parents dict for path reconstruction
        use_geometric_costs: If True, use actual geometric distances (pixels) instead of
            physics-weighted costs. Use this for PBRS normalization where you need the
            actual path length in pixels, not the physics-optimal cost.
        track_geometric_distances: If True AND use_geometric_costs=False, also track
            geometric distances (pixels) along the physics-optimal path. This allows
            returning the pixel length of the physics-optimal path. The priority queue
            still uses physics costs for ordering.
        mine_sdf: Optional MineSignedDistanceField for velocity-aware hazard costs

    Returns:
        Tuple of (distances_dict, target_distance, parents_dict, geometric_distances_dict):
        - distances_dict: Map of node -> distance from start (physics or geometric based on use_geometric_costs)
        - target_distance: Distance to target_node if found, None otherwise
        - parents_dict: Map of node -> parent node (only if return_parents=True, else None)
        - geometric_distances_dict: Map of node -> geometric distance along physics-optimal path
          (only if track_geometric_distances=True AND use_geometric_costs=False, else None)
    """
    # Priority queue: (distance, node)
    pq = [(0.0, start_node)]
    distances = {start_node: 0.0}
    parents = {start_node: None}  # Track parents for horizontal rule
    grandparents = {start_node: None}  # Track grandparents for momentum inference
    visited = set()

    # Track geometric distances along physics-optimal path if requested
    # This gives us the pixel length of the physics-optimal path
    geometric_distances: Optional[Dict[Tuple[int, int], float]] = None
    if track_geometric_distances and not use_geometric_costs:
        geometric_distances = {start_node: 0.0}

    # OPTIMIZATION: Early termination heuristic for unreachable cases
    # Calculate Euclidean distance to target for bounding search
    euclidean_to_target = None
    if target_node is not None:
        dx_target = target_node[0] - start_node[0]
        dy_target = target_node[1] - start_node[1]
        euclidean_to_target = (dx_target * dx_target + dy_target * dy_target) ** 0.5
        # Termination threshold: 2x Euclidean (generous for winding paths)
        early_termination_distance = euclidean_to_target * 2.0

    while pq:
        current_dist, current = heapq.heappop(pq)

        # Skip if already visited
        if current in visited:
            continue
        visited.add(current)

        # Early termination if we found target and it's requested
        if target_node is not None and current == target_node:
            parents_result = parents if return_parents else None
            return distances, current_dist, parents_result, geometric_distances

        # Early termination if we've exceeded max distance
        if max_distance is not None and current_dist > max_distance:
            continue

        # OPTIMIZATION: Early termination for unreachable cases
        # If exploring nodes beyond 2x Euclidean distance to target, likely unreachable
        # This prevents exhaustive search of entire graph for impossible paths
        if (
            euclidean_to_target is not None
            and current_dist > early_termination_distance
        ):
            # Check if we've found ANY path to target yet
            if target_node not in distances:
                # Still searching, but current path is very long - likely unreachable
                # Continue for a bit longer in case there's a winding path
                pass  # Let it continue, termination is soft
            else:
                # Already found target, no need to explore further
                break

        # OPTIMIZATION: Direct dict access instead of .get() (hot loop, millions of calls)
        neighbors = adjacency.get(current, [])
        # Cache parent and grandparent lookups to avoid repeated dict.get() calls
        current_parent = parents.get(current)
        current_grandparent = grandparents.get(current)

        for neighbor_info in neighbors:
            neighbor_pos, _ = (
                neighbor_info  # Ignore stored cost, calculate physics-aware cost
            )

            if neighbor_pos in visited:
                continue

            # Check horizontal rule before accepting edge (uses physics cache)
            if _violates_horizontal_rule_util(
                current, neighbor_pos, parents, base_adjacency, physics_cache
            ):
                continue

            # Calculate edge cost
            if use_geometric_costs:
                # Use actual geometric distance in pixels
                # Sub-nodes are spaced 12 pixels apart
                dx = neighbor_pos[0] - current[0]
                dy = neighbor_pos[1] - current[1]
                # Cardinal: 12px, Diagonal: ~17px (12 * sqrt(2))
                cost = (dx * dx + dy * dy) ** 0.5
            else:
                # Calculate physics-aware edge cost with momentum tracking (uses base_adjacency)
                # OPTIMIZATION: Pass cached parent/grandparent instead of dict.get() calls
                cost = _calculate_physics_aware_cost(
                    current,
                    neighbor_pos,
                    base_adjacency,
                    current_parent,
                    physics_cache,
                    level_data,
                    mine_proximity_cache,
                    0,  # aerial_upward_chain not tracked here (optimization)
                    current_grandparent,  # Pass cached grandparent
                    mine_sdf,  # Pass SDF for velocity-aware mine costs
                )

            new_dist = current_dist + cost

            # OPTIMIZATION: Direct comparison instead of .get() with default
            if neighbor_pos not in distances or new_dist < distances[neighbor_pos]:
                distances[neighbor_pos] = new_dist
                parents[neighbor_pos] = current  # Track parent for horizontal rule
                grandparents[neighbor_pos] = (
                    current_parent  # Track grandparent (cached)
                )
                heapq.heappush(pq, (new_dist, neighbor_pos))

                # Track geometric distance along physics-optimal path
                if geometric_distances is not None:
                    dx = neighbor_pos[0] - current[0]
                    dy = neighbor_pos[1] - current[1]
                    geometric_edge_cost = (dx * dx + dy * dy) ** 0.5
                    # OPTIMIZATION: Direct access instead of .get()
                    current_geometric_dist = geometric_distances.get(current, 0.0)
                    geometric_distances[neighbor_pos] = (
                        current_geometric_dist + geometric_edge_cost
                    )

    # Return distances dict, target distance, optionally parents, and optionally geometric distances
    target_distance = distances.get(target_node) if target_node else None
    parents_result = parents if return_parents else None
    return distances, target_distance, parents_result, geometric_distances


def calculate_geometric_path_distance(
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
    spatial_hash: Optional[Any] = None,
    subcell_lookup: Optional[Any] = None,
    entity_radius: float = 0.0,
    ninja_radius: float = 10.0,
    level_data: Optional[Any] = None,
    mine_proximity_cache: Optional[Any] = None,
    mine_sdf: Optional[Any] = None,
) -> float:
    """
    Calculate the geometric (pixel) path distance along the physics-optimal path.

    Uses physics-aware costs for pathfinding (to find the "easiest" path according
    to N++ physics), but returns the actual path length in pixels. This ensures
    PBRS uses the physics-optimal path's length, not the geometrically-shortest path.

    This is important for levels where the physics-optimal path differs from the
    geometrically-shortest path (e.g., taking a longer but easier route that
    avoids difficult jumps).

    Args:
        start_pos: Start position (x, y) in pixels
        goal_pos: Goal position (x, y) in pixels
        adjacency: Masked graph adjacency structure (for pathfinding)
        base_adjacency: Base graph adjacency structure (pre-entity-mask, for physics checks)
        physics_cache: Pre-computed physics properties for O(1) lookups
        spatial_hash: Optional spatial hash for fast node lookup
        subcell_lookup: Optional subcell lookup for node snapping
        entity_radius: Collision radius of the goal entity (default 0.0)
        ninja_radius: Collision radius of the ninja (default 10.0)
        level_data: Optional LevelData for mine proximity checks
        mine_proximity_cache: Optional MineProximityCostCache for mine cost calculations
        mine_sdf: Optional MineSignedDistanceField for velocity-aware hazard costs

    Returns:
        Geometric path distance in pixels along the physics-optimal path,
        or float('inf') if unreachable
    """
    # Find closest nodes to start and goal positions
    start_node = find_ninja_node(
        start_pos,
        adjacency,
        spatial_hash=spatial_hash,
        subcell_lookup=subcell_lookup,
        ninja_radius=ninja_radius,
    )

    goal_node = find_closest_node_to_position(
        goal_pos,
        adjacency,
        threshold=None,  # Will be calculated from radii
        entity_radius=entity_radius,
        ninja_radius=ninja_radius,
        spatial_hash=spatial_hash,
        subcell_lookup=subcell_lookup,
    )

    if start_node is None or goal_node is None:
        return float("inf")

    if start_node == goal_node:
        return 0.0

    # Use physics costs for pathfinding, but track geometric distances along the path
    # This gives us the pixel length of the physics-optimal path
    _, _, _, geometric_distances = bfs_distance_from_start(
        start_node=start_node,
        target_node=goal_node,
        adjacency=adjacency,
        base_adjacency=base_adjacency,
        physics_cache=physics_cache,
        level_data=level_data,
        mine_proximity_cache=mine_proximity_cache,
        mine_sdf=mine_sdf,
        use_geometric_costs=False,  # Use physics costs for pathfinding priority
        track_geometric_distances=True,  # Track pixel distances along physics path
    )

    if geometric_distances is None:
        return float("inf")

    # Get the geometric distance to the goal node along the physics-optimal path
    target_geometric_distance = geometric_distances.get(goal_node)
    if target_geometric_distance is None or target_geometric_distance == float("inf"):
        return float("inf")

    # Adjust for collision radii
    return max(0.0, target_geometric_distance - (ninja_radius + entity_radius))


def find_shortest_path(
    start_node: Tuple[int, int],
    end_node: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
    level_data: Optional[Any] = None,
    mine_proximity_cache: Optional[Any] = None,
) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    """
    Find shortest path from start to end node using Dijkstra's algorithm with physics validation.

    Returns both the path (list of nodes) and the distance.
    Applies physics-aware costs and validation rules during search:
    - Horizontal edges: Checks consecutive non-grounded rule
    - Diagonal upward edges: Requires source to be grounded
    - Edge costs: Physics-based (grounded horizontal fastest, upward expensive, etc.)

    Uses Dijkstra's algorithm (priority queue) to find the lowest-cost path according to physics costs.

    Args:
        start_node: Starting node position
        end_node: Target node position
        adjacency: Masked graph adjacency structure (for pathfinding)
        base_adjacency: Base graph adjacency structure (pre-entity-mask, for physics checks)
        physics_cache: Optional pre-computed physics properties for O(1) lookups
        level_data: Optional LevelData for mine proximity checks (fallback)
        mine_proximity_cache: Optional MineProximityCostCache for O(1) mine cost lookup

    Returns:
        Tuple of (path, distance):
        - path: List of node positions from start to end, or None if unreachable
        - distance: Total path distance (physics-aware), or float('inf') if unreachable
    """
    if start_node == end_node:
        return [start_node], 0.0

    if physics_cache is None:
        raise ValueError("Physics cache is required for physics-aware cost calculation")
    if level_data is None:
        raise ValueError("Level data is required for mine proximity cost calculation")
    if mine_proximity_cache is None:
        raise ValueError(
            "Mine proximity cache is required for mine proximity cost calculation"
        )

    # Priority queue: (distance, node)
    pq = [(0.0, start_node)]
    distances = {start_node: 0.0}
    parents = {start_node: None}
    grandparents = {start_node: None}  # Track grandparents for momentum inference
    visited = set()

    while pq:
        current_dist, current = heapq.heappop(pq)

        # Skip if already visited
        if current in visited:
            continue
        visited.add(current)

        if current == end_node:
            # Reconstruct path
            path = []
            node = end_node
            while node is not None:
                path.append(node)
                node = parents.get(node)
            path.reverse()
            return path, distances[end_node]

        # OPTIMIZATION: Cache parent/grandparent lookups to avoid repeated dict.get()
        current_parent = parents.get(current)
        current_grandparent = grandparents.get(current)

        neighbors = adjacency.get(current, [])
        for neighbor_info in neighbors:
            neighbor_pos, _ = (
                neighbor_info  # Ignore stored cost, calculate physics-aware cost
            )

            if neighbor_pos in visited:
                continue

            # Check horizontal rule before accepting edge (uses physics cache)
            if _violates_horizontal_rule_util(
                current, neighbor_pos, parents, base_adjacency, physics_cache
            ):
                continue

            # Calculate physics-aware edge cost with momentum tracking (uses base_adjacency)
            # OPTIMIZATION: Pass cached parent/grandparent
            cost = _calculate_physics_aware_cost(
                current,
                neighbor_pos,
                base_adjacency,
                current_parent,
                physics_cache,
                level_data,
                mine_proximity_cache,
                0,  # aerial_upward_chain not tracked here (optimization)
                current_grandparent,  # Pass cached grandparent
            )

            new_dist = current_dist + cost

            # OPTIMIZATION: Direct comparison instead of .get() with default
            if neighbor_pos not in distances or new_dist < distances[neighbor_pos]:
                distances[neighbor_pos] = new_dist
                parents[neighbor_pos] = current
                grandparents[neighbor_pos] = current_parent  # Track cached grandparent
                heapq.heappush(pq, (new_dist, neighbor_pos))

    return None, float("inf")


def flood_fill_reachable_nodes(
    start_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    spatial_hash: Optional[any] = None,
    subcell_lookup: Optional[any] = None,
    player_radius: float = 10.0,
    prefer_grounded_start: bool = True,
) -> set:
    """
    Perform flood fill from start position to find all reachable nodes.

    Handles coordinate space conversion and uses optimal spatial lookups.
    This consolidates the flood fill logic used across the codebase.

    Coordinate Systems:
    - start_pos is in world space (includes 1-tile padding)
    - adjacency keys are in tile data space (excludes 1-tile padding)
    - Conversion: tile_data = world - NODE_WORLD_COORD_OFFSET (24px)

    Args:
        start_pos: Starting position in world space (x, y) pixels
        adjacency: Graph adjacency structure (keys in tile data space)
        spatial_hash: Optional SpatialHash instance for O(1) lookup
        subcell_lookup: Optional SubcellNodeLookupLoader instance
        player_radius: Player collision radius in pixels (default 10.0)
        prefer_grounded_start: If True, prefer grounded start nodes (default True)

    Returns:
        Set of reachable node positions (in tile data space)

    Raises:
        RuntimeError: If adjacency is empty or None
    """
    if not adjacency:
        raise RuntimeError(
            "flood_fill_reachable_nodes: adjacency graph is empty or None"
        )

    # Get subcell_lookup if not provided
    if subcell_lookup is None:
        subcell_lookup = _get_subcell_lookup_loader()

    # Find closest node(s) within player radius using optimal lookups
    # Try with player radius first, preferring grounded nodes for better path quality
    closest_node = find_closest_node_to_position(
        start_pos,
        adjacency,
        threshold=player_radius,
        spatial_hash=spatial_hash,
        subcell_lookup=subcell_lookup,
        prefer_grounded=prefer_grounded_start,
    )

    _logger.debug(
        f"[FLOOD_FILL] First attempt (radius={player_radius}, prefer_grounded={prefer_grounded_start}): "
        f"closest_node={closest_node}"
    )

    # Fallback: try with larger threshold if nothing found
    if closest_node is None:
        closest_node = find_closest_node_to_position(
            start_pos,
            adjacency,
            threshold=50.0,
            spatial_hash=spatial_hash,
            subcell_lookup=subcell_lookup,
            prefer_grounded=prefer_grounded_start,
        )
        _logger.debug(
            f"[FLOOD_FILL] Fallback attempt (radius=50.0, prefer_grounded={prefer_grounded_start}): "
            f"closest_node={closest_node}"
        )

    # Check if we found a valid starting node
    if closest_node is None or closest_node not in adjacency:
        _logger.error(
            f"[FLOOD_FILL] FAILED: No valid starting node found. "
            f"start_pos={start_pos} (world space), closest_node={closest_node}, "
            f"in_adjacency={closest_node in adjacency if closest_node else False}, "
            f"adjacency_size={len(adjacency)}"
        )
        # Include sample of adjacency keys for debugging
        if adjacency:
            sample_keys = list(adjacency.keys())[:10]
            _logger.error(
                f"[FLOOD_FILL] Sample adjacency keys (tile data space): {sample_keys}"
            )

        # Return empty set to allow caller to handle gracefully
        return set()

    # Perform BFS flood fill from starting node(s)
    reachable = set()
    queue = deque([closest_node])
    visited = set([closest_node])

    while queue:
        current = queue.popleft()
        reachable.add(current)

        # Get neighbors from adjacency
        neighbors = adjacency.get(current, [])
        for neighbor_info in neighbors:
            # Handle both tuple format (neighbor_pos, cost) and just neighbor_pos
            if isinstance(neighbor_info, tuple) and len(neighbor_info) >= 2:
                neighbor_pos = neighbor_info[0]
            else:
                neighbor_pos = neighbor_info

            if neighbor_pos not in visited:
                visited.add(neighbor_pos)
                queue.append(neighbor_pos)

    return reachable


def compute_reachable_surface_area(
    start_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    graph_data: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute reachable surface area as number of nodes from start position.

    This provides a normalized scale for distance calculations based on the
    actual reachable area of the level. Uses flood fill to count all nodes
    reachable from the starting position.

    Args:
        start_pos: Start position in world space (x, y) pixels
        adjacency: Graph adjacency structure (keys in tile data space)
        graph_data: Optional graph data dict with spatial_hash for optimization

    Returns:
        Total number of reachable nodes (float)

    Raises:
        RuntimeError: If adjacency is empty or no nodes reachable from start
    """
    if not adjacency:
        raise RuntimeError(
            "compute_reachable_surface_area: adjacency graph is empty or None.\n"
            "Surface area calculation requires valid graph data."
        )

    # DEBUG: Log adjacency size and start_pos
    _logger.debug(
        f"[SURFACE_AREA_COMPUTE] start_pos={start_pos} (world space), "
        f"adjacency_size={len(adjacency)}"
    )

    # Log a few sample adjacency keys for debugging
    if adjacency:
        sample_keys = list(adjacency.keys())[:5]
        _logger.debug(f"[SURFACE_AREA_COMPUTE] Sample adjacency keys: {sample_keys}")

    # Extract spatial lookups from graph_data for optimization
    spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(graph_data)

    # Perform flood fill to find all reachable nodes
    reachable_nodes = flood_fill_reachable_nodes(
        start_pos, adjacency, spatial_hash, subcell_lookup
    )

    if not reachable_nodes:
        _logger.error(
            f"[SURFACE_AREA_COMPUTE] FAILED: No reachable nodes from start_pos={start_pos}"
        )

        sample_keys = list(adjacency.keys())[:10] if adjacency else []
        converted_start = (
            start_pos[0] - NODE_WORLD_COORD_OFFSET,
            start_pos[1] - NODE_WORLD_COORD_OFFSET,
        )  # Convert to tile data space

        raise RuntimeError(
            f"compute_reachable_surface_area: no nodes reachable from start position.\n"
            f"start_pos={start_pos} (world space) = {converted_start} (tile data space)\n"
            f"adjacency_size={len(adjacency)}, sample_nodes={sample_keys}\n"
            f"This indicates either:\n"
            f"  1. Degenerate map (too few traversable tiles)\n"
            f"  2. Start position is isolated from graph\n"
            f"  3. Coordinate space mismatch\n"
            f"CRITICAL: This is likely a map generation bug and should not occur in training."
        )

    return float(len(reachable_nodes))


def get_cached_surface_area(
    level_id: str,
    start_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    graph_data: Optional[Dict[str, Any]] = None,
    force_recompute: bool = False,
) -> float:
    """
    Get or compute reachable surface area with caching by level ID.

    Caches surface area per unique level configuration. Cache key is level_id,
    which should uniquely identify the level tiles, entities, and start position.
    This avoids expensive flood fill recomputation for the same level.

    Args:
        level_id: Unique identifier for level configuration (should include
                 switch states if they affect reachability)
        start_pos: Start position in world space (x, y) pixels
        adjacency: Graph adjacency structure (keys in tile data space)
        graph_data: Optional graph data dict with spatial_hash for optimization
        force_recompute: If True, bypass cache and recompute (default False)

    Returns:
        Reachable surface area (float) - number of nodes reachable from start

    Raises:
        RuntimeError: If surface area computation fails
    """
    # Check cache unless force recompute requested
    if not force_recompute and level_id in _surface_area_cache:
        cached_value = _surface_area_cache[level_id]
        # Move to end to maintain LRU ordering
        _surface_area_cache.move_to_end(level_id)
        _logger.debug(
            f"[SURFACE_AREA_CACHE] HIT - cache_key={level_id[:50]}..., "
            f"surface_area={cached_value:.1f}"
        )
        return cached_value

    _logger.debug(
        f"[SURFACE_AREA_CACHE] MISS - cache_key={level_id[:50]}..., "
        f"computing surface area (force_recompute={force_recompute})"
    )

    # Compute surface area
    surface_area = compute_reachable_surface_area(start_pos, adjacency, graph_data)

    # Cache the result with LRU eviction
    _surface_area_cache[level_id] = surface_area
    _surface_area_cache.move_to_end(level_id)

    # Evict oldest entry if cache exceeds max size
    if len(_surface_area_cache) > _SURFACE_AREA_CACHE_MAX_SIZE:
        evicted_key, _ = _surface_area_cache.popitem(last=False)
        _logger.debug(
            f"[SURFACE_AREA_CACHE] EVICTED - cache_key={evicted_key[:50]}..., "
            f"cache_size={len(_surface_area_cache)}"
        )

    _logger.debug(
        f"[SURFACE_AREA_CACHE] CACHED - cache_key={level_id[:50]}..., "
        f"surface_area={surface_area:.1f}, cache_size={len(_surface_area_cache)}"
    )

    return surface_area


def clear_surface_area_cache(level_id: Optional[str] = None) -> None:
    """
    Clear surface area cache.

    This should be called when level data changes or when memory cleanup is needed.

    Args:
        level_id: If provided, clear only this specific level from cache.
                 If None, clear the entire cache.
    """
    if level_id is None:
        _surface_area_cache.clear()
    else:
        _surface_area_cache.pop(level_id, None)


def find_path_subgoals(
    path_nodes: List[Tuple[int, int]],
    threshold_angle_degrees: float = 45.0,
    min_segment_length: float = 24.0,
) -> List[Tuple[int, int]]:
    """
    Find subgoals from path inflection points.

    Subgoals are positions where the path direction changes significantly
    (> threshold angle). These represent key decision points where the
    agent needs to make important navigation choices.

    Uses for RL:
    - Intermediate milestone rewards without breaking PBRS guarantees
    - Hierarchical reward structure without hierarchical policies
    - Debug visualization of optimal path structure

    Args:
        path_nodes: List of (x, y) positions forming the path (in any coord space)
        threshold_angle_degrees: Minimum angle change to consider as inflection point
                                (default 45 degrees = significant turn)
        min_segment_length: Minimum distance between path segments to consider
                           (default 24px = 1 tile, filters out micro-adjustments)

    Returns:
        List of (x, y) positions representing subgoals (inflection points)
        Does NOT include start or end of path.
    """
    import math

    if len(path_nodes) < 3:
        return []  # Need at least 3 nodes to have an inflection

    subgoals = []
    threshold_radians = math.radians(threshold_angle_degrees)

    for i in range(1, len(path_nodes) - 1):
        prev_node = path_nodes[i - 1]
        curr_node = path_nodes[i]
        next_node = path_nodes[i + 1]

        # Direction vector from prev to current
        dir_in_x = curr_node[0] - prev_node[0]
        dir_in_y = curr_node[1] - prev_node[1]
        len_in = math.sqrt(dir_in_x**2 + dir_in_y**2)

        # Direction vector from current to next
        dir_out_x = next_node[0] - curr_node[0]
        dir_out_y = next_node[1] - curr_node[1]
        len_out = math.sqrt(dir_out_x**2 + dir_out_y**2)

        # Skip if segments are too short (noise filtering)
        if len_in < min_segment_length or len_out < min_segment_length:
            continue

        # Normalize directions
        dir_in_x /= len_in
        dir_in_y /= len_in
        dir_out_x /= len_out
        dir_out_y /= len_out

        # Compute angle between directions using dot product
        # dot = cos(angle), so angle = arccos(dot)
        dot_product = dir_in_x * dir_out_x + dir_in_y * dir_out_y
        # Clamp to [-1, 1] to avoid numerical issues with arccos
        dot_product = max(-1.0, min(1.0, dot_product))
        angle = math.acos(dot_product)

        # If angle exceeds threshold, this is a subgoal
        if angle > threshold_radians:
            subgoals.append(curr_node)

    return subgoals


def find_path_subgoals_from_cache(
    level_cache: Any,
    start_pos: Tuple[int, int],
    goal_id: str,
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    spatial_hash: Optional[Any] = None,
    subcell_lookup: Optional[Any] = None,
    threshold_angle_degrees: float = 45.0,
) -> List[Tuple[int, int]]:
    """
    Find subgoals by tracing the optimal path using cached next_hop data.

    This reconstructs the optimal path from current position to goal using
    the level cache's precomputed shortest path data, then finds inflection
    points where the path direction changes significantly.

    Args:
        level_cache: LevelBasedPathDistanceCache with precomputed paths
        start_pos: Starting position in world space (x, y) pixels
        goal_id: Goal identifier ("switch" or "exit")
        adjacency: Graph adjacency structure
        spatial_hash: Optional SpatialHash for O(1) node lookup
        subcell_lookup: Optional subcell lookup for node snapping
        threshold_angle_degrees: Minimum angle for inflection detection

    Returns:
        List of subgoal positions (world space coordinates)
    """
    if level_cache is None:
        return []

    # Find starting node
    start_node = find_ninja_node(
        (int(start_pos[0]), int(start_pos[1])),
        adjacency,
        spatial_hash=spatial_hash,
        subcell_lookup=subcell_lookup,
        ninja_radius=10.0,
    )

    if start_node is None:
        return []

    # Reconstruct path by following next_hop chain
    path_nodes = [start_node]
    current_node = start_node
    max_hops = 500  # Safety limit to prevent infinite loops

    for _ in range(max_hops):
        next_hop = level_cache.get_next_hop(current_node, goal_id)
        if next_hop is None:
            break  # Reached goal or no path

        path_nodes.append(next_hop)
        current_node = next_hop

        # Check if we've reached a goal node (would have no next_hop)
        if level_cache.get_next_hop(next_hop, goal_id) is None:
            break

    # Find inflection points
    subgoals_tile_space = find_path_subgoals(
        path_nodes, threshold_angle_degrees=threshold_angle_degrees
    )

    # Convert to world space
    subgoals_world = [
        (node[0] + NODE_WORLD_COORD_OFFSET, node[1] + NODE_WORLD_COORD_OFFSET)
        for node in subgoals_tile_space
    ]

    return subgoals_world


def compute_distance_to_nearest_subgoal(
    player_pos: Tuple[float, float],
    subgoals: List[Tuple[int, int]],
) -> Tuple[float, Optional[Tuple[int, int]]]:
    """
    Compute Euclidean distance to nearest subgoal.

    Used for intermediate milestone rewards - agent gets bonus when
    reaching a subgoal for the first time.

    Args:
        player_pos: Player position (x, y) in world space
        subgoals: List of subgoal positions in world space

    Returns:
        Tuple of (distance_to_nearest, nearest_subgoal)
        Returns (inf, None) if no subgoals
    """
    import math

    if not subgoals:
        return float("inf"), None

    min_dist = float("inf")
    nearest = None

    for subgoal in subgoals:
        dx = player_pos[0] - subgoal[0]
        dy = player_pos[1] - subgoal[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < min_dist:
            min_dist = dist
            nearest = subgoal

    return min_dist, nearest
