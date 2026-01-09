"""
Spatial context features for graph-free observation.

This module provides compact fixed-size spatial features that replace the full
graph observation, achieving ~99% memory reduction while preserving local
geometry understanding for platforming decisions.

Features:
1. 8×8 local tile grid (64 dims) - simplified tile categories
2. 8 nearest mines (48 dims) - 6 features per mine:
   - relative_x, relative_y: position relative to ninja
   - state: mine state (-1=deadly, 0=transitioning, 1=safe)
   - radius: normalized collision radius
   - velocity_dot_direction: dot(velocity, mine_direction) / max_speed
     > 0 = approaching mine, < 0 = moving away (Markov: current state only)
   - distance_rate: velocity projected toward mine / max_speed
     Negative = getting closer (Markov: current state only)

Total: 112 dimensions (~0.5 KB vs ~162 KB for full graph)
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from ..constants.physics_constants import TILE_PIXEL_SIZE
from ..constants.entity_types import EntityType
from ..constants import MAX_HOR_SPEED  # 3.333 px/frame

# Local tile grid constants
LOCAL_GRID_SIZE = 8  # 8×8 tiles around ninja
LOCAL_GRID_DIM = LOCAL_GRID_SIZE * LOCAL_GRID_SIZE  # 64 features
CELL_SIZE = TILE_PIXEL_SIZE  # 24 pixels

# Mine overlay constants
MAX_NEAREST_MINES = 8
# Extended features: relative_x, relative_y, state, radius, velocity_dot, distance_rate
MINE_FEATURES_PER = 6
MINE_OVERLAY_DIM = MAX_NEAREST_MINES * MINE_FEATURES_PER  # 48 features

# Total spatial context dimension
SPATIAL_CONTEXT_DIM = LOCAL_GRID_DIM + MINE_OVERLAY_DIM  # 112 features

# Tile category mapping (38 tile types -> 5 simplified categories)
# Category 0: Empty (fully traversable)
# Category 1: Solid (fully blocked)
# Category 2: Half tiles (partially blocked, simple geometry)
# Category 3: Slopes (diagonal movement surfaces)
# Category 4: Curved (quarter circles/pipes)
TILE_CATEGORIES = np.array(
    [
        0,  # Type 0: Empty
        1,  # Type 1: Full solid
        2,
        2,
        2,
        2,  # Types 2-5: Half tiles
        3,
        3,
        3,
        3,  # Types 6-9: 45-degree slopes
        4,
        4,
        4,
        4,  # Types 10-13: Quarter circles (convex)
        4,
        4,
        4,
        4,  # Types 14-17: Quarter pipes (concave)
        3,
        3,
        3,
        3,  # Types 18-21: Short mild slopes
        3,
        3,
        3,
        3,  # Types 22-25: Raised mild slopes
        3,
        3,
        3,
        3,  # Types 26-29: Short steep slopes
        3,
        3,
        3,
        3,  # Types 30-33: Raised steep slopes
        0,
        0,
        0,
        0,  # Types 34-37: Glitched tiles (treat as empty)
    ],
    dtype=np.float32,
)

# Normalize categories to [0, 1] range (divide by 4.0 = max category)
TILE_CATEGORIES_NORMALIZED = TILE_CATEGORIES / 4.0

# Mine state encoding
MINE_STATE_SAFE = 1.0  # Untoggled - safe to touch
MINE_STATE_TRANSITIONING = 0.0  # Toggling - dangerous
MINE_STATE_DEADLY = -1.0  # Toggled - instant death

# Mine radii from physics constants (normalized by max radius)
MINE_RADIUS_SAFE = 3.5 / 5.0  # 3.5px normalized
MINE_RADIUS_TRANSITIONING = 4.5 / 5.0  # 4.5px normalized
MINE_RADIUS_DEADLY = 4.0 / 5.0  # 4.0px normalized

# Module-level cache for mine overlay (position-based caching)
_mine_overlay_cache = {
    "last_pos": None,
    "result": None,
}
MINE_OVERLAY_CACHE_THRESHOLD = 12.0  # pixels - recompute when ninja moves > 12px (OPTIMIZED: doubled for better cache hit rate)


def compute_local_tile_grid(
    ninja_pos: Tuple[float, float],
    tiles: np.ndarray,
) -> np.ndarray:
    """
    Compute 8×8 local tile grid centered on ninja position.

    Returns simplified tile categories (0-4) normalized to [0, 1].
    Tiles outside level bounds are treated as solid (category 1).

    OPTIMIZED: Uses vectorized numpy operations instead of Python loops.

    Args:
        ninja_pos: Ninja position (x, y) in world coordinates
        tiles: 2D numpy array of tile types [height, width]

    Returns:
        64-dimensional numpy array with normalized tile categories [0, 1]
    """
    # Convert ninja position to tile coordinates
    ninja_x, ninja_y = ninja_pos
    ninja_tile_col = int(ninja_x // CELL_SIZE)
    ninja_tile_row = int(ninja_y // CELL_SIZE)

    # Grid is centered on ninja, so offset by half grid size
    half_grid = LOCAL_GRID_SIZE // 2

    # Calculate slice bounds
    row_start = ninja_tile_row - half_grid
    row_end = row_start + LOCAL_GRID_SIZE
    col_start = ninja_tile_col - half_grid
    col_end = col_start + LOCAL_GRID_SIZE

    height, width = tiles.shape

    # VECTORIZED: Extract tile slice with padding for out-of-bounds
    # Default to solid (category 1) for out-of-bounds tiles
    grid_2d = np.ones((LOCAL_GRID_SIZE, LOCAL_GRID_SIZE), dtype=np.int32)

    # Calculate valid slice bounds (clipped to tile array)
    valid_row_start = max(0, row_start)
    valid_row_end = min(height, row_end)
    valid_col_start = max(0, col_start)
    valid_col_end = min(width, col_end)

    # Calculate corresponding positions in grid
    grid_row_start = valid_row_start - row_start
    grid_row_end = grid_row_start + (valid_row_end - valid_row_start)
    grid_col_start = valid_col_start - col_start
    grid_col_end = grid_col_start + (valid_col_end - valid_col_start)

    # Copy valid tile data into grid
    if valid_row_end > valid_row_start and valid_col_end > valid_col_start:
        tile_slice = tiles[valid_row_start:valid_row_end, valid_col_start:valid_col_end]
        # Clamp to valid range (0-37)
        tile_slice = np.clip(tile_slice, 0, 37)
        grid_2d[grid_row_start:grid_row_end, grid_col_start:grid_col_end] = tile_slice

    # VECTORIZED: Convert tile types to normalized categories using lookup table
    # TILE_CATEGORIES_NORMALIZED is a precomputed array indexed by tile type
    grid_normalized = TILE_CATEGORIES_NORMALIZED[grid_2d]

    # Flatten to 1D array (64 elements)
    return grid_normalized.flatten().astype(np.float32)


def compute_mine_overlay(
    ninja_pos: Tuple[float, float],
    ninja_velocity: Tuple[float, float],
    entities: List[Dict[str, Any]],
    level_width: float = 1056.0,
    level_height: float = 600.0,
) -> np.ndarray:
    """
    Compute overlay of 8 nearest mines with position, state, and velocity-hazard features.

    Each mine contributes 6 features (all Markovian - depend only on current state):
    - relative_x: (mine_x - ninja_x) / level_width, normalized to [-1, 1]
    - relative_y: (mine_y - ninja_y) / level_height, normalized to [-1, 1]
    - state: -1.0 (deadly), 0.0 (transitioning), 1.0 (safe)
    - radius: normalized collision radius [0, 1]
    - velocity_dot_direction: dot(velocity, mine_direction) / max_speed
      > 0 = approaching mine, < 0 = moving away
    - distance_rate: velocity component toward mine / max_speed
      Negative = getting closer (collision risk)

    Markov Property: All features are computed from current state only.
    No history dependence - fully Markovian.

    Args:
        ninja_pos: Ninja position (x, y) in world coordinates
        ninja_velocity: Ninja velocity (vx, vy) in pixels per frame
        entities: List of entity dictionaries from level_data
        level_width: Level width for normalization
        level_height: Level height for normalization

    Returns:
        48-dimensional numpy array (8 mines × 6 features)
    """
    overlay = np.zeros(MINE_OVERLAY_DIM, dtype=np.float32)

    ninja_x, ninja_y = ninja_pos
    ninja_vx, ninja_vy = ninja_velocity

    # Collect all mines with distance
    mines_with_distance = []

    for entity in entities:
        entity_type = entity.get("type")

        # Check for toggle mines (both untoggled and toggled types)
        is_toggle_mine = entity_type == EntityType.TOGGLE_MINE
        is_toggle_mine_toggled = entity_type == EntityType.TOGGLE_MINE_TOGGLED

        if not (is_toggle_mine or is_toggle_mine_toggled):
            continue

        # Get mine position
        mine_x = entity.get("x", 0.0)
        mine_y = entity.get("y", 0.0)

        # Calculate distance to ninja
        dx = mine_x - ninja_x
        dy = mine_y - ninja_y
        distance = (dx * dx + dy * dy) ** 0.5

        # Get mine state (for TOGGLE_MINE_TOGGLED, state is always deadly)
        # Per EntityToggleMine: 0=toggled/deadly, 1=untoggled/safe, 2=toggling
        if is_toggle_mine_toggled:
            state = 0  # Deadly
        else:
            state = entity.get("state", 1)  # Default: untoggled (safe) = state 1

        # Map state to encoded value and radius
        # Per EntityToggleMine:
        #   State 0: Toggled (deadly)
        #   State 1: Untoggled (safe)
        #   State 2: Toggling (transitioning)
        if state == 1:  # Untoggled (safe)
            state_encoded = MINE_STATE_SAFE
            radius_encoded = MINE_RADIUS_SAFE
        elif state == 2:  # Toggling (transitioning)
            state_encoded = MINE_STATE_TRANSITIONING
            radius_encoded = MINE_RADIUS_TRANSITIONING
        else:  # 0: Toggled (deadly)
            state_encoded = MINE_STATE_DEADLY
            radius_encoded = MINE_RADIUS_DEADLY

        # Compute velocity-hazard features (Markovian: uses current velocity only)
        if distance > 1e-6:
            # Normalized direction from ninja to mine
            dir_x = dx / distance
            dir_y = dy / distance

            # Velocity dot direction: positive = approaching, negative = moving away
            # Normalized by MAX_HOR_SPEED for consistent scale
            velocity_dot = (ninja_vx * dir_x + ninja_vy * dir_y) / MAX_HOR_SPEED

            # Distance rate: velocity component toward mine (negative = getting closer)
            # This is the same as velocity_dot but with explicit sign for "approaching"
            # Positive velocity_dot means approaching, so distance_rate = -velocity_dot
            distance_rate = -velocity_dot
        else:
            # At mine position - velocity doesn't matter much
            velocity_dot = 0.0
            distance_rate = 0.0

        mines_with_distance.append(
            {
                "distance": distance,
                "dx": dx,
                "dy": dy,
                "state": state_encoded,
                "radius": radius_encoded,
                "velocity_dot": np.clip(velocity_dot, -1.0, 1.0),
                "distance_rate": np.clip(distance_rate, -1.0, 1.0),
            }
        )

    # Sort by distance and take nearest 8
    mines_with_distance.sort(key=lambda m: m["distance"])
    nearest_mines = mines_with_distance[:MAX_NEAREST_MINES]

    # Fill overlay with mine features
    for i, mine in enumerate(nearest_mines):
        base_idx = i * MINE_FEATURES_PER
        overlay[base_idx + 0] = np.clip(mine["dx"] / level_width, -1.0, 1.0)
        overlay[base_idx + 1] = np.clip(mine["dy"] / level_height, -1.0, 1.0)
        overlay[base_idx + 2] = mine["state"]
        overlay[base_idx + 3] = mine["radius"]
        overlay[base_idx + 4] = mine["velocity_dot"]
        overlay[base_idx + 5] = mine["distance_rate"]

    return overlay


def compute_mine_overlay_from_entities(
    ninja_pos: Tuple[float, float],
    ninja_velocity: Tuple[float, float],
    toggle_mines: List,  # EntityToggleMine objects (type 1)
    toggled_mines: List,  # EntityToggleMine objects (type 21)
    level_width: float = 1056.0,
    level_height: float = 600.0,
) -> np.ndarray:
    """
    Optimized mine overlay computation with position-based caching and direct entity access.

    This function provides two key optimizations over compute_mine_overlay():
    1. Direct entity access: Accepts mine entity objects directly, avoiding iteration
       through all entities and type filtering overhead
    2. Position-based caching: Caches results and only recomputes when ninja moves
       more than MINE_OVERLAY_CACHE_THRESHOLD (6px) from last cached position

    Since mines don't move, their relative positions change slowly with ninja movement.
    Velocity features may be slightly stale between cache hits but remain useful.

    Cache invalidation: Position change > 6px only (no mine state tracking for simplicity)

    Args:
        ninja_pos: Ninja position (x, y) in world coordinates
        ninja_velocity: Ninja velocity (vx, vy) in pixels per frame
        toggle_mines: List of EntityToggleMine objects (type 1 - start untoggled)
        toggled_mines: List of EntityToggleMine objects (type 21 - start toggled)
        level_width: Level width for normalization
        level_height: Level height for normalization

    Returns:
        48-dimensional numpy array (8 mines × 6 features)
    """
    global _mine_overlay_cache

    # Check cache validity (position threshold)
    if _mine_overlay_cache["last_pos"] is not None:
        dx = ninja_pos[0] - _mine_overlay_cache["last_pos"][0]
        dy = ninja_pos[1] - _mine_overlay_cache["last_pos"][1]
        dist_sq = dx * dx + dy * dy
        if dist_sq < MINE_OVERLAY_CACHE_THRESHOLD**2:
            # Cache hit - return cached result
            return _mine_overlay_cache["result"]

    # Cache miss - compute overlay with direct entity attribute access
    overlay = _compute_mine_overlay_impl(
        ninja_pos,
        ninja_velocity,
        toggle_mines,
        toggled_mines,
        level_width,
        level_height,
    )

    # Update cache
    _mine_overlay_cache["last_pos"] = ninja_pos
    _mine_overlay_cache["result"] = overlay

    return overlay


def _compute_mine_overlay_impl(
    ninja_pos: Tuple[float, float],
    ninja_velocity: Tuple[float, float],
    toggle_mines: List,
    toggled_mines: List,
    level_width: float,
    level_height: float,
) -> np.ndarray:
    """
    Implementation of mine overlay computation with direct entity attribute access.

    This function processes mine entity objects directly without dictionary lookups,
    providing significant performance improvement over the dictionary-based approach.

    Args:
        ninja_pos: Ninja position (x, y) in world coordinates
        ninja_velocity: Ninja velocity (vx, vy) in pixels per frame
        toggle_mines: List of EntityToggleMine objects (type 1)
        toggled_mines: List of EntityToggleMine objects (type 21)
        level_width: Level width for normalization
        level_height: Level height for normalization

    Returns:
        48-dimensional numpy array (8 mines × 6 features)
    """
    overlay = np.zeros(MINE_OVERLAY_DIM, dtype=np.float32)

    ninja_x, ninja_y = ninja_pos
    ninja_vx, ninja_vy = ninja_velocity

    # Collect all mines with distance (direct attribute access)
    mines_with_distance = []

    # Process toggle mines (type 1 - start untoggled/safe)
    for mine in toggle_mines:
        # Direct attribute access (no dictionary lookups)
        mine_x = mine.xpos
        mine_y = mine.ypos
        state = mine.state  # 0=toggled/deadly, 1=untoggled/safe, 2=toggling

        # Calculate distance to ninja
        dx = mine_x - ninja_x
        dy = mine_y - ninja_y
        distance = (dx * dx + dy * dy) ** 0.5

        # Map state to encoded value and radius
        if state == 1:  # Untoggled (safe)
            state_encoded = MINE_STATE_SAFE
            radius_encoded = MINE_RADIUS_SAFE
        elif state == 2:  # Toggling (transitioning)
            state_encoded = MINE_STATE_TRANSITIONING
            radius_encoded = MINE_RADIUS_TRANSITIONING
        else:  # 0: Toggled (deadly)
            state_encoded = MINE_STATE_DEADLY
            radius_encoded = MINE_RADIUS_DEADLY

        # Compute velocity-hazard features
        if distance > 1e-6:
            dir_x = dx / distance
            dir_y = dy / distance
            velocity_dot = (ninja_vx * dir_x + ninja_vy * dir_y) / MAX_HOR_SPEED
            distance_rate = -velocity_dot
        else:
            velocity_dot = 0.0
            distance_rate = 0.0

        mines_with_distance.append(
            {
                "distance": distance,
                "dx": dx,
                "dy": dy,
                "state": state_encoded,
                "radius": radius_encoded,
                "velocity_dot": np.clip(velocity_dot, -1.0, 1.0),
                "distance_rate": np.clip(distance_rate, -1.0, 1.0),
            }
        )

    # Process toggled mines (type 21 - start toggled/deadly)
    for mine in toggled_mines:
        # Direct attribute access
        mine_x = mine.xpos
        mine_y = mine.ypos
        # Type 21 always starts in deadly state (0)
        state = mine.state

        # Calculate distance to ninja
        dx = mine_x - ninja_x
        dy = mine_y - ninja_y
        distance = (dx * dx + dy * dy) ** 0.5

        # Map state to encoded value and radius
        if state == 1:  # Untoggled (safe)
            state_encoded = MINE_STATE_SAFE
            radius_encoded = MINE_RADIUS_SAFE
        elif state == 2:  # Toggling (transitioning)
            state_encoded = MINE_STATE_TRANSITIONING
            radius_encoded = MINE_RADIUS_TRANSITIONING
        else:  # 0: Toggled (deadly)
            state_encoded = MINE_STATE_DEADLY
            radius_encoded = MINE_RADIUS_DEADLY

        # Compute velocity-hazard features
        if distance > 1e-6:
            dir_x = dx / distance
            dir_y = dy / distance
            velocity_dot = (ninja_vx * dir_x + ninja_vy * dir_y) / MAX_HOR_SPEED
            distance_rate = -velocity_dot
        else:
            velocity_dot = 0.0
            distance_rate = 0.0

        mines_with_distance.append(
            {
                "distance": distance,
                "dx": dx,
                "dy": dy,
                "state": state_encoded,
                "radius": radius_encoded,
                "velocity_dot": np.clip(velocity_dot, -1.0, 1.0),
                "distance_rate": np.clip(distance_rate, -1.0, 1.0),
            }
        )

    # Sort by distance and take nearest 8
    mines_with_distance.sort(key=lambda m: m["distance"])
    nearest_mines = mines_with_distance[:MAX_NEAREST_MINES]

    # Fill overlay with mine features
    for i, mine in enumerate(nearest_mines):
        base_idx = i * MINE_FEATURES_PER
        overlay[base_idx + 0] = np.clip(mine["dx"] / level_width, -1.0, 1.0)
        overlay[base_idx + 1] = np.clip(mine["dy"] / level_height, -1.0, 1.0)
        overlay[base_idx + 2] = mine["state"]
        overlay[base_idx + 3] = mine["radius"]
        overlay[base_idx + 4] = mine["velocity_dot"]
        overlay[base_idx + 5] = mine["distance_rate"]

    return overlay


def reset_mine_overlay_cache():
    """Reset mine overlay cache on episode reset.

    This should be called at the start of each episode to ensure
    fresh computation with the new level configuration.
    """
    global _mine_overlay_cache
    _mine_overlay_cache["last_pos"] = None
    _mine_overlay_cache["result"] = None


def compute_spatial_context(
    ninja_pos: Tuple[float, float],
    ninja_velocity: Tuple[float, float],
    tiles: np.ndarray,
    entities: List[Dict[str, Any]],
    level_width: float = 1056.0,
    level_height: float = 600.0,
) -> np.ndarray:
    """
    Compute complete spatial context features.

    Combines:
    1. 8×8 local tile grid (64 dims)
    2. 8 nearest mines overlay (48 dims) with velocity-hazard alignment

    Total: 112 dimensions

    Markov Property: All features depend only on current state (position, velocity).
    No history dependence.

    Args:
        ninja_pos: Ninja position (x, y) in world coordinates
        ninja_velocity: Ninja velocity (vx, vy) in pixels per frame
        tiles: 2D numpy array of tile types
        entities: List of entity dictionaries
        level_width: Level width for mine normalization
        level_height: Level height for mine normalization

    Returns:
        112-dimensional numpy array of spatial context features
    """
    # Compute local tile grid
    tile_grid = compute_local_tile_grid(ninja_pos, tiles)

    # Compute mine overlay with velocity-hazard features
    mine_overlay = compute_mine_overlay(
        ninja_pos, ninja_velocity, entities, level_width, level_height
    )

    # Concatenate features
    return np.concatenate([tile_grid, mine_overlay])
