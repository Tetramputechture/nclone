"""Collision validator for analytically precomputed jump arc trajectories.

Given a free-space trajectory (from analytical_physics.JumpArcTable) and the
level tile occupancy grid, determines:

  1. Whether the arc passes through any solid tile geometry.
  2. The frame at which the ninja would land (if any) before reaching the
     precomputed endpoint.

Strategy
--------
- Build a boolean occupancy grid from map_data once per level.
- For each arc frame's absolute position (ax, ay), sample the cell below the
  ninja center (ay + NINJA_RADIUS + 1) and adjacent cells.
- Detect a "floor hit" when the cell directly below the ninja center is solid
  and the ninja is moving downward (vy > 0 or vy_prev < vy).
- Detect a "blocked" arc when the ninja center cell itself is inside a solid tile.

This is an approximation — it uses tile cell occupancy rather than the full
segment-based collision geometry.  For the common case (open corridors, flat
floors), it is exact.  For complex geometry near tile corners, the simulation
fallback in edge_enumeration.py handles edge cases.

Tile layout:
  - N++ map: 44 columns × 25 rows of 24×24-pixel tiles.
  - Cell (col, row) occupies x ∈ [col*24, (col+1)*24), y ∈ [row*24, (row+1)*24).
  - Tile type 0 = empty; non-zero = solid (full block or slope).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..constants.physics_constants import NINJA_RADIUS

# N++ map dimensions
_MAP_COLS: int = 44
_MAP_ROWS: int = 25
_TILE_SIZE: float = 24.0
_INV_TILE: float = 1.0 / _TILE_SIZE

# Occupancy check: sample a small cross pattern around ninja center to be
# robust against the ninja hitting a wall vs a floor.
# Offsets are (dx, dy) relative to ninja center in pixels.
_FLOOR_CHECK_DY: float = NINJA_RADIUS + 1.0   # just below center
_WALL_CHECK_DX_L: float = -(NINJA_RADIUS + 1.0)  # left edge
_WALL_CHECK_DX_R: float = NINJA_RADIUS + 1.0    # right edge


def build_tile_occupancy(map_data: List[int]) -> np.ndarray:
    """Build a boolean occupancy grid (non-zero tile = solid).

    Args:
        map_data: Flat integer list of length MAP_COLS * MAP_ROWS (1100 ints),
            row-major order (row 0 is the topmost row).

    Returns:
        Boolean numpy array of shape (MAP_COLS, MAP_ROWS).
        ``occupancy[col, row]`` is True if the tile is solid.
    """
    arr = np.array(map_data[:_MAP_COLS * _MAP_ROWS], dtype=np.int16)
    arr = arr.reshape(_MAP_ROWS, _MAP_COLS).T  # shape (44, 25), column-major
    return arr != 0


def _cell(x: float, y: float) -> Tuple[int, int]:
    """Map pixel position to tile cell indices (col, row)."""
    col = int(x * _INV_TILE)
    row = int(y * _INV_TILE)
    return col, row


def _is_solid(occupancy: np.ndarray, col: int, row: int) -> bool:
    """Return True if (col, row) is a solid tile (or out of bounds = solid wall)."""
    if col < 0 or col >= _MAP_COLS or row < 0 or row >= _MAP_ROWS:
        return True  # treat boundary as solid
    return bool(occupancy[col, row])


def validate_arc(
    arc_frames: list,   # List[JumpArcFrame] from analytical_physics
    launch_x: float,
    launch_y: float,
    occupancy: np.ndarray,
) -> Tuple[Optional[int], Optional[Tuple[float, float, float, float]]]:
    """Validate a precomputed arc against level geometry.

    Checks each frame's absolute position for tile occupancy.  Detects:
      - **Blocked**: ninja center is inside a solid tile → arc is invalid.
      - **Floor landing**: solid tile below ninja, vy turning from negative to
        positive (peak passed) and vy > 0 (descending) → landing detected.
      - **Wall landing**: solid tile to the left or right of ninja.

    Args:
        arc_frames: List of JumpArcFrame (relative positions from
            analytical_physics.compute_jump_arc_relative).
        launch_x: Absolute x of the arc launch position.
        launch_y: Absolute y of the arc launch position.
        occupancy: Boolean tile grid from build_tile_occupancy().

    Returns:
        (valid_frame, landing_state)
            valid_frame: 1-based frame index of landing/block, or None if arc
                ends without collision (arc reached max_frames — caller should
                treat as timed-out).
            landing_state: (abs_x, abs_y, vx, vy) at landing frame, or None.
    """
    prev_vy = arc_frames[0].vy if arc_frames else 0.0

    for i, f in enumerate(arc_frames):
        abs_x = launch_x + f.rel_x
        abs_y = launch_y + f.rel_y
        frame_idx = i + 1  # 1-based

        # --- Check 1: ninja center in solid tile (blocked) ---
        c_col, c_row = _cell(abs_x, abs_y)
        if _is_solid(occupancy, c_col, c_row):
            return frame_idx, (abs_x, abs_y, f.vx, f.vy)

        # --- Check 2: floor hit (solid tile directly below center) ---
        f_col, f_row = _cell(abs_x, abs_y + _FLOOR_CHECK_DY)
        if _is_solid(occupancy, f_col, f_row) and f.vy > 0:
            # Landing: ninja is descending and floor is immediately below
            # Snap y to the top surface of the tile below
            floor_y = f_row * _TILE_SIZE - NINJA_RADIUS
            return frame_idx, (abs_x, floor_y, f.vx, 0.0)

        # --- Check 3: ceiling hit (solid tile directly above center) ---
        ceil_col, ceil_row = _cell(abs_x, abs_y - (_FLOOR_CHECK_DY))
        if _is_solid(occupancy, ceil_col, ceil_row) and f.vy < 0:
            return frame_idx, (abs_x, abs_y, f.vx, f.vy)

        # --- Check 4: wall hits ---
        l_col, l_row = _cell(abs_x + _WALL_CHECK_DX_L, abs_y)
        r_col, r_row = _cell(abs_x + _WALL_CHECK_DX_R, abs_y)
        if _is_solid(occupancy, l_col, l_row) or _is_solid(occupancy, r_col, r_row):
            return frame_idx, (abs_x, abs_y, f.vx, f.vy)

        prev_vy = f.vy

    return None, None  # arc reached max_frames without collision


def find_arc_landing(
    arc_frames: list,   # List[JumpArcFrame]
    launch_x: float,
    launch_y: float,
    occupancy: np.ndarray,
) -> Optional[Tuple[float, float, float, float, int]]:
    """Find the landing point of an arc in the level.

    Convenience wrapper around validate_arc that returns a 5-tuple compatible
    with _landing_to_node in edge_enumeration.py.

    Returns:
        (land_x, land_y, land_vx, land_vy, state_code) or None if no landing.
        state_code: 1 (Running) for floor landings, 5 (WallSlide) for wall hits.
    """
    from ..constants.physics_constants import GROUND_STATES, WALL_STATES

    _STATE_RUNNING = 1
    _STATE_WALL_SLIDING = 5

    valid_frame, landing = validate_arc(arc_frames, launch_x, launch_y, occupancy)
    if landing is None:
        return None

    lx, ly, lvx, lvy = landing
    # Determine state code from landing geometry: if |lvy| > |lvx|, probably wall
    # Simple heuristic: check what surface we landed on
    # For floor landing: vy ≈ 0 (snapped), |vy| < |vx| or vy was positive (descending)
    # For wall landing: |lvx| ≈ 0
    if abs(lvy) < 0.1 and abs(lvx) <= abs(lvy) + 0.5:
        state_code = _STATE_RUNNING
    elif abs(lvx) < 0.1:
        state_code = _STATE_WALL_SLIDING
    else:
        # Ambiguous — default to floor
        state_code = _STATE_RUNNING

    return (lx, ly, lvx, lvy, state_code)
