"""Analytical ballistic propagator.

For the airborne middle of a trajectory, the ninja's path is a pure
function of its launch state ``(vx, vy, hold_remaining, h_during, h_after)``
and gravity / drag.  Tile geometry along the flight path matters only as
a *collision test*: we don't need to simulate the level, we just need to
know when the analytical arc first enters a solid tile cell.

``propagate_and_sweep`` combines:
- :func:`analytical_physics.compute_jump_arc_relative` to produce the
  frame-by-frame ballistic trajectory in closed form (~6 us for 60 frames
  in NumPy-less scalar code).
- A cheap per-frame tile occupancy lookup against a level tile dict to
  detect the first solid-tile hit.

Returns
-------
The return type is a tagged union (``BallisticResult``) with three
variants:

* ``Landed(cell, frame, pos, vel)`` - the arc first intersected a solid
  tile at ``cell`` on ``frame``.  The caller feeds the (cell, pos, vel)
  into the landing cache to resolve the final surface node.
* ``Timeout(frames_used)`` - arc ran to the frame budget without hitting
  anything.  The caller may discard the arc or fall back to a full sim.
* ``Died`` - the ninja hit a lethal tile or entered a mine-kill region.

Reuses ``arc_collision_validator.build_tile_occupancy`` for the occupancy
grid so we share the existing tile-dict convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from ...constants import NINJA_RADIUS, TILE_PIXEL_SIZE
from ..analytical_physics import compute_jump_arc_relative

# Re-export for callers that want to build occupancy without importing
# arc_collision_validator themselves.
from ..arc_collision_validator import build_tile_occupancy  # noqa: F401

_INV_TILE: float = 1.0 / TILE_PIXEL_SIZE
_FLOOR_CHECK_DY: float = NINJA_RADIUS + 1.0
_WALL_CHECK_DX_L: float = -(NINJA_RADIUS + 1.0)
_WALL_CHECK_DX_R: float = NINJA_RADIUS + 1.0


# ---------------------------------------------------------------------------
# Tile-dict occupancy adapter
# ---------------------------------------------------------------------------
def tile_dict_is_solid(
    tile_dict: Dict[Tuple[int, int], int],
    col: int,
    row: int,
    n_cols: int,
    n_rows: int,
) -> bool:
    """True if ``(col, row)`` is solid in the tile dict.

    ``tile_dict`` maps ``(col, row) -> tile_type`` in the padded 44x25
    loader coordinate space.  Missing keys mean empty (tile type 0).
    Out-of-bounds cells are treated as solid to match the simulator's
    border behaviour.
    """
    if col < 0 or col >= n_cols or row < 0 or row >= n_rows:
        return True
    tile = tile_dict.get((col, row), 0)
    return tile != 0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class Landed:
    """Arc hit a solid tile at ``cell`` on ``frame``."""
    cell: Tuple[int, int]       # (col, row) padded-grid coords
    frame: int                  # frames elapsed since launch (1-based)
    pos: Tuple[float, float]    # ninja centre pos at hit
    vel: Tuple[float, float]    # ninja velocity at hit
    hit_kind: str               # 'floor' | 'ceiling' | 'wall' | 'center'


@dataclass
class Timeout:
    """Arc exhausted its frame budget without any collision."""
    frames_used: int
    end_pos: Tuple[float, float]
    end_vel: Tuple[float, float]


@dataclass
class Died:
    """Arc entered a lethal region (reserved; currently unused in this phase)."""
    frame: int
    pos: Tuple[float, float]


BallisticResult = Union[Landed, Timeout, Died]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def propagate_and_sweep(
    launch_pos: Tuple[float, float],
    launch_vel: Tuple[float, float],
    hold_remaining: int,
    h_during: int,
    h_after: int,
    tile_dict: Dict[Tuple[int, int], int],
    n_cols: int,
    n_rows: int,
    max_frames: int = 300,
) -> BallisticResult:
    """Propagate a ballistic trajectory until it first intersects a solid tile.

    Args:
        launch_pos: ``(x, y)`` absolute ninja-centre pixel coordinates at
            the start of the ballistic phase.
        launch_vel: ``(vx, vy)`` velocity at the start of the ballistic
            phase (i.e. the handoff state from the launch cache).
        hold_remaining: Remaining jump-hold frames (``0`` if jump already
            released or if this is a non-jump handoff).
        h_during: Horizontal input during the remaining hold phase.
        h_after: Horizontal input after hold release.
        tile_dict: Padded-grid ``{(col, row): tile_type}`` dict for the
            level.
        n_cols, n_rows: Padded-grid dimensions (typically 44, 25).
        max_frames: Hard cap on the ballistic arc length.

    Returns:
        A :class:`Landed` / :class:`Timeout` / :class:`Died` instance.
    """
    launch_vx, launch_vy = launch_vel
    arc = compute_jump_arc_relative(
        launch_vx=launch_vx,
        launch_vy=launch_vy,
        hold_frames=max(0, int(hold_remaining)),
        h_during=int(h_during),
        h_after=int(h_after),
        max_frames=int(max_frames),
    )

    lx, ly = launch_pos
    prev_vy = launch_vy

    for i, f in enumerate(arc):
        frame_idx = i + 1  # 1-based
        abs_x = lx + f.rel_x
        abs_y = ly + f.rel_y

        # --- Center-in-solid ---
        c_col = int(math.floor(abs_x * _INV_TILE))
        c_row = int(math.floor(abs_y * _INV_TILE))
        if tile_dict_is_solid(tile_dict, c_col, c_row, n_cols, n_rows):
            return Landed(
                cell=(c_col, c_row), frame=frame_idx,
                pos=(abs_x, abs_y), vel=(f.vx, f.vy), hit_kind="center",
            )

        # --- Floor hit: cell below centre solid, descending ---
        f_col = int(math.floor(abs_x * _INV_TILE))
        f_row = int(math.floor((abs_y + _FLOOR_CHECK_DY) * _INV_TILE))
        if f.vy > 0 and tile_dict_is_solid(tile_dict, f_col, f_row, n_cols, n_rows):
            floor_y = f_row * TILE_PIXEL_SIZE - NINJA_RADIUS
            return Landed(
                cell=(f_col, f_row), frame=frame_idx,
                pos=(abs_x, floor_y), vel=(f.vx, 0.0), hit_kind="floor",
            )

        # --- Ceiling hit: cell above centre solid, ascending ---
        if f.vy < 0:
            ce_col = int(math.floor(abs_x * _INV_TILE))
            ce_row = int(math.floor((abs_y - _FLOOR_CHECK_DY) * _INV_TILE))
            if tile_dict_is_solid(tile_dict, ce_col, ce_row, n_cols, n_rows):
                return Landed(
                    cell=(ce_col, ce_row), frame=frame_idx,
                    pos=(abs_x, abs_y), vel=(f.vx, f.vy), hit_kind="ceiling",
                )

        # --- Wall hits (left / right) ---
        lx_col = int(math.floor((abs_x + _WALL_CHECK_DX_L) * _INV_TILE))
        lx_row = int(math.floor(abs_y * _INV_TILE))
        rx_col = int(math.floor((abs_x + _WALL_CHECK_DX_R) * _INV_TILE))
        rx_row = int(math.floor(abs_y * _INV_TILE))
        if tile_dict_is_solid(tile_dict, lx_col, lx_row, n_cols, n_rows):
            return Landed(
                cell=(lx_col, lx_row), frame=frame_idx,
                pos=(abs_x, abs_y), vel=(f.vx, f.vy), hit_kind="wall",
            )
        if tile_dict_is_solid(tile_dict, rx_col, rx_row, n_cols, n_rows):
            return Landed(
                cell=(rx_col, rx_row), frame=frame_idx,
                pos=(abs_x, abs_y), vel=(f.vx, f.vy), hit_kind="wall",
            )

        prev_vy = f.vy

    # Arc reached the frame budget with no collision.
    if arc:
        last = arc[-1]
        return Timeout(
            frames_used=len(arc),
            end_pos=(lx + last.rel_x, ly + last.rel_y),
            end_vel=(last.vx, last.vy),
        )
    return Timeout(frames_used=0, end_pos=launch_pos, end_vel=launch_vel)


# ---------------------------------------------------------------------------
# Tile dict helper (build from map_data if needed)
# ---------------------------------------------------------------------------

def tile_dict_from_map_data(map_data) -> Dict[Tuple[int, int], int]:
    """Build the padded 44x25 ``{(col, row): tile_type}`` dict from raw bytes.

    Matches the loader's ``load_tiles`` routine - indices 184..1150 hold
    tile data at ``(col+1, row+1)`` in the padded grid, and the outer
    border is tile type 1.
    """
    from ...constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT, FULL_MAP_WIDTH, FULL_MAP_HEIGHT

    td: Dict[Tuple[int, int], int] = {}
    # Border
    for x in range(FULL_MAP_WIDTH):
        td[(x, 0)] = 1
        td[(x, FULL_MAP_HEIGHT - 1)] = 1
    for y in range(FULL_MAP_HEIGHT):
        td[(0, y)] = 1
        td[(FULL_MAP_WIDTH - 1, y)] = 1
    # Playable area
    for x in range(MAP_TILE_WIDTH):
        for y in range(MAP_TILE_HEIGHT):
            t = int(map_data[184 + x + y * MAP_TILE_WIDTH]) & 0xFF
            if t != 0:
                td[(x + 1, y + 1)] = t
    return td
