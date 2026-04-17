"""Synthetic harness maps used to populate the tile cache.

The launch / landing cache is populated by running isolated simulator
instances on tiny maps whose only non-trivial geometry is the 3x3
signature under study.  This module builds the raw map bytes ready for
``nclone_native.Simulator.load``.

Harness layout
--------------
The harness is three concentric regions centred in the 42x23 playable
area:

  * **Center 3x3** (``HARNESS_WINDOW``) - the signature being indexed.
  * **Empty buffer** (default 2 tiles thick, giving a 7x7 region) -
    all tile type 0, so the ninja can cross the boundary of the 3x3
    window without hitting a wall.  The builder samples the ninja
    state at the first frame its bounding box leaves the 3x3 window;
    the empty buffer ensures that sample reflects free-flight rather
    than the harness itself.
  * **Solid outer ring** - tile type 1 around the buffer (and filling
    the rest of the playable area), so the simulator's collision grid
    is never faced with an unbounded trajectory.

Ninja spawn is irrelevant for harness simulation (we inject state
directly via ``inject_ninja_for_value_iteration_grounded`` and similar)
so we place it at the center of the signature cell.

Coordinates
-----------
- Tiles are ``TILE_PIXEL_SIZE == 24`` px square.
- The full map grid is ``44x25`` once the border padding is applied by
  the loader.
- We intentionally place the 3x3 signature near the middle of the
  playable area so all surrounding cells are also inside the playable
  area and thus addressable from the raw map-file byte layout.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from ...constants import (
    MAP_TILE_WIDTH,
    MAP_TILE_HEIGHT,
    TILE_PIXEL_SIZE,
)

# Size of the signature window in tiles (3x3).
HARNESS_WINDOW: int = 3
# Buffer thickness (in tiles) around the signature in which tiles are
# empty so the ninja can leave the 3x3 window cleanly.
HARNESS_BUFFER: int = 2
# Total harness width in tiles: signature (3) + buffer*2 = 7.
HARNESS_SIZE: int = HARNESS_WINDOW + HARNESS_BUFFER * 2

# The 3x3 signature sits in the center of the full harness.  All coords
# below are in the raw-file tile grid (0..MAP_TILE_WIDTH-1, 0..MAP_TILE_HEIGHT-1)
# NOT in the padded 44x25 loader grid.
HARNESS_CENTER_COL: int = MAP_TILE_WIDTH // 2  # col 21 in the raw tile grid
HARNESS_CENTER_ROW: int = MAP_TILE_HEIGHT // 2  # row 11

# Raw-file layout constants.  See ``loader.rs::load_map``.
_TILE_OFFSET: int = 184         # first tile byte
_TOTAL_FILE_SIZE: int = 1234    # minimal map file length (no entities)
_SPAWN_X_INDEX: int = 1231
_SPAWN_Y_INDEX: int = 1232

# Tile id used for solid outer fill.
_RING_TILE: int = 1
# Empty tile used in the buffer region around the signature.
_EMPTY_TILE: int = 0
# Solid tile used for all cells not otherwise set (far exterior).
_BACKGROUND_TILE: int = 1


def _raw_tile_index(col: int, row: int) -> int:
    """Offset into ``map_data`` for tile at ``(col, row)`` in the 42x23 grid."""
    if not (0 <= col < MAP_TILE_WIDTH):
        raise ValueError(f"col {col} outside raw tile width {MAP_TILE_WIDTH}")
    if not (0 <= row < MAP_TILE_HEIGHT):
        raise ValueError(f"row {row} outside raw tile height {MAP_TILE_HEIGHT}")
    return _TILE_OFFSET + col + row * MAP_TILE_WIDTH


def launch_cell_origin_px(center_col: int = HARNESS_CENTER_COL,
                          center_row: int = HARNESS_CENTER_ROW) -> Tuple[float, float]:
    """Return the top-left pixel corner of the center (launch) tile.

    In the padded loader grid, the raw-file tile at ``(col, row)`` ends
    up at ``(col+1, row+1)``, so its pixel origin is
    ``((col+1) * TILE_PIXEL_SIZE, (row+1) * TILE_PIXEL_SIZE)``.
    """
    ox = float((center_col + 1) * TILE_PIXEL_SIZE)
    oy = float((center_row + 1) * TILE_PIXEL_SIZE)
    return ox, oy


def _make_base_map() -> list:
    """All-solid map with empty header/entities section."""
    return [0] * 184 + [_BACKGROUND_TILE] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT) + \
        [0] * (_TOTAL_FILE_SIZE - (184 + MAP_TILE_WIDTH * MAP_TILE_HEIGHT))


def _place_signature(
    map_data: list,
    signature: Sequence[int],
    center_col: int,
    center_row: int,
) -> None:
    """Place a 3x3 signature into ``map_data`` centred at the given cell."""
    if len(signature) != 9:
        raise ValueError(f"signature must be length 9, got {len(signature)}")
    k = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            map_data[_raw_tile_index(center_col + dc, center_row + dr)] = int(
                signature[k]
            ) & 0xFF
            k += 1


def _install_buffer_and_ring(
    map_data: list,
    center_col: int,
    center_row: int,
) -> None:
    """Install empty buffer + solid outer ring around the 3x3 signature.

    The empty buffer (thickness ``HARNESS_BUFFER``) lets the ninja leave
    the 3x3 signature cleanly; the outermost cells of the harness are
    forced to solid (tile type 1) so any trajectory that escapes the
    buffer is caught.
    """
    half = HARNESS_SIZE // 2  # == 3 when HARNESS_SIZE == 7
    inner_half = HARNESS_WINDOW // 2  # == 1
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            if abs(dr) <= inner_half and abs(dc) <= inner_half:
                continue  # inner 3x3 reserved for the signature
            is_outer = (abs(dr) == half) or (abs(dc) == half)
            tile = _RING_TILE if is_outer else _EMPTY_TILE
            map_data[_raw_tile_index(center_col + dc, center_row + dr)] = tile


def _set_spawn_at_center(
    map_data: list,
    center_col: int,
    center_row: int,
) -> None:
    """Set the ninja spawn inside the center (signature) tile.

    Spawn coordinates in the file are multiplied by 6 to get pixels, so
    one tile (24 px) == 4 file units.  Placing the spawn at the middle
    of the padded-grid center cell keeps the simulator happy even
    though we inject state directly after load.
    """
    ox_px, oy_px = launch_cell_origin_px(center_col, center_row)
    # center-of-tile in pixels, divided by 6 to get file units.
    map_data[_SPAWN_X_INDEX] = int((ox_px + TILE_PIXEL_SIZE / 2.0) / 6.0)
    map_data[_SPAWN_Y_INDEX] = int((oy_px + TILE_PIXEL_SIZE / 2.0) / 6.0)


def build_launch_harness_map(
    signature: Sequence[int],
    center_col: int = HARNESS_CENTER_COL,
    center_row: int = HARNESS_CENTER_ROW,
) -> list:
    """Build a minimal harness ``map_data`` list for a launch signature.

    The returned list is suitable for ``nclone_native.Simulator.load``.
    The 3x3 signature is placed at the requested center; the 5x5 ring
    around it (and everything else) is solid.

    Args:
        signature: 9 ``uint8`` tile ids in row-major order (NW..SE).
        center_col, center_row: Position in the raw 42x23 tile grid
            where the signature center sits.  Must leave room for the
            5x5 window - i.e. ``2 <= col <= MAP_TILE_WIDTH-3`` etc.

    Returns:
        Flat ``list[int]`` of length :data:`_TOTAL_FILE_SIZE`.
    """
    half = HARNESS_SIZE // 2
    if not half <= center_col <= MAP_TILE_WIDTH - 1 - half:
        raise ValueError(
            f"center_col {center_col} leaves no room for "
            f"{HARNESS_SIZE}x{HARNESS_SIZE} harness window"
        )
    if not half <= center_row <= MAP_TILE_HEIGHT - 1 - half:
        raise ValueError(
            f"center_row {center_row} leaves no room for "
            f"{HARNESS_SIZE}x{HARNESS_SIZE} harness window"
        )

    map_data = _make_base_map()
    _install_buffer_and_ring(map_data, center_col, center_row)
    _place_signature(map_data, signature, center_col, center_row)
    _set_spawn_at_center(map_data, center_col, center_row)
    return map_data


def build_landing_harness_map(
    signature: Sequence[int],
    center_col: int = HARNESS_CENTER_COL,
    center_row: int = HARNESS_CENTER_ROW,
) -> list:
    """Build a harness for the landing cache.

    The launch and landing harnesses share the same 5x5 layout; the
    difference is how the ninja is injected (the launch harness uses
    ``inject_ninja_for_value_iteration_grounded`` at the source
    surface, while the landing harness uses
    ``inject_ninja_for_value_iteration`` with an airborne state).
    This function is kept separate for readability; we may diverge
    later if the landing harness needs extra padding.
    """
    return build_launch_harness_map(signature, center_col, center_row)
