"""Shared utilities for graph-based reachability augmentations.

Bridges the AttractData format (map-data units, flat tile list) and the
LevelData / GraphBuilder types used by the reachability graph system.

Coordinate conventions
----------------------
AttractData entity/spawn x, y  : map-data units (1 tile = 4 units, range 0-176 for
                                  the full 44-tile-wide map)
LevelData start_position        : world-space pixels (multiply map-data units by 6)
EntityMask expects entity x, y  : world-space pixels
Graph adjacency keys            : sub-node pixel positions in tile-data space
                                  (tile-data = world − 24; 1 tile = 24 px)
Tile coordinates returned here  : (col, row) in the inner 42×23 playable grid,
                                  0-indexed
"""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np

from .base_augmentation import AttractData
from ....graph.level_data import LevelData

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_MAP_TILE_WIDTH: int = 42
_MAP_TILE_HEIGHT: int = 23
_TILE_COUNT: int = _MAP_TILE_WIDTH * _MAP_TILE_HEIGHT  # 966

# Map-data coordinate constants
# 1 tile = 4 map-data units; full map (44 tiles including 1-tile padding on
# each side) = 176 units wide, 100 units tall.
_MAP_UNITS_PER_TILE: int = 4
_FULL_MAP_WIDTH_UNITS: int = 176   # 44 tiles * 4
_FULL_MAP_HEIGHT_UNITS: int = 100  # 25 tiles * 4

# Pixel constants (must match nclone physics constants)
_PIXELS_PER_MAP_UNIT: int = 6   # map_data_units * 6 = world_space_pixels
_CELL_SIZE: int = 24             # pixels per tile; also NODE_WORLD_COORD_OFFSET


# ---------------------------------------------------------------------------
# AttractData → LevelData conversion
# ---------------------------------------------------------------------------


def attract_data_to_level_data(data: AttractData) -> LevelData:
    """Convert an AttractData dict to a LevelData object for the graph system.

    Performs the necessary coordinate-space conversions:
    - Flat 966-element tile list → (23, 42) NumPy array
    - Ninja spawn from map-data units → world-space pixels
    - Entity x/y from map-data units → world-space pixels

    Args:
        data: Decoded attract-file data as produced by NppAttractDecoder or
              AttractAugmentor._decode().

    Returns:
        LevelData ready to pass to GraphBuilder.build_graph().

    Raises:
        ValueError: If tiles list does not contain exactly 966 values.
    """
    tiles_flat = data["tiles"]
    if len(tiles_flat) != _TILE_COUNT:
        raise ValueError(
            f"attract_data_to_level_data: expected {_TILE_COUNT} tiles, "
            f"got {len(tiles_flat)}"
        )

    # Reshape flat tile list to (height=23, width=42) array.
    tiles = np.array(tiles_flat, dtype=np.int32).reshape(_MAP_TILE_HEIGHT, _MAP_TILE_WIDTH)

    # Ninja spawn in AttractData is (x, y) in map-data units.
    # Multiply by 6 to obtain world-space pixels.
    spawn_x_units, spawn_y_units = data["ninja_spawn"]
    start_position = (
        spawn_x_units * _PIXELS_PER_MAP_UNIT,
        spawn_y_units * _PIXELS_PER_MAP_UNIT,
    )

    # Convert entity coordinates from map-data units to world-space pixels.
    # EntityMask uses entity["x"] and entity["y"] as world-space pixel values.
    entities = []
    for ent in data.get("entities", []):
        converted = dict(ent)
        converted["x"] = ent["x"] * _PIXELS_PER_MAP_UNIT
        converted["y"] = ent["y"] * _PIXELS_PER_MAP_UNIT
        entities.append(converted)

    return LevelData(
        start_position=start_position,
        tiles=tiles,
        entities=entities,
        switch_states={},
    )


# ---------------------------------------------------------------------------
# Reachability helpers
# ---------------------------------------------------------------------------


def get_reachable_tiles(data: AttractData) -> Set[Tuple[int, int]]:
    """Return the set of tile coordinates reachable from the ninja spawn.

    Uses the reachability graph system to flood-fill from the spawn position and
    derives tile-level coverage from the sub-node adjacency keys.

    The returned set is in (col, row) coordinates in the inner 42×23 playable
    grid (0-indexed, tile-data space).

    Args:
        data: Decoded attract-file data.

    Returns:
        Set of (col, row) tile coordinates that have at least one reachable
        sub-node (i.e., the ninja can reach some part of that tile).

    Raises:
        ValueError: If the tile data is structurally invalid.
    """
    from ....graph.reachability.graph_builder import GraphBuilder, CELL_SIZE

    level_data = attract_data_to_level_data(data)
    builder = GraphBuilder()
    graph_data = builder.build_graph(level_data, filter_by_reachability=True)

    # With filter_by_reachability=True the adjacency keys are exactly the
    # sub-node pixel positions (in tile-data space) reachable from spawn.
    # Convert each sub-node to the tile it belongs to.
    reachable_tiles: Set[Tuple[int, int]] = {
        (px // CELL_SIZE, py // CELL_SIZE)
        for (px, py) in graph_data["adjacency"].keys()
    }
    return reachable_tiles


def get_protected_tiles(
    reachable: Set[Tuple[int, int]],
    buffer: int = 1,
) -> Set[Tuple[int, int]]:
    """Expand the reachable tile set by a safety buffer.

    Any tile within *buffer* tiles of a reachable tile may interact with the
    ninja's physics (collision geometry, slope normals, etc.) and must not be
    modified by augmentations.

    Args:
        reachable: Set of (col, row) tile coordinates that are reachable from
                   spawn (as returned by get_reachable_tiles).
        buffer:    Number of tiles to expand in each direction (default 1).
                   A value of 1 is conservative given the ninja's 10 px radius
                   versus the 24 px tile size.

    Returns:
        Set of (col, row) tiles that must remain untouched.
    """
    protected: Set[Tuple[int, int]] = set()
    for col, row in reachable:
        for dc in range(-buffer, buffer + 1):
            for dr in range(-buffer, buffer + 1):
                nc, nr = col + dc, row + dr
                if 0 <= nc < _MAP_TILE_WIDTH and 0 <= nr < _MAP_TILE_HEIGHT:
                    protected.add((nc, nr))
    return protected


def get_reachable_bounding_box(
    tile_set: Set[Tuple[int, int]],
) -> Tuple[int, int, int, int]:
    """Return the axis-aligned bounding box of a tile set.

    Args:
        tile_set: Non-empty set of (col, row) tile coordinates.

    Returns:
        Tuple (min_col, min_row, max_col, max_row) – inclusive on all sides.

    Raises:
        ValueError: If *tile_set* is empty.
    """
    if not tile_set:
        raise ValueError("get_reachable_bounding_box: tile_set is empty")
    cols = [c for c, _ in tile_set]
    rows = [r for _, r in tile_set]
    return min(cols), min(rows), max(cols), max(rows)
