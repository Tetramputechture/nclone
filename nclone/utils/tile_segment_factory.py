"""
Centralized tile segment factory for consistent segment creation across MapLoader and PreciseCollision.

This module consolidates the tile-to-segment conversion logic to eliminate duplication
and ensure consistency between the main simulation and collision detection systems.

## Overview

The TileSegmentFactory centralizes the complex logic for converting tile data into
collision segments, supporting multiple input formats and ensuring exact compatibility
with the original MapLoader implementation.

## Key Features

- **Format Flexibility**: Supports dict, list, and numpy array tile formats
- **Exact Compatibility**: Replicates MapLoader logic precisely for consistency
- **Performance Optimized**: Uses efficient defaultdict for orthogonal segment processing
- **Comprehensive Coverage**: Handles all tile types (solid, diagonal, circular)
- **Constants Integration**: Uses physics constants for map dimensions (44x25 tiles)

## Usage Examples

### Basic Usage
```python
from nclone.utils.tile_segment_factory import TileSegmentFactory

# Dictionary format
tiles = {(5, 5): 1, (6, 6): 6}  # Solid and diagonal tiles
segments = TileSegmentFactory.create_segment_dictionary(tiles)

# List format
tiles = [[1, 0, 6], [0, 1, 0]]
segments = TileSegmentFactory.create_segment_dictionary(tiles)
```

### Integration with Simulator Objects
```python
# For MapLoader-style objects with orthogonal segment dictionaries
TileSegmentFactory.create_segments_for_simulator(simulator, tiles)
```

## Architecture

The factory uses a two-phase approach:
1. **Orthogonal Processing**: Builds horizontal/vertical segment dictionaries
2. **Segment Creation**: Converts processed data into final segment objects

This matches the original MapLoader architecture while providing a clean,
reusable interface for all collision systems.
"""

import math
import os
from typing import Dict, Any, List, Tuple, Union
from collections import defaultdict

from ..entities import GridSegmentLinear

#: Restore the pre-§11 split into 12 px half-faces.
_HALF_SEGMENTS = os.environ.get("NCLONE_AB_HALF_SEGMENTS") == "1"
from ..tile_definitions import (
    TILE_EDGE_GRID_TERNARY,
    TILE_GRID_EDGE_MAP,
    TILE_EDGE_OCCUPANCY,
    SIDE_SEGMENT_TEMPLATES,
    TILE_INTERIOR_LINEAR_EXTRA,
    MAX_SEGMENTS_PER_CELL,
    TILE_SEGMENT_ORTHO_MAP,
)
from ..constants.physics_constants import (
    FULL_MAP_WIDTH,
    FULL_MAP_HEIGHT,
)
from .tile_segment_cache import TileSegmentCache


#: Restore the old 0/1 edge table and its `2*outer + inner` indexing.
#: `NCLONE_AB_LEGACY_EDGES=1`. The edge grid is shared by every entity that walks
#: the cell grid -- floor guards, drones, thwumps -- so this one flag moves all
#: of them at once, which is exactly why it needs to stay A/B-able.
_LEGACY_EDGES = os.environ.get("NCLONE_AB_LEGACY_EDGES") == "1"


def _combine_edge(old: int, new: int) -> int:
    """Fold one tile's grid-edge contribution into the shared edge grid.

    `FUN_1050CE60` (`docs/decomp/TILE_SEGMENT_TABLES.md` §7) combines as

        new = (old == 2 && v == 2) ? 0 : max(old, v)

    i.e. an edge is cancelled ONLY when two **full** faces both claim it -- the
    shared face between solid neighbours, which is interior and not a wall. Two
    PARTIAL faces (`1`, a slope or a curve) meeting do *not* cancel, and that
    asymmetry is the whole reason the grid is ternary.

    The port used to store 1 for solid and test `old and new`, which cancelled
    partial pairs the binary keeps. It now carries the binary's own 0/1/2 values
    (`TILE_EDGE_GRID_TERNARY`), so the rule is written as the binary writes it.

    Before that it was a plain toggle, `new if old == 0 else 0`, which also
    cleared on `old=1, new=0` -- so every solid tile's OUTER edge was wiped by the
    empty tile next to it and the whole grid came out empty. Nothing reads it but
    the drone/floor-guard/thwump cell walks, which is why it survived so long:
    they simply never saw a wall. Round 9 §1.4.
    """
    if _LEGACY_EDGES:
        # The legacy table is 0/1, so `== 2` would never cancel anything and the
        # flag would not be a revert at all. Keep the old test with the old table.
        return 0 if (old and new) else max(old, new)
    if old == 2 and new == 2:
        return 0
    return max(old, new)


class TileSegmentFactory:
    """
    Factory class for creating tile segments from tile definitions.

    This class centralizes the logic for converting tile data into collision segments,
    ensuring consistency between MapLoader and PreciseCollision systems.
    """

    @staticmethod
    def create_segment_dictionary(
        tiles: Union[Dict, List, Any],
    ) -> Dict[Tuple[int, int], List]:
        """
        Create a complete segment dictionary from tile data.

        All N++ levels are exactly 44x25 tiles as defined in physics constants.

        Args:
            tiles: Tile data in various formats (dict, list, numpy array)

        Returns:
            Dictionary mapping (x, y) coordinates to lists of segments
        """
        # Initialize segment dictionary with empty lists for all cells
        # All N++ maps are exactly FULL_MAP_WIDTH x FULL_MAP_HEIGHT tiles
        segment_dic = {}
        for x in range(FULL_MAP_WIDTH):
            for y in range(FULL_MAP_HEIGHT):
                segment_dic[(x, y)] = []

        # Initialize temporary dictionaries for orthogonal segments
        hor_segment_dic = defaultdict(int)
        ver_segment_dic = defaultdict(int)

        if TileSegmentFactory._use_table_builder():
            TileSegmentFactory._build_segments_from_tables(
                TileSegmentFactory._tile_accessor(tiles), segment_dic
            )
            return segment_dic

        # Process tiles to build segments
        TileSegmentFactory._process_tiles_for_segments(
            tiles, segment_dic, hor_segment_dic, ver_segment_dic
        )

        # Convert orthogonal segment dictionaries to actual segments
        TileSegmentFactory._process_orthogonal_segments(
            hor_segment_dic, ver_segment_dic, segment_dic
        )

        return segment_dic

    @staticmethod
    def _use_table_builder():
        """Opt in to the binary-faithful builder with `NCLONE_TILE_SEGMENTS=tables`.

        **Off by default, and the reason is a measurement.** The table builder is
        the literal shape of `FUN_1050D6E0` and the accumulator below is not, but
        the two are *behaviourally identical*: 0 demos flipped across 1200, and 0
        across the 128-demo no-hazard subset. Our endpoints, winding and all eight
        arc centres were already correct -- only the packaging differs (one
        full-span segment vs two halves, the tile's own cell vs the neighbour's,
        and a 4-per-cell cap that never binds for a single tile), and none of that
        reaches `get_single_closest_point`.

        So it stays off until the Rust twin exists, because turning it on alone
        would break Python/Rust parity for no measured gain. The tables in
        `tile_definitions.py` are [V] and worth keeping either way.
        """
        return os.environ.get("NCLONE_TILE_SEGMENTS") == "tables"

    @staticmethod
    def _tile_accessor(tiles):
        """Normalise the several tile containers into one `(x, y) -> id` lookup."""
        if isinstance(tiles, dict):
            return lambda x, y: tiles.get((x, y), 0)
        if hasattr(tiles, "shape") and len(tiles.shape) == 2:
            h, w = tiles.shape
            return lambda x, y: int(tiles[y, x]) if 0 <= x < w and 0 <= y < h else 0

        def _seq(x, y):
            try:
                return tiles[y][x]
            except (IndexError, KeyError, TypeError):
                return 0

        return _seq

    @staticmethod
    def _process_tiles_for_segments(
        tiles, segment_dic, hor_segment_dic, ver_segment_dic
    ):
        """
        Process tile data and create segments using standardized logic.

        Args:
            tiles: Tile data in various formats
            segment_dic: Dictionary to populate with segments
            hor_segment_dic: Horizontal segment accumulator
            ver_segment_dic: Vertical segment accumulator
        """
        if isinstance(tiles, dict):
            # Dictionary format: {(x, y): tile_id}
            for (xcoord, ycoord), tile_id in tiles.items():
                if tile_id != 0:
                    TileSegmentFactory._process_single_tile(
                        tile_id,
                        xcoord,
                        ycoord,
                        segment_dic,
                        hor_segment_dic,
                        ver_segment_dic,
                    )
        elif hasattr(tiles, "shape") and len(tiles.shape) == 2:
            # NumPy array format
            height, width = tiles.shape
            for ycoord in range(height):
                for xcoord in range(width):
                    tile_id = tiles[ycoord, xcoord]
                    if tile_id != 0:
                        TileSegmentFactory._process_single_tile(
                            tile_id,
                            xcoord,
                            ycoord,
                            segment_dic,
                            hor_segment_dic,
                            ver_segment_dic,
                        )
        elif isinstance(tiles, (list, tuple)):
            # List/tuple format
            for ycoord, row in enumerate(tiles):
                if hasattr(row, "__getitem__"):
                    for xcoord, tile_id in enumerate(row):
                        if tile_id != 0:
                            TileSegmentFactory._process_single_tile(
                                tile_id,
                                xcoord,
                                ycoord,
                                segment_dic,
                                hor_segment_dic,
                                ver_segment_dic,
                            )

    @staticmethod
    def _process_single_tile(
        tile_id, xcoord, ycoord, segment_dic, hor_segment_dic, ver_segment_dic
    ):
        """
        Process a single tile and add its segments using the exact MapLoader logic.

        Args:
            tile_id: Tile type ID
            xcoord: Tile x coordinate
            ycoord: Tile y coordinate
            segment_dic: Main segment dictionary
            hor_segment_dic: Horizontal segment accumulator
            ver_segment_dic: Vertical segment accumulator
        """
        coord = (xcoord, ycoord)

        # Process orthogonal segments (grid edges and linear segments)
        if tile_id in TILE_GRID_EDGE_MAP and tile_id in TILE_SEGMENT_ORTHO_MAP:
            segment_ortho_list = TILE_SEGMENT_ORTHO_MAP[tile_id]

            # Process horizontal segments - matches MapLoader exactly
            for y_loop_idx in range(3):
                for x_loop_idx in range(2):
                    hor_key = (2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx)
                    hor_segment_dic[hor_key] += segment_ortho_list[
                        2 * y_loop_idx + x_loop_idx
                    ]

            # Process vertical segments - matches MapLoader exactly
            for x_loop_idx in range(3):
                for y_loop_idx in range(2):
                    ver_key = (2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx)
                    ver_segment_dic[ver_key] += segment_ortho_list[
                        2 * x_loop_idx + y_loop_idx + 6
                    ]

        # Process non-orthogonal segments (diagonal and circular)
        # Use TileSegmentCache for efficient template-based instantiation
        non_ortho_segments = TileSegmentCache.instantiate_segments(
            tile_id, xcoord, ycoord
        )
        segment_dic[coord].extend(non_ortho_segments)

    #: Border tile ids the loader writes around the playfield. `FUN_1051A400`
    #: overwrites the border with four dedicated ids that emit exactly one
    #: inward-facing side each, plus four corner ids that emit nothing --
    #: NOT "solid tile 1", which would also emit its outward and perpendicular
    #: edges and cancel differently against its interior neighbour.
    _BORDER_TOP, _BORDER_BOTTOM, _BORDER_LEFT, _BORDER_RIGHT = 36, 34, 35, 37
    _BORDER_CORNERS = {(0, 0): 38, (43, 0): 39, (0, 24): 40, (43, 24): 41}

    @staticmethod
    def _border_aware_tile(tiles_get, x, y, width=44, height=25):
        """Tile id at (x, y), with the border rewritten to its dedicated ids."""
        corner = TileSegmentFactory._BORDER_CORNERS.get((x, y))
        if corner is not None:
            return corner
        if y == 0:
            return TileSegmentFactory._BORDER_TOP
        if y == height - 1:
            return TileSegmentFactory._BORDER_BOTTOM
        if x == 0:
            return TileSegmentFactory._BORDER_LEFT
        if x == width - 1:
            return TileSegmentFactory._BORDER_RIGHT
        return tiles_get(x, y)

    @staticmethod
    def _build_segments_from_tables(tiles_get, segment_dic, width=44, height=25):
        """Emit collision segments the way `FUN_1050D6E0` does.

        Per cell, per side k in (left, bottom, right, top): a half-edge survives
        only if this tile has it AND the neighbour across that side does not have
        the opposing one. Both halves surviving emits a single full-span segment;
        one half emits that half; neither emits nothing. The tile's interior
        segment follows. Everything lands in the tile's OWN cell, capped at four.
        """
        # side k of this tile is checked against side (k + 2) % 4 of the neighbour
        neighbour_delta = ((-1, 0), (0, 1), (1, 0), (0, -1))
        for y in range(height):
            for x in range(width):
                tile = TileSegmentFactory._border_aware_tile(tiles_get, x, y, width, height)
                occ = TILE_EDGE_OCCUPANCY.get(tile)
                if occ is None:
                    continue
                out = []
                for k in range(4):
                    dx, dy = neighbour_delta[k]
                    nb = TileSegmentFactory._border_aware_tile(
                        tiles_get, x + dx, y + dy, width, height
                    )
                    nocc = TILE_EDGE_OCCUPANCY.get(nb, (0,) * 8)
                    # `(2k+4) & 7` / `(2k+5) & 7` -- the opposing side's two slots
                    exp_a = occ[2 * k] and not nocc[(2 * k + 4) & 7]
                    exp_b = occ[2 * k + 1] and not nocc[(2 * k + 5) & 7]
                    tpl = SIDE_SEGMENT_TEMPLATES[k]
                    if exp_a and exp_b:
                        out.append(tpl[2])
                    elif exp_a:
                        out.append(tpl[0])
                    elif exp_b:
                        out.append(tpl[1])
                ox, oy = 24 * x, 24 * y
                cell = (x, y)
                for p1, p2 in out[:MAX_SEGMENTS_PER_CELL]:
                    segment_dic[cell].append(
                        GridSegmentLinear(
                            (ox + p1[0], oy + p1[1]), (ox + p2[0], oy + p2[1])
                        )
                    )
                if len(out) >= MAX_SEGMENTS_PER_CELL:
                    continue  # the interior segment is what the cap drops
                interior = TILE_INTERIOR_LINEAR_EXTRA.get(tile)
                if interior is not None:
                    p1, p2 = interior
                    segment_dic[cell].append(
                        GridSegmentLinear(
                            (ox + p1[0], oy + p1[1]), (ox + p2[0], oy + p2[1])
                        )
                    )
                else:
                    # The diagonal and circular families. Their endpoints and all
                    # eight arc centres were diffed against the binary's table and
                    # match exactly, so the existing cache is reused as-is -- only
                    # the ORDER changes, since the game emits the interior last.
                    segment_dic[cell].extend(
                        TileSegmentCache.instantiate_segments(tile, x, y)
                    )

    @staticmethod
    def _process_orthogonal_segments(hor_segment_dic, ver_segment_dic, segment_dic):
        """
        Convert orthogonal segment dictionaries to actual GridSegmentLinear objects.

        This matches the MapLoader logic exactly where segments with opposite orientations
        cancel each other out (state == 0 means no segment).

        Args:
            hor_segment_dic: Horizontal segment accumulator
            ver_segment_dic: Vertical segment accumulator
            segment_dic: Main segment dictionary to populate
        """
        # A full tile face is ONE 24 px segment, not two 12 px halves.
        #
        # This is what §11's denormal floor normal was. `get_closest_point`
        # evaluates `a = x1 + (t / lensq) * px`, and for a position resting on
        # flat ground the two parametrisations are not numerically equal: the
        # 12 px halves return `pos.x` exactly, while the 24 px face returns it
        # one ulp low. That one ulp is the whole of `floor_normal_x`, and
        # dividing it by the post-gravity contact distance reproduces the game's
        # captured `floor_x` **bit for bit** on `10137734` frame 9, `10137755`
        # frame 20 and `10133925` frame 213, with the right sign on all five
        # captured cases. Neither 12 px winding produces anything but 0.
        #
        # The game's face is wound right-to-left for a floor -- `(72,264)`
        # to `(48,264)` -- which is what `state == -1` already gives once the
        # halves are joined.
        #
        # `NCLONE_AB_HALF_SEGMENTS=1` restores the split halves.
        half = _HALF_SEGMENTS

        def _emit(dic, horizontal):
            done = set()
            for coord, state in dic.items():
                if state == 0 or coord in done:
                    continue
                xcoord, ycoord = coord
                inner = xcoord if horizontal else ycoord
                partner = ((xcoord + (1 if inner % 2 == 0 else -1), ycoord)
                           if horizontal
                           else (xcoord, ycoord + (1 if inner % 2 == 0 else -1)))
                joined = not half and dic.get(partner) == state
                done.add(coord)
                if joined:
                    done.add(partner)
                if horizontal:
                    lo = min(xcoord, partner[0]) if joined else xcoord
                    span = 24 if joined else 12
                    cell = (math.floor(lo / 2),
                            math.floor((ycoord - 0.1 * state) / 2))
                    point1 = (12 * lo, 12 * ycoord)
                    point2 = (12 * lo + span, 12 * ycoord)
                else:
                    lo = min(ycoord, partner[1]) if joined else ycoord
                    span = 24 if joined else 12
                    cell = (math.floor((xcoord - 0.1 * state) / 2),
                            math.floor(lo / 2))
                    point1 = (12 * xcoord, 12 * lo + span)
                    point2 = (12 * xcoord, 12 * lo)
                if state == -1:
                    point1, point2 = point2, point1
                segment_dic[cell].append(GridSegmentLinear(point1, point2))

        _emit(hor_segment_dic, True)
        _emit(ver_segment_dic, False)

    @staticmethod
    def _process_tile_with_grid_edges(tile_id, xcoord, ycoord, simulator):
        """
        Process a single tile for both grid edges and segments - consolidates MapLoader logic.

        Uses the original's toggle logic for grid edges: if current value is 0, set to
        new value; if current is non-zero, reset to 0. This correctly cancels shared
        edges between adjacent tiles.

        Args:
            tile_id: Tile type ID
            xcoord: Tile x coordinate
            ycoord: Tile y coordinate
            simulator: Simulator object with all required dictionaries
        """
        # Grid edges, from the binary's own ternary tables. `outer` is the
        # 3-valued axis, `inner` the 2-valued one, and the binary reads
        # `outer*4 + inner*0xC` bytes = `outer + 3*inner` int32s -- see
        # `TILE_EDGE_GRID_TERNARY`. The port used to read `2*outer + inner` off a
        # 0/1 table, which is wrong twice over.
        if _LEGACY_EDGES:
            if tile_id in TILE_GRID_EDGE_MAP:
                grid_edge_list = TILE_GRID_EDGE_MAP[tile_id]
                for y_loop_idx in range(3):
                    for x_loop_idx in range(2):
                        key = (2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx)
                        new_val = grid_edge_list[2 * y_loop_idx + x_loop_idx]
                        simulator.hor_grid_edge_dic[key] = _combine_edge(
                            simulator.hor_grid_edge_dic[key], new_val
                        )
                for x_loop_idx in range(3):
                    for y_loop_idx in range(2):
                        key = (2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx)
                        new_val = grid_edge_list[2 * x_loop_idx + y_loop_idx + 6]
                        simulator.ver_grid_edge_dic[key] = _combine_edge(
                            simulator.ver_grid_edge_dic[key], new_val
                        )
        elif tile_id in TILE_EDGE_GRID_TERNARY:
            hor_row, ver_row = TILE_EDGE_GRID_TERNARY[tile_id]

            # Horizontal edges: 3 rows (top / interior / bottom) x 2 columns.
            for y_loop_idx in range(3):
                for x_loop_idx in range(2):
                    key = (2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx)
                    new_val = hor_row[y_loop_idx + 3 * x_loop_idx]
                    simulator.hor_grid_edge_dic[key] = _combine_edge(
                        simulator.hor_grid_edge_dic[key], new_val
                    )

            # Vertical edges: 3 columns (left / interior / right) x 2 rows.
            for x_loop_idx in range(3):
                for y_loop_idx in range(2):
                    key = (2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx)
                    new_val = ver_row[x_loop_idx + 3 * y_loop_idx]
                    simulator.ver_grid_edge_dic[key] = _combine_edge(
                        simulator.ver_grid_edge_dic[key], new_val
                    )

        # Process segments using existing logic
        TileSegmentFactory._process_single_tile(
            tile_id,
            xcoord,
            ycoord,
            simulator.segment_dic,
            simulator.hor_segment_dic,
            simulator.ver_segment_dic,
        )

    @staticmethod
    def create_segments_for_simulator(simulator, tiles, process_grid_edges=False):
        """
        Create segments directly in a simulator object (for MapLoader compatibility).

        When process_grid_edges=True, ALL tiles (including empty tile_id=0) are processed
        to match the original's grid edge cancellation behavior where empty tiles clear
        previously set edges.

        Args:
            simulator: Simulator object with segment_dic, hor_segment_dic, ver_segment_dic
            tiles: Tile data dictionary {(x, y): tile_id}
            process_grid_edges: If True, also populate grid edge dictionaries for MapLoader
        """
        # Process each tile using the centralized logic
        for (xcoord, ycoord), tile_id in tiles.items():
            if process_grid_edges:
                # Process ALL tiles (including empty) for grid edge cancellation
                TileSegmentFactory._process_tile_with_grid_edges(
                    tile_id, xcoord, ycoord, simulator
                )
            elif tile_id != 0:
                TileSegmentFactory._process_single_tile(
                    tile_id,
                    xcoord,
                    ycoord,
                    simulator.segment_dic,
                    simulator.hor_segment_dic,
                    simulator.ver_segment_dic,
                )

        # The table builder is the literal shape of `FUN_1050D6E0`. Until now the
        # opt-in only reached `build_segments`, which the replay path does not
        # use -- `map_loader` comes through here -- so setting the variable
        # appeared to do nothing at all.
        #
        # The grid-edge pass above still runs either way: §10's entity walk reads
        # `hor_grid_edge_dic` / `ver_grid_edge_dic`, and only the *segments* are
        # rebuilt. The per-tile loop has already appended its own segments, so
        # they are cleared first rather than added to.
        if TileSegmentFactory._use_table_builder():
            for cell in simulator.segment_dic:
                simulator.segment_dic[cell] = []
            TileSegmentFactory._build_segments_from_tables(
                TileSegmentFactory._tile_accessor(tiles), simulator.segment_dic
            )
            return

        # Convert orthogonal segments to actual segments
        TileSegmentFactory._process_orthogonal_segments(
            simulator.hor_segment_dic, simulator.ver_segment_dic, simulator.segment_dic
        )
