"""
This module contains the definitions for tile properties, including grid edges,
orthogonal segments, diagonal segments, and circular segments. These definitions
are used by the simulator to construct the level geometry from map data.

IMPORTANT: Tile types 34-37 are glitched/unused tiles and are treated as empty
(no collision). They are included in the definitions for completeness but should
not generate any collision segments.

COMPREHENSIVE TILE TYPE DOCUMENTATION:

Tile Types 0-1: Basic tiles
- 0: Empty tile (no collision)
- 1: Full solid tile (complete collision)

Tile Types 2-5: Half tiles
- 2: Top half solid (bottom half traversable)
- 3: Right half solid (left half traversable)
- 4: Bottom half solid (top half traversable)
- 5: Left half solid (right half traversable)

Tile Types 6-9: 45-degree slopes
- 6: Slope from top-left to bottom-right (\\ shape)
- 7: Slope from top-right to bottom-left (/ shape)
- 8: Slope from bottom-left to top-right (/ shape, inverted)
- 9: Slope from bottom-right to top-left (\\ shape, inverted)

Tile Types 10-13: Quarter circles (convex corners)
- 10: Bottom-right quarter circle (solid in bottom-right corner)
- 11: Bottom-left quarter circle (solid in bottom-left corner)
- 12: Top-left quarter circle (solid in top-left corner)
- 13: Top-right quarter circle (solid in top-right corner)

Tile Types 14-17: Quarter pipes (concave corners)
- 14: Top-left quarter pipe (hollow in top-left, solid elsewhere)
- 15: Top-right quarter pipe (hollow in top-right, solid elsewhere)
- 16: Bottom-right quarter pipe (hollow in bottom-right, solid elsewhere)
- 17: Bottom-left quarter pipe (hollow in bottom-left, solid elsewhere)

Tile Types 18-21: Short mild slopes (gentle slopes, short rise)
- 18: Mild slope up-left, short rise from left edge to middle
- 19: Mild slope up-right, short rise from middle to right edge
- 20: Mild slope down-right, short drop from middle to right edge
- 21: Mild slope down-left, short drop from left edge to middle

Tile Types 22-25: Raised mild slopes (gentle slopes, raised platform)
- 22: Raised mild slope, platform on left with gentle rise
- 23: Raised mild slope, platform on right with gentle rise
- 24: Raised mild slope, platform on right with gentle drop
- 25: Raised mild slope, platform on left with gentle drop

Tile Types 26-29: Short steep slopes (steep slopes, short rise)
- 26: Steep slope up-left, sharp rise from left edge to middle
- 27: Steep slope up-right, sharp rise from middle to right edge
- 28: Steep slope down-right, sharp drop from middle to right edge
- 29: Steep slope down-left, sharp drop from left edge to middle

Tile Types 30-33: Raised steep slopes (steep slopes, raised platform)
- 30: Raised steep slope, platform on left with sharp rise
- 31: Raised steep slope, platform on right with sharp rise
- 32: Raised steep slope, platform on right with sharp drop
- 33: Raised steep slope, platform on left with sharp drop

Tile Types 34-37: Glitched tiles (TREATED AS EMPTY - NO COLLISION)
- 34-37: Various glitched configurations - included for completeness but
  collision system treats these as tile type 0 (empty) for performance

DIAGONAL SEGMENT COORDINATES:
The TILE_SEGMENT_DIAG_MAP defines diagonal lines within tiles using two points:
- Point coordinates are relative to tile corner (0,0) = top-left, (24,24) = bottom-right
- Lines connect the two specified points to create slopes and ramps

CIRCULAR SEGMENT PARAMETERS:
The TILE_SEGMENT_CIRCULAR_MAP defines circular arcs using:
- Center point: (x, y) relative to tile corner
- Quadrant: (dx, dy) direction vector indicating which quadrant
- Convex flag: True for solid corners, False for hollow pipes
"""

# This is a dictionary mapping every tile id to the grid edges it contains.
# The first 6 values represent horizontal half-tile edges, from left to right then top to bottom.
# The last 6 values represent vertical half-tile edges, from top to bottom then left to right.
# 1 if there is a grid edge, 0 otherwise.
TILE_GRID_EDGE_MAP = {
    0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    1: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],  # 0-1: Empty and full tiles
    # 2-5: Half tiles
    2: [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    3: [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
    4: [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
    5: [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0],
    # 6-9: 45 degree slopes
    6: [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    7: [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1],
    8: [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    9: [1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    # 10-13: Quarter moons
    10: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    11: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    12: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    13: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    # 14-17: Quarter pipes
    14: [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    15: [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1],
    16: [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    17: [1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    # 18-21: Short mild slopes
    18: [1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    19: [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    20: [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
    21: [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
    # 22-25: Raised mild slopes
    22: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    23: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    24: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    25: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    # 26-29: Short steep slopes
    26: [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0],
    27: [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
    28: [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
    29: [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0],
    # 30-33: Raised steep slopes
    30: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    31: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    32: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    33: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    # 34-37: Glitched tiles
    34: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    35: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    36: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    37: [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
}

# This is a dictionary mapping every tile id to the orthogonal linear segments it contains,
# same order as grid edges.
# 0 if no segment, -1 if normal facing left or up, 1 if normal right or down.
TILE_SEGMENT_ORTHO_MAP = {
    0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    1: [-1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 1, 1],  # 0-1: Empty and full tiles
    # 2-5: Half tiles
    2: [-1, -1, 1, 1, 0, 0, -1, 0, 0, 0, 1, 0],
    3: [0, -1, 0, 0, 0, 1, 0, 0, -1, -1, 1, 1],
    4: [0, 0, -1, -1, 1, 1, 0, -1, 0, 0, 0, 1],
    5: [-1, 0, 0, 0, 1, 0, -1, -1, 1, 1, 0, 0],
    # 6-9: 45 degree slopes
    6: [-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0],
    7: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    8: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    9: [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
    # 10-13: Quarter moons
    10: [-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0],
    11: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    12: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    13: [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
    # 14-17: Quarter pipes
    14: [-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0],
    15: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    16: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    17: [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
    # 18-21: Short mild slopes
    18: [-1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    19: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    20: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    21: [0, 0, 0, 0, 1, 1, 0, -1, 0, 0, 0, 0],
    # 22-25: Raised mild slopes
    22: [-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 1, 0],
    23: [-1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1],
    24: [0, 0, 0, 0, 1, 1, 0, -1, 0, 0, 1, 1],
    25: [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 1],
    # 26-29: Short steep slopes
    26: [-1, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0],
    27: [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    28: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    29: [0, 0, 0, 0, 1, 0, -1, -1, 0, 0, 0, 0],
    # 30-33: Raised steep slopes
    30: [-1, -1, 0, 0, 1, 0, -1, -1, 0, 0, 0, 0],
    31: [-1, -1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    32: [0, -1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    33: [-1, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
    # 34-37: Glitched tiles (not solid or used in the game)
    34: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    35: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    36: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    37: [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0],
}

# This is a dictionary mapping every tile id to the diagonal linear segment it contains.
# Segments are defined by two sets of point that need to be added to the position inside the grid.
TILE_SEGMENT_DIAG_MAP = {
    6: ((0, 24), (24, 0)),
    7: ((0, 0), (24, 24)),
    8: ((24, 0), (0, 24)),
    9: ((24, 24), (0, 0)),
    18: ((0, 12), (24, 0)),
    19: ((0, 0), (24, 12)),
    20: ((24, 12), (0, 24)),
    21: ((24, 24), (0, 12)),
    22: ((0, 24), (24, 12)),
    23: ((0, 12), (24, 24)),
    24: ((24, 0), (0, 12)),
    25: ((24, 12), (0, 0)),
    26: ((0, 24), (12, 0)),
    27: ((12, 0), (24, 24)),
    28: ((24, 0), (12, 24)),
    29: ((12, 24), (0, 0)),
    30: ((12, 24), (24, 0)),
    31: ((0, 0), (12, 24)),
    32: ((12, 0), (0, 24)),
    33: ((24, 24), (12, 0)),
}

# This is a dictionary mapping every tile id to the circular segment it contains.
# Segments defined by their center point and the quadrant.
TILE_SEGMENT_CIRCULAR_MAP = {
    10: ((0, 0), (1, 1), True),
    11: ((24, 0), (-1, 1), True),
    12: ((24, 24), (-1, -1), True),
    13: ((0, 24), (1, -1), True),
    14: ((24, 24), (-1, -1), False),
    15: ((0, 24), (1, -1), False),
    16: ((0, 0), (1, 1), False),
    17: ((24, 0), (-1, 1), False),
}


def get_tile_definitions():
    """Return a mapping of tile definition tables used by navigation/surface parsing."""
    return {
        "TILE_GRID_EDGE_MAP": TILE_GRID_EDGE_MAP,
        "TILE_SEGMENT_ORTHO_MAP": TILE_SEGMENT_ORTHO_MAP,
        "TILE_SEGMENT_DIAG_MAP": TILE_SEGMENT_DIAG_MAP,
        "TILE_SEGMENT_CIRCULAR_MAP": TILE_SEGMENT_CIRCULAR_MAP,
    }


# ---------------------------------------------------------------------------
# The binary's own tile->segment tables (round 7, `docs/decomp/TILE_SEGMENT_TABLES.md`).
#
# These describe the SAME geometry as TILE_SEGMENT_ORTHO_MAP above, but in the
# shape the game actually builds it, which differs in three ways that change the
# candidate set `get_single_closest_point` sees:
#
#   * a side with both halves exposed emits ONE full-span 24px segment, where the
#     ortho map emits two 12px halves;
#   * every segment lands in the tile's OWN cell, where the ortho map assigns it
#     to the neighbouring cell via the `-0.1 * state` nudge;
#   * a cell holds at most FOUR segments; the fifth is silently dropped.
#
# Endpoints, winding and all eight arc centres were verified identical to the
# port's tables before these were added, so this is a restructure, not a
# regeometry.
# ---------------------------------------------------------------------------

#: `0x10585F40`, 8 bytes per tile: which half-edges of each side are solid.
#: Slot order `left(0,1) | bottom(2,3) | right(4,5) | top(6,7)`, in the winding
#: order of the templates below.
TILE_EDGE_OCCUPANCY = {
    0: (0, 0, 0, 0, 0, 0, 0, 0), 1: (1, 1, 1, 1, 1, 1, 1, 1),
    2: (1, 0, 0, 0, 1, 0, 1, 1), 3: (0, 0, 0, 1, 1, 1, 0, 1),
    4: (0, 1, 1, 1, 0, 1, 0, 0), 5: (1, 1, 1, 0, 0, 0, 1, 0),
    6: (1, 1, 0, 0, 0, 0, 1, 1), 7: (0, 0, 0, 0, 1, 1, 1, 1),
    8: (0, 0, 1, 1, 1, 1, 0, 0), 9: (1, 1, 1, 1, 0, 0, 0, 0),
    10: (1, 1, 0, 0, 0, 0, 1, 1), 11: (0, 0, 0, 0, 1, 1, 1, 1),
    12: (0, 0, 1, 1, 1, 1, 0, 0), 13: (1, 1, 1, 1, 0, 0, 0, 0),
    14: (1, 1, 0, 0, 0, 0, 1, 1), 15: (0, 0, 0, 0, 1, 1, 1, 1),
    16: (0, 0, 1, 1, 1, 1, 0, 0), 17: (1, 1, 1, 1, 0, 0, 0, 0),
    18: (1, 0, 0, 0, 0, 0, 1, 1), 19: (0, 0, 0, 0, 1, 0, 1, 1),
    20: (0, 0, 1, 1, 0, 1, 0, 0), 21: (0, 1, 1, 1, 0, 0, 0, 0),
    22: (1, 1, 0, 0, 1, 0, 1, 1), 23: (1, 0, 0, 0, 1, 1, 1, 1),
    24: (0, 1, 1, 1, 1, 1, 0, 0), 25: (1, 1, 1, 1, 0, 1, 0, 0),
    26: (1, 1, 0, 0, 0, 0, 1, 0), 27: (0, 0, 0, 0, 1, 1, 0, 1),
    28: (0, 0, 0, 1, 1, 1, 0, 0), 29: (1, 1, 1, 0, 0, 0, 0, 0),
    30: (1, 1, 1, 0, 0, 0, 1, 1), 31: (0, 0, 0, 1, 1, 1, 1, 1),
    32: (0, 0, 1, 1, 1, 1, 0, 1), 33: (1, 1, 1, 1, 0, 0, 1, 0),
    # 34-37 are the map border, each emitting exactly one inward-facing side.
    34: (0, 0, 0, 0, 0, 0, 1, 1), 35: (0, 0, 0, 0, 1, 1, 0, 0),
    36: (0, 0, 1, 1, 0, 0, 0, 0), 37: (1, 1, 0, 0, 0, 0, 0, 0),
    # 38-41 are the four border corners and emit nothing at all.
    38: (0,) * 8, 39: (0,) * 8, 40: (0,) * 8, 41: (0,) * 8,
}

#: `0x107BAFA0`, tile-independent: per side, the (halfA, halfB, full) templates as
#: pixel offsets from the cell's top-left corner. Winding runs left-down,
#: bottom-right, right-up, top-left, so the outward normal of `d = p2 - p1` is
#: `(-dy, dx)`.
SIDE_SEGMENT_TEMPLATES = (
    (((0, 0), (0, 12)), ((0, 12), (0, 24)), ((0, 0), (0, 24))),          # left
    (((0, 24), (12, 24)), ((12, 24), (24, 24)), ((0, 24), (24, 24))),    # bottom
    (((24, 12), (24, 0)), ((24, 24), (24, 12)), ((24, 24), (24, 0))),    # right
    (((12, 0), (0, 0)), ((24, 0), (12, 0)), ((24, 0), (0, 0))),          # top
)

#: `0x107BB240`, the per-tile interior segment, for the tiles whose interior is a
#: straight line. The diagonal families already live in `TILE_SEGMENT_DIAG_MAP`
#: and were verified to match; only the four half-tiles were missing, because the
#: port carried their flat face in the ortho map instead.
TILE_INTERIOR_LINEAR_EXTRA = {
    2: ((0, 12), (24, 12)),
    3: ((12, 0), (12, 24)),
    4: ((24, 12), (0, 12)),
    5: ((12, 24), (12, 0)),
}

#: `FUN_1050D640` refuses a cell's fifth segment. Emission order is
#: left, bottom, right, top, interior, so the interior is what gets dropped.
MAX_SEGMENTS_PER_CELL = 4
#: The binary's edge-grid tables, `0x10585890` (horizontal edges, writes
#: array 2 at `sim+0x3248C0`) and `0x105854A0` (vertical edges, array 1 at
#: `sim+0x320400`), verbatim from `docs/decomp/TILE_TABLES_2.txt` §C.
#:
#: **Ternary, and that is the point.** `0` no edge, `1` a PARTIAL face (a
#: slope or a curve), `2` a full flat face. `TILE_GRID_EDGE_MAP` above
#: collapses 1 and 2 into 1, which is fine for "is anything there" but
#: destroys the only distinction the floor guard turns on: `FUN_1040B900`
#: walks onto a cell only when its floor reads exactly `2`.
#:
#: Each row is the binary's 6 int32 consumed as
#: `base + tile*0x18 + outer*4 + inner*0xC`, i.e. **`index = outer + 3*inner`**
#: -- `outer` is the 3-valued axis (y for horizontal edges, x for vertical),
#: `inner` the 2-valued one. `TILE_SEGMENT_TABLES.md` §8 lists this mapping as
#: [U]; tile 1 settles it. A full solid block reads `2 0 2 2 0 2`, which under
#: this indexing is "both outer faces solid, interior empty" on both axes --
#: the only reading that describes a solid block.
TILE_EDGE_GRID_TERNARY = {
     0: ([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
     1: ([2, 0, 2, 2, 0, 2], [2, 0, 2, 2, 0, 2]),
     2: ([2, 2, 0, 2, 2, 0], [2, 0, 2, 0, 0, 0]),
     3: ([0, 0, 0, 2, 0, 2], [0, 2, 2, 0, 2, 2]),
     4: ([0, 2, 2, 0, 2, 2], [0, 0, 0, 2, 0, 2]),
     5: ([2, 0, 2, 0, 0, 0], [2, 2, 0, 2, 2, 0]),
     6: ([2, 0, 1, 2, 1, 0], [2, 0, 1, 2, 1, 0]),
     7: ([2, 1, 0, 2, 0, 1], [1, 0, 2, 0, 1, 2]),
     8: ([0, 1, 2, 1, 0, 2], [0, 1, 2, 1, 0, 2]),
     9: ([1, 0, 2, 0, 1, 2], [2, 1, 0, 2, 0, 1]),
    10: ([2, 0, 1, 2, 0, 1], [2, 0, 1, 2, 0, 1]),
    11: ([2, 0, 1, 2, 0, 1], [1, 0, 2, 1, 0, 2]),
    12: ([1, 0, 2, 1, 0, 2], [1, 0, 2, 1, 0, 2]),
    13: ([1, 0, 2, 1, 0, 2], [2, 0, 1, 2, 0, 1]),
    14: ([2, 0, 1, 2, 1, 0], [2, 0, 1, 2, 1, 0]),
    15: ([2, 1, 0, 2, 0, 1], [1, 0, 2, 0, 1, 2]),
    16: ([0, 1, 2, 1, 0, 2], [0, 1, 2, 1, 0, 2]),
    17: ([1, 0, 2, 0, 1, 2], [2, 1, 0, 2, 0, 1]),
    18: ([2, 1, 0, 2, 1, 0], [2, 0, 1, 0, 0, 0]),
    19: ([2, 1, 0, 2, 1, 0], [1, 0, 2, 0, 0, 0]),
    20: ([0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 0, 2]),
    21: ([0, 1, 2, 0, 1, 2], [0, 0, 0, 2, 0, 1]),
    22: ([2, 0, 1, 2, 0, 1], [2, 0, 2, 2, 0, 1]),
    23: ([2, 0, 1, 2, 0, 1], [2, 0, 2, 1, 0, 2]),
    24: ([1, 0, 2, 1, 0, 2], [1, 0, 2, 2, 0, 2]),
    25: ([1, 0, 2, 1, 0, 2], [2, 0, 1, 2, 0, 2]),
    26: ([2, 0, 1, 0, 0, 0], [2, 1, 0, 2, 1, 0]),
    27: ([0, 0, 0, 2, 0, 1], [0, 1, 2, 0, 1, 2]),
    28: ([0, 0, 0, 1, 0, 2], [0, 1, 2, 0, 1, 2]),
    29: ([1, 0, 2, 0, 0, 0], [2, 1, 0, 2, 1, 0]),
    30: ([2, 0, 2, 2, 0, 1], [2, 0, 1, 2, 0, 1]),
    31: ([2, 0, 1, 2, 0, 2], [1, 0, 2, 1, 0, 2]),
    32: ([1, 0, 2, 2, 0, 2], [1, 0, 2, 1, 0, 2]),
    33: ([2, 0, 2, 1, 0, 2], [2, 0, 1, 2, 0, 1]),
    34: ([2, 0, 0, 2, 0, 0], [0, 0, 0, 0, 0, 0]),
    35: ([0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 2]),
    36: ([0, 0, 2, 0, 0, 2], [0, 0, 0, 0, 0, 0]),
    37: ([0, 0, 0, 0, 0, 0], [2, 0, 0, 2, 0, 0]),
    38: ([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
    39: ([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
    40: ([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
    41: ([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
    42: ([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
    43: ([0, 0, 0, 0, 0, 0], [2, 0, 2, 2, 0, 2]),
}
