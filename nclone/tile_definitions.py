"""
This module contains the definitions for tile properties, including grid edges,
orthogonal segments, diagonal segments, and circular segments. These definitions
are used by the simulator to construct the level geometry from map data.
"""

# This is a dictionary mapping every tile id to the grid edges it contains.
# The first 6 values represent horizontal half-tile edges, from left to right then top to bottom.
# The last 6 values represent vertical half-tile edges, from top to bottom then left to right.
# 1 if there is a grid edge, 0 otherwise.
TILE_GRID_EDGE_MAP = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],  # 0-1: Empty and full tiles
                      # 2-5: Half tiles
                      2: [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0], 3: [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
                      4: [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1], 5: [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0],
                      # 6-9: 45 degree slopes
                      6: [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0], 7: [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1],
                      8: [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], 9: [1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
                      # 10-13: Quarter moons
                      10: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], 11: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                      12: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], 13: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                      # 14-17: Quarter pipes
                      14: [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0], 15: [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1],
                      16: [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], 17: [1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
                      # 18-21: Short mild slopes
                      18: [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0], 19: [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                      20: [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1], 21: [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                      # 22-25: Raised mild slopes
                      22: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], 23: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                      24: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], 25: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                      # 26-29: Short steep slopes
                      26: [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0], 27: [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
                      28: [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1], 29: [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0],
                      # 30-33: Raised steep slopes
                      30: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], 31: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                      32: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], 33: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                      # 34-37: Glitched tiles
                      34: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 35: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                      36: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], 37: [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]}

# This is a dictionary mapping every tile id to the orthogonal linear segments it contains,
# same order as grid edges.
# 0 if no segment, -1 if normal facing left or up, 1 if normal right or down.
TILE_SEGMENT_ORTHO_MAP = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [-1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 1, 1],  # 0-1: Empty and full tiles
                          # 2-5: Half tiles
                          2: [-1, -1, 1, 1, 0, 0, -1, 0, 0, 0, 1, 0], 3: [0, -1, 0, 0, 0, 1, 0, 0, -1, -1, 1, 1],
                          4: [0, 0, -1, -1, 1, 1, 0, -1, 0, 0, 0, 1], 5: [-1, 0, 0, 0, 1, 0, -1, -1, 1, 1, 0, 0],
                          # 6-9: 45 degree slopes
                          6: [-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0], 7: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          8: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], 9: [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
                          # 10-13: Quarter moons
                          10: [-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0], 11: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          12: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], 13: [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
                          # 14-17: Quarter pipes
                          14: [-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0], 15: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          16: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], 17: [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
                          # 18-21: Short mild slopes
                          18: [-1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], 19: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          20: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1], 21: [0, 0, 0, 0, 1, 1, 0, -1, 0, 0, 0, 0],
                          # 22-25: Raised mild slopes
                          22: [-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 1, 0], 23: [-1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1],
                          24: [0, 0, 0, 0, 1, 1, 0, -1, 0, 0, 1, 1], 25: [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 1],
                          # 26-29: Short steep slopes
                          26: [-1, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0], 27: [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          28: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1], 29: [0, 0, 0, 0, 1, 0, -1, -1, 0, 0, 0, 0],
                          # 30-33: Raised steep slopes
                          30: [-1, -1, 0, 0, 1, 0, -1, -1, 0, 0, 0, 0], 31: [-1, -1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
                          32: [0, -1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], 33: [-1, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
                          # 34-37: Glitched tiles
                          34: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 35: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          36: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], 37: [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0]}

# This is a dictionary mapping every tile id to the diagonal linear segment it contains.
# Segments are defined by two sets of point that need to be added to the position inside the grid.
TILE_SEGMENT_DIAG_MAP = {6: ((0, 24), (24, 0)), 7: ((0, 0), (24, 24)),
                         8: ((24, 0), (0, 24)), 9: ((24, 24), (0, 0)),
                         18: ((0, 12), (24, 0)), 19: ((0, 0), (24, 12)),
                         20: ((24, 12), (0, 24)), 21: ((24, 24), (0, 12)),
                         22: ((0, 24), (24, 12)), 23: ((0, 12), (24, 24)),
                         24: ((24, 0), (0, 12)), 25: ((24, 12), (0, 0)),
                         26: ((0, 24), (12, 0)), 27: ((12, 0), (24, 24)),
                         28: ((24, 0), (12, 24)), 29: ((12, 24), (0, 0)),
                         30: ((12, 24), (24, 0)), 31: ((0, 0), (12, 24)),
                         32: ((12, 0), (0, 24)), 33: ((24, 24), (12, 0))}

# This is a dictionary mapping every tile id to the circular segment it contains.
# Segments defined by their center point and the quadrant.
TILE_SEGMENT_CIRCULAR_MAP = {10: ((0, 0), (1, 1), True), 11: ((24, 0), (-1, 1), True),
                             12: ((24, 24), (-1, -1), True), 13: ((0, 24), (1, -1), True),
                             14: ((24, 24), (-1, -1), False), 15: ((0, 24), (1, -1), False),
                             16: ((0, 0), (1, 1), False), 17: ((24, 0), (-1, 1), False)} 

def get_tile_definitions():
    """Return a mapping of tile definition tables used by pathfinding/surface parsing."""
    return {
        'TILE_GRID_EDGE_MAP': TILE_GRID_EDGE_MAP,
        'TILE_SEGMENT_ORTHO_MAP': TILE_SEGMENT_ORTHO_MAP,
        'TILE_SEGMENT_DIAG_MAP': TILE_SEGMENT_DIAG_MAP,
        'TILE_SEGMENT_CIRCULAR_MAP': TILE_SEGMENT_CIRCULAR_MAP,
    }