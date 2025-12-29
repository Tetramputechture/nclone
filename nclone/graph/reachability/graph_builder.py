"""
Fast graph builder using precomputed tile connectivity.

Combines precomputed tile traversability lookups with dynamic entity masking
to build level-specific navigation graphs efficiently.

Improved implementation with sub-tile nodes:
- Each 24px tile divided into 4 sub-nodes (2x2 grid at 12px resolution)
- Respects tile types (type 0=empty, type 1=solid)
- Builds graph only for reachable nodes from player spawn
- More accurate pathfinding for 10px radius player
"""

import logging
import time
import numpy as np
from typing import Dict, Set, Tuple, List, Any, Optional
from collections import OrderedDict

from .tile_connectivity_loader import TileConnectivityLoader
from .entity_mask import EntityMask
from .spatial_hash import SpatialHash
from .pathfinding_utils import flood_fill_reachable_nodes

logger = logging.getLogger(__name__)

# Hardcoded cell size as per N++ constants
CELL_SIZE = 24
SUB_NODE_SIZE = 12  # Divide each 24px tile into 2x2 grid of 12px nodes
PLAYER_RADIUS = 10  # Player collision radius in pixels

# Sub-node offsets within a 24px tile
# Creates 2x2 grid: (6,6), (18,6), (6,18), (18,18)
# Maps to (sub_x, sub_y): (0,0), (1,0), (0,1), (1,1)
SUB_NODE_OFFSETS = [(6, 6), (18, 6), (6, 18), (18, 18)]
SUB_NODE_COORDS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def _check_subnode_validity_simple(tile_type: int, pixel_x: int, pixel_y: int) -> bool:
    """
    O(1) check if sub-node position is in dark grey (non-solid) area of tile.

    Args:
        tile_type: Tile type ID (0-37)
        pixel_x: X position within tile (0-24)
        pixel_y: Y position within tile (0-24)

    Returns:
        True if position is in non-solid area (dark grey), False otherwise
    """
    # Type 0: entirely non-solid
    if tile_type == 0:
        return True
    # Type 1: entirely solid
    if tile_type == 1:
        return False

    # Half tiles (2-5): simple rectangle bounds checks
    if tile_type == 2:  # Top half solid (bottom half traversable)
        return pixel_y >= 12  # Bottom half
    if tile_type == 3:  # Right half solid (left half traversable)
        return pixel_x < 12  # Left half
    if tile_type == 4:  # Bottom half solid (top half traversable)
        return pixel_y < 12  # Top half
    if tile_type == 5:  # Left half solid (right half traversable)
        return pixel_x >= 12  # Right half

    # Diagonal slopes (6-9): line equation checks
    # NOTE: Rendering uses Y-down coordinate system (0 at top, 24 at bottom)
    if tile_type == 6:  # Slope \ - Triangle vertices: (0,24), (24,0), (0,0)
        # Fills top-right triangle. Line: y = 24 - x
        # Solid is ABOVE/RIGHT of diagonal (smaller y values)
        # Traversable is BELOW/LEFT of diagonal (larger y values)
        return pixel_y > (24 - pixel_x)
    if tile_type == 7:  # Slope / - Triangle vertices: (0,0), (24,24), (0,24)
        # Fills left half. Line: y = x from top-left to bottom-right
        # Solid is LEFT of diagonal (x >= y region)
        # Traversable is RIGHT of diagonal (x < y region)
        return pixel_x < pixel_y
    if tile_type == 8:  # Slope / inverted - Triangle vertices: (24,0), (0,24), (24,24)
        # Fills bottom-right triangle. Line: y = 24 - x
        # Solid is BELOW/RIGHT of diagonal (larger y values)
        # Traversable is ABOVE/LEFT of diagonal (smaller y values) OR on the diagonal (y = x)
        return pixel_y < (24 - pixel_x) or pixel_y == pixel_x
    if tile_type == 9:  # Slope \ inverted - Triangle vertices: (24,24), (0,0), (0,24)
        # Fills left side except top-right corner. Line: y = x
        # Solid is LEFT of or ON diagonal (x <= y region)
        # Traversable is RIGHT of diagonal (x > y region)
        return pixel_x > pixel_y

    # Quarter circles (10-13): solid fills most of tile with curved edge in one corner
    # Arc center positions match shared_tile_renderer.py exactly:
    #   dx = tile_size if (tile_type == 11 or tile_type == 12) else 0
    #   dy = tile_size if (tile_type == 12 or tile_type == 13) else 0
    # Solid is INSIDE the arc (dist < 24), traversable is OUTSIDE (dist >= 24)
    # Only 1 of the 4 sub-nodes is traversable (the corner opposite the arc center)
    if tile_type == 10:  # Arc center at (0, 0) - only bottom-right corner traversable
        dist_sq = pixel_x * pixel_x + pixel_y * pixel_y
        return dist_sq >= 24 * 24
    if tile_type == 11:  # Arc center at (24, 0) - only bottom-left corner traversable
        dx = pixel_x - 24
        dist_sq = dx * dx + pixel_y * pixel_y
        return dist_sq >= 24 * 24
    if tile_type == 12:  # Arc center at (24, 24) - only top-left corner traversable
        dx = pixel_x - 24
        dy = pixel_y - 24
        dist_sq = dx * dx + dy * dy
        return dist_sq >= 24 * 24
    if tile_type == 13:  # Arc center at (0, 24) - only top-right corner traversable
        dy = pixel_y - 24
        dist_sq = pixel_x * pixel_x + dy * dy
        return dist_sq >= 24 * 24

    # Quarter pipes (14-17): "inverted" quarter circles - solid OUTSIDE the arc
    # Arc center positions match shared_tile_renderer.py exactly:
    #   dx2 = tile_size if (tile_type == 14 or tile_type == 17) else 0
    #   dy2 = tile_size if (tile_type == 14 or tile_type == 15) else 0

    # Traversable is INSIDE the arc (dist < 24), solid is OUTSIDE (dist >= 24)
    # Only 1 of the 4 sub-nodes is solid (the corner opposite the arc center)
    if tile_type == 14:  # Arc center at (24, 24) - top-left corner solid
        dx = pixel_x - 24
        dy = pixel_y - 24
        dist_sq = dx * dx + dy * dy
        return dist_sq < 24 * 24
    if tile_type == 15:  # Arc center at (0, 24) - top-right corner solid
        dy = pixel_y - 24
        dist_sq = pixel_x * pixel_x + dy * dy
        return dist_sq < 24 * 24
    if tile_type == 16:  # Arc center at (0, 0) - bottom-right corner solid
        dist_sq = pixel_x * pixel_x + pixel_y * pixel_y
        return dist_sq < 24 * 24
    if tile_type == 17:  # Arc center at (24, 0) - bottom-left corner solid
        dx = pixel_x - 24
        dist_sq = dx * dx + pixel_y * pixel_y
        return dist_sq < 24 * 24

    # Short mild slopes (18-21)
    if tile_type == 18:  # Mild slope up-left, short rise from left edge to middle
        # Line from (0, 12) to (24, 0): y = 12 - 0.5x
        # Solid is above line, traversable is below line
        return pixel_y >= (12 - pixel_x * 12 / 24)
    if tile_type == 19:  # Mild slope up-right, short rise from middle to right edge
        # Line from (0, 0) to (24, 12): y = 0.5x
        # Solid is above line, traversable is below line
        return pixel_y >= (pixel_x * 12 / 24)
    if tile_type == 20:  # Mild slope down-right, short drop from middle to right edge
        # Line from (24, 12) to (0, 24): y = 24 - 0.5x
        # Solid is below line, traversable is above line
        return pixel_y <= (24 - pixel_x * 12 / 24)
    if tile_type == 21:  # Mild slope down-left, short drop from left edge to middle
        # Line from (24, 24) to (0, 12): y = 12 + 0.5x
        # Solid is below line, traversable is above/on line
        return pixel_y <= (12 + pixel_x * 12 / 24)

    # Raised mild slopes (22-25)
    if tile_type == 22:  # Raised mild slope, platform on left with gentle rise
        # Line from (0, 24) to (24, 12): y = 24 - 0.5x
        # Solid is above line, traversable is below/on line (raised platform)
        return pixel_y >= (24 - pixel_x * 12 / 24)
    if tile_type == 23:  # Raised mild slope, platform on right with gentle rise
        # Line from (0, 12) to (24, 24): y = 12 + 0.5x
        # Solid is above line, traversable is below/on line (raised platform)
        return pixel_y >= (12 + pixel_x * 12 / 24)
    if tile_type == 24:  # Raised mild slope, platform on right with gentle drop
        # Line from (24, 0) to (0, 12): y = 12 - 0.5x
        # Solid is below/on line, traversable is above (triangle at top)
        return pixel_y < (12 - pixel_x * 12 / 24)
    if tile_type == 25:  # Raised mild slope, platform on left with gentle drop
        # Line from (24, 12) to (0, 0): y = 0.5x
        # Solid is below/on line, traversable is above (triangle at top)
        return pixel_y < (pixel_x * 12 / 24)
    # Short steep slopes (26-29)
    if tile_type == 26:  # Steep slope up-left, sharp rise from left edge to middle
        # Line from (0, 24) to (12, 0)
        # Solid is below line
        return pixel_y >= (24 - pixel_x * 24 / 12) if pixel_x <= 12 else True
    if tile_type == 27:  # Steep slope down-left, starts middle descends right
        # Line from (12, 24) to (0, 0)
        # Solid is below line
        return pixel_y >= (pixel_x * 24 / 12)
    if tile_type == 28:  # Steep slope down-right, sharp drop from middle to right edge
        # Line from (12, 24) to (24, 0)
        # Solid is above line
        return pixel_y <= (24 - (pixel_x - 12) * 24 / 12) if pixel_x >= 12 else True
    if tile_type == 29:  # Steep slope down-left, sharp drop from left edge to middle
        # Line from (12, 24) to (0, 0)
        # Solid is left triangle: vertices (12,24), (0,0), (0,24)
        # Traversable is right of x=12 OR above diagonal
        return pixel_x > 12 or pixel_y < (pixel_x * 24 / 12)

    # Raised steep slopes (30-33)
    if tile_type == 30:  # Raised steep slope, platform on left with sharp rise
        # Solid quad: (12,24), (0,24), (0,0), (24,0)
        # Line from (12, 24) to (24, 0)
        # Traversable is right triangle: (12,24), (24,0), (24,24)
        return (
            pixel_y >= (24 - (pixel_x - 12) * 24 / 12)
            if pixel_x >= 12
            else pixel_y >= 24
        )
    if tile_type == 31:  # Raised steep slope, platform on right with sharp rise
        # Solid quad: (12,24), (24,24), (24,0), (0,0)
        # Line from (0, 0) to (12, 24)
        # Traversable is left of diagonal or left of x=12
        return pixel_y >= (pixel_x * 24 / 12) if pixel_x <= 12 else pixel_y >= 24
    if tile_type == 32:  # Raised steep slope, platform on right with sharp drop
        # Solid quad: (12,0), (0,24), (24,24), (24,0)
        # Line from (12,0) to (0,24): y = 24 - 2x
        # For x < 12: TRAV if above line (y < 24-2x), SOLID if below line
        # For x >= 12: Always SOLID (in rectangle portion)
        return pixel_x < 12 and pixel_y < (24 - 2 * pixel_x)
    if tile_type == 33:  # Raised steep slope, platform on left with sharp drop
        # Solid quad: (12,0), (24,24), (0,24), (0,0)
        # Line from (12,0) to (24,24): y = 2(x-12) = 2x - 24
        # For x <= 12: Always SOLID (in rectangle portion)
        # For x > 12: TRAV if above line (y < 2x-24), SOLID if below line
        return pixel_x > 12 and pixel_y < (2 * pixel_x - 24)

    # Glitched tiles (34-37): treat as non-solid for safety
    if tile_type >= 34:
        return True

    # Unknown tile type: assume non-solid
    return True


def _precompute_subnode_validity_table() -> Dict[int, Dict[Tuple[int, int], bool]]:
    """
    Precompute which sub-nodes are valid (in dark grey areas) for each tile type.

    Returns:
        Dict mapping tile_type -> {(sub_x, sub_y): bool}
    """
    table = {}
    for tile_type in range(38):  # All tile types 0-37
        table[tile_type] = {}
        for (offset_x, offset_y), (sub_x, sub_y) in zip(
            SUB_NODE_OFFSETS, SUB_NODE_COORDS
        ):
            # Check if this sub-node position is in non-solid area
            is_valid = _check_subnode_validity_simple(tile_type, offset_x, offset_y)
            table[tile_type][(sub_x, sub_y)] = is_valid
    return table


# Precomputed lookup table: O(1) access to sub-node validity
SUBNODE_VALIDITY_TABLE = _precompute_subnode_validity_table()


def _line_crosses_diagonal(
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
) -> bool:
    """
    Check if line segment p1-p2 crosses diagonal line segment line_start-line_end.

    Uses determinant-based line intersection test (O(1)).
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = line_start
    x4, y4 = line_end

    # Calculate determinants
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return False  # Parallel lines

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # Intersection occurs if both t and u are in [0, 1]
    return 0 < t < 1 and 0 < u < 1


def _line_crosses_circle(
    p1: Tuple[int, int], p2: Tuple[int, int], center: Tuple[int, int], radius: float
) -> bool:
    """
    Check if line segment p1-p2 crosses circle boundary.

    For convex circles (quarter circles), checks if line crosses from inside to outside.
    """
    cx, cy = center
    x1, y1 = p1
    x2, y2 = p2

    # Distance from center to endpoints
    dist1_sq = (x1 - cx) ** 2 + (y1 - cy) ** 2
    dist2_sq = (x2 - cx) ** 2 + (y2 - cy) ** 2
    radius_sq = radius**2

    # Check if line crosses circle boundary
    # (one endpoint inside, one outside)
    inside1 = dist1_sq < radius_sq
    inside2 = dist2_sq < radius_sq

    return inside1 != inside2


def _line_passes_through_circle(
    p1: Tuple[int, int], p2: Tuple[int, int], center: Tuple[int, int], radius: float
) -> bool:
    """
    Check if line segment p1-p2 passes through or is inside a circular region.

    More comprehensive than _line_crosses_circle - also detects:
    - Both endpoints inside the circle (entirely blocked)
    - Line passing through circle even if both endpoints outside

    Args:
        p1, p2: Line segment endpoints (tile-relative coordinates)
        center: Circle center
        radius: Circle radius

    Returns:
        True if any part of the line is inside the circle (blocked)
    """
    cx, cy = center
    x1, y1 = p1
    x2, y2 = p2
    radius_sq = radius * radius

    # Distance from center to endpoints
    dist1_sq = (x1 - cx) ** 2 + (y1 - cy) ** 2
    dist2_sq = (x2 - cx) ** 2 + (y2 - cy) ** 2

    # If either endpoint is inside the circle, line passes through solid
    if dist1_sq < radius_sq or dist2_sq < radius_sq:
        return True

    # Check if line segment passes through circle (both endpoints outside)
    # Find closest point on line segment to circle center
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy

    if len_sq == 0:
        # Degenerate line (single point), already checked above
        return False

    # Project center onto line, clamped to segment
    t = max(0, min(1, ((cx - x1) * dx + (cy - y1) * dy) / len_sq))

    # Closest point on segment to center
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Distance from closest point to center
    closest_dist_sq = (closest_x - cx) ** 2 + (closest_y - cy) ** 2

    return closest_dist_sq < radius_sq


def _can_traverse_within_tile(
    tile_type: int, src_sub: Tuple[int, int], dst_sub: Tuple[int, int]
) -> bool:
    """
    O(1) check if two sub-nodes within the same tile can reach each other.

    Considers the solid geometry between them using simple geometric checks.
    """
    if src_sub == dst_sub:
        return True

    # Get pixel positions of sub-nodes
    src_offset = SUB_NODE_OFFSETS[SUB_NODE_COORDS.index(src_sub)]
    dst_offset = SUB_NODE_OFFSETS[SUB_NODE_COORDS.index(dst_sub)]

    # Type 0: All sub-nodes connected
    if tile_type == 0:
        return True

    # Type 1: No sub-nodes (all solid)
    if tile_type == 1:
        return False

    # Quarter pipes (14-17): All sub-nodes connected (entire tile is traversable)
    if tile_type in [14, 15, 16, 17]:
        return True

    # Half tiles (2-5): Sub-nodes on same side connected
    if tile_type == 2:  # Top half solid
        # Both must be in bottom half (y >= 12)
        return src_offset[1] >= 12 and dst_offset[1] >= 12
    if tile_type == 3:  # Right half solid
        # Both must be in left half (x < 12)
        return src_offset[0] < 12 and dst_offset[0] < 12
    if tile_type == 4:  # Bottom half solid
        # Both must be in top half (y < 12)
        return src_offset[1] < 12 and dst_offset[1] < 12
    if tile_type == 5:  # Left half solid
        # Both must be in right half (x >= 12)
        return src_offset[0] >= 12 and dst_offset[0] >= 12

    # Diagonal slopes (6-9): Check if line crosses the slope
    if tile_type == 6:  # Slope from top-left to bottom-right (\)
        # Line: from (0,24) to (24,0), solid is below/left
        return not _line_crosses_diagonal(src_offset, dst_offset, (0, 24), (24, 0))
    if tile_type == 7:  # Slope from top-right to bottom-left (/)
        # Line: from (0,0) to (24,24), solid is above/right
        return not _line_crosses_diagonal(src_offset, dst_offset, (0, 0), (24, 24))
    if tile_type == 8:  # Slope / inverted
        # Line: from (24,0) to (0,24), solid is below/right
        return not _line_crosses_diagonal(src_offset, dst_offset, (24, 0), (0, 24))
    if tile_type == 9:  # Slope \ inverted
        # Line: from (24,24) to (0,0), solid is above/left
        return not _line_crosses_diagonal(src_offset, dst_offset, (24, 24), (0, 0))

    # Quarter circles (10-13): Convex corners - check if line crosses circle
    if tile_type == 10:  # Bottom-right quarter circle, center at (24,24)
        return not _line_crosses_circle(src_offset, dst_offset, (24, 24), 24)
    if tile_type == 11:  # Bottom-left quarter circle, center at (0,24)
        return not _line_crosses_circle(src_offset, dst_offset, (0, 24), 24)
    if tile_type == 12:  # Top-left quarter circle, center at (0,0)
        return not _line_crosses_circle(src_offset, dst_offset, (0, 0), 24)
    if tile_type == 13:  # Top-right quarter circle, center at (24,0)
        return not _line_crosses_circle(src_offset, dst_offset, (24, 0), 24)

    # Quarter pipes (14-17) handled earlier - all connections are valid

    # Mild and steep slopes (18-33): Check if line crosses the slope
    # For simplicity, use line intersection checks with the slope segments
    if tile_type in [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]:
        # Get slope line from TILE_SEGMENT_DIAG_MAP
        from ...tile_definitions import TILE_SEGMENT_DIAG_MAP

        if tile_type in TILE_SEGMENT_DIAG_MAP:
            line_start, line_end = TILE_SEGMENT_DIAG_MAP[tile_type]
            return not _line_crosses_diagonal(
                src_offset, dst_offset, line_start, line_end
            )

    # Default: assume connected (conservative)
    return True


def _check_line_crosses_tile_geometry(
    tile_type: int,
    src_pixel: Tuple[int, int],
    dst_pixel: Tuple[int, int],
    tile_x: int,
    tile_y: int,
) -> bool:
    """
    Check if a line segment crosses solid geometry within a tile.

    Used for cross-tile edge validation to ensure the path segment within
    each tile doesn't pass through solid areas (slopes, circles, etc.).

    Args:
        tile_type: Tile type ID (0-37)
        src_pixel: Source pixel position (absolute coordinates)
        dst_pixel: Destination pixel position (absolute coordinates)
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate

    Returns:
        True if line crosses solid geometry (invalid), False if clear (valid)
    """
    # Convert to tile-relative coordinates
    tile_origin_x = tile_x * CELL_SIZE
    tile_origin_y = tile_y * CELL_SIZE

    src_rel = (src_pixel[0] - tile_origin_x, src_pixel[1] - tile_origin_y)
    dst_rel = (dst_pixel[0] - tile_origin_x, dst_pixel[1] - tile_origin_y)

    # Type 0: entirely non-solid - no obstruction
    if tile_type == 0:
        return False

    # Type 1: entirely solid - always blocked
    if tile_type == 1:
        return True

    # Quarter pipes (14-17): All traversable
    if tile_type in [14, 15, 16, 17]:
        return False

    # Half tiles (2-5): Check if line crosses the half-tile boundary into solid area
    if tile_type == 2:  # Top half solid (y < 12 is solid)
        # Line crosses if it goes through the y=12 boundary into solid region
        return _line_crosses_diagonal(src_rel, dst_rel, (0, 12), (24, 12))
    if tile_type == 3:  # Right half solid (x >= 12 is solid)
        return _line_crosses_diagonal(src_rel, dst_rel, (12, 0), (12, 24))
    if tile_type == 4:  # Bottom half solid (y >= 12 is solid)
        return _line_crosses_diagonal(src_rel, dst_rel, (0, 12), (24, 12))
    if tile_type == 5:  # Left half solid (x < 12 is solid)
        return _line_crosses_diagonal(src_rel, dst_rel, (12, 0), (12, 24))

    # Diagonal slopes (6-9): Check if line crosses the slope
    if tile_type == 6:  # Slope \ - solid is top-right
        return _line_crosses_diagonal(src_rel, dst_rel, (0, 24), (24, 0))
    if tile_type == 7:  # Slope / - solid is top-left
        return _line_crosses_diagonal(src_rel, dst_rel, (0, 0), (24, 24))
    if tile_type == 8:  # Slope / inverted - solid is bottom-right
        return _line_crosses_diagonal(src_rel, dst_rel, (24, 0), (0, 24))
    if tile_type == 9:  # Slope \ inverted - solid is bottom-left
        return _line_crosses_diagonal(src_rel, dst_rel, (24, 24), (0, 0))

    # Quarter circles (10-13): Check if line passes through solid curved region
    # Use _line_passes_through_circle which catches:
    # - Both endpoints inside solid region
    # - Line passing through solid even if both endpoints outside
    if tile_type == 10:  # Bottom-right quarter circle, center at (0,0)
        return _line_passes_through_circle(src_rel, dst_rel, (0, 0), 24)
    if tile_type == 11:  # Bottom-left quarter circle, center at (24,0)
        return _line_passes_through_circle(src_rel, dst_rel, (24, 0), 24)
    if tile_type == 12:  # Top-left quarter circle, center at (24,24)
        return _line_passes_through_circle(src_rel, dst_rel, (24, 24), 24)
    if tile_type == 13:  # Top-right quarter circle, center at (0,24)
        return _line_passes_through_circle(src_rel, dst_rel, (0, 24), 24)

    # Mild and steep slopes (18-33): Check if line crosses the slope
    if tile_type in [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]:
        from ...tile_definitions import TILE_SEGMENT_DIAG_MAP

        if tile_type in TILE_SEGMENT_DIAG_MAP:
            line_start, line_end = TILE_SEGMENT_DIAG_MAP[tile_type]
            return _line_crosses_diagonal(src_rel, dst_rel, line_start, line_end)

    # Default: assume no obstruction (conservative for unknown types)
    return False


def _check_cross_tile_line_clear(
    src_pixel_x: int,
    src_pixel_y: int,
    dst_pixel_x: int,
    dst_pixel_y: int,
    src_tile_type: int,
    dst_tile_type: int,
    src_tile_x: int,
    src_tile_y: int,
    dst_tile_x: int,
    dst_tile_y: int,
) -> bool:
    """
    Check if a cross-tile edge path is clear of solid geometry in both tiles.

    For cardinal direction cross-tile edges, validates that the line segment
    doesn't pass through solid areas (slopes, quarter circles, etc.) in either
    the source or destination tile.

    Args:
        src_pixel_x, src_pixel_y: Source sub-node pixel position
        dst_pixel_x, dst_pixel_y: Destination sub-node pixel position
        src_tile_type: Source tile type
        dst_tile_type: Destination tile type
        src_tile_x, src_tile_y: Source tile coordinates
        dst_tile_x, dst_tile_y: Destination tile coordinates

    Returns:
        True if path is clear, False if blocked by geometry
    """
    src_pixel = (src_pixel_x, src_pixel_y)
    dst_pixel = (dst_pixel_x, dst_pixel_y)

    # Check if line crosses solid geometry in source tile
    if _check_line_crosses_tile_geometry(
        src_tile_type, src_pixel, dst_pixel, src_tile_x, src_tile_y
    ):
        return False

    # Check if line crosses solid geometry in destination tile
    if _check_line_crosses_tile_geometry(
        dst_tile_type, src_pixel, dst_pixel, dst_tile_x, dst_tile_y
    ):
        return False

    return True


def _precompute_within_tile_connectivity() -> Dict[
    int, Dict[Tuple[int, int], Set[Tuple[int, int]]]
]:
    """
    Precompute which sub-nodes can reach each other within the same tile.

    For each tile type and each valid sub-node, determines which other
    sub-nodes in that tile it can reach without crossing solid geometry.

    Returns:
        Dict mapping tile_type -> {sub_node: {reachable_sub_nodes}}
    """
    connectivity = {}

    for tile_type in range(38):  # All tile types 0-37
        connectivity[tile_type] = {}

        # Get all valid sub-nodes for this tile
        valid_subs = []
        for sub_x, sub_y in SUB_NODE_COORDS:
            if SUBNODE_VALIDITY_TABLE.get(tile_type, {}).get((sub_x, sub_y), False):
                valid_subs.append((sub_x, sub_y))

        # For each valid sub-node, determine which others it can reach
        for src_sub in valid_subs:
            reachable = set()
            for dst_sub in valid_subs:
                if _can_traverse_within_tile(tile_type, src_sub, dst_sub):
                    reachable.add(dst_sub)
            connectivity[tile_type][src_sub] = reachable

    return connectivity


# Precomputed lookup table: O(1) access to within-tile connectivity
WITHIN_TILE_CONNECTIVITY = _precompute_within_tile_connectivity()


class GraphBuilder:
    """
    Builds level traversability graph using precomputed tile connectivity.

    Combines:
    1. Static tile connectivity (precomputed, O(1) lookup)
    2. Dynamic entity mask (doors, mines)
    3. Result: Complete traversability graph for pathfinding

    Performance:
    - First call per level: <0.2ms (builds and caches graph)
    - Subsequent calls: <0.05ms (uses cached graph + state mask)

    Caching Strategy:
    - Tiles never change during a run → cache base graph
    - Entity POSITIONS never change → cache in base graph
    - Entity STATES change → lightweight mask update only
    """

    def __init__(self, debug: bool = False):
        """Initialize graph builder with connectivity loader."""
        self.connectivity_loader = TileConnectivityLoader()
        self.debug = debug

        # Per-level caching with LRU eviction to prevent unbounded growth
        # Uses OrderedDict for O(1) LRU operations
        self._level_graph_cache = OrderedDict()  # level_id -> base graph
        self._flood_fill_cache = (
            OrderedDict()
        )  # level_id -> (last_ninja_pos, reachable_set)
        self._max_cache_size = 100  # Reasonable limit for level graphs
        self._current_level_id = None

        # Flood fill caching (per level)
        # Only recompute flood fill when player moves >= 12px from last cached position
        # 12px matches sub-node spacing - recomputing more frequently adds overhead without
        # improving PBRS accuracy (sub-node interpolation handles dense rewards within a cell)
        self._flood_fill_distance_threshold = 12  # 12px - one sub-node spacing

        # Performance tracking
        self.build_times = []
        self.cache_hits = 0

        # Debug statistics
        self.debug_stats = {
            "total_tiles": 0,
            "total_sub_nodes": 0,
            "sub_nodes_by_tile_type": {},
            "traversability_checks": 0,
            "within_tile_checks": 0,
            "between_tile_checks": 0,
            "blocked_by_geometry": 0,
            "blocked_by_connectivity": 0,
            "blocked_by_diagonal": 0,
            "physics_filtered_horizontal": 0,  # Long horizontal edges filtered by physics
            "physics_filtered_diagonal_upward": 0,  # Diagonal upward edges filtered by physics
        }
        self.cache_misses = 0

    def _debug_log(self, message: str):
        """Log debug message if debug mode is enabled."""
        if self.debug:
            print(f"[GraphBuilder DEBUG] {message}")

    def _debug_log_traversability_decision(
        self,
        src_tile_type: int,
        dst_tile_type: int,
        direction: str,
        result: bool,
        reason: str,
    ):
        """Log traversability decisions for debugging."""
        if self.debug:
            status = "✓ ALLOWED" if result else "✗ BLOCKED"
            self._debug_log(
                f"Traversability: Tile {src_tile_type} → {dst_tile_type} ({direction}): "
                f"{status} - {reason}"
            )

    def _debug_print_tile_statistics(self, tiles: np.ndarray, all_sub_nodes: Dict):
        """Print statistics about sub-nodes per tile type."""
        if not self.debug:
            return

        self._debug_log("=== Tile Statistics ===")
        self._debug_log(f"Total tiles in level: {self.debug_stats['total_tiles']}")
        self._debug_log(
            f"Total sub-nodes generated: {self.debug_stats['total_sub_nodes']}"
        )
        self._debug_log("")
        self._debug_log("Sub-nodes by tile type:")

        for tile_type in sorted(self.debug_stats["sub_nodes_by_tile_type"].keys()):
            count = self.debug_stats["sub_nodes_by_tile_type"][tile_type]
            self._debug_log(f"  Type {tile_type:2d}: {count:4d} sub-nodes")

        self._debug_log("")
        self._debug_log("=== Traversability Check Statistics ===")
        self._debug_log(f"Total checks: {self.debug_stats['traversability_checks']}")
        self._debug_log(f"Within-tile checks: {self.debug_stats['within_tile_checks']}")
        self._debug_log(
            f"Between-tile checks: {self.debug_stats['between_tile_checks']}"
        )
        self._debug_log(
            f"Blocked by geometry: {self.debug_stats['blocked_by_geometry']}"
        )
        self._debug_log(
            f"Blocked by connectivity: {self.debug_stats['blocked_by_connectivity']}"
        )
        self._debug_log(
            f"Blocked by diagonal: {self.debug_stats['blocked_by_diagonal']}"
        )

    def build_graph(
        self,
        level_data: Dict[str, Any],
        ninja_pos: Optional[Tuple[int, int]] = None,
        level_id: Optional[str] = None,
        filter_by_reachability: bool = True,
    ) -> Dict[str, Any]:
        """
        Build complete traversability graph for level.

        The adjacency graph is filtered to only include nodes reachable from the initial
        player position (from LevelData.start_position) when filter_by_reachability=True.
        This ensures path caching only calculates distances for reachable areas, preventing
        caching of unreachable or isolated regions.

        When rebuilding mid-gameplay (e.g., after switch state changes), set
        filter_by_reachability=False to avoid filtering by spawn reachability, as the ninja
        may have moved to areas that are no longer reachable from spawn.

        Args:
            level_data: Level data with tiles, entities, switch_states, start_position
                       Can be a dict or LevelData object
            ninja_pos: Optional ninja position for reachability analysis (current position)
            level_id: Optional level identifier for caching
            filter_by_reachability: If True, filter adjacency to nodes reachable from spawn.
                                   Set to False when rebuilding during gameplay.

        Returns:
            Dictionary containing:
            - 'adjacency': Dict mapping (x, y) -> List[(neighbor_x, neighbor_y, cost)]
                          Filtered to reachable nodes if filter_by_reachability=True
            - 'reachable': Set of reachable positions from ninja_pos (if provided)
            - 'blocked_positions': Set of blocked positions (by entities)
            - 'blocked_edges': Set of blocked edges
        """
        start_time = time.perf_counter()

        # Handle both dict and LevelData object
        if hasattr(level_data, "tiles"):
            # LevelData object
            tiles = level_data.tiles
            level_data_dict = {
                "tiles": tiles,
                "entities": level_data.entities,
                "start_position": level_data.start_position,
                "switch_states": level_data.switch_states,
            }
        else:
            # Dict format
            tiles = level_data["tiles"]  # 2D numpy array
            level_data_dict = level_data

        height, width = tiles.shape

        # Generate level ID if not provided
        if level_id is None:
            level_id = f"level_{hash(tiles.tobytes())}"

        # Check cache
        if level_id in self._level_graph_cache:
            # Use cached base graph
            base_graph = self._level_graph_cache[level_id]
            # Move to end to maintain LRU ordering
            self._level_graph_cache.move_to_end(level_id)
            self.cache_hits += 1
        else:
            # Build new base graph (includes tile connectivity, entity positions)
            base_graph = self._build_base_graph(tiles, level_data_dict)
            self._level_graph_cache[level_id] = base_graph
            self._level_graph_cache.move_to_end(level_id)

            # Evict oldest entry if cache exceeds max size
            if len(self._level_graph_cache) > self._max_cache_size:
                self._level_graph_cache.popitem(last=False)

            self.cache_misses += 1

        # Apply dynamic entity state mask
        entity_mask = EntityMask(level_data_dict)

        # Get all node positions from base graph for radius-based blocking
        all_nodes = set(base_graph["adjacency"].keys())

        # Get blocked pixel positions (radius-based for mines, tile-based for doors)
        blocked_pixel_positions = entity_mask.get_blocked_pixel_positions(all_nodes)

        # Get blocked edges (currently none, but extensible)
        blocked_edges = entity_mask.get_blocked_edges()

        # Create final adjacency by filtering blocked positions/edges
        adjacency = self._apply_entity_mask(
            base_graph["adjacency"], blocked_pixel_positions, blocked_edges
        )

        # Filter adjacency to only include nodes reachable from initial player position
        # This ensures path caching only operates on reachable areas
        # Skip filtering when rebuilding mid-gameplay (ninja may be in isolated area)
        if filter_by_reachability:
            initial_player_pos = self._find_player_spawn(level_data_dict, tiles)

            # Build spatial hash for optimization (not in base_graph yet)
            temp_spatial_hash = SpatialHash(cell_size=CELL_SIZE)
            temp_spatial_hash.build(list(adjacency.keys()))

            # Use shared flood fill utility
            reachable_nodes = flood_fill_reachable_nodes(
                initial_player_pos,
                adjacency,
                spatial_hash=temp_spatial_hash,
                subcell_lookup=None,  # Will be auto-loaded by flood_fill
                player_radius=PLAYER_RADIUS,
            )

            # Validate that flood fill found reachable nodes
            # If empty, the level may have invalid spawn location or broken tile connectivity
            if len(reachable_nodes) == 0:
                logger.error(
                    f"CRITICAL: Flood fill found ZERO reachable nodes from spawn position "
                    f"{initial_player_pos}. Level may have invalid spawn location or broken "
                    f"tile connectivity. Using ALL adjacency nodes as fallback."
                )
                # Use all adjacency nodes as fallback to prevent complete failure
                reachable_nodes = set(adjacency.keys())

            # Filter adjacency to only reachable nodes
            # Critical: This ensures all cached distances are for reachable areas only
            adjacency = self._filter_adjacency_to_reachable(adjacency, reachable_nodes)

            # # Apply smart truncation if graph exceeds N_MAX_NODES
            # # This preserves full exploration space when possible, only truncating
            # # when the level is exceptionally open
            # if len(reachable_nodes) > N_MAX_NODES:
            #     # Extract goal positions from entities
            #     goal_positions = self._extract_goal_positions(level_data_dict)

            #     adjacency, reachable_nodes, was_truncated = truncate_graph_if_needed(
            #         adjacency=adjacency,
            #         reachable_nodes=reachable_nodes,
            #         start_pos=initial_player_pos,
            #         goal_positions=goal_positions,
            #         max_nodes=N_MAX_NODES,
            #     )

        # Build spatial hash for O(1) node lookup
        spatial_hash = SpatialHash(cell_size=CELL_SIZE)
        spatial_hash.build(list(adjacency.keys()))

        # PERFORMANCE OPTIMIZATION: Pre-compute node physics properties
        # Caching grounded/walled status eliminates expensive per-edge checks during pathfinding
        # This trades O(nodes) upfront cost for O(1) lookups during A* (called frequently)
        node_physics_cache = self._compute_node_physics_cache(base_graph["adjacency"])

        result = {
            "adjacency": adjacency,  # Masked adjacency (for pathfinding)
            "base_adjacency": base_graph[
                "adjacency"
            ],  # Unmasked adjacency (for physics checks)
            "blocked_positions": blocked_pixel_positions,
            "blocked_edges": blocked_edges,
            "base_graph_cached": level_id in self._level_graph_cache,
            "spatial_hash": spatial_hash,  # For fast node lookup
            "reachable_node_count": len(adjacency),  # For adaptive caching strategy
            "node_physics": node_physics_cache,  # Pre-computed physics properties (grounded, walled)
        }

        # If ninja position provided, compute reachable set for that position
        # Note: This may differ from initial_player_pos if ninja has moved
        # Optimization: Only recompute flood fill if player moved >= 12px from last cached position
        if ninja_pos is not None:
            reachable = self._get_cached_flood_fill(
                level_id, ninja_pos, adjacency, spatial_hash
            )
            result["reachable"] = reachable

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.build_times.append(elapsed_ms)

        return result

    def _compute_node_physics_cache(
        self,
        base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    ) -> Dict[Tuple[int, int], Dict[str, bool]]:
        """
        Pre-compute physics properties for all nodes to optimize pathfinding.

        Caches grounded and walled status for each node, eliminating expensive
        per-edge checks during A* pathfinding. This is a one-time O(nodes) cost
        that enables O(1) lookups during physics-aware cost calculations.

        Args:
            base_adjacency: Base graph adjacency (pre-entity-mask) for physics checks

        Returns:
            Dict mapping node positions to {"grounded": bool, "walled": bool}
        """
        SUB_NODE_SIZE = 12  # Sub-node spacing in pixels

        physics_cache = {}

        for node_pos in base_adjacency.keys():
            x, y = node_pos

            # Check if node is grounded (has solid surface below)
            below_pos = (x, y + SUB_NODE_SIZE)
            is_grounded = True

            if below_pos in base_adjacency:
                # If there's a neighbor below, node is airborne
                neighbors = base_adjacency[node_pos]
                for neighbor_pos, _ in neighbors:
                    if neighbor_pos == below_pos:
                        is_grounded = False
                        break

            # Check if node has a wall (left or right neighbor unreachable)
            left_pos = (x - SUB_NODE_SIZE, y)
            right_pos = (x + SUB_NODE_SIZE, y)
            is_walled = (
                left_pos not in base_adjacency or right_pos not in base_adjacency
            )

            physics_cache[node_pos] = {
                "grounded": is_grounded,
                "walled": is_walled,
            }

        return physics_cache

    def _build_base_graph(
        self, tiles: np.ndarray, level_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build base graph from tile layout with sub-tile nodes (cached per level).

        Note: This builds adjacency for ALL traversable nodes. Filtering to only
        reachable nodes from initial player position is done in build_graph() after
        applying entity masks.

        Key improvements:
        1. Each 24px tile → 4 sub-nodes at 12px resolution (2x2 grid)
        2. Respects tile types (type 0=empty OK, type 1=solid blocked)
        3. More accurate collision detection with 10px player radius
        """
        height, width = tiles.shape

        # Find player spawn position
        player_spawn = self._find_player_spawn(level_data, tiles)

        # Generate all sub-nodes and check which are valid (not in solid tiles)
        all_sub_nodes = self._generate_sub_nodes(tiles)

        # Build adjacency only for reachable sub-nodes from player spawn
        adjacency = self._build_reachable_adjacency(tiles, all_sub_nodes, player_spawn)

        return {"adjacency": adjacency}

    def _find_player_spawn(
        self, level_data: Dict[str, Any], tiles: np.ndarray
    ) -> Tuple[int, int]:
        """
        Find player spawn position from level data.

        Returns pixel coordinates of spawn point in tile data coordinate space.
        Falls back to center of level if not found.
        """
        # First, check if start_position is provided in level_data
        # This is the preferred method as it's extracted directly from map_data
        if isinstance(level_data, dict):
            start_position = level_data.get("start_position")
            if start_position is not None:
                return tuple(start_position)
        elif hasattr(level_data, "start_position"):
            # Handle LevelData object
            return level_data.start_position

        # Fallback: Check for entities with type 14 (ninja spawn)
        # This is deprecated but kept for backward compatibility
        entities = (
            level_data.get("entities", [])
            if isinstance(level_data, dict)
            else getattr(level_data, "entities", [])
        )
        for entity in entities:
            if entity.get("type") == 14:  # Ninja spawn point
                # Entity position is in pixels
                x = entity.get("x", 0)
                y = entity.get("y", 0)
                return (int(x), int(y))

        # Final fallback: center of level (skip padding)
        height, width = tiles.shape
        center_x = (width // 2) * CELL_SIZE + CELL_SIZE // 2
        center_y = (height // 2) * CELL_SIZE + CELL_SIZE // 2
        return (center_x, center_y)

    def _extract_goal_positions(
        self, level_data: Dict[str, Any]
    ) -> List[Tuple[int, int]]:
        """
        Extract goal positions (exit switch, exit door) from level data.

        Used for smart truncation to prioritize nodes on paths to objectives.

        Args:
            level_data: Level data dict with entities

        Returns:
            List of goal positions (x, y) in pixels
        """
        goal_positions = []

        entities = level_data.get("entities", [])
        for entity in entities:
            entity_type = entity.get("type")
            # Exit switch (type 3) and exit door (type 4)
            if entity_type in [3, 4]:
                x = entity.get("x", 0)
                y = entity.get("y", 0)
                goal_positions.append((int(x), int(y)))

        return goal_positions

    def _is_position_in_non_solid_area(
        self, pixel_x: int, pixel_y: int, tile_type: int, tile_x: int, tile_y: int
    ) -> bool:
        """
        Check if position is in dark grey (non-solid) area of tile - O(1) lookup.

        Args:
            pixel_x: Pixel X coordinate
            pixel_y: Pixel Y coordinate
            tile_type: Tile type ID (0-37)
            tile_x: Tile X coordinate
            tile_y: Tile Y coordinate

        Returns:
            True if position is in non-solid area (dark grey), False otherwise
        """
        # Calculate position relative to tile origin
        rel_x = pixel_x - (tile_x * CELL_SIZE)
        rel_y = pixel_y - (tile_y * CELL_SIZE)

        # Determine which sub-node this position corresponds to
        # Sub-nodes are at (6,6), (18,6), (6,18), (18,18)
        sub_x = 0 if rel_x < 12 else 1
        sub_y = 0 if rel_y < 12 else 1

        # O(1) lookup from precomputed table
        return SUBNODE_VALIDITY_TABLE.get(tile_type, {}).get((sub_x, sub_y), False)

    def _generate_sub_nodes(
        self, tiles: np.ndarray
    ) -> Dict[Tuple[int, int], Tuple[int, int, int, int]]:
        """
        Generate all sub-nodes for level, filtering by dark grey (non-solid) areas.

        Returns:
            Dict mapping (pixel_x, pixel_y) -> (tile_x, tile_y, sub_x, sub_y)
            where sub_x, sub_y are 0 or 1 indicating position within tile
        """
        height, width = tiles.shape
        sub_nodes = {}

        # Reset debug stats
        if self.debug:
            self.debug_stats["total_tiles"] = height * width
            self.debug_stats["sub_nodes_by_tile_type"] = {}

        for tile_y in range(height):
            for tile_x in range(width):
                tile_type = tiles[tile_y, tile_x]

                # Skip completely solid tiles (type 1)
                if tile_type == 1:
                    continue

                # Track sub-nodes per tile type
                if self.debug:
                    if tile_type not in self.debug_stats["sub_nodes_by_tile_type"]:
                        self.debug_stats["sub_nodes_by_tile_type"][tile_type] = 0

                # Create sub-nodes for this tile, checking if they're in dark grey areas
                for (offset_x, offset_y), (sub_x, sub_y) in zip(
                    SUB_NODE_OFFSETS, SUB_NODE_COORDS
                ):
                    pixel_x = tile_x * CELL_SIZE + offset_x
                    pixel_y = tile_y * CELL_SIZE + offset_y

                    # Check if this sub-node is in dark grey area (non-solid)
                    if not self._is_position_in_non_solid_area(
                        pixel_x, pixel_y, tile_type, tile_x, tile_y
                    ):
                        continue  # Skip sub-nodes in solid areas

                    sub_nodes[(pixel_x, pixel_y)] = (tile_x, tile_y, sub_x, sub_y)

                    # Track debug stats
                    if self.debug:
                        self.debug_stats["sub_nodes_by_tile_type"][tile_type] += 1

        if self.debug:
            self.debug_stats["total_sub_nodes"] = len(sub_nodes)
            self._debug_print_tile_statistics(tiles, sub_nodes)

        return sub_nodes

    def _is_node_grounded(
        self,
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

        This is O(N) where N is the number of neighbors (typically small).

        Args:
            node_pos: Node position (x, y) in pixels
            adjacency: Current adjacency graph being built

        Returns:
            True if node is grounded (on a surface), False otherwise (mid-air)
        """
        x, y = node_pos
        below_pos = (x, y + SUB_NODE_SIZE)  # 12px down (y increases downward)

        # If node directly below doesn't exist in graph, this node is on solid surface
        if below_pos not in adjacency:
            return True

        # Node below exists - check if we have a direct vertical edge to it
        # If we CAN fall to it (edge exists), we're NOT grounded
        # If we CAN'T fall to it (no edge), we ARE grounded
        if node_pos in adjacency:
            neighbors = adjacency[node_pos]
            for neighbor_pos, _ in neighbors:
                if neighbor_pos == below_pos:
                    # Found direct vertical edge downward - NOT grounded (can fall)
                    return False

        # No direct downward edge found - node is grounded
        return True

    def _build_full_adjacency(
        self,
        tiles: np.ndarray,
        all_sub_nodes: Dict[Tuple[int, int], Tuple[int, int, int, int]],
    ) -> Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]:
        """
        Build full adjacency graph without physics filtering.

        This method builds the complete adjacency graph based on tile connectivity
        and traversability, without applying physics-based movement constraints.

        Args:
            tiles: 2D numpy array of tile types
            all_sub_nodes: Dict mapping node positions to (tile_x, tile_y, sub_x, sub_y)

        Returns:
            Full adjacency dictionary for all traversable sub-nodes
        """
        adjacency = {}

        # Direction mappings for 8-connectivity (cardinal and diagonal directions)
        directions = {
            "N": (0, -12),
            "E": (12, 0),
            "S": (0, 12),
            "W": (-12, 0),
            "NE": (12, -12),
            "SE": (12, 12),
            "SW": (-12, 12),
            "NW": (-12, -12),
        }

        # Build adjacency for all nodes
        for current_pos, (tile_x, tile_y, sub_x, sub_y) in all_sub_nodes.items():
            current_x, current_y = current_pos
            tile_type = tiles[tile_y, tile_x]

            neighbors = []

            # Check each direction for potential neighbors
            for dir_name, (dx, dy) in directions.items():
                neighbor_x = current_x + dx
                neighbor_y = current_y + dy
                neighbor_pos = (neighbor_x, neighbor_y)

                # Check if neighbor sub-node exists
                if neighbor_pos not in all_sub_nodes:
                    continue

                n_tile_x, n_tile_y, n_sub_x, n_sub_y = all_sub_nodes[neighbor_pos]
                neighbor_tile_type = tiles[n_tile_y, n_tile_x]

                # Check traversability using precomputed connectivity
                if self._is_sub_node_traversable(
                    tile_type,
                    neighbor_tile_type,
                    tile_x,
                    tile_y,
                    n_tile_x,
                    n_tile_y,
                    dir_name,
                    tiles,
                    current_x,
                    current_y,
                    neighbor_x,
                    neighbor_y,
                ):
                    # Use uniform cost in graph building
                    # Physics-aware costs are now applied during pathfinding
                    cost = 1.0

                    neighbors.append((neighbor_pos, cost))

            adjacency[current_pos] = neighbors

        return adjacency

    def _apply_physics_filtering(
        self,
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    ) -> Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]:
        """
        Apply physics-based filtering to remove invalid movement edges.

        NOTE: Physics constraints (horizontal grounding, diagonal upward grounding)
        have been moved to pathfinding for more flexible path planning.
        This method currently returns the adjacency graph unchanged but is kept
        for future physics constraints that should apply at the graph level.

        Args:
            adjacency: Full adjacency graph

        Returns:
            Adjacency graph (currently unchanged, physics applied during pathfinding)
        """
        # Physics filtering moved to pathfinding - return graph unchanged
        # This allows more flexible path planning with physics constraints applied
        # at search time rather than graph build time
        return adjacency

    def _build_reachable_adjacency(
        self,
        tiles: np.ndarray,
        all_sub_nodes: Dict[Tuple[int, int], Tuple[int, int, int, int]],
        player_spawn: Tuple[int, int],
    ) -> Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]:
        """
        Build adjacency graph with uniform edge costs.

        Two-phase approach:
        1. Build full adjacency based on tile connectivity (uniform cost)
        2. Apply physics filtering (currently none, kept for extensibility)

        Physics constraints (costs, grounding rules) are now applied during
        pathfinding for more flexible path planning.

        Note: Filtering to only reachable nodes from initial player position
        is done in build_graph() after applying entity masks and flood fill.

        Args:
            tiles: 2D numpy array of tile types
            all_sub_nodes: Dict mapping node positions to (tile_x, tile_y, sub_x, sub_y)
            player_spawn: Player spawn position (used for naming, filtering done later)

        Returns:
            Adjacency dictionary with uniform edge costs
        """
        # Phase 1: Build full adjacency without physics filtering
        adjacency_full = self._build_full_adjacency(tiles, all_sub_nodes)

        # Phase 2: Apply physics filtering to remove invalid edges
        adjacency_filtered = self._apply_physics_filtering(adjacency_full)

        return adjacency_filtered

    def _find_closest_node(
        self,
        pos: Tuple[int, int],
        sub_nodes: Dict[Tuple[int, int], Tuple[int, int, int, int]],
        adjacency: Optional[
            Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]
        ] = None,
        prefer_grounded: bool = False,
    ) -> Optional[Tuple[int, int]]:
        """
        Find closest sub-node to given position - O(1) with grid snapping.

        Uses the fact that sub-nodes are on a 12px grid at positions:
        tile*24 + {6, 18} for each dimension.

        Args:
            pos: Query position (x, y) in pixels
            sub_nodes: Dict mapping node positions to tile info
            adjacency: Optional adjacency graph for grounding checks
            prefer_grounded: If True and adjacency provided, prefer grounded nodes
        """
        if not sub_nodes:
            return None

        px, py = pos

        # Fast path: Snap to nearest 12px grid position - O(1)
        # Sub-nodes are at: ..., -18, -6, 6, 18, 30, 42, ...
        snap_x = round((px - 6) / 12) * 12 + 6
        snap_y = round((py - 6) / 12) * 12 + 6

        # Check if snapped position exists
        if (snap_x, snap_y) in sub_nodes:
            snapped_node = (snap_x, snap_y)
            # If not preferring grounded or no adjacency, return immediately
            if not prefer_grounded or adjacency is None:
                return snapped_node
            # If snapped node is grounded, return it
            if self._is_node_grounded(snapped_node, adjacency):
                return snapped_node
            # Otherwise, continue searching for grounded alternative

        # Collect candidates from 8 neighboring sub-nodes
        candidates = []
        if (snap_x, snap_y) in sub_nodes:
            candidates.append((snap_x, snap_y))

        for dx in [-12, 0, 12]:
            for dy in [-12, 0, 12]:
                if dx == 0 and dy == 0:
                    continue  # Already checked above
                candidate = (snap_x + dx, snap_y + dy)
                if candidate in sub_nodes:
                    candidates.append(candidate)

        # If preferring grounded and we have adjacency, separate candidates
        if prefer_grounded and adjacency is not None and candidates:
            grounded_candidates = []
            air_candidates = []

            for candidate in candidates:
                if self._is_node_grounded(candidate, adjacency):
                    grounded_candidates.append(candidate)
                else:
                    air_candidates.append(candidate)

            # Prefer grounded candidates, fall back to air if none grounded
            search_list = grounded_candidates if grounded_candidates else air_candidates

            # Among preferred candidates, find closest
            min_dist = float("inf")
            closest = None
            for candidate in search_list:
                nx, ny = candidate
                dist_sq = (nx - px) ** 2 + (ny - py) ** 2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    closest = candidate

            if closest is not None:
                return closest

        # Standard path: return closest candidate without grounding preference
        if candidates:
            min_dist = float("inf")
            closest = None
            for candidate in candidates:
                nx, ny = candidate
                dist_sq = (nx - px) ** 2 + (ny - py) ** 2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    closest = candidate
            return closest

        # Rare fallback: Linear search if grid snapping fails
        # This only happens if pos is very far from any sub-node
        closest_node = None
        min_dist = float("inf")
        for node_pos in sub_nodes.keys():
            nx, ny = node_pos
            dist_sq = (nx - px) ** 2 + (ny - py) ** 2
            if dist_sq < min_dist:
                min_dist = dist_sq
                closest_node = node_pos

        return closest_node

    def _find_nodes_within_radius(
        self,
        pos: Tuple[int, int],
        sub_nodes: Dict[Tuple[int, int], Tuple[int, int, int, int]],
        radius: float = PLAYER_RADIUS,
        adjacency: Optional[
            Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]
        ] = None,
        prefer_grounded: bool = False,
    ) -> List[Tuple[int, int]]:
        """
        Find all sub-nodes within radius of given position.

        This accounts for the 10px player circle - any node within the player's
        collision radius should be considered a valid starting position.

        Args:
            pos: Center position (player position)
            sub_nodes: Dictionary of all sub-nodes
            radius: Search radius in pixels (default: PLAYER_RADIUS = 10px)
            adjacency: Optional adjacency graph for grounding checks
            prefer_grounded: If True and adjacency provided, prioritize grounded nodes

        Returns:
            List of node positions within radius, sorted by priority (grounded first) and distance
        """
        if not sub_nodes:
            return []

        px, py = pos
        radius_sq = radius * radius
        grounded_nodes = []
        air_nodes = []

        # Fast grid-based search: only check sub-nodes in nearby tiles
        # Player radius is 10px, sub-nodes are 12px apart
        # So we need to check within ceil(10/12) = 1 tile in each direction
        snap_x = round((px - 6) / 12) * 12 + 6
        snap_y = round((py - 6) / 12) * 12 + 6

        # Check 5x5 grid around snapped position (covers 60x60px area)
        for dx in [-24, -12, 0, 12, 24]:
            for dy in [-24, -12, 0, 12, 24]:
                candidate = (snap_x + dx, snap_y + dy)
                if candidate in sub_nodes:
                    nx, ny = candidate
                    dist_sq = (nx - px) ** 2 + (ny - py) ** 2
                    if dist_sq <= radius_sq:
                        # Separate grounded vs air nodes if preferring grounded
                        if prefer_grounded and adjacency is not None:
                            if self._is_node_grounded(candidate, adjacency):
                                grounded_nodes.append((candidate, dist_sq))
                            else:
                                air_nodes.append((candidate, dist_sq))
                        else:
                            grounded_nodes.append((candidate, dist_sq))

        # Sort each category by distance
        grounded_nodes.sort(key=lambda x: x[1])
        air_nodes.sort(key=lambda x: x[1])

        # Prioritize grounded nodes if preference enabled
        if prefer_grounded and adjacency is not None:
            # Return grounded first, then air
            result = [node for node, _ in grounded_nodes] + [
                node for node, _ in air_nodes
            ]
        else:
            # Just return all nodes sorted by distance
            result = [node for node, _ in grounded_nodes]

        return result

    def _is_sub_node_traversable(
        self,
        src_tile_type: int,
        dst_tile_type: int,
        src_tile_x: int,
        src_tile_y: int,
        dst_tile_x: int,
        dst_tile_y: int,
        direction: str,
        tiles: np.ndarray,
        src_pixel_x: int,
        src_pixel_y: int,
        dst_pixel_x: int,
        dst_pixel_y: int,
    ) -> bool:
        """
        Check if movement from source sub-node to destination sub-node is traversable.

        Uses precomputed 4-connectivity (cardinal directions only) for tile-to-tile movement and verifies
        both sub-nodes are in dark grey (non-solid) areas.
        """
        # Track debug stats
        if self.debug:
            self.debug_stats["traversability_checks"] += 1

        # Check both sub-nodes are in dark grey areas (non-solid)
        src_valid = self._is_position_in_non_solid_area(
            src_pixel_x, src_pixel_y, src_tile_type, src_tile_x, src_tile_y
        )
        dst_valid = self._is_position_in_non_solid_area(
            dst_pixel_x, dst_pixel_y, dst_tile_type, dst_tile_x, dst_tile_y
        )

        if not src_valid:
            if self.debug:
                self.debug_stats["blocked_by_geometry"] += 1
                self._debug_log_traversability_decision(
                    src_tile_type,
                    dst_tile_type,
                    direction,
                    False,
                    "Source sub-node in solid area",
                    src_pos=f"({src_pixel_x},{src_pixel_y})",
                    src_tile=f"({src_tile_x},{src_tile_y})",
                )
            return False
        if not dst_valid:
            if self.debug:
                self.debug_stats["blocked_by_geometry"] += 1
                self._debug_log_traversability_decision(
                    src_tile_type,
                    dst_tile_type,
                    direction,
                    False,
                    "Destination sub-node in solid area",
                    dst_pos=f"({dst_pixel_x},{dst_pixel_y})",
                    dst_tile=f"({dst_tile_x},{dst_tile_y})",
                )
            return False

        # Calculate tile-to-tile direction
        tile_dx = dst_tile_x - src_tile_x
        tile_dy = dst_tile_y - src_tile_y

        # If in same tile, check within-tile connectivity (O(1) lookup)
        if tile_dx == 0 and tile_dy == 0:
            if self.debug:
                self.debug_stats["within_tile_checks"] += 1

            # Get sub-node indices for source and destination
            src_rel_x = src_pixel_x - (src_tile_x * CELL_SIZE)
            src_rel_y = src_pixel_y - (src_tile_y * CELL_SIZE)
            src_sub_x = 0 if src_rel_x < 12 else 1
            src_sub_y = 0 if src_rel_y < 12 else 1

            dst_rel_x = dst_pixel_x - (dst_tile_x * CELL_SIZE)
            dst_rel_y = dst_pixel_y - (dst_tile_y * CELL_SIZE)
            dst_sub_x = 0 if dst_rel_x < 12 else 1
            dst_sub_y = 0 if dst_rel_y < 12 else 1

            # O(1) lookup in precomputed within-tile connectivity table
            reachable = WITHIN_TILE_CONNECTIVITY.get(src_tile_type, {}).get(
                (src_sub_x, src_sub_y), set()
            )
            result = (dst_sub_x, dst_sub_y) in reachable

            if self.debug and not result:
                self.debug_stats["blocked_by_connectivity"] += 1
                self._debug_log_traversability_decision(
                    src_tile_type,
                    dst_tile_type,
                    "SAME_TILE",
                    result,
                    f"Within-tile blocked: {(src_sub_x, src_sub_y)} → {(dst_sub_x, dst_sub_y)}",
                )

            return result

        if self.debug:
            self.debug_stats["between_tile_checks"] += 1

        # Calculate sub-node indices from pixel positions
        src_sub_x = 0 if (src_pixel_x % CELL_SIZE) < 12 else 1
        src_sub_y = 0 if (src_pixel_y % CELL_SIZE) < 12 else 1
        dst_sub_x = 0 if (dst_pixel_x % CELL_SIZE) < 12 else 1
        dst_sub_y = 0 if (dst_pixel_y % CELL_SIZE) < 12 else 1

        # For diagonal movements, check intermediate tiles aren't blocking
        if direction in ["NE", "SE", "SW", "NW"]:
            if not self._check_diagonal_clear(
                src_tile_x,
                src_tile_y,
                dst_tile_x,
                dst_tile_y,
                src_sub_x,
                src_sub_y,
                dst_sub_x,
                dst_sub_y,
                tiles,
            ):
                if self.debug:
                    self.debug_stats["blocked_by_diagonal"] += 1
                    self._debug_log_traversability_decision(
                        src_tile_type,
                        dst_tile_type,
                        direction,
                        False,
                        "Diagonal movement blocked by intermediate tiles",
                    )
                return False

        # For cardinal cross-tile movements, check if line crosses solid geometry
        # in either source or destination tile (slopes, quarter circles, etc.)
        if direction in ["N", "S", "E", "W"]:
            if not _check_cross_tile_line_clear(
                src_pixel_x,
                src_pixel_y,
                dst_pixel_x,
                dst_pixel_y,
                src_tile_type,
                dst_tile_type,
                src_tile_x,
                src_tile_y,
                dst_tile_x,
                dst_tile_y,
            ):
                if self.debug:
                    self.debug_stats["blocked_by_geometry"] += 1
                    self._debug_log_traversability_decision(
                        src_tile_type,
                        dst_tile_type,
                        direction,
                        False,
                        "Cardinal cross-tile movement blocked by slope/geometry",
                    )
                return False

        # Use precomputed sub-node-aware connectivity for tile-to-tile movement
        try:
            result = self.connectivity_loader.is_traversable(
                src_tile_type,
                dst_tile_type,
                direction,
                src_sub_x,
                src_sub_y,
                dst_sub_x,
                dst_sub_y,
            )
            if self.debug and not result:
                self.debug_stats["blocked_by_connectivity"] += 1
                self._debug_log_traversability_decision(
                    src_tile_type,
                    dst_tile_type,
                    direction,
                    result,
                    f"Precomputed connectivity blocked (sub-nodes: ({src_sub_x},{src_sub_y})→({dst_sub_x},{dst_sub_y}))",
                )
            return result
        except Exception as e:
            # Fallback: disallow if connectivity check fails
            if self.debug:
                self.debug_stats["blocked_by_connectivity"] += 1
                self._debug_log_traversability_decision(
                    src_tile_type,
                    dst_tile_type,
                    direction,
                    False,
                    f"Connectivity check exception: {e}",
                )
            return False

    def _check_diagonal_clear(
        self,
        src_x: int,
        src_y: int,
        dst_x: int,
        dst_y: int,
        src_sub_x: int,
        src_sub_y: int,
        dst_sub_x: int,
        dst_sub_y: int,
        tiles: np.ndarray,
    ) -> bool:
        """
        Check if diagonal movement doesn't cut through solid corners.

        Enhanced to check specific sub-node positions in intermediate tiles,
        not just whether tiles are type 1 (fully solid).

        For diagonal movement, we check if the specific corner being used
        in each intermediate tile is traversable.
        """
        height, width = tiles.shape

        dx = dst_x - src_x
        dy = dst_y - src_y

        # Get intermediate tiles
        side_x = src_x + dx  # Horizontal neighbor
        side_y = src_y
        vert_x = src_x  # Vertical neighbor
        vert_y = src_y + dy

        # Bounds check
        if not (0 <= side_x < width and 0 <= side_y < height):
            return False
        if not (0 <= vert_x < width and 0 <= vert_y < height):
            return False

        side_tile = tiles[side_y, side_x]
        vert_tile = tiles[vert_y, vert_x]

        # If both are fully solid, definitely blocked
        if side_tile == 1 and vert_tile == 1:
            return False

        # For NE movement: check if we can pass through the corner
        if dx == 1 and dy == -1:  # Northeast
            # Moving from bottom-left region to top-right region
            # Need: East tile's bottom-left corner clear AND North tile's bottom-right corner clear
            # East tile: check sub-node at position that aligns with source
            side_check_x = 0  # Left side of East tile
            side_check_y = src_sub_y  # Same Y as source
            side_pixel_x = side_x * CELL_SIZE + (6 if side_check_x == 0 else 18)
            side_pixel_y = side_y * CELL_SIZE + (6 if side_check_y == 0 else 18)

            # North tile: check sub-node at position that aligns with source
            vert_check_x = src_sub_x  # Same X as source
            vert_check_y = 1  # Bottom of North tile
            vert_pixel_x = vert_x * CELL_SIZE + (6 if vert_check_x == 0 else 18)
            vert_pixel_y = vert_y * CELL_SIZE + (6 if vert_check_y == 0 else 18)

        elif dx == 1 and dy == 1:  # Southeast
            # East tile: check top-left
            side_check_x = 0
            side_check_y = src_sub_y
            side_pixel_x = side_x * CELL_SIZE + (6 if side_check_x == 0 else 18)
            side_pixel_y = side_y * CELL_SIZE + (6 if side_check_y == 0 else 18)

            # South tile: check top-right
            vert_check_x = src_sub_x
            vert_check_y = 0
            vert_pixel_x = vert_x * CELL_SIZE + (6 if vert_check_x == 0 else 18)
            vert_pixel_y = vert_y * CELL_SIZE + (6 if vert_check_y == 0 else 18)

        elif dx == -1 and dy == 1:  # Southwest
            # West tile: check top-right
            side_check_x = 1
            side_check_y = src_sub_y
            side_pixel_x = side_x * CELL_SIZE + (6 if side_check_x == 0 else 18)
            side_pixel_y = side_y * CELL_SIZE + (6 if side_check_y == 0 else 18)

            # South tile: check top-left
            vert_check_x = src_sub_x
            vert_check_y = 0
            vert_pixel_x = vert_x * CELL_SIZE + (6 if vert_check_x == 0 else 18)
            vert_pixel_y = vert_y * CELL_SIZE + (6 if vert_check_y == 0 else 18)

        elif dx == -1 and dy == -1:  # Northwest
            # West tile: check bottom-right
            side_check_x = 1
            side_check_y = src_sub_y
            side_pixel_x = side_x * CELL_SIZE + (6 if side_check_x == 0 else 18)
            side_pixel_y = side_y * CELL_SIZE + (6 if side_check_y == 0 else 18)

            # North tile: check bottom-left
            vert_check_x = src_sub_x
            vert_check_y = 1
            vert_pixel_x = vert_x * CELL_SIZE + (6 if vert_check_x == 0 else 18)
            vert_pixel_y = vert_y * CELL_SIZE + (6 if vert_check_y == 0 else 18)
        else:
            # Not a diagonal, shouldn't reach here
            return True

        # Check if both positions in intermediate tiles are traversable
        side_clear = self._is_position_in_non_solid_area(
            side_pixel_x, side_pixel_y, side_tile, side_x, side_y
        )
        vert_clear = self._is_position_in_non_solid_area(
            vert_pixel_x, vert_pixel_y, vert_tile, vert_x, vert_y
        )

        # Need BOTH paths clear for valid diagonal movement (can't cut through corners)
        # In N++, diagonal movement requires both intermediate tile corners to be traversable
        return side_clear and vert_clear

    def _apply_entity_mask(
        self,
        base_adjacency: Dict,
        blocked_pixel_positions: Set[Tuple[int, int]],
        blocked_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
    ) -> Dict:
        """
        Apply entity state mask to base adjacency graph.

        Args:
            base_adjacency: Base adjacency graph
            blocked_pixel_positions: Set of blocked pixel positions (sub-nodes)
            blocked_edges: Set of blocked edges

        Returns:
            Filtered adjacency graph
        """
        # Build filtered adjacency
        filtered = {}

        for pos, neighbors in base_adjacency.items():
            # Skip if position itself is blocked
            if pos in blocked_pixel_positions:
                continue

            # Filter neighbors
            valid_neighbors = []
            for neighbor_pos, cost in neighbors:
                # Skip if neighbor is blocked
                if neighbor_pos in blocked_pixel_positions:
                    continue

                # Skip if edge is blocked
                edge = (pos, neighbor_pos)
                if edge in blocked_edges:
                    continue

                valid_neighbors.append((neighbor_pos, cost))

            filtered[pos] = valid_neighbors

        return filtered

    def _filter_adjacency_to_reachable(
        self,
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        reachable_nodes: Set[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]:
        """
        Filter adjacency graph to only include nodes reachable from initial player position.

        This ensures path caching only operates on areas reachable from the starting position,
        preventing caching of distances for isolated or unreachable areas.

        Args:
            adjacency: Full adjacency graph (may include unreachable nodes)
            reachable_nodes: Set of node positions reachable from initial player position

        Returns:
            Filtered adjacency graph containing only reachable nodes and edges between them
        """
        filtered = {}
        for pos, neighbors in adjacency.items():
            # Only include nodes that are reachable
            if pos not in reachable_nodes:
                continue

            # Filter neighbors to only include reachable nodes
            valid_neighbors = [
                (neighbor_pos, cost)
                for neighbor_pos, cost in neighbors
                if neighbor_pos in reachable_nodes
            ]

            # Only add node if it has valid neighbors (or if it's isolated but reachable)
            if valid_neighbors:
                filtered[pos] = valid_neighbors
            elif pos in reachable_nodes:
                # Include isolated reachable nodes (no neighbors but still reachable)
                filtered[pos] = []

        return filtered

    def _get_cached_flood_fill(
        self,
        level_id: str,
        ninja_pos: Tuple[int, int],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        spatial_hash: Any,
    ) -> Set[Tuple[int, int]]:
        """
        Get flood fill results with caching based on player position.

        Only recomputes flood fill when player moves >= 12px from last cached position.
        This optimization avoids expensive BFS operations when player moves small amounts.

        Args:
            level_id: Level identifier for cache key
            ninja_pos: Current ninja position (x, y) in pixels
            adjacency: Graph adjacency structure
            spatial_hash: SpatialHash for optimization

        Returns:
            Set of reachable node positions from ninja_pos
        """
        # Check if we have cached flood fill for this level
        if level_id in self._flood_fill_cache:
            last_ninja_pos, cached_reachable = self._flood_fill_cache[level_id]

            # Calculate distance moved since last flood fill
            dx = ninja_pos[0] - last_ninja_pos[0]
            dy = ninja_pos[1] - last_ninja_pos[1]
            distance_moved = (dx * dx + dy * dy) ** 0.5

            # If player hasn't moved significantly, return cached result
            if distance_moved < self._flood_fill_distance_threshold:
                # Move to end to maintain LRU ordering
                self._flood_fill_cache.move_to_end(level_id)
                return cached_reachable

        # Need to recompute flood fill (either no cache or player moved enough)
        reachable = flood_fill_reachable_nodes(
            ninja_pos,
            adjacency,
            spatial_hash=spatial_hash,
            subcell_lookup=None,  # Will be auto-loaded by flood_fill
            player_radius=PLAYER_RADIUS,
        )

        # Cache the result with LRU eviction
        self._flood_fill_cache[level_id] = (ninja_pos, reachable)
        self._flood_fill_cache.move_to_end(level_id)

        # Evict oldest entry if cache exceeds max size
        if len(self._flood_fill_cache) > self._max_cache_size:
            self._flood_fill_cache.popitem(last=False)

        return reachable

    def clear_cache(self):
        """Clear level graph cache (call on environment reset)."""
        self._level_graph_cache.clear()
        self._flood_fill_cache.clear()
        self._current_level_id = None

    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_time = np.mean(self.build_times) if self.build_times else 0
        max_time = np.max(self.build_times) if self.build_times else 0

        return {
            "total_builds": len(self.build_times),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0,
            "avg_build_time_ms": avg_time,
            "max_build_time_ms": max_time,
            "cached_levels": len(self._level_graph_cache),
        }
