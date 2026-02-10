"""Position validity checking for value function precomputation.

This module computes a validity mask indicating which grid positions are
physically valid for the ninja to occupy. Invalid positions include:
- Positions where the ninja center falls inside solid tile geometry
- Positions where the ninja overlaps with toggled toggle mines

The validity check uses the definitive tile geometry from tile_definitions.py
and considers the ninja's radius (NINJA_RADIUS=10) when checking overlap.
"""

import numpy as np
import math
from typing import Tuple

from ..tile_definitions import (
    TILE_SEGMENT_DIAG_MAP,
    TILE_SEGMENT_CIRCULAR_MAP,
    TILE_SEGMENT_ORTHO_MAP,
)
from ..constants.physics_constants import NINJA_RADIUS, TOGGLE_MINE_RADII
from ..physics import overlap_circle_vs_circle


def compute_position_validity_mask(sim, discretizer) -> np.ndarray:
    """Compute boolean mask of valid ninja positions.
    
    A position is invalid if:
    1. The ninja's center (radius NINJA_RADIUS=10) would clip into solid tile geometry
    2. The ninja would overlap with a toggled toggle mine (deadly state)
    
    Args:
        sim: Simulator instance with loaded level (tile_dic, entity_dic)
        discretizer: StateDiscretizer for grid dimensions and conversions
    
    Returns:
        Boolean array of shape (nx, ny), True = valid position, False = invalid
    """
    config = discretizer.config
    valid_mask = np.ones((config.nx, config.ny), dtype=bool)
    
    # Check each grid position
    for ix in range(config.nx):
        for iy in range(config.ny):
            # Get continuous position at grid cell center
            state = discretizer.index_to_state((ix, iy, 0, 0, 0))
            px, py = state[0], state[1]
            
            # Check tile geometry overlap
            if _overlaps_tile_geometry(px, py, sim.tile_dic):
                valid_mask[ix, iy] = False
                continue
            
            # Check toggled toggle mine overlap
            if _overlaps_toggled_mines(px, py, sim.entity_dic):
                valid_mask[ix, iy] = False
    
    return valid_mask


def _overlaps_tile_geometry(px: float, py: float, tile_dic: dict) -> bool:
    """Check if ninja at (px, py) overlaps with any solid tile geometry.
    
    Considers all tiles within NINJA_RADIUS + tile_size range, since the ninja's
    circular collision shape can extend into adjacent tiles.
    
    Args:
        px, py: Ninja center position in pixels
        tile_dic: Dictionary mapping (grid_x, grid_y) to tile type
    
    Returns:
        True if ninja overlaps solid geometry, False otherwise
    """
    result = _overlaps_tile_geometry_debug(px, py, tile_dic)
    return result is not None


def _overlaps_tile_geometry_debug(px: float, py: float, tile_dic: dict):
    """Check if ninja at (px, py) overlaps with any solid tile geometry.
    
    Returns:
        None if no overlap, or the tile type that caused the overlap
    """
    # Determine which grid cells to check (3x3 neighborhood around ninja)
    # Tile grid coordinates (each tile is 24x24 pixels, offset by +1 for border)
    center_gx = int(px / 24)
    center_gy = int(py / 24)
    
    # Check 3x3 neighborhood (handles ninja radius extending into adjacent tiles)
    for dgx in range(-1, 2):
        for dgy in range(-1, 2):
            gx = center_gx + dgx
            gy = center_gy + dgy
            
            tile_type = tile_dic.get((gx, gy), 0)
            
            if tile_type == 0 or tile_type > 33:
                # Empty or glitched tiles (types 34-37) - no collision
                continue
            
            # Check if ninja at (px, py) overlaps this tile's solid region
            if _circle_overlaps_tile_solid_region(px, py, NINJA_RADIUS, gx, gy, tile_type):
                return tile_type
    
    return None


def _overlaps_toggled_mines(px: float, py: float, entity_dic: dict) -> bool:
    """Check if ninja at (px, py) overlaps with any toggled toggle mine.
    
    Toggle mines in toggled state (state=0) are deadly. We check overlap using
    circular collision detection.
    
    Args:
        px, py: Ninja center position in pixels
        entity_dic: Dictionary mapping entity type to list of entities
    
    Returns:
        True if ninja overlaps a toggled mine, False otherwise
    """
    # Check toggle mine types (1 and 21)
    for entity_type in [1, 21]:
        if entity_type in entity_dic:
            for mine in entity_dic[entity_type]:
                # Only check mines in toggled state (state=0, deadly)
                if hasattr(mine, 'state') and mine.state == 0:
                    mine_radius = TOGGLE_MINE_RADII[0]  # Toggled radius = 4.0
                    if overlap_circle_vs_circle(
                        mine.xpos, mine.ypos, mine_radius,
                        px, py, NINJA_RADIUS
                    ):
                        return True
    
    return False


def _circle_overlaps_tile_solid_region(
    px: float, py: float, radius: float,
    gx: int, gy: int, tile_type: int
) -> bool:
    """Check if a circle at (px, py) with given radius overlaps a tile's solid region.
    
    Uses definitive tile geometry from tile_definitions.py for all 34 tile types.
    
    Args:
        px, py: Circle center position in pixels
        radius: Circle radius (NINJA_RADIUS = 10)
        gx, gy: Tile grid coordinates
        tile_type: Tile type (1-33, types 0 and 34-37 should be filtered out)
    
    Returns:
        True if circle overlaps tile's solid region, False otherwise
    """
    # Tile corner in pixel space
    tile_x = gx * 24
    tile_y = gy * 24
    
    # Type 1: Full solid tile
    if tile_type == 1:
        return _circle_overlaps_rectangle(
            px, py, radius,
            tile_x, tile_y, 24, 24
        )
    
    # Types 2-5: Half tiles
    if tile_type == 2:  # Top half
        return _circle_overlaps_rectangle(px, py, radius, tile_x, tile_y, 24, 12)
    if tile_type == 3:  # Right half
        return _circle_overlaps_rectangle(px, py, radius, tile_x + 12, tile_y, 12, 24)
    if tile_type == 4:  # Bottom half
        return _circle_overlaps_rectangle(px, py, radius, tile_x, tile_y + 12, 24, 12)
    if tile_type == 5:  # Left half
        return _circle_overlaps_rectangle(px, py, radius, tile_x, tile_y, 12, 24)
    
    # Types 6-9: 45-degree slopes
    if tile_type in TILE_SEGMENT_DIAG_MAP:
        return _circle_overlaps_diagonal_tile(
            px, py, radius, tile_x, tile_y, tile_type
        )
    
    # Types 10-13: Quarter circles (convex)
    # Types 14-17: Quarter pipes (concave)
    if tile_type in TILE_SEGMENT_CIRCULAR_MAP:
        return _circle_overlaps_circular_tile(
            px, py, radius, tile_x, tile_y, tile_type
        )
    
    # Types 18-33: Mild and steep slopes
    if tile_type >= 18 and tile_type <= 33:
        return _circle_overlaps_slope_tile(
            px, py, radius, tile_x, tile_y, tile_type
        )
    
    # Unknown tile type - conservatively return False
    return False


def _circle_overlaps_rectangle(
    px: float, py: float, radius: float,
    rect_x: float, rect_y: float, rect_w: float, rect_h: float
) -> bool:
    """Check if circle overlaps axis-aligned rectangle.
    
    Uses the standard circle-rectangle collision algorithm: find closest point
    on rectangle to circle center, then check distance.
    """
    # Find closest point on rectangle to circle center
    closest_x = max(rect_x, min(px, rect_x + rect_w))
    closest_y = max(rect_y, min(py, rect_y + rect_h))
    
    # Check distance from circle center to closest point
    dx = px - closest_x
    dy = py - closest_y
    dist_sq = dx * dx + dy * dy
    
    return dist_sq < radius * radius


def _circle_overlaps_diagonal_tile(
    px: float, py: float, radius: float,
    tile_x: float, tile_y: float, tile_type: int
) -> bool:
    """Check if circle overlaps diagonal slope tile (types 6-9, 18-33).
    
    Diagonal tiles have a diagonal line segment dividing the tile into solid/empty.
    The solid region is a triangle or trapezoid bounded by the diagonal and tile edges.
    """
    # Get diagonal endpoints from tile_definitions
    (p1x, p1y), (p2x, p2y) = TILE_SEGMENT_DIAG_MAP[tile_type]
    
    # Convert to world coordinates
    p1x_world = tile_x + p1x
    p1y_world = tile_y + p1y
    p2x_world = tile_x + p2x
    p2y_world = tile_y + p2y
    
    # Determine which side of the diagonal is solid based on tile type
    # For types 6-9 (45-degree slopes), use the diagonal direction
    # For types 18-33 (mild/steep slopes), check TILE_SEGMENT_ORTHO_MAP for edges
    
    # Check if circle center is on the solid side of the diagonal
    # Use cross product to determine side
    dx_line = p2x_world - p1x_world
    dy_line = p2y_world - p1y_world
    dx_point = px - p1x_world
    dy_point = py - p1y_world
    cross = dx_line * dy_point - dy_line * dx_point
    
    # Determine solid side based on tile type
    # Types 6, 7: solid below/right of diagonal
    # Types 8, 9: solid above/left of diagonal
    # For slopes (18-33), the orthogonal edges tell us which side is solid
    solid_side = _get_solid_side_for_diagonal(tile_type)
    
    if solid_side == 1:  # Solid on positive side of diagonal
        on_solid_side = cross > 0
    else:  # Solid on negative side of diagonal
        on_solid_side = cross < 0
    
    if not on_solid_side:
        # Circle center is on empty side; check if circle is close enough to diagonal
        # to potentially overlap the solid region
        dist_to_line = abs(cross) / math.sqrt(dx_line * dx_line + dy_line * dy_line)
        return dist_to_line < radius
    else:
        # Circle center is on solid side; need to check if it's within the tile bounds
        # and close enough to the solid region
        return _point_in_tile_bounds(px, py, tile_x, tile_y, 24, 24) or \
               _circle_near_tile_edge(px, py, radius, tile_x, tile_y, 24, 24)


def _circle_overlaps_circular_tile(
    px: float, py: float, radius: float,
    tile_x: float, tile_y: float, tile_type: int
) -> bool:
    """Check if circle overlaps circular tile (types 10-17).
    
    Types 10-13 are quarter circles (convex corners) - solid in the arc region.
    Types 14-17 are quarter pipes (concave corners) - hollow in the arc region.
    """
    center, quadrant, is_convex = TILE_SEGMENT_CIRCULAR_MAP[tile_type]
    
    # Arc center in world coordinates
    arc_cx = tile_x + center[0]
    arc_cy = tile_y + center[1]
    
    # Distance from ninja center to arc center
    dx = px - arc_cx
    dy = py - arc_cy
    dist = math.sqrt(dx * dx + dy * dy)
    
    # Arc radius is 24px (tile size)
    arc_radius = 24.0
    
    # Check if point is in the correct quadrant
    qx, qy = quadrant
    in_quadrant = (dx * qx >= 0) and (dy * qy >= 0)
    
    if is_convex:
        # Quarter circle (types 10-13): solid inside the arc within the quadrant
        if in_quadrant and dist < arc_radius:
            # Point is inside the convex arc - this is solid
            return True
        # Check if circle edge is close to arc boundary
        if in_quadrant and abs(dist - arc_radius) < radius:
            return True
    else:
        # Quarter pipe (types 14-17): hollow inside the arc, solid outside in the quadrant
        if in_quadrant and dist > arc_radius:
            # Point is outside the hollow arc within the quadrant - this is solid
            # Check if we're within the tile bounds
            if _point_in_tile_bounds(px, py, tile_x, tile_y, 24, 24):
                return True
        # Check if circle overlaps the arc boundary
        if in_quadrant and abs(dist - arc_radius) < radius:
            return True
    
    return False


def _circle_overlaps_slope_tile(
    px: float, py: float, radius: float,
    tile_x: float, tile_y: float, tile_type: int
) -> bool:
    """Check if circle overlaps slope tile (types 18-33: mild/steep slopes).
    
    These tiles have diagonal segments plus orthogonal edges. The solid region
    is typically a trapezoid or complex polygon.
    """
    # Reuse diagonal tile logic, as slopes are also defined by diagonals
    return _circle_overlaps_diagonal_tile(px, py, radius, tile_x, tile_y, tile_type)


def _get_solid_side_for_diagonal(tile_type: int) -> int:
    """Determine which side of a diagonal is solid.
    
    Returns:
        1 if solid is on positive side of cross product, -1 if negative side
    """
    # Types 6-9: 45-degree slopes
    if tile_type == 6:  # Top-left to bottom-right, solid below-right
        return 1
    if tile_type == 7:  # Top-right to bottom-left, solid below-left
        return -1
    if tile_type == 8:  # Bottom-left to top-right, solid above-right
        return -1
    if tile_type == 9:  # Bottom-right to top-left, solid above-left
        return 1
    
    # Types 18-33: Mild and steep slopes
    # Use TILE_SEGMENT_ORTHO_MAP to infer solid side
    # If there are orthogonal segments, the solid side is where those edges are
    ortho = TILE_SEGMENT_ORTHO_MAP.get(tile_type)
    if ortho:
        # Check if there are more edges on top/bottom vs left/right
        # This is a heuristic; for precise handling, we'd need to analyze each type
        # For now, use the diagonal endpoints to infer direction
        (p1x, p1y), (p2x, p2y) = TILE_SEGMENT_DIAG_MAP[tile_type]
        if p1y < p2y:  # Diagonal goes downward
            return 1
        else:
            return -1
    
    return 1  # Default


def _point_in_tile_bounds(
    px: float, py: float,
    tile_x: float, tile_y: float,
    tile_w: float, tile_h: float
) -> bool:
    """Check if point is within tile bounds."""
    return (tile_x <= px <= tile_x + tile_w) and (tile_y <= py <= tile_y + tile_h)


def _circle_near_tile_edge(
    px: float, py: float, radius: float,
    tile_x: float, tile_y: float,
    tile_w: float, tile_h: float
) -> bool:
    """Check if circle is close enough to tile bounds to potentially overlap."""
    return (
        tile_x - radius <= px <= tile_x + tile_w + radius and
        tile_y - radius <= py <= tile_y + tile_h + radius
    )
