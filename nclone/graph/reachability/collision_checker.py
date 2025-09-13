"""
Collision detection utilities for reachability analysis.

This module handles all collision detection logic including:
- Circle vs tile collision detection
- Circle vs shaped tile collision using segments
- Circle vs circular segment collision (quarter-circles)
"""

import math
import numpy as np
from typing import Dict, Tuple, List

from ...constants.physics_constants import TILE_PIXEL_SIZE


class CollisionChecker:
    """Handles collision detection between ninja and level geometry."""

    def __init__(self, debug: bool = False):
        """
        Initialize collision checker.

        Args:
            debug: Enable debug output
        """
        self.debug = debug

    def check_circle_tile_collision(
        self, x: float, y: float, tile_x: int, tile_y: int, radius: float
    ) -> bool:
        """
        Check if a circle collides with a solid tile using simple geometry.

        Args:
            x: Circle center x coordinate
            y: Circle center y coordinate
            tile_x: Tile x coordinate (in tile units)
            tile_y: Tile y coordinate (in tile units)
            radius: Circle radius

        Returns:
            True if collision detected
        """
        # Tile bounds in world coordinates
        tile_left = tile_x * TILE_PIXEL_SIZE
        tile_right = tile_left + TILE_PIXEL_SIZE
        tile_top = tile_y * TILE_PIXEL_SIZE
        tile_bottom = tile_top + TILE_PIXEL_SIZE

        # Find closest point on tile to circle center
        closest_x = max(tile_left, min(x, tile_right))
        closest_y = max(tile_top, min(y, tile_bottom))

        # Check if distance to closest point is less than radius
        dx = x - closest_x
        dy = y - closest_y
        distance_squared = dx * dx + dy * dy

        return distance_squared < (radius * radius)

    def check_circle_shaped_tile_collision(
        self,
        x: float,
        y: float,
        tile_x: int,
        tile_y: int,
        tiles: np.ndarray,
        radius: float,
    ) -> bool:
        """
        Check if a circle collides with a shaped tile using segment-based collision detection.

        Args:
            x: Circle center x coordinate
            y: Circle center y coordinate
            tile_x: Tile x coordinate (in tile units)
            tile_y: Tile y coordinate (in tile units)
            tiles: Tile data array
            radius: Circle radius

        Returns:
            True if collision detected
        """
        from ...utils.tile_segment_factory import TileSegmentFactory
        from ...physics import overlap_circle_vs_segment

        # Get the tile ID
        tile_id = tiles[tile_y, tile_x]

        # Debug output for problematic positions
        debug_pos = abs(x - 135) < 1 and abs(y - 447) < 1
        if debug_pos and self.debug:
            print(
                f"DEBUG SHAPED COLLISION: pos=({x:.1f},{y:.1f}) "
                f"tile=({tile_x},{tile_y}) tile_id={tile_id}"
            )

        # Create a single-tile dictionary for the segment factory
        single_tile = {(tile_x, tile_y): tile_id}

        # Generate segments for this tile
        segment_dict = TileSegmentFactory.create_segment_dictionary(single_tile)

        # Check collision with all segments in this tile
        tile_coord = (tile_x, tile_y)
        if tile_coord in segment_dict:
            segments = segment_dict[tile_coord]
            if debug_pos and self.debug:
                print(
                    f"DEBUG SHAPED COLLISION: Found {len(segments)} segments "
                    f"for tile ({tile_x},{tile_y})"
                )

            for i, segment in enumerate(segments):
                if hasattr(segment, "x1") and hasattr(segment, "y1"):
                    # Linear segment
                    collision = overlap_circle_vs_segment(
                        x, y, radius, segment.x1, segment.y1, segment.x2, segment.y2
                    )
                    if debug_pos and self.debug:
                        print(
                            f"DEBUG SHAPED COLLISION: Linear segment {i}: "
                            f"({segment.x1},{segment.y1})-({segment.x2},{segment.y2}) "
                            f"collision={collision}"
                        )
                    if collision:
                        return True
                elif hasattr(segment, "xpos") and hasattr(segment, "ypos"):
                    # Circular segment
                    collision = self.check_circle_vs_circular_segment(
                        x, y, radius, segment
                    )
                    if debug_pos and self.debug:
                        print(
                            f"DEBUG SHAPED COLLISION: Circular segment {i}: "
                            f"center=({segment.xpos},{segment.ypos}) collision={collision}"
                        )
                    if collision:
                        return True
        elif debug_pos and self.debug:
            print(
                f"DEBUG SHAPED COLLISION: No segments found for tile ({tile_x},{tile_y})"
            )

        return False

    def check_circle_vs_circular_segment(
        self, x: float, y: float, radius: float, segment
    ) -> bool:
        """
        Check if a circle collides with a circular segment (quarter-circle).

        Args:
            x: Circle center x coordinate
            y: Circle center y coordinate
            radius: Circle radius
            segment: Circular segment object with xpos, ypos, radius, convex, hor, ver

        Returns:
            True if collision detected
        """
        # Distance from circle center to arc center
        dx = x - segment.xpos
        dy = y - segment.ypos
        distance = math.sqrt(dx * dx + dy * dy)

        # Check if we're in the right quadrant
        in_quadrant = (dx * segment.hor >= 0) and (dy * segment.ver >= 0)

        if segment.convex:
            # Convex arc (quarter-pipe) - collision if inside the arc and in quadrant
            if in_quadrant and distance < (segment.radius + radius):
                return True
        else:
            # Concave arc (quarter-moon) - collision if outside inner radius but inside outer radius
            if in_quadrant and (segment.radius - radius) < distance < (
                segment.radius + radius
            ):
                return True

        return False

    def point_to_line_segment_distance(
        self, px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """
        Calculate the shortest distance from a point to a line segment.

        Args:
            px: Point x coordinate
            py: Point y coordinate
            x1: Line segment start x coordinate
            y1: Line segment start y coordinate
            x2: Line segment end x coordinate
            y2: Line segment end y coordinate

        Returns:
            Shortest distance from point to line segment
        """
        # Vector from line start to point
        dx_p = px - x1
        dy_p = py - y1

        # Vector along the line
        dx_line = x2 - x1
        dy_line = y2 - y1

        # Length squared of the line segment
        line_length_sq = dx_line * dx_line + dy_line * dy_line

        if line_length_sq < 1e-6:  # Degenerate line segment (point)
            return (dx_p * dx_p + dy_p * dy_p) ** 0.5

        # Project point onto line (parameter t)
        t = (dx_p * dx_line + dy_p * dy_line) / line_length_sq

        # Clamp t to [0, 1] to stay within line segment
        t = max(0.0, min(1.0, t))

        # Find closest point on line segment
        closest_x = x1 + t * dx_line
        closest_y = y1 + t * dy_line

        # Return distance from point to closest point on segment
        dx_closest = px - closest_x
        dy_closest = py - closest_y
        return (dx_closest * dx_closest + dy_closest * dy_closest) ** 0.5
