"""
Position validation and traversability checking for reachability analysis.

This module handles all position-related validation logic including:
- Sub-grid bounds checking
- Tile-based traversability with ninja radius
- Integration with optimized collision detection
"""

import numpy as np
from typing import Optional, Tuple

from ..common import SUB_GRID_WIDTH, SUB_GRID_HEIGHT, SUB_CELL_SIZE
from ..optimized_collision import get_collision_detector
from ...constants.physics_constants import TILE_PIXEL_SIZE


class PositionValidator:
    """Handles position validation and traversability checking."""

    def __init__(self, debug: bool = False):
        """
        Initialize position validator.

        Args:
            debug: Enable debug output
        """
        self.debug = debug
        self.collision_detector = get_collision_detector()

    def initialize_for_level(self, tiles: np.ndarray):
        """
        Initialize collision detector for the current level.

        Args:
            tiles: Level tile data array
        """
        self.collision_detector.initialize_for_level(tiles)

    def is_valid_sub_grid_position(self, sub_row: int, sub_col: int) -> bool:
        """
        Check if sub-grid position is within level bounds.

        Args:
            sub_row: Sub-grid row coordinate
            sub_col: Sub-grid column coordinate

        Returns:
            True if position is within valid sub-grid bounds
        """
        return 0 <= sub_row < SUB_GRID_HEIGHT and 0 <= sub_col < SUB_GRID_WIDTH

    def is_position_traversable(
        self,
        level_data,
        sub_row: int,
        sub_col: int,
        ninja_position_override: Optional[Tuple[float, float]] = None,
    ) -> bool:
        """
        Check if a sub-grid position is traversable by the ninja.

        Uses segment-based collision detection to accurately determine if
        the ninja (10px radius) can occupy the given position.

        Args:
            level_data: Level data containing tiles and entities
            sub_row: Sub-grid row coordinate
            sub_col: Sub-grid column coordinate
            ninja_position_override: If provided, use this position instead of sub-cell center

        Returns:
            True if position can be occupied by ninja
        """
        # Use ninja's actual position if provided, otherwise use sub-cell center
        if ninja_position_override is not None:
            pixel_x, pixel_y = ninja_position_override
        else:
            # Convert to pixel coordinates (center of sub-cell)
            pixel_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            pixel_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2

        # Check bounds first
        tile_x = int(pixel_x // TILE_PIXEL_SIZE)
        tile_y = int(pixel_y // TILE_PIXEL_SIZE)

        # Account for padding: tile data is unpadded, but coordinates assume padding
        # Visual cell (5,18) corresponds to tile_data[17][4] (subtract 1 from both x,y)
        data_tile_x = tile_x - 1
        data_tile_y = tile_y - 1

        if not (
            0 <= data_tile_y < len(level_data.tiles)
            and 0 <= data_tile_x < len(level_data.tiles[0])
        ):
            return False

        # Use proper tile-based traversability check
        # This accounts for the ninja's 10px radius and handles all tile types correctly
        is_traversable = self.is_position_traversable_with_radius(
            pixel_x,
            pixel_y,
            level_data.tiles,
            10.0,  # ninja radius
        )

        if self.debug:
            tile_value = level_data.tiles[data_tile_y][data_tile_x]
            print(
                f"DEBUG: Position ({sub_row}, {sub_col}) -> pixel ({pixel_x}, {pixel_y}) "
                f"tile ({tile_x}, {tile_y}) -> data[{data_tile_y}][{data_tile_x}] "
                f"tile_value={tile_value} traversable: {is_traversable}"
            )

        return is_traversable

    def is_position_traversable_with_radius(
        self, x: float, y: float, tiles: np.ndarray, radius: float
    ) -> bool:
        """
        Check if a position is traversable considering ninja radius and proper tile definitions.

        Uses the optimized collision detector with full segment-based collision detection
        to handle all tile types correctly.

        Args:
            x: X coordinate (padded coordinate system)
            y: Y coordinate (padded coordinate system)
            tiles: Level tile data (unpadded)
            radius: Ninja collision radius

        Returns:
            True if position is traversable, False if blocked
        """
        return self.collision_detector.is_circle_position_clear(x, y, radius, tiles)

    def convert_sub_grid_to_pixel(
        self, sub_row: int, sub_col: int
    ) -> Tuple[float, float]:
        """
        Convert sub-grid coordinates to pixel coordinates (center of sub-cell).

        Args:
            sub_row: Sub-grid row coordinate
            sub_col: Sub-grid column coordinate

        Returns:
            Tuple of (pixel_x, pixel_y) coordinates
        """
        pixel_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        pixel_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        return pixel_x, pixel_y

    def convert_pixel_to_sub_grid(
        self, pixel_x: float, pixel_y: float
    ) -> Tuple[int, int]:
        """
        Convert pixel coordinates to sub-grid coordinates.

        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels

        Returns:
            Tuple of (sub_row, sub_col) coordinates
        """
        sub_row = int(pixel_y // SUB_CELL_SIZE)
        sub_col = int(pixel_x // SUB_CELL_SIZE)
        return sub_row, sub_col
