"""Grid cell validity classification at 6px resolution.

Classifies each cell of a 176x100 grid (6px per cell over 1056x600 level)
as valid (open space) or invalid (inside solid geometry). Uses the same
tile geometry logic as graph_builder._check_subnode_validity_simple() but
evaluated at 6px grid centers.

The 6px resolution is approximately half the ninja radius (10px), providing
fine-grained gradients for PBRS potential fields.
"""

import numpy as np
from typing import Dict, Tuple

from ..constants.physics_constants import FULL_MAP_WIDTH_PX, FULL_MAP_HEIGHT_PX

# Grid resolution and dimensions
GRID_RESOLUTION = 6  # pixels per cell
GRID_WIDTH = FULL_MAP_WIDTH_PX // GRID_RESOLUTION  # 1056 // 6 = 176
GRID_HEIGHT = FULL_MAP_HEIGHT_PX // GRID_RESOLUTION  # 600 // 6 = 100

# Tile size in pixels
TILE_SIZE = 24


def _check_pixel_validity(tile_type: int, pixel_x: int, pixel_y: int) -> bool:
    """Check if a pixel position is in non-solid area of its tile.

    This is a direct adaptation of graph_builder._check_subnode_validity_simple()
    extended to work at arbitrary pixel positions within a tile.

    Args:
        tile_type: Tile type ID (0-37)
        pixel_x: X position within tile (0-24)
        pixel_y: Y position within tile (0-24)

    Returns:
        True if position is in non-solid (traversable) area
    """
    if tile_type == 0:
        return True
    if tile_type == 1:
        return False

    # Half tiles (2-5)
    if tile_type == 2:
        return pixel_y >= 12
    if tile_type == 3:
        return pixel_x < 12
    if tile_type == 4:
        return pixel_y < 12
    if tile_type == 5:
        return pixel_x >= 12

    # 45-degree slopes (6-9)
    if tile_type == 6:
        return pixel_y > (24 - pixel_x)
    if tile_type == 7:
        return pixel_x < pixel_y
    if tile_type == 8:
        return pixel_y < (24 - pixel_x) or pixel_y == pixel_x
    if tile_type == 9:
        return pixel_x > pixel_y

    # Quarter circles (10-13): solid inside arc, traversable outside
    if tile_type == 10:
        return pixel_x * pixel_x + pixel_y * pixel_y >= 576  # 24^2
    if tile_type == 11:
        dx = pixel_x - 24
        return dx * dx + pixel_y * pixel_y >= 576
    if tile_type == 12:
        dx = pixel_x - 24
        dy = pixel_y - 24
        return dx * dx + dy * dy >= 576
    if tile_type == 13:
        dy = pixel_y - 24
        return pixel_x * pixel_x + dy * dy >= 576

    # Quarter pipes (14-17): traversable inside arc, solid outside
    if tile_type == 14:
        dx = pixel_x - 24
        dy = pixel_y - 24
        return dx * dx + dy * dy < 576
    if tile_type == 15:
        dy = pixel_y - 24
        return pixel_x * pixel_x + dy * dy < 576
    if tile_type == 16:
        return pixel_x * pixel_x + pixel_y * pixel_y < 576
    if tile_type == 17:
        dx = pixel_x - 24
        return dx * dx + pixel_y * pixel_y < 576

    # Short mild slopes (18-21)
    if tile_type == 18:
        return pixel_y >= (12 - pixel_x * 12 / 24)
    if tile_type == 19:
        return pixel_y >= (pixel_x * 12 / 24)
    if tile_type == 20:
        return pixel_y <= (24 - pixel_x * 12 / 24)
    if tile_type == 21:
        return pixel_y <= (12 + pixel_x * 12 / 24)

    # Raised mild slopes (22-25)
    if tile_type == 22:
        return pixel_y >= (24 - pixel_x * 12 / 24)
    if tile_type == 23:
        return pixel_y >= (12 + pixel_x * 12 / 24)
    if tile_type == 24:
        return pixel_y < (12 - pixel_x * 12 / 24)
    if tile_type == 25:
        return pixel_y < (pixel_x * 12 / 24)

    # Short steep slopes (26-29)
    if tile_type == 26:
        return pixel_y >= (24 - pixel_x * 24 / 12) if pixel_x <= 12 else True
    if tile_type == 27:
        return pixel_y >= (pixel_x * 24 / 12)
    if tile_type == 28:
        return pixel_y <= (24 - (pixel_x - 12) * 24 / 12) if pixel_x >= 12 else True
    if tile_type == 29:
        return pixel_x > 12 or pixel_y < (pixel_x * 24 / 12)

    # Raised steep slopes (30-33)
    if tile_type == 30:
        if pixel_x >= 12:
            return pixel_y >= (24 - (pixel_x - 12) * 24 / 12)
        return pixel_y >= 24
    if tile_type == 31:
        if pixel_x <= 12:
            return pixel_y >= (pixel_x * 24 / 12)
        return pixel_y >= 24
    if tile_type == 32:
        return pixel_x < 12 and pixel_y < (24 - 2 * pixel_x)
    if tile_type == 33:
        return pixel_x > 12 and pixel_y < (2 * pixel_x - 24)

    # Glitched tiles (34-37): treat as non-solid
    if tile_type >= 34:
        return True

    return True


class GridClassifier:
    """Classifies level cells as valid (traversable) or invalid (solid).

    Produces a boolean numpy array of shape (GRID_WIDTH, GRID_HEIGHT) where
    True = open space the ninja can occupy, False = inside solid geometry.
    """

    @staticmethod
    def classify(tile_dic: Dict[Tuple[int, int], int]) -> np.ndarray:
        """Classify all cells in the level grid.

        For each cell, computes the pixel center, determines which tile it
        falls in, and checks if that pixel position is in non-solid area.

        Args:
            tile_dic: Dict mapping (tile_col, tile_row) -> tile_type,
                as stored in sim.tile_dic. Tile coordinates are 0-indexed
                (0-43 for width, 0-24 for height).

        Returns:
            Boolean numpy array of shape (GRID_WIDTH, GRID_HEIGHT) where
            valid_mask[ix, iy] = True means cell (ix, iy) is traversable.
        """
        valid_mask = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=bool)

        for ix in range(GRID_WIDTH):
            for iy in range(GRID_HEIGHT):
                # Pixel center of this cell
                px = ix * GRID_RESOLUTION + GRID_RESOLUTION // 2  # ix*6 + 3
                py = iy * GRID_RESOLUTION + GRID_RESOLUTION // 2  # iy*6 + 3

                # Which tile does this pixel fall in?
                tile_col = px // TILE_SIZE
                tile_row = py // TILE_SIZE

                # Look up tile type from dict (default to full solid if missing)
                tile_type = tile_dic.get((tile_col, tile_row), 1)

                # Position relative to tile origin
                rel_x = px - tile_col * TILE_SIZE
                rel_y = py - tile_row * TILE_SIZE

                valid_mask[ix, iy] = _check_pixel_validity(tile_type, rel_x, rel_y)

        return valid_mask

    @staticmethod
    def pixel_to_cell(px: float, py: float) -> Tuple[int, int]:
        """Convert pixel coordinates to grid cell indices.

        Args:
            px: X position in pixels
            py: Y position in pixels

        Returns:
            (ix, iy) grid cell indices, clamped to valid range
        """
        ix = int(px / GRID_RESOLUTION)
        iy = int(py / GRID_RESOLUTION)
        ix = max(0, min(GRID_WIDTH - 1, ix))
        iy = max(0, min(GRID_HEIGHT - 1, iy))
        return ix, iy

    @staticmethod
    def cell_to_pixel(ix: int, iy: int) -> Tuple[float, float]:
        """Convert grid cell indices to pixel center coordinates.

        Args:
            ix: X cell index
            iy: Y cell index

        Returns:
            (px, py) pixel center coordinates
        """
        px = ix * GRID_RESOLUTION + GRID_RESOLUTION / 2.0
        py = iy * GRID_RESOLUTION + GRID_RESOLUTION / 2.0
        return px, py
