"""
Precise tile collision detection system for graph traversability.

This module provides a wrapper around the existing nclone collision system
to enable accurate navigation that matches the actual ninja physics.
It reuses the existing segment-based collision detection from physics.py
and entities.py to ensure perfect consistency.
"""

import math
import numpy as np

from ..constants.physics_constants import (
    NINJA_RADIUS,
    FULL_MAP_WIDTH,
    FULL_MAP_HEIGHT,
    TILE_PIXEL_SIZE,
)
from ..physics import sweep_circle_vs_tiles
from ..utils.tile_segment_factory import TileSegmentFactory


class PreciseTileCollision:
    """
    Precise tile collision detection using the existing nclone collision system.

    This class provides a navigation-friendly interface to the same collision
    detection used by the ninja physics, ensuring perfect consistency between
    simulation and navigation.
    """

    def __init__(self):
        """Initialize precise collision detector."""
        # Cache for simulator segment dictionaries (level_id -> segment_dic)
        self._segment_cache = {}
        self._current_level_id = None

    def is_path_traversable(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        tiles: np.ndarray,
        ninja_radius: float = NINJA_RADIUS,
    ) -> bool:
        """
        Check if a path is traversable using the existing nclone collision system.

        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            tiles: Level tile data as NumPy array
            ninja_radius: Ninja collision radius

        Returns:
            True if path is traversable, False if blocked by tile geometry
        """
        # Calculate movement vector
        dx = tgt_x - src_x
        dy = tgt_y - src_y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 1e-6:  # No movement
            return True

        # Check if source or destination positions are traversable using segment-based collision
        if not self._is_position_traversable_segments(
            src_x, src_y, tiles, ninja_radius
        ):
            return False
        if not self._is_position_traversable_segments(
            tgt_x, tgt_y, tiles, ninja_radius
        ):
            return False

        # Create a mock simulator object with the segment dictionary
        mock_sim = self._create_mock_simulator(tiles)

        # Use the existing sweep_circle_vs_tiles function
        # This uses the same collision detection as the actual ninja physics
        collision_time = sweep_circle_vs_tiles(
            mock_sim, src_x, src_y, dx, dy, ninja_radius
        )

        # If collision_time < 1.0, there's a collision before reaching the target
        return collision_time >= 1.0

    def _is_position_traversable_segments(
        self, x: float, y: float, tiles: np.ndarray, ninja_radius: float
    ) -> bool:
        """
        Check if a position is traversable using hybrid collision detection.

        Uses different strategies based on tile type:
        - Fully solid tiles (1, >33): Simple geometric check
        - Shaped tiles (2-33): Precise segment-based collision detection

        Args:
            x: X coordinate to check
            y: Y coordinate to check
            tiles: Level tile data as NumPy array
            ninja_radius: Required clearance radius around the position

        Returns:
            True if position is traversable, False if blocked by tile geometry
        """
        height, width = tiles.shape

        min_tile_x = int(math.floor((x - ninja_radius) / TILE_PIXEL_SIZE))
        max_tile_x = int(math.ceil((x + ninja_radius) / TILE_PIXEL_SIZE))
        min_tile_y = int(math.floor((y - ninja_radius) / TILE_PIXEL_SIZE))
        max_tile_y = int(math.ceil((y + ninja_radius) / TILE_PIXEL_SIZE))

        # Check each tile in the range
        for check_tile_y in range(min_tile_y, max_tile_y + 1):
            for check_tile_x in range(min_tile_x, max_tile_x + 1):
                # Skip tiles outside the map bounds
                if (
                    check_tile_x < 0
                    or check_tile_x >= width
                    or check_tile_y < 0
                    or check_tile_y >= height
                ):
                    continue

                tile_id = tiles[check_tile_y, check_tile_x]
                if tile_id == 0:
                    continue  # Empty tile, no collision

                # For fully solid tiles, use simple geometric check
                if tile_id == 1 or tile_id > 33:
                    if self._check_solid_tile_collision(
                        x, y, check_tile_x, check_tile_y, ninja_radius
                    ):
                        return False

                # For shaped tiles (2-33), use segment-based collision detection
                elif 2 <= tile_id <= 33:
                    if self._check_shaped_tile_collision(
                        x, y, check_tile_x, check_tile_y, tiles, ninja_radius
                    ):
                        return False

        return True

    def _check_solid_tile_collision(
        self, x: float, y: float, tile_x: int, tile_y: int, ninja_radius: float
    ) -> bool:
        """
        Check collision with a fully solid tile using simple geometry.

        Args:
            x, y: Ninja center position
            tile_x, tile_y: Tile coordinates
            ninja_radius: Ninja collision radius

        Returns:
            True if collision detected, False otherwise
        """
        # Calculate tile boundaries
        tile_left = tile_x * TILE_PIXEL_SIZE
        tile_right = (tile_x + 1) * TILE_PIXEL_SIZE
        tile_top = tile_y * TILE_PIXEL_SIZE
        tile_bottom = (tile_y + 1) * TILE_PIXEL_SIZE

        # Find closest point on tile boundary to ninja center
        closest_x = max(tile_left, min(x, tile_right))
        closest_y = max(tile_top, min(y, tile_bottom))

        # Calculate distance from ninja center to closest point on tile
        dist_x = x - closest_x
        dist_y = y - closest_y
        distance = math.sqrt(dist_x * dist_x + dist_y * dist_y)

        # Collision if ninja radius overlaps with tile
        return distance < ninja_radius

    def _check_shaped_tile_collision(
        self,
        x: float,
        y: float,
        tile_x: int,
        tile_y: int,
        tiles: np.ndarray,
        ninja_radius: float,
    ) -> bool:
        """
        Check collision with a shaped tile using segment-based detection.

        Args:
            x, y: Ninja center position
            tile_x, tile_y: Tile coordinates
            tiles: Full tile array
            ninja_radius: Ninja collision radius

        Returns:
            True if collision detected, False otherwise
        """
        # Create a mock simulator object with the segment dictionary
        mock_sim = self._create_mock_simulator(tiles)

        # Get segments for this specific tile
        segments = mock_sim.segment_dic.get((tile_x, tile_y), [])

        # Check if the ninja circle intersects with any segment in this tile
        for segment in segments:
            # Get the closest point on the segment to the ninja center
            is_back_facing, closest_x, closest_y = segment.get_closest_point(x, y)

            # Calculate distance from ninja center to closest point on segment
            dist_x = x - closest_x
            dist_y = y - closest_y
            distance = math.sqrt(dist_x * dist_x + dist_y * dist_y)

            # Collision if ninja radius overlaps with segment
            # Only consider front-facing segments (back-facing segments are inside geometry)
            if not is_back_facing and distance < ninja_radius:
                return True

        return False

    def _create_mock_simulator(self, tiles: np.ndarray):
        """
        Create a mock simulator object with segment dictionary for collision detection.

        Args:
            tiles: Level tile data as NumPy array

        Returns:
            Mock simulator object with segment_dic attribute
        """
        # Cache segments for performance
        level_id = id(tiles)
        if self._current_level_id != level_id:
            self._build_segment_dictionary(tiles)
            self._current_level_id = level_id

        # Create a simple mock object with the segment dictionary
        class MockSimulator:
            def __init__(self, segment_dic):
                self.segment_dic = segment_dic

        return MockSimulator(self._segment_cache[level_id])

    def _build_segment_dictionary(self, tiles: np.ndarray) -> None:
        """
        Build the segment dictionary using the centralized TileSegmentFactory.

        Args:
            tiles: Level tile data as NumPy array
        """
        level_id = id(tiles)

        # Check if tiles array is empty
        if tiles is None or tiles.size == 0:
            # Create empty segment dictionary for empty level
            segment_dic = {}
            for x in range(FULL_MAP_WIDTH):
                for y in range(FULL_MAP_HEIGHT):
                    segment_dic[(x, y)] = []
            self._segment_cache[level_id] = segment_dic
            return

        # Use centralized factory to create segments
        segment_dic = TileSegmentFactory.create_segment_dictionary(tiles)
        self._segment_cache[level_id] = segment_dic
