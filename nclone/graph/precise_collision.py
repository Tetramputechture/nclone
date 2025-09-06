"""
Precise tile collision detection system for graph traversability.

This module provides a wrapper around the existing nclone collision system
to enable accurate pathfinding that matches the actual ninja physics.
It reuses the existing segment-based collision detection from physics.py
and entities.py to ensure perfect consistency.
"""

import math
import numpy as np

from ..constants.physics_constants import NINJA_RADIUS, FULL_MAP_WIDTH, FULL_MAP_HEIGHT
from ..physics import sweep_circle_vs_tiles
from ..utils.tile_segment_factory import TileSegmentFactory


class PreciseTileCollision:
    """
    Precise tile collision detection using the existing nclone collision system.
    
    This class provides a pathfinding-friendly interface to the same collision
    detection used by the ninja physics, ensuring perfect consistency between
    simulation and pathfinding.
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
        ninja_radius: float = NINJA_RADIUS
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
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 1e-6:  # No movement
            return True
        
        # CRITICAL FIX: Check if source or destination is inside a solid tile
        # The ninja needs clearance (NINJA_RADIUS) around its center position
        if not self._is_position_traversable(src_x, src_y, tiles, ninja_radius):
            return False
        if not self._is_position_traversable(tgt_x, tgt_y, tiles, ninja_radius):
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
    
    def _is_position_traversable(
        self, 
        x: float, 
        y: float, 
        tiles: np.ndarray, 
        ninja_radius: float
    ) -> bool:
        """
        Check if a position is traversable (ninja can exist there with required clearance).
        
        The ninja is a circle with radius NINJA_RADIUS. For the ninja to be able to exist
        at position (x, y), there must be no solid tiles within ninja_radius of that position.
        
        Args:
            x: X coordinate to check
            y: Y coordinate to check  
            tiles: Level tile data as NumPy array
            ninja_radius: Required clearance radius around the position
            
        Returns:
            True if position is traversable, False if blocked by solid tiles
        """
        from ..constants import TILE_PIXEL_SIZE
        
        # Convert to tile coordinates
        tile_x = x / TILE_PIXEL_SIZE
        tile_y = y / TILE_PIXEL_SIZE
        
        # Calculate the range of tiles that could intersect with the ninja's radius
        # We need to check all tiles that are within ninja_radius of the position
        radius_in_tiles = ninja_radius / TILE_PIXEL_SIZE
        
        min_tile_x = int(math.floor(tile_x - radius_in_tiles))
        max_tile_x = int(math.ceil(tile_x + radius_in_tiles))
        min_tile_y = int(math.floor(tile_y - radius_in_tiles))
        max_tile_y = int(math.ceil(tile_y + radius_in_tiles))
        
        # Check each tile in the range
        height, width = tiles.shape
        for check_tile_y in range(min_tile_y, max_tile_y + 1):
            for check_tile_x in range(min_tile_x, max_tile_x + 1):
                # Skip tiles outside the map bounds
                if (check_tile_x < 0 or check_tile_x >= width or 
                    check_tile_y < 0 or check_tile_y >= height):
                    continue
                
                # If this tile is solid (tile_id = 1), check if it's too close
                if tiles[check_tile_y, check_tile_x] == 1:
                    # Calculate distance from position to closest point on this tile
                    tile_center_x = (check_tile_x + 0.5) * TILE_PIXEL_SIZE
                    tile_center_y = (check_tile_y + 0.5) * TILE_PIXEL_SIZE
                    
                    # For a solid tile, find the closest point on the tile boundary to the ninja center
                    tile_left = check_tile_x * TILE_PIXEL_SIZE
                    tile_right = (check_tile_x + 1) * TILE_PIXEL_SIZE
                    tile_top = check_tile_y * TILE_PIXEL_SIZE
                    tile_bottom = (check_tile_y + 1) * TILE_PIXEL_SIZE
                    
                    # Find closest point on tile boundary to ninja center
                    closest_x = max(tile_left, min(x, tile_right))
                    closest_y = max(tile_top, min(y, tile_bottom))
                    
                    # Calculate distance from ninja center to closest point on tile
                    dist_x = x - closest_x
                    dist_y = y - closest_y
                    distance = math.sqrt(dist_x * dist_x + dist_y * dist_y)
                    
                    # If the ninja's radius overlaps with the solid tile, position is not traversable
                    if distance < ninja_radius:
                        return False
        
        return True
    
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
