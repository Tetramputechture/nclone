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
        
        # Create a mock simulator object with the segment dictionary
        mock_sim = self._create_mock_simulator(tiles)
        
        # Use the existing sweep_circle_vs_tiles function
        # This uses the same collision detection as the actual ninja physics
        collision_time = sweep_circle_vs_tiles(
            mock_sim, src_x, src_y, dx, dy, ninja_radius
        )
        
        # If collision_time < 1.0, there's a collision before reaching the target
        return collision_time >= 1.0
    
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
