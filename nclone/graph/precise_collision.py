"""
Precise tile collision detection system for graph traversability.

This module provides a wrapper around the existing nclone collision system
to enable accurate pathfinding that matches the actual ninja physics.
It reuses the existing segment-based collision detection from physics.py
and entities.py to ensure perfect consistency.
"""

import math
from typing import Dict, Any, List, Optional, Tuple, Union

from ..constants.physics_constants import TILE_PIXEL_SIZE, NINJA_RADIUS
from ..physics import sweep_circle_vs_tiles, gather_segments_from_region
from ..entities import GridSegmentLinear, GridSegmentCircular
from ..tile_definitions import (
    TILE_GRID_EDGE_MAP, TILE_SEGMENT_ORTHO_MAP, 
    TILE_SEGMENT_DIAG_MAP, TILE_SEGMENT_CIRCULAR_MAP
)


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
        level_data: Dict[str, Any],
        ninja_radius: float = NINJA_RADIUS
    ) -> bool:
        """
        Check if a path is traversable using the existing nclone collision system.
        
        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            level_data: Level tile data and structure
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
        mock_sim = self._create_mock_simulator(level_data)
        
        # Use the existing sweep_circle_vs_tiles function
        # This uses the same collision detection as the actual ninja physics
        collision_time = sweep_circle_vs_tiles(
            mock_sim, src_x, src_y, dx, dy, ninja_radius
        )
        
        # If collision_time < 1.0, there's a collision before reaching the target
        return collision_time >= 1.0
    
    def _create_mock_simulator(self, level_data: Dict[str, Any]):
        """
        Create a mock simulator object with segment dictionary for collision detection.
        
        Args:
            level_data: Level tile data and structure
            
        Returns:
            Mock simulator object with segment_dic attribute
        """
        # Cache segments for performance
        level_id = level_data.get('level_id', id(level_data))
        if self._current_level_id != level_id:
            self._build_segment_dictionary(level_data)
            self._current_level_id = level_id
        
        # Create a simple mock object with the segment dictionary
        class MockSimulator:
            def __init__(self, segment_dic):
                self.segment_dic = segment_dic
        
        return MockSimulator(self._segment_cache[level_id])
    
    def _build_segment_dictionary(self, level_data: Dict[str, Any]) -> None:
        """
        Build the segment dictionary using the same logic as MapLoader.
        
        Args:
            level_data: Level tile data
        """
        level_id = level_data.get('level_id', id(level_data))
        segment_dic = {}
        
        # Initialize segment dictionary with empty lists for all cells
        for x in range(44):  # Standard nclone map width
            for y in range(25):  # Standard nclone map height
                segment_dic[(x, y)] = []
        
        # Get tile data
        tiles = level_data.get('tiles', {})
        if not tiles:
            self._segment_cache[level_id] = segment_dic
            return
        
        # Handle different tile data formats and build segments
        # This replicates the logic from MapLoader.load_map_tiles()
        self._process_tiles_for_segments(tiles, segment_dic)
        
        self._segment_cache[level_id] = segment_dic
    
    def _process_tiles_for_segments(self, tiles, segment_dic):
        """
        Process tile data and create segments using the same logic as MapLoader.
        
        Args:
            tiles: Tile data in various formats
            segment_dic: Dictionary to populate with segments
        """
        # Initialize temporary dictionaries for orthogonal segments
        hor_segment_dic = {}
        ver_segment_dic = {}
        
        # Initialize all grid positions
        for x in range(88):  # 44 * 2 for half-tile precision
            for y in range(50):  # 25 * 2 for half-tile precision
                hor_segment_dic[(x, y)] = 0
                ver_segment_dic[(x, y)] = 0
        
        # Process each tile to build segments (replicating MapLoader logic)
        if isinstance(tiles, dict):
            # Dictionary format: {(x, y): tile_id}
            for (xcoord, ycoord), tile_id in tiles.items():
                if tile_id != 0:
                    self._process_single_tile(
                        tile_id, xcoord, ycoord, segment_dic, 
                        hor_segment_dic, ver_segment_dic
                    )
        elif hasattr(tiles, 'shape') and len(tiles.shape) == 2:
            # NumPy array format
            height, width = tiles.shape
            for ycoord in range(height):
                for xcoord in range(width):
                    tile_id = tiles[ycoord, xcoord]
                    if tile_id != 0:
                        self._process_single_tile(
                            tile_id, xcoord, ycoord, segment_dic,
                            hor_segment_dic, ver_segment_dic
                        )
        elif isinstance(tiles, (list, tuple)):
            # List/tuple format
            for ycoord, row in enumerate(tiles):
                if hasattr(row, '__getitem__'):
                    for xcoord, tile_id in enumerate(row):
                        if tile_id != 0:
                            self._process_single_tile(
                                tile_id, xcoord, ycoord, segment_dic,
                                hor_segment_dic, ver_segment_dic
                            )
        
        # Process orthogonal segments (replicating MapLoader logic)
        self._process_orthogonal_segments(hor_segment_dic, ver_segment_dic, segment_dic)
    
    def _process_single_tile(self, tile_id, xcoord, ycoord, segment_dic, hor_segment_dic, ver_segment_dic):
        """
        Process a single tile and add its segments (replicating MapLoader logic).
        
        Args:
            tile_id: Tile type ID
            xcoord: Tile x coordinate
            ycoord: Tile y coordinate
            segment_dic: Main segment dictionary
            hor_segment_dic: Horizontal segment dictionary
            ver_segment_dic: Vertical segment dictionary
        """
        coord = (xcoord, ycoord)
        
        # Assign every grid edge and orthogonal linear segment to the dictionaries
        if tile_id in TILE_GRID_EDGE_MAP and tile_id in TILE_SEGMENT_ORTHO_MAP:
            grid_edge_list = TILE_GRID_EDGE_MAP[tile_id]
            segment_ortho_list = TILE_SEGMENT_ORTHO_MAP[tile_id]
            
            # Process horizontal segments (replicating MapLoader logic exactly)
            for y_loop_idx in range(3):
                for x_loop_idx in range(2):
                    hor_key = (2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx)
                    # Note: We don't need grid edges for collision, just segments
                    hor_segment_dic[hor_key] += segment_ortho_list[2 * y_loop_idx + x_loop_idx]
            
            # Process vertical segments (note different loop structure)
            for x_loop_idx in range(3):
                for y_loop_idx in range(2):
                    ver_key = (2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx)
                    # Note: We don't need grid edges for collision, just segments
                    ver_segment_dic[ver_key] += segment_ortho_list[2 * x_loop_idx + y_loop_idx + 6]
        
        # Initiate non-orthogonal linear and circular segments
        xtl = xcoord * TILE_PIXEL_SIZE
        ytl = ycoord * TILE_PIXEL_SIZE
        
        if tile_id in TILE_SEGMENT_DIAG_MAP:
            ((x1, y1), (x2, y2)) = TILE_SEGMENT_DIAG_MAP[tile_id]
            segment_dic[coord].append(
                GridSegmentLinear((xtl + x1, ytl + y1), (xtl + x2, ytl + y2))
            )
        
        if tile_id in TILE_SEGMENT_CIRCULAR_MAP:
            ((x_center, y_center), quadrant, convex) = TILE_SEGMENT_CIRCULAR_MAP[tile_id]
            segment_dic[coord].append(
                GridSegmentCircular((xtl + x_center, ytl + y_center), quadrant, convex)
            )
    
    def _process_orthogonal_segments(self, hor_segment_dic, ver_segment_dic, segment_dic):
        """
        Process orthogonal segments and add them to segment dictionary (replicating MapLoader logic).
        
        Args:
            hor_segment_dic: Horizontal segment dictionary
            ver_segment_dic: Vertical segment dictionary
            segment_dic: Main segment dictionary
        """
        # Process horizontal segments
        for coord, state in hor_segment_dic.items():
            if state:
                xcoord, ycoord = coord
                cell = (math.floor(xcoord / 2), math.floor((ycoord - 0.1 * state) / 2))
                point1 = (12 * xcoord, 12 * ycoord)
                point2 = (12 * xcoord + 12, 12 * ycoord)
                if state == -1:
                    point1, point2 = point2, point1
                segment_dic[cell].append(GridSegmentLinear(point1, point2))
        
        # Process vertical segments
        for coord, state in ver_segment_dic.items():
            if state:
                xcoord, ycoord = coord
                cell = (math.floor((xcoord - 0.1 * state) / 2), math.floor(ycoord / 2))
                point1 = (12 * xcoord, 12 * ycoord + 12)
                point2 = (12 * xcoord, 12 * ycoord)
                if state == -1:
                    point1, point2 = point2, point1
                segment_dic[cell].append(GridSegmentLinear(point1, point2))