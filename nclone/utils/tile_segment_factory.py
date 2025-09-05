"""
Centralized tile segment factory for consistent segment creation across MapLoader and PreciseCollision.

This module consolidates the tile-to-segment conversion logic to eliminate duplication
and ensure consistency between the main simulation and collision detection systems.

## Overview

The TileSegmentFactory centralizes the complex logic for converting tile data into
collision segments, supporting multiple input formats and ensuring exact compatibility
with the original MapLoader implementation.

## Key Features

- **Format Flexibility**: Supports dict, list, and numpy array tile formats
- **Exact Compatibility**: Replicates MapLoader logic precisely for consistency
- **Performance Optimized**: Uses efficient defaultdict for orthogonal segment processing
- **Comprehensive Coverage**: Handles all tile types (solid, diagonal, circular)
- **Constants Integration**: Uses physics constants for map dimensions (44x25 tiles)

## Usage Examples

### Basic Usage
```python
from nclone.utils.tile_segment_factory import TileSegmentFactory

# Dictionary format
tiles = {(5, 5): 1, (6, 6): 6}  # Solid and diagonal tiles
segments = TileSegmentFactory.create_segment_dictionary(tiles)

# List format
tiles = [[1, 0, 6], [0, 1, 0]]
segments = TileSegmentFactory.create_segment_dictionary(tiles)
```

### Integration with Simulator Objects
```python
# For MapLoader-style objects with orthogonal segment dictionaries
TileSegmentFactory.create_segments_for_simulator(simulator, tiles)
```

## Architecture

The factory uses a two-phase approach:
1. **Orthogonal Processing**: Builds horizontal/vertical segment dictionaries
2. **Segment Creation**: Converts processed data into final segment objects

This matches the original MapLoader architecture while providing a clean,
reusable interface for all collision systems.
"""

import math
from typing import Dict, Any, List, Tuple, Union
from collections import defaultdict

from ..entities import GridSegmentLinear, GridSegmentCircular
from ..tile_definitions import (
    TILE_GRID_EDGE_MAP, TILE_SEGMENT_ORTHO_MAP, TILE_SEGMENT_DIAG_MAP, TILE_SEGMENT_CIRCULAR_MAP
)
from ..constants.physics_constants import TILE_PIXEL_SIZE, FULL_MAP_WIDTH, FULL_MAP_HEIGHT


class TileSegmentFactory:
    """
    Factory class for creating tile segments from tile definitions.
    
    This class centralizes the logic for converting tile data into collision segments,
    ensuring consistency between MapLoader and PreciseCollision systems.
    """
    
    @staticmethod
    def create_segment_dictionary(tiles: Union[Dict, List, Any]) -> Dict[Tuple[int, int], List]:
        """
        Create a complete segment dictionary from tile data.
        
        All N++ levels are exactly 44x25 tiles as defined in physics constants.
        
        Args:
            tiles: Tile data in various formats (dict, list, numpy array)
            
        Returns:
            Dictionary mapping (x, y) coordinates to lists of segments
        """
        # Initialize segment dictionary with empty lists for all cells
        # All N++ maps are exactly FULL_MAP_WIDTH x FULL_MAP_HEIGHT tiles
        segment_dic = {}
        for x in range(FULL_MAP_WIDTH):
            for y in range(FULL_MAP_HEIGHT):
                segment_dic[(x, y)] = []
        
        # Initialize temporary dictionaries for orthogonal segments
        hor_segment_dic = defaultdict(int)
        ver_segment_dic = defaultdict(int)
        
        # Process tiles to build segments
        TileSegmentFactory._process_tiles_for_segments(
            tiles, segment_dic, hor_segment_dic, ver_segment_dic
        )
        
        # Convert orthogonal segment dictionaries to actual segments
        TileSegmentFactory._process_orthogonal_segments(
            hor_segment_dic, ver_segment_dic, segment_dic
        )
        
        return segment_dic
    
    @staticmethod
    def _process_tiles_for_segments(tiles, segment_dic, hor_segment_dic, ver_segment_dic):
        """
        Process tile data and create segments using standardized logic.
        
        Args:
            tiles: Tile data in various formats
            segment_dic: Dictionary to populate with segments
            hor_segment_dic: Horizontal segment accumulator
            ver_segment_dic: Vertical segment accumulator
        """
        if isinstance(tiles, dict):
            # Dictionary format: {(x, y): tile_id}
            for (xcoord, ycoord), tile_id in tiles.items():
                if tile_id != 0:
                    TileSegmentFactory._process_single_tile(
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
                        TileSegmentFactory._process_single_tile(
                            tile_id, xcoord, ycoord, segment_dic,
                            hor_segment_dic, ver_segment_dic
                        )
        elif isinstance(tiles, (list, tuple)):
            # List/tuple format
            for ycoord, row in enumerate(tiles):
                if hasattr(row, '__getitem__'):
                    for xcoord, tile_id in enumerate(row):
                        if tile_id != 0:
                            TileSegmentFactory._process_single_tile(
                                tile_id, xcoord, ycoord, segment_dic,
                                hor_segment_dic, ver_segment_dic
                            )
    
    @staticmethod
    def _process_single_tile(tile_id, xcoord, ycoord, segment_dic, hor_segment_dic, ver_segment_dic):
        """
        Process a single tile and add its segments using the exact MapLoader logic.
        
        Args:
            tile_id: Tile type ID
            xcoord: Tile x coordinate
            ycoord: Tile y coordinate
            segment_dic: Main segment dictionary
            hor_segment_dic: Horizontal segment accumulator
            ver_segment_dic: Vertical segment accumulator
        """
        coord = (xcoord, ycoord)
        
        # Process orthogonal segments (grid edges and linear segments)
        if tile_id in TILE_GRID_EDGE_MAP and tile_id in TILE_SEGMENT_ORTHO_MAP:
            grid_edge_list = TILE_GRID_EDGE_MAP[tile_id]
            segment_ortho_list = TILE_SEGMENT_ORTHO_MAP[tile_id]
            
            # Process horizontal segments - matches MapLoader exactly
            for y_loop_idx in range(3):
                for x_loop_idx in range(2):
                    hor_key = (2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx)
                    hor_segment_dic[hor_key] += segment_ortho_list[2 * y_loop_idx + x_loop_idx]
            
            # Process vertical segments - matches MapLoader exactly
            for x_loop_idx in range(3):
                for y_loop_idx in range(2):
                    ver_key = (2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx)
                    ver_segment_dic[ver_key] += segment_ortho_list[2 * x_loop_idx + y_loop_idx + 6]
        
        # Process non-orthogonal segments (diagonal and circular)
        xtl = xcoord * TILE_PIXEL_SIZE
        ytl = ycoord * TILE_PIXEL_SIZE
        
        # Add diagonal segments
        if tile_id in TILE_SEGMENT_DIAG_MAP:
            ((x1, y1), (x2, y2)) = TILE_SEGMENT_DIAG_MAP[tile_id]
            segment_dic[coord].append(
                GridSegmentLinear((xtl + x1, ytl + y1), (xtl + x2, ytl + y2))
            )
        
        # Add circular segments
        if tile_id in TILE_SEGMENT_CIRCULAR_MAP:
            ((x_center, y_center), quadrant, convex) = TILE_SEGMENT_CIRCULAR_MAP[tile_id]
            segment_dic[coord].append(
                GridSegmentCircular((xtl + x_center, ytl + y_center), quadrant, convex)
            )
    
    @staticmethod
    def _process_orthogonal_segments(hor_segment_dic, ver_segment_dic, segment_dic):
        """
        Convert orthogonal segment dictionaries to actual GridSegmentLinear objects.
        
        This matches the MapLoader logic exactly where segments with opposite orientations
        cancel each other out (state == 0 means no segment).
        
        Args:
            hor_segment_dic: Horizontal segment accumulator
            ver_segment_dic: Vertical segment accumulator
            segment_dic: Main segment dictionary to populate
        """
        # Process horizontal segments
        for coord, state in hor_segment_dic.items():
            if state != 0:  # Non-zero state means segment exists
                xcoord, ycoord = coord
                # Calculate which cell this segment belongs to
                cell = (math.floor(xcoord / 2), math.floor((ycoord - 0.1 * state) / 2))
                
                # Create segment endpoints
                point1 = (12 * xcoord, 12 * ycoord)
                point2 = (12 * xcoord + 12, 12 * ycoord)
                
                # Reverse direction for negative state
                if state == -1:
                    point1, point2 = point2, point1
                
                segment_dic[cell].append(GridSegmentLinear(point1, point2))
        
        # Process vertical segments
        for coord, state in ver_segment_dic.items():
            if state != 0:  # Non-zero state means segment exists
                xcoord, ycoord = coord
                # Calculate which cell this segment belongs to
                cell = (math.floor((xcoord - 0.1 * state) / 2), math.floor(ycoord / 2))
                
                # Create segment endpoints
                point1 = (12 * xcoord, 12 * ycoord + 12)
                point2 = (12 * xcoord, 12 * ycoord)
                
                # Reverse direction for negative state
                if state == -1:
                    point1, point2 = point2, point1
                
                segment_dic[cell].append(GridSegmentLinear(point1, point2))
    
    @staticmethod
    def create_segments_for_simulator(simulator, tiles):
        """
        Create segments directly in a simulator object (for MapLoader compatibility).
        
        Args:
            simulator: Simulator object with segment_dic, hor_segment_dic, ver_segment_dic
            tiles: Tile data dictionary {(x, y): tile_id}
        """
        # Process each tile using the centralized logic
        for (xcoord, ycoord), tile_id in tiles.items():
            if tile_id != 0:
                TileSegmentFactory._process_single_tile(
                    tile_id, xcoord, ycoord, simulator.segment_dic,
                    simulator.hor_segment_dic, simulator.ver_segment_dic
                )
        
        # Convert orthogonal segments to actual segments
        TileSegmentFactory._process_orthogonal_segments(
            simulator.hor_segment_dic, simulator.ver_segment_dic, simulator.segment_dic
        )