#!/usr/bin/env python3
"""
Debug what segments are created for different tile types.
"""

import numpy as np
from nclone.utils.tile_segment_factory import TileSegmentFactory
from nclone.tile_definitions import TILE_GRID_EDGE_MAP, TILE_SEGMENT_ORTHO_MAP

def debug_tile_segments():
    """Debug what segments are created for different tile types."""
    
    # Create a simple test case with different tile types
    tiles = np.zeros((3, 3), dtype=int)
    tiles[1, 1] = 1   # Solid tile
    tiles[0, 0] = 18  # Complex shaped tile (three-quarter circle)
    
    print("🔍 TILE SEGMENT ANALYSIS")
    print("=" * 50)
    
    # Create segment dictionary
    segment_dic = TileSegmentFactory.create_segment_dictionary(tiles)
    
    # Check what segments exist for each tile
    for y in range(3):
        for x in range(3):
            tile_value = tiles[y, x]
            segments = segment_dic.get((x, y), [])
            
            print(f"Tile ({x}, {y}) [value={tile_value}]: {len(segments)} segments")
            
            if tile_value in TILE_GRID_EDGE_MAP:
                grid_edges = TILE_GRID_EDGE_MAP[tile_value]
                ortho_map = TILE_SEGMENT_ORTHO_MAP[tile_value]
                print(f"  Grid edges: {grid_edges}")
                print(f"  Ortho map:  {ortho_map}")
            
            for i, segment in enumerate(segments):
                print(f"  Segment {i}: {type(segment).__name__}")
                if hasattr(segment, 'x1'):
                    print(f"    Linear: ({segment.x1}, {segment.y1}) -> ({segment.x2}, {segment.y2})")
                elif hasattr(segment, 'center_x'):
                    print(f"    Circular: center=({segment.center_x}, {segment.center_y})")
            print()

if __name__ == "__main__":
    debug_tile_segments()