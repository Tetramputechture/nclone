#!/usr/bin/env python3
"""
Compare the old conservative collision detection with the new segment-based approach.
"""

import numpy as np
from nclone.graph.precise_collision import PreciseTileCollision
from nclone.constants.physics_constants import NINJA_RADIUS

def test_collision_methods():
    """Test both collision detection methods on a simple case."""
    
    # Create a simple test case: solid tile at (1,1)
    tiles = np.zeros((5, 5), dtype=int)
    tiles[1, 1] = 1  # Solid tile
    tiles[2, 2] = 18  # Complex shaped tile (three-quarter circle)
    
    collision_detector = PreciseTileCollision()
    
    print("🔍 COLLISION DETECTION COMPARISON")
    print("=" * 50)
    print(f"Test tiles shape: {tiles.shape}")
    print(f"Ninja radius: {NINJA_RADIUS}")
    print()
    
    # Test positions around the solid tile
    test_positions = [
        (12, 12),   # Center of empty tile (0,0) - should be traversable
        (36, 36),   # Center of solid tile (1,1) - should NOT be traversable  
        (24, 24),   # Edge between tiles - might be traversable
        (60, 60),   # Center of shaped tile (2,2) - depends on shape
        (48, 48),   # Between solid and shaped tile
    ]
    
    for x, y in test_positions:
        tile_x, tile_y = int(x // 24), int(y // 24)
        tile_value = tiles[tile_y, tile_x] if 0 <= tile_x < 5 and 0 <= tile_y < 5 else 0
        
        # Test segment-based method
        try:
            segment_result = collision_detector._is_position_traversable_segments(x, y, tiles, NINJA_RADIUS)
        except Exception as e:
            segment_result = f"ERROR: {e}"
        
        print(f"Position ({x:2}, {y:2}) in tile ({tile_x}, {tile_y}) [value={tile_value}]:")
        print(f"  Segment-based: {segment_result}")
        print()

if __name__ == "__main__":
    test_collision_methods()