#!/usr/bin/env python3
"""
Debug what's blocking paths between empty tile positions.
"""

import os
import sys
import math

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.precise_collision import PreciseTileCollision


def debug_path_blocking():
    """Debug what's blocking paths between positions."""
    print("=" * 80)
    print("DEBUGGING PATH BLOCKING BETWEEN EMPTY TILE POSITIONS")
    print("=" * 80)
    
    # Create environment
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42
    )
    
    # Reset to load the map
    env.reset()
    
    # Get level data
    level_data = env.level_data
    tiles = level_data.tiles
    
    print(f"Map: {level_data.width}x{level_data.height} tiles")
    
    # Create collision detector
    collision_detector = PreciseTileCollision()
    
    # Test the failing path: (156, 252) -> (180, 252)
    src_x, src_y = 156, 252
    tgt_x, tgt_y = 180, 252
    
    print(f"\nDebugging path: ({src_x}, {src_y}) -> ({tgt_x}, {tgt_y})")
    
    # Check source and target positions
    print(f"\nChecking source and target positions:")
    
    src_traversable = collision_detector._is_position_traversable(src_x, src_y, tiles, 10)
    tgt_traversable = collision_detector._is_position_traversable(tgt_x, tgt_y, tiles, 10)
    
    print(f"Source ({src_x}, {src_y}): {'✅ Traversable' if src_traversable else '❌ Not traversable'}")
    print(f"Target ({tgt_x}, {tgt_y}): {'✅ Traversable' if tgt_traversable else '❌ Not traversable'}")
    
    # Check tiles along the path
    print(f"\nAnalyzing tiles along the path:")
    
    # Sample points along the path
    num_samples = 10
    for i in range(num_samples + 1):
        t = i / num_samples
        sample_x = src_x + t * (tgt_x - src_x)
        sample_y = src_y + t * (tgt_y - src_y)
        
        # Check tile at this position
        tile_x = int(sample_x // 24)
        tile_y = int(sample_y // 24)
        
        if 0 <= tile_x < level_data.width and 0 <= tile_y < level_data.height:
            tile_value = level_data.get_tile(tile_y, tile_x)
            tile_type = "empty" if tile_value == 0 else "solid" if tile_value == 1 else f"other({tile_value})"
            
            # Check if position is traversable
            pos_traversable = collision_detector._is_position_traversable(sample_x, sample_y, tiles, 10)
            status = "✅" if pos_traversable else "❌"
            
            print(f"  Sample {i:2d}: ({sample_x:5.1f}, {sample_y:5.1f}) -> Tile ({tile_x:2d}, {tile_y:2d}) = {tile_type:8s} {status}")
        else:
            print(f"  Sample {i:2d}: ({sample_x:5.1f}, {sample_y:5.1f}) -> OUT OF BOUNDS")
    
    # Check the actual collision detection
    print(f"\nTesting collision detection:")
    
    path_traversable = collision_detector.is_path_traversable(src_x, src_y, tgt_x, tgt_y, tiles, 10)
    print(f"Path traversable: {'✅ Yes' if path_traversable else '❌ No'}")
    
    # Let's manually check what sweep_circle_vs_tiles returns
    print(f"\nManual collision detection analysis:")
    
    # Create mock simulator
    mock_sim = collision_detector._create_mock_simulator(tiles)
    
    # Calculate movement vector
    dx = tgt_x - src_x
    dy = tgt_y - src_y
    distance = math.sqrt(dx*dx + dy*dy)
    
    print(f"Movement vector: ({dx}, {dy}), distance: {distance:.1f}")
    
    # Import the collision function
    from nclone.physics import sweep_circle_vs_tiles
    
    # Test collision
    collision_time = sweep_circle_vs_tiles(mock_sim, src_x, src_y, dx, dy, 10)
    
    print(f"Collision time: {collision_time:.6f}")
    print(f"Expected: >= 1.0 for no collision")
    print(f"Result: {'✅ No collision' if collision_time >= 1.0 else '❌ Collision detected'}")
    
    if collision_time < 1.0:
        collision_x = src_x + collision_time * dx
        collision_y = src_y + collision_time * dy
        print(f"Collision point: ({collision_x:.1f}, {collision_y:.1f})")
        
        # Check what tile the collision is in
        collision_tile_x = int(collision_x // 24)
        collision_tile_y = int(collision_y // 24)
        
        if 0 <= collision_tile_x < level_data.width and 0 <= collision_tile_y < level_data.height:
            collision_tile_value = level_data.get_tile(collision_tile_y, collision_tile_x)
            collision_tile_type = "empty" if collision_tile_value == 0 else "solid" if collision_tile_value == 1 else f"other({collision_tile_value})"
            print(f"Collision tile: ({collision_tile_x}, {collision_tile_y}) = {collision_tile_type}")
    
    # Test with smaller radius
    print(f"\n" + "=" * 60)
    print("TESTING WITH SMALLER NINJA RADIUS")
    print("=" * 60)
    
    for test_radius in [8, 6, 4, 2]:
        print(f"\nTesting with radius {test_radius}:")
        
        # Check positions
        src_ok = collision_detector._is_position_traversable(src_x, src_y, tiles, test_radius)
        tgt_ok = collision_detector._is_position_traversable(tgt_x, tgt_y, tiles, test_radius)
        
        # Check path
        path_ok = collision_detector.is_path_traversable(src_x, src_y, tgt_x, tgt_y, tiles, test_radius)
        
        # Manual collision check
        collision_time = sweep_circle_vs_tiles(mock_sim, src_x, src_y, dx, dy, test_radius)
        
        print(f"  Source: {'✅' if src_ok else '❌'}, Target: {'✅' if tgt_ok else '❌'}, Path: {'✅' if path_ok else '❌'}, Collision time: {collision_time:.6f}")
    
    # Test a working path for comparison
    print(f"\n" + "=" * 60)
    print("COMPARING WITH WORKING PATH: (228, 276) -> (204, 300)")
    print("=" * 60)
    
    work_src_x, work_src_y = 228, 276
    work_tgt_x, work_tgt_y = 204, 300
    
    work_dx = work_tgt_x - work_src_x
    work_dy = work_tgt_y - work_src_y
    work_distance = math.sqrt(work_dx*work_dx + work_dy*work_dy)
    
    work_collision_time = sweep_circle_vs_tiles(mock_sim, work_src_x, work_src_y, work_dx, work_dy, 10)
    work_path_ok = collision_detector.is_path_traversable(work_src_x, work_src_y, work_tgt_x, work_tgt_y, tiles, 10)
    
    print(f"Working path:")
    print(f"  Movement: ({work_dx}, {work_dy}), distance: {work_distance:.1f}")
    print(f"  Collision time: {work_collision_time:.6f}")
    print(f"  Path traversable: {'✅' if work_path_ok else '❌'}")
    
    # Sample tiles along working path
    print(f"\nTiles along working path:")
    for i in range(6):
        t = i / 5
        sample_x = work_src_x + t * work_dx
        sample_y = work_src_y + t * work_dy
        
        tile_x = int(sample_x // 24)
        tile_y = int(sample_y // 24)
        
        if 0 <= tile_x < level_data.width and 0 <= tile_y < level_data.height:
            tile_value = level_data.get_tile(tile_y, tile_x)
            tile_type = "empty" if tile_value == 0 else "solid" if tile_value == 1 else f"other({tile_value})"
            print(f"  Sample {i}: ({sample_x:5.1f}, {sample_y:5.1f}) -> Tile ({tile_x:2d}, {tile_y:2d}) = {tile_type}")


if __name__ == '__main__':
    debug_path_blocking()