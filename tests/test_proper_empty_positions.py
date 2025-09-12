#!/usr/bin/env python3
"""
Test navigation between positions that are actually in the center of empty tiles.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.precise_collision import PreciseTileCollision


def test_proper_empty_positions():
    """Test navigation between positions in the center of empty tiles."""
    print("=" * 80)
    print("TESTING PATHFINDING BETWEEN PROPER EMPTY TILE CENTERS")
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
    
    # Find actual empty tiles and their centers
    print(f"\nFinding empty tiles and their centers:")
    
    empty_tile_centers = []
    
    for y in range(level_data.height):
        for x in range(level_data.width):
            if level_data.get_tile(y, x) == 0:  # Empty tile
                # Calculate center of tile
                center_x = x * 24 + 12
                center_y = y * 24 + 12
                empty_tile_centers.append((center_x, center_y, x, y))
    
    print(f"Found {len(empty_tile_centers)} empty tiles")
    
    # Show first 10 empty tile centers
    print(f"\nFirst 10 empty tile centers:")
    for i, (center_x, center_y, tile_x, tile_y) in enumerate(empty_tile_centers[:10], 1):
        print(f"  {i:2d}. Tile ({tile_x:2d}, {tile_y:2d}) -> Center ({center_x:3d}, {center_y:3d})")
    
    # Create collision detector
    collision_detector = PreciseTileCollision()
    
    # Test traversability of empty tile centers
    print(f"\nTesting traversability of empty tile centers:")
    
    traversable_centers = []
    
    for center_x, center_y, tile_x, tile_y in empty_tile_centers:
        is_traversable = collision_detector._is_position_traversable(center_x, center_y, tiles, 10)
        
        if is_traversable:
            traversable_centers.append((center_x, center_y, tile_x, tile_y))
        
        status = "✅" if is_traversable else "❌"
        if len(traversable_centers) <= 10 or not is_traversable:  # Show first 10 good ones and all bad ones
            print(f"  Tile ({tile_x:2d}, {tile_y:2d}) center ({center_x:3d}, {center_y:3d}): {status}")
    
    print(f"\nTraversable empty tile centers: {len(traversable_centers)}/{len(empty_tile_centers)} ({len(traversable_centers)/len(empty_tile_centers)*100:.1f}%)")
    
    # Test navigation between traversable centers
    print(f"\n" + "=" * 60)
    print("TESTING PATHFINDING BETWEEN TRAVERSABLE EMPTY TILE CENTERS")
    print("=" * 60)
    
    if len(traversable_centers) < 2:
        print("❌ Not enough traversable centers for navigation tests")
        return
    
    # Select some test pairs
    test_pairs = []
    
    # Add nearby pairs (same row/column)
    for i, (x1, y1, tx1, ty1) in enumerate(traversable_centers):
        for j, (x2, y2, tx2, ty2) in enumerate(traversable_centers[i+1:], i+1):
            # Same row or column
            if tx1 == tx2 or ty1 == ty2:
                distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                if distance <= 100:  # Within 100 pixels
                    test_pairs.append(((x1, y1), (x2, y2), distance, "same_row_col"))
            
            if len(test_pairs) >= 5:
                break
        if len(test_pairs) >= 5:
            break
    
    # Add diagonal pairs
    for i, (x1, y1, tx1, ty1) in enumerate(traversable_centers):
        for j, (x2, y2, tx2, ty2) in enumerate(traversable_centers[i+1:], i+1):
            distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            if 50 <= distance <= 150:  # Medium distance diagonal
                test_pairs.append(((x1, y1), (x2, y2), distance, "diagonal"))
            
            if len(test_pairs) >= 10:
                break
        if len(test_pairs) >= 10:
            break
    
    print(f"Testing {len(test_pairs)} path pairs:")
    
    successful_paths = 0
    
    for i, (src_pos, tgt_pos, distance, path_type) in enumerate(test_pairs, 1):
        src_x, src_y = src_pos
        tgt_x, tgt_y = tgt_pos
        
        # Test path traversability
        is_traversable = collision_detector.is_path_traversable(src_x, src_y, tgt_x, tgt_y, tiles, 10)
        
        status = "✅" if is_traversable else "❌"
        print(f"  {i:2d}. ({src_x:3d}, {src_y:3d}) -> ({tgt_x:3d}, {tgt_y:3d}) [{path_type:12s}] (dist: {distance:5.1f}): {status}")
        
        if is_traversable:
            successful_paths += 1
    
    success_rate = (successful_paths / len(test_pairs)) * 100 if test_pairs else 0
    print(f"\nPath success rate: {successful_paths}/{len(test_pairs)} ({success_rate:.1f}%)")
    
    # Test with different ninja radii
    print(f"\n" + "=" * 60)
    print("TESTING WITH DIFFERENT NINJA RADII")
    print("=" * 60)
    
    if test_pairs:
        # Use first few test pairs
        sample_pairs = test_pairs[:3]
        
        for radius in [10, 8, 6, 4]:
            print(f"\nNinja radius: {radius} pixels")
            
            radius_successful = 0
            
            for i, (src_pos, tgt_pos, distance, path_type) in enumerate(sample_pairs, 1):
                src_x, src_y = src_pos
                tgt_x, tgt_y = tgt_pos
                
                is_traversable = collision_detector.is_path_traversable(src_x, src_y, tgt_x, tgt_y, tiles, radius)
                
                status = "✅" if is_traversable else "❌"
                print(f"  {i}. ({src_x:3d}, {src_y:3d}) -> ({tgt_x:3d}, {tgt_y:3d}) (dist: {distance:5.1f}): {status}")
                
                if is_traversable:
                    radius_successful += 1
            
            radius_rate = (radius_successful / len(sample_pairs)) * 100
            print(f"  Success rate: {radius_successful}/{len(sample_pairs)} ({radius_rate:.1f}%)")
    
    # Analysis
    print(f"\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    if success_rate >= 80:
        print("✅ EXCELLENT: Pathfinding between empty tile centers works well!")
        print("   The collision detection system is working correctly.")
        print("   The graph fragmentation issue is likely in the edge building logic.")
    elif success_rate >= 50:
        print("✅ GOOD: Most paths between empty tile centers work.")
        print("   Some improvements possible but system is functional.")
    elif success_rate >= 20:
        print("⚠️  MODERATE: Some paths work but many are blocked.")
        print("   Need to investigate specific blocking issues.")
    else:
        print("❌ POOR: Very few paths work between empty tile centers.")
        print("   Collision detection may be too restrictive.")
    
    return {
        'empty_tiles': len(empty_tile_centers),
        'traversable_centers': len(traversable_centers),
        'path_success_rate': success_rate,
        'successful_paths': successful_paths,
        'total_test_pairs': len(test_pairs)
    }


if __name__ == '__main__':
    results = test_proper_empty_positions()
    print(f"\nTest completed. Results: {results}")