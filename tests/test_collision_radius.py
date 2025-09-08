#!/usr/bin/env python3
"""
Test the impact of ninja radius on collision detection and graph connectivity.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nclone"))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import (
    BasicLevelNoGold,
)
from nclone.graph.precise_collision import PreciseTileCollision
from nclone.graph.common import SUB_CELL_SIZE


def test_collision_radius_impact():
    """Test how ninja radius affects collision detection."""
    print("=" * 80)
    print("TESTING NINJA RADIUS IMPACT ON COLLISION DETECTION")
    print("=" * 80)

    # Create environment
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42,
    )

    # Reset to load the map
    env.reset()

    # Get level data
    level_data = env.level_data
    tiles = level_data.tiles

    print(f"Map: {level_data.width}x{level_data.height} tiles")
    print(f"Sub-cell size: {SUB_CELL_SIZE} pixels")

    # Create collision detector
    collision_detector = PreciseTileCollision()

    # Test different ninja radii
    test_radii = [10, 8, 6, 4, 2]  # Original is 10

    print(f"\nTesting traversability with different ninja radii:")

    # Test positions in empty tiles
    test_positions = [
        (156, 252),  # Empty tile
        (180, 252),  # Empty tile
        (228, 276),  # Empty tile
        (204, 300),  # Empty tile
        (156, 324),  # Empty tile
    ]

    for radius in test_radii:
        print(f"\n--- Ninja Radius: {radius} pixels ---")

        traversable_positions = 0

        for pos in test_positions:
            x, y = pos

            # Check if position is traversable
            is_traversable = collision_detector._is_position_traversable(
                x, y, tiles, radius
            )

            # Check tile type
            tile_x = int(x // 24)
            tile_y = int(y // 24)
            tile_value = (
                level_data.get_tile(tile_y, tile_x)
                if (0 <= tile_x < level_data.width and 0 <= tile_y < level_data.height)
                else -1
            )
            tile_type = (
                "empty" if tile_value == 0 else "solid" if tile_value == 1 else "other"
            )

            status = "✅" if is_traversable else "❌"
            print(f"  {pos} ({tile_type}): {status}")

            if is_traversable:
                traversable_positions += 1

        traversability_rate = (traversable_positions / len(test_positions)) * 100
        print(
            f"  Traversability rate: {traversable_positions}/{len(test_positions)} ({traversability_rate:.1f}%)"
        )

    # Test path traversability between positions
    print(f"\n" + "=" * 60)
    print("TESTING PATH TRAVERSABILITY BETWEEN POSITIONS")
    print("=" * 60)

    # Test paths between empty tile positions
    path_tests = [
        ((156, 252), (180, 252)),  # Same row
        ((156, 252), (156, 276)),  # Same column
        ((156, 252), (228, 276)),  # Diagonal
        ((228, 276), (204, 300)),  # Different areas
    ]

    for radius in test_radii:
        print(f"\n--- Ninja Radius: {radius} pixels ---")

        traversable_paths = 0

        for src_pos, tgt_pos in path_tests:
            src_x, src_y = src_pos
            tgt_x, tgt_y = tgt_pos

            # Check if path is traversable
            is_traversable = collision_detector.is_path_traversable(
                src_x, src_y, tgt_x, tgt_y, tiles, radius
            )

            status = "✅" if is_traversable else "❌"
            distance = ((tgt_x - src_x) ** 2 + (tgt_y - src_y) ** 2) ** 0.5
            print(f"  {src_pos} -> {tgt_pos} (dist: {distance:.1f}): {status}")

            if is_traversable:
                traversable_paths += 1

        path_traversability_rate = (traversable_paths / len(path_tests)) * 100
        print(
            f"  Path traversability rate: {traversable_paths}/{len(path_tests)} ({path_traversability_rate:.1f}%)"
        )


if __name__ == "__main__":
    test_collision_radius_impact()
