#!/usr/bin/env python3
"""Test script to verify entity positions are correctly adjusted for map padding."""

from nclone.map_generation.map_islands import MapIslands
from nclone.constants import MAP_TILE_WIDTH


def test_entity_positions():
    """Test that entities are positioned correctly (not inside tiles)."""
    print("Testing entity position adjustments for map padding...")

    # Test with multiple seeds
    for seed in [42, 100, 200, 300]:
        print(f"\n--- Testing seed {seed} ---")

        map_gen = MapIslands(seed=seed)
        map_gen.generate(seed=seed)

        # Extract entity positions (in grid coordinates)
        ninja_x = int(map_gen.ninja_spawn_x // 24)
        ninja_y = int(map_gen.ninja_spawn_y // 24)

        # Find switch and door positions from entity data
        entity_data = map_gen.entity_data
        i = 0
        door_x, door_y, switch_x, switch_y = None, None, None, None

        while i < len(entity_data):
            entity_type = entity_data[i]
            if entity_type == 3:  # Exit door
                door_x = int(entity_data[i + 1] // 24)
                door_y = int(entity_data[i + 2] // 24)
                switch_x = int(entity_data[i + 6] // 24)
                switch_y = int(entity_data[i + 7] // 24)
                break
            i += 5

        print(f"Entity positions (grid coords):")
        print(f"  Ninja:  ({ninja_x}, {ninja_y})")
        print(f"  Switch: ({switch_x}, {switch_y})")
        print(f"  Door:   ({door_x}, {door_y})")

        # Check what's at the entity positions and one tile to the right/down
        def check_tile(x, y, name):
            """Check tile at position and surrounding tiles."""
            if 0 <= x < MAP_TILE_WIDTH and 0 <= y < 25:
                tile_idx = x + y * MAP_TILE_WIDTH
                tile_at_pos = map_gen.tile_data[tile_idx]

                # Check tile below (entity should be standing on this)
                tile_below_idx = x + (y + 1) * MAP_TILE_WIDTH
                tile_below = (
                    map_gen.tile_data[tile_below_idx]
                    if tile_below_idx < len(map_gen.tile_data)
                    else -1
                )

                # Check tile to the right+below (diagonal)
                tile_diag_idx = (x + 1) + (y + 1) * MAP_TILE_WIDTH
                tile_diag = (
                    map_gen.tile_data[tile_diag_idx]
                    if tile_diag_idx < len(map_gen.tile_data)
                    else -1
                )

                print(f"  {name}:")
                print(
                    f"    At position ({x},{y}): tile={tile_at_pos} {'[EMPTY]' if tile_at_pos == 0 else '[SOLID]'}"
                )
                print(
                    f"    Below ({x},{y + 1}): tile={tile_below} {'[EMPTY]' if tile_below == 0 else '[SOLID]'}"
                )
                print(
                    f"    Diagonal ({x + 1},{y + 1}): tile={tile_diag} {'[EMPTY]' if tile_diag == 0 else '[SOLID]'}"
                )

                # Entity should be in empty space with island below or diagonally below
                is_valid = tile_at_pos == 0 and (tile_below != 0 or tile_diag != 0)
                return is_valid
            return False

        print(f"\nValidation:")
        ninja_valid = check_tile(ninja_x, ninja_y, "Ninja")
        switch_valid = check_tile(switch_x, switch_y, "Switch")
        door_valid = check_tile(door_x, door_y, "Door")

        all_valid = ninja_valid and switch_valid and door_valid
        print(f"\n  ✓ All entities correctly positioned: {all_valid}")

    print("\n" + "=" * 50)
    print("✓ Entity position adjustment test completed!")


if __name__ == "__main__":
    test_entity_positions()
