#!/usr/bin/env python3
"""Visualize entity placement to debug positioning."""

from nclone.map_generation.map_islands import MapIslands
from nclone.constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


def main():
    """Visualize map with entity positions."""
    print("Generating island map to visualize entity positions...")
    print("Seed: 42\n")

    map_gen = MapIslands(seed=42)
    map_gen.generate(seed=42)

    # Get entity positions
    ninja_x = int(map_gen.ninja_spawn_x // 24)
    ninja_y = int(map_gen.ninja_spawn_y // 24)

    entity_data = map_gen.entity_data
    i = 0
    door_x, door_y, switch_x, switch_y = None, None, None, None

    while i < len(entity_data):
        entity_type = entity_data[i]
        if entity_type == 3:
            door_x = int(entity_data[i + 1] // 24)
            door_y = int(entity_data[i + 2] // 24)
            switch_x = int(entity_data[i + 6] // 24)
            switch_y = int(entity_data[i + 7] // 24)
            break
        i += 5

    print(f"Entity positions:")
    print(f"  Ninja:  ({ninja_x}, {ninja_y})")
    print(f"  Switch: ({switch_x}, {switch_y})")
    print(f"  Door:   ({door_x}, {door_y})")

    # Show a zoomed-in view of the top-left area where entities are
    print("\nZoomed view of entity area (first 15x15 tiles):")
    print("  N=Ninja, S=Switch, D=Door, #=Solid, .=Empty")
    print("  " + "".join([str(i % 10) for i in range(15)]))

    for y in range(15):
        row = f"{y:2d}"
        for x in range(15):
            tile_idx = x + y * MAP_TILE_WIDTH

            # Check for entity
            if (x, y) == (ninja_x, ninja_y):
                row += "N"
            elif (x, y) == (switch_x, switch_y):
                row += "S"
            elif (x, y) == (door_x, door_y):
                row += "D"
            # Check tile type
            elif map_gen.tile_data[tile_idx] == 0:
                row += "."
            else:
                row += "#"
        print(row)

    print("\nAnalysis:")
    for name, ex, ey in [
        ("Ninja", ninja_x, ninja_y),
        ("Switch", switch_x, switch_y),
        ("Door", door_x, door_y),
    ]:
        if 0 <= ex < MAP_TILE_WIDTH and 0 <= ey < MAP_TILE_HEIGHT:
            tile_idx = ex + ey * MAP_TILE_WIDTH
            tile_at = map_gen.tile_data[tile_idx]
            print(
                f"  {name} at ({ex},{ey}): tile={tile_at} {'[SOLID - INSIDE TILE!]' if tile_at != 0 else '[EMPTY - OK]'}"
            )


if __name__ == "__main__":
    main()
