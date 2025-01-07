"""Test the map generator module."""

import random
from map_generation.map_generator import MapGenerator
import sys
import os

# Temporarily add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_map(map_gen: MapGenerator):
    """Print a text representation of the map."""
    for y in range(map_gen.height):
        for x in range(map_gen.width):
            if (x, y) == map_gen.ninja_pos:
                print("N", end=" ")
            elif (x, y) == map_gen.exit_door_pos:
                print("D", end=" ")
            elif (x, y) == map_gen.exit_switch_pos:
                print("S", end=" ")
            elif (x, y) in map_gen.gold_positions:
                print("G", end=" ")
            else:
                print("█" if map_gen.grid[y][x] == 1 else " ", end=" ")
        print()


def main():
    """Test the map generator."""
    # Test maze generation
    print("Testing maze generation...")
    map_gen = MapGenerator(10, 10, seed=42)
    map_gen.generate("MAZE")
    print("Generated Maze:")
    print_map(map_gen)
    print()

    # Test simple horizontal level generation
    print("Testing simple horizontal level generation...")
    map_gen = MapGenerator()
    map_gen.generate("SIMPLE_HORIZONTAL_NO_BACKTRACK", seed=42)
    print("Generated Simple Horizontal Level:")
    # Note: Simple horizontal levels use the Map class directly, so we can't print them
    # in the same way. Instead, we'll print some stats about the generated level
    print(f"Ninja spawn: ({map_gen.ninja_spawn_x}, {map_gen.ninja_spawn_y})")
    print(f"Exit door count: {map_gen.entity_counts['exit_door']}")
    print(f"Gold count: {map_gen.entity_counts['gold']}")


if __name__ == "__main__":
    main()
