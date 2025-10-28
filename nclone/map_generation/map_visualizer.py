"""ASCII visualization utility for N++ map debugging.

This module provides tools to visualize map tiles and entities as ASCII art
for quick debugging and verification of map generation.
"""

import sys
import pickle
from pathlib import Path
from typing import List
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


# Tile visualization characters
TILE_CHARS = {
    0: ".",  # Empty
    1: "#",  # Full solid
    2: "=",  # Top half solid
    3: "]",  # Right half solid
    4: "=",  # Bottom half solid
    5: "[",  # Left half solid
    6: "\\",  # Slope top-left to bottom-right
    7: "/",  # Slope top-right to bottom-left
    8: "/",  # Slope bottom-left to top-right
    9: "\\",  # Slope bottom-right to top-left
    10: "o",  # Bottom-right quarter circle
    11: "o",  # Bottom-left quarter circle
    12: "o",  # Top-left quarter circle
    13: "o",  # Top-right quarter circle
    14: "c",  # Top-left quarter pipe
    15: "c",  # Top-right quarter pipe
    16: "c",  # Bottom-right quarter pipe
    17: "c",  # Bottom-left quarter pipe
    18: "~",  # Mild slope up-left
    19: "~",  # Mild slope up-right
    20: "~",  # Mild slope down-right
    21: "~",  # Mild slope down-left
    22: "~",  # Raised mild slope
    23: "~",  # Raised mild slope
    24: "~",  # Raised mild slope
    25: "~",  # Raised mild slope
    26: "^",  # Steep slope up-left
    27: "^",  # Steep slope up-right
    28: "v",  # Steep slope down-right
    29: "v",  # Steep slope down-left
    30: "^",  # Raised steep slope
    31: "^",  # Raised steep slope
    32: "v",  # Raised steep slope
    33: "v",  # Raised steep slope
}


def get_tile_char(tile_type: int) -> str:
    """Get ASCII character for a tile type."""
    return TILE_CHARS.get(tile_type, "?")


def parse_entities_from_map_data(map_data: List[int]) -> dict:
    """Parse entities from map data.

    Returns dictionary with entity positions:
    - ninja: (x, y, orientation)
    - doors: [(x, y), ...]
    - switches: [(x, y), ...]
    - mines: [(x, y), ...]
    """
    entities = {
        "ninja": None,
        "doors": [],
        "switches": [],
        "mines": [],
    }

    # Parse ninja spawn (indices 1231-1233)
    ninja_x = map_data[1231] // 6  # Convert from map data units to tiles
    ninja_y = map_data[1232] // 6
    ninja_orientation = map_data[1233]
    entities["ninja"] = (ninja_x, ninja_y, ninja_orientation)

    # Parse entity data (starts at index 1235)
    entity_data = map_data[1235:]
    i = 0
    while i < len(entity_data):
        if i + 4 >= len(entity_data):
            break

        entity_type = entity_data[i]
        entity_x = entity_data[i + 1] // 6  # Convert to tiles
        entity_y = entity_data[i + 2] // 6

        if entity_type == 3:  # Exit door
            entities["doors"].append((entity_x, entity_y))
            # Next entity should be switch (type 4)
            i += 5
            if i + 4 < len(entity_data) and entity_data[i] == 4:
                switch_x = entity_data[i + 1] // 6
                switch_y = entity_data[i + 2] // 6
                entities["switches"].append((switch_x, switch_y))
                i += 5
        elif entity_type in (1, 21):  # Mines (toggle mine, toggled mine)
            entities["mines"].append((entity_x, entity_y))
            i += 5
        elif entity_type in (6, 8):  # Locked door or trap door (9 bytes)
            i += 9
        else:
            i += 5

    return entities


def visualize_map(map_data: List[int], show_coords: bool = True) -> str:
    """Generate ASCII visualization of a map.

    Args:
        map_data: Map data array (1235+ elements)
        show_coords: Whether to show coordinate labels

    Returns:
        String containing ASCII art representation
    """
    # Extract tile data (indices 184-1149)
    tile_data = map_data[184:1150]

    # Parse entities
    entities = parse_entities_from_map_data(map_data)

    # Create entity position lookup for quick access
    entity_positions = {}
    if entities["ninja"]:
        x, y, _ = entities["ninja"]
        entity_positions[(x, y)] = "N"
    for x, y in entities["doors"]:
        entity_positions[(x, y)] = "D"
    for x, y in entities["switches"]:
        entity_positions[(x, y)] = "S"
    for x, y in entities["mines"]:
        # If position already has an entity, use combined marker
        if (x, y) in entity_positions:
            entity_positions[(x, y)] = "X"
        else:
            entity_positions[(x, y)] = "M"

    # Build the visualization
    lines = []

    if show_coords:
        # Top coordinate labels (every 5 tiles)
        header = "    "
        for x in range(0, MAP_TILE_WIDTH, 5):
            header += f"{x:<5}"
        lines.append(header)
        lines.append("   +" + "-" * MAP_TILE_WIDTH + "+")
    else:
        lines.append("+" + "-" * MAP_TILE_WIDTH + "+")

    # Build each row
    for y in range(MAP_TILE_HEIGHT):
        row_chars = []
        for x in range(MAP_TILE_WIDTH):
            idx = x + y * MAP_TILE_WIDTH
            tile_type = tile_data[idx]

            # Check if there's an entity at this position
            if (x, y) in entity_positions:
                row_chars.append(entity_positions[(x, y)])
            else:
                row_chars.append(get_tile_char(tile_type))

        if show_coords:
            lines.append(f"{y:2d} |" + "".join(row_chars) + "|")
        else:
            lines.append("|" + "".join(row_chars) + "|")

    if show_coords:
        lines.append("   +" + "-" * MAP_TILE_WIDTH + "+")
    else:
        lines.append("+" + "-" * MAP_TILE_WIDTH + "+")

    # Add legend
    lines.append("")
    lines.append("Legend:")
    lines.append(
        "  . = empty   # = solid   / \\ = slopes   ~ = mild slope   ^ v = steep slope"
    )
    lines.append(
        "  N = ninja   D = door    S = switch     M = mine         X = overlapping"
    )

    return "\n".join(lines)


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python -m nclone.map_generation.map_visualizer <map_file.pkl>")
        print("       python -m nclone.map_generation.map_visualizer --test")
        sys.exit(1)

    if sys.argv[1] == "--test":
        # Create a simple test map
        from .map import Map

        test_map = Map(seed=42)
        test_map.reset()
        # Fill with some test tiles
        for y in range(10, 15):
            for x in range(5, 20):
                if y == 14:
                    test_map.set_tile(x, y, 1)  # Floor
                else:
                    test_map.set_tile(x, y, 0)  # Empty
        # Add boundaries
        for x in range(5, 20):
            test_map.set_tile(x, 9, 1)
            test_map.set_tile(x, 15, 1)
        for y in range(9, 16):
            test_map.set_tile(4, y, 1)
            test_map.set_tile(20, y, 1)
        # Add entities
        test_map.set_ninja_spawn(7, 14, 1)
        test_map.add_entity(3, 16, 14, 0, 0, 12, 14)  # Door and switch
        test_map.add_entity(1, 10, 14)  # Mine

        map_data = test_map.map_data()
        print(visualize_map(map_data))
    else:
        map_file = Path(sys.argv[1])
        if not map_file.exists():
            print(f"Error: File not found: {map_file}")
            sys.exit(1)

        with open(map_file, "rb") as f:
            level_data = pickle.load(f)

        if isinstance(level_data, dict) and "map_data" in level_data:
            map_data = level_data["map_data"]
        elif isinstance(level_data, list):
            map_data = level_data
        else:
            print("Error: Unknown map data format")
            sys.exit(1)

        print(f"\nMap: {map_file.name}")
        print(visualize_map(map_data))


if __name__ == "__main__":
    main()
