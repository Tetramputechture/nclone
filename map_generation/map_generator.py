"""Map generator module for procedural level generation."""

from typing import Literal, Optional
from map_generation.map import Map
from map_generation.map_maze import MazeGenerator
from map_generation.map_single_chamber import SingleChamberGenerator
from map_generation.map_multi_chamber import MultiChamberGenerator


def generate_map(level_type: Literal["MAZE", "SIMPLE_HORIZONTAL_NO_BACKTRACK", "MULTI_CHAMBER"] = "MAZE",
                 width: int = 10,
                 height: int = 10,
                 seed: Optional[int] = None) -> Map:
    """Generate a level of the specified type.

    Args:
        level_type: Type of level to generate ("MAZE", "SIMPLE_HORIZONTAL_NO_BACKTRACK", or "MULTI_CHAMBER")
        width: Width of the map (only used for maze generation)
        height: Height of the map (only used for maze generation)
        seed: Random seed for reproducible generation
    Returns:
        Map: A Map instance with the generated level
    """
    if level_type == "MAZE":
        generator = MazeGenerator(width=width, height=height, seed=seed)
        return generator.generate()
    elif level_type == "SIMPLE_HORIZONTAL_NO_BACKTRACK":
        generator = SingleChamberGenerator(seed=seed)
        return generator.generate()
    elif level_type == "MULTI_CHAMBER":
        generator = MultiChamberGenerator(seed=seed)
        return generator.generate()
    else:
        raise ValueError(f"Unknown level type: {level_type}")
