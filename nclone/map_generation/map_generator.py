"""Map generator module for procedural level generation."""

import os
import random
from typing import Literal, Optional
from .map import Map
from .map_maze import MazeGenerator
from .map_single_chamber import SingleChamberGenerator
from .map_multi_chamber import MultiChamberGenerator
from .map_jump_required import MapJumpRequired


def generate_map(level_type: Literal["MAZE", "SIMPLE_HORIZONTAL_NO_BACKTRACK", "MULTI_CHAMBER", "JUMP_REQUIRED"] = "MAZE",
                 seed: Optional[int] = None) -> Map:
    """Generate a level of the specified type.

    Args:
        level_type: Type of level to generate ("MAZE", "SIMPLE_HORIZONTAL_NO_BACKTRACK", "MULTI_CHAMBER", or "JUMP_REQUIRED")
        width: Width of the map (only used for maze generation)
        height: Height of the map (only used for maze generation)
        seed: Random seed for reproducible generation
    Returns:
        Map: A Map instance with the generated level
    """
    if level_type == "MAZE":
        generator = MazeGenerator(seed=seed)
        return generator.generate()
    elif level_type == "SIMPLE_HORIZONTAL_NO_BACKTRACK":
        generator = SingleChamberGenerator(seed=seed)
        return generator.generate()
    elif level_type == "MULTI_CHAMBER":
        generator = MultiChamberGenerator(seed=seed)
        return generator.generate()
    elif level_type == "JUMP_REQUIRED":
        generator = MapJumpRequired(seed=seed)
        return generator.generate()


def random_official_map(rng: random.Random):
    """
    Load a random official map from the maps/official folder.
    """
    map_files = [f for f in os.listdir('maps/official')]
    map_file = rng.choice(map_files)
    map_path = os.path.join('maps/official', map_file)
    map_data = [int(b) for b in open(map_path, "rb").read()]
    return map_data
