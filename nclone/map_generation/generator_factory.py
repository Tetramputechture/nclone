"""
Generator factory for creating configured map generators.

This module provides a centralized factory for instantiating map generators
with preset configurations. It acts as the bridge between configuration
and generator implementation.
"""

from typing import Dict, Type, Callable
from .map import Map
from .map_maze import MazeGenerator
from .map_jump_required import MapJumpRequired
from .map_hills import MapHills
from .map_vertical_corridor import MapVerticalCorridor
from .map_mine_maze import MapMineMaze
from .map_islands import MapIslands
from .map_jump_platforms import MapJumpPlatforms
from .map_multi_chamber import MultiChamberGenerator
from .map_single_chamber import SingleChamberGenerator
from .generator_configs import GENERATOR_PRESETS


GENERATOR_REGISTRY: Dict[str, Type[Map]] = {
    "maze": MazeGenerator,
    "jump_required": MapJumpRequired,
    "hills": MapHills,
    "vertical_corridor": MapVerticalCorridor,
    "mine_maze": MapMineMaze,
    "islands": MapIslands,
    "jump_platforms": MapJumpPlatforms,
    "multi_chamber": MultiChamberGenerator,
    "single_chamber": SingleChamberGenerator,
}


class GeneratorFactory:
    """Factory for creating and configuring map generators."""

    @staticmethod
    def create_generator(generator_type: str, config, seed: int) -> Map:
        """Create and configure a generator with the given config.

        Args:
            generator_type: Key from GENERATOR_REGISTRY
            config: Config dataclass instance with parameters
            seed: Random seed for generation

        Returns:
            Configured Map generator ready to call .generate()

        Raises:
            ValueError: If generator_type is not recognized
        """
        if generator_type not in GENERATOR_REGISTRY:
            raise ValueError(
                f"Unknown generator type: {generator_type}. "
                f"Available: {list(GENERATOR_REGISTRY.keys())}"
            )

        generator_class = GENERATOR_REGISTRY[generator_type]
        generator = generator_class(seed=seed)

        # Apply config to generator attributes (convert to uppercase class constants)
        for key, value in config.__dict__.items():
            attr_name = key.upper()
            setattr(generator, attr_name, value)

        return generator

    @staticmethod
    def create_from_preset(generator_type: str, preset: str, seed: int) -> Map:
        """Convenience method to create generator using preset configs.

        Args:
            generator_type: Key from GENERATOR_REGISTRY
            preset: Preset name from GENERATOR_PRESETS
            seed: Random seed for generation

        Returns:
            Configured Map generator ready to call .generate()

        Raises:
            ValueError: If generator_type or preset is not recognized
        """
        if generator_type not in GENERATOR_PRESETS:
            raise ValueError(
                f"Unknown generator type: {generator_type}. "
                f"Available: {list(GENERATOR_PRESETS.keys())}"
            )

        if preset not in GENERATOR_PRESETS[generator_type]:
            raise ValueError(
                f"Unknown preset '{preset}' for generator '{generator_type}'. "
                f"Available: {list(GENERATOR_PRESETS[generator_type].keys())}"
            )

        config = GENERATOR_PRESETS[generator_type][preset]
        return GeneratorFactory.create_generator(generator_type, config, seed)


def create_horizontal_generator(
    seed: int, index: int, height: int = None
) -> Callable[[int], Map]:
    """Create a horizontal level generator function.

    This is a special case for horizontal minimal levels that use the base Map class
    with custom generation logic from TestSuiteGenerator.

    Args:
        seed: Random seed
        index: Level index for parameter variation
        height: Optional fixed height

    Returns:
        Generator function that creates a horizontal level
    """

    def generator(s: int) -> Map:
        from .generate_test_suite_maps import TestSuiteGenerator

        gen = TestSuiteGenerator()
        return gen._create_minimal_simple_level_horizontal(s, index, height)

    return generator
