"""
Registry for map generators by category.

This module provides a centralized, configuration-driven system for mapping
difficulty categories to their corresponding map generation functions.
"""

from ..map_generation.map import Map
from ..map_generation.generator_factory import GeneratorFactory
from ..map_generation.generator_configs import CATEGORIES


class GeneratorRegistry:
    """Registry mapping difficulty categories to generators using config system.

    This class provides a clean, extensible way to access map generators.
    Adding new categories or generators requires only configuration changes
    in generator_configs.py - no code changes needed here.
    """

    def __init__(self, rng):
        """Initialize registry with random number generator.

        Args:
            rng: Random number generator for generator selection
        """
        self.rng = rng
        # Load from centralized config
        self.categories = CATEGORIES

    def get_map(self, category: str, seed: int) -> Map:
        """Generate a map for the specified category.

        Args:
            category: Difficulty category name
            seed: Random seed for generation

        Returns:
            Generated Map instance

        Raises:
            ValueError: If category is invalid
        """
        if category not in self.categories:
            raise ValueError(
                f"Unknown category: {category}. "
                f"Available: {list(self.categories.keys())}"
            )

        config = self.categories[category]

        # Select random generator from category's generator list
        gen_type, preset = self.rng.choice(config.generators)

        # Use factory for all generators
        map_gen = GeneratorFactory.create_from_preset(gen_type, preset, seed)
        map_gen.generate(seed=seed)
        return map_gen
