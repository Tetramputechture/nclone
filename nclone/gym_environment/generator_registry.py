"""
Registry for map generators by category.

This module provides a centralized, data-driven system for mapping difficulty
categories to their corresponding map generation functions.
"""

from typing import Callable, List
from ..map_generation.generate_test_suite_maps import TestSuiteGenerator
from ..map_generation.map import Map


class GeneratorRegistry:
    """Registry mapping difficulty categories to generator functions.

    This class provides a clean, extensible way to add new map generation
    functions without modifying the map loading logic.
    """

    def __init__(self, generator: TestSuiteGenerator, rng):
        """Initialize registry with a generator instance.

        Args:
            generator: TestSuiteGenerator instance
            rng: Random number generator for selection
        """
        self.generator = generator
        self.rng = rng

        # Registry: category_name -> list of generator functions
        # Each generator function takes a seed and returns a Map
        self._registry = {
            "simplest": self._get_simplest_generators(),
            "simpler": self._get_simpler_generators(),
            "simple": self._get_simple_generators(),
            "medium": self._get_medium_generators(),
            "complex": self._get_complex_generators(),
            "mine_heavy": self._get_mine_heavy_generators(),
            "exploration": self._get_exploration_generators(),
        }

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
        if category not in self._registry:
            raise ValueError(f"Unknown category: {category}")

        # Select random generator from category's generator list
        generators = self._registry[category]
        generator_func = self.rng.choice(generators)

        # Generate and return map
        return generator_func(seed)

    # ========================================================================
    # Generator function lists for each category
    # ========================================================================

    def _get_simplest_generators(self) -> List[Callable[[int], Map]]:
        """Get generators for simplest difficulty (no randomness)."""
        return [
            lambda seed: self.generator._create_minimal_simple_level_vertical_corridor(
                seed, 0
            ),
            lambda seed: self.generator._create_minimal_simple_level_horizontal(
                seed, 0, height=1
            ),
        ]

    def _get_simpler_generators(self) -> List[Callable[[int], Map]]:
        """Get generators for simpler difficulty (slight variations)."""
        return [
            lambda seed: self.generator._create_minimal_simple_level_horizontal(
                seed, self.rng.randint(0, 100000, height=1)
            ),
            lambda seed: self.generator._create_minimal_simple_level_vertical_corridor(
                seed, self.rng.randint(0, 100000)
            ),
        ]

    def _get_simple_generators(self) -> List[Callable[[int], Map]]:
        """Get generators for simple difficulty."""
        return [
            lambda seed: self.generator._create_minimal_simple_level_horizontal(
                seed, self.rng.randint(0, 100000)
            ),
            self.generator._create_tiny_maze,
            self.generator._create_simple_jump_level,
            self.generator._create_simple_hills_level,
            self.generator._create_simple_vertical_corridor,
        ]

    def _get_medium_generators(self) -> List[Callable[[int], Map]]:
        """Get generators for medium difficulty."""
        return [
            self.generator._create_medium_jump_level,
            self.generator._create_medium_hills_level,
            self.generator._create_medium_vertical_corridor,
            self.generator._create_medium_jump_platforms,
        ]

    def _get_complex_generators(self) -> List[Callable[[int], Map]]:
        """Get generators for complex difficulty."""
        return [
            self.generator._create_complex_mine_maze,
            self.generator._create_complex_jump_level,
            self.generator._create_complex_hills_level,
            self.generator._create_complex_islands_map,
        ]

    def _get_mine_heavy_generators(self) -> List[Callable[[int], Map]]:
        """Get generators for mine-heavy difficulty."""
        return [
            self.generator._create_heavy_mine_maze,
            self.generator._create_heavy_mine_jump,
        ]

    def _get_exploration_generators(self) -> List[Callable[[int], Map]]:
        """Get generators for exploration difficulty."""
        return [
            self.generator._create_exploration_maze,
            self.generator._create_exploration_multi_chamber,
        ]
