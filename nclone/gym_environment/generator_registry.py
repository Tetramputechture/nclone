"""
Registry for map generators by category.

This module provides a centralized, configuration-driven system for mapping
difficulty categories to their corresponding map generation functions.
"""

from ..map_generation.map import Map
from ..map_generation.generator_factory import GeneratorFactory
from ..map_generation.generator_configs import (
    bucket_difficulty_view,
    get_category_config,
    get_category_names,
)


class GeneratorRegistry:
    """Registry mapping difficulty bands to generators using the config system.

    The per-band ``(generator, preset)`` lists are sourced from
    ``bucket_difficulty_view()`` -- a derived projection of ``BUCKET_CATALOG``
    (the single source of truth). This means every generator wired into a
    mechanic bucket (now all 23 ``map_*`` classes) is reachable from this
    difficulty-banded interactive/test path, and the two taxonomies cannot drift.

    Adding new generators requires only catalog/preset changes in
    generator_configs.py -- no code changes here.
    """

    def __init__(self, rng):
        """Initialize registry with random number generator.

        Args:
            rng: Random number generator for generator selection
        """
        self.rng = rng
        # Per-band generator lists derived from the mechanic bucket catalog.
        self.categories = bucket_difficulty_view()

    def get_map(self, category: str, seed: int) -> Map:
        """Generate a map for the specified difficulty band.

        Args:
            category: Difficulty band name (see curriculum_config.CATEGORY_NAMES)
            seed: Random seed for generation

        Returns:
            Generated Map instance

        Raises:
            ValueError: If the band is unknown or empty.
        """
        if category in self.categories:
            generators = self.categories[category]
        elif category in get_category_names():
            # Legacy CATEGORIES entry that isn't wired to a mechanic bucket, so it
            # doesn't appear in the bucket-derived difficulty view above (e.g.
            # "extremely_simple", an on-ramp below "simplest"). curriculum_config can
            # still weight these, so serve them faithfully from the category catalog
            # -- both stores use the same (generator_type, preset) pair format --
            # rather than crashing an env reset / obs-space probe with a seed that
            # happens to sample one.
            generators = get_category_config(category).generators
        else:
            raise ValueError(
                f"Unknown category: {category}. "
                f"Available: {list(self.categories.keys())} "
                f"(+ catalog: {get_category_names()})"
            )

        if not generators:
            raise ValueError(f"Difficulty band {category!r} has no generators")

        # Select a random (generator_type, preset) pair for this band.
        gen_type, preset = self.rng.choice(generators)

        # Use factory for all generators
        map_gen = GeneratorFactory.create_from_preset(gen_type, preset, seed)
        map_gen.generate(seed=seed)
        return map_gen
