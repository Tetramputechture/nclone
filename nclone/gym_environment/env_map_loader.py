"""
Map loading utilities for N++ environment.

This module handles map loading for training and evaluation, with support for
curriculum learning and procedural generation.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from ..evaluation import TestSuiteLoader
from ..map_generation.generate_test_suite_maps import TestSuiteGenerator
from .curriculum_config import (
    CATEGORY_NAMES,
    get_default_weights,
    validate_category,
    validate_weights,
)
from .generator_registry import GeneratorRegistry


# Constants
TRAIN_SEED_START = 100000  # Starting seed for training map generation


class EnvMapLoader:
    """
    Handles map loading functionality for the N++ environment.

    This class provides centralized logic for loading different types of maps
    including custom maps, evaluation maps, and training maps.
    """

    def __init__(
        self,
        nplay_headless,
        rng,
        eval_mode: bool = False,
        custom_map_path: Optional[str] = None,
        curriculum_stage: Optional[str] = None,
        test_dataset_path: Optional[str] = None,
    ):
        """Initialize the map loader.

        Args:
            nplay_headless: NPlayHeadless instance to load maps into
            rng: Random number generator for map selection
            eval_mode: Whether to use evaluation maps (sequential)
            custom_map_path: Path to custom map file (overrides all other loading)
            curriculum_stage: Current curriculum difficulty stage (optional)
                             See curriculum_config.CATEGORY_NAMES for options
            test_dataset_path: Path to test dataset directory for evaluation (defaults to "datasets/test")
        """
        self.nplay_headless = nplay_headless
        self.rng = rng
        self.eval_mode = eval_mode
        self.custom_map_path = custom_map_path
        self.curriculum_stage = curriculum_stage
        # Resolve test_dataset_path to absolute path to ensure consistent resolution
        # regardless of current working directory (important for distributed training)
        # Paths are typically already resolved by ArchitectureTrainer, but we resolve here
        # as a safety measure in case paths are passed directly
        # When test_dataset_path is None (single-level mode), store None to skip test loading
        if test_dataset_path is not None:
            path_obj = Path(test_dataset_path)
            # If path is already absolute, resolve() will return it unchanged
            # If relative, resolve() will make it absolute relative to current working directory
            self.test_dataset_path = str(path_obj.resolve())
        else:
            self.test_dataset_path = None

        # Current map state
        self.current_map_name = None
        self.random_map_type = None

        # Test suite for evaluation (sequential loading)
        # Skip test suite loading if using custom map (not needed)
        self._test_suite_levels = self._load_test_suite_levels()
        self._test_suite_index = 0

        # Training map generation (lazy initialization)
        self._train_generator = None
        self._generator_registry = None
        self._train_seed_counter = TRAIN_SEED_START

        # Curriculum weights (category name -> relative weight)
        self._curriculum_weights = get_default_weights()

    def load_initial_map(self):
        """Load the first map based on configuration."""
        if self.eval_mode:
            self.current_map_name = f"eval_map_{uuid.uuid4()}"
            self.random_map_type = self.rng.choice(["JUMP_REQUIRED", "MAZE"])
            self.nplay_headless.load_random_map(self.random_map_type)
        else:
            # Load random map for training
            self.current_map_name = f"random_map_{uuid.uuid4()}"
            self.random_map_type = self.rng.choice(
                [
                    "SIMPLE_HORIZONTAL_NO_BACKTRACK",
                    "JUMP_REQUIRED",
                    "MAZE",
                ]
            )
            self.nplay_headless.load_random_map(self.random_map_type)

    def load_map(self):
        """Load the map specified by custom_map_path or follow default logic."""
        if self.custom_map_path:
            # Extract map name from path for display purposes
            # Use Path.stem to get filename without extension for consistent matching with demo seeding
            from pathlib import Path

            map_name = Path(self.custom_map_path).stem
            if not map_name:  # Handle edge case
                map_name = os.path.basename(self.custom_map_path)
            self.current_map_name = map_name
            self.random_map_type = None

            # Add logging to make custom map usage clear
            import logging

            logger = logging.getLogger(__name__)
            logger.info(f"Custom map path set: {self.custom_map_path}")
            logger.info("  → Bypassing ALL dataset selection (train/test/curriculum)")
            logger.info(
                "  → This single level will be used for all training and evaluation"
            )

            self.nplay_headless.load_map(self.custom_map_path)
            return

        # If we are in eval mode, load evaluation maps
        if self.eval_mode:
            # Load test suite maps sequentially
            if not self._test_suite_levels:
                print(
                    "Warning: No test suite levels loaded, falling back to random map"
                )
                self.random_map_type = "SIMPLE_HORIZONTAL_NO_BACKTRACK"
                self.current_map_name = f"eval_map_{uuid.uuid4()}"
                self.nplay_headless.load_random_map(self.random_map_type)
                return

            # Get the current level ID and load it
            level_id = self._test_suite_levels[self._test_suite_index]
            loader = TestSuiteLoader(self.test_dataset_path)
            level = loader.get_level(level_id)
            self.nplay_headless.load_map_from_map_data(level["map_data"])

            # Update state
            self.current_map_name = level_id
            self.random_map_type = None

            # Advance to next level (with wraparound)
            self._test_suite_index = (self._test_suite_index + 1) % len(
                self._test_suite_levels
            )
            return

        # Load training maps using procedural generation
        # Lazy initialization of generator and registry
        if self._train_generator is None:
            self._train_generator = TestSuiteGenerator("datasets/train_runtime")
            self._generator_registry = GeneratorRegistry(self.rng)

        # Select difficulty category
        category = self._select_category()

        # Generate map using registry
        self._train_seed_counter += 1
        seed = self._train_seed_counter
        map_gen = self._generator_registry.get_map(category, seed)

        # Get map data
        map_data = map_gen.map_data()

        # Validate that map has required entities (exit door at index 1156)
        exit_door_count = map_data[1156]
        if exit_door_count == 0:
            raise RuntimeError(
                f"Generated map is invalid: no exit door!\n"
                f"  Category: {category}\n"
                f"  Seed: {seed}\n"
                f"This indicates a bug in the map generator."
            )

        # Validate ninja spawn position
        ninja_spawn_x = map_data[1231]
        ninja_spawn_y = map_data[1232]
        if ninja_spawn_x == 0 and ninja_spawn_y == 0:
            raise RuntimeError(
                f"Generated map has invalid ninja spawn at (0, 0)!\n"
                f"  Category: {category}\n"
                f"  Seed: {seed}\n"
                f"This indicates a bug in the map generator."
            )

        # Load the validated map
        self.nplay_headless.load_map_from_map_data(map_data)

        # Update state
        self.current_map_name = f"train_{category}_{seed}"
        self.random_map_type = None

    def _select_category(self) -> str:
        """Select a difficulty category for map generation.

        Uses curriculum stage if set, otherwise samples based on weights.

        Returns:
            Category name
        """
        # Use curriculum stage if explicitly set
        if self.curriculum_stage:
            return self.curriculum_stage

        # Otherwise, use weighted random sampling
        import random

        if isinstance(self.rng, random.Random):
            # Use weighted sampling for training diversity
            category = self.rng.choices(
                CATEGORY_NAMES, weights=self._curriculum_weights, k=1
            )[0]
        else:
            # Fallback to uniform sampling if RNG doesn't support choices
            category = self.rng.choice(CATEGORY_NAMES)

        return category

    def get_map_display_name(self) -> str:
        """Get the display name for the current map."""
        return (
            self.current_map_name
            if self.random_map_type is None
            else f"Random {self.random_map_type}"
        )

    def _load_test_suite_levels(self) -> list:
        """
        Load the test suite level IDs from metadata JSON in sequential order.

        Returns:
            Ordered list of level IDs starting with simple levels, or empty list if not available
        """
        # If using custom map or test_dataset_path is None, skip test suite loading
        if self.custom_map_path or self.test_dataset_path is None:
            logger = logging.getLogger(__name__)
            if self.custom_map_path:
                logger.debug("Using custom map - skipping test suite metadata loading")
            else:
                logger.debug(
                    "No test_dataset_path provided - skipping test suite metadata loading"
                )
            return []

        # self.test_dataset_path is already resolved to absolute path in __init__
        metadata_path = Path(self.test_dataset_path) / "test_metadata.json"

        if not metadata_path.exists():
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Test suite metadata not found at {metadata_path} "
                f"(resolved from: {self.test_dataset_path}). "
                f"Evaluation mode will fall back to random maps."
            )
            return []

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Extract levels in order: simple, medium, complex, mine_heavy, exploration
        ordered_levels = []
        category_order = TestSuiteLoader.CATEGORIES

        for category in category_order:
            if category in metadata.get("categories", {}):
                ordered_levels.extend(metadata["categories"][category]["level_ids"])

        return ordered_levels

    def set_curriculum_stage(self, stage: str) -> None:
        """Set the current curriculum stage for map selection.

        Args:
            stage: Curriculum stage name (see curriculum_config.CATEGORY_NAMES)

        Raises:
            ValueError: If stage is invalid
        """
        if not validate_category(stage):
            raise ValueError(
                f"Invalid curriculum stage '{stage}'. Must be one of: {CATEGORY_NAMES}"
            )

        self.curriculum_stage = stage
        print(f"Curriculum stage set to: {stage}")

    def get_curriculum_stage(self) -> Optional[str]:
        """Get the current curriculum stage.

        Returns:
            Current curriculum stage name, or None if not using curriculum learning
        """
        return self.curriculum_stage

    def set_curriculum_weights(self, weights: dict) -> None:
        """Set custom category weights for curriculum learning.

        Args:
            weights: Dictionary mapping category names to relative weights
                    Example: {'simplest': 10, 'simple': 50, 'medium': 30, ...}

        Raises:
            ValueError: If weights are invalid
        """
        # Validate weights using centralized validation
        is_valid, error_msg = validate_weights(weights)
        if not is_valid:
            raise ValueError(error_msg)

        # Update weights (convert dict to list in category order)
        self._curriculum_weights = [weights[cat] for cat in CATEGORY_NAMES]
        print(f"Curriculum weights updated: {weights}")

    def reset_curriculum_weights(self) -> None:
        """Reset curriculum weights to default values."""
        self._curriculum_weights = get_default_weights()
        print("Curriculum weights reset to default")

    def get_curriculum_info(self) -> dict:
        """Get curriculum learning configuration info.

        Returns:
            Dictionary with curriculum settings including weights and stage
        """
        # Create weights dictionary from current weights list
        weights_dict = {
            cat: weight for cat, weight in zip(CATEGORY_NAMES, self._curriculum_weights)
        }

        return {
            "current_stage": self.curriculum_stage,
            "category_weights": weights_dict,
            "eval_mode": self.eval_mode,
            "using_curriculum": self.curriculum_stage is not None,
        }
