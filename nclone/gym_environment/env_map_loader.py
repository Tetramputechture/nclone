"""
Map loading utilities for N++ environment.

This module handles map loading for training and evaluation, with support for
curriculum learning and procedural generation.
"""

import json
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
MAP_CATEGORIZATION_PATH = (
    Path(__file__).parent.parent.parent / "map_categorization.json"
)


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
        self.test_dataset_path = test_dataset_path or "datasets/test"

        # Current map state
        self.current_map_name = None
        self.random_map_type = None

        # Map categorization data (for legacy categorized map loading)
        self._map_categories = self._load_map_categorization()

        # Test suite for evaluation (sequential loading)
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
            map_name = os.path.basename(self.custom_map_path)
            if not map_name:  # Handle trailing slash case
                map_name = os.path.basename(os.path.dirname(self.custom_map_path))
            self.current_map_name = map_name
            self.random_map_type = None
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
            self._generator_registry = GeneratorRegistry(
                self._train_generator, self.rng
            )

        # Select difficulty category
        category = self._select_category()

        # Generate map using registry
        self._train_seed_counter += 1
        seed = self._train_seed_counter
        map_gen = self._generator_registry.get_map(category, seed)

        # Load the generated map
        self.nplay_headless.load_map_from_map_data(map_gen.map_data())

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

    def _load_map_categorization(self) -> dict:
        """
        Load the map categorization JSON file.

        Returns:
            Dictionary containing simple and complex map lists, or empty dict if file not found
        """
        if not MAP_CATEGORIZATION_PATH.exists():
            return {"simple": [], "complex": []}

        try:
            with open(MAP_CATEGORIZATION_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load map categorization: {e}")
            return {"simple": [], "complex": []}

    def _load_test_suite_levels(self) -> list:
        """
        Load the test suite level IDs from metadata JSON in sequential order.

        Returns:
            Ordered list of level IDs starting with simple levels
        """
        metadata_path = Path(self.test_dataset_path) / "test_metadata.json"

        if not metadata_path.exists():
            print(f"Warning: Test suite metadata not found at {metadata_path}")
            return []

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Extract levels in order: simple, medium, complex, mine_heavy, exploration
            ordered_levels = []
            category_order = [
                "simple",
                "medium",
                "complex",
                "mine_heavy",
                "exploration",
            ]

            for category in category_order:
                if category in metadata.get("categories", {}):
                    ordered_levels.extend(metadata["categories"][category]["level_ids"])

            return ordered_levels
        except Exception as e:
            print(f"Warning: Failed to load test suite metadata: {e}")
            return []

    def load_random_categorized_map(self, category: str = "simple"):
        """
        Load a random map from the specified category (simple or complex).

        Args:
            category: Either "simple" or "complex" to select map category

        Raises:
            ValueError: If category is invalid or no maps available in that category
        """
        if category not in ["simple", "complex"]:
            raise ValueError(
                f"Invalid category '{category}'. Must be 'simple' or 'complex'."
            )

        maps_list = self._map_categories.get(category, [])
        if not maps_list:
            raise ValueError(f"No {category} maps available in categorization file.")

        # Select a random map from the category
        map_entry = self.rng.choice(maps_list)
        map_name = map_entry["name"]

        # Construct the full path to the map file
        map_path = (
            Path(__file__).parent.parent
            / "maps"
            / "official"
            / map_entry["folder"]
            / map_name
        )

        # Update current map state
        self.current_map_name = f"{category.capitalize()}: {map_name}"
        self.random_map_type = None

        # Load the map
        self.nplay_headless.load_map(str(map_path))

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
