"""
Map loading utilities for N++ environment.

This module contains logic for loading different types of maps in the N++ environment.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Optional

from ..evaluation import TestSuiteLoader
from ..map_generation.generate_test_suite_maps import TestSuiteGenerator


# Path to the map categorization JSON file
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
    ):
        """
        Initialize the map loader.

        Args:
            nplay_headless: The NPlayHeadless instance to load maps into
            rng: Random number generator for map selection
            eval_mode: Whether to use evaluation maps
            custom_map_path: Path to custom map file
            curriculum_stage: Current curriculum stage for difficulty control (optional)
                             Options: 'simple', 'medium', 'complex', 'mine_heavy', 'exploration'
        """
        self.nplay_headless = nplay_headless
        self.rng = rng
        self.eval_mode = eval_mode
        self.custom_map_path = custom_map_path
        self.curriculum_stage = curriculum_stage

        # Map state
        self.current_map_name = None
        self.random_map_type = None

        # Load map categorization data
        self._map_categories = self._load_map_categorization()

        # Test suite for evaluation (sequential loading)
        self._test_suite_levels = self._load_test_suite_levels()
        self._test_suite_index = 0

        # Test suite generator for training (random map generation)
        self._train_generator = None  # Lazy initialization
        self._train_seed_counter = 100000  # Start from high seed range for training

        # Curriculum learning weights (can be dynamically adjusted)
        self._curriculum_weights = self._get_default_curriculum_weights()

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
            loader = TestSuiteLoader("datasets/test")
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

        # Load training maps using Test Suite Generator
        # This ensures diverse, procedurally generated levels for robust training
        if self._train_generator is None:
            self._train_generator = TestSuiteGenerator("datasets/train_runtime")

        # Determine difficulty category based on curriculum stage or weighted sampling
        if self.curriculum_stage:
            # Use curriculum stage directly if specified
            category = self.curriculum_stage
        else:
            # Randomly select difficulty category for training diversity
            # Use weighted sampling to favor simpler levels early in training
            categories = [
                "simplest",
                "simpler",
                "simple",
                "medium",
                "complex",
                "mine_heavy",
                "exploration",
            ]
            category_weights = self._curriculum_weights

            # Convert to cumulative weights for random.choices
            import random

            if isinstance(self.rng, random.Random):
                category = self.rng.choices(categories, weights=category_weights, k=1)[
                    0
                ]
            else:
                # Fallback to simple choice if not using random.Random
                category = self.rng.choice(categories)

        # Generate a map from the selected category
        self._train_seed_counter += 1
        seed = self._train_seed_counter

        # Generate map based on category using available generator methods
        if category == "simplest":
            # Extremely simple levels (no jump, width 3)
            # Minimal simple level takes an index parameter
            def map_func(seed):
                return self._train_generator._create_minimal_simple_level(seed, 0)

            map_gen = map_func(seed)
        if category == "simpler":
            # Very simple levels
            # Minimal simple level takes an index parameter
            def map_func(seed):
                return self._train_generator._create_minimal_simple_level(
                    seed, self.rng.randint(0, 100000)
                )

            map_gen = map_func(seed)
        elif category == "simple":
            # Randomly select from simple level generators
            generators = [
                self._train_generator._create_simple_jump_level,
                self._train_generator._create_simple_hills_level,
                self._train_generator._create_simple_vertical_corridor,
            ]
            map_gen = self.rng.choice(generators)(seed)
        elif category == "medium":
            # Randomly select from medium level generators
            generators = [
                self._train_generator._create_medium_jump_level,
                self._train_generator._create_medium_hills_level,
                self._train_generator._create_medium_vertical_corridor,
                self._train_generator._create_medium_jump_platforms,
            ]
            map_gen = self.rng.choice(generators)(seed)
        elif category == "complex":
            # Randomly select from complex level generators
            generators = [
                self._train_generator._create_complex_mine_maze,
                self._train_generator._create_complex_jump_level,
                self._train_generator._create_complex_hills_level,
                self._train_generator._create_complex_islands_map,
            ]
            map_gen = self.rng.choice(generators)(seed)
        elif category == "mine_heavy":
            # Heavy mine levels
            generators = [
                self._train_generator._create_heavy_mine_maze,
                self._train_generator._create_heavy_mine_jump,
            ]
            map_gen = self.rng.choice(generators)(seed)
        else:  # exploration
            # Exploration levels
            generators = [
                self._train_generator._create_exploration_maze,
                self._train_generator._create_exploration_multi_chamber,
            ]
            map_gen = self.rng.choice(generators)(seed)

        # Load the generated map
        self.nplay_headless.load_map_from_map_data(map_gen.map_data())

        # Update state
        self.current_map_name = f"train_{category}_{seed}"
        self.random_map_type = None

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
        metadata_path = (
            Path(__file__).parent.parent.parent
            / "datasets"
            / "test"
            / "test_metadata.json"
        )

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
            stage: Curriculum stage name ('simple', 'medium', 'complex', 'mine_heavy', 'exploration')

        Raises:
            ValueError: If stage is invalid
        """
        valid_stages = [
            "simpler",
            "simple",
            "medium",
            "complex",
            "mine_heavy",
            "exploration",
        ]
        if stage not in valid_stages:
            raise ValueError(
                f"Invalid curriculum stage '{stage}'. Must be one of: {valid_stages}"
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
                    Example: {'simple': 50, 'medium': 30, 'complex': 15, 'mine_heavy': 5, 'exploration': 0}

        Raises:
            ValueError: If weights are invalid
        """
        categories = [
            "simplest",
            "simpler",
            "simple",
            "medium",
            "complex",
            "mine_heavy",
            "exploration",
        ]

        # Validate all categories are present
        for cat in categories:
            if cat not in weights:
                raise ValueError(f"Missing weight for category '{cat}'")

        # Validate all weights are non-negative
        for cat, weight in weights.items():
            if weight < 0:
                raise ValueError(
                    f"Weight for '{cat}' must be non-negative, got {weight}"
                )

        # Update weights
        self._curriculum_weights = [weights[cat] for cat in categories]
        print(f"Curriculum weights updated: {weights}")

    def _get_default_curriculum_weights(self) -> list:
        """Get default curriculum weights for training.

        Returns:
            List of weights for [simple, medium, complex, mine_heavy, exploration]
        """
        # Default weights: favor simpler levels
        return [10, 10, 30, 30, 20, 10, 10]

    def reset_curriculum_weights(self) -> None:
        """Reset curriculum weights to default values."""
        self._curriculum_weights = self._get_default_curriculum_weights()
        print("Curriculum weights reset to default")

    def get_curriculum_info(self) -> dict:
        """Get curriculum learning configuration info.

        Returns:
            Dictionary with curriculum settings
        """
        categories = [
            "simplest",
            "simpler",
            "simple",
            "medium",
            "complex",
            "mine_heavy",
            "exploration",
        ]
        weights_dict = {
            cat: weight for cat, weight in zip(categories, self._curriculum_weights)
        }

        return {
            "current_stage": self.curriculum_stage,
            "category_weights": weights_dict,
            "eval_mode": self.eval_mode,
            "using_curriculum": self.curriculum_stage is not None,
        }
