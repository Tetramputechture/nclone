"""
Map loading utilities for N++ environment.

This module contains logic for loading different types of maps in the N++ environment.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Optional


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
    ):
        """
        Initialize the map loader.

        Args:
            nplay_headless: The NPlayHeadless instance to load maps into
            rng: Random number generator for map selection
            eval_mode: Whether to use evaluation maps
            custom_map_path: Path to custom map file
        """
        self.nplay_headless = nplay_headless
        self.rng = rng
        self.eval_mode = eval_mode
        self.custom_map_path = custom_map_path

        # Map state
        self.current_map_name = None
        self.random_map_type = None

        # Load map categorization data
        self._map_categories = self._load_map_categorization()

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
        # If a custom map path is provided, use that instead of default behavior
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
        if True:
            # Eval mode will load a random JUMP_REQUIRED or MAZE map
            self.random_map_type = self.rng.choice(
                [
                    # "JUMP_REQUIRED",
                    # "MAZE",
                    "SIMPLE_HORIZONTAL_NO_BACKTRACK",
                    # "MULTI_CHAMBER",
                    # "MINE_MAZE",
                ]
            )
            self.current_map_name = f"eval_map_{uuid.uuid4()}"
            self.nplay_headless.load_random_map(self.random_map_type)
            # self.load_random_categorized_map()
            return

        # Load the test map 'doortest' for training
        # TODO: This is hardcoded for testing, should be made configurable
        self.current_map_name = "complex-path-switch-required"
        self.nplay_headless.load_map("nclone/test_maps/complex-path-switch-required")

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

    def get_categorization_stats(self) -> dict:
        """
        Get statistics about available categorized maps.

        Returns:
            Dictionary with counts of simple and complex maps
        """
        return {
            "simple_count": len(self._map_categories.get("simple", [])),
            "complex_count": len(self._map_categories.get("complex", [])),
            "total_count": len(self._map_categories.get("simple", []))
            + len(self._map_categories.get("complex", [])),
        }
