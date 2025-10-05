"""
Map loading utilities for N++ environment.

This module contains logic for loading different types of maps in the N++ environment.
"""

import os
import uuid
from typing import Optional


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
                    # "SIMPLE_HORIZONTAL_NO_BACKTRACK",
                    # "MULTI_CHAMBER",
                    "MINE_MAZE",
                ]
            )
            self.current_map_name = f"eval_map_{uuid.uuid4()}"
            self.nplay_headless.load_random_map(self.random_map_type)
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
