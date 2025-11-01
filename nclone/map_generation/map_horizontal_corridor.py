"""Horizontal corridor map generator for N++ levels.

This generator creates simple horizontal corridors with minimal obstacles.
These are among the simplest level types in the game.
"""

from typing import Optional
import numpy as np

from .map import Map
from .constants import VALID_TILE_TYPES, GRID_SIZE_FACTOR
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class MapHorizontalCorridor(Map):
    """Generates minimal horizontal corridor levels."""

    # Configuration parameters
    MIN_WIDTH = 3
    MAX_WIDTH = 23
    MIN_HEIGHT = 1
    MAX_HEIGHT = 5
    RANDOM_EDGE_TILES = False
    FIXED_HEIGHT = None

    def generate(self, seed: Optional[int] = None) -> "MapHorizontalCorridor":
        """Generate a minimal horizontal corridor level.

        Args:
            seed: Random seed for reproducible generation

        Returns:
            Self for method chaining
        """
        if seed is not None:
            self.rng.seed(seed)

        self.reset()

        # Calculate level index from seed for parameter variation
        index = seed % 100 if seed is not None else 0

        # Determine dimensions based on index
        max_width = self.MIN_WIDTH + (index % 20)
        max_height = self.MIN_HEIGHT + (index % 5)

        width = self.rng.randint(self.MIN_WIDTH, min(max_width, self.MAX_WIDTH))

        if self.FIXED_HEIGHT is not None:
            height = self.FIXED_HEIGHT
        else:
            height = self.rng.randint(self.MIN_HEIGHT, min(max_height, self.MAX_HEIGHT))

        # Random offset for the chamber
        max_start_x = MAP_TILE_WIDTH - width - 1
        max_start_y = MAP_TILE_HEIGHT - height - 1
        start_x = self.rng.randint(1, max_start_x)
        start_y = self.rng.randint(1, max_start_y)

        # Fill with random tiles
        tile_types = [
            self.rng.randint(0, VALID_TILE_TYPES)
            for _ in range(MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
        ]
        self.set_tiles_bulk(tile_types)

        # Create empty chamber
        for y in range(start_y, start_y + height):
            for x in range(start_x, start_x + width):
                self.set_tile(x, y, 0)

        self.set_hollow_rectangle(
            start_x - 1,
            start_y - 1,
            start_x + width,
            start_y + height,
            use_random_tiles_type=True,
            chaotic_random_tiles=self.RANDOM_EDGE_TILES,
        )

        # Randomly choose ninja starting side
        ninja_on_left = self.rng.choice([True, False])

        if ninja_on_left:
            ninja_x = start_x
            ninja_orientation = 1
        else:
            ninja_x = start_x + width - 1
            ninja_orientation = -1

        ninja_y = start_y + height - 1

        # Check if we should add a locked door
        can_add_locked_door = height == 1 and width >= 4
        add_locked_door = can_add_locked_door and self.rng.choice([True, False])

        # Generate positions with quarter-tile increments
        num_positions = (width - 1) * 4
        available_positions = [start_x + i * 0.25 for i in range(num_positions)]

        # Filter positions to avoid ninja spawn and edges
        available_positions = [
            pos
            for pos in available_positions
            if pos > start_x + 0.25 and pos < start_x + width - 0.25
        ]
        available_positions = [
            pos for pos in available_positions if abs(pos - ninja_x) >= 1
        ]

        # For doors, filter to only integer positions (24-pixel boundaries)
        door_positions = [pos for pos in available_positions if pos == int(pos)]

        # Place entities based on layout complexity
        locked_door_viable = False
        if (
            add_locked_door
            and len(door_positions) >= 2
            and len(available_positions) >= 4
        ):
            # Sample 2 door positions (must be integers for 24-pixel alignment)
            door_pos = sorted(self.rng.sample(door_positions, k=2))

            # For each door, find valid switch positions
            switch_available = [p for p in available_positions if p not in door_pos]

            # Check if we have enough switch positions between ninja and doors
            if ninja_on_left:
                locked_switch_candidates = [
                    p for p in switch_available if p < door_pos[0]
                ]
                exit_switch_candidates = [
                    p for p in switch_available if door_pos[0] < p < door_pos[1]
                ]
            else:
                locked_switch_candidates = [
                    p for p in switch_available if p > door_pos[1]
                ]
                exit_switch_candidates = [
                    p for p in switch_available if door_pos[0] < p < door_pos[1]
                ]

            # Only proceed if we have valid switch positions
            if locked_switch_candidates and exit_switch_candidates:
                locked_door_viable = True
                if ninja_on_left:
                    locked_switch_x = self.rng.choice(locked_switch_candidates)
                    exit_switch_x = self.rng.choice(exit_switch_candidates)
                    locked_door_x = door_pos[0]
                    exit_door_x = door_pos[1]
                else:
                    locked_switch_x = self.rng.choice(locked_switch_candidates)
                    exit_switch_x = self.rng.choice(exit_switch_candidates)
                    locked_door_x = door_pos[1]
                    exit_door_x = door_pos[0]
                entity_y = start_y

        if locked_door_viable:
            # Place locked door and exit door
            self.set_ninja_spawn(ninja_x, ninja_y, orientation=ninja_orientation)

            # Convert switch positions preserving fractional coordinates
            locked_switch_grid = (
                int(locked_switch_x * GRID_SIZE_FACTOR) / GRID_SIZE_FACTOR
            )
            exit_switch_grid = int(exit_switch_x * GRID_SIZE_FACTOR) / GRID_SIZE_FACTOR

            self.add_entity(
                6,
                int(locked_door_x),
                entity_y,
                4,
                0,
                locked_switch_grid,
                entity_y,
            )
            self.add_entity(
                3,
                int(exit_door_x),
                entity_y,
                0,
                0,
                exit_switch_grid,
                entity_y,
            )
        else:
            # Exit door only - use integer positions for door
            if len(door_positions) >= 2:
                positions = sorted(self.rng.sample(door_positions, k=2))
            else:
                # Fallback if not enough integer positions
                positions = sorted(self.rng.sample(available_positions, k=2))

            if not ninja_on_left:
                positions = positions[::-1]

            exit_switch_x, exit_door_x = positions
            exit_switch_y = start_y
            exit_door_y = start_y

            if height > 1:
                exit_switch_y = start_y + self.rng.randint(1, height - 1) * 0.25
                exit_door_y = start_y + self.rng.randint(1, height - 1) * 0.25

            self.set_ninja_spawn(ninja_x, ninja_y, orientation=ninja_orientation)
            self.add_entity(
                3,
                exit_door_x,
                exit_door_y,
                0,
                0,
                exit_switch_x,
                exit_switch_y,
            )

        # Situations where height is 1 and random edge tiles are almost impossible to achieve at first.
        if not (self.RANDOM_EDGE_TILES and height == 1) and (
            not self.RANDOM_EDGE_TILES and height != 2
        ):
            # Add mines evenly spaced along the ceiling of the corridor to discourage random jumping.
            min_mines = min(1, width - 1)
            max_mines = max(1, width - 1)
            num_mines = self.rng.randint(min_mines, max_mines)

            mine_y = start_y + 1

            if width >= 4:
                x_start = start_x + 0.5
                x_end = start_x + width + 0.5
                if ninja_on_left:
                    mine_x_positions = np.linspace(x_start, x_end, num=num_mines)
                else:
                    mine_x_positions = np.linspace(x_end, x_start, num=num_mines)
            else:
                # fallback: just pack them left to right (or right to left) for very narrow corridors
                if ninja_on_left:
                    mine_x_positions = [start_x + i + 2 for i in range(num_mines)]
                else:
                    mine_x_positions = [
                        start_x + width - 2 - i for i in range(num_mines)
                    ]

            # make sure no mine x is within 12px of ninja_x
            mine_x_positions = [
                x for x in mine_x_positions if abs(x - ninja_x - 1) >= 1
            ]

            for mine_x in mine_x_positions:
                self.add_entity(1, float(mine_x), mine_y)

        # Add random entities outside the playspace
        self.add_random_entities_outside_playspace(
            start_x - 2,
            start_y - 2,
            start_x + width + 2,
            start_y + height + 2,
        )

        return self
