"""Single chamber generation module for N++ levels."""

from .map import Map
from typing import Optional
from .constants import VALID_TILE_TYPES
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class SingleChamberGenerator(Map):
    """Generates simple horizontal N++ levels with no backtracking required."""

    # Simple horizontal level constants
    MIN_WIDTH = 4
    MAX_WIDTH = 30
    MIN_HEIGHT = 4
    MAX_HEIGHT = 10
    GLOBAL_MAX_UP_DEVIATION = 5
    GLOBAL_MAX_DOWN_DEVIATION = 1
    ADD_CENTER_OBSTACLE = False
    ADD_FLOOR_MINES = False
    ADD_PLATFORM_GAP = False
    ADD_CEILING_OBSTACLES = False

    def set_empty_rectangle(self, x1: int, y1: int, x2: int, y2: int):
        """Set a rectangular region of the map to empty space."""
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                self.set_tile(x, y, 0)

    def generate(
        self,
        seed: Optional[int] = None,
        add_center_obstacle: Optional[bool] = None,
        add_floor_mines: Optional[bool] = None,
        add_platform_gap: Optional[bool] = None,
        add_ceiling_obstacles: Optional[bool] = None,
    ) -> Map:
        """Generate a simple horizontal level with no backtracking required.

        Args:
            seed: Random seed for reproducible generation
            add_center_obstacle: Add single tile outcropping in middle (defaults to class attribute)
            add_floor_mines: Scatter mines on floor (defaults to class attribute)
            add_platform_gap: Create small gap requiring jump (defaults to class attribute)
            add_ceiling_obstacles: Add hanging obstacles from ceiling (defaults to class attribute)

        Returns:
            Map: A Map instance with the generated level
        """
        if seed is not None:
            self.rng.seed(seed)

        self.reset()

        # Use class attributes as defaults if parameters not provided
        if add_center_obstacle is None:
            add_center_obstacle = self.ADD_CENTER_OBSTACLE
        if add_floor_mines is None:
            add_floor_mines = self.ADD_FLOOR_MINES
        if add_platform_gap is None:
            add_platform_gap = self.ADD_PLATFORM_GAP
        if add_ceiling_obstacles is None:
            add_ceiling_obstacles = self.ADD_CEILING_OBSTACLES

        # Generate random dimensions for play space
        width = self.rng.randint(self.MIN_WIDTH, self.MAX_WIDTH)
        height = self.rng.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)

        # Calculate maximum possible starting positions
        max_start_x = MAP_TILE_WIDTH - width - 1
        max_start_y = MAP_TILE_HEIGHT - height - 1

        # Randomize the starting position
        play_x1 = self.rng.randint(2, max(3, max_start_x))
        play_y1 = self.rng.randint(2, max(3, max_start_y))
        play_x2 = min(play_x1 + width, MAP_TILE_WIDTH - 2)
        play_y2 = min(play_y1 + height, MAP_TILE_HEIGHT - 2)

        actual_width = play_x2 - play_x1
        actual_height = play_y2 - play_y1

        # Pre-generate all random tiles at once
        # Choose if tiles will be random or solid for the border
        choice = self.rng.randint(0, 2)
        if choice == 0:
            tile_types = [
                self.rng.randint(0, VALID_TILE_TYPES)
                for _ in range(MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
            ]
        elif choice == 1:
            tile_types = [1] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT)  # Solid walls
        else:
            tile_types = [0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT)  # Empty tiles
        self.set_tiles_bulk(tile_types)
        # Create the empty play space
        self.set_empty_rectangle(play_x1, play_y1, play_x2, play_y2)

        # Create boundary tiles
        for x in range(play_x1, play_x2 + 1):
            self.set_tile(x, play_y2 + 1, 1)
            self.set_tile(x, play_y1 - 1, 1)

        for y in range(play_y1, play_y2 + 1):
            self.set_tile(play_x1 - 1, y, 1)
            self.set_tile(play_x2 + 1, y, 1)

        # Calculate entity positions
        usable_width = actual_width - 2
        section_width = max(3, usable_width // 3)

        # Pre-calculate max deviation values
        max_up = min(actual_height - 1, self.GLOBAL_MAX_UP_DEVIATION)
        max_down = 0

        # Place entities on X axis, choosing switch or door first
        if self.rng.choice([True, False]):
            switch_x = play_x1 + 1 + self.rng.randint(1, section_width - 1)
            door_x = switch_x + self.rng.randint(1, max(1, play_x2 - switch_x - 1))
            ninja_x = switch_x - self.rng.randint(1, max(1, switch_x - play_x1 - 1))
        else:
            door_x = play_x1 + self.rng.randint(1, section_width - 1)
            switch_x = door_x + self.rng.randint(1, max(1, play_x2 - door_x - 1))
            ninja_x = switch_x + self.rng.randint(1, max(1, play_x2 - switch_x))

        # Handle surface deviations
        deviations = {}
        should_deviate = self.rng.choice([True, False])
        should_deviate_tiles = self.rng.choice([True, False])

        for x in range(play_x1, play_x2 + 1):
            if should_deviate:
                deviation = self.rng.randint(-max_down, max_up)
            else:
                deviation = 0

            deviations[x] = deviation

            if should_deviate_tiles:
                if deviation < 0:
                    for y in range(play_y2 + 2, play_y2 - deviation, 1):
                        self.set_tile(x, y, 0)
                elif deviation > 0:
                    for y in range(play_y2, play_y2 - deviation, -1):
                        random_tile = self.rng.randint(1, VALID_TILE_TYPES)
                        self.set_tile(x, y, random_tile)

        # Calculate final entity positions
        if should_deviate_tiles:
            ninja_y = play_y2 - deviations.get(ninja_x, 0)
        else:
            ninja_y = play_y2
        door_y = play_y2 - deviations.get(door_x, 0)
        switch_y = play_y2 - deviations.get(switch_x, 0)

        # Ninja should be facing a random direction
        ninja_orientation = self.rng.choice([1, -1])

        # Convert to screen coordinates and place entities
        self.set_ninja_spawn(ninja_x, ninja_y, ninja_orientation)
        self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)

        # Add optional obstacles/hazards
        reserved_positions = {ninja_x, door_x, switch_x}

        if add_center_obstacle and actual_width >= 2:
            # Place obstacle in middle of chamber, 1 tile high from floor
            center_x = play_x1 + actual_width // 2
            if center_x not in reserved_positions:
                obstacle_y = play_y2 - 1
                # Randomly choose between solid tile or mine entity
                if self.rng.choice([True, False]):
                    # Solid tile (randomly choose from 1-33)
                    tile_type = self.rng.randint(1, 33)
                    self.set_tile(center_x, obstacle_y, tile_type)
                else:
                    # Mine entity (type 1 or 21)
                    mine_type = self.rng.choice([1, 21])
                    self.add_entity(mine_type, center_x, obstacle_y)

        if add_floor_mines:
            # Scatter 2-5 mines at floor level
            num_mines = self.rng.randint(2, min(5, actual_width - 3))
            available_x = [
                x for x in range(play_x1 + 1, play_x2) if x not in reserved_positions
            ]
            if len(available_x) >= num_mines:
                mine_positions = self.rng.sample(available_x, num_mines)
                for mine_x in mine_positions:
                    mine_y = play_y2 - deviations.get(mine_x, 0)
                    mine_type = self.rng.choice([1, 21])
                    self.add_entity(mine_type, mine_x, mine_y)

        if add_platform_gap and actual_width >= 6:
            # Remove 2-3 floor tiles in middle section to create gap
            gap_width = self.rng.randint(2, 3)
            gap_center = play_x1 + actual_width // 2
            gap_start = max(play_x1 + 2, gap_center - gap_width // 2)
            gap_end = min(play_x2 - 2, gap_start + gap_width)

            # Make sure gap doesn't overlap with entity positions
            if not any(x in reserved_positions for x in range(gap_start, gap_end)):
                for x in range(gap_start, gap_end):
                    self.set_tile(x, play_y2, 0)
                    self.set_tile(x, play_y2 + 1, 0)

        if add_ceiling_obstacles and actual_height >= 4:
            # Add 1-2 tiles hanging from ceiling
            num_obstacles = self.rng.randint(1, 2)
            available_x = [
                x for x in range(play_x1 + 1, play_x2) if x not in reserved_positions
            ]
            if len(available_x) >= num_obstacles:
                obstacle_positions = self.rng.sample(available_x, num_obstacles)
                for obs_x in obstacle_positions:
                    obs_y = play_y1
                    tile_type = self.rng.randint(1, 33)
                    self.set_tile(obs_x, obs_y, tile_type)

        # Add random entities outside playspace
        playspace = (play_x1 - 4, play_y1 - 4, play_x2 + 4, play_y2 + 4)
        self.add_random_entities_outside_playspace(
            playspace[0], playspace[1], playspace[2], playspace[3]
        )

        return self
