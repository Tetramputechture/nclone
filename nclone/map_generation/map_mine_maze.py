"""Map generation for a mine maze level.

The level consists of a single chamber with mines scattered throughout.
The player must navigate through the mines from one side to the other.

Key features:
- Single chamber with configurable dimensions
- Ninja spawns on left or right side
- Exit door on opposite side from ninja
- Exit switch placed between ninja and exit
- Mines placed in columns with random spacing (skip 1-3 columns) for navigation
- Each column has 1-10 mines, placed in a vertical sequence
- Mines in a sequence have variable vertical offset (12px or 24px = 2 or 4 tiles)
- At least 1 mine present
- Mines never within 14 pixels of player spawn
- Mines never overlap switch, door, or player position
"""

from .map import Map
from typing import Optional, Tuple, Set
from .constants import VALID_TILE_TYPES, GRID_SIZE_FACTOR
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class MapMineMaze(Map):
    """Generates N++ levels with mine-filled chambers."""

    # Chamber size constants
    MIN_WIDTH = 6
    MAX_WIDTH = MAP_TILE_WIDTH
    MIN_HEIGHT = 3
    MAX_HEIGHT = 6

    # Mine placement constants
    MIN_PLAYER_MINE_DISTANCE_PX = 14
    MIN_PLAYER_MINE_DISTANCE_TILES = (
        int(MIN_PLAYER_MINE_DISTANCE_PX / GRID_SIZE_FACTOR) - 2
    )  # 4 tiles
    MIN_SKIP_COLUMNS = 2
    MAX_SKIP_COLUMNS = 4
    MIN_MINES_PER_COLUMN = 1
    MAX_MINES_PER_COLUMN = 10

    # Vertical offset options for mine sequences (in tiles)
    # 12px = 2 tiles, 24px = 4 tiles
    MINE_VERTICAL_OFFSETS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

    def set_empty_rectangle(self, x1: int, y1: int, x2: int, y2: int):
        """Set a rectangular region of the map to empty space."""
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                self.set_tile(x, y, 0)

    def place_mines_in_chamber(
        self,
        chamber_x1: int,
        chamber_y1: int,
        chamber_x2: int,
        chamber_y2: int,
        ninja_x: int,
        ninja_y: int,
        door_x: int,
        door_y: int,
        switch_x: int,
        switch_y: int,
    ) -> int:
        """Place mines throughout the chamber with variable vertical offsets.

        Mines are placed in vertical sequences where each mine has a vertical

        offset of either 12px (2 tiles) or 24px (4 tiles) from the previous mine.

        Args:
            chamber_x1: Left edge of chamber
            chamber_y1: Top edge of chamber
            chamber_x2: Right edge of chamber
            chamber_y2: Bottom edge of chamber
            ninja_x: X position of ninja spawn
            ninja_y: Y position of ninja spawn
            door_x: X position of exit door
            door_y: Y position of exit door
            switch_x: X position of exit switch
            switch_y: Y position of exit switch

        Returns:
            Number of mines placed
        """
        total_mines = 0
        ninja_on_left = ninja_x < door_x

        # Collect forbidden positions (switch, door, near player)
        forbidden_positions: Set[Tuple[int, int]] = set()
        forbidden_positions.add((switch_x, switch_y))
        forbidden_positions.add((door_x, door_y))

        # Add positions near player spawn
        for dx in range(
            -self.MIN_PLAYER_MINE_DISTANCE_TILES,
            self.MIN_PLAYER_MINE_DISTANCE_TILES + 1,
        ):
            for dy in range(
                -self.MIN_PLAYER_MINE_DISTANCE_TILES,
                self.MIN_PLAYER_MINE_DISTANCE_TILES + 1,
            ):
                forbidden_positions.add((ninja_x + dx, ninja_y + dy))

        # Process columns with random skipping (2-4 columns per skip)
        col_x = chamber_x1 + 1 if ninja_on_left else chamber_x2 - 1
        end_col = chamber_x2 if ninja_on_left else chamber_x1
        direction = 1 if ninja_on_left else -1

        while (ninja_on_left and col_x <= end_col) or (
            not ninja_on_left and col_x >= end_col
        ):
            # Skip columns too close to door
            if abs(col_x - door_x) < 2:
                col_x += direction * self.rng.randint(
                    self.MIN_SKIP_COLUMNS, self.MAX_SKIP_COLUMNS
                )
                continue

            # Determine number of mines for this column
            num_mines = self.rng.randint(
                self.MIN_MINES_PER_COLUMN, self.MAX_MINES_PER_COLUMN
            )

            # Chamber bounds for mine placement
            # Limit mines to no more than 3 tiles from ground floor (chamber_y2)
            min_y = max(chamber_y1 + 2, chamber_y2 - 1)
            max_y = chamber_y2 + 2  # Include ground level

            # Pick a random starting position for the mine sequence
            # Start from bottom or top with equal probability
            start_from_bottom = self.rng.choice([True, False])

            if start_from_bottom:
                # Start from bottom and work upward
                current_y = max_y
                y_direction = -1  # Move up
            else:
                # Start from top and work downward
                current_y = min_y
                y_direction = 1  # Move down

            # Build mine sequence with variable offsets
            mine_positions = []
            for _ in range(num_mines):
                # Check if current position is valid
                if current_y < min_y or current_y > max_y:
                    break

                # Check if position is forbidden
                if (col_x, current_y) in forbidden_positions:
                    # Try next position with offset
                    offset = self.rng.choice(self.MINE_VERTICAL_OFFSETS)
                    current_y += y_direction * offset
                    continue

                # Valid position, add to sequence
                mine_positions.append(current_y)

                # Choose next offset (12px or 24px = 2 or 4 tiles)
                offset = self.rng.choice(self.MINE_VERTICAL_OFFSETS)
                current_y += y_direction * offset

            # Place mines at the positions
            for row_y in mine_positions:
                # Type 1 = toggle mine, mode 1 = active
                # Add plus or minus a range of 0.5 to col_x
                col_x += self.rng.choice([-0.5, 0.5])
                mine_type = self.rng.choice([1, 21])
                self.add_entity(mine_type, col_x, row_y, 0, 1)
                total_mines += 1

            # Skip 2-4 columns randomly
            col_x += direction * self.rng.randint(
                self.MIN_SKIP_COLUMNS, self.MAX_SKIP_COLUMNS
            )

        return total_mines

    def generate(self, seed: Optional[int] = None) -> Map:
        """Generate a mine maze level.

        Args:
            seed: Random seed for reproducible generation

        Returns:
            Map: A Map instance with the generated level
        """
        if seed is not None:
            self.rng.seed(seed)

        self.reset()

        # Generate random dimensions for chamber
        width = self.rng.randint(self.MIN_WIDTH, self.MAX_WIDTH)
        height = self.rng.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)

        # Calculate chamber position (centered or random)
        max_start_x = MAP_TILE_WIDTH - width - 1
        max_start_y = MAP_TILE_HEIGHT - height - 1

        # Randomize the starting position of the chamber
        chamber_x1 = self.rng.randint(2, max(3, max_start_x))
        chamber_y1 = self.rng.randint(2, max(3, max_start_y))
        chamber_x2 = min(chamber_x1 + width, MAP_TILE_WIDTH - 2)
        chamber_y2 = min(chamber_y1 + height, MAP_TILE_HEIGHT - 2)

        # Pre-generate background tiles (random, solid, or empty)
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

        # Create the empty chamber
        self.set_empty_rectangle(chamber_x1, chamber_y1, chamber_x2, chamber_y2)

        # Create boundary walls
        for x in range(chamber_x1, chamber_x2 + 1):
            self.set_tile(x, chamber_y2 + 1, 1)  # Bottom wall
            self.set_tile(x, chamber_y1 - 1, 1)  # Top wall

        for y in range(chamber_y1, chamber_y2 + 1):
            self.set_tile(chamber_x1 - 1, y, 1)  # Left wall
            self.set_tile(chamber_x2 + 1, y, 1)  # Right wall

        # Randomly choose which side the ninja starts on
        ninja_on_left = self.rng.choice([True, False])

        # Place ninja and door on opposite sides, on the ground
        ground_y = chamber_y2  # Bottom of chamber (ground level)

        if ninja_on_left:
            ninja_x = chamber_x1 + 1
            ninja_y = ground_y
            ninja_orientation = 1  # Face right
            door_x = chamber_x2 - 1
            door_y = ground_y
        else:
            ninja_x = chamber_x2 - 1
            ninja_y = ground_y
            ninja_orientation = -1  # Face left
            door_x = chamber_x1 + 1
            door_y = ground_y

        # Place switch between ninja and door, on the ground
        # Switch should be roughly in the middle, but with some randomness
        if ninja_on_left:
            min_switch_x = ninja_x + self.MIN_PLAYER_MINE_DISTANCE_TILES + 2
            max_switch_x = door_x - 2
            # Ensure valid range
            if min_switch_x >= max_switch_x:
                switch_x = (ninja_x + door_x) // 2  # Place in middle
            else:
                switch_x = self.rng.randint(min_switch_x, max_switch_x)
        else:
            min_switch_x = door_x + 2
            max_switch_x = ninja_x - self.MIN_PLAYER_MINE_DISTANCE_TILES - 2
            # Ensure valid range
            if min_switch_x >= max_switch_x:
                switch_x = (ninja_x + door_x) // 2  # Place in middle
            else:
                switch_x = self.rng.randint(min_switch_x, max_switch_x)

        # Switch y position is on the ground
        switch_y = ground_y

        # Place entities
        self.set_ninja_spawn(ninja_x, ninja_y, ninja_orientation)
        self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)

        # Place mines in chamber
        mines_placed = self.place_mines_in_chamber(
            chamber_x1,
            chamber_y1,
            chamber_x2,
            chamber_y2,
            ninja_x,
            ninja_y,
            door_x,
            door_y,
            switch_x,
            switch_y,
        )

        # Ensure at least 1 mine was placed
        # If no mines were placed, force place one in a valid location
        if mines_placed == 0:
            # Find a valid location for a mine (not near player, not on switch/door)
            valid_positions = []
            for x in range(chamber_x1 + 1, chamber_x2):
                for y in range(chamber_y1 + 1, chamber_y2):
                    # Check distance from player
                    if abs(x - ninja_x) < self.MIN_PLAYER_MINE_DISTANCE_TILES:
                        continue
                    if abs(y - ninja_y) < self.MIN_PLAYER_MINE_DISTANCE_TILES:
                        continue
                    # Check not on switch or door
                    if (x, y) == (switch_x, switch_y) or (x, y) == (door_x, door_y):
                        continue
                    valid_positions.append((x, y))

            if valid_positions:
                mine_x, mine_y = self.rng.choice(valid_positions)
                self.add_entity(1, mine_x, mine_y, 0, 1)

        # Add random entities outside the playspace
        playspace = (chamber_x1 - 4, chamber_y1 - 4, chamber_x2 + 4, chamber_y2 + 4)
        self.add_random_entities_outside_playspace(
            playspace[0], playspace[1], playspace[2], playspace[3]
        )

        return self
