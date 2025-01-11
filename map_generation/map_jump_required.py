"""Map generation for a level with a jump required.

The level's geometry is composed of a single chamber,
with a pit in the center of the chamber with mines at the bottom.

The switch is located somewhere between the top of the pit and the top of the chamber.

The chamber is surrounded by walls, and the ninja must jump over the pit to reach the switch.

The ninja starts at either the left or right side of the chamber, with the exit door on the opposite side.

The background of the map should either be:
- Empty tiles
- Solid tiles
- Random tiles

Additionally, the platforms should have mines randomly placed on the ground level, with a maximum of 5 mines per platform.
The mines should not be directly below the ninja's starting position, or the exit door.
"""

from map_generation.map import Map
from typing import Optional, Tuple
from map_generation.constants import VALID_TILE_TYPES


class MapJumpRequired(Map):
    """Generates N++ levels that require a jump to complete."""

    # Chamber size constants
    MIN_WIDTH = 16  # Minimum width to allow for pit and platforms
    MAX_WIDTH = 40
    MIN_HEIGHT = 8  # Minimum height to allow for pit and jump
    MAX_HEIGHT = 16
    MIN_PIT_WIDTH = 3  # Minimum width of the pit
    MAX_PIT_WIDTH = 5

    MAX_MINES_PER_PLATFORM = 5

    def set_empty_rectangle(self, x1: int, y1: int, x2: int, y2: int):
        """Set a rectangular region of the map to empty space."""
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                self.set_tile(x, y, 0)

    def create_pit_with_mines(self, chamber_x1: int, chamber_y1: int, chamber_x2: int, chamber_y2: int) -> Tuple[int, int, int]:
        """Create a pit in the center of the chamber with mines at the bottom.
        Also creates elevated platforms on both sides of the pit.
        Returns the pit's x1, x2 coordinates and the pit's top y coordinate."""
        chamber_width = chamber_x2 - chamber_x1
        pit_width = self.rng.randint(self.MIN_PIT_WIDTH, min(
            self.MAX_PIT_WIDTH, chamber_width - 8))  # Leave more space for platforms

        # Center the pit horizontally
        pit_x1 = chamber_x1 + (chamber_width - pit_width) // 2
        pit_x2 = pit_x1 + pit_width

        # Pit should extend from bottom of chamber up about 2/3 of chamber height
        pit_height = (chamber_y2 - chamber_y1) * 2 // 3
        pit_top_y = chamber_y2 - pit_height

        # Create the pit (empty space)
        self.set_empty_rectangle(pit_x1, pit_top_y, pit_x2, chamber_y2)

        # Add mines at the bottom of the pit (at chamber ground level)
        for x in range(pit_x1, pit_x2 + 2):
            self.add_entity(1, x + 1, chamber_y2 + 2, 0,
                            1)  # Type 1 = toggle mine

        # Create elevated platforms on both sides
        # Left platform
        for y in range(pit_top_y + 1, chamber_y2 + 1):
            for x in range(chamber_x1, pit_x1):
                self.set_tile(x, y, 1)  # Fill with solid tiles

        # Right platform
        for y in range(pit_top_y + 1, chamber_y2 + 1):
            for x in range(pit_x2 + 1, chamber_x2 + 1):
                self.set_tile(x, y, 1)  # Fill with solid tiles

        return pit_x1, pit_x2, pit_top_y

    def place_mines_on_platforms(self, pit_x1: int, pit_x2: int, pit_top_y: int, chamber_x1: int, chamber_x2: int, chamber_y2: int, ninja_x: int, door_x: int):
        """Place mines on the platforms.

        Args:
            pit_x1: Left edge of pit
            pit_x2: Right edge of pit
            pit_top_y: Top edge of pit
            chamber_x1: Left edge of chamber
            chamber_x2: Right edge of chamber
            chamber_y2: Bottom edge of chamber
            ninja_x: X position of ninja spawn
            door_x: X position of exit door
        """
        # Left platform potential mine positions (exclude ninja spawn and door positions)
        left_platform_positions = []
        for x in range(chamber_x1, pit_x1):
            # Check if position is not within 2 tiles of ninja or door
            if not (ninja_x - 2 <= x + 1 <= ninja_x + 2) and not (door_x - 2 <= x + 1 <= door_x + 2):
                left_platform_positions.append(x)

        # Right platform potential mine positions (exclude ninja spawn and door positions)
        right_platform_positions = []
        for x in range(pit_x2 + 1, chamber_x2 + 1):
            # Check if position is not within 2 tiles of ninja or door
            if not (ninja_x - 2 <= x + 1 <= ninja_x + 2) and not (door_x - 2 <= x + 1 <= door_x + 2):
                right_platform_positions.append(x)

        # Randomly select number of mines for each platform
        left_mine_count = self.rng.randint(
            0, min(self.MAX_MINES_PER_PLATFORM, len(left_platform_positions)))
        right_mine_count = self.rng.randint(
            0, min(self.MAX_MINES_PER_PLATFORM, len(right_platform_positions)))

        # Place mines on left platform
        if left_mine_count > 0 and left_platform_positions:
            mine_positions = self.rng.sample(
                left_platform_positions, left_mine_count)
            for x in mine_positions:
                # Type 1 = toggle mine
                self.add_entity(1, x + 1, pit_top_y + 2, 0, 1)

        # Place mines on right platform
        if right_mine_count > 0 and right_platform_positions:
            mine_positions = self.rng.sample(
                right_platform_positions, right_mine_count)
            for x in mine_positions:
                # Type 1 = toggle mine
                self.add_entity(1, x + 1, pit_top_y + 2, 0, 1)

    def generate(self, seed: Optional[int] = None) -> Map:
        """Generate a level that requires a jump to complete.

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

        # Calculate maximum possible starting positions
        max_start_x = self.MAP_WIDTH - width - 1
        max_start_y = self.MAP_HEIGHT - height - 1

        # Randomize the starting position of the chamber
        chamber_x1 = self.rng.randint(2, max(3, max_start_x))
        chamber_y1 = self.rng.randint(2, max(3, max_start_y))
        chamber_x2 = min(chamber_x1 + width, self.MAP_WIDTH - 2)
        chamber_y2 = min(chamber_y1 + height, self.MAP_HEIGHT - 2)

        # Pre-generate all random tiles at once
        # Choose if tiles will be random, solid, or empty for the border
        choice = self.rng.randint(0, 2)
        if choice == 0:
            tile_types = [self.rng.randint(0, VALID_TILE_TYPES) for _ in range(
                self.MAP_WIDTH * self.MAP_HEIGHT)]
        elif choice == 1:
            tile_types = [1] * (self.MAP_WIDTH *
                                self.MAP_HEIGHT)  # Solid walls
        else:
            tile_types = [0] * (self.MAP_WIDTH *
                                self.MAP_HEIGHT)  # Empty tiles
        self.set_tiles_bulk(tile_types)

        # Create the empty chamber
        self.set_empty_rectangle(
            chamber_x1, chamber_y1, chamber_x2, chamber_y2)

        # Create boundary walls
        for x in range(chamber_x1, chamber_x2 + 1):
            self.set_tile(x, chamber_y2 + 1, 1)  # Bottom wall
            self.set_tile(x, chamber_y1 - 1, 1)  # Top wall

        for y in range(chamber_y1, chamber_y2 + 1):
            self.set_tile(chamber_x1 - 1, y, 1)  # Left wall
            self.set_tile(chamber_x2 + 1, y, 1)  # Right wall

        # Create pit with mines
        pit_x1, pit_x2, pit_top_y = self.create_pit_with_mines(
            chamber_x1, chamber_y1, chamber_x2, chamber_y2)

        # Randomly choose which side the ninja starts on
        ninja_on_left = self.rng.choice([True, False])

        # Place ninja
        if ninja_on_left:
            ninja_x = chamber_x1 + 1
            ninja_orientation = 1  # Face right
            door_x = chamber_x2 - 1
        else:
            ninja_x = chamber_x2 - 1
            ninja_orientation = -1  # Face left
            door_x = chamber_x1 + 1

        # Place ninja and door on the elevated platforms
        ninja_y = pit_top_y  # Place on elevated platform level
        door_y = pit_top_y  # Place on elevated platform level

        # Place switch somewhere between pit top and chamber top (-3 so its not too high)
        switch_x = self.rng.randint(pit_x1, pit_x2)
        switch_y = self.rng.randint(pit_top_y - 3, pit_top_y - 1)

        # Place mines on the platforms
        self.place_mines_on_platforms(
            pit_x1, pit_x2, pit_top_y, chamber_x1, chamber_x2, chamber_y2, ninja_x, door_x)

        # Convert to screen coordinates and place entities
        self.set_ninja_spawn(ninja_x, ninja_y, ninja_orientation)
        self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)

        # Add random entities outside the playspace
        # Our playspace is inside the chamber, so we need to add entities outside the chamber
        playspace = (chamber_x1 - 4, chamber_y1 - 4,
                     chamber_x2 + 4, chamber_y2 + 4)
        self.add_random_entities_outside_playspace(
            playspace[0], playspace[1], playspace[2], playspace[3])

        return self
