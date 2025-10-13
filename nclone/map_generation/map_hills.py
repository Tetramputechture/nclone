"""Hills map generation module for N++ levels.

Generates levels with procedurally generated rolling hills using slope tiles.
The terrain is created using a combination of mild slopes, steep slopes, and 45-degree tiles.
"""

from .map import Map
from typing import Optional
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class MapHills(Map):
    """Generates N++ levels with rolling hills terrain using slope tiles."""

    # Chamber dimension constraints
    MIN_WIDTH = 10
    MAX_WIDTH = MAP_TILE_WIDTH - 4
    MIN_HEIGHT = 8
    MAX_HEIGHT = MAP_TILE_HEIGHT - 4

    # Terrain generation parameters
    MIN_HILLS = 1
    MAX_HILLS = 8
    MIN_HILL_WIDTH = 2
    MAX_HILL_WIDTH = 8
    MIN_HEIGHT_CHANGE = 1
    MAX_HEIGHT_CHANGE = 10

    def generate(self, seed: Optional[int] = None) -> Map:
        """Generate a hills level with procedurally generated terrain.

        Args:
            seed: Random seed for reproducible generation.

        Returns:
            Map: A Map instance with the generated level
        """
        if seed is not None:
            self.rng.seed(seed)

        self.reset()

        # Step 1: Fill map with either solid or empty tiles
        fill_solid = self.rng.choice([True, False])
        fill_tile = 1 if fill_solid else 0
        for y in range(MAP_TILE_HEIGHT):
            for x in range(MAP_TILE_WIDTH):
                self.set_tile(x, y, fill_tile)

        # Step 2: Determine chamber dimensions (at least 5x5)
        width = self.rng.randint(max(self.MIN_WIDTH, 5), self.MAX_WIDTH)
        height = self.rng.randint(max(self.MIN_HEIGHT, 5), self.MAX_HEIGHT)

        # Calculate chamber position (centered with some randomness)
        max_start_x = MAP_TILE_WIDTH - width - 2
        max_start_y = MAP_TILE_HEIGHT - height - 2
        chamber_x = self.rng.randint(2, max(2, max_start_x))
        chamber_y = self.rng.randint(2, max(2, max_start_y))

        # Step 3: Carve out empty chamber
        for y in range(chamber_y, chamber_y + height):
            for x in range(chamber_x, chamber_x + width):
                self.set_tile(x, y, 0)

        # Step 4: Create boundary walls
        use_random_tiles = self.rng.choice([True, False])

        # Step 5: Generate rolling hills terrain
        floor_y = chamber_y + height

        # Use new hill pattern approach for more natural terrain
        self._apply_hill_patterns(chamber_x - 1, width, floor_y, chamber_y)

        # Step 6: Place entities
        # Ninja at one end, exit switch in middle, exit door at other end
        ninja_on_left = self.rng.choice([True, False])

        if ninja_on_left:
            ninja_x = chamber_x + 1
            switch_x = chamber_x + width // 2
            door_x = chamber_x + width - 2
            ninja_orientation = 1  # Facing right
        else:
            ninja_x = chamber_x + width - 2
            switch_x = chamber_x + width // 2
            door_x = chamber_x + 1
            ninja_orientation = -1  # Facing left

        # Find ground level at entity positions
        ninja_y = self._find_ground_level(ninja_x, floor_y, chamber_y) - 1
        switch_y = self._find_ground_level(switch_x, floor_y, chamber_y) - 1

        # move switch_y up a random number of tiles, respecting the ceiling, with a max of 4 tiles up
        switch_y = switch_y - self.rng.randint(0, min(4, switch_y - chamber_y))

        door_y = self._find_ground_level(door_x, floor_y, chamber_y) - 1

        self.set_hollow_rectangle(
            chamber_x - 1,
            chamber_y - 1,
            chamber_x + width,
            chamber_y + height,
            use_random_tiles_type=use_random_tiles,
        )

        self.set_ninja_spawn(ninja_x, ninja_y, ninja_orientation)
        self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)

        return self

    def _apply_hill_patterns(
        self, start_x: int, width: int, floor_y: int, chamber_y: int
    ) -> None:
        """Apply terrain using the new hill pattern functions.

        This creates more natural-looking hills by using complete hill patterns
        (mild, steep, 45-degree, mixed) rather than individual slope tiles.

        Args:
            start_x: Starting x coordinate of the chamber
            width: Width of the chamber
            floor_y: Y coordinate of the floor
            chamber_y: Y coordinate of the chamber ceiling (top of playable area)
        """
        # Calculate chamber boundaries
        chamber_left = start_x
        chamber_right = start_x + width + 1
        chamber_ceiling = chamber_y

        # Fill the floor with solid tiles first
        for x in range(start_x, start_x + width):
            for y in range(floor_y + 1, floor_y + 3):
                self.set_tile(x, y, 1)

        # Calculate how many hills to create based on width
        num_hills = self.rng.randint(self.MIN_HILLS, max(self.MAX_HILLS, width // 6))

        # Determine max height based on chamber size
        chamber_height = floor_y - chamber_y
        max_hill_height = min(self.MAX_HEIGHT_CHANGE, chamber_height - 1)
        hill_types_to_use = ["mild", "steep", "45", "mixed"]

        # Create rolling hills across the chamber
        current_x = start_x
        hills_created = 0

        while current_x < start_x + width and hills_created < num_hills:
            # Randomly choose hill parameters
            hill_height = self.rng.randint(1, max_hill_height)
            hill_type = self.rng.choice(hill_types_to_use)
            # hill_type = "steep"

            # Add some flat ground occasionally
            if hills_created > 0 and self.rng.random() < 0.3:
                flat_width = self.rng.randint(1, 3)
                for _ in range(flat_width):
                    if current_x >= start_x + width:
                        break
                    self.set_tile(current_x, floor_y, 1)
                    current_x += 1

            # Create the hill based on type, with boundary constraints
            if hill_type == "mild":
                current_x = self.create_mild_hill(
                    current_x,
                    floor_y,
                    hill_height,
                    min_x=chamber_left,
                    max_x=chamber_right,
                    min_y=chamber_ceiling,
                    max_y=floor_y,
                )
            elif hill_type == "steep":
                current_x = self.create_steep_hill(
                    current_x,
                    floor_y,
                    hill_height,
                    min_x=chamber_left,
                    max_x=chamber_right,
                    min_y=chamber_ceiling,
                    max_y=floor_y,
                )
            elif hill_type == "45":
                current_x = self.create_45_degree_hill(
                    current_x,
                    floor_y,
                    hill_height,
                    min_x=chamber_left,
                    max_x=chamber_right,
                    min_y=chamber_ceiling,
                    max_y=floor_y,
                )
            elif hill_type == "mixed":
                ascent_type = self.rng.choice(["mild", "steep", "45"])
                descent_type = self.rng.choice(["mild", "steep", "45"])
                current_x = self.create_mixed_hill(
                    current_x,
                    floor_y,
                    hill_height,
                    ascent_type,
                    descent_type,
                    min_x=chamber_left,
                    max_x=chamber_right,
                    min_y=chamber_ceiling,
                    max_y=floor_y,
                )

            hills_created += 1

        # Fill remaining space with flat ground
        while current_x < start_x + width:
            self.set_tile(current_x, floor_y, 1)
            current_x += 1

    def _find_ground_level(self, x: int, floor_y: int, ceiling_y: int) -> int:
        """Find the ground level (first solid tile from top) at position x.

        Args:
            x: X coordinate to check
            floor_y: Bottom of the chamber
            ceiling_y: Top of the chamber

        Returns:
            Y coordinate of the ground surface (empty tile just above solid)
        """
        # Clamp x to valid range to prevent out-of-bounds access
        x = max(0, min(x, MAP_TILE_WIDTH - 1))

        for y in range(ceiling_y, min(floor_y + 1, MAP_TILE_HEIGHT)):
            tile = self.tile_data[x + y * MAP_TILE_WIDTH]
            # Check if this is a solid or slope tile (non-zero)
            if tile != 0:
                return y

        # If no solid tile found, return floor level
        return min(floor_y, MAP_TILE_HEIGHT - 1)
