"""Island-style map generation module for N++ levels.

Creates maps with 4x4 groups of tiles spread across an empty chamber,
resembling islands floating in space.
"""

import math
from typing import Optional, List, Tuple
from .map import Map
from .constants import VALID_TILE_TYPES
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class MapIslands(Map):
    """Generates island-style N++ levels with 4x4 tile groups spread across empty space."""

    # Chamber dimension constraints
    MIN_WIDTH = 36
    MAX_WIDTH = MAP_TILE_WIDTH - 4  # Leave room for borders
    MIN_HEIGHT = 12
    MAX_HEIGHT = MAP_TILE_HEIGHT - 4  # Leave room for borders

    # Island spacing constraints
    MIN_ISLAND_SPACING = 1  # Minimum empty tiles between islands
    MAX_ISLAND_SPACING = 1  # Maximum empty tiles between islands
    MIN_ISLAND_SIZE = 1  # Minimum island dimension
    MAX_ISLAND_SIZE = 4  # Maximum island dimension
    BORDER_DISTANCE = 1  # Minimum distance from playspace borders

    def generate(self, seed: Optional[int] = None) -> Map:
        """Generate an island-style level with variable-sized tile groups (1x1 to 4x4).

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

        # Calculate chamber position (centered with some randomness)
        max_start_x = MAP_TILE_WIDTH - width - 2
        max_start_y = MAP_TILE_HEIGHT - height - 2
        chamber_x1 = self.rng.randint(2, max(2, max_start_x))
        chamber_y1 = self.rng.randint(2, max(2, max_start_y))
        chamber_x2 = chamber_x1 + width - 1
        chamber_y2 = chamber_y1 + height - 1

        # Fill everything with walls first (start with solid background)
        should_fill_walls = self.rng.choice([True, False])
        if should_fill_walls:
            tile_types = [1] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
        else:
            tile_types = [
                self.rng.randint(0, VALID_TILE_TYPES)
                for _ in range(MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
            ]
        self.set_tiles_bulk(tile_types)

        # Carve out empty chamber
        self.set_empty_rectangle(chamber_x1, chamber_y1, chamber_x2, chamber_y2)

        # Add boundary walls around chamber
        use_random_border = self.rng.choice([True, False])
        self.set_hollow_rectangle(
            chamber_x1 - 1,
            chamber_y1 - 1,
            chamber_x2 + 1,
            chamber_y2 + 1,
            use_random_tiles_type=use_random_border,
        )

        # Calculate valid area for island placement (respecting border distance)
        # Use MAX_ISLAND_SIZE to ensure largest islands can fit
        island_area_x1 = chamber_x1 + self.BORDER_DISTANCE
        island_area_y1 = chamber_y1 + self.BORDER_DISTANCE
        island_area_x2 = chamber_x2 - self.BORDER_DISTANCE - self.MAX_ISLAND_SIZE + 4
        island_area_y2 = chamber_y2 - self.BORDER_DISTANCE - self.MAX_ISLAND_SIZE + 4

        # Generate islands with spacing constraints
        islands = self._generate_islands(
            island_area_x1, island_area_y1, island_area_x2, island_area_y2
        )

        # Place the islands
        for island_x, island_y, island_width, island_height in islands:
            self._place_island(island_x, island_y, island_width, island_height)

        # Find island top positions for entity placement
        island_positions = self._get_island_top_positions(
            chamber_x1, chamber_y1, chamber_x2, chamber_y2
        )

        # Place ninja, exit switch, and exit door on islands with minimum distance constraint
        if len(island_positions) < 3:
            # Fallback: if not enough islands, place entities at bottom of chamber
            ninja_x, ninja_y = chamber_x1 + 1, chamber_y2 - 1
            switch_x, switch_y = chamber_x1 + width // 3, chamber_y2 - 1
            door_x, door_y = chamber_x1 + (2 * width) // 3, chamber_y2 - 1
        else:
            ninja_x, ninja_y, switch_x, switch_y, door_x, door_y = (
                self._place_entities_with_distance(island_positions, min_distance=3)
            )

        # Adjust entity positions to account for map padding (subtract 1 tile from x and y)
        # This prevents entities from spawning inside tiles
        ninja_y -= 1
        switch_y -= 1
        door_y -= 1

        # Set ninja spawn with random orientation
        ninja_orientation = self.rng.choice([1, -1])
        self.set_ninja_spawn(ninja_x, ninja_y, ninja_orientation)

        # Add exit door with its switch
        self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)

        return self

    def _generate_islands(
        self, area_x1: int, area_y1: int, area_x2: int, area_y2: int
    ) -> List[Tuple[int, int, int, int]]:
        """Generate island positions and sizes with spacing constraints.

        Uses a grid-based approach to ensure spacing constraints are met.
        Each island has a random size between 1x1 and 4x4 with varying width and height.

        Args:
            area_x1, area_y1: Top-left corner of valid placement area
            area_x2, area_y2: Bottom-right corner of valid placement area

        Returns:
            List of (x, y, width, height) tuples representing islands
        """
        islands = []

        # Calculate grid spacing (max island size + spacing between islands)
        spacing = self.rng.randint(self.MIN_ISLAND_SPACING, self.MAX_ISLAND_SPACING)
        grid_step = self.MAX_ISLAND_SIZE + spacing

        # Calculate how many islands can fit
        available_width = area_x2 - area_x1
        available_height = area_y2 - area_y1

        if available_width <= 0 or available_height <= 0:
            return islands

        # Generate islands on a grid with some randomness
        x = area_x1
        while x <= area_x2:
            y = area_y1
            while y <= area_y2:
                # Generate random island size
                island_width = self.rng.randint(
                    self.MIN_ISLAND_SIZE, self.MAX_ISLAND_SIZE
                )
                island_height = self.rng.randint(
                    self.MIN_ISLAND_SIZE, self.MAX_ISLAND_SIZE
                )

                # Add some randomness to position (but maintain minimum spacing)
                max_offset_x = min(spacing, area_x2 - x - island_width + 1)
                max_offset_y = min(spacing, area_y2 - y - island_height + 1)

                offset_x = self.rng.randint(0, max(0, max_offset_x))
                offset_y = self.rng.randint(0, max(0, max_offset_y))

                island_x = x + offset_x
                island_y = y + offset_y

                # Check if island fits within bounds
                if (
                    island_x + island_width - 1 <= area_x2
                    and island_y + island_height - 1 <= area_y2
                ):
                    islands.append((island_x, island_y, island_width, island_height))

                y += grid_step
            x += grid_step

        return islands

    def _place_island(self, x: int, y: int, width: int, height: int) -> None:
        """Place an island of random tiles at the specified position.

        Args:
            x, y: Top-left corner of the island
            width: Width of the island in tiles
            height: Height of the island in tiles
        """
        for dy in range(height):
            for dx in range(width):
                tile_x = x + dx
                tile_y = y + dy
                # Use tiles 1 to VALID_TILE_TYPES (excluding 0 which is empty)
                random_tile = self.rng.randint(1, VALID_TILE_TYPES)
                self.set_tile(tile_x, tile_y, random_tile)

    def _get_island_top_positions(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> List[Tuple[int, int]]:
        """Get all positions on top of islands within the specified area.

        Finds tiles that are solid (non-empty) and have empty space above them,
        suitable for placing entities.

        Args:
            x1, y1: Top-left corner of area
            x2, y2: Bottom-right corner of area

        Returns:
            List of (x, y) tuples representing positions on top of islands
        """
        island_top_positions = []
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if 0 <= x < MAP_TILE_WIDTH and 0 <= y < MAP_TILE_HEIGHT:
                    tile_idx = x + y * MAP_TILE_WIDTH

                    # Check if this tile is solid (part of an island)
                    if tile_idx < len(self.tile_data) and self.tile_data[tile_idx] != 0:
                        # Check if the tile above is empty (or we're at the top)
                        if y == y1:
                            # Top row - always valid for entity placement
                            island_top_positions.append((x, y))
                        else:
                            above_idx = x + (y - 1) * MAP_TILE_WIDTH
                            if above_idx >= 0 and above_idx < len(self.tile_data):
                                if self.tile_data[above_idx] == 0:
                                    # This is a top surface of an island
                                    island_top_positions.append((x, y))

        return island_top_positions

    def _place_entities_with_distance(
        self, positions: List[Tuple[int, int]], min_distance: int = 3
    ) -> Tuple[int, int, int, int, int, int]:
        """Place three entities (ninja, switch, door) with minimum distance between them.

        Args:
            positions: List of available positions
            min_distance: Minimum diagonal distance between entities

        Returns:
            Tuple of (ninja_x, ninja_y, switch_x, switch_y, door_x, door_y)
        """
        max_attempts = 1000

        for _ in range(max_attempts):
            # Randomly select three positions
            if len(positions) < 3:
                break

            selected = self.rng.sample(positions, 3)
            ninja_pos, switch_pos, door_pos = selected

            # Calculate diagonal distances
            dist_ninja_switch = math.sqrt(
                (ninja_pos[0] - switch_pos[0]) ** 2
                + (ninja_pos[1] - switch_pos[1]) ** 2
            )
            dist_ninja_door = math.sqrt(
                (ninja_pos[0] - door_pos[0]) ** 2 + (ninja_pos[1] - door_pos[1]) ** 2
            )
            dist_switch_door = math.sqrt(
                (switch_pos[0] - door_pos[0]) ** 2 + (switch_pos[1] - door_pos[1]) ** 2
            )

            # Check if all distances meet minimum requirement
            if (
                dist_ninja_switch >= min_distance
                and dist_ninja_door >= min_distance
                and dist_switch_door >= min_distance
            ):
                return (
                    ninja_pos[0],
                    ninja_pos[1],
                    switch_pos[0],
                    switch_pos[1],
                    door_pos[0],
                    door_pos[1],
                )

        # Fallback: use the last selected positions if we couldn't meet constraints
        if len(positions) >= 3:
            selected = self.rng.sample(positions, 3)
            ninja_pos, switch_pos, door_pos = selected
            return (
                ninja_pos[0],
                ninja_pos[1],
                switch_pos[0],
                switch_pos[1],
                door_pos[0],
                door_pos[1],
            )

        # Ultimate fallback: use first available positions
        ninja_pos = positions[0] if len(positions) > 0 else (1, 1)
        switch_pos = positions[1] if len(positions) > 1 else (3, 1)
        door_pos = positions[2] if len(positions) > 2 else (5, 1)

        return (
            ninja_pos[0],
            ninja_pos[1],
            switch_pos[0],
            switch_pos[1],
            door_pos[0],
            door_pos[1],
        )
