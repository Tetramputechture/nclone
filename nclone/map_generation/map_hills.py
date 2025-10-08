"""Hills map generation module for N++ levels.

Generates levels with procedurally generated rolling hills using slope tiles.
The terrain is created using a combination of mild slopes, steep slopes, and 45-degree tiles.
"""

from .map import Map
from typing import Optional
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
import math


class MapHills(Map):
    """Generates N++ levels with rolling hills terrain using slope tiles."""

    # Chamber dimension constraints
    MIN_WIDTH = 10
    MAX_WIDTH = MAP_TILE_WIDTH - 4
    MIN_HEIGHT = 8
    MAX_HEIGHT = MAP_TILE_HEIGHT - 4

    # Terrain generation parameters
    MIN_HILLS = 2
    MAX_HILLS = 8
    MIN_HILL_WIDTH = 2
    MAX_HILL_WIDTH = 8
    MIN_HEIGHT_CHANGE = 1
    MAX_HEIGHT_CHANGE = 4

    # Tile types for different slopes
    # 45-degree slopes
    SLOPE_45_UP = [8, 7]  # ascending 45-degree slopes
    SLOPE_45_DOWN = [6, 9]  # descending 45-degree slopes

    # Mild slopes (gentle)
    MILD_SLOPE_UP = [18, 19]  # short mild slopes ascending
    MILD_SLOPE_DOWN = [20, 21]  # short mild slopes descending
    MILD_RAISED_UP = [22, 23]  # raised mild slopes
    MILD_RAISED_DOWN = [24, 25]  # raised mild slopes descending

    # Steep slopes
    STEEP_SLOPE_UP = [26, 27]  # short steep slopes ascending
    STEEP_SLOPE_DOWN = [28, 29]  # short steep slopes descending
    STEEP_RAISED_UP = [30, 31]  # raised steep slopes
    STEEP_RAISED_DOWN = [32, 33]  # raised steep slopes descending

    def generate(self, seed: Optional[int] = None) -> Map:
        """Generate a hills level with procedurally generated terrain.

        Args:
            seed: Random seed for reproducible generation

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
        self.set_hollow_rectangle(
            chamber_x - 1,
            chamber_y - 1,
            chamber_x + width,
            chamber_y + height,
            use_random_tiles_type=use_random_tiles,
        )

        # Step 5: Generate rolling hills terrain
        floor_y = chamber_y + height - 1
        terrain_heights = self._generate_terrain_heights(
            chamber_x, width, floor_y, height
        )

        # Apply terrain heights to the map
        self._apply_terrain_to_map(chamber_x, width, terrain_heights, floor_y)

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

        # Place entities one tile above ground
        ninja_y = terrain_heights[ninja_x - chamber_x] - 1
        switch_y = terrain_heights[switch_x - chamber_x] - 1
        door_y = terrain_heights[door_x - chamber_x] - 1

        self.set_ninja_spawn(ninja_x, ninja_y, ninja_orientation)
        self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)

        # Add random entities outside playspace
        self.add_random_entities_outside_playspace(
            chamber_x - 4, chamber_y - 4, chamber_x + width + 4, chamber_y + height + 4
        )

        return self

    def _generate_terrain_heights(
        self, start_x: int, width: int, floor_y: int, chamber_height: int
    ) -> list:
        """Generate height map for rolling hills terrain.

        Uses a combination of sine waves and random perturbations to create
        natural-looking rolling hills.

        Args:
            start_x: Starting x coordinate of the chamber
            width: Width of the chamber
            floor_y: Y coordinate of the floor
            chamber_height: Total height of the chamber

        Returns:
            List of y-coordinates for each x position (ground level)
        """
        heights = []
        max_height_variance = min(chamber_height - 3, self.MAX_HEIGHT_CHANGE)

        # Generate number of hills
        num_hills = self.rng.randint(self.MIN_HILLS, self.MAX_HILLS)

        # Use sine wave for base terrain
        frequency = (num_hills * 2 * math.pi) / width
        amplitude = max_height_variance

        for i in range(width):
            # Base sine wave
            base_height = math.sin(i * frequency) * amplitude

            # Add random perturbation
            noise = self.rng.uniform(-0.5, 0.5)

            # Calculate final height
            height_offset = int(base_height + noise)
            # Clamp to valid range
            height_offset = max(
                -max_height_variance, min(max_height_variance, height_offset)
            )

            # Ground y position (higher y = lower on screen in tile coords)
            ground_y = floor_y + height_offset
            ground_y = max(floor_y - max_height_variance, min(floor_y, ground_y))

            heights.append(ground_y)

        # Smooth out extreme changes
        heights = self._smooth_terrain(heights, floor_y)

        return heights

    def _smooth_terrain(self, heights: list, floor_y: int) -> list:
        """Smooth terrain to prevent impossible jumps.

        Args:
            heights: List of ground heights
            floor_y: Base floor y coordinate

        Returns:
            Smoothed list of heights
        """
        smoothed = heights.copy()
        max_change = 2  # Maximum height change between adjacent tiles

        for i in range(1, len(smoothed)):
            height_diff = abs(smoothed[i] - smoothed[i - 1])
            if height_diff > max_change:
                # Interpolate to smooth the transition
                if smoothed[i] > smoothed[i - 1]:
                    smoothed[i] = smoothed[i - 1] + max_change
                else:
                    smoothed[i] = smoothed[i - 1] - max_change

        return smoothed

    def _apply_terrain_to_map(
        self, start_x: int, width: int, heights: list, floor_y: int
    ) -> None:
        """Apply terrain heights to the map using appropriate slope tiles.

        Args:
            start_x: Starting x coordinate of the chamber
            width: Width of the chamber
            heights: List of ground heights for each x position
            floor_y: Base floor y coordinate
        """
        for i in range(width):
            x = start_x + i
            ground_y = heights[i]

            # Fill in solid tiles below ground level
            for y in range(ground_y + 1, floor_y + 2):
                self.set_tile(x, y, 1)

            # Determine slope tile to use based on height change
            if i < width - 1:
                current_height = heights[i]
                next_height = heights[i + 1]
                height_diff = next_height - current_height

                slope_tile = self._get_slope_tile(height_diff)

                if slope_tile is not None:
                    # Place slope tile at the ground level
                    self.set_tile(x, ground_y, slope_tile)
                else:
                    # Flat ground - use full tile
                    self.set_tile(x, ground_y, 1)
            else:
                # Last tile - use full tile
                self.set_tile(x, ground_y, 1)

    def _get_slope_tile(self, height_diff: int) -> Optional[int]:
        """Get appropriate slope tile based on height difference.

        Args:
            height_diff: Height difference to next tile (negative = ascending, positive = descending)

        Returns:
            Tile type ID for the appropriate slope, or None for flat ground
        """
        if height_diff == 0:
            # Flat ground
            return None
        elif height_diff < 0:
            # Ascending (going up)
            if abs(height_diff) >= 2:
                # Steep slope
                return self.rng.choice(self.STEEP_SLOPE_UP)
            else:
                # Mild slope
                return self.rng.choice(self.MILD_SLOPE_UP)
        else:
            # Descending (going down)
            if abs(height_diff) >= 2:
                # Steep slope
                return self.rng.choice(self.STEEP_SLOPE_DOWN)
            else:
                # Mild slope
                return self.rng.choice(self.MILD_SLOPE_DOWN)
