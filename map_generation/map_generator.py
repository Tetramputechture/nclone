"""Map generator module for procedural level generation."""

from map import Map
import random
import sys
import os
from typing import List, Tuple, Optional, Dict, Set, Literal

# Temporarily add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MapGenerator(Map):
    """Generates procedural N++ levels including mazes and simple horizontal levels."""

    # Maze generation constants
    MIN_WIDTH = 4
    MAX_WIDTH = 42
    MIN_HEIGHT = 4
    MAX_HEIGHT = 23

    # Simple horizontal level constants
    SIMPLE_MAP_MIN_WIDTH = 4
    SIMPLE_MAP_MAX_WIDTH = 6
    SIMPLE_MAP_MIN_HEIGHT = 1
    SIMPLE_MAP_MAX_HEIGHT = 3
    GLOBAL_MAX_UP_DEVIATION = 5
    GLOBAL_MAX_DOWN_DEVIATION = 0

    def __init__(self, width: int = 10, height: int = 10, seed: Optional[int] = None):
        """Initialize the map generator.

        Args:
            width: Width of the map (4-42)
            height: Height of the map (4-23)
            seed: Random seed for reproducible generation
        """
        super().__init__()
        self.width = max(self.MIN_WIDTH, min(width, self.MAX_WIDTH))
        self.height = max(self.MIN_HEIGHT, min(height, self.MAX_HEIGHT))
        self.rng = random.Random(seed)

        # Initialize tracking variables
        self.visited: Set[Tuple[int, int]] = set()
        self.grid: List[List[int]] = []

        # Initialize the map with solid walls
        self._init_solid_map()

    def _init_solid_map(self):
        """Initialize both the grid and map tiles with solid walls."""
        # Initialize the grid for maze algorithm logic
        self.grid = [[1] * self.width for _ in range(self.height)]

        # Fill the map with solid walls
        self._fill_with_walls()

    def _fill_with_walls(self):
        """Fill the entire map with solid walls."""
        for y in range(self.height):
            for x in range(self.width):
                self.set_tile(x, y, 1)  # 1 = wall

    def _is_valid_cell(self, x: int, y: int) -> bool:
        """Check if coordinates are within maze bounds."""
        return 0 <= x < self.width and 0 <= y < self.height

    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells for maze generation."""
        neighbors = []
        for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)]:  # Check cells 2 steps away
            new_x, new_y = x + dx, y + dy
            if self._is_valid_cell(new_x, new_y) and (new_x, new_y) not in self.visited:
                neighbors.append((new_x, new_y))
        return neighbors

    def _carve_empty_space(self, x: int, y: int):
        """Carve an empty space at the given coordinates in both grid and map."""
        self.grid[y][x] = 0  # Update grid for maze logic
        self.set_tile(x, y, 0)  # Update actual map tiles (0 = empty space)

    def _carve_path(self, x: int, y: int):
        """Recursively carve paths using depth-first search.

        This method carves paths through solid walls, creating a maze pattern.
        It marks each visited cell and carves paths between cells that are two steps apart,
        ensuring walls remain between parallel paths.
        """
        # Mark current cell as visited and carve it
        self.visited.add((x, y))
        self._carve_empty_space(x, y)

        # Get valid neighbors and randomize their order
        neighbors = self._get_neighbors(x, y)
        self.rng.shuffle(neighbors)

        # For each unvisited neighbor, carve a path to it
        for next_x, next_y in neighbors:
            if (next_x, next_y) not in self.visited:
                # Carve the connecting path between current cell and next cell
                mid_x = (x + next_x) // 2
                mid_y = (y + next_y) // 2
                self._carve_empty_space(mid_x, mid_y)

                # Recursively continue from the next cell
                self._carve_path(next_x, next_y)

    def _place_ninja(self):
        """Place the ninja in a random valid starting position."""
        # Find valid empty cells in the first two columns
        valid_positions = []
        for y in range(self.height):
            for x in range(2):
                if self.grid[y][x + 1] == 0:  # Use grid for checking valid positions
                    valid_positions.append((x + 1, y))

        if valid_positions:
            grid_x, grid_y = self.rng.choice(valid_positions)
            # Convert grid coordinates to pixel coordinates (24 pixel grid)
            # Add 1 for map border offset and multiply by 4 for pixel coordinates
            self.set_ninja_spawn((grid_x) * 4 + 6, (grid_y) * 4 + 6)
        else:
            # If no valid positions in first two columns, find first empty cell
            for y in range(self.height):
                for x in range(self.width):
                    if self.grid[y][x] == 0:
                        # Convert grid coordinates to pixel coordinates
                        # Add 1 for map border offset and multiply by 4 for pixel coordinates
                        self.set_ninja_spawn((x) * 4 + 6, (y) * 4 + 6)
                        return

    def _place_exit(self):
        """Place the exit door and switch in valid positions."""
        # Place exit door on the right edge
        valid_door_positions = []
        for y in range(self.height):
            if self.grid[y][self.width-2] == 0:  # Check if cell next to edge is path
                valid_door_positions.append((self.width-2, y))

        if valid_door_positions:
            grid_x, grid_y = self.rng.choice(valid_door_positions)
            # Add offset of 6 like in simple horizontal
            door_x = (grid_x) * 4 + 6
            door_y = (grid_y) * 4 + 6

            # Find valid switch positions (empty cells not too close to ninja)
            valid_switch_positions = []
            min_distance = (self.width + self.height) // 4

            for y in range(self.height):
                for x in range(self.width):
                    if self.grid[y][x] == 0:
                        # Calculate distance using grid coordinates
                        ninja_x = self.ninja_spawn_x // 4 - 1
                        ninja_y = self.ninja_spawn_y // 4 - 1
                        if abs(x - ninja_x) + abs(y - ninja_y) >= min_distance:
                            valid_switch_positions.append((x, y))

            if valid_switch_positions:
                grid_x, grid_y = self.rng.choice(valid_switch_positions)
                # Add offset of 6 like in simple horizontal
                switch_x = (grid_x) * 4 + 6
                switch_y = (grid_y) * 4 + 6

                # Add exit door and switch using Map's add_entity method
                self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)

    def _place_gold(self, count: int = 5):
        """Place gold pieces in valid positions."""
        valid_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 0:
                    # Convert to pixel coordinates for comparison
                    pixel_x = (x + 1) * 4
                    pixel_y = (y + 1) * 4
                    # Don't place gold at ninja spawn or switch
                    if (pixel_x, pixel_y) != (self.ninja_spawn_x, self.ninja_spawn_y):
                        valid_positions.append((x, y))

        count = min(count, len(valid_positions))
        selected_positions = self.rng.sample(valid_positions, count)

        # Add gold entities using Map's add_entity method
        for x, y in selected_positions:
            self.add_entity(2, (x + 1) * 4 - 6, (y + 1) * 4 - 6, 0, 0)

    def generate_maze(self) -> Map:
        """Generate a complete maze with entities.

        This method follows these steps:
        1. Start with a completely solid level (all walls)
        2. Carve paths through the walls using a depth-first search algorithm
        3. Place the ninja spawn point near the left side
        4. Place the exit door and switch in valid positions
        5. Add gold pieces in valid positions

        Returns:
            Map: A Map instance with the generated maze and entities
        """
        self.reset()
        # Reset state and ensure we start with a solid level
        self.visited.clear()

        # Pre-generate all random tiles at once
        # Choose if tiles will be random or solid for the border
        if self.rng.choice([True, False]):
            tile_types = [self.rng.randint(0, 37) for _ in range(
                self.MAP_WIDTH * self.MAP_HEIGHT)]
        else:
            tile_types = [1] * (self.MAP_WIDTH * self.MAP_HEIGHT)
        self.set_tiles_bulk(tile_types)
        self._init_solid_map()

        # Start maze generation from a random position on the left side
        start_x = 0  # Start from leftmost column
        start_y = self.rng.randrange(0, self.height, 2)  # Random even row
        self._carve_path(start_x, start_y)

        # self.fill_boundary()

        # Place entities in the carved maze
        self._place_ninja()
        self._place_exit()
        self._place_gold()

        return self

    def fill_boundary(self):
        # Set our outer edge of rectangle with solid tiles
        for x in range(self.MAP_WIDTH):
            self.set_tile(x, 0, 1)
            self.set_tile(x, self.MAP_HEIGHT - 1, 1)
        for y in range(self.MAP_HEIGHT):
            self.set_tile(0, y, 1)
            self.set_tile(self.MAP_WIDTH - 1, y, 1)

    def generate_simple_horizontal(self, seed: Optional[int] = None) -> Map:
        """Generate a simple horizontal level with no backtracking required.

        Args:
            seed: Random seed for reproducible generation

        Returns:
            Map: A Map instance with the generated level
        """
        self.reset()
        if seed is not None:
            self.rng.seed(seed)

        # Generate random dimensions for play space
        width = self.rng.randint(
            self.SIMPLE_MAP_MIN_WIDTH, self.SIMPLE_MAP_MAX_WIDTH)
        height = self.rng.randint(
            self.SIMPLE_MAP_MIN_HEIGHT, self.SIMPLE_MAP_MAX_HEIGHT)

        # Calculate maximum possible starting positions
        max_start_x = self.MAP_WIDTH - width - 1
        max_start_y = self.MAP_HEIGHT - height - 1

        # Randomize the starting position
        play_x1 = self.rng.randint(2, max(3, max_start_x))
        play_y1 = self.rng.randint(2, max(3, max_start_y))
        play_x2 = min(play_x1 + width, self.MAP_WIDTH - 2)
        play_y2 = min(play_y1 + height, self.MAP_HEIGHT - 2)

        actual_width = play_x2 - play_x1
        actual_height = play_y2 - play_y1

        # self.fill_boundary()

        # Pre-generate all random tiles at once
        # Choose if tiles will be random or solid for the border
        if self.rng.choice([True, False]):
            tile_types = [self.rng.randint(0, 37) for _ in range(
                self.MAP_WIDTH * self.MAP_HEIGHT)]
        else:
            tile_types = [1] * (self.MAP_WIDTH * self.MAP_HEIGHT)
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
        max_up = min(actual_height, max(5, self.GLOBAL_MAX_UP_DEVIATION))
        max_down = min(self.GLOBAL_MAX_DOWN_DEVIATION,
                       max(5, 21 - actual_height))

        # Place entities on X axis, choosing switch or door first
        if self.rng.choice([True, False]):
            switch_x = play_x1 + 1 + self.rng.randint(1, section_width - 1)
            door_x = switch_x + \
                self.rng.randint(1, max(1, play_x2 - switch_x - 1))
            ninja_x = switch_x - \
                self.rng.randint(1, max(1, switch_x - play_x1 - 1))
        else:
            door_x = play_x1 + self.rng.randint(1, section_width - 1)
            switch_x = door_x + \
                self.rng.randint(1, max(1, play_x2 - door_x - 1))
            ninja_x = switch_x + \
                self.rng.randint(1, max(1, play_x2 - switch_x))

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
                        random_tile = self.rng.randint(1, 33)
                        self.set_tile(x, y, random_tile)

        # Calculate final entity positions
        if should_deviate_tiles:
            ninja_y = play_y2 - deviations.get(ninja_x, 0)
        else:
            ninja_y = play_y2
        door_y = play_y2 - deviations.get(door_x, 0)
        switch_y = play_y2 - deviations.get(switch_x, 0)

        # Convert to screen coordinates and place entities
        self.set_ninja_spawn(ninja_x * 4 + 6, ninja_y * 4 + 6)
        self.add_entity(3, door_x * 4 + 6, door_y * 4 + 6, 0, 0,
                        switch_x * 4 + 6, switch_y * 4 + 6)

        return self

    def generate(self, level_type: Literal["MAZE", "SIMPLE_HORIZONTAL_NO_BACKTRACK"] = "MAZE", seed: Optional[int] = None) -> Map:
        """Generate a level of the specified type.

        Args:
            level_type: Type of level to generate ("MAZE" or "SIMPLE_HORIZONTAL_NO_BACKTRACK")
            seed: Random seed for reproducible generation

        Returns:
            Map: A Map instance with the generated level
        """
        if seed is not None:
            self.rng.seed(seed)

        if level_type == "MAZE":
            return self.generate_maze()
        elif level_type == "SIMPLE_HORIZONTAL_NO_BACKTRACK":
            return self.generate_simple_horizontal(seed)
        else:
            raise ValueError(f"Unknown level type: {level_type}")
