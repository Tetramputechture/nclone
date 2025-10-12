"""Maze generation module for N++ levels."""

from .map import Map
from typing import List, Tuple, Optional, Set
from .constants import VALID_TILE_TYPES, NINJA_SPAWN_OFFSET_UNITS
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class MazeGenerator(Map):
    """Generates maze-style N++ levels."""

    # Maze generation constants
    MIN_WIDTH = 6
    MAX_WIDTH = 20
    MIN_HEIGHT = 6
    MAX_HEIGHT = 10
    MAX_CELL_SIZE = 4

    def __init__(self, seed: Optional[int] = None):
        """Initialize the maze generator.

        Args:
            seed: Random seed for reproducible generation
        """
        super().__init__(seed)

        # Initialize tracking variables
        # Width and height will be set in generate() to allow modification of MIN/MAX constants
        self.width = 0
        self.height = 0
        self.start_x = 0
        self.start_y = 0
        self.visited: Set[Tuple[int, int]] = set()
        self.grid: List[List[int]] = []
        self.ninja_orientation = -1  # Default orientation (facing right)
        self.cell_size = 1  # Size of each cell in tiles (1x1, 2x2, 3x3, or 4x4)

    def _init_solid_map(self):
        """Initialize both the grid and map tiles with solid walls."""
        # Initialize the grid for maze algorithm logic
        self.grid = [[1] * self.width for _ in range(self.height)]

        # Fill the map with solid walls
        self._fill_with_walls()

        # Add solid walls around maze boundaries (scaled by cell_size)
        use_random_tiles_type = self.rng.choice([True, False])
        self.set_hollow_rectangle(
            self.start_x - self.cell_size,
            self.start_y - self.cell_size,
            self.start_x + self.width * self.cell_size,
            self.start_y + self.height * self.cell_size,
            use_random_tiles_type=use_random_tiles_type,
        )

    def _fill_with_walls(self):
        """Fill the maze area with solid walls, scaled by cell_size."""
        # Fill an area that's (width * cell_size) x (height * cell_size) tiles
        for y in range((self.height + 1) * self.cell_size):
            for x in range((self.width + 1) * self.cell_size):
                self.set_tile(self.start_x + x, self.start_y + y, 1)  # 1 = wall

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
        """Carve an empty space at the given coordinates in both grid and map.

        When cell_size > 1, this carves a block of cell_size x cell_size tiles.
        """
        self.grid[y][x] = 0  # Update grid for maze logic

        # Carve a block of cell_size x cell_size tiles
        for dy in range(self.cell_size):
            for dx in range(self.cell_size):
                tile_x = self.start_x + x * self.cell_size + dx
                tile_y = self.start_y + y * self.cell_size + dy
                self.set_tile(tile_x, tile_y, 0)  # 0 = empty space

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
        """Place the ninja in a random valid starting position (corners)."""
        # Define possible corner regions
        corners = {
            "top_left": ((0, 2), (0, 2)),  # (x_range, y_range)
            "top_right": ((self.width - 3, self.width - 1), (0, 2)),
            "bottom_left": ((0, 2), (self.height - 3, self.height - 1)),
            "bottom_right": (
                (self.width - 3, self.width - 1),
                (self.height - 3, self.height - 1),
            ),
        }

        # Randomly select a corner
        corner_name = self.rng.choice(list(corners.keys()))
        (x_min, x_max), (y_min, y_max) = corners[corner_name]

        # Find valid positions in the selected corner
        valid_positions = []
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if self.grid[y][x] == 0:  # Check if position is empty
                    valid_positions.append((x, y))

        if valid_positions:
            grid_x, grid_y = self.rng.choice(valid_positions)
            # Set orientation based on position
            orientation = 1  # Default facing right
            if "right" in corner_name:  # If on right side, face left
                orientation = -1
            # Convert grid coordinates to map coordinates (scaled by cell_size)
            # Place ninja in the center of the cell block
            map_x = self.start_x + grid_x * self.cell_size + self.cell_size // 2
            map_y = self.start_y + grid_y * self.cell_size + self.cell_size // 2
            self.set_ninja_spawn(map_x, map_y, orientation)
        else:
            # Fallback to first empty cell if no valid corner positions
            for y in range(self.height):
                for x in range(self.width):
                    if self.grid[y][x] == 0:
                        map_x = self.start_x + x * self.cell_size + self.cell_size // 2
                        map_y = self.start_y + y * self.cell_size + self.cell_size // 2
                        self.set_ninja_spawn(map_x, map_y, 1)
                        return

    def _place_exit(self):
        """Place the exit door and switch in valid positions.

        Args:
            add_locked_doors: If True, may place locked doors in front of exit door or switch
        """
        # Place exit door on the opposite side from ninja
        # Convert ninja spawn from map data units to global grid coordinates (in tiles)
        ninja_x_global, ninja_y_global = self._from_map_data_units(
            self.ninja_spawn_x, self.ninja_spawn_y, NINJA_SPAWN_OFFSET_UNITS
        )
        # Convert to local grid coordinates (relative to maze start position, scaled by cell_size)
        ninja_x = (ninja_x_global - self.start_x) // self.cell_size
        ninja_y = (ninja_y_global - self.start_y) // self.cell_size

        # Determine which side to place the exit based on ninja position
        if ninja_x < self.width // 2:  # Ninja on left side
            valid_door_positions = []
            for y in range(self.height):
                # Check if cell next to right edge is path
                if self.grid[y][self.width - 2] == 0:
                    valid_door_positions.append((self.width - 2, y))
            door_orientation = 2  # Face left
        else:  # Ninja on right side
            valid_door_positions = []
            for y in range(self.height):
                if self.grid[y][1] == 0:  # Check if cell next to left edge is path
                    valid_door_positions.append((1, y))
            door_orientation = 0  # Face right

        if valid_door_positions:
            door_x, door_y = self.rng.choice(valid_door_positions)

            # Find valid switch positions (empty cells not too close to ninja, and not on top of exit door)
            valid_switch_positions = []
            min_distance = max(2, (self.width + self.height) // 4)

            for y in range(self.height):
                for x in range(self.width):
                    if self.grid[y][x] == 0:
                        # Check: far enough from ninja, not on exit door, and not within 1 tile radius of ninja
                        # Use Chebyshev distance (max of x and y distances) for "1 tile radius"
                        if (
                            abs(x - ninja_x) + abs(y - ninja_y) >= min_distance
                            and (x, y) != (door_x, door_y)
                            and max(abs(x - ninja_x), abs(y - ninja_y)) > 1
                        ):
                            valid_switch_positions.append((x, y))

            if valid_switch_positions:
                switch_x, switch_y = self.rng.choice(valid_switch_positions)

                # Convert grid coordinates to map coordinates (scaled by cell_size, centered in cell)
                door_map_x = (
                    self.start_x + door_x * self.cell_size + self.cell_size // 2
                )
                door_map_y = (
                    self.start_y + door_y * self.cell_size + self.cell_size // 2
                )
                switch_map_x = (
                    self.start_x + switch_x * self.cell_size + self.cell_size // 2
                )
                switch_map_y = (
                    self.start_y + switch_y * self.cell_size + self.cell_size // 2
                )

                # Add exit door and switch using Map's add_entity method
                self.add_entity(
                    3,
                    door_map_x,
                    door_map_y,
                    door_orientation,
                    0,
                    switch_map_x,
                    switch_map_y,
                )

    def generate(self, seed: Optional[int] = None) -> Map:
        """Generate a complete maze with entities.

        This method follows these steps:
        1. Start with a completely solid level (all walls)
        2. Carve paths through the walls using a depth-first search algorithm
        3. Place the ninja spawn point near the left side
        4. Place the exit door and switch in valid positions
        6. Add random entities outside the playspace

        Args:
            seed: Random seed for reproducible generation

        Returns:
            Map: A Map instance with the generated maze and entities
        """
        if seed is not None:
            self.rng.seed(seed)

        self.reset()
        # Reset state and ensure we start with a solid level
        self.visited.clear()

        # Randomly choose cell size (1x1, 2x2, 3x3, or 4x4)
        self.cell_size = self.rng.randint(1, self.MAX_CELL_SIZE)

        # Calculate width and height based on MIN/MAX constants
        # This allows modification of constants before calling generate()
        self.width = self.rng.randint(self.MIN_WIDTH, self.MAX_WIDTH)
        self.height = self.rng.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)

        # Calculate random offset for the maze, ensuring it fits within map bounds
        # Account for actual size in tiles (width * cell_size) plus boundary walls
        actual_width_tiles = self.width * self.cell_size + 2 * self.cell_size
        actual_height_tiles = self.height * self.cell_size + 2 * self.cell_size
        max_start_x = MAP_TILE_WIDTH - actual_width_tiles
        max_start_y = MAP_TILE_HEIGHT - actual_height_tiles
        self.start_x = self.rng.randint(0, max(0, max_start_x))
        self.start_y = self.rng.randint(0, max(0, max_start_y))

        # Pre-generate all random tiles at once
        # Choose if tiles will be random, solid, or empty for the border
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
        self._init_solid_map()

        # Start maze generation from a random position on the left side
        start_x = self.rng.randrange(0, self.width, 2)
        start_y = self.rng.randrange(0, self.height, 2)  # Random even row
        self._carve_path(start_x, start_y)

        # Place entities in the carved maze
        self._place_ninja()
        self._place_exit()

        # Add random entities outside the playspace
        # For maze, playspace is the maze area with offset (scaled by cell_size)
        playspace = (
            self.start_x - self.cell_size,
            self.start_y - self.cell_size,
            self.start_x + self.width * self.cell_size + self.cell_size,
            self.start_y + self.height * self.cell_size + self.cell_size,
        )
        self.add_random_entities_outside_playspace(
            playspace[0], playspace[1], playspace[2], playspace[3]
        )

        return self
