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
        self.set_hollow_rectangle(
            self.start_x - self.cell_size,
            self.start_y - self.cell_size,
            self.start_x + self.width * self.cell_size,
            self.start_y + self.height * self.cell_size,
            use_random_tiles_type=True,
        )

    def _fill_with_walls(self):
        """Fill the maze area with solid walls, scaled by cell_size."""
        # Fill an area that's (width * cell_size) x (height * cell_size) tiles
        for y in range(self.height * self.cell_size):
            for x in range(self.width * self.cell_size):
                tile_x = self.start_x + x
                tile_y = self.start_y + y
                random_tile = self.rng.randint(1, VALID_TILE_TYPES)
                # Ensure we're within map boundaries
                if 0 <= tile_x < MAP_TILE_WIDTH and 0 <= tile_y < MAP_TILE_HEIGHT:
                    self.set_tile(tile_x, tile_y, random_tile)

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

    def _is_tile_empty(self, tile_x: int, tile_y: int) -> bool:
        """Check if a tile is empty (walkable) and within bounds."""
        if not (0 <= tile_x < MAP_TILE_WIDTH and 0 <= tile_y < MAP_TILE_HEIGHT):
            return False
        return self.get_tile(tile_x, tile_y) == 0

    def _flood_fill_reachable(
        self, start_tile_x: int, start_tile_y: int
    ) -> Set[Tuple[int, int]]:
        """Find all tiles reachable from a starting tile using flood fill.

        Args:
            start_tile_x: Starting tile x coordinate (in map tiles)
            start_tile_y: Starting tile y coordinate (in map tiles)

        Returns:
            Set of (x, y) tile coordinates reachable from start
        """
        if not self._is_tile_empty(start_tile_x, start_tile_y):
            return set()

        reachable = set()
        to_visit = [(start_tile_x, start_tile_y)]

        while to_visit:
            tx, ty = to_visit.pop()
            if (tx, ty) in reachable:
                continue
            if not self._is_tile_empty(tx, ty):
                continue

            reachable.add((tx, ty))

            # Check 4-connected neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_x, next_y = tx + dx, ty + dy
                if (next_x, next_y) not in reachable:
                    to_visit.append((next_x, next_y))

        return reachable

    def _are_positions_connected(
        self, pos1_x: int, pos1_y: int, pos2_x: int, pos2_y: int
    ) -> bool:
        """Check if two tile positions are connected via empty tiles.

        Args:
            pos1_x, pos1_y: First position (in map tiles)
            pos2_x, pos2_y: Second position (in map tiles)

        Returns:
            True if positions are connected, False otherwise
        """
        reachable = self._flood_fill_reachable(pos1_x, pos1_y)
        return (pos2_x, pos2_y) in reachable

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
                # Ensure we're within map boundaries
                if 0 <= tile_x < MAP_TILE_WIDTH and 0 <= tile_y < MAP_TILE_HEIGHT:
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

                # Validate that the middle cell is within bounds
                if self._is_valid_cell(mid_x, mid_y):
                    self._carve_empty_space(mid_x, mid_y)
                    # Recursively continue from the next cell
                    self._carve_path(next_x, next_y)

    def _place_ninja(self):
        """Place the ninja in a random valid starting position (corners)."""
        # Find all empty cells to identify the largest connected component
        all_empty_cells = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 0:
                    all_empty_cells.append((x, y))

        if not all_empty_cells:
            # Fallback: place at start position if no empty cells (shouldn't happen)
            self.set_ninja_spawn(self.start_x + 1, self.start_y + 1, 1)
            return

        # Use the first empty cell to find the main connected component
        first_cell = all_empty_cells[0]
        first_tile_x = (
            self.start_x + first_cell[0] * self.cell_size + self.cell_size // 2
        )
        first_tile_y = (
            self.start_y + first_cell[1] * self.cell_size + self.cell_size // 2
        )
        main_component = self._flood_fill_reachable(first_tile_x, first_tile_y)

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

        # Try each corner in random order
        corner_names = list(corners.keys())
        self.rng.shuffle(corner_names)

        for corner_name in corner_names:
            (x_min, x_max), (y_min, y_max) = corners[corner_name]

            # Find valid positions in this corner that are in the main component
            valid_positions = []
            for y in range(max(0, y_min), min(self.height, y_max + 1)):
                for x in range(max(0, x_min), min(self.width, x_max + 1)):
                    if self.grid[y][x] == 0:  # Check if position is empty
                        # Convert to tile coordinates
                        tile_x = self.start_x + x * self.cell_size + self.cell_size // 2
                        tile_y = self.start_y + y * self.cell_size + self.cell_size // 2
                        # Check if in main component
                        if (tile_x, tile_y) in main_component:
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
                return

        # Fallback: place at any position in the main component
        if main_component:
            # Pick a random tile from main component
            random_tile = self.rng.choice(list(main_component))
            self.set_ninja_spawn(random_tile[0], random_tile[1], 1)
        else:
            # Last resort fallback (shouldn't reach here)
            self.set_ninja_spawn(self.start_x + 1, self.start_y + 1, 1)

    def _place_exit(self, max_retries: int = 100):
        """Place the exit door and switch in valid positions, ensuring connectivity.

        Args:
            max_retries: Maximum number of attempts to place exit with valid connectivity
        """
        # Convert ninja spawn from map data units to tile coordinates
        ninja_x_tile, ninja_y_tile = self._from_map_data_units(
            self.ninja_spawn_x, self.ninja_spawn_y, NINJA_SPAWN_OFFSET_UNITS
        )

        # Get the connected component containing the ninja
        ninja_component = self._flood_fill_reachable(ninja_x_tile, ninja_y_tile)

        if not ninja_component:
            # If ninja isn't on an empty tile, something went wrong
            return

        # Convert to local grid coordinates for determining door side
        ninja_x_local = ninja_x_tile - self.start_x

        # Determine which side to place the exit based on ninja position
        if ninja_x_local < self.width * self.cell_size // 2:  # Ninja on left side
            door_orientation = 2  # Face left
            # Prefer right side for door
            door_preference = "right"
        else:  # Ninja on right side
            door_orientation = 0  # Face right
            # Prefer left side for door
            door_preference = "left"

        # Collect all valid door and switch positions within the ninja's component
        valid_door_tiles = []
        valid_switch_tiles = []
        min_distance_tiles = max(2, (self.width + self.height) * self.cell_size // 4)

        for tile_x, tile_y in ninja_component:
            # Check if this tile is far enough from ninja
            dist_from_ninja = abs(tile_x - ninja_x_tile) + abs(tile_y - ninja_y_tile)

            # Convert tile to grid cell
            rel_x = tile_x - self.start_x
            rel_y = tile_y - self.start_y

            # Check if within maze bounds
            if rel_x < 0 or rel_y < 0:
                continue
            if (
                rel_x >= self.width * self.cell_size
                or rel_y >= self.height * self.cell_size
            ):
                continue

            grid_x = rel_x // self.cell_size

            # For door: prefer edges based on ninja position
            is_edge = False
            if door_preference == "right" and grid_x >= self.width - 2:
                is_edge = True
            elif door_preference == "left" and grid_x <= 1:
                is_edge = True

            if is_edge and dist_from_ninja >= min_distance_tiles // 2:
                valid_door_tiles.append((tile_x, tile_y))

            # For switch: any position far from ninja
            if dist_from_ninja >= min_distance_tiles:
                valid_switch_tiles.append((tile_x, tile_y))

        if not valid_door_tiles:
            # If no edge positions available, use any position far from ninja
            valid_door_tiles = [
                (tx, ty)
                for tx, ty in ninja_component
                if abs(tx - ninja_x_tile) + abs(ty - ninja_y_tile)
                >= min_distance_tiles // 2
            ]

        if not valid_door_tiles or not valid_switch_tiles:
            # Not enough valid positions, fallback to simpler placement
            all_tiles = [(tx, ty) for tx, ty in ninja_component]
            if len(all_tiles) >= 2:
                # Pick two random tiles from the component
                self.rng.shuffle(all_tiles)
                door_tile = all_tiles[0]
                switch_tile = all_tiles[1]

                door_map_x, door_map_y = door_tile
                switch_map_x, switch_map_y = switch_tile

                self.add_entity(
                    3,
                    door_map_x,
                    door_map_y,
                    door_orientation,
                    0,
                    switch_map_x,
                    switch_map_y,
                )
            return

        # Try to find positions where door and switch are far apart
        best_distance = 0
        best_door_pos = None
        best_switch_pos = None

        # Sample random combinations to find well-separated positions
        for attempt in range(
            min(max_retries, len(valid_door_tiles) * len(valid_switch_tiles))
        ):
            door_x, door_y = self.rng.choice(valid_door_tiles)
            switch_x, switch_y = self.rng.choice(valid_switch_tiles)

            # Skip if they're the same position
            if door_x == switch_x and door_y == switch_y:
                continue

            # Calculate distance between door and switch
            distance = abs(door_x - switch_x) + abs(door_y - switch_y)

            # Verify connectivity (should always be true since they're in same component)
            if distance > best_distance:
                best_distance = distance
                best_door_pos = (door_x, door_y)
                best_switch_pos = (switch_x, switch_y)

                # If we found a good separation, accept it
                if best_distance >= min_distance_tiles // 2:
                    break

        # Use best positions found
        if best_door_pos and best_switch_pos:
            door_map_x, door_map_y = best_door_pos
            switch_map_x, switch_map_y = best_switch_pos

            self.add_entity(
                3,
                door_map_x,
                door_map_y,
                door_orientation,
                0,
                switch_map_x,
                switch_map_y,
            )
        else:
            # Final fallback: just use first available positions
            door_map_x, door_map_y = valid_door_tiles[0]
            switch_map_x, switch_map_y = valid_switch_tiles[0]

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

        # Calculate width and height based on MIN/MAX constants, accounting for cell_size
        # Ensure the maze doesn't exceed map boundaries (42x23)
        # Account for actual size in tiles (width * cell_size) plus boundary walls (2 * cell_size)
        max_possible_width = (MAP_TILE_WIDTH - 2 * self.cell_size) // self.cell_size
        max_possible_height = (MAP_TILE_HEIGHT - 2 * self.cell_size) // self.cell_size

        # Constrain width and height to fit within boundaries
        adjusted_max_width = min(self.MAX_WIDTH, max_possible_width)
        adjusted_max_height = min(self.MAX_HEIGHT, max_possible_height)

        # Ensure we have valid ranges
        self.width = self.rng.randint(
            self.MIN_WIDTH, max(self.MIN_WIDTH, adjusted_max_width)
        )
        self.height = self.rng.randint(
            self.MIN_HEIGHT, max(self.MIN_HEIGHT, adjusted_max_height)
        )

        # Ensure dimensions are odd for proper maze generation (allows even spacing for paths)
        # The algorithm works on even coordinates (0, 2, 4, ...) with walls at odd coordinates
        if self.width % 2 == 0:
            self.width += 1
        if self.height % 2 == 0:
            self.height += 1

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

        # Start maze generation from a random even position (0, 2, 4, ...)
        # Ensure the position is valid by clamping to valid even coordinates
        max_start_x = (self.width - 1) // 2 * 2  # Largest even coordinate < width
        max_start_y = (self.height - 1) // 2 * 2  # Largest even coordinate < height

        # Generate random even coordinate within valid range
        if max_start_x >= 2:
            start_x = self.rng.randrange(0, max_start_x + 1, 2)
        else:
            start_x = 0

        if max_start_y >= 2:
            start_y = self.rng.randrange(0, max_start_y + 1, 2)
        else:
            start_y = 0

        self._carve_path(start_x, start_y)

        # Place entities in the carved maze
        self._place_ninja()
        self._place_exit()

        return self
