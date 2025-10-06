"""Maze generation module for N++ levels."""

from .map import Map
from typing import List, Tuple, Optional, Set
from .constants import VALID_TILE_TYPES, NINJA_SPAWN_OFFSET_PX, SWITCH_OFFSET_PX
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class MazeGenerator(Map):
    """Generates maze-style N++ levels."""

    # Maze generation constants
    MIN_WIDTH = 6
    MAX_WIDTH = 20
    MIN_HEIGHT = 6
    MAX_HEIGHT = 10

    MAX_GOLD = 0

    def __init__(self, seed: Optional[int] = None):
        """Initialize the maze generator.

        Args:
            seed: Random seed for reproducible generation
        """
        super().__init__(seed)
        self.width = self.rng.randint(self.MIN_WIDTH, self.MAX_WIDTH)
        self.height = self.rng.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)

        # Initialize tracking variables
        self.visited: Set[Tuple[int, int]] = set()
        self.grid: List[List[int]] = []
        self.ninja_orientation = -1  # Default orientation (facing right)

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
        for y in range(self.height + 1):
            for x in range(self.width + 1):
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
            # Convert grid coordinates to pixel coordinates
            self.set_ninja_spawn(grid_x, grid_y, orientation)
        else:
            # Fallback to first empty cell if no valid corner positions
            for y in range(self.height):
                for x in range(self.width):
                    if self.grid[y][x] == 0:
                        self.set_ninja_spawn(x, y, 1)
                        return

    def _place_exit(self, add_locked_doors: bool = False):
        """Place the exit door and switch in valid positions.
        
        Args:
            add_locked_doors: If True, may place locked doors in front of exit door or switch
        """
        # Place exit door on the opposite side from ninja
        ninja_x, _ = self._from_screen_coordinates(
            self.ninja_spawn_x, self.ninja_spawn_y, NINJA_SPAWN_OFFSET_PX
        )

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
            min_distance = (self.width + self.height) // 4

            for y in range(self.height):
                for x in range(self.width):
                    if self.grid[y][x] == 0:
                        ninja_grid_x, ninja_grid_y = self._from_screen_coordinates(
                            self.ninja_spawn_x,
                            self.ninja_spawn_y,
                            NINJA_SPAWN_OFFSET_PX,
                        )
                        if abs(x - ninja_grid_x) + abs(
                            y - ninja_grid_y
                        ) >= min_distance and (x, y) != (door_x, door_y):
                            valid_switch_positions.append((x, y))

            if valid_switch_positions:
                switch_x, switch_y = self.rng.choice(valid_switch_positions)

                # Add exit door and switch using Map's add_entity method
                self.add_entity(
                    3, door_x, door_y, door_orientation, 0, switch_x, switch_y
                )
                
                # Optionally add locked doors in front of exit door or switch
                if add_locked_doors:
                    self._place_locked_doors(door_x, door_y, switch_x, switch_y, door_orientation)
    
    def _place_locked_doors(self, exit_door_x: int, exit_door_y: int, 
                            exit_switch_x: int, exit_switch_y: int, 
                            exit_door_orientation: int):
        """Place 1-2 locked doors in front of the exit door or switch.
        
        Args:
            exit_door_x, exit_door_y: Position of the exit door
            exit_switch_x, exit_switch_y: Position of the exit switch
            exit_door_orientation: Orientation of the exit door
        """
        # Decide how many locked doors: 0 (33%), 1 (50%), or 2 (17%)
        num_locked_doors = self.rng.choices([0, 1, 2], weights=[33, 50, 17])[0]
        
        if num_locked_doors == 0:
            return
        
        # Find valid positions for locked door switches (away from ninja and exit points)
        ninja_grid_x, ninja_grid_y = self._from_screen_coordinates(
            self.ninja_spawn_x, self.ninja_spawn_y, NINJA_SPAWN_OFFSET_PX
        )
        
        valid_switch_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 0:
                    # Not too close to ninja, exit door, or exit switch
                    if (abs(x - ninja_grid_x) + abs(y - ninja_grid_y) >= 2 and
                        (x, y) != (exit_door_x, exit_door_y) and
                        (x, y) != (exit_switch_x, exit_switch_y)):
                        valid_switch_positions.append((x, y))
        
        if not valid_switch_positions:
            return  # Can't place locked doors without switch positions
        
        # Shuffle to randomize selection
        self.rng.shuffle(valid_switch_positions)
        
        if num_locked_doors == 1:
            # Place locked door in front of exit door OR exit switch (50/50)
            if self.rng.random() < 0.5:
                # Block exit door
                self._add_locked_door_blocking(exit_door_x, exit_door_y, 
                                               valid_switch_positions[0], 
                                               exit_door_orientation)
            else:
                # Block exit switch
                self._add_locked_door_blocking(exit_switch_x, exit_switch_y,
                                               valid_switch_positions[0],
                                               exit_door_orientation)
        elif num_locked_doors == 2 and len(valid_switch_positions) >= 2:
            # Block both exit door and exit switch
            self._add_locked_door_blocking(exit_door_x, exit_door_y,
                                           valid_switch_positions[0],
                                           exit_door_orientation)
            self._add_locked_door_blocking(exit_switch_x, exit_switch_y,
                                           valid_switch_positions[1],
                                           exit_door_orientation)
    
    def _add_locked_door_blocking(self, target_x: int, target_y: int, 
                                   switch_pos: Tuple[int, int], orientation: int):
        """Add a locked door that blocks access to a target position.
        
        Args:
            target_x, target_y: Grid position of the entity to block
            switch_pos: (x, y) tuple for the locked door switch position
            orientation: Door orientation (determines which direction the door faces)
        """
        switch_x, switch_y = switch_pos
        
        # Find an adjacent empty tile to place the locked door
        # The door should be in a position that the player must pass through to reach the target
        ninja_grid_x, ninja_grid_y = self._from_screen_coordinates(
            self.ninja_spawn_x, self.ninja_spawn_y, NINJA_SPAWN_OFFSET_PX
        )
        
        # Check all adjacent tiles and find one that's empty and closer to the ninja
        adjacent_positions = [
            (target_x - 1, target_y),  # Left
            (target_x + 1, target_y),  # Right
            (target_x, target_y - 1),  # Up
            (target_x, target_y + 1),  # Down
        ]
        
        valid_door_positions = []
        for adj_x, adj_y in adjacent_positions:
            if (self._is_valid_cell(adj_x, adj_y) and 
                self.grid[adj_y][adj_x] == 0 and  # Empty tile
                (adj_x, adj_y) != (ninja_grid_x, ninja_grid_y)):  # Not ninja spawn
                # Check if this tile is between ninja and target
                ninja_dist = abs(adj_x - ninja_grid_x) + abs(adj_y - ninja_grid_y)
                target_dist = abs(adj_x - target_x) + abs(adj_y - target_y)
                if ninja_dist < abs(target_x - ninja_grid_x) + abs(target_y - ninja_grid_y):
                    valid_door_positions.append((adj_x, adj_y))
        
        # If we found valid positions, place the door at one of them
        if valid_door_positions:
            door_x, door_y = self.rng.choice(valid_door_positions)
            # Determine orientation based on position relative to target
            if door_x < target_x:
                door_orient = 0  # Face right (blocking path to target)
            elif door_x > target_x:
                door_orient = 2  # Face left
            elif door_y < target_y:
                door_orient = 3  # Face down
            else:
                door_orient = 1  # Face up
            
            self.add_entity(6, door_x, door_y, door_orient, 0, switch_x, switch_y)
        # If no valid adjacent position, don't place the locked door

    def generate(self, seed: Optional[int] = None, add_locked_doors: bool = False) -> Map:
        """Generate a complete maze with entities.

        This method follows these steps:
        1. Start with a completely solid level (all walls)
        2. Carve paths through the walls using a depth-first search algorithm
        3. Place the ninja spawn point near the left side
        4. Place the exit door and switch in valid positions
        5. Optionally add locked doors in front of exit door or switch
        6. Add random entities outside the playspace

        Args:
            seed: Random seed for reproducible generation
            add_locked_doors: If True, may place locked doors in front of exit door or switch

        Returns:
            Map: A Map instance with the generated maze and entities
        """
        if seed is not None:
            self.rng.seed(seed)

        self.reset()
        # Reset state and ensure we start with a solid level
        self.visited.clear()

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
        self._place_exit(add_locked_doors=add_locked_doors)

        # Add random entities outside the playspace
        # For maze, playspace is the maze area plus some padding
        playspace = (2, 2, 2 + self.width * 2, 2 + self.height * 2)
        self.add_random_entities_outside_playspace(
            playspace[0], playspace[1], playspace[2], playspace[3]
        )

        return self
