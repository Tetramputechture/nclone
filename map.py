import random


class Map:
    """Class for manually constructing simulator maps."""

    MAP_WIDTH = 42
    MAP_HEIGHT = 23

    def __init__(self):
        # Initialize empty tile data (all tiles are empty, type 0)
        self.tile_data = [0] * (self.MAP_WIDTH * self.MAP_HEIGHT)

        # Initialize empty entity data
        self.entity_data = []

        # Initialize entity counts
        self.entity_counts = {
            'exit_door': 0,
            'gold': 0,
            'death_ball': 0
        }

        # Initialize ninja spawn point (defaults to top-left corner)
        self.ninja_spawn_x = 1
        self.ninja_spawn_y = 1

    def set_tile(self, x, y, tile_type):
        """Set a tile at the given coordinates to the specified type."""
        if 0 <= x < self.MAP_WIDTH and 0 <= y < self.MAP_HEIGHT:
            self.tile_data[x + y * self.MAP_WIDTH] = tile_type

    def set_ninja_spawn(self, x, y):
        """Set the ninja spawn point coordinates.
        Converts tile coordinates to screen coordinates (x6 multiplier)."""
        self.ninja_spawn_x = x
        self.ninja_spawn_y = y

    def add_entity(self, entity_type, x, y, orientation=0, mode=0, switch_x=None, switch_y=None):
        """Add an entity to the map.
        For doors that require switch coordinates (types 6 and 8), provide switch_x and switch_y.
        For exit doors (type 3), provide switch_x and switch_y for the switch location.
        Converts tile coordinates to screen coordinates (x4.5 multiplier)."""

        # Basic entity data
        entity_data = [entity_type, x, y, orientation, mode]

        # Handle special cases
        if entity_type == 3:  # Exit door
            # Store the exit door data
            self.entity_data.extend(entity_data)
            # Store the switch data right after all exit doors
            self.entity_counts['exit_door'] += 1
            self.entity_data.extend(
                [4, switch_x, switch_y, 0, 0])  # Switch is type 4
        elif entity_type in (6, 8):  # Locked door or trap door
            if switch_x is None or switch_y is None:
                raise ValueError(
                    f"Door type {entity_type} requires switch coordinates")
            entity_data.extend([switch_x, switch_y])
            self.entity_data.extend(entity_data)
        else:
            self.entity_data.extend(entity_data)

        # Update other entity counts
        if entity_type == 2:  # Gold
            self.entity_counts['gold'] += 1
        elif entity_type == 25:  # Death ball
            self.entity_counts['death_ball'] += 1

    def generate(self):
        """Generate the map data in the format expected by the simulator.
        The map data format is:
        [0-183]: Header (zeros)
        [184-1149]: Tile data (42x23 = 966 tiles)
        [1150-1155]: Unknown (zeros)
        [1156]: Exit door count
        [1157-1199]: Unknown (zeros)
        [1200]: Death ball count
        [1201-1229]: Unknown (zeros)
        [1230-1231]: Ninja spawn coordinates
        [1232-1234]: Unknown (zeros)
        [1235+]: Entity data"""

        # Create the map data array
        map_data = [0] * 184  # Header
        map_data.extend(self.tile_data)  # Tile data (184-1149)
        map_data.extend([0] * 6)  # Unknown section (1150-1155)
        # Exit door count at 1156
        map_data.append(self.entity_counts['exit_door'])
        map_data.extend([0] * 43)  # Unknown section (1157-1199)
        # Death ball count at 1200
        map_data.append(self.entity_counts['death_ball'])
        map_data.extend([0] * 30)  # Unknown section (1201-1230)
        # Ninja spawn at 1231-1232 to match Ninja class expectations
        map_data.extend([self.ninja_spawn_x, self.ninja_spawn_y])
        map_data.extend([0] * 2)  # Unknown section (1233-1234)
        map_data.extend(self.entity_data)  # Entity data starts at 1235

        return map_data

    def reset(self):
        self.tile_data = [0] * (self.MAP_WIDTH * self.MAP_HEIGHT)
        self.ninja_spawn_x = 1
        self.ninja_spawn_y = 1
        self.entity_data = []
        self.entity_counts = {
            'exit_door': 0,
            'gold': 0,
            'death_ball': 0
        }


class MapGenerator(Map):
    """Class for generating maps with common patterns and features."""

    def __init__(self):
        super().__init__()
        self.rng = random.Random()

    def random_tile_type(self):
        """Return a random tile type. Tile types defined in TILE_GRID_EDGE_MAP."""
        return self.rng.randint(0, 37)

    def fill_boundaries(self, width=1, left_width=None, right_width=None, top_width=None, bottom_width=None):
        """Fill the map boundaries with solid tiles.

        Args:
            width (int or tuple): Default width for all boundaries if no specific width is provided
            left_width (int): Width of left boundary (max 26)
            right_width (int): Width of right boundary (max 26)
            top_width (int): Width of top boundary (max 11)
            bottom_width (int): Width of bottom boundary (max 12)
        """
        # Set default widths if not specified
        left = min(left_width if left_width is not None else width, 26)
        right = min(right_width if right_width is not None else width, 26)
        top = min(top_width if top_width is not None else width, 11)
        bottom = min(bottom_width if bottom_width is not None else width, 12)

        # Fill left boundary
        for x in range(left):
            for y in range(self.MAP_HEIGHT):
                self.set_tile(x, y, 1)

        # Fill right boundary
        for x in range(self.MAP_WIDTH - right, self.MAP_WIDTH):
            for y in range(self.MAP_HEIGHT):
                self.set_tile(x, y, 1)

        # Fill top boundary
        for y in range(top):
            for x in range(self.MAP_WIDTH):
                self.set_tile(x, y, 1)

        # Fill bottom boundary
        for y in range(self.MAP_HEIGHT - bottom, self.MAP_HEIGHT):
            for x in range(self.MAP_WIDTH):
                self.set_tile(x, y, 1)

    def set_empty_rectangle(self, x1, y1, x2, y2):
        """Set a rectangular area to empty tiles.

        Args:
            x1 (int): Starting X coordinate
            y1 (int): Starting Y coordinate
            x2 (int): Ending X coordinate (inclusive)
            y2 (int): Ending Y coordinate (inclusive)
        """
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, self.MAP_WIDTH - 1))
        x2 = max(0, min(x2, self.MAP_WIDTH - 1))
        y1 = max(0, min(y1, self.MAP_HEIGHT - 1))
        y2 = max(0, min(y2, self.MAP_HEIGHT - 1))

        # Ensure x1,y1 is top-left and x2,y2 is bottom-right
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Fill the rectangle with empty tiles
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                self.set_tile(x, y, 0)

    def generate_random_level(self, level_type="SIMPLE_HORIZONTAL_NO_BACKTRACK", seed=None):
        """Generate a random level based on the specified type.

        Args:
            level_type (str): Type of level to generate. Currently supports:
                - SIMPLE_HORIZONTAL_NO_BACKTRACK: A simple horizontal level with ninja, switch, and exit
            seed (int, optional): Random seed for level generation
        """
        self.reset()
        if seed is not None:
            self.rng.seed(seed)

        if level_type == "SIMPLE_HORIZONTAL_NO_BACKTRACK":
            # Generate random dimensions for play space
            # Increased minimum width to ensure enough space
            width = self.rng.randint(9, 40)
            height = self.rng.randint(1, 21)

            # Calculate maximum possible starting positions while leaving room for the play space
            max_start_x = self.MAP_WIDTH - width - 1
            max_start_y = self.MAP_HEIGHT - height - 1

            # Randomize the starting position (but keep at least 1 tile from edges)
            play_x1 = self.rng.randint(2, max(3, max_start_x))
            play_y1 = self.rng.randint(2, max(3, max_start_y))
            play_x2 = min(play_x1 + width, self.MAP_WIDTH - 2)
            play_y2 = min(play_y1 + height, self.MAP_HEIGHT - 2)

            # Fill the entire map with random tile types
            for y in range(self.MAP_HEIGHT):
                for x in range(self.MAP_WIDTH):
                    self.set_tile(x, y, self.random_tile_type())

            # Create the empty play space
            self.set_empty_rectangle(play_x1, play_y1, play_x2, play_y2)

            # Now, to make sure the play space is bounded on the bottom, create
            # a line of type 1 tiles with a width of the level
            # at the bottom of the play space
            for x in range(play_x1, play_x2 + 1):
                self.set_tile(x, play_y2 + 1, 1)

            # Floor Y coordinate (where entities will be placed)
            floor_y = play_y2 * 4  # Bottom of the play space

            # Calculate available space for elements
            usable_width = play_x2 - play_x1 - 2  # -2 for wall padding

            # Divide the usable space into three sections
            # Ensure minimum section width
            section_width = max(3, usable_width // 3)

            # Place switch and door with proper spacing
            if self.rng.choice([True, False]):  # Switch on left, door on right
                # Place switch in first third of space
                switch_x = play_x1 + 1 + self.rng.randint(1, section_width - 1)
                # Place door to the right of the switch, a random distance
                # away from the switch, where the max distance is the
                # distance between the switch and the right edge of the play space -1
                door_x = switch_x + \
                    self.rng.randint(1, max(1, play_x2 - switch_x - 1))
                # Ninja spawns to the left of the switch a random distance
                # away from the switch, where the max distance is the
                # distance between the switch and the left edge of the play space +1
                ninja_x = switch_x * 4 - \
                    self.rng.randint(1, max(1, switch_x - play_x1 - 1))
            else:  # Door on left, switch on right
                # Place door in first third of space
                door_x = play_x1 + self.rng.randint(1, section_width - 1)
                # Place switch to the right of the door, a random distance
                # away from the door, where the max distance is the
                # distance between the door and the right edge of the play space -1
                switch_x = door_x + \
                    self.rng.randint(1, max(1, play_x2 - door_x - 1))
                # Ninja spawns to the right of the switch a random distance
                # away from the switch, where the max distance is the
                # distance between the switch and the right edge of the play space -1
                ninja_x = switch_x * 4 + \
                    self.rng.randint(1, max(1, play_x2 - switch_x)) + 10

            self.set_ninja_spawn(ninja_x, floor_y + 6)

            # Add exit door and switch (convert from tile coordinates to screen coordinates)
            self.add_entity(3, door_x * 4 + 6, floor_y +
                            6, 0, 0, switch_x * 4 + 6, floor_y + 6)

            return True

        return False  # Unsupported level type
