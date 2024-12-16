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
        """Set the ninja spawn point coordinates."""
        self.ninja_spawn_x = x
        self.ninja_spawn_y = y

    def add_entity(self, entity_type, x, y, orientation=0, mode=0, switch_x=None, switch_y=None):
        """Add an entity to the map.
        For doors that require switch coordinates (types 6 and 8), provide switch_x and switch_y.
        For exit doors (type 3), provide switch_x and switch_y for the switch location."""

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
