from typing import Tuple
from map_generation.constants import GRID_SIZE_FACTOR, NINJA_SPAWN_OFFSET_PX, EXIT_DOOR_OFFSET_PX, SWITCH_OFFSET_PX, GOLD_OFFSET_PX


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
        self.ninja_orientation = -1  # 1 = right, -1 = left

    def _to_screen_coordinates(self, grid_x: int, grid_y: int, offset: int = 0) -> Tuple[int, int]:
        """Convert grid coordinates to screen coordinates."""
        if grid_x is None or grid_y is None:
            return None, None
        return grid_x * GRID_SIZE_FACTOR + offset, grid_y * GRID_SIZE_FACTOR + offset

    def _from_screen_coordinates(self, screen_x: int, screen_y: int, offset: int = 0) -> Tuple[int, int]:
        """Convert screen coordinates to grid coordinates."""
        if screen_x is None or screen_y is None:
            return None, None
        return (screen_x - offset) // GRID_SIZE_FACTOR, (screen_y - offset) // GRID_SIZE_FACTOR

    def set_tile(self, x, y, tile_type):
        """Set a tile at the given coordinates to the specified type."""
        if 0 <= x < self.MAP_WIDTH and 0 <= y < self.MAP_HEIGHT:
            self.tile_data[x + y * self.MAP_WIDTH] = tile_type

    def set_tiles_bulk(self, tile_types):
        """Set all tiles at once using a pre-generated array."""
        if len(tile_types) == len(self.tile_data):
            self.tile_data = tile_types

    def set_ninja_spawn(self, grid_x, grid_y, orientation=None):
        """Set the ninja spawn point coordinates and optionally orientation.
        Converts tile coordinates to screen coordinates (x6 multiplier).
        Orientation: 1 = right, -1 = left"""
        self.ninja_spawn_x, self.ninja_spawn_y = self._to_screen_coordinates(
            grid_x, grid_y, NINJA_SPAWN_OFFSET_PX)
        if orientation is not None:
            self.ninja_orientation = orientation

    def add_entity(self, entity_type, grid_x, grid_y, orientation=0, mode=0, switch_x=None, switch_y=None):
        """Add an entity to the map.
        For doors that require switch coordinates (types 6 and 8), provide switch_x and switch_y.
        For exit doors (type 3), provide switch_x and switch_y for the switch location.
        Converts tile coordinates to screen coordinates (x4.5 multiplier)."""

        # Convert grid coords to screen
        screen_x, screen_y = self._to_screen_coordinates(grid_x, grid_y)
        switch_screen_x, switch_screen_y = self._to_screen_coordinates(
            switch_x, switch_y)

        # Handle entity offsets
        if entity_type == 3:
            screen_x += EXIT_DOOR_OFFSET_PX
            screen_y += EXIT_DOOR_OFFSET_PX
        elif entity_type == 2:
            screen_x += GOLD_OFFSET_PX
            screen_y += GOLD_OFFSET_PX

        # Basic entity data
        entity_data = [entity_type, screen_x, screen_y, orientation, mode]

        # Handle special cases
        if entity_type == 3:  # Exit door
            # Store the exit door data
            self.entity_data.extend(entity_data)
            # Store the switch data right after all exit doors
            self.entity_counts['exit_door'] += 1
            self.entity_data.extend(
                [4, switch_screen_x + SWITCH_OFFSET_PX, switch_screen_y + SWITCH_OFFSET_PX, 0, 0])  # Switch is type 4
        elif entity_type in (6, 8):  # Locked door or trap door
            if switch_x is None or switch_y is None:
                raise ValueError(
                    f"Door type {entity_type} requires switch coordinates")
            entity_data.extend([switch_screen_x, switch_screen_y])
            self.entity_data.extend(entity_data)
        else:
            self.entity_data.extend(entity_data)

        # Update other entity counts
        if entity_type == 2:  # Gold
            self.entity_counts['gold'] += 1
        elif entity_type == 25:  # Death ball
            self.entity_counts['death_ball'] += 1

    def map_data(self):
        """Generate the map data in the format expected by the simulator."""
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
        # Ninja orientation at 1233
        map_data.extend([self.ninja_orientation, 0])
        map_data.extend(self.entity_data)  # Entity data starts at 1235

        return map_data

    def reset(self):
        self.tile_data = [0] * (self.MAP_WIDTH * self.MAP_HEIGHT)
        self.ninja_spawn_x = 1
        self.ninja_spawn_y = 1
        self.ninja_orientation = -1
        self.entity_data = []
        self.entity_counts = {
            'exit_door': 0,
            'gold': 0,
            'death_ball': 0
        }

    def set_empty_rectangle(self, x1, y1, x2, y2):
        """Set a rectangular area to empty tiles efficiently."""
        x1 = max(0, min(x1, self.MAP_WIDTH - 1))
        x2 = max(0, min(x2, self.MAP_WIDTH - 1))
        y1 = max(0, min(y1, self.MAP_HEIGHT - 1))
        y2 = max(0, min(y2, self.MAP_HEIGHT - 1))

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        for y in range(y1, y2 + 1):
            start_idx = x1 + y * self.MAP_WIDTH
            end_idx = x2 + 1 + y * self.MAP_WIDTH
            self.tile_data[start_idx:end_idx] = [0] * (x2 - x1 + 1)
