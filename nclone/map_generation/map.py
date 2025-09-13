from typing import Tuple, Optional, List
import random
from .constants import (
    GRID_SIZE_FACTOR,
    NINJA_SPAWN_OFFSET_PX,
    EXIT_DOOR_OFFSET_PX,
    SWITCH_OFFSET_PX,
    GOLD_OFFSET_PX,
)
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT

# Valid entity types that can be randomly placed
VALID_RANDOM_ENTITIES = [
    1,  # Toggle mine
    2,  # Gold
    10,  # Launch pad
    11,  # One way platform
    14,  # Drone zap
    17,  # Bounce block,
    21,  # Toggle mine (toggled)
    20,  # Thwump
    24,  # Boost pad
    25,  # Death ball
    26,  # Mini drone
]


class Map:
    """Class for manually constructing simulator maps."""

    def __init__(self, seed: Optional[int] = None):
        # Initialize empty tile data (all tiles are empty, type 0)
        self.tile_data = [0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT)

        # Initialize empty entity data
        self.entity_data = []

        # Initialize entity counts
        self.entity_counts = {"exit_door": 0, "gold": 0, "death_ball": 0}

        # Initialize ninja spawn point (defaults to top-left corner)
        self.ninja_spawn_x = 1
        self.ninja_spawn_y = 1
        self.ninja_orientation = -1  # 1 = right, -1 = left

        self.rng = random.Random(seed)

    def _to_screen_coordinates(
        self, grid_x: int, grid_y: int, offset: int = 0
    ) -> Tuple[int, int]:
        """Convert grid coordinates to screen coordinates."""
        return grid_x * GRID_SIZE_FACTOR + offset, grid_y * GRID_SIZE_FACTOR + offset

    def _from_screen_coordinates(
        self, screen_x: int, screen_y: int, offset: int = 0
    ) -> Tuple[int, int]:
        """Convert screen coordinates to grid coordinates."""
        return (screen_x - offset) // GRID_SIZE_FACTOR, (
            screen_y - offset
        ) // GRID_SIZE_FACTOR

    def set_tile(self, x, y, tile_type):
        """Set a tile at the given coordinates to the specified type."""
        if 0 <= x < MAP_TILE_WIDTH and 0 <= y < MAP_TILE_HEIGHT:
            self.tile_data[x + y * MAP_TILE_WIDTH] = tile_type

    def set_tiles_bulk(self, tile_types):
        """Set all tiles at once using a pre-generated array."""
        if len(tile_types) == len(self.tile_data):
            self.tile_data = tile_types

    def set_ninja_spawn(self, grid_x, grid_y, orientation=None):
        """Set the ninja spawn point coordinates and optionally orientation.
        Converts tile coordinates to screen coordinates (x6 multiplier).
        Orientation: 1 = right, -1 = left"""
        self.ninja_spawn_x, self.ninja_spawn_y = self._to_screen_coordinates(
            grid_x, grid_y, NINJA_SPAWN_OFFSET_PX
        )
        if orientation is not None:
            self.ninja_orientation = orientation

    def add_entity(
        self,
        entity_type,
        grid_x,
        grid_y,
        orientation=0,
        mode=0,
        switch_x=None,
        switch_y=None,
    ):
        """Add an entity to the map.
        For doors that require switch coordinates (types 6 and 8), provide switch_x and switch_y.
        For exit doors (type 3), provide switch_x and switch_y for the switch location.
        Converts tile coordinates to screen coordinates (x4.5 multiplier)."""

        # Convert grid coords to screen
        screen_x, screen_y = self._to_screen_coordinates(grid_x, grid_y)

        # Convert switch grid coords to screen coords, if provided
        switch_screen_x, switch_screen_y = None, None
        if switch_x is not None and switch_y is not None:
            switch_screen_x, switch_screen_y = self._to_screen_coordinates(
                switch_x, switch_y
            )
        elif (
            switch_x is not None or switch_y is not None
        ):  # If one is provided but not the other
            raise ValueError(
                "If switch coordinates are partially provided, both switch_x and switch_y must be set."
            )

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
            self.entity_counts["exit_door"] += 1
            self.entity_data.extend(
                [
                    4,
                    switch_screen_x + SWITCH_OFFSET_PX,
                    switch_screen_y + SWITCH_OFFSET_PX,
                    0,
                    0,
                ]
            )  # Switch is type 4
        elif entity_type in (6, 8):  # Locked door or trap door
            if switch_x is None or switch_y is None:
                raise ValueError(f"Door type {entity_type} requires switch coordinates")
            self.entity_data.extend(entity_data)
        else:
            self.entity_data.extend(entity_data)

        # Update other entity counts
        if entity_type == 2:  # Gold
            self.entity_counts["gold"] += 1
        elif entity_type == 25:  # Death ball
            self.entity_counts["death_ball"] += 1

    def map_data(self):
        """Generate the map data in the format expected by the simulator."""
        # Create the map data array
        map_data = [0] * 184  # Header
        map_data.extend(self.tile_data)  # Tile data (184-1149)
        map_data.extend([0] * 6)  # Unknown section (1150-1155)
        # Exit door count at 1156
        map_data.append(self.entity_counts["exit_door"])
        map_data.extend([0] * 43)  # Unknown section (1157-1199)
        # Death ball count at 1200
        map_data.append(self.entity_counts["death_ball"])
        map_data.extend([0] * 30)  # Unknown section (1201-1230)
        # Ninja spawn at 1231-1232 to match Ninja class expectations
        map_data.extend([self.ninja_spawn_x, self.ninja_spawn_y])
        # Ninja orientation at 1233
        map_data.extend([self.ninja_orientation, 0])
        map_data.extend(self.entity_data)  # Entity data starts at 1235

        return map_data

    def reset(self):
        self.tile_data = [0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
        self.ninja_spawn_x = 1
        self.ninja_spawn_y = 1
        self.ninja_orientation = -1
        self.entity_data = []
        self.entity_counts = {"exit_door": 0, "gold": 0, "death_ball": 0}

    def set_empty_rectangle(self, x1, y1, x2, y2):
        """Set a rectangular area to empty tiles efficiently."""
        x1 = max(0, min(x1, MAP_TILE_WIDTH - 1))
        x2 = max(0, min(x2, MAP_TILE_WIDTH - 1))
        y1 = max(0, min(y1, MAP_TILE_HEIGHT - 1))
        y2 = max(0, min(y2, MAP_TILE_HEIGHT - 1))

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        for y in range(y1, y2 + 1):
            start_idx = x1 + y * MAP_TILE_WIDTH
            end_idx = x2 + 1 + y * MAP_TILE_WIDTH
            self.tile_data[start_idx:end_idx] = [0] * (x2 - x1 + 1)

    @staticmethod
    def from_map_data(map_data: list) -> "Map":
        """Create a new Map instance from map data.

        Args:
            map_data: List of integers representing the map data in simulator format.
                     Format:
                     - 0-183: Header
                     - 184-1149: Tile data
                     - 1150-1155: Unknown section
                     - 1156: Exit door count
                     - 1157-1199: Unknown section
                     - 1200: Death ball count
                     - 1201-1230: Unknown section
                     - 1231-1232: Ninja spawn coordinates
                     - 1233-1234: Unknown
                     - 1235+: Entity data (5 integers per entity)

        Returns:
            A new Map instance with the data from map_data.
        """
        map_instance = Map()

        # Set tile data (indices 184-1149)
        map_instance.tile_data = map_data[184:1150]

        # Set entity counts
        map_instance.entity_counts["exit_door"] = map_data[1156]
        map_instance.entity_counts["death_ball"] = map_data[1200]

        # Set ninja spawn and orientation
        map_instance.ninja_spawn_x = map_data[1231]
        map_instance.ninja_spawn_y = map_data[1232]

        # Process entity data starting at index 1235
        map_instance.entity_data = map_data[1235:]

        # Update gold count by counting type 2 entities
        gold_count = 0
        for i in range(0, len(map_instance.entity_data), 5):
            if i + 4 < len(
                map_instance.entity_data
            ):  # Ensure we have a complete entity
                if map_instance.entity_data[i] == 2:  # Type 2 is gold
                    gold_count += 1
        map_instance.entity_counts["gold"] = gold_count

        return map_instance

    def add_random_entities_outside_playspace(
        self, playspace_x1: int, playspace_y1: int, playspace_x2: int, playspace_y2: int
    ) -> None:
        """Add random entities outside the playspace of the map.

        Args:
            playspace_x1: Left bound of playspace
            playspace_y1: Top bound of playspace
            playspace_x2: Right bound of playspace
            playspace_y2: Bottom bound of playspace
        """
        # Get number of entities to add (0-128)
        num_entities = self.rng.randint(0, 128)

        # Get valid positions outside playspace
        valid_positions: List[Tuple[int, int]] = []

        # Add positions from all regions outside the playspace
        # Top region
        for y in range(2, min(playspace_y1, MAP_TILE_HEIGHT)):
            for x in range(2, MAP_TILE_WIDTH - 2):
                valid_positions.append((x, y))

        # Bottom region
        for y in range(playspace_y2 + 1, MAP_TILE_HEIGHT - 2):
            for x in range(2, MAP_TILE_WIDTH - 2):
                valid_positions.append((x, y))

        # Left region (excluding parts already covered by top/bottom)
        for y in range(playspace_y1, min(playspace_y2 + 1, MAP_TILE_HEIGHT)):
            for x in range(2, playspace_x1):
                valid_positions.append((x, y))

        # Right region (excluding parts already covered by top/bottom)
        for y in range(playspace_y1, min(playspace_y2 + 1, MAP_TILE_HEIGHT)):
            for x in range(playspace_x2 + 1, MAP_TILE_WIDTH - 2):
                valid_positions.append((x, y))

        # Ensure x is not above 43 and y is not above 24
        valid_positions = [(x, y) for x, y in valid_positions if x <= 43 and y <= 24]

        # Add random entities
        for _ in range(num_entities):
            if not valid_positions:  # Break if we run out of positions
                break

            # Choose random position and remove it from available positions
            pos_idx = self.rng.randint(0, len(valid_positions) - 1)
            x, y = valid_positions.pop(pos_idx)

            # Choose random entity type
            entity_type = self.rng.choice(VALID_RANDOM_ENTITIES)

            # Add entity to map
            # For entities that need orientation, randomly choose one
            # These entity types need orientation
            if entity_type in [10, 11, 14, 20, 26]:
                orientation = self.rng.randint(0, 7)
                self.add_entity(entity_type, x, y, orientation)
            else:
                self.add_entity(entity_type, x, y)
