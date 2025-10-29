from typing import Tuple, Optional, List
import random
from .constants import (
    GRID_SIZE_FACTOR,
    NINJA_SPAWN_OFFSET_UNITS,
    EXIT_DOOR_OFFSET_UNITS,
    SWITCH_OFFSET_UNITS,
    GOLD_OFFSET_UNITS,
    LOCKED_DOOR_OFFSET_UNITS,
)
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from . import terrain_utils
from .constants import VALID_TILE_TYPES

# Valid entity types that can be randomly placed
VALID_RANDOM_ENTITIES = [
    1,  # Toggle mine
    # 2,  # Gold
    # 10,  # Launch pad
    # 11,  # One way platform
    # 14,  # Drone zap
    # 17,  # Bounce block,
    21,  # Toggle mine (toggled)
    # 20,  # Thwump
    # 24,  # Boost pad
    # 25,  # Death ball
    # 26,  # Mini drone
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

    def _to_map_data_units(
        self, grid_x: int, grid_y: int, offset: int = 0
    ) -> Tuple[int, int]:
        """Convert grid coordinates to map data units.

        Map data units are multiplied by 6 during entity loading to get pixel coordinates.
        """
        return grid_x * GRID_SIZE_FACTOR + offset, grid_y * GRID_SIZE_FACTOR + offset

    def _from_map_data_units(
        self, units_x: int, units_y: int, offset: int = 0
    ) -> Tuple[int, int]:
        """Convert map data units to grid coordinates."""
        return (units_x - offset) // GRID_SIZE_FACTOR, (
            units_y - offset
        ) // GRID_SIZE_FACTOR

    def set_tile(self, x, y, tile_type):
        """Set a tile at the given coordinates to the specified type."""
        if 0 <= x < MAP_TILE_WIDTH and 0 <= y < MAP_TILE_HEIGHT:
            self.tile_data[x + y * MAP_TILE_WIDTH] = tile_type

    def get_tile(self, x, y):
        """Get the tile type at the given coordinates.

        Args:
            x: X coordinate (in tiles)
            y: Y coordinate (in tiles)

        Returns:
            Tile type (int), or -1 if out of bounds
        """
        # Ensure coordinates are integers
        x, y = int(x), int(y)
        if 0 <= x < MAP_TILE_WIDTH and 0 <= y < MAP_TILE_HEIGHT:
            return self.tile_data[x + y * MAP_TILE_WIDTH]
        return -1

    def set_tiles_bulk(self, tile_types):
        """Set all tiles at once using a pre-generated array."""
        if len(tile_types) == len(self.tile_data):
            self.tile_data = tile_types

    def _find_closest_valid_tile(
        self, x: int, y: int, tile_type: int = 0
    ) -> Tuple[int, int]:
        """Find the closest tile of a given type within map boundaries.

        Uses BFS to find the nearest tile of the specified type that's within bounds.

        Args:
            x: Starting x coordinate
            y: Starting y coordinate
            tile_type: The tile type to search for (default: 0 for empty)

        Returns:
            Tuple of (x, y) for the closest valid tile, or original (x, y) if none found
        """
        # First check if the position is already valid
        if 0 <= x < MAP_TILE_WIDTH and 0 <= y < MAP_TILE_HEIGHT:
            if self.tile_data[x + y * MAP_TILE_WIDTH] == tile_type:
                return (x, y)

        # Clamp to boundaries first
        x = max(0, min(x, MAP_TILE_WIDTH - 1))
        y = max(0, min(y, MAP_TILE_HEIGHT - 1))

        # BFS to find closest valid tile
        from collections import deque

        visited = set()
        queue = deque([(x, y, 0)])  # (x, y, distance)
        visited.add((x, y))

        while queue:
            curr_x, curr_y, dist = queue.popleft()

            # Check if this tile is valid
            if 0 <= curr_x < MAP_TILE_WIDTH and 0 <= curr_y < MAP_TILE_HEIGHT:
                if self.tile_data[curr_x + curr_y * MAP_TILE_WIDTH] == tile_type:
                    return (curr_x, curr_y)

            # Add neighbors (8-directional search for better coverage)
            for dx, dy in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                next_x, next_y = curr_x + dx, curr_y + dy
                if (
                    (next_x, next_y) not in visited
                    and 0 <= next_x < MAP_TILE_WIDTH
                    and 0 <= next_y < MAP_TILE_HEIGHT
                ):
                    visited.add((next_x, next_y))
                    queue.append((next_x, next_y, dist + 1))

        # If no valid tile found, return clamped position
        return (x, y)

    def set_ninja_spawn(self, grid_x, grid_y, orientation=None):
        """Set the ninja spawn point coordinates and optionally orientation.
        Converts tile coordinates to map data units (multiplied by 6 to get pixels during loading).
        Orientation: 1 = right, -1 = left"""
        self.ninja_spawn_x, self.ninja_spawn_y = self._to_map_data_units(
            grid_x, grid_y, NINJA_SPAWN_OFFSET_UNITS
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
        Converts tile coordinates to map data units (multiplied by 6 to get pixels during loading)."""

        # Convert grid coords to map data units
        units_x, units_y = self._to_map_data_units(grid_x, grid_y)

        # Convert switch grid coords to map data units, if provided
        switch_units_x, switch_units_y = None, None
        if switch_x is not None and switch_y is not None:
            switch_units_x, switch_units_y = self._to_map_data_units(switch_x, switch_y)
        elif (
            switch_x is not None or switch_y is not None
        ):  # If one is provided but not the other
            raise ValueError(
                "If switch coordinates are partially provided, both switch_x and switch_y must be set."
            )

        # Handle entity offsets
        if entity_type == 3:
            units_x += EXIT_DOOR_OFFSET_UNITS
            units_y += EXIT_DOOR_OFFSET_UNITS
        elif entity_type == 2:
            units_x += GOLD_OFFSET_UNITS
            units_y += GOLD_OFFSET_UNITS
        elif entity_type in (6, 8):  # Locked door and trap door
            units_x += LOCKED_DOOR_OFFSET_UNITS
            units_y += LOCKED_DOOR_OFFSET_UNITS

        # Basic entity data
        entity_data = [entity_type, units_x, units_y, orientation, mode]

        # Handle special cases
        if entity_type == 3:  # Exit door
            # Store the exit door data
            self.entity_data.extend(entity_data)
            # Store the switch data right after all exit doors
            self.entity_counts["exit_door"] += 1
            self.entity_data.extend(
                [
                    4,
                    switch_units_x + SWITCH_OFFSET_UNITS,
                    switch_units_y + SWITCH_OFFSET_UNITS,
                    0,
                    0,
                ]
            )  # Switch is type 4
        elif entity_type in (6, 8):  # Locked door or trap door
            if switch_x is None or switch_y is None:
                raise ValueError(f"Door type {entity_type} requires switch coordinates")
            # Add entity data and switch coordinates
            # Use 9-byte format to match existing map files:
            # [type, x, y, orientation, mode, 7, switch_x, switch_y, 0]
            # Switch coordinates must be at index+6 and index+7 for entity loader
            # Byte 5 is always 7 in real N++ maps (compatibility/format identifier)
            # Note: No offset applied to door position (already done above as "pass")
            # but switch still gets offset for proper positioning
            self.entity_data.extend(entity_data)
            self.entity_data.extend(
                [
                    7,  # format byte at index+5 (always 7 in real maps)
                    switch_units_x + SWITCH_OFFSET_UNITS,  # switch_x at index+6
                    switch_units_y + SWITCH_OFFSET_UNITS,  # switch_y at index+7
                    0,  # padding byte at index+8
                ]
            )
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

    def set_hollow_rectangle(
        self,
        x1,
        y1,
        x2,
        y2,
        use_random_tiles_type: bool = False,
        chaotic_random_tiles: bool = False,
    ):
        """Set a hollow rectangle border with wall-appropriate tile types.

        When use_random_tiles_type is True, each wall uses tiles with appropriate
        solid faces: left wall uses right-facing tiles, right wall uses left-facing
        tiles, top wall uses bottom-facing tiles, bottom wall uses top-facing tiles.

        Args:
            x1, y1: Top-left corner of the rectangle
            x2, y2: Bottom-right corner of the rectangle
            use_random_tiles_type: If True, use random tiles appropriate for each wall
        """
        # Tile sets for each wall direction based on solid face orientation
        LEFT_WALL_TILES = [1, 3, 7, 8, 11, 11, 15, 16, 23, 24, 27, 28, 31, 32]
        RIGHT_WALL_TILES = [1, 5, 6, 9, 10, 13, 14, 17, 22, 25, 26, 29, 30, 33]
        TOP_WALL_TILES = [1, 4, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 32, 33]
        BOTTOM_WALL_TILES = [1, 2, 6, 7, 10, 11, 14, 15, 18, 19, 30, 31]

        # Top wall
        for x in range(x1, x2 + 1):
            if chaotic_random_tiles:
                self.set_tile(x, y1, self.rng.randint(1, VALID_TILE_TYPES))
            elif use_random_tiles_type:
                self.set_tile(x, y1, self.rng.choice(TOP_WALL_TILES))
            else:
                self.set_tile(x, y1, 1)

        # Bottom wall
        for x in range(x1, x2 + 1):
            if chaotic_random_tiles:
                self.set_tile(x, y2, self.rng.randint(1, VALID_TILE_TYPES))
            elif use_random_tiles_type:
                self.set_tile(x, y2, self.rng.choice(BOTTOM_WALL_TILES))
            else:
                self.set_tile(x, y2, 1)

        # Left wall (excluding corners already set)
        for y in range(y1 + 1, y2):
            if chaotic_random_tiles:
                self.set_tile(x1, y, self.rng.randint(1, VALID_TILE_TYPES))
            elif use_random_tiles_type:
                self.set_tile(x1, y, self.rng.choice(LEFT_WALL_TILES))
            else:
                self.set_tile(x1, y, 1)

        # Right wall (excluding corners already set)
        for y in range(y1 + 1, y2):
            if chaotic_random_tiles:
                self.set_tile(x2, y, self.rng.randint(1, VALID_TILE_TYPES))
            elif use_random_tiles_type:
                self.set_tile(x2, y, self.rng.choice(RIGHT_WALL_TILES))
            else:
                self.set_tile(x2, y, 1)

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

    def create_mild_hill(
        self,
        x: int,
        y: int,
        height: int,
        width: int = None,
        min_x: int = None,
        max_x: int = None,
        min_y: int = None,
        max_y: int = None,
    ) -> int:
        """Create a gentle hill using mild slope tiles.

        See terrain_utils.create_mild_hill for full documentation.
        """
        return terrain_utils.create_mild_hill(
            self,
            x,
            y,
            height,
            width,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )

    def create_steep_hill(
        self,
        x: int,
        y: int,
        height: int,
        width: int = None,
        min_x: int = None,
        max_x: int = None,
        min_y: int = None,
        max_y: int = None,
    ) -> int:
        """Create a steep hill using steep slope tiles.

        See terrain_utils.create_steep_hill for full documentation.
        """
        return terrain_utils.create_steep_hill(
            self,
            x,
            y,
            height,
            width,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )

    def create_45_degree_hill(
        self,
        x: int,
        y: int,
        height: int,
        min_x: int = None,
        max_x: int = None,
        min_y: int = None,
        max_y: int = None,
    ) -> int:
        """Create a sharp 45-degree hill.

        See terrain_utils.create_45_degree_hill for full documentation.
        """
        return terrain_utils.create_45_degree_hill(
            self, x, y, height, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y
        )

    def create_mixed_hill(
        self,
        x: int,
        y: int,
        height: int,
        ascent_type: str = "mild",
        descent_type: str = "mild",
        min_x: int = None,
        max_x: int = None,
        min_y: int = None,
        max_y: int = None,
    ) -> int:
        """Create a hill with different slope types for ascent and descent.

        See terrain_utils.create_mixed_hill for full documentation.
        """
        return terrain_utils.create_mixed_hill(
            self,
            x,
            y,
            height,
            ascent_type,
            descent_type,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )

    def to_ascii(self, show_coords: bool = True) -> str:
        """Generate ASCII visualization of this map for debugging.

        Args:
            show_coords: Whether to show coordinate labels

        Returns:
            String containing ASCII art representation of the map
        """
        from .map_visualizer import visualize_map

        return visualize_map(self.map_data(), show_coords=show_coords)
