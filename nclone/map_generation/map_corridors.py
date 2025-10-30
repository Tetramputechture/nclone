"""Corridors map generation module for N++ levels.

Generates levels with alternating horizontal and vertical corridors that create
a zig-zag navigation pattern. Supports optional drop-down connections between
horizontal sections for more complex routing.
"""

from .map import Map
from typing import Optional, List, Tuple
from .constants import VALID_TILE_TYPES
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class CorridorSegment:
    """Represents a single corridor segment in the map."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        orientation: str,
        direction: str = "right",
    ):
        """Initialize a corridor segment.

        Args:
            x: Starting x coordinate
            y: Starting y coordinate
            width: Width in tiles
            height: Height in tiles
            orientation: "horizontal" or "vertical"
            direction: "right", "left", "up", or "down"
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.orientation = orientation
        self.direction = direction

    def get_start_point(self) -> Tuple[int, int]:
        """Get the start point of the corridor."""
        if self.orientation == "horizontal":
            if self.direction == "right":
                return (self.x, self.y + self.height // 2)
            else:  # left
                return (self.x + self.width - 1, self.y + self.height // 2)
        else:  # vertical
            # For vertical corridors, place connection on the side at a reachable height
            # Use the side of the corridor (not center) so ninja can reach it
            side_x = self.x if self.width <= 2 else self.x + 1
            if self.direction == "down":
                # Start near top but not at ceiling (3 tiles down from top)
                return (side_x, self.y + min(3, self.height // 3))
            else:  # up
                # Start at bottom
                return (side_x, self.y + self.height - 1)

    def get_end_point(self) -> Tuple[int, int]:
        """Get the end point of the corridor."""
        if self.orientation == "horizontal":
            if self.direction == "right":
                return (self.x + self.width - 1, self.y + self.height // 2)
            else:  # left
                return (self.x, self.y + self.height // 2)
        else:  # vertical
            # For vertical corridors, place connection on the side at a reachable height
            # Use the side of the corridor (not center) so ninja can reach it
            side_x = (
                self.x + self.width - 1 if self.width <= 2 else self.x + self.width - 2
            )
            if self.direction == "down":
                # End at bottom
                return (side_x, self.y + self.height - 1)
            else:  # up
                # End near top but not at ceiling (3 tiles down from top)
                return (side_x, self.y + min(3, self.height // 3))


class MapCorridors(Map):
    """Generates N++ levels with alternating horizontal and vertical corridors."""

    # Number of corridors
    MIN_CORRIDORS = 2
    MAX_CORRIDORS = 5

    # Horizontal corridor dimensions
    MIN_HORIZONTAL_WIDTH = 8
    MAX_HORIZONTAL_WIDTH = 25
    MIN_HORIZONTAL_HEIGHT = 1
    MAX_HORIZONTAL_HEIGHT = 3

    # Vertical corridor dimensions
    MIN_VERTICAL_WIDTH = 1
    MAX_VERTICAL_WIDTH = 6
    MIN_VERTICAL_HEIGHT = 6
    MAX_VERTICAL_HEIGHT = 15

    # Optional features
    ADD_ZIG_ZAG_DROPS = False
    ADD_CHAOTIC_WALLS = False

    def __init__(self, seed: Optional[int] = None):
        """Initialize the corridors generator.

        Args:
            seed: Random seed for reproducible generation
        """
        super().__init__(seed)
        self.corridors: List[CorridorSegment] = []

    def generate(
        self,
        seed: Optional[int] = None,
        num_corridors: Optional[int] = None,
        add_zig_zag_drops: Optional[bool] = None,
        max_horizontal_height: Optional[int] = None,
        max_vertical_width: Optional[int] = None,
        add_chaotic_walls: Optional[bool] = None,
    ) -> Map:
        """Generate a corridors level with alternating horizontal and vertical sections.

        Args:
            seed: Random seed for reproducible generation
            num_corridors: Number of corridor segments to generate (defaults to class attribute)
            add_zig_zag_drops: Enable zig-zag drop patterns (defaults to class attribute)
            max_horizontal_height: Override max height for horizontal corridors (defaults to class attribute)
            max_vertical_width: Override max width for vertical corridors (defaults to class attribute)

        Returns:
            Map: A Map instance with the generated level
        """
        if seed is not None:
            self.rng.seed(seed)

        self.reset()
        self.corridors.clear()

        # Use class attributes as defaults if parameters not provided
        if num_corridors is None:
            num_corridors = self.rng.randint(self.MIN_CORRIDORS, self.MAX_CORRIDORS)
        if add_zig_zag_drops is None:
            add_zig_zag_drops = self.ADD_ZIG_ZAG_DROPS
        if max_horizontal_height is None:
            max_horizontal_height = self.MAX_HORIZONTAL_HEIGHT
        if max_vertical_width is None:
            max_vertical_width = self.MAX_VERTICAL_WIDTH
        if add_chaotic_walls is None:
            add_chaotic_walls = self.ADD_CHAOTIC_WALLS

        # Randomly decide if zig-zag drops should be applied this generation
        use_zig_zag = add_zig_zag_drops and self.rng.choice([True, False])

        # Fill background with solid tiles or random solid tiles
        # Never use empty (0) tiles for background to avoid confusion
        choice = self.rng.randint(0, 1)
        if choice == 0:
            # Random solid tiles (1-VALID_TILE_TYPES)
            tile_types = [
                self.rng.randint(1, VALID_TILE_TYPES)
                for _ in range(MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
            ]
        else:
            # All solid walls (1)
            tile_types = [1] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
        self.set_tiles_bulk(tile_types)

        # Randomly choose starting direction
        start_horizontal = self.rng.choice([True, False])

        # Generate corridors with alternating orientations
        for i in range(num_corridors):
            is_horizontal = (i % 2 == 0) == start_horizontal

            if i == 0:
                # First corridor - place near the start of the map
                corridor = self._create_first_corridor(
                    is_horizontal, max_horizontal_height, max_vertical_width
                )
            else:
                # Subsequent corridors - connect to the previous one
                prev_corridor = self.corridors[-1]
                corridor = self._create_next_corridor(
                    prev_corridor,
                    is_horizontal,
                    use_zig_zag and i > 1,
                    max_horizontal_height,
                    max_vertical_width,
                )

            if corridor:
                self.corridors.append(corridor)
                self._carve_corridor_interior(corridor)

        # Connect corridors with transition sections
        for i in range(len(self.corridors) - 1):
            self._connect_corridors(self.corridors[i], self.corridors[i + 1])

        # Add boundary walls around all corridors
        self._add_boundary_walls()

        # Place entities
        self._place_entities()

        # Add random entities outside playspace
        self._add_random_entities_outside()

        return self

    def _create_first_corridor(
        self, is_horizontal: bool, max_horizontal_height: int, max_vertical_width: int
    ) -> CorridorSegment:
        """Create the first corridor segment near the start of the map.

        Args:
            is_horizontal: Whether this should be a horizontal corridor
            max_horizontal_height: Maximum height for horizontal corridors
            max_vertical_width: Maximum width for vertical corridors

        Returns:
            CorridorSegment: The created corridor
        """
        if is_horizontal:
            width = self.rng.randint(
                self.MIN_HORIZONTAL_WIDTH, self.MAX_HORIZONTAL_WIDTH
            )
            height = self.rng.randint(self.MIN_HORIZONTAL_HEIGHT, max_horizontal_height)
            # Start near the left side
            x = self.rng.randint(2, 5)
            y = self.rng.randint(2, MAP_TILE_HEIGHT - height - 2)
            direction = "right"
        else:
            width = self.rng.randint(self.MIN_VERTICAL_WIDTH, max_vertical_width)
            height = self.rng.randint(
                self.MIN_VERTICAL_HEIGHT, self.MAX_VERTICAL_HEIGHT
            )
            # Start near the bottom
            x = self.rng.randint(2, MAP_TILE_WIDTH - width - 2)
            y = MAP_TILE_HEIGHT - height - self.rng.randint(2, 4)
            direction = "up"

        return CorridorSegment(
            x,
            y,
            width,
            height,
            "horizontal" if is_horizontal else "vertical",
            direction,
        )

    def _create_next_corridor(
        self,
        prev_corridor: CorridorSegment,
        is_horizontal: bool,
        allow_zig_zag: bool,
        max_horizontal_height: int,
        max_vertical_width: int,
    ) -> Optional[CorridorSegment]:
        """Create the next corridor segment connected to the previous one.

        Args:
            prev_corridor: The previous corridor segment
            is_horizontal: Whether this should be a horizontal corridor
            allow_zig_zag: Whether to allow zig-zag drop patterns
            max_horizontal_height: Maximum height for horizontal corridors
            max_vertical_width: Maximum width for vertical corridors

        Returns:
            CorridorSegment or None: The created corridor, or None if placement failed
        """
        prev_end = prev_corridor.get_end_point()

        if is_horizontal:
            width = self.rng.randint(
                self.MIN_HORIZONTAL_WIDTH, self.MAX_HORIZONTAL_WIDTH
            )
            height = self.rng.randint(self.MIN_HORIZONTAL_HEIGHT, max_horizontal_height)

            # Decide direction based on previous corridor and zig-zag setting
            if allow_zig_zag and prev_corridor.orientation == "vertical":
                # For zig-zag, alternate horizontal direction
                if (
                    len(self.corridors) >= 2
                    and self.corridors[-2].orientation == "horizontal"
                ):
                    # Reverse direction from two corridors ago
                    prev_horizontal_dir = self.corridors[-2].direction
                    direction = "left" if prev_horizontal_dir == "right" else "right"
                else:
                    direction = self.rng.choice(["right", "left"])
            else:
                # Default: continue in a consistent direction
                direction = "right" if prev_end[0] < MAP_TILE_WIDTH // 2 else "left"

            # Position the corridor
            if direction == "right":
                x = prev_end[0]
                y = prev_end[1] - height // 2
            else:
                x = max(2, prev_end[0] - width + 1)
                y = prev_end[1] - height // 2

            # Clamp to map bounds
            x = max(2, min(x, MAP_TILE_WIDTH - width - 2))
            y = max(2, min(y, MAP_TILE_HEIGHT - height - 2))

        else:  # vertical
            width = self.rng.randint(self.MIN_VERTICAL_WIDTH, max_vertical_width)
            height = self.rng.randint(
                self.MIN_VERTICAL_HEIGHT, self.MAX_VERTICAL_HEIGHT
            )

            # Determine vertical direction based on position and space
            if prev_end[1] < MAP_TILE_HEIGHT // 2:
                direction = "down"
                x = prev_end[0] - width // 2
                y = prev_end[1]
            else:
                direction = "up"
                x = prev_end[0] - width // 2
                y = max(2, prev_end[1] - height + 1)

            # Clamp to map bounds
            x = max(2, min(x, MAP_TILE_WIDTH - width - 2))
            y = max(2, min(y, MAP_TILE_HEIGHT - height - 2))

        return CorridorSegment(
            x,
            y,
            width,
            height,
            "horizontal" if is_horizontal else "vertical",
            direction,
        )

    def _carve_corridor_interior(self, corridor: CorridorSegment) -> None:
        """Carve out a corridor interior without adding walls.

        Args:
            corridor: The corridor segment to carve
        """
        # Carve out the interior
        for y in range(corridor.y, corridor.y + corridor.height):
            for x in range(corridor.x, corridor.x + corridor.width):
                if 0 <= x < MAP_TILE_WIDTH and 0 <= y < MAP_TILE_HEIGHT:
                    self.set_tile(x, y, 0)

    def _connect_corridors(
        self, corridor1: CorridorSegment, corridor2: CorridorSegment
    ) -> None:
        """Create a connection between two corridor segments.

        The connection width/height respects the dimensions of both corridors.
        For example, a height-1 horizontal corridor connecting to a width-1
        vertical corridor will have a 1-tile connection.

        Args:
            corridor1: The first corridor
            corridor2: The second corridor
        """
        end_point = corridor1.get_end_point()
        start_point = corridor2.get_start_point()

        # Calculate appropriate connection dimensions based on corridor sizes
        # The connection should be as wide/tall as the smaller corridor dimension
        if (
            corridor1.orientation == "horizontal"
            and corridor2.orientation == "vertical"
        ):
            # Horizontal to vertical: connection width is min of both
            connection_width = min(corridor1.height, corridor2.width)
        elif (
            corridor1.orientation == "vertical"
            and corridor2.orientation == "horizontal"
        ):
            # Vertical to horizontal: connection width is min of both
            connection_width = min(corridor1.width, corridor2.height)
        else:
            # Same orientation (shouldn't happen with alternating, but handle it)
            if corridor1.orientation == "horizontal":
                connection_width = min(corridor1.height, corridor2.height)
            else:
                connection_width = min(corridor1.width, corridor2.width)

        # Ensure connection is at least 1 tile
        connection_width = max(1, connection_width)

        # Calculate the offset for centering the connection
        offset = connection_width // 2

        # Clear connection area at both endpoints
        for point in [end_point, start_point]:
            for dy in range(-offset, -offset + connection_width):
                for dx in range(-offset, -offset + connection_width):
                    nx, ny = point[0] + dx, point[1] + dy
                    if 0 <= nx < MAP_TILE_WIDTH and 0 <= ny < MAP_TILE_HEIGHT:
                        self.set_tile(nx, ny, 0)

        # Create a path of empty tiles between the corridors
        current_x, current_y = end_point
        target_x, target_y = start_point

        # Simple pathfinding: move horizontally first, then vertically
        while current_x != target_x or current_y != target_y:
            # Clear a corridor with width matching connection_width
            for dy in range(-offset, -offset + connection_width):
                for dx in range(-offset, -offset + connection_width):
                    nx, ny = current_x + dx, current_y + dy
                    if 0 <= nx < MAP_TILE_WIDTH and 0 <= ny < MAP_TILE_HEIGHT:
                        self.set_tile(nx, ny, 0)

            # Move towards target
            if current_x != target_x:
                current_x += 1 if current_x < target_x else -1
            elif current_y != target_y:
                current_y += 1 if current_y < target_y else -1

    def _add_boundary_walls(self) -> None:
        """Add walls around all empty corridor tiles to create proper boundaries."""
        # Find all empty tiles that are part of the corridor system
        empty_tiles = set()
        for y in range(MAP_TILE_HEIGHT):
            for x in range(MAP_TILE_WIDTH):
                if self.get_tile(x, y) == 0:
                    empty_tiles.add((x, y))

        # For each empty tile, check its neighbors and add walls where needed
        use_random_tiles = self.ADD_CHAOTIC_WALLS and self.rng.choice([True, False])
        walls_to_add = set()

        for x, y in empty_tiles:
            # Check all 8 neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < MAP_TILE_WIDTH and 0 <= ny < MAP_TILE_HEIGHT:
                        # If neighbor is not part of corridor system, mark it for wall
                        if (nx, ny) not in empty_tiles:
                            walls_to_add.add((nx, ny))

        # Add all the walls
        for x, y in walls_to_add:
            if use_random_tiles:
                self.set_tile(x, y, self.rng.randint(1, VALID_TILE_TYPES))
            else:
                self.set_tile(x, y, 1)

    def _place_entities(self) -> None:
        """Place ninja spawn, exit door, and switch in the corridors."""
        if not self.corridors:
            return

        # Place ninja in the first corridor
        first_corridor = self.corridors[0]
        start_point = first_corridor.get_start_point()

        # Find a good position on the floor of the first corridor
        ninja_x = start_point[0]
        ninja_y = first_corridor.y + first_corridor.height - 1

        # Ensure ninja is on an empty tile
        ninja_x, ninja_y = self._find_closest_valid_tile(ninja_x, ninja_y, tile_type=0)

        # Set ninja orientation based on first corridor direction
        if first_corridor.orientation == "horizontal":
            ninja_orientation = 1 if first_corridor.direction == "right" else -1
        else:
            ninja_orientation = self.rng.choice([1, -1])

        self.set_ninja_spawn(ninja_x, ninja_y, ninja_orientation)

        # Place exit door and switch in the last corridor
        last_corridor = self.corridors[-1]
        end_point = last_corridor.get_end_point()

        # Find positions for door and switch
        if last_corridor.orientation == "horizontal":
            door_x = end_point[0]
            door_y = last_corridor.y + last_corridor.height - 1
            switch_x = max(last_corridor.x, door_x - self.rng.randint(2, 4))
            switch_y = door_y
        else:
            door_x = last_corridor.x + last_corridor.width // 2
            door_y = end_point[1]
            switch_x = door_x
            switch_y = min(
                last_corridor.y + last_corridor.height - 1,
                door_y + self.rng.randint(2, 4),
            )

        # Ensure positions are valid
        door_x, door_y = self._find_closest_valid_tile(door_x, door_y, tile_type=0)
        switch_x, switch_y = self._find_closest_valid_tile(
            switch_x, switch_y, tile_type=0
        )

        # Add exit door and switch
        self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)

    def _add_random_entities_outside(self) -> None:
        """Add random entities outside the corridor playspace."""
        if not self.corridors:
            return

        # Calculate the bounding box of all corridors
        min_x = min(c.x for c in self.corridors) - 2
        min_y = min(c.y for c in self.corridors) - 2
        max_x = max(c.x + c.width for c in self.corridors) + 2
        max_y = max(c.y + c.height for c in self.corridors) + 2

        self.add_random_entities_outside_playspace(min_x, min_y, max_x, max_y)
