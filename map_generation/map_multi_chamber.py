"""Multi-chamber generation module for N++ levels."""

from map_generation.map import Map
import random
from typing import List, Tuple, Optional, Set
from map_generation.constants import VALID_TILE_TYPES, NINJA_SPAWN_OFFSET_PX, EXIT_DOOR_OFFSET_PX, SWITCH_OFFSET_PX, GOLD_OFFSET_PX


class Chamber:
    """Represents a single chamber in the multi-chamber map."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        # IDs of chambers this is connected to
        self.connected_to: Set[int] = set()
        self.has_ninja = False
        self.has_exit = False
        self.has_switch = False
        self.gold_count = 0

    def overlaps(self, other: 'Chamber', padding: int = 1) -> bool:
        """Check if this chamber overlaps with another, including padding."""
        return not (
            self.x + self.width + padding < other.x or
            other.x + other.width + padding < self.x or
            self.y + self.height + padding < other.y or
            other.y + other.height + padding < self.y
        )

    def get_random_wall_point(self, rng: random.Random, side: str) -> Tuple[int, int]:
        """Get a random point on the specified wall of the chamber that is guaranteed to be accessible."""
        if side == 'left':
            return (self.x, self.y + rng.randint(1, self.height - 2))
        elif side == 'right':
            return (self.x + self.width - 1, self.y + rng.randint(1, self.height - 2))
        elif side == 'top':
            return (self.x + rng.randint(1, self.width - 2), self.y)
        else:  # bottom
            return (self.x + rng.randint(1, self.width - 2), self.y + self.height - 1)


class MultiChamberGenerator(Map):
    """Generates N++ levels with multiple connected chambers."""

    # Chamber generation constants
    MIN_CHAMBERS = 2
    MAX_CHAMBERS = 4
    MIN_CHAMBER_WIDTH = 4
    MAX_CHAMBER_WIDTH = 10
    MIN_CHAMBER_HEIGHT = 3
    MAX_CHAMBER_HEIGHT = 6
    MIN_CORRIDOR_LENGTH = 1
    MAX_CORRIDOR_LENGTH = 5
    MIN_CORRIDOR_WIDTH = 1
    MAX_CORRIDOR_WIDTH = 3
    MAX_GOLD_PER_CHAMBER = 0

    def __init__(self, seed: Optional[int] = None):
        """Initialize the multi-chamber generator.

        Args:
            seed: Random seed for reproducible generation
        """
        super().__init__()
        self.rng = random.Random(seed)
        self.chambers: List[Chamber] = []

    def _try_place_chamber(self) -> Optional[Chamber]:
        """Attempt to place a new chamber without overlapping existing ones."""
        width = self.rng.randint(
            self.MIN_CHAMBER_WIDTH, self.MAX_CHAMBER_WIDTH)
        height = self.rng.randint(
            self.MIN_CHAMBER_HEIGHT, self.MAX_CHAMBER_HEIGHT)

        # Try 50 random positions
        for _ in range(50):
            x = self.rng.randint(1, self.MAP_WIDTH - width - 1)
            y = self.rng.randint(1, self.MAP_HEIGHT - height - 1)

            new_chamber = Chamber(x, y, width, height)

            # Check for overlaps with existing chambers
            if not any(new_chamber.overlaps(chamber, padding=2) for chamber in self.chambers):
                return new_chamber

        return None

    def _connect_chambers(self, chamber1: Chamber, chamber2: Chamber):
        """Create a corridor connecting two chambers."""
        # Choose random points on the walls of each chamber
        sides = ['left', 'right', 'top', 'bottom']

        # Keep trying different connection points until we find a valid path
        for _ in range(50):  # Limit attempts to avoid infinite loops
            side1 = self.rng.choice(sides)
            side2 = self.rng.choice(sides)

            # Get connection points that are guaranteed to be inside the chambers
            if side1 == 'left':
                start_x = chamber1.x
                start_y = chamber1.y + self.rng.randint(1, chamber1.height - 2)
            elif side1 == 'right':
                start_x = chamber1.x + chamber1.width - 1
                start_y = chamber1.y + self.rng.randint(1, chamber1.height - 2)
            elif side1 == 'top':
                start_x = chamber1.x + self.rng.randint(1, chamber1.width - 2)
                start_y = chamber1.y
            else:  # bottom
                start_x = chamber1.x + self.rng.randint(1, chamber1.width - 2)
                start_y = chamber1.y + chamber1.height - 1

            if side2 == 'left':
                end_x = chamber2.x
                end_y = chamber2.y + self.rng.randint(1, chamber2.height - 2)
            elif side2 == 'right':
                end_x = chamber2.x + chamber2.width - 1
                end_y = chamber2.y + self.rng.randint(1, chamber2.height - 2)
            elif side2 == 'top':
                end_x = chamber2.x + self.rng.randint(1, chamber2.width - 2)
                end_y = chamber2.y
            else:  # bottom
                end_x = chamber2.x + self.rng.randint(1, chamber2.width - 2)
                end_y = chamber2.y + chamber2.height - 1

            # Determine corridor width (at least 2 to ensure connectivity)
            corridor_width = max(2, self.rng.randint(
                self.MIN_CORRIDOR_WIDTH, self.MAX_CORRIDOR_WIDTH))

            # Create corridor with proper spacing
            # First extend straight out from the first chamber
            extension1 = 2  # Extend at least 2 tiles out from chamber
            if side1 == 'left':
                for x in range(start_x - extension1, start_x + 2):
                    for y in range(start_y - corridor_width, start_y + corridor_width + 1):
                        if 0 <= x < self.MAP_WIDTH and 0 <= y < self.MAP_HEIGHT:
                            self.set_tile(x, y, 0)
                start_x -= extension1
            elif side1 == 'right':
                for x in range(start_x - 1, start_x + extension1 + 1):
                    for y in range(start_y - corridor_width, start_y + corridor_width + 1):
                        if 0 <= x < self.MAP_WIDTH and 0 <= y < self.MAP_HEIGHT:
                            self.set_tile(x, y, 0)
                start_x += extension1
            elif side1 == 'top':
                for y in range(start_y - extension1, start_y + 2):
                    for x in range(start_x - corridor_width, start_x + corridor_width + 1):
                        if 0 <= x < self.MAP_WIDTH and 0 <= y < self.MAP_HEIGHT:
                            self.set_tile(x, y, 0)
                start_y -= extension1
            else:  # bottom
                for y in range(start_y - 1, start_y + extension1 + 1):
                    for x in range(start_x - corridor_width, start_x + corridor_width + 1):
                        if 0 <= x < self.MAP_WIDTH and 0 <= y < self.MAP_HEIGHT:
                            self.set_tile(x, y, 0)
                start_y += extension1

            # Then extend straight out from the second chamber
            extension2 = 2  # Extend at least 2 tiles out from chamber
            if side2 == 'left':
                for x in range(end_x - extension2, end_x + 2):
                    for y in range(end_y - corridor_width, end_y + corridor_width + 1):
                        if 0 <= x < self.MAP_WIDTH and 0 <= y < self.MAP_HEIGHT:
                            self.set_tile(x, y, 0)
                end_x -= extension2
            elif side2 == 'right':
                for x in range(end_x - 1, end_x + extension2 + 1):
                    for y in range(end_y - corridor_width, end_y + corridor_width + 1):
                        if 0 <= x < self.MAP_WIDTH and 0 <= y < self.MAP_HEIGHT:
                            self.set_tile(x, y, 0)
                end_x += extension2
            elif side2 == 'top':
                for y in range(end_y - extension2, end_y + 2):
                    for x in range(end_x - corridor_width, end_x + corridor_width + 1):
                        if 0 <= x < self.MAP_WIDTH and 0 <= y < self.MAP_HEIGHT:
                            self.set_tile(x, y, 0)
                end_y -= extension2
            else:  # bottom
                for y in range(end_y - 1, end_y + extension2 + 1):
                    for x in range(end_x - corridor_width, end_x + corridor_width + 1):
                        if 0 <= x < self.MAP_WIDTH and 0 <= y < self.MAP_HEIGHT:
                            self.set_tile(x, y, 0)
                end_y += extension2

            # Now connect the extended points with a wide path
            # Calculate midpoint for L-shaped connection
            mid_x = start_x + (end_x - start_x) // 2

            # Create horizontal connection
            min_x = min(start_x, mid_x)
            max_x = max(start_x, mid_x)
            for x in range(min_x, max_x + 1):
                for y in range(start_y - corridor_width, start_y + corridor_width + 1):
                    if 0 <= x < self.MAP_WIDTH and 0 <= y < self.MAP_HEIGHT:
                        self.set_tile(x, y, 0)

            # Create vertical connection
            min_y = min(start_y, end_y)
            max_y = max(start_y, end_y)
            for y in range(min_y, max_y + 1):
                for x in range(mid_x - corridor_width, mid_x + corridor_width + 1):
                    if 0 <= x < self.MAP_WIDTH and 0 <= y < self.MAP_HEIGHT:
                        self.set_tile(x, y, 0)

            # Create horizontal connection to end point
            min_x = min(mid_x, end_x)
            max_x = max(mid_x, end_x)
            for x in range(min_x, max_x + 1):
                for y in range(end_y - corridor_width, end_y + corridor_width + 1):
                    if 0 <= x < self.MAP_WIDTH and 0 <= y < self.MAP_HEIGHT:
                        self.set_tile(x, y, 0)

            # If we got here, we successfully created a corridor
            return

        # If we failed to create a corridor after 50 attempts, raise an error
        raise RuntimeError("Failed to connect chambers after 50 attempts")

    def _carve_chamber(self, chamber: Chamber):
        """Carve out the space for a chamber."""
        for y in range(chamber.y, chamber.y + chamber.height):
            for x in range(chamber.x, chamber.x + chamber.width):
                self.set_tile(x, y, 0)

    def _add_walls(self):
        """Add walls around all empty spaces while preserving corridors."""
        # First pass: Mark all empty spaces that are part of corridors or chambers
        walkable = set()
        for y in range(self.MAP_HEIGHT):
            for x in range(self.MAP_WIDTH):
                if self.tile_data[x + y * self.MAP_WIDTH] == 0:
                    walkable.add((x, y))

        # Second pass: Add walls only where they won't block paths
        for y in range(self.MAP_HEIGHT):
            for x in range(self.MAP_WIDTH):
                if (x, y) not in walkable:  # Only consider non-walkable tiles for walls
                    # Check if this position is adjacent to a walkable tile
                    adjacent_to_walkable = False
                    would_block_path = False

                    # Count walkable neighbors
                    walkable_neighbors = 0
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) in walkable:
                            walkable_neighbors += 1
                            adjacent_to_walkable = True

                    # If this position would create a diagonal wall between walkable spaces,
                    # it might block a path
                    for dx, dy in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) in walkable:
                            # Check if adding a wall here would create a diagonal barrier
                            if ((x + dx, y) in walkable and (x, y + dy) in walkable):
                                would_block_path = True
                                break

                    # Add wall only if it's adjacent to walkable space and won't block paths
                    if adjacent_to_walkable and not would_block_path:
                        self.set_tile(x, y, 1)

    def _place_entities(self):
        """Place ninja spawn, exit door, switch, and gold in the chambers."""
        # Choose random chambers for ninja and exit
        ninja_chamber = self.rng.choice(self.chambers)
        ninja_chamber.has_ninja = True

        # Choose a different chamber for exit
        available_chambers = [c for c in self.chambers if not c.has_ninja]
        exit_chamber = self.rng.choice(available_chambers)
        exit_chamber.has_exit = True

        # Choose a different chamber for switch
        available_chambers = [
            c for c in self.chambers if not c.has_ninja and not c.has_exit]
        switch_chamber = self.rng.choice(available_chambers or [exit_chamber])
        switch_chamber.has_switch = True

        # Place ninja on the floor
        ninja_x = ninja_chamber.x + \
            self.rng.randint(1, ninja_chamber.width - 2)
        # Place ninja one tile above the floor
        ninja_y = ninja_chamber.y + ninja_chamber.height - 1
        ninja_orientation = self.rng.choice([-1, 1])
        self.set_ninja_spawn(ninja_x, ninja_y, ninja_orientation)

        # Place a solid tile under ninja spawn so we always spawn grounded
        self.set_tile(ninja_x, ninja_y + 1, 1)

        # Place exit door and switch
        exit_x = exit_chamber.x + self.rng.randint(1, exit_chamber.width - 2)
        exit_y = exit_chamber.y + self.rng.randint(1, exit_chamber.height - 2)
        switch_x = switch_chamber.x + \
            self.rng.randint(1, switch_chamber.width - 2)
        switch_y = switch_chamber.y + \
            self.rng.randint(1, switch_chamber.height - 2)

        # Ensure switch is not on top of exit
        while switch_x == exit_x and switch_y == exit_y:
            switch_x = switch_chamber.x + \
                self.rng.randint(1, switch_chamber.width - 2)
            switch_y = switch_chamber.y + \
                self.rng.randint(1, switch_chamber.height - 2)

        self.add_entity(3, exit_x, exit_y, 0, 0, switch_x, switch_y)

        # Place solid tile under exit so its always available,
        # but not on top of the switch
        if exit_y + 1 != switch_y and exit_x != switch_x:
            self.set_tile(exit_x, exit_y + 1, 1)

        # Place solid tile under switch so its always available,
        # but not on top of the exit
        if switch_y + 1 != exit_y or switch_x != exit_x:
            self.set_tile(switch_x, switch_y + 1, 1)

        # Add gold to random chambers
        if self.MAX_GOLD_PER_CHAMBER > 0:
            for chamber in self.chambers:
                if self.rng.random() < 0.7:  # 70% chance for gold in each chamber
                    gold_count = self.rng.randint(
                        1, self.MAX_GOLD_PER_CHAMBER)
                chamber.gold_count = gold_count

                for _ in range(gold_count):
                    gold_x = chamber.x + self.rng.randint(1, chamber.width - 2)
                    gold_y = chamber.y + \
                        self.rng.randint(1, chamber.height - 2)
                    self.add_entity(2, gold_x + 2, gold_y + 2, 0, 0)

    def generate(self, seed: Optional[int] = None) -> Map:
        """Generate a multi-chamber level.

        Args:
            seed: Random seed for reproducible generation

        Returns:
            Map: A Map instance with the generated level
        """
        if seed is not None:
            self.rng.seed(seed)

        self.reset()
        self.chambers.clear()

        # Fill the map with random tiles
        tile_types = [self.rng.randint(0, VALID_TILE_TYPES)
                      for _ in range(self.MAP_WIDTH * self.MAP_HEIGHT)]
        self.set_tiles_bulk(tile_types)

        # Generate chambers
        num_chambers = self.rng.randint(self.MIN_CHAMBERS, self.MAX_CHAMBERS)
        while len(self.chambers) < num_chambers:
            if chamber := self._try_place_chamber():
                self.chambers.append(chamber)

        # Carve out chambers
        for chamber in self.chambers:
            self._carve_chamber(chamber)

        # Connect chambers
        unconnected = set(range(len(self.chambers)))
        connected = {0}
        unconnected.remove(0)

        while unconnected:
            chamber1_idx = self.rng.choice(list(connected))
            chamber2_idx = self.rng.choice(list(unconnected))

            chamber1 = self.chambers[chamber1_idx]
            chamber2 = self.chambers[chamber2_idx]

            self._connect_chambers(chamber1, chamber2)
            chamber1.connected_to.add(chamber2_idx)
            chamber2.connected_to.add(chamber1_idx)

            connected.add(chamber2_idx)
            unconnected.remove(chamber2_idx)

        # Add some random additional connections for variety
        for _ in range(self.rng.randint(0, 2)):
            chamber1_idx = self.rng.randint(0, len(self.chambers) - 1)
            chamber2_idx = self.rng.randint(0, len(self.chambers) - 1)
            if (chamber1_idx != chamber2_idx and
                    chamber2_idx not in self.chambers[chamber1_idx].connected_to):
                self._connect_chambers(
                    self.chambers[chamber1_idx], self.chambers[chamber2_idx])
                self.chambers[chamber1_idx].connected_to.add(chamber2_idx)
                self.chambers[chamber2_idx].connected_to.add(chamber1_idx)

        # Add walls around empty spaces
        self._add_walls()

        # Place entities
        self._place_entities()

        return self
