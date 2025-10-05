"""Switch puzzle generation module for N++ levels."""

from .map import Map
from typing import Optional, List, Tuple, Dict, Literal
from .constants import VALID_TILE_TYPES
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class SwitchPuzzleGenerator(Map):
    """Generates switch puzzle N++ levels with locked doors.

    Creates single-chamber levels where the player must navigate through
    locked doors by collecting switches in the correct order. Supports
    simple to complex puzzles with varying numbers of locked doors and
    dependencies.
    """

    # Chamber dimensions
    MIN_WIDTH = 8
    MAX_WIDTH = 18
    MIN_HEIGHT = 4
    MAX_HEIGHT = 8

    # Puzzle complexity (number of outcroppings)
    MIN_OUTCROPPINGS = 1
    MAX_OUTCROPPINGS = 5

    def __init__(self, seed: Optional[int] = None):
        """Initialize the switch puzzle generator.

        Args:
            seed: Random seed for reproducible generation
        """
        super().__init__(seed)
        self.locked_doors: List[Dict] = []  # Track locked doors and their switches
        self.outcroppings: List[Dict] = []  # Track outcropping data

    def _create_chamber(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Create a rectangular chamber with walls.

        Args:
            x1, y1: Top-left corner
            x2, y2: Bottom-right corner
        """
        # Create empty space
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                self.set_tile(x, y, 0)

        # Create boundary walls
        for x in range(x1, x2 + 1):
            self.set_tile(x, y2 + 1, 1)  # Floor
            self.set_tile(x, y1 - 1, 1)  # Ceiling

        for y in range(y1, y2 + 1):
            self.set_tile(x1 - 1, y, 1)  # Left wall
            self.set_tile(x2 + 1, y, 1)  # Right wall

    def _create_vertical_outcropping(
        self,
        chamber_x1: int,
        chamber_y1: int,
        chamber_x2: int,
        chamber_y2: int,
        direction: Literal["up", "down"],
        tunnel_depth: int,
    ) -> Dict:
        """Create a vertical hollow tunnel extending up or down from chamber.

        Args:
            chamber_x1, chamber_y1, chamber_x2, chamber_y2: Chamber bounds
            direction: "up" extends upward from ceiling, "down" extends downward from floor
            tunnel_depth: How far tunnel extends (2-4 tiles)

        Returns:
            Dict with outcropping data including entrance position and bounds
        """
        # Tunnel is always exactly 1 tile wide
        # Choose random X position for tunnel along chamber width
        tunnel_x = self.rng.randint(chamber_x1 + 1, chamber_x2 - 1)

        if direction == "up":
            # Tunnel extends upward from ceiling
            tunnel_y1 = max(1, chamber_y1 - tunnel_depth)
            tunnel_y2 = chamber_y1 - 1
            # Door should be placed just inside the chamber at the ceiling
            entrance_x = tunnel_x
            entrance_y = chamber_y1 - 1  # First tile of the tunnel
        else:
            # Tunnel extends downward from floor
            tunnel_y1 = chamber_y2 + 1
            tunnel_y2 = min(MAP_TILE_HEIGHT - 2, chamber_y2 + tunnel_depth)
            # Door should be placed just inside the chamber at the floor
            entrance_x = tunnel_x
            entrance_y = chamber_y2 + 1  # First tile of the tunnel

        # Create hollow tunnel (1 tile wide)
        for y in range(tunnel_y1, tunnel_y2 + 1):
            self.set_tile(tunnel_x, y, 0)

        # Create walls around tunnel
        for y in range(tunnel_y1, tunnel_y2 + 1):
            self.set_tile(tunnel_x - 1, y, 1)  # Left wall
            self.set_tile(tunnel_x + 1, y, 1)  # Right wall

        if direction == "up":
            # Close top of tunnel
            for x in range(tunnel_x - 1, tunnel_x + 2):
                self.set_tile(x, tunnel_y1 - 1, 1)
        else:
            # Close bottom of tunnel
            for x in range(tunnel_x - 1, tunnel_x + 2):
                self.set_tile(x, tunnel_y2 + 1, 1)

        # Interior positions exclude the entrance tile (where door will be placed)
        interior_positions = [
            (tunnel_x, y) for y in range(tunnel_y1, tunnel_y2 + 1) if y != entrance_y
        ]

        return {
            "type": "vertical",
            "direction": direction,
            "x1": tunnel_x,
            "y1": tunnel_y1,
            "x2": tunnel_x,
            "y2": tunnel_y2,
            "entrance_x": entrance_x,
            "entrance_y": entrance_y,
            "interior_positions": interior_positions,
        }

    def _create_horizontal_outcropping(
        self,
        chamber_x1: int,
        chamber_y1: int,
        chamber_x2: int,
        chamber_y2: int,
        direction: Literal["left", "right"],
        tunnel_depth: int,
    ) -> Dict:
        """Create a horizontal hollow tunnel extending left or right from chamber.

        Args:
            chamber_x1, chamber_y1, chamber_x2, chamber_y2: Chamber bounds
            direction: "left" extends left from left wall, "right" extends right from right wall
            tunnel_depth: How far tunnel extends (2-4 tiles)

        Returns:
            Dict with outcropping data including entrance position and bounds
        """
        # Tunnel is always exactly 1 tile tall
        # Choose random Y position for tunnel along chamber height
        tunnel_y = self.rng.randint(chamber_y1 + 1, chamber_y2 - 1)

        if direction == "left":
            # Tunnel extends left from left wall
            tunnel_x1 = max(1, chamber_x1 - tunnel_depth)
            tunnel_x2 = chamber_x1 - 1
            # Door should be placed just inside the chamber at the left wall
            entrance_x = chamber_x1 - 1  # First tile of the tunnel
            entrance_y = tunnel_y
        else:
            # Tunnel extends right from right wall
            tunnel_x1 = chamber_x2 + 1
            tunnel_x2 = min(MAP_TILE_WIDTH - 2, chamber_x2 + tunnel_depth)
            # Door should be placed just inside the chamber at the right wall
            entrance_x = chamber_x2 + 1  # First tile of the tunnel
            entrance_y = tunnel_y

        # Create hollow tunnel (1 tile tall)
        for x in range(tunnel_x1, tunnel_x2 + 1):
            self.set_tile(x, tunnel_y, 0)

        # Create walls around tunnel
        for x in range(tunnel_x1, tunnel_x2 + 1):
            self.set_tile(x, tunnel_y - 1, 1)  # Top wall
            self.set_tile(x, tunnel_y + 1, 1)  # Bottom wall

        if direction == "left":
            # Close left end of tunnel
            for y in range(tunnel_y - 1, tunnel_y + 2):
                self.set_tile(tunnel_x1 - 1, y, 1)
        else:
            # Close right end of tunnel
            for y in range(tunnel_y - 1, tunnel_y + 2):
                self.set_tile(tunnel_x2 + 1, y, 1)

        # Interior positions exclude the entrance tile (where door will be placed)
        interior_positions = [
            (x, tunnel_y) for x in range(tunnel_x1, tunnel_x2 + 1) if x != entrance_x
        ]

        return {
            "type": "horizontal",
            "direction": direction,
            "x1": tunnel_x1,
            "y1": tunnel_y,
            "x2": tunnel_x2,
            "y2": tunnel_y,
            "entrance_x": entrance_x,
            "entrance_y": entrance_y,
            "interior_positions": interior_positions,
        }

    def _place_locked_door_at_entrance(
        self, outcropping: Dict, switch_x: int, switch_y: int
    ) -> None:
        """Place a locked door at the entrance of an outcropping.

        For vertical outcroppings: use horizontal door (orientation 2)
        For horizontal outcroppings: use vertical door (orientation 0)

        Args:
            outcropping: Outcropping dict with entrance position
            switch_x: X position of switch
            switch_y: Y position of switch
        """
        door_x = outcropping["entrance_x"]
        door_y = outcropping["entrance_y"]

        if outcropping["type"] == "vertical":
            # Horizontal door blocks vertical tunnel
            orientation = 2
        else:
            # Vertical door blocks horizontal tunnel
            orientation = 0

        self.add_entity(6, door_x, door_y, orientation, 0, switch_x, switch_y)
        self.locked_doors.append(
            {
                "door_x": door_x,
                "door_y": door_y,
                "switch_x": switch_x,
                "switch_y": switch_y,
                "orientation": orientation,
                "outcropping_index": len(self.outcroppings) - 1,
            }
        )

    def _place_ninja_on_floor(self, floor_y: int, x_min: int, x_max: int) -> int:
        """Place ninja on the floor in the left portion of the chamber.

        Args:
            floor_y: Y coordinate of floor
            x_min: Minimum X coordinate
            x_max: Maximum X coordinate

        Returns:
            X coordinate where ninja was placed
        """
        # Place ninja in left quarter of chamber
        spawn_range = max(1, (x_max - x_min) // 4)
        ninja_x = x_min + self.rng.randint(0, spawn_range)
        ninja_orientation = self.rng.choice([1, -1])

        self.set_ninja_spawn(ninja_x, floor_y, ninja_orientation)
        return ninja_x

    def _get_accessible_outcroppings(
        self, chamber_positions: List[Tuple[int, int]]
    ) -> List[int]:
        """Determine which outcroppings are accessible from chamber (no locked doors).

        Args:
            chamber_positions: List of positions in main chamber

        Returns:
            List of outcropping indices that are accessible
        """
        accessible = []
        for i, outcropping in enumerate(self.outcroppings):
            # Check if this outcropping has a locked door
            has_locked_door = any(
                door["outcropping_index"] == i for door in self.locked_doors
            )
            if not has_locked_door:
                accessible.append(i)
        return accessible

    def generate(self, seed: Optional[int] = None) -> Map:
        """Generate a complete switch puzzle level.

        This method creates a single-chamber level with hollow tunnel outcroppings
        that extend from the chamber. Each outcropping is barricaded by a locked door,
        and switches are placed to create dependency puzzles.

        Steps:
        1. Create main chamber
        2. Place ninja on floor
        3. Create 1-5 outcroppings (hollow tunnels extending from chamber)
        4. Place locked doors at outcropping entrances
        5. Place exit door and/or switch in outcroppings
        6. Create switch dependencies (some switches in other outcroppings)
        7. Ensure at least one outcropping is immediately accessible
        8. Add random entities outside playspace

        Args:
            seed: Random seed for reproducible generation

        Returns:
            Map: A Map instance with the generated puzzle
        """
        if seed is not None:
            self.rng.seed(seed)

        self.reset()
        self.locked_doors.clear()
        self.outcroppings.clear()

        # Generate chamber dimensions
        width = self.rng.randint(self.MIN_WIDTH, self.MAX_WIDTH)
        height = self.rng.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)

        # Calculate chamber position
        max_start_x = MAP_TILE_WIDTH - width - 4
        max_start_y = MAP_TILE_HEIGHT - height - 4

        chamber_x1 = self.rng.randint(3, max(4, max_start_x))
        chamber_y1 = self.rng.randint(3, max(4, max_start_y))
        chamber_x2 = min(chamber_x1 + width - 1, MAP_TILE_WIDTH - 4)
        chamber_y2 = min(chamber_y1 + height - 1, MAP_TILE_HEIGHT - 4)

        floor_y = chamber_y2

        # Pre-generate background tiles
        choice = self.rng.randint(0, 2)
        if choice == 0:
            tile_types = [
                self.rng.randint(0, VALID_TILE_TYPES)
                for _ in range(MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
            ]
        elif choice == 1:
            tile_types = [1] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
        else:
            tile_types = [0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
        self.set_tiles_bulk(tile_types)

        # Create main chamber
        self._create_chamber(chamber_x1, chamber_y1, chamber_x2, chamber_y2)

        # Place ninja on floor
        ninja_x = self._place_ninja_on_floor(floor_y, chamber_x1, chamber_x2)

        # Determine number of outcroppings (puzzle complexity)
        num_outcroppings = self.rng.randint(
            self.MIN_OUTCROPPINGS, self.MAX_OUTCROPPINGS
        )

        # Create outcroppings (hollow tunnels extending from chamber)
        outcropping_directions = ["up", "down", "left", "right"]
        self.rng.shuffle(outcropping_directions)

        for i in range(num_outcroppings):
            direction = outcropping_directions[i % len(outcropping_directions)]

            # Tunnel depth (how far it extends)
            tunnel_depth = self.rng.randint(2, 4)

            if direction in ["up", "down"]:
                # Vertical outcropping (1 tile wide)
                outcrop = self._create_vertical_outcropping(
                    chamber_x1,
                    chamber_y1,
                    chamber_x2,
                    chamber_y2,
                    direction,
                    tunnel_depth,
                )
            else:
                # Horizontal outcropping (1 tile tall)
                outcrop = self._create_horizontal_outcropping(
                    chamber_x1,
                    chamber_y1,
                    chamber_x2,
                    chamber_y2,
                    direction,
                    tunnel_depth,
                )

            self.outcroppings.append(outcrop)

        # Choose outcroppings for exit door and switch
        # At least one of them should be in an outcropping
        if num_outcroppings >= 2:
            # Both in outcroppings (different ones)
            exit_door_outcrop_idx = self.rng.randint(0, num_outcroppings - 1)
            available_switch_outcrops = [
                i for i in range(num_outcroppings) if i != exit_door_outcrop_idx
            ]
            exit_switch_outcrop_idx = (
                self.rng.choice(available_switch_outcrops)
                if available_switch_outcrops
                else exit_door_outcrop_idx
            )
        else:
            # Only one outcropping - both go in it or one in chamber
            exit_door_outcrop_idx = 0
            exit_switch_outcrop_idx = None if self.rng.choice([True, False]) else 0

        # Place exit door in its designated outcropping
        exit_door_outcrop = self.outcroppings[exit_door_outcrop_idx]
        exit_door_positions = exit_door_outcrop["interior_positions"]
        door_x, door_y = self.rng.choice(exit_door_positions)
        door_orientation = 0

        # Place exit switch
        if exit_switch_outcrop_idx is not None:
            # Switch in another outcropping
            exit_switch_outcrop = self.outcroppings[exit_switch_outcrop_idx]
            exit_switch_positions = exit_switch_outcrop["interior_positions"]
            switch_x, switch_y = self.rng.choice(exit_switch_positions)
        else:
            # Switch in main chamber
            switch_x = self.rng.randint(chamber_x1 + 1, chamber_x2 - 1)
            switch_y = floor_y

        # Add exit door and switch
        self.add_entity(3, door_x, door_y, door_orientation, 0, switch_x, switch_y)

        # Create switch dependency puzzle
        # Determine which outcroppings get locked doors
        # Strategy: At least one outcropping must be accessible (no locked door)
        # This ensures solvability

        # Collect available positions for switches (chamber + all outcroppings)
        chamber_positions = [
            (x, floor_y)
            for x in range(chamber_x1 + 1, chamber_x2)
            if abs(x - ninja_x) > 1 and (x, floor_y) != (switch_x, switch_y)
        ]

        all_switch_positions = chamber_positions.copy()
        for outcrop in self.outcroppings:
            all_switch_positions.extend(outcrop["interior_positions"])

        # Remove exit door and switch positions
        all_switch_positions = [
            pos
            for pos in all_switch_positions
            if pos != (door_x, door_y) and pos != (switch_x, switch_y)
        ]

        # Decide how many outcroppings to lock (at least 1, at most all but one)
        num_to_lock = min(num_outcroppings - 1, self.rng.randint(1, num_outcroppings))

        # Mark one outcropping as always accessible (starting point)
        accessible_outcrop_idx = self.rng.randint(0, num_outcroppings - 1)

        # Place locked doors on outcroppings
        for i, outcrop in enumerate(self.outcroppings):
            if i == accessible_outcrop_idx:
                # This outcropping is accessible - no locked door
                continue

            if num_to_lock > 0:
                # Place a locked door at the entrance
                # Choose switch position from available positions
                if all_switch_positions:
                    switch_pos_idx = self.rng.randint(0, len(all_switch_positions) - 1)
                    sw_x, sw_y = all_switch_positions.pop(switch_pos_idx)
                else:
                    # Fallback: place in chamber
                    sw_x = self.rng.randint(chamber_x1 + 1, chamber_x2 - 1)
                    sw_y = floor_y

                self._place_locked_door_at_entrance(outcrop, sw_x, sw_y)
                num_to_lock -= 1

        # Add random entities outside playspace
        playspace = (chamber_x1 - 4, chamber_y1 - 4, chamber_x2 + 4, chamber_y2 + 4)
        self.add_random_entities_outside_playspace(
            playspace[0], playspace[1], playspace[2], playspace[3]
        )

        return self
