"""Vertical corridor map generation module for N++ levels.

Generates levels with narrow vertical chambers where the player must navigate
upward from the bottom to reach the exit at the top. Walls may have mines
placed at regular intervals to increase difficulty.
"""

from .map import Map
from typing import Optional
from .constants import VALID_TILE_TYPES
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class MapVerticalCorridor(Map):
    """Generates N++ levels with narrow vertical corridors."""

    # Chamber dimension constraints
    MIN_WIDTH = 1
    MAX_WIDTH = 6
    MIN_HEIGHT = 8
    MAX_HEIGHT = 22

    # Mine placement constraints
    MIN_MINE_SPACING = 2  # Minimum vertical spacing between mines (tiles)
    MAX_MINE_SPACING = 5  # Maximum vertical spacing between mines (tiles)

    # Optional features
    ADD_PLATFORMS = False
    ADD_MID_MINES = False

    def generate(
        self,
        seed: Optional[int] = None,
        swap_top_and_bottom: bool = False,
        door_at_top: bool = False,
        width: Optional[int] = None,
        height: Optional[int] = None,
        add_platforms: Optional[bool] = None,
        add_mid_mines: Optional[bool] = None,
    ) -> Map:
        """Generate a vertical corridor level with exit at the top.

        Args:
            seed: Random seed for reproducible generation
            swap_top_and_bottom: If True, swap the top and bottom of the corridor, so that the
                exit is at the bottom and the ninja starts at the top. Defaults to False.
            door_at_top: If True, the exit door will always be at the topmost tile of the corridor.
                Defaults to False. Respects swap_top_and_bottom.
            width: Width of the corridor. Defaults to None, in which case a random width is chosen.
            height: Height of the corridor. Defaults to None, in which case a random height is chosen.
            add_platforms: Add 1-2 horizontal platforms jutting from walls (only if width > 1, defaults to class attribute).
            add_mid_mines: Place mines at mid-height floating positions (defaults to class attribute).

        Returns:
            Map: A Map instance with the generated level
        """
        if seed is not None:
            self.rng.seed(seed)

        self.reset()

        # Use class attributes as defaults if parameters not provided
        if add_platforms is None:
            add_platforms = self.ADD_PLATFORMS
        if add_mid_mines is None:
            add_mid_mines = self.ADD_MID_MINES

        # Step 1: Determine chamber dimensions
        if width is None:
            width = self.rng.randint(self.MIN_WIDTH, self.MAX_WIDTH)
        if height is None:
            height = self.rng.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)

        # Step 2: Calculate chamber position (centered with some randomness)
        max_start_x = MAP_TILE_WIDTH - width - 2
        max_start_y = MAP_TILE_HEIGHT - height - 2
        chamber_x = self.rng.randint(2, max(2, max_start_x))
        chamber_y = self.rng.randint(2, max(2, max_start_y))

        tile_types = [
            self.rng.randint(0, VALID_TILE_TYPES)
            for _ in range(MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
        ]
        self.set_tiles_bulk(tile_types)

        # Step 4: Carve out empty chamber
        for y in range(chamber_y, chamber_y + height):
            for x in range(chamber_x, chamber_x + width):
                self.set_tile(x, y, 0)

        # Step 5: Create boundary walls
        use_random_tiles = self.rng.choice([True, False])
        self.set_hollow_rectangle(
            chamber_x - 1,
            chamber_y - 1,
            chamber_x + width,
            chamber_y + height,
            use_random_tiles_type=use_random_tiles,
        )

        # Step 6: Place mines on walls
        should_place_mines = width > 1 and self.rng.choice([True, False])
        if should_place_mines:
            self._place_wall_mines(chamber_x, chamber_y, width, height)

        # Step 7: Place entities
        # Ninja at bottom on ground floor at random X position
        ninja_x = chamber_x + self.rng.randint(0, max(0, width - 2))
        ninja_y = chamber_y + height - 1
        ninja_orientation = self.rng.choice([1, -1])

        # ensure ninja is on an empty tile
        ninja_x, ninja_y = self._find_closest_valid_tile(ninja_x, ninja_y, tile_type=0)

        # Exit switch and exit door at top, next to each other
        # Place them side by side at a random height between midpoint and top
        if width >= 2:
            # If width allows, place them next to each other
            switch_x = chamber_x + self.rng.randint(0, width - 2)
            door_x = switch_x + 1
        else:
            # If width is 1, place them at the same X coordinate
            switch_x = chamber_x
            door_x = chamber_x

        # Random Y position between top and midpoint of chamber
        midpoint_y = chamber_y + height // 2
        if door_at_top:
            door_y = chamber_y
        else:
            door_y = self.rng.randint(chamber_y, midpoint_y)
        switch_y = self.rng.randint(chamber_y, door_y)

        # If swapping  top and bottom, the exit door and ninja positions need to be swapped
        if swap_top_and_bottom:
            tmp_door_y = door_y
            door_y = ninja_y
            ninja_y = tmp_door_y

        self.set_ninja_spawn(ninja_x, ninja_y, ninja_orientation)
        self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)

        # Add optional platforms and mines
        if add_platforms and width > 1:
            # Add 1-2 horizontal platforms jutting from walls
            num_platforms = self.rng.randint(1, 2)
            # Place platforms in middle third of corridor height
            min_plat_y = chamber_y + height // 3
            max_plat_y = chamber_y + 2 * height // 3

            for _ in range(num_platforms):
                plat_y = self.rng.randint(min_plat_y, max_plat_y)
                # Choose which wall to jut from
                from_left = self.rng.choice([True, False])

                if from_left:
                    # Platform jutting from left wall
                    plat_x = chamber_x
                    tile_type = self.rng.randint(1, 33)
                    self.set_tile(plat_x, plat_y, tile_type)
                else:
                    # Platform jutting from right wall
                    plat_x = chamber_x + width - 1
                    tile_type = self.rng.randint(1, 33)
                    self.set_tile(plat_x, plat_y, tile_type)

        if add_mid_mines:
            # Place 2-4 mines floating in middle of corridor at various heights
            num_mines = self.rng.randint(2, min(4, height - 4))
            # Distribute mines across the height
            mine_spacing = height // (num_mines + 1)

            for i in range(num_mines):
                mine_y = chamber_y + (i + 1) * mine_spacing
                # Place in middle of corridor width
                if width == 1:
                    mine_x = chamber_x
                else:
                    mine_x = chamber_x + width // 2

                mine_type = self.rng.choice([1, 21])
                self.add_entity(mine_type, mine_x, mine_y)

        return self

    def _place_wall_mines(
        self, chamber_x: int, chamber_y: int, width: int, height: int
    ) -> None:
        """Place mines on the left and right walls of the chamber.

        Mines are placed at regular intervals (2-5 tiles apart vertically).

        Args:
            chamber_x: Starting x coordinate of the chamber
            chamber_y: Starting y coordinate of the chamber
            width: Width of the chamber
            height: Height of the chamber
        """
        # Determine vertical spacing between mines
        mine_spacing = self.rng.randint(self.MIN_MINE_SPACING, self.MAX_MINE_SPACING)

        # Decide which walls to place mines on
        place_on_left = self.rng.choice([True, False])
        place_on_right = self.rng.choice([True, False])

        # Ensure at least one wall has mines if we decided to place mines
        if not place_on_left and not place_on_right:
            if self.rng.choice([True, False]):
                place_on_left = True
            else:
                place_on_right = True

        # Start placing mines from the bottom, leaving some space from the floor
        # and ceiling
        start_offset = self.rng.randint(1, min(3, mine_spacing))
        end_offset = 2  # Leave space near the top where exit is

        # Calculate valid Y range for mine placement
        # Mines are placed in the air (empty space) next to the walls
        min_y = chamber_y + start_offset
        max_y = chamber_y + height - end_offset

        # Place mines on left wall
        if place_on_left and width >= 1:
            mine_x = chamber_x - 1  # Position mine in wall/air boundary
            current_y = min_y
            while current_y <= max_y:
                # Convert to grid position for mine placement
                self.add_entity(1, mine_x, current_y)  # Type 1 = toggle mine
                current_y += mine_spacing

        # Place mines on right wall
        if place_on_right and width >= 1:
            mine_x = chamber_x + width + 1  # Position mine in wall/air boundary
            current_y = min_y
            while current_y <= max_y:
                # Convert to grid position for mine placement
                self.add_entity(1, mine_x, current_y)  # Type 1 = toggle mine
                current_y += mine_spacing
