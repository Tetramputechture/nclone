"""Vertical corridor map generation module for N++ levels.

Generates levels with narrow vertical chambers where the player must navigate
upward from the bottom to reach the exit at the top. Walls may have mines
placed at regular intervals to increase difficulty.
"""

from .map import Map
from typing import Optional, Tuple, List
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
    ADD_CHAOTIC_WALLS = False
    ADD_WALL_MINES = False
    ADD_BOUNDARY_MINES = False

    def _calculate_chebyshev_distance(
        self, pos1: Tuple[int, int], pos2: Tuple[int, int]
    ) -> int:
        """Calculate Chebyshev distance between two tile positions.

        Args:
            pos1: First position (x, y) in tiles
            pos2: Second position (x, y) in tiles

        Returns:
            Chebyshev distance in tiles (max of x and y differences)
        """
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return max(dx, dy)

    def _get_valid_empty_tiles(
        self, chamber_x: int, chamber_y: int, width: int, height: int
    ) -> List[Tuple[int, int]]:
        """Get all valid empty tile positions in the chamber.

        Args:
            chamber_x, chamber_y: Chamber start position
            width, height: Chamber dimensions

        Returns:
            List of (x, y) tuples for all empty tiles in the chamber
        """
        valid_tiles = []
        for y in range(chamber_y, chamber_y + height):
            for x in range(chamber_x, chamber_x + width):
                if self.get_tile(x, y) == 0:
                    valid_tiles.append((x, y))
        return valid_tiles

    def _find_best_door_switch_positions(
        self,
        ninja_pos: Tuple[int, int],
        chamber_x: int,
        chamber_y: int,
        width: int,
        height: int,
        min_distance_tiles: int = 1,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Find optimal positions for door and switch that maximize distances.

        Ensures door and switch are at least min_distance_tiles away from ninja
        and from each other, while maximizing the sum of all pairwise distances.

        Args:
            ninja_pos: Ninja position (x, y) in tiles
            chamber_x, chamber_y: Chamber start position
            width, height: Chamber dimensions
            min_distance_tiles: Minimum distance in tiles (24 pixels = 1 tile)

        Returns:
            Tuple of (door_pos, switch_pos) where each is (x, y)
        """
        valid_tiles = self._get_valid_empty_tiles(chamber_x, chamber_y, width, height)

        if len(valid_tiles) < 2:
            # Fallback: use top of chamber
            door_pos = (chamber_x + width // 2, chamber_y)
            switch_pos = (chamber_x + width // 2, chamber_y + 1)
            return door_pos, switch_pos

        # Filter tiles that meet minimum distance from ninja
        candidate_tiles = [
            tile
            for tile in valid_tiles
            if self._calculate_chebyshev_distance(tile, ninja_pos) >= min_distance_tiles
        ]

        if len(candidate_tiles) < 2:
            # Not enough candidates, use all valid tiles
            candidate_tiles = valid_tiles

        best_door_pos = None
        best_switch_pos = None
        best_total_distance = -1

        # Try multiple random combinations to find good positions
        num_attempts = min(100, len(candidate_tiles) * 10)

        for _ in range(num_attempts):
            # Randomly select door position
            door_candidate = self.rng.choice(candidate_tiles)

            # Filter switch candidates that meet distance requirements
            switch_candidates = [
                tile
                for tile in candidate_tiles
                if tile != door_candidate
                and self._calculate_chebyshev_distance(tile, door_candidate)
                >= min_distance_tiles
            ]

            if not switch_candidates:
                continue

            # Try a few switch positions for this door
            for _ in range(min(5, len(switch_candidates))):
                switch_candidate = self.rng.choice(switch_candidates)

                # Calculate total pairwise distance
                ninja_door_dist = self._calculate_chebyshev_distance(
                    ninja_pos, door_candidate
                )
                ninja_switch_dist = self._calculate_chebyshev_distance(
                    ninja_pos, switch_candidate
                )
                door_switch_dist = self._calculate_chebyshev_distance(
                    door_candidate, switch_candidate
                )

                total_distance = ninja_door_dist + ninja_switch_dist + door_switch_dist

                if total_distance > best_total_distance:
                    best_total_distance = total_distance
                    best_door_pos = door_candidate
                    best_switch_pos = switch_candidate

        # Fallback if no valid combination found
        if best_door_pos is None or best_switch_pos is None:
            # Use top of chamber as fallback
            door_pos = (chamber_x + width // 2, chamber_y)
            switch_pos = (chamber_x + width // 2, chamber_y + 1)
            return door_pos, switch_pos

        return best_door_pos, best_switch_pos

    def generate(
        self,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        add_platforms: Optional[bool] = None,
        add_mid_mines: Optional[bool] = None,
        add_wall_mines: Optional[bool] = None,
        add_chaotic_walls: Optional[bool] = None,
    ) -> Map:
        """Generate a vertical corridor level with exit at the top.

        Args:
            seed: Random seed for reproducible generation
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
        if add_chaotic_walls is None:
            add_chaotic_walls = self.ADD_CHAOTIC_WALLS
        if add_wall_mines is None:
            add_wall_mines = self.ADD_WALL_MINES

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
        self.set_hollow_rectangle(
            chamber_x - 1,
            chamber_y - 1,
            chamber_x + width,
            chamber_y + height,
            use_random_tiles_type=True,
            chaotic_random_tiles=add_chaotic_walls,
        )

        # Step 6: Place mines on walls
        should_place_mines = (
            add_wall_mines and width > 1 and self.rng.choice([True, False])
        )
        if should_place_mines:
            self._place_wall_mines(chamber_x, chamber_y, width, height)

        # Step 7: Place entities
        # Ninja at bottom on ground floor at random X position
        ninja_x = chamber_x + self.rng.randint(0, max(0, width - 2))
        ninja_y = chamber_y + height - 1
        ninja_orientation = self.rng.choice([1, -1])

        # ensure ninja is on an empty tile
        ninja_x, ninja_y = self._find_closest_valid_tile(ninja_x, ninja_y, tile_type=0)
        ninja_pos = (ninja_x, ninja_y)

        # Find optimal positions for door and switch that maximize distances
        # Minimum distance: 24 pixels = 1 tile
        door_pos, switch_pos = self._find_best_door_switch_positions(
            ninja_pos, chamber_x, chamber_y, width, height, min_distance_tiles=1
        )

        door_x, door_y = door_pos
        switch_x, switch_y = switch_pos

        # Verify minimum distances are met (sanity check)
        ninja_door_dist = self._calculate_chebyshev_distance(ninja_pos, door_pos)
        ninja_switch_dist = self._calculate_chebyshev_distance(ninja_pos, switch_pos)
        door_switch_dist = self._calculate_chebyshev_distance(door_pos, switch_pos)

        # Ensure all distances are at least 1 tile (24 pixels)
        if ninja_door_dist < 1:
            # Move door away from ninja
            dx = door_x - ninja_x
            dy = door_y - ninja_y
            if dx == 0 and dy == 0:
                door_y = ninja_y - 1  # Move up
            else:
                door_x = ninja_x + (1 if dx >= 0 else -1)
                door_y = ninja_y + (1 if dy >= 0 else -1)
            door_x, door_y = self._find_closest_valid_tile(door_x, door_y, tile_type=0)
            door_pos = (door_x, door_y)
            # Recalculate distances after door update
            ninja_door_dist = self._calculate_chebyshev_distance(ninja_pos, door_pos)
            door_switch_dist = self._calculate_chebyshev_distance(door_pos, switch_pos)

        if ninja_switch_dist < 1:
            # Move switch away from ninja
            dx = switch_x - ninja_x
            dy = switch_y - ninja_y
            if dx == 0 and dy == 0:
                switch_y = ninja_y - 1  # Move up
            else:
                switch_x = ninja_x + (1 if dx >= 0 else -1)
                switch_y = ninja_y + (1 if dy >= 0 else -1)
            switch_x, switch_y = self._find_closest_valid_tile(
                switch_x, switch_y, tile_type=0
            )
            switch_pos = (switch_x, switch_y)
            # Recalculate distances after switch update
            ninja_switch_dist = self._calculate_chebyshev_distance(
                ninja_pos, switch_pos
            )
            door_switch_dist = self._calculate_chebyshev_distance(door_pos, switch_pos)

        if door_switch_dist < 1:
            # Move switch away from door
            dx = switch_x - door_x
            dy = switch_y - door_y
            if dx == 0 and dy == 0:
                switch_y = door_y + 1
            else:
                switch_x = door_x + (1 if dx >= 0 else -1)
                switch_y = door_y + (1 if dy >= 0 else -1)
            switch_x, switch_y = self._find_closest_valid_tile(
                switch_x, switch_y, tile_type=0
            )

        # Randomly swap door and switch positions (50% chance)
        swap_door_switch = self.rng.choice([True, False])
        if swap_door_switch:
            door_x, switch_x = switch_x, door_x
            door_y, switch_y = switch_y, door_y

        self.set_ninja_spawn(ninja_x, ninja_y, ninja_orientation)
        self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)

        # Add ceiling mines at the top of the corridor
        if self.ADD_BOUNDARY_MINES:
            self._place_corridor_ceiling_mines(
                chamber_x, chamber_y, width, height, "vertical", ninja_x, ninja_y
            )

        # Add floor mines if corridor is tall enough
        if self.ADD_BOUNDARY_MINES:
            self._place_corridor_floor_mines(
                chamber_x, chamber_y, width, height, "vertical", ninja_x, ninja_y
            )

        can_add_platforms = height > 8 and width > 2

        # Add optional platforms and mines
        if can_add_platforms and add_platforms and width > 1:
            # Add 1-2 horizontal platforms jutting from walls
            num_platforms = self.rng.randint(1, max(2, height // 3))
            # Place platforms in middle third of corridor height
            min_plat_y = chamber_y + height // 3
            max_plat_y = chamber_y + 2 * height // 3

            # platforms should not be placed on the same y position as the switch,
            # door, ninja, or any other platform += 1 vertical
            invalid_positions = [switch_y, door_y, ninja_y]

            for _ in range(num_platforms):
                placement_attempts = 0
                while placement_attempts < 10:
                    plat_y = self.rng.randint(min_plat_y, max_plat_y)
                    if plat_y not in invalid_positions:
                        break
                    placement_attempts += 1

                if plat_y not in invalid_positions:
                    invalid_positions += [plat_y + 1, plat_y, plat_y - 1]
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
                mine_x = chamber_x + width // 2
                mine_type = self.rng.choice([1, 21])
                self.add_entity(mine_type, mine_x, mine_y)

        # add random entities outside the playspace
        playspace = (
            chamber_x - 1,
            chamber_y - 1,
            chamber_x + width + 1,
            chamber_y + height + 1,
        )
        self.add_random_entities_outside_playspace(
            playspace[0], playspace[1], playspace[2], playspace[3]
        )

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
            mine_x = chamber_x + 1  # Position mine in wall/air boundary
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
