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
    MAX_MINES = -1

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

        # CRITICAL: Remove ninja's exact position from candidates to prevent overlap
        valid_tiles = [tile for tile in valid_tiles if tile != ninja_pos]

        if len(valid_tiles) < 2:
            # Fallback: place at opposite end from ninja
            if ninja_pos[1] == chamber_y:
                # Ninja at top, place door/switch at bottom
                door_pos = (chamber_x + width // 2, chamber_y + height - 1)
                switch_pos = (chamber_x + width // 2, chamber_y + height - 2)
            else:
                # Ninja at bottom, place door/switch at top
                door_pos = (chamber_x + width // 2, chamber_y)
                switch_pos = (chamber_x + width // 2, chamber_y + 1)
            return door_pos, switch_pos

        best_door_pos = None
        best_switch_pos = None
        best_total_distance = -1

        # Use exhaustive search for small chambers, sampling for large ones
        max_combinations = 500  # Limit to avoid performance issues

        # For small chambers, try all combinations
        if len(valid_tiles) * (len(valid_tiles) - 1) <= max_combinations:
            # Exhaustive search: try all pairs of positions
            for door_candidate in valid_tiles:
                # Check minimum distance from ninja to door
                ninja_door_dist = self._calculate_chebyshev_distance(
                    ninja_pos, door_candidate
                )

                for switch_candidate in valid_tiles:
                    if switch_candidate == door_candidate:
                        continue

                    # Check minimum distances
                    ninja_switch_dist = self._calculate_chebyshev_distance(
                        ninja_pos, switch_candidate
                    )
                    door_switch_dist = self._calculate_chebyshev_distance(
                        door_candidate, switch_candidate
                    )

                    # Skip if any pair is too close
                    if (
                        ninja_door_dist < min_distance_tiles
                        or ninja_switch_dist < min_distance_tiles
                        or door_switch_dist < min_distance_tiles
                    ):
                        continue

                    # Calculate total pairwise distance
                    total_distance = (
                        ninja_door_dist + ninja_switch_dist + door_switch_dist
                    )

                    if total_distance > best_total_distance:
                        best_total_distance = total_distance
                        best_door_pos = door_candidate
                        best_switch_pos = switch_candidate
        else:
            # For large chambers, use smart sampling
            # Sample door positions uniformly across the chamber
            num_door_samples = min(50, len(valid_tiles))
            door_candidates = self.rng.sample(valid_tiles, num_door_samples)

            for door_candidate in door_candidates:
                ninja_door_dist = self._calculate_chebyshev_distance(
                    ninja_pos, door_candidate
                )

                if ninja_door_dist < min_distance_tiles:
                    continue

                # Sample switch positions
                num_switch_samples = min(20, len(valid_tiles) - 1)
                switch_candidates = [t for t in valid_tiles if t != door_candidate]
                if len(switch_candidates) > num_switch_samples:
                    switch_candidates = self.rng.sample(
                        switch_candidates, num_switch_samples
                    )

                for switch_candidate in switch_candidates:
                    ninja_switch_dist = self._calculate_chebyshev_distance(
                        ninja_pos, switch_candidate
                    )
                    door_switch_dist = self._calculate_chebyshev_distance(
                        door_candidate, switch_candidate
                    )

                    # Skip if any pair is too close
                    if (
                        ninja_switch_dist < min_distance_tiles
                        or door_switch_dist < min_distance_tiles
                    ):
                        continue

                    total_distance = (
                        ninja_door_dist + ninja_switch_dist + door_switch_dist
                    )

                    if total_distance > best_total_distance:
                        best_total_distance = total_distance
                        best_door_pos = door_candidate
                        best_switch_pos = switch_candidate

        # If no valid combination found with min_distance, try with reduced distance
        if best_door_pos is None or best_switch_pos is None:
            # Relax constraint and try again with distance of 2
            for door_candidate in valid_tiles:
                ninja_door_dist = self._calculate_chebyshev_distance(
                    ninja_pos, door_candidate
                )

                if ninja_door_dist < 2:
                    continue

                for switch_candidate in valid_tiles:
                    if switch_candidate == door_candidate:
                        continue

                    ninja_switch_dist = self._calculate_chebyshev_distance(
                        ninja_pos, switch_candidate
                    )
                    door_switch_dist = self._calculate_chebyshev_distance(
                        door_candidate, switch_candidate
                    )

                    if ninja_switch_dist < 2 or door_switch_dist < 2:
                        continue

                    total_distance = (
                        ninja_door_dist + ninja_switch_dist + door_switch_dist
                    )

                    if total_distance > best_total_distance:
                        best_total_distance = total_distance
                        best_door_pos = door_candidate
                        best_switch_pos = switch_candidate

        # Final fallback if still no valid combination
        if best_door_pos is None or best_switch_pos is None:
            # Use top and bottom of chamber
            if ninja_pos[1] == chamber_y:
                # Ninja at top, place door/switch at bottom
                door_pos = (chamber_x + width // 2, chamber_y + height - 1)
                switch_pos = (chamber_x + width // 2, chamber_y + height - 2)
            else:
                # Ninja at bottom, place door/switch at top
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
        should_place_mines = add_wall_mines and width > 1
        if should_place_mines:
            self._place_wall_mines(chamber_x, chamber_y, width, height)

        # Step 7: Place entities
        # Ninja can spawn at top or bottom of corridor (50/50 chance)
        spawn_at_top = self.rng.choice([True, False])

        if spawn_at_top:
            # Ninja at top of corridor
            ninja_x = chamber_x + self.rng.randint(0, max(0, width - 1))
            ninja_y = chamber_y
        else:
            # Ninja at bottom on ground floor at random X position
            ninja_x = chamber_x + self.rng.randint(0, max(0, width - 1))
            ninja_y = chamber_y + height - 1

        ninja_orientation = self.rng.choice([1, -1])

        # ensure ninja is on an empty tile
        ninja_x, ninja_y = self._find_closest_valid_tile(ninja_x, ninja_y, tile_type=0)
        ninja_pos = (ninja_x, ninja_y)

        # Find optimal positions for door and switch that maximize distances
        # Default minimum distance: 3 tiles (72 pixels)
        door_pos, switch_pos = self._find_best_door_switch_positions(
            ninja_pos, chamber_x, chamber_y, width, height
        )

        door_x, door_y = door_pos
        switch_x, switch_y = switch_pos

        # Place entities (door and switch positions are already optimized)
        self.set_ninja_spawn(ninja_x, ninja_y, ninja_orientation)
        self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)

        # Add ceiling mines at the top of the corridor
        if self.ADD_BOUNDARY_MINES:
            self._place_corridor_ceiling_mines(
                chamber_x,
                chamber_y,
                width,
                height,
                "vertical",
                ninja_x,
                ninja_y,
                max_mines=self.MAX_MINES,
            )

        # Add floor mines if corridor is tall enough
        if self.ADD_BOUNDARY_MINES:
            self._place_corridor_floor_mines(
                chamber_x,
                chamber_y,
                width,
                height,
                ninja_x,
                ninja_y,
                max_mines=self.MAX_MINES,
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

        # Calculate valid Y range for mine placement
        # Mines are placed in the air (empty space) next to the walls
        min_y = chamber_y + start_offset + 0.1
        max_y = chamber_y + height

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
