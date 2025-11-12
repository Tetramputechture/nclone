"""Horizontal corridor map generator for N++ levels.

This generator creates simple horizontal corridors with minimal obstacles.
These are among the simplest level types in the game.
"""

from typing import Optional, Tuple, List
import numpy as np

from .map import Map
from .constants import VALID_TILE_TYPES, GRID_SIZE_FACTOR
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT, TILE_PIXEL_SIZE


class MapHorizontalCorridor(Map):
    """Generates minimal horizontal corridor levels."""

    # Configuration parameters
    MIN_WIDTH = 3
    MAX_WIDTH = 23
    MIN_HEIGHT = 1
    MAX_HEIGHT = 5
    RANDOM_EDGE_TILES = False
    FIXED_HEIGHT = None
    ADD_MINES = False
    MAX_MINES = -1

    def _calculate_pixel_distance(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """Calculate Chebyshev distance in pixels between two positions.

        Args:
            x1, y1: First position (can be fractional tiles)
            x2, y2: Second position (can be fractional tiles)

        Returns:
            Distance in pixels (Chebyshev distance)
        """
        dx_pixels = abs(x1 - x2) * TILE_PIXEL_SIZE
        dy_pixels = abs(y1 - y2) * TILE_PIXEL_SIZE
        return max(dx_pixels, dy_pixels)

    def _find_best_door_switch_positions(
        self,
        ninja_x: int,
        ninja_y: int,
        door_positions: List[float],
        switch_positions: List[float],
        base_y: int,
    ) -> Tuple[float, float]:
        """Find optimal positions for door and switch that maximize distances.

        Args:
            ninja_x, ninja_y: Ninja position
            door_positions: List of candidate door X positions
            switch_positions: List of candidate switch X positions
            base_y: Base Y coordinate for distance calculations

        Returns:
            Tuple of (door_x, switch_x)
        """
        if not door_positions:
            door_positions = switch_positions if switch_positions else [ninja_x + 2]
        if not switch_positions:
            switch_positions = door_positions if door_positions else [ninja_x + 3]

        best_door_x = None
        best_switch_x = None
        best_total_distance = -1

        # Try multiple combinations to maximize distance
        num_attempts = min(50, len(door_positions) * len(switch_positions))

        for _ in range(num_attempts):
            door_candidate = self.rng.choice(door_positions)
            switch_candidate = self.rng.choice(switch_positions)

            # Skip if same position
            if abs(door_candidate - switch_candidate) < 0.01:
                continue

            # Calculate distances
            ninja_door_dist = self._calculate_pixel_distance(
                door_candidate, base_y, ninja_x, ninja_y
            )
            ninja_switch_dist = self._calculate_pixel_distance(
                switch_candidate, base_y, ninja_x, ninja_y
            )
            door_switch_dist = self._calculate_pixel_distance(
                door_candidate, base_y, switch_candidate, base_y
            )

            # Ensure minimum distances
            if (
                ninja_door_dist >= TILE_PIXEL_SIZE
                and ninja_switch_dist >= TILE_PIXEL_SIZE
                and door_switch_dist >= TILE_PIXEL_SIZE
            ):
                total_distance = ninja_door_dist + ninja_switch_dist + door_switch_dist
                if total_distance > best_total_distance:
                    best_total_distance = total_distance
                    best_door_x = door_candidate
                    best_switch_x = switch_candidate

        # Fallback if no valid combination found
        if best_door_x is None or best_switch_x is None:
            # Use positions that are at least 1 tile away from ninja
            valid_doors = [
                d
                for d in door_positions
                if self._calculate_pixel_distance(d, base_y, ninja_x, ninja_y)
                >= TILE_PIXEL_SIZE
            ]
            valid_switches = [
                s
                for s in switch_positions
                if self._calculate_pixel_distance(s, base_y, ninja_x, ninja_y)
                >= TILE_PIXEL_SIZE
            ]

            if valid_doors:
                best_door_x = self.rng.choice(valid_doors)
            else:
                best_door_x = door_positions[0] if door_positions else ninja_x + 2

            if valid_switches:
                # Filter switches that are at least 1 tile from door
                valid_switches = [
                    s
                    for s in valid_switches
                    if self._calculate_pixel_distance(s, base_y, best_door_x, base_y)
                    >= TILE_PIXEL_SIZE
                ]
                if valid_switches:
                    best_switch_x = self.rng.choice(valid_switches)
                else:
                    best_switch_x = (
                        switch_positions[0] if switch_positions else best_door_x + 1
                    )
            else:
                best_switch_x = (
                    switch_positions[0] if switch_positions else best_door_x + 1
                )

        return best_door_x, best_switch_x

    def _adjust_positions_for_min_distance(
        self,
        ninja_x: int,
        ninja_y: int,
        door_x: float,
        door_y: float,
        switch_x: float,
        switch_y: float,
        start_x: int,
        start_y: int,
        width: int,
        height: int,
    ) -> Tuple[float, float, float, float]:
        """Adjust door and switch positions to ensure minimum distances.

        Args:
            ninja_x, ninja_y: Ninja position
            door_x, door_y: Door position
            switch_x, switch_y: Switch position
            start_x, start_y: Corridor start position
            width, height: Corridor dimensions

        Returns:
            Tuple of (door_x, door_y, switch_x, switch_y) adjusted positions
        """
        # Adjust door if too close to ninja
        ninja_door_dist = self._calculate_pixel_distance(
            door_x, door_y, ninja_x, ninja_y
        )
        if ninja_door_dist < TILE_PIXEL_SIZE:
            dx = door_x - ninja_x
            if abs(dx) < 1.0:
                door_x = ninja_x + (1.0 if dx >= 0 else -1.0)
            door_x = max(start_x, min(door_x, start_x + width - 1))

        # Adjust switch if too close to ninja
        ninja_switch_dist = self._calculate_pixel_distance(
            switch_x, switch_y, ninja_x, ninja_y
        )
        if ninja_switch_dist < TILE_PIXEL_SIZE:
            dx = switch_x - ninja_x
            if abs(dx) < 1.0:
                switch_x = ninja_x + (1.0 if dx >= 0 else -1.0)
            switch_x = max(start_x, min(switch_x, start_x + width - 1))

        # Adjust switch if too close to door
        door_switch_dist = self._calculate_pixel_distance(
            door_x, door_y, switch_x, switch_y
        )
        if door_switch_dist < TILE_PIXEL_SIZE:
            dx = switch_x - door_x
            if abs(dx) < 1.0:
                switch_x = door_x + (1.0 if dx >= 0 else -1.0)
            switch_x = max(start_x, min(switch_x, start_x + width - 1))

        return door_x, door_y, switch_x, switch_y

    def _place_uniform_ceiling_mines(
        self,
        start_x: float,
        start_y: float,
        width: float,
        height: float,
        ninja_x: float,
        ninja_y: float,
        exit_switch_x: float,
        num_mines: int,
    ) -> None:
        """Place ceiling mines uniformly with guaranteed coverage between ninja and exit.

        Args:
            start_x: Starting x coordinate of the corridor
            start_y: Starting y coordinate of the corridor
            width: Width of the corridor
            height: Height of the corridor
            ninja_x: X coordinate of ninja spawn
            ninja_y: Y coordinate of ninja spawn
            exit_switch_x: X coordinate of exit switch
            num_mines: Exact number of mines to place
        """
        # Skip ceiling mines for horizontal corridors that are exactly 2 tiles high
        if height == 2:
            return

        if num_mines <= 0:
            return

        # Calculate mine y position (one tile below the top of the corridor)
        mine_y = start_y + 1

        # Ensure mine_y is within map bounds
        if mine_y < 0 or mine_y >= MAP_TILE_HEIGHT:
            return

        # Generate uniform positions across the corridor width
        offset = 1
        x_start = start_x + 0.5 + offset
        x_end = start_x + width - 0.5 + offset

        # Generate uniformly spaced positions
        mine_x_positions = np.linspace(x_start, x_end, num=num_mines).tolist()

        # Filter by distance from ninja spawn
        SPAWN_BUFFER_TILES = 1

        filtered_positions = []
        for mine_x in mine_x_positions:
            distance_to_ninja = (
                (float(mine_x) - float(ninja_x + 1)) ** 2
                + (float(mine_y) - float(ninja_y + 1)) ** 2
            ) ** 0.5

            if distance_to_ninja >= SPAWN_BUFFER_TILES:
                filtered_positions.append(mine_x)

        # Ensure at least one mine between ninja and exit switch
        min_x = min(ninja_x, exit_switch_x)
        max_x = max(ninja_x, exit_switch_x)
        mines_in_range = [x for x in filtered_positions if min_x < x < max_x]

        # If no mines in range OR no filtered positions at all, add one in the middle
        if not mines_in_range or not filtered_positions:
            middle_x = (ninja_x + exit_switch_x) / 2
            # Add a mine at the middle position if it's valid
            distance_to_ninja_middle = (
                (float(middle_x) - float(ninja_x + 1)) ** 2
                + (float(mine_y) - float(ninja_y + 1)) ** 2
            ) ** 0.5
            if distance_to_ninja_middle >= SPAWN_BUFFER_TILES:
                filtered_positions.append(middle_x)
                filtered_positions.sort()
            elif width >= 3:
                # If middle is too close, try positions further from ninja
                for fraction in [0.6, 0.4, 0.7, 0.3]:
                    test_x = ninja_x + fraction * (exit_switch_x - ninja_x)
                    distance_test = (
                        (float(test_x) - float(ninja_x + 1)) ** 2
                        + (float(mine_y) - float(ninja_y + 1)) ** 2
                    ) ** 0.5
                    if (
                        distance_test >= SPAWN_BUFFER_TILES
                        and start_x <= test_x <= start_x + width
                    ):
                        filtered_positions.append(test_x)
                        break

        # Place all filtered mines
        for mine_x in filtered_positions:
            if 0 <= mine_x < MAP_TILE_WIDTH:
                self.add_entity(1, float(mine_x), float(mine_y))

    def _place_uniform_floor_mines(
        self,
        start_x: float,
        start_y: float,
        width: float,
        height: float,
        ninja_x: float,
        ninja_y: float,
        exit_switch_x: float,
        num_mines: int,
    ) -> None:
        """Place floor mines uniformly with guaranteed coverage between ninja and exit.

        Args:
            start_x: Starting x coordinate of the corridor
            start_y: Starting y coordinate of the corridor
            width: Width of the corridor
            height: Height of the corridor
            ninja_x: X coordinate of ninja spawn
            ninja_y: Y coordinate of ninja spawn
            exit_switch_x: X coordinate of exit switch
            num_mines: Exact number of mines to place
        """
        # Only place floor mines if corridor is at least 2 tiles high
        if height < 2:
            return

        if num_mines <= 0:
            return

        # Skip if corridor is too narrow
        if width < 2:
            return

        # Calculate mine y position (floor of the corridor)
        mine_y = start_y + height - 1

        # Ensure mine_y is within map bounds
        if mine_y < 0 or mine_y >= MAP_TILE_HEIGHT:
            return

        # Generate uniform positions with minimum spacing
        MIN_SEPARATION_TILES = 1.25
        max_possible_mines = max(
            1, int((width - MIN_SEPARATION_TILES) / MIN_SEPARATION_TILES) + 1
        )

        # Limit to what's actually possible given spacing constraints
        actual_num_mines = min(num_mines, max_possible_mines)

        # Calculate mine positions with proper spacing
        if actual_num_mines == 1:
            mine_x_positions = [start_x + width / 2]
        elif actual_num_mines > 1:
            usable_width = width - MIN_SEPARATION_TILES
            spacing = (
                usable_width / (actual_num_mines - 1) if actual_num_mines > 1 else 0
            )
            mine_x_positions = [
                start_x + 1 + MIN_SEPARATION_TILES / 2 + i * spacing
                for i in range(actual_num_mines)
            ]

        # Filter by distance from ninja spawn
        SPAWN_BUFFER_TILES = 2

        filtered_positions = []
        for mine_x in mine_x_positions:
            distance_to_ninja = (
                (float(mine_x) - float(ninja_x)) ** 2
                + (float(mine_y) - float(ninja_y)) ** 2
            ) ** 0.5

            if distance_to_ninja >= SPAWN_BUFFER_TILES:
                filtered_positions.append(mine_x)

        # Ensure at least one mine between ninja and exit switch
        min_x = min(ninja_x, exit_switch_x)
        max_x = max(ninja_x, exit_switch_x)
        mines_in_range = [x for x in filtered_positions if min_x < x < max_x]

        # If no mines in range OR no filtered positions at all, try to add one in the middle
        if not mines_in_range or not filtered_positions:
            middle_x = (ninja_x + exit_switch_x) / 2
            distance_to_ninja_middle = (
                (float(middle_x) - float(ninja_x)) ** 2
                + (float(mine_y) - float(ninja_y)) ** 2
            ) ** 0.5
            if distance_to_ninja_middle >= SPAWN_BUFFER_TILES:
                # Check if this position maintains minimum separation from existing mines
                valid_position = True
                for existing_x in filtered_positions:
                    if abs(middle_x - existing_x) < MIN_SEPARATION_TILES:
                        valid_position = False
                        break

                if valid_position:
                    filtered_positions.append(middle_x)
                    filtered_positions.sort()
            elif width >= 3:
                # If middle is too close, try positions further from ninja
                # Try positions at 1/3 and 2/3 of the distance between ninja and exit
                for fraction in [0.6, 0.4, 0.7, 0.3]:
                    test_x = ninja_x + fraction * (exit_switch_x - ninja_x)
                    distance_test = (
                        (float(test_x) - float(ninja_x)) ** 2
                        + (float(mine_y) - float(ninja_y)) ** 2
                    ) ** 0.5
                    if distance_test >= SPAWN_BUFFER_TILES:
                        valid_position = True
                        for existing_x in filtered_positions:
                            if abs(test_x - existing_x) < MIN_SEPARATION_TILES:
                                valid_position = False
                                break

                        if valid_position and start_x <= test_x <= start_x + width:
                            filtered_positions.append(test_x)
                            break

        # Place all filtered mines
        for mine_x in filtered_positions:
            if 0 <= mine_x < MAP_TILE_WIDTH:
                self.add_entity(1, float(mine_x), float(mine_y + height))

    def generate(self, seed: Optional[int] = None) -> "MapHorizontalCorridor":
        """Generate a minimal horizontal corridor level.

        Args:
            seed: Random seed for reproducible generation

        Returns:
            Self for method chaining
        """
        if seed is not None:
            self.rng.seed(seed)

        self.reset()

        # Calculate level index from seed for parameter variation
        index = seed % 100 if seed is not None else 0

        # Determine dimensions based on index
        max_width = self.MIN_WIDTH + (index % 20)
        max_height = self.MIN_HEIGHT + (index % 5)

        width = self.rng.randint(self.MIN_WIDTH, min(max_width, self.MAX_WIDTH))

        if self.FIXED_HEIGHT is not None:
            height = self.FIXED_HEIGHT
        else:
            height = self.rng.randint(self.MIN_HEIGHT, min(max_height, self.MAX_HEIGHT))

        # Random offset for the chamber
        max_start_x = MAP_TILE_WIDTH - width - 1
        max_start_y = MAP_TILE_HEIGHT - height - 1
        start_x = self.rng.randint(1, max_start_x)
        start_y = self.rng.randint(1, max_start_y)

        # Fill with random tiles
        tile_types = [
            self.rng.randint(0, VALID_TILE_TYPES)
            for _ in range(MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
        ]
        self.set_tiles_bulk(tile_types)

        # Create empty chamber
        for y in range(start_y, start_y + height):
            for x in range(start_x, start_x + width):
                self.set_tile(x, y, 0)

        self.set_hollow_rectangle(
            start_x - 1,
            start_y - 1,
            start_x + width,
            start_y + height,
            use_random_tiles_type=True,
            chaotic_random_tiles=self.RANDOM_EDGE_TILES,
        )

        # Randomly choose ninja starting side
        ninja_on_left = self.rng.choice([True, False])

        if ninja_on_left:
            ninja_x = start_x
            ninja_orientation = 1
        else:
            ninja_x = start_x + width - 1
            ninja_orientation = -1

        ninja_y = start_y + height - 1

        # Check if we should add a locked door
        add_locked_door = height == 1 and width >= 4

        # Generate positions with half-tile increments (0.5)
        num_positions = (width - 1) * 4
        available_positions = [start_x + i * 0.25 for i in range(num_positions)]

        # Filter positions to avoid ninja spawn and edges
        available_positions = [
            pos
            for pos in available_positions
            if pos > start_x + 0.5 and pos < start_x + width - 0.5
        ]
        # Exclude positions too close to ninja spawn (minimum 24 pixels = 1 tile distance)
        available_positions = [
            pos
            for pos in available_positions
            if self._calculate_pixel_distance(pos, ninja_y, ninja_x, ninja_y)
            >= TILE_PIXEL_SIZE
        ]

        # For doors, filter to only integer positions (24-pixel boundaries)
        door_positions = [pos for pos in available_positions if pos == int(pos)]

        # Place entities based on layout complexity
        locked_door_viable = False
        if (
            add_locked_door
            and len(door_positions) >= 1
            and len(available_positions) >= 1
        ):
            # Sample 2 door positions (must be integers for 24-pixel alignment)
            door_pos = sorted(self.rng.sample(door_positions, k=2))

            # For each door, find valid switch positions
            switch_available = [p for p in available_positions if p not in door_pos]

            # Exclude switch positions too close to ninja (minimum 24 pixels = 1 tile distance)
            switch_available = [
                p
                for p in switch_available
                if self._calculate_pixel_distance(p, start_y, ninja_x, ninja_y) >= 6
            ]

            # Exclude switch positions too close to doors (minimum 24 pixels = 1 tile distance)
            switch_available = [
                p
                for p in switch_available
                if all(
                    self._calculate_pixel_distance(p, start_y, door_x, start_y) >= 6
                    for door_x in door_pos
                )
            ]

            # Check if we have enough switch positions between ninja and doors
            if ninja_on_left:
                locked_switch_candidates = [
                    p for p in switch_available if p < door_pos[0]
                ]
                exit_switch_candidates = [
                    p for p in switch_available if door_pos[0] < p < door_pos[1]
                ]
            else:
                locked_switch_candidates = [
                    p for p in switch_available if p > door_pos[1]
                ]
                exit_switch_candidates = [
                    p for p in switch_available if door_pos[0] < p < door_pos[1]
                ]

            # Only proceed if we have valid switch positions
            # Also ensure switches are at least 24 pixels apart
            if locked_switch_candidates and exit_switch_candidates:
                # Filter to ensure switches are at least 24 pixels apart
                valid_combinations = []
                for locked_sw in locked_switch_candidates:
                    for exit_sw in exit_switch_candidates:
                        if (
                            self._calculate_pixel_distance(
                                locked_sw, start_y, exit_sw, start_y
                            )
                            >= TILE_PIXEL_SIZE
                        ):
                            valid_combinations.append((locked_sw, exit_sw))

                if valid_combinations:
                    locked_door_viable = True
                    locked_switch_x, exit_switch_x = self.rng.choice(valid_combinations)
                    if ninja_on_left:
                        locked_door_x = door_pos[0]
                        exit_door_x = door_pos[1]
                    else:
                        locked_door_x = door_pos[1]
                        exit_door_x = door_pos[0]
                    entity_y = start_y

        if locked_door_viable:
            # Place locked door and exit door
            self.set_ninja_spawn(ninja_x, ninja_y, orientation=ninja_orientation)

            # Convert switch positions preserving fractional coordinates
            locked_switch_grid = (
                int(locked_switch_x * GRID_SIZE_FACTOR) / GRID_SIZE_FACTOR
            )
            exit_switch_grid = int(exit_switch_x * GRID_SIZE_FACTOR) / GRID_SIZE_FACTOR

            # Apply random vertical offset of ±4px to exit door and exit switch
            exit_door_y = entity_y + (self.rng.random() - 0.5) * 2 * (4.0 / 24.0)
            exit_switch_y = entity_y + (self.rng.random() - 0.5) * 2 * (4.0 / 24.0)

            self.add_entity(
                6,
                int(locked_door_x),
                entity_y,
                4,
                0,
                locked_switch_grid,
                entity_y,
            )
            self.add_entity(
                3,
                int(exit_door_x),
                exit_door_y,
                0,
                0,
                exit_switch_grid,
                exit_switch_y,
            )
        else:
            # Exit door only - use integer positions for door
            # Filter door positions to exclude those too close to ninja (minimum 24 pixels = 1 tile distance)
            filtered_door_positions = [
                p
                for p in door_positions
                if self._calculate_pixel_distance(p, start_y, ninja_x, ninja_y)
                >= TILE_PIXEL_SIZE
            ]
            filtered_available_positions = [
                p
                for p in available_positions
                if self._calculate_pixel_distance(p, start_y, ninja_x, ninja_y)
                >= TILE_PIXEL_SIZE
            ]

            # Find positions that maximize distance between ninja, door, and switch
            exit_door_x, exit_switch_x = self._find_best_door_switch_positions(
                ninja_x,
                ninja_y,
                filtered_door_positions,
                filtered_available_positions,
                start_y,
            )

            exit_switch_y = start_y
            exit_door_y = start_y

            if height > 1:
                exit_switch_y = start_y + self.rng.randint(1, height - 1) * 0.5
                exit_door_y = start_y + self.rng.randint(1, height - 1) * 0.5

            # Apply random vertical offset of ±4px (4px = 1/6 tiles ≈ 0.1667 tiles)
            # rng.random() returns [0, 1), so (rng.random() - 0.5) * 2 gives [-1, 1)
            # Multiply by (4/24) to get ±4px offset
            exit_switch_y += (self.rng.random() - 0.5) * 2 * (4.0 / 24.0)
            exit_door_y += (self.rng.random() - 0.5) * 2 * (4.0 / 24.0)

            # Verify final positions meet minimum distance requirements
            ninja_door_dist = self._calculate_pixel_distance(
                exit_door_x, exit_door_y, ninja_x, ninja_y
            )
            ninja_switch_dist = self._calculate_pixel_distance(
                exit_switch_x, exit_switch_y, ninja_x, ninja_y
            )
            door_switch_dist = self._calculate_pixel_distance(
                exit_door_x, exit_door_y, exit_switch_x, exit_switch_y
            )

            # Ensure all distances are at least 24 pixels (1 tile)
            if (
                ninja_door_dist < TILE_PIXEL_SIZE
                or ninja_switch_dist < TILE_PIXEL_SIZE
                or door_switch_dist < TILE_PIXEL_SIZE
            ):
                # Adjust positions to meet minimum distance
                exit_door_x, exit_door_y, exit_switch_x, exit_switch_y = (
                    self._adjust_positions_for_min_distance(
                        ninja_x,
                        ninja_y,
                        exit_door_x,
                        exit_door_y,
                        exit_switch_x,
                        exit_switch_y,
                        start_x,
                        start_y,
                        width,
                        height,
                    )
                )

            self.set_ninja_spawn(ninja_x, ninja_y, orientation=ninja_orientation)
            self.add_entity(
                3,
                exit_door_x,
                exit_door_y,
                0,
                0,
                exit_switch_x,
                exit_switch_y,
            )

        # Situations where height is 1 and random edge tiles are almost impossible to achieve at first.
        if self.ADD_MINES and not (self.RANDOM_EDGE_TILES and height == 1):
            # Use MAX_MINES as the exact target count for uniform distribution
            num_mines = self.MAX_MINES if self.MAX_MINES > 0 else max(1, width // 2)

            # Add ceiling mines (skip if height == 2) with uniform distribution
            self._place_uniform_ceiling_mines(
                start_x,
                start_y,
                width,
                height,
                ninja_x,
                ninja_y,
                exit_switch_x,
                num_mines=num_mines,
            )

            # Add floor mines (if height >= 2) with uniform distribution
            self._place_uniform_floor_mines(
                start_x,
                start_y,
                width,
                height,
                ninja_x,
                ninja_y,
                exit_switch_x,
                num_mines=num_mines,
            )

        # Add random entities outside the playspace
        self.add_random_entities_outside_playspace(
            start_x - 2,
            start_y - 2,
            start_x + width + 2,
            start_y + height + 2,
        )

        return self
